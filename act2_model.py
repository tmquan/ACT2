import math
import random
import torch
import hashlib
from typing import Dict, List, Tuple
from collections import OrderedDict

from lightning import LightningModule

from cosmos_predict2.models.video2world_model import (
    Predict2Video2WorldModelConfig,
    Predict2ModelManagerConfig,
)
from cosmos_predict2.configs.base.config_video2world import PREDICT2_VIDEO2WORLD_PIPELINE_2B
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.auxiliary.text_encoder import CosmosT5TextEncoder
from cosmos_predict2.conditioner import DataType, ReMapkey, BooleanFlag
import imaginaire.utils.misc as misc
from einops import rearrange

from imaginaire.lazy_config import LazyCall
from cosmos_predict2.models.video2world_dit import MinimalV1LVGDiT
from cosmos_predict2.models.text2image_dit import SACConfig
from cosmos_predict2.configs.base.config_video2world import (
    Video2WorldPipelineConfig,
    Vid2VidConditioner,
    TextAttr,
    ConditioningStrategy,
    CosmosReason1Config,
    CosmosGuardrailConfig,
)
from cosmos_predict2.configs.base.defaults.ema import EMAConfig
from cosmos_predict2.tokenizers.tokenizer import TokenizerInterface


# Cosmos Predict2 Image2Image 2B
PREDICT2_IMAGE2IMAGE_NET_2B = LazyCall(MinimalV1LVGDiT)(
    max_img_h=256,
    max_img_w=256,
    max_frames=2,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    # attention settings
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    atten_backend="minimal_a2a",
    # positional embedding settings
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_h_extrapolation_ratio=2.0,
    rope_w_extrapolation_ratio=2.0,
    rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False,
    sac_config=LazyCall(SACConfig)(
        every_n_blocks=1,
        mode="predict2_2b_720",
    ),
)

PREDICT2_IMAGE2IMAGE_PIPELINE_2B = Video2WorldPipelineConfig(
    adjust_video_noise=True,
    conditioner=LazyCall(Vid2VidConditioner)(
        fps=LazyCall(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="fps",
            output_key="fps",
        ),
        padding_mask=LazyCall(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="padding_mask",
            output_key="padding_mask",
        ),
        text=LazyCall(TextAttr)(
            dropout_rate=0.2,
            input_key=["t5_text_embeddings"],
        ),
        use_video_condition=LazyCall(BooleanFlag)(
            dropout_rate=0.0,
            input_key="fps",
            output_key="use_video_condition",
        ),
    ),
    conditioning_strategy=str(ConditioningStrategy.FRAME_REPLACE),
    min_num_conditional_frames=1,
    max_num_conditional_frames=1,
    net=PREDICT2_IMAGE2IMAGE_NET_2B,
    precision="bfloat16",
    rectified_flow_t_scaling_factor=1.0,
    resize_online=True,
    resolution="720",
    ema=LazyCall(EMAConfig)(enabled=True),  # Enable EMA
    sigma_conditional=0.0001,
    sigma_data=1.0,
    state_ch=16,
    state_t=2,  # Was 24
    text_encoder_class="T5",
    tokenizer=LazyCall(TokenizerInterface)(
        chunk_duration=2,
        temporal_window=16,
        load_mean_std=False,
        name="tokenizer",
        vae_pth="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth",
    ),
    prompt_refiner_config=CosmosReason1Config(
        checkpoint_dir="checkpoints/nvidia/Cosmos-Reason1-7B",
        offload_model_to_cpu=True,
        enabled=True,
    ),
    guardrail_config=CosmosGuardrailConfig(
        checkpoint_dir="checkpoints/",
        offload_model_to_cpu=True,
        enabled=True,
    ),
)

class ACT2CosmosPredict2Model(LightningModule):
    def __init__(self, 
        dit_path: str, 
        text_encoder_path: str, 
        learning_rate: float,
        max_cache_mem_size: int = 10000,  # Maximum number of cached embeddings
        enable_text_cache: bool = True,
        cache_on_gpu: bool = True,  # Store embeddings on GPU for faster access
        gpu_cache_memory_limit_mb: int = 8192,  # GPU memory limit for cache in MB
    ):
        super().__init__()
        self.save_hyperparameters()

        self.precision = torch.bfloat16

        self.pipe = Video2WorldPipeline.from_config(
            PREDICT2_IMAGE2IMAGE_PIPELINE_2B, 
            dit_path=dit_path
        )
        if dit_path:
            state_dict = torch.load(dit_path, map_location="cpu")
            state_dict_dit_compatible = {k.replace("net.", ""): v for k, v in state_dict.items() if k.startswith("net.")}
            self.pipe.dit.load_state_dict(state_dict_dit_compatible, strict=False)

        self.loss_reduce = "mean"
        self.loss_scale = 10.0
        self.video_noise_multiplier = math.sqrt(2) 

        self.last_prediction = None
        
        # Text embedding cache for performance optimization (FIFO)
        self.enable_text_cache = enable_text_cache
        self.max_cache_mem_size = max_cache_mem_size
        self.cache_on_gpu = cache_on_gpu
        self.gpu_cache_memory_limit_mb = gpu_cache_memory_limit_mb
        self.text_embedding_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.cache_memory_usage_mb = 0.0
        
        print(f"Text embedding cache enabled: {self.enable_text_cache}")
        if self.enable_text_cache:
            print(f"Maximum cache size: {self.max_cache_mem_size}")
            print(f"Cache storage: {'GPU' if self.cache_on_gpu else 'CPU'} memory")
            print(f"Cache policy: FIFO (First In, First Out)")
            if self.cache_on_gpu:
                print(f"GPU cache memory limit: {self.gpu_cache_memory_limit_mb} MB")

    def _get_prompt_hash(self, prompt: str) -> str:
        """Generate a hash for a prompt to use as cache key."""
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()

    def _estimate_tensor_memory_mb(self, tensor: torch.Tensor) -> float:
        """Estimate tensor memory usage in MB."""
        return tensor.numel() * tensor.element_size() / (1024 * 1024)

    def _clean_cache_by_memory_limit(self):
        """Clean cache using FIFO policy when memory limit is exceeded - remove one oldest item."""
        if not self.cache_on_gpu or self.cache_memory_usage_mb <= self.gpu_cache_memory_limit_mb:
            return
        
        if len(self.text_embedding_cache) > 0:
            # Remove exactly one oldest entry (FIFO)
            key, tensor = self.text_embedding_cache.popitem(last=False)
            freed_memory = self._estimate_tensor_memory_mb(tensor)
            self.cache_memory_usage_mb -= freed_memory
            print(f"FIFO cache cleanup: Removed 1 oldest entry, freed {freed_memory:.2f} MB")

    def _get_cached_text_embeddings(self, prompts: List[str]) -> Tuple[torch.Tensor, List[str]]:
        """
        Get cached text embeddings for prompts, returning embeddings and list of uncached prompts.
        Updates access order for FIFO management.
        """
        if not self.enable_text_cache:
            return None, prompts
        
        batch_size = len(prompts)
        cached_embeddings = []
        uncached_prompts = []
        uncached_indices = []
        
        for i, prompt in enumerate(prompts):
            prompt_hash = self._get_prompt_hash(prompt)
            if prompt_hash in self.text_embedding_cache:
                # Move to end to update access order (most recently used)
                cached_embedding = self.text_embedding_cache.pop(prompt_hash)
                self.text_embedding_cache[prompt_hash] = cached_embedding
                
                # Move to correct device and dtype if needed
                if self.cache_on_gpu:
                    # Already on GPU, just ensure correct dtype
                    cached_embedding = cached_embedding.to(dtype=self.precision)
                else:
                    # Move from CPU to GPU
                    cached_embedding = cached_embedding.to(device=self.device, dtype=self.precision)
                
                cached_embeddings.append(cached_embedding)
                self.cache_hit_count += 1
            else:
                uncached_prompts.append(prompt)
                uncached_indices.append(i)
                cached_embeddings.append(None)
                self.cache_miss_count += 1
        
        # If all prompts are cached, return the cached embeddings
        if not uncached_prompts:
            final_embeddings = torch.stack([emb for emb in cached_embeddings])
            return final_embeddings, []
        
        # If some prompts are cached, we'll need to encode the uncached ones
        return cached_embeddings, uncached_prompts

    def _cache_text_embeddings(self, prompts: List[str], embeddings: torch.Tensor):
        """Cache text embeddings using FIFO policy - remove oldest when full."""
        if not self.enable_text_cache:
            return
        
        # Cache the new embeddings (will be added at the end, making them newest)
        for prompt, embedding in zip(prompts, embeddings):
            prompt_hash = self._get_prompt_hash(prompt)
            
            # Remove oldest item if cache is at capacity
            if len(self.text_embedding_cache) >= self.max_cache_mem_size:
                # Remove exactly one oldest entry (FIFO)
                removed_key, removed_tensor = self.text_embedding_cache.popitem(last=False)
                if self.cache_on_gpu:
                    self.cache_memory_usage_mb -= self._estimate_tensor_memory_mb(removed_tensor)
                print(f"FIFO text cache: Removed oldest entry to make space")
            
            if self.cache_on_gpu:
                # Store on GPU for faster access
                cached_embedding = embedding.detach().to(dtype=self.precision)
                # Update memory usage tracking
                self.cache_memory_usage_mb += self._estimate_tensor_memory_mb(cached_embedding)
            else:
                # Store on CPU to save GPU memory
                cached_embedding = embedding.detach().cpu()
            
            # Add new entry (becomes the newest in FIFO order)
            self.text_embedding_cache[prompt_hash] = cached_embedding
            
            # If GPU memory limit is exceeded after adding, remove one oldest item
            if self.cache_on_gpu and self.cache_memory_usage_mb > self.gpu_cache_memory_limit_mb:
                self._clean_cache_by_memory_limit()

    def _encode_prompts_with_cache(self, prompts: List[str]) -> torch.Tensor:
        """
        Encode prompts using cache when possible.
        """
        if not self.enable_text_cache:
            # No caching, encode all prompts
            return self.pipe.encode_prompt(prompts).to(dtype=self.precision)
        
        # Try to get cached embeddings
        cached_result, uncached_prompts = self._get_cached_text_embeddings(prompts)
        
        if isinstance(cached_result, torch.Tensor):
            # All prompts were cached
            return cached_result
        
        # Some prompts need to be encoded
        if uncached_prompts:
            # Encode uncached prompts
            new_embeddings = self.pipe.encode_prompt(uncached_prompts).to(dtype=self.precision)
            
            # Cache the new embeddings
            self._cache_text_embeddings(uncached_prompts, new_embeddings)
            
            # Reconstruct the full batch
            final_embeddings = []
            uncached_idx = 0
            
            for i, prompt in enumerate(prompts):
                if cached_result[i] is not None:
                    # Use cached embedding
                    final_embeddings.append(cached_result[i])
                else:
                    # Use newly encoded embedding
                    final_embeddings.append(new_embeddings[uncached_idx])
                    uncached_idx += 1
            
            return torch.stack(final_embeddings)
        else:
            # This shouldn't happen, but handle it gracefully
            return self.pipe.encode_prompt(prompts).to(dtype=self.precision)

    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics for monitoring."""
        total_requests = self.cache_hit_count + self.cache_miss_count
        hit_rate = (self.cache_hit_count / total_requests * 100) if total_requests > 0 else 0
        
        # Get current GPU memory usage if available
        gpu_memory_used_mb = 0.0
        gpu_memory_total_mb = 0.0
        if torch.cuda.is_available() and self.device.type == 'cuda':
            gpu_memory_used_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            gpu_memory_total_mb = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
        
        return {
            "cache_mem_size": len(self.text_embedding_cache),
            "cache_hits": self.cache_hit_count,
            "cache_misses": self.cache_miss_count,
            "hit_rate_percent": hit_rate,
            "total_requests": total_requests,
            "cache_memory_mb": self.cache_memory_usage_mb if self.cache_on_gpu else 0.0,
            "gpu_memory_used_mb": gpu_memory_used_mb,
            "gpu_memory_total_mb": gpu_memory_total_mb,
            "cache_storage": "GPU" if self.cache_on_gpu else "CPU"
        }

    def setup(self, stage: str):
        if stage == "fit":
            self.pipe.text_encoder = CosmosT5TextEncoder(device=self.device, cache_dir=self.hparams.text_encoder_path)
            # Freeze the text encoder's parameters and set it to evaluation mode
            self.pipe.text_encoder.requires_grad_(False)
            self.pipe.text_encoder.eval()

            # Freeze the tokenizer's parameters and set it to evaluation mode
            self.pipe.tokenizer.model.model.requires_grad_(False)
            self.pipe.tokenizer.model.model.eval()

            # Unfreeze the DiT
            self.pipe.denoising_model().requires_grad_(True)
            self.pipe.denoising_model().train()

    def process_batch(self, batch: dict) -> dict:
        prompts = batch["txt"]
        cond_frame = batch["png"]
        target_frame = batch["tif"]

        if self.training:
            shuffled_prompts = []
            for prompt in prompts:
                words = prompt.split(' ')
                random.shuffle(words)
                shuffled_prompts.append(' '.join(words))
            prompts = shuffled_prompts
            batch['txt'] = prompts

        video = torch.stack([cond_frame, target_frame], dim=2)
        video = (video * 255.0).to(torch.uint8)
        B, C, T, H, W = video.shape
        
        # Use cached text embeddings for better performance
        text_embeddings = self._encode_prompts_with_cache(prompts)
        
        return {
            "video": video,
            "t5_text_embeddings": text_embeddings,
            "dataset_name": "video_data",
            "num_conditional_frames": 1,
            "fps": torch.ones((B,), device=video.device, dtype=torch.long),
            "padding_mask": torch.zeros(B, 1, H, W, device=video.device, dtype=self.precision),
        }

    def draw_training_sigma_and_epsilon(self, x0_size: torch.Size, is_video_batch: bool) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x0_size[0]
        epsilon = torch.randn(x0_size, device=self.device)
        sigma_B = self.pipe.scheduler.sample_sigma(batch_size).to(device=self.device)
        sigma_B_1 = rearrange(sigma_B, "b -> b 1")
        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B_1 = sigma_B_1 * multiplier
        return sigma_B_1, epsilon

    def get_per_sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.pipe.sigma_data**2) / (sigma * self.pipe.sigma_data) ** 2

    def compute_loss(self, x0, condition, epsilon, sigma) -> tuple[dict, torch.Tensor]:
        mean, std = x0, sigma
        xt = mean + epsilon * rearrange(std, "b t -> b 1 t 1 1")
        out_pred = self.pipe.denoise(xt, sigma, condition)
        weights = self.get_per_sigma_loss_weights(sigma=sigma)
        mse_loss = (x0 - out_pred.x0) ** 2
        edm_loss = mse_loss * rearrange(weights, "b t -> b 1 t 1 1")
        
        output_batch = {
            "out_pred": out_pred,
            "mse_loss": mse_loss.mean(),
            "edm_loss": edm_loss.mean(),
        }
        
        ret_loss = edm_loss
        
        return output_batch, ret_loss


    def _shared_step(self, data_batch):
        _, x0, condition = self.pipe.get_data_and_condition(data_batch)
        sigma, epsilon = self.draw_training_sigma_and_epsilon(x0.size(), condition.data_type == DataType.VIDEO)
        x0, condition, epsilon, sigma = self.pipe.broadcast_split_for_model_parallelsim(x0, condition, epsilon, sigma)
        output, loss_tensor = self.compute_loss(x0, condition, epsilon, sigma)
        
        if self.loss_reduce == "mean":
            loss = loss_tensor.mean() * self.loss_scale
        else:
            loss = loss_tensor.sum(dim=1).mean() * self.loss_scale
        return output, loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        data_batch = self.process_batch(batch)
        output_batch, loss = self._shared_step(data_batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mse_loss', output_batch['mse_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_edm_loss', output_batch['edm_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        # Log cache statistics every 100 steps
        if batch_idx % 100 == 0 and self.enable_text_cache:
            cache_stats = self.get_cache_stats()
            self.log('cache_hit_rate', cache_stats['hit_rate_percent'], on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('cache_mem_size', cache_stats['cache_mem_size'], on_step=True, on_epoch=False, prog_bar=False, logger=True)
            if self.cache_on_gpu:
                self.log('cache_memory_mb', cache_stats['cache_memory_mb'], on_step=True, on_epoch=False, prog_bar=False, logger=True)
                self.log('gpu_memory_used_mb', cache_stats['gpu_memory_used_mb'], on_step=True, on_epoch=False, prog_bar=False, logger=True)
        
        if batch_idx == 0:
            self._generate_denoised_image(data_batch)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        data_batch = self.process_batch(batch)
        output_batch, loss = self._shared_step(data_batch)
        # Log the evaluation/test loss.
        self.log(f'val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx == 0:
            self._generate_denoised_image(data_batch)
        return loss
    
    def test_step(self, batch: dict, batch_idx: int) -> None:
        data_batch = self.process_batch(batch)
        _, loss = self._shared_step(data_batch)
        self.log(f'test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def _generate_denoised_image(self, data_batch: dict):
        with torch.no_grad():
            x0_fn = self.pipe.get_x0_fn_from_batch(data_batch, guidance=7.0, is_negative_prompt=False)
            _T, _H, _W = data_batch["video"].shape[-3:]
            state_shape = [
                self.pipe.config.state_ch,
                self.pipe.tokenizer.get_latent_num_frames(_T),
                _H // self.pipe.tokenizer.spatial_compression_factor,
                _W // self.pipe.tokenizer.spatial_compression_factor,
            ]
            x_sigma_max = misc.arch_invariant_rand(
                (data_batch["video"].shape[0],) + tuple(state_shape), torch.float32, self.device, 0
            ) * self.pipe.scheduler.config.sigma_max
            
            scheduler = self.pipe.scheduler
            scheduler.set_timesteps(35, device=x_sigma_max.device)
            sample = x_sigma_max.to(dtype=torch.float32)
            x0_prev = None
            for i, _ in enumerate(scheduler.timesteps):
                sigma_t = scheduler.sigmas[i].to(sample.device, dtype=torch.float32)
                sigma_in = sigma_t.repeat(sample.shape[0])
                x0_pred = x0_fn(sample, sigma_in)
                sample, x0_prev = scheduler.step(x0_pred, i, sample, x0_prev)

            sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
            samples = x0_fn(sample, sigma_min.repeat(sample.shape[0]))
            video = self.pipe.decode(samples)
            self.last_prediction = video[:, :, -1, :, :]

    def configure_optimizers(self):
        trainable_params = [p for p in self.pipe.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable_params, lr=self.hparams.learning_rate)
    
    def on_train_epoch_end(self):
        """Print cache statistics at the end of each epoch."""
        if self.enable_text_cache:
            cache_stats = self.get_cache_stats()
            print(f"\n=== Text Embedding Cache Stats (Epoch {self.current_epoch}) ===")
            print(f"Cache size: {cache_stats['cache_mem_size']}/{self.max_cache_mem_size}")
            print(f"Hit rate: {cache_stats['hit_rate_percent']:.1f}%")
            print(f"Total requests: {cache_stats['total_requests']}")
            print(f"Cache hits: {cache_stats['cache_hits']}")
            print(f"Cache misses: {cache_stats['cache_misses']}")
            print(f"Cache storage: {cache_stats['cache_storage']}")
            if self.cache_on_gpu:
                print(f"Cache memory usage: {cache_stats['cache_memory_mb']:.2f} MB")
                print(f"GPU memory used: {cache_stats['gpu_memory_used_mb']:.2f} MB")
                print(f"GPU memory total: {cache_stats['gpu_memory_total_mb']:.2f} MB")
            print("="*50)
    
    