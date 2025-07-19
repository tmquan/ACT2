import math
import random
import torch
import hashlib
from typing import Dict, List, Tuple
from collections import OrderedDict

from lightning import LightningModule
from lightning.pytorch.utilities.model_summary import summarize

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
    VideoConditioner,
    TextAttr,
    ConditioningStrategy,
    CosmosReason1Config,
    CosmosGuardrailConfig,
)
from cosmos_predict2.configs.base.defaults.ema import EMAConfig
from cosmos_predict2.tokenizers.tokenizer import TokenizerInterface

# Import kornia for HSV conversion
import kornia
import kornia.color


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
    conditioner=LazyCall(VideoConditioner)(
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
            dropout_rate=0.0,
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
    ema=LazyCall(EMAConfig)(enabled=False),  # Disable EMA
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


class CosmosPredict2Module(LightningModule):
    def __init__(self, 
        dit_path: str, 
        tokenizer_path: str,
        text_encoder_path: str, 
        hparams: Dict = {},
    ):  
        super().__init__()
        self.save_hyperparameters()
        
        self.dit_path = dit_path
        self.tokenizer_path = tokenizer_path
        self.text_encoder_path = text_encoder_path
        self.learning_rate = hparams.get("learning_rate", 1e-5)

        # Initialize a placeholder pipeline - implement according to your needs
        self.pipe = None

    def setup(self, stage: str):
        if stage == "fit":
            pass
        if stage == "test":
            pass
        if stage == "predict":
            pass

    def configure_optimizers(self):
        if self.pipe is not None:
            trainable_params = [p for p in self.pipe.parameters() if p.requires_grad]
            return torch.optim.AdamW(trainable_params, lr=self.learning_rate)
        # Create a dummy parameter for when pipe is None
        dummy_param = torch.nn.Parameter(torch.tensor(0.0))
        return torch.optim.AdamW([dummy_param], lr=self.learning_rate)

    def _shared_step(self, data_batch):
        # Implement shared step logic
        return {"loss": torch.tensor(0.0, requires_grad=True)}

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        step_output = self._shared_step(batch)
        return step_output["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        step_output = self._shared_step(batch)
        self.log("val_loss", step_output["loss"])

    def test_step(self, batch: dict, batch_idx: int) -> None:
        step_output = self._shared_step(batch)
        self.log("test_loss", step_output["loss"])
   
    def predict_step(self, batch: dict, batch_idx: int) -> None:
        pass

    def on_train_epoch_start(self):
        pass
    
    def on_validation_epoch_start(self):
        pass
    
    def on_test_epoch_start(self):
        pass
    
    def on_predict_epoch_start(self):
        pass

    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_end(self):
        pass
    
    def on_test_epoch_end(self):
        pass
    
    def on_predict_epoch_end(self):
        pass


class ACT2CosmosPredict2Module(LightningModule):
    def __init__(self, 
        dit_path: str = "checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps-natten.pt",
        tokenizer_path: str = "checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth",
        text_encoder_path: str = "checkpoints/google-t5/t5-11b",
        hparams: Dict = {},
        learning_rate: float = 1e-5,
        video_noise_multiplier: float = 1.0,
        hsv_weight: float = 0.1,
        img_weight: float = 0.1,
        cache_size: int = 1000,
        enable_cache: bool = True,
        loss_scale: float = 10.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.dit_path = dit_path
        self.tokenizer_path = tokenizer_path
        self.text_encoder_path = text_encoder_path
        self.learning_rate = learning_rate
        self.video_noise_multiplier = video_noise_multiplier
        self.hsv_weight = hsv_weight
        self.img_weight = img_weight
        self.loss_scale = loss_scale
        self.precision = torch.bfloat16
        self.last_prediction = None
        
        # Text embedding cache configuration
        self.cache_size = cache_size
        self.enable_cache = enable_cache
        self._text_cache = OrderedDict()  # OrderedDict for LRU tracking
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Initialize pipeline
        self.pipe = Video2WorldPipeline.from_config(
            PREDICT2_IMAGE2IMAGE_PIPELINE_2B,
            dit_path=dit_path or ""  # Provide empty string instead of None
        )
        
        # Load checkpoints
        self._load_checkpoints()

    def _load_checkpoints(self):
        """Load all model checkpoints and configure them appropriately."""
        # Load DiT model if path provided
        if self.dit_path:
            try:
                print(f"Loading DiT model from: {self.dit_path}")
                state_dict = torch.load(self.dit_path, map_location="cpu")
                state_dict_dit_compatible = {k.replace("net.", ""): v for k, v in state_dict.items() if k.startswith("net.")}
                self.pipe.denoising_model().load_state_dict(state_dict_dit_compatible, strict=False)
                self.pipe.denoising_model().requires_grad_(True)
                self.pipe.denoising_model().train()
                print(f"✓ DiT model loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load DiT model: {e}")
                raise
            
        # Initialize tokenizer if path provided
        if self.tokenizer_path:
            try:
                print(f"Loading tokenizer from: {self.tokenizer_path}")
                # The tokenizer should be frozen and set to eval mode
                self.pipe.tokenizer.model.model.requires_grad_(False)
                self.pipe.tokenizer.model.model.eval()
                print(f"✓ Tokenizer loaded and frozen successfully")
            except Exception as e:
                print(f"✗ Failed to configure tokenizer: {e}")
                raise

        # Initialize text encoder if path provided
        if self.text_encoder_path:
            try:
                print(f"Loading text encoder from: {self.text_encoder_path}")
                self.pipe.text_encoder = CosmosT5TextEncoder(device=self.device, cache_dir=self.text_encoder_path)
                self.pipe.text_encoder.requires_grad_(False)
                self.pipe.text_encoder.eval()
                print(f"✓ Text encoder loaded and frozen successfully")
            except Exception as e:
                print(f"✗ Failed to load text encoder: {e}")
                raise

    def _process_batch(self, batch: dict) -> dict:
        """Process input batch and prepare for training/inference."""
        prompts = batch["txt"]
        cond_frame = batch["tif"]
        true_frame = batch["png"]
        _device = cond_frame.device
        
        if self.training:
            shuffled_prompts = []
            for prompt in prompts:
                words = prompt.split(' ')
                random.shuffle(words)
                shuffled_prompts.append(' '.join(words))
            prompts = shuffled_prompts
            batch['txt'] = prompts

        init_frame = cond_frame.clone().unsqueeze(2)
        last_frame = true_frame.clone().unsqueeze(2)
        
        # Create base video with two frames: conditioning frame + target frame
        video = torch.cat([init_frame, last_frame], dim=2)   
        
        # Get expected frame count from tokenizer configuration and extend if needed
        try:
            expected_length = self.pipe.tokenizer.get_pixel_num_frames(self.pipe.config.state_t)
            original_length = video.shape[2]  # Should be 2
            
            if original_length < expected_length:
                # Need more frames - repeat the last frame only
                additional_frames_needed = expected_length - original_length
                repeated_last_frame = last_frame.repeat(1, 1, additional_frames_needed, 1, 1)
                video = torch.cat([video, repeated_last_frame], dim=2)
            elif original_length > expected_length:
                # This shouldn't happen with our 2-frame setup, but handle it safely
                # Keep the first frame (conditioning) and truncate the rest
                video = video[:, :, :expected_length, :, :]
            
        except Exception as e:
            print(f"Warning: Could not apply temporal sampling: {e}")
            # Continue with 2-frame video if sampling fails

        # Convert to uint8
        video = (video * 255.0).to(torch.uint8)
        B, C, T, H, W = video.shape
        # Use cached text embeddings for better performance
        text_embeddings = self._encode_prompts_with_cache(prompts=prompts, device=_device)
        
        return {
            "video": video,  # This is the key the pipeline expects for video input
            "prompt": prompts,
            "t5_text_embeddings": text_embeddings,
            "dataset_name": "video_data",
            "num_conditional_frames": 1,
            "fps": torch.ones((B,), device=video.device, dtype=torch.long),
            "padding_mask": torch.zeros(B, 1, H, W, device=video.device, dtype=self.precision),
        }

    def _encode_prompts_with_cache(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        """Encode text prompts using cached embeddings for performance."""
        if not self.enable_cache:
            # If caching is disabled, encode directly
            return self._encode_prompts_direct(prompts=prompts, device=device)
        
        batch_embeddings = []
        prompts_to_encode = []
        prompt_indices_to_encode = []
        
        # Check cache for each individual prompt
        for i, prompt in enumerate(prompts):
            cache_key = self._get_cache_key_single(prompt)
            
            if cache_key in self._text_cache:
                # Cache hit - reuse existing embedding and move to end (most recent)
                self._cache_hits += 1
                cached_embedding = self._text_cache[cache_key]
                # Move to end for LRU tracking (most recently used)
                self._text_cache.move_to_end(cache_key)
                batch_embeddings.append(cached_embedding)
                self.log("cache_hits", self._cache_hits, on_step=True, logger=True)
            else:
                # Cache miss - need to encode this prompt
                self._cache_misses += 1
                prompts_to_encode.append(prompt)
                prompt_indices_to_encode.append(i)
                batch_embeddings.append(None)  # Placeholder
                self.log("cache_misses", self._cache_misses, on_step=True, logger=True)
        
        # Encode missing prompts if any
        if prompts_to_encode:
            new_embeddings = self._encode_prompts_direct(prompts=prompts_to_encode, device=device)
            
            # Store individual embeddings in cache and update batch
            for j, prompt_id in enumerate(prompt_indices_to_encode):
                individual_embedding = new_embeddings[j:j+1]  # Keep batch dimension
                cache_key = self._get_cache_key_single(prompts_to_encode[j])
                
                # Check if cache is full and evict oldest (first) item
                if len(self._text_cache) >= self.cache_size:
                    # Remove least recently used item (first in OrderedDict)
                    self._text_cache.popitem(last=False)
                    self.log("cache_evictions", 1, on_step=True, logger=True)
                
                # Insert new embedding (will be added to end - most recent)
                self._text_cache[cache_key] = individual_embedding.clone().detach()
                batch_embeddings[prompt_id] = individual_embedding
        
        # Concatenate all embeddings to form the final batch
        final_embeddings = torch.cat(batch_embeddings, dim=0)
        return final_embeddings

    def _encode_prompts_direct(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        """Directly encode prompts without caching."""
        if hasattr(self.pipe, 'encode_prompt'):
            try:
                # Fix device mismatch between wrapper and actual encoder
                if hasattr(self.pipe, 'text_encoder') and self.pipe.text_encoder is not None:
                    # Get the actual encoder device
                    actual_device = device  # Default fallback
                    if hasattr(self.pipe.text_encoder, 'text_encoder'):
                        try:
                            actual_device = next(self.pipe.text_encoder.text_encoder.parameters()).device
                        except StopIteration:
                            pass
                    
                    # Ensure the wrapper's device attribute matches the actual model
                    if hasattr(self.pipe.text_encoder, 'device'):
                        if self.pipe.text_encoder.device != actual_device:
                            self.pipe.text_encoder.device = actual_device
                    
                    # Also move the wrapper to the correct device if it has a 'to' method
                    if hasattr(self.pipe.text_encoder, 'to'):
                        self.pipe.text_encoder = self.pipe.text_encoder.to(actual_device)
                
                # Encode prompts
                embeddings = self.pipe.encode_prompt(prompts)
                
                # Convert to target device and precision
                return embeddings.to(dtype=self.precision, device=device)
                
            except Exception as e:
                print(f"Warning: Text encoding failed: {e}")
                print("Falling back to dummy embeddings")
                
                # Fall back to dummy embeddings if encoding fails
                B = len(prompts)
                return torch.zeros((B, 77, 4096), dtype=self.precision, device=device)
        else:
            # Return dummy embeddings if encode_prompt method not available
            B = len(prompts)
            return torch.zeros((B, 77, 4096), dtype=self.precision, device=device)

    def _get_cache_key_single(self, prompt: str) -> str:
        """Generate a unique cache key for a single prompt."""
        # Create a hash of the prompt
        cache_key = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        return cache_key

    def _get_cache_key(self, prompts: List[str]) -> str:
        """Generate a unique cache key for a list of prompts (legacy method)."""
        # Sort prompts to ensure consistent hashing regardless of order
        sorted_prompts = sorted(prompts)
        # Create a hash of the concatenated prompts
        prompt_str = "|".join(sorted_prompts)
        cache_key = hashlib.md5(prompt_str.encode('utf-8')).hexdigest()
        return cache_key

    def _clear_cache(self):
        """Clear the text embedding cache."""
        cache_size_before = len(self._text_cache)
        self._text_cache.clear()  # OrderedDict.clear() works the same
        self._cache_hits = 0
        self._cache_misses = 0
        # Log cache clearing
        self.log("cache_cleared", cache_size_before, logger=True)

    def _log_cache_stats(self):
        """Log cache performance statistics to the logger."""
        stats = self._get_cache_stats()
        
        # Log all cache metrics
        self.log("cache_size", stats["cache_size"], logger=True)
        self.log("cache_hit_rate", stats["hit_rate"], logger=True)
        self.log("cache_hits_total", stats["cache_hits"], logger=True)
        self.log("cache_misses_total", stats["cache_misses"], logger=True)
        self.log("cache_enabled", float(stats["enabled"]), logger=True)

    def _put_cache_stats(self):
        """Print cache performance statistics (for debugging only)."""
        stats = self._get_cache_stats()
        print("\n" + "="*50)
        print("TEXT EMBEDDING CACHE STATISTICS (OrderedDict LRU)")
        print("="*50)
        print(f"Cache Status: {'Enabled' if stats['enabled'] else 'Disabled'}")
        print(f"Cache Size: {stats['cache_size']}/{stats['max_cache_size']}")
        print(f"Cache Hits: {stats['cache_hits']}")
        print(f"Cache Misses: {stats['cache_misses']}")
        print(f"Hit Rate: {stats['hit_rate']:.2%}")
        print("="*50)
        
        # Also log to the logger
        self._log_cache_stats()

    def _get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_size": len(self._text_cache),
            "max_cache_size": self.cache_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "enabled": self.enable_cache
        }

    def _convert_to_rgb_range(self, x):
        """Convert from model latent space to RGB [0, 1] range for image domain losses."""
        # Decode latents to image space using the pipeline decoder
        video = self.pipe.decode(x) / 2.0 + 0.5
        # Ensure values are in [0, 1] range
        rgb = torch.clamp(video, 0.0, 1.0)
        return rgb

    def _draw_training_sigma_and_epsilon(self, x0_size: torch.Size, is_video_batch: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate training noise parameters for EDM loss."""
        batch_size = x0_size[0]
        epsilon = torch.randn(x0_size, device=self.device)
        sigma_B = self.pipe.scheduler.sample_sigma(batch_size).to(device=self.device)
        sigma_B_1 = rearrange(sigma_B, "b -> b 1")
        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B_1 = sigma_B_1 * multiplier
        return sigma_B_1, epsilon

    def _get_per_sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute per-sigma loss weights for EDM loss."""
        return (sigma**2 + self.pipe.sigma_data**2) / (sigma * self.pipe.sigma_data) ** 2

    def _compute_loss(self, x0, condition, epsilon, sigma) -> tuple[dict, torch.Tensor]:
        """Compute EDM loss given input tensors."""
        mean, std = x0, sigma
        xt = mean + epsilon * rearrange(std, "b t -> b 1 t 1 1")
        out_pred = self.pipe.denoise(xt, sigma, condition)
        weights = self._get_per_sigma_loss_weights(sigma=sigma)
        mse_loss = (x0 - out_pred.x0) ** 2
        edm_loss = mse_loss * rearrange(weights, "b t -> b 1 t 1 1")
        
        # Image domain loss - compute on full frame
        img_loss = torch.tensor(0.0, device=x0.device, dtype=x0.dtype)
        if self.img_weight > 0:
            # Convert from model space to RGB [0, 1] range
            rgb_true = self._convert_to_rgb_range(x0)  # Shape: [B, C, T, H, W]
            rgb_pred = self._convert_to_rgb_range(out_pred.x0)  # Shape: [B, C, T, H, W]
            
            # Simple MSE loss in image space (on full frame)
            img_loss = ((rgb_true - rgb_pred) ** 2)

        # HSV domain loss - compute on full frame
        hsv_loss = torch.tensor(0.0, device=x0.device, dtype=x0.dtype)
        if self.hsv_weight > 0:
            # Convert from model space to RGB [0, 1] range
            rgb_true = self._convert_to_rgb_range(x0)  # Shape: [B, C, T, H, W]
            rgb_pred = self._convert_to_rgb_range(out_pred.x0)  # Shape: [B, C, T, H, W]
            
            # Reshape video tensors to combine batch and time dimensions for HSV conversion
            # einops: combine B and T dimensions together (full frame)
            B, C, T, H, W = rgb_true.shape
            rgb_true_flat = rearrange(rgb_true, 'b c t h w -> (b t) c h w')
            rgb_pred_flat = rearrange(rgb_pred, 'b c t h w -> (b t) c h w')
            
            # Convert to HSV space using kornia
            hsv_true_flat = kornia.color.rgb_to_hsv(rgb_true_flat)
            hsv_pred_flat = kornia.color.rgb_to_hsv(rgb_pred_flat)
            
            # Reshape back to original video shape (full frame)
            hsv_true = rearrange(hsv_true_flat, '(b t) c h w -> b c t h w', b=B, t=T)
            hsv_pred = rearrange(hsv_pred_flat, '(b t) c h w -> b c t h w', b=B, t=T)
            
            # Simple MSE loss in HSV space (on full frame)
            hsv_loss = ((hsv_true - hsv_pred) ** 2)
        
        output_batch = {
            "out_pred": out_pred,
            "mse_loss": mse_loss.mean(),
            "edm_loss": edm_loss.mean(),
            "img_loss": img_loss.mean(),
            "hsv_loss": hsv_loss.mean(),
        }
        
        # Combine losses - ensure all are on the same device
        img_loss = img_loss.to(device=edm_loss.device)
        hsv_loss = hsv_loss.to(device=edm_loss.device)
        ret_loss = (
            self.loss_scale * edm_loss.mean() 
            + self.img_weight * img_loss.mean()
            + self.hsv_weight * hsv_loss.mean()
        )
        
        return output_batch, ret_loss

    def _generate_denoised_image(self, data_batch: dict, frame_id: int=1):
        """Generate denoised image for visualization/validation."""
        # Ensure all models are in eval mode for consistent generation
        was_training = self.pipe.denoising_model().training
        self.pipe.denoising_model().eval()
        
        # Ensure frozen components stay frozen and in eval mode
        if hasattr(self.pipe, 'tokenizer') and hasattr(self.pipe.tokenizer, 'model'):
            self.pipe.tokenizer.model.model.eval()
            self.pipe.tokenizer.model.model.requires_grad_(False)
        
        if hasattr(self.pipe, 'text_encoder') and self.pipe.text_encoder is not None:
            self.pipe.text_encoder.eval()
            self.pipe.text_encoder.requires_grad_(False)
        
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
            self.last_prediction = video[:, :, frame_id, :, :]
            
        # Restore original training state
        if was_training:
            self.pipe.denoising_model().train()

    def _debug_model_states(self):
        """Debug method to check model states and gradients."""
        print("\n" + "="*60)
        print("MODEL STATE DEBUGGING")
        print("="*60)
        
        # Check DiT model
        dit_training = self.pipe.denoising_model().training
        dit_requires_grad = any(p.requires_grad for p in self.pipe.denoising_model().parameters())
        print(f"DiT Model - Training: {dit_training}, Requires Grad: {dit_requires_grad}")
        
        # Check tokenizer
        if hasattr(self.pipe, 'tokenizer') and hasattr(self.pipe.tokenizer, 'model'):
            tok_training = self.pipe.tokenizer.model.model.training
            tok_requires_grad = any(p.requires_grad for p in self.pipe.tokenizer.model.model.parameters())
            print(f"Tokenizer - Training: {tok_training}, Requires Grad: {tok_requires_grad}")
        
        # Check text encoder
        if hasattr(self.pipe, 'text_encoder') and self.pipe.text_encoder is not None:
            if hasattr(self.pipe.text_encoder, 'training'):
                te_training = self.pipe.text_encoder.training
                te_requires_grad = any(p.requires_grad for p in self.pipe.text_encoder.parameters())
                print(f"Text Encoder - Training: {te_training}, Requires Grad: {te_requires_grad}")
        
        print("="*60)

    def setup(self, stage: str):
        """Configure models for different training stages."""
        if stage == "fit":
            # Ensure DiT model is in training mode with gradients enabled
            if hasattr(self.pipe, 'denoising_model') and self.pipe.denoising_model():
                self.pipe.denoising_model().requires_grad_(True)
                self.pipe.denoising_model().train()
                print("✓ DiT model set to training mode")

            # Ensure tokenizer remains frozen and in eval mode
            if hasattr(self.pipe, 'tokenizer') and hasattr(self.pipe.tokenizer, 'model'):
                self.pipe.tokenizer.model.model.requires_grad_(False)
                self.pipe.tokenizer.model.model.eval()
                print("✓ Tokenizer kept frozen in eval mode")

            # Ensure text encoder remains frozen and in eval mode
            if hasattr(self.pipe, 'text_encoder') and self.pipe.text_encoder is not None:
                self.pipe.text_encoder.requires_grad_(False)
                self.pipe.text_encoder.eval()
                print("✓ Text encoder kept frozen in eval mode")

        elif stage in ["test", "predict"]:
            # Set all models to eval mode for testing/prediction
            if hasattr(self.pipe, 'denoising_model') and self.pipe.denoising_model():
                self.pipe.denoising_model().eval()
                print("✓ DiT model set to eval mode")
                
            if hasattr(self.pipe, 'tokenizer') and hasattr(self.pipe.tokenizer, 'model'):
                self.pipe.tokenizer.model.model.eval()
                print("✓ Tokenizer in eval mode")
                
            if hasattr(self.pipe, 'text_encoder') and self.pipe.text_encoder is not None:
                self.pipe.text_encoder.eval()
                print("✓ Text encoder in eval mode")

    def configure_optimizers(self):
        """Configure optimizers"""
        # Only train parameters that require gradients
        trainable_params = [p for p in self.pipe.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable_params, lr=self.learning_rate)
    
    def _shared_step(self, data_batch):
        """Shared step for training and validation."""
        # Process the batch to get it in the right format
        result_batch = self._process_batch(data_batch)
        
        # Use the pipeline's get_data_and_condition method for consistent processing
        raw_state, latent_state, condition = self.pipe.get_data_and_condition(result_batch)
        
        # Draw training noise parameters
        sigma, epsilon = self._draw_training_sigma_and_epsilon(
            latent_state.shape, 
            is_video_batch=True
        )
        
        # Handle model parallelism (context parallelism)
        latent_state, condition, epsilon, sigma = self.pipe.broadcast_split_for_model_parallelsim(
            latent_state, condition, epsilon, sigma
        )
        
        # Compute EDM loss
        output_batch, loss = self._compute_loss(
            x0=latent_state,
            condition=condition,
            epsilon=epsilon,
            sigma=sigma
        )
        
        return {
            "loss": loss.mean(),
            "mse_loss": output_batch["mse_loss"],
            "edm_loss": output_batch["edm_loss"],
            "img_loss": output_batch["img_loss"],
            "hsv_loss": output_batch["hsv_loss"],
            "result_batch": result_batch,
            "output_batch": output_batch
        }

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step using EDM loss."""
        step_output = self._shared_step(batch)
        
        # Log losses
        self.log("train_loss", step_output["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mse_loss", step_output["mse_loss"], on_step=True, on_epoch=True)
        self.log("train_edm_loss", step_output["edm_loss"], on_step=True, on_epoch=True)
        self.log("train_img_loss", step_output["img_loss"], on_step=True, on_epoch=True)
        self.log("train_hsv_loss", step_output["hsv_loss"], on_step=True, on_epoch=True)
        
        # Generate sample predictions for visualization (on first batch only)
        if batch_idx == 0:
            # Debug model states before generation
            # self._debug_model_states()
            self._generate_denoised_image(step_output["result_batch"], frame_id=1)

        return step_output["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Validation step using EDM loss."""
        step_output = self._shared_step(batch)
        
        # Log validation losses
        self.log("val_loss", step_output["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mse_loss", step_output["mse_loss"], on_step=False, on_epoch=True)
        self.log("val_edm_loss", step_output["edm_loss"], on_step=False, on_epoch=True)
        self.log("val_img_loss", step_output["img_loss"], on_step=False, on_epoch=True)
        self.log("val_hsv_loss", step_output["hsv_loss"], on_step=False, on_epoch=True)
        
        # Generate sample predictions for visualization (on first batch only)
        if batch_idx == 0:
            # Debug model states before generation
            # self._debug_model_states()
            self._generate_denoised_image(step_output["result_batch"], frame_id=1)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Test step using EDM loss."""
        step_output = self._shared_step(batch)
        
        # Log test losses
        self.log("test_loss", step_output["loss"], on_step=False, on_epoch=True)
        self.log("test_mse_loss", step_output["mse_loss"], on_step=False, on_epoch=True)
        self.log("test_edm_loss", step_output["edm_loss"], on_step=False, on_epoch=True)
        self.log("test_img_loss", step_output["img_loss"], on_step=False, on_epoch=True)
        self.log("test_hsv_loss", step_output["hsv_loss"], on_step=False, on_epoch=True)
    
    def predict_step(self, batch: dict, batch_idx: int) -> None:
        """Predict step for inference."""
        result_batch = self._process_batch(batch)
        self._generate_denoised_image(result_batch, frame_id=1)
        return self.last_prediction
    
    def on_train_epoch_start(self):
        """Log cache statistics at the start of each training epoch."""
        if self.enable_cache:
            self._log_cache_stats()

    def on_validation_epoch_start(self):
        """Log cache statistics at the start of validation."""
        if self.enable_cache:
            self._log_cache_stats()

    def on_test_epoch_start(self):
        """Log cache statistics at the start of testing."""
        if self.enable_cache:
            self._log_cache_stats()

    def on_predict_epoch_start(self):
        """Log cache statistics at the start of prediction."""
        if self.enable_cache:
            self._log_cache_stats()
    
    def on_train_epoch_end(self):
        """Log cache performance metrics at the end of training epoch."""
        if self.enable_cache:
            stats = self._get_cache_stats()
            self.log("epoch_cache_hit_rate", stats["hit_rate"], on_epoch=True)
            self.log("epoch_cache_size", stats["cache_size"], on_epoch=True)
            # Log comprehensive stats
            self._log_cache_stats()

    def on_validation_epoch_end(self):
        """Log cache statistics after validation."""
        if self.enable_cache:
            self._log_cache_stats()

    def on_test_epoch_end(self):
        """Log cache statistics after testing."""
        if self.enable_cache:
            self._log_cache_stats()

    def on_predict_epoch_end(self):
        """Log cache statistics after prediction."""
        if self.enable_cache:
            self._log_cache_stats()


def main():
    print("Testing ACT2CosmosPredict2Module...")
    
    try:
        print("Testing model initialization and detailed summary...")
        hparams = {
            "dit_path": "checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps-natten.pt",
            "tokenizer_path": "checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth",
            "text_encoder_path": "checkpoints/google-t5/t5-11b", 
            "learning_rate": 1e-5,
        }
        
        model = ACT2CosmosPredict2Module(
            dit_path=hparams["dit_path"],
            tokenizer_path=hparams["tokenizer_path"],
            text_encoder_path=hparams["text_encoder_path"],
            learning_rate=hparams["learning_rate"],
            cache_size=3,  # Small cache for testing LRU eviction
            enable_cache=True,
            loss_scale=10.0,  # Default EDM loss scaling
        )
        print("✓ Model initialized successfully")
        
        # Test caching functionality
        print("\n" + "="*50)
        print("TESTING OrderedDict LRU CACHE (cache_size=3)")
        print("="*50)
        
        test_prompts = [
            ["prompt_A", "prompt_B"],  # Fill cache with 2 items
            ["prompt_C"],  # Add 3rd item (cache full)
            ["prompt_D"],  # Add 4th item - should evict prompt_A (oldest)
            ["prompt_A"],  # Should be cache miss (was evicted)
            ["prompt_B", "prompt_C"],  # Should be cache hits, move to end
            ["prompt_E"],  # Add 5th item - should evict prompt_D (oldest)
            ["prompt_D"],  # Should be cache miss (was evicted)
        ]
        
        for i, prompts in enumerate(test_prompts):
            print(f"\nTest {i+1}: Encoding batch of {len(prompts)} prompts")
            print(f"Prompts: {prompts}")
            embeddings = model._encode_prompts_with_cache(prompts=prompts, device=torch.device('cpu'))
            print(f"Embeddings shape: {embeddings.shape}")
            
            # Show current cache stats after each batch
            stats = model._get_cache_stats()
            print(f"Cache: {stats['cache_size']}/{stats['max_cache_size']}, "
                  f"Hits: {stats['cache_hits']}, Misses: {stats['cache_misses']}")
            
            # Show cache contents (keys only for debugging)
            cache_keys = list(model._text_cache.keys())
            cache_prompts = [list(model._text_cache.keys())]  # Just show we have the keys
            print(f"Cache contains {len(cache_keys)} items")
        
        # Print final cache statistics for debugging
        model._put_cache_stats()
        
        # Also test logging functionality
        print("\nTesting logging functionality...")
        model._log_cache_stats()
        
        # Print model summary using Lightning utilities
        print("\n" + "="*50)
        print("MODEL SUMMARY (max_depth=2)")
        print("="*50)
        model_summary = summarize(model, max_depth=2)
        print(model_summary)
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    
