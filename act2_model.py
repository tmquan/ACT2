import torch
from lightning import LightningModule
import math
import random

from cosmos_predict2.models.video2world_model import (
    Predict2Video2WorldModelConfig,
    Predict2ModelManagerConfig,
)
from cosmos_predict2.configs.base.config_video2world import PREDICT2_VIDEO2WORLD_PIPELINE_2B
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.auxiliary.text_encoder import CosmosT5TextEncoder
from cosmos_predict2.module.denoise_prediction import DenoisePrediction
from cosmos_predict2.conditioner import T2VCondition, DataType, ReMapkey, BooleanFlag
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

# Try to import kornia for HSV conversion, fallback to pure PyTorch if not available
try:
    import kornia
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("Warning: kornia not available, using pure PyTorch HSV conversion")
    print("Install kornia with: pip install kornia")


def rgb_to_hsv_pytorch_simple(rgb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Simple RGB to HSV conversion using pure PyTorch as fallback.
    rgb: tensor with values in [0, 1]
    Returns HSV tensor, H in [0, 2π], S and V in [0, 1]
    """
    r, g, b = rgb.unbind(dim=-3)
    
    max_rgb, argmax_rgb = rgb.max(dim=-3)
    min_rgb = rgb.min(dim=-3)[0]
    delta = max_rgb - min_rgb
    
    # Hue calculation
    h = torch.zeros_like(max_rgb)
    
    # Avoid division by zero
    mask = delta > eps
    
    # Red is max
    idx = (argmax_rgb == 0) & mask
    h[idx] = (60.0 * ((g[idx] - b[idx]) / delta[idx]) + 360.0) % 360.0
    
    # Green is max  
    idx = (argmax_rgb == 1) & mask
    h[idx] = 60.0 * ((b[idx] - r[idx]) / delta[idx]) + 120.0
    
    # Blue is max
    idx = (argmax_rgb == 2) & mask
    h[idx] = 60.0 * ((r[idx] - g[idx]) / delta[idx]) + 240.0
    
    # Convert to radians (0 to 2π)
    h = h * math.pi / 180.0
    
    # Saturation and Value
    s = torch.where(max_rgb > eps, delta / max_rgb, torch.zeros_like(max_rgb))
    v = max_rgb
    
    return torch.stack([h, s, v], dim=-3)

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
    state_t=2, #24,
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
    def __init__(self, dit_path: str, text_encoder_path: str, learning_rate: float, 
                 hsv_weight: float = 0.1, 
                 hue_weight: float = 1.0, 
                 sat_weight: float = 1.0, 
                 val_weight: float = 1.0):
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

        self.pipe.text_encoder = CosmosT5TextEncoder(device="cpu", cache_dir=self.hparams.text_encoder_path)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.text_encoder.eval()

        self.loss_reduce = "mean"
        self.loss_scale = 10.0
        self.video_noise_multiplier = math.sqrt(2) 

        # HSV loss parameters
        self.hsv_weight = hsv_weight
        self.hue_weight = hue_weight
        self.sat_weight = sat_weight
        self.val_weight = val_weight

        self.last_prediction = None

    def setup(self, stage: str):
        if stage == "fit":
            # Create text encoder on CPU and keep it there to save GPU memory
            self.pipe.text_encoder = CosmosT5TextEncoder(device="cpu", cache_dir=self.hparams.text_encoder_path)
            self.pipe.text_encoder.requires_grad_(False)
            self.pipe.text_encoder.eval()

            # Ensure tokenizer is on the correct device for distributed training
            device = next(self.parameters()).device
            self._ensure_models_on_device(device, move_text_encoder=False, move_tokenizer=True)

            # Freeze the tokenizer's parameters and set it to evaluation mode
            self.pipe.tokenizer.model.model.requires_grad_(False)
            self.pipe.tokenizer.model.model.eval()

            # Unfreeze the DiT
            self.pipe.denoising_model().requires_grad_(True)
            self.pipe.denoising_model().train()

    def _ensure_models_on_device(self, device='cpu', move_text_encoder=True, move_tokenizer=False):
        """Manage device placement for text encoder and tokenizer
        
        Args:
            device: Target device (default: 'cpu')
            move_text_encoder: Whether to move text encoder (default: True)
            move_tokenizer: Whether to move tokenizer (default: False)
        """
        if move_text_encoder and hasattr(self.pipe, 'text_encoder') and self.pipe.text_encoder is not None:
            self.pipe.text_encoder = self.pipe.text_encoder.to(device)
        
        if move_tokenizer and hasattr(self.pipe, 'tokenizer') and hasattr(self.pipe.tokenizer, 'model'):
            if hasattr(self.pipe.tokenizer.model, 'model'):
                self.pipe.tokenizer.model.model = self.pipe.tokenizer.model.model.to(device)

    def process_batch(self, batch: dict) -> dict:
        # Keep text encoder on CPU to save GPU memory
        self._ensure_models_on_device()
        
        prompts = batch["txt"]
        if self.training:
            shuffled_prompts = []
            for prompt in prompts:
                words = prompt.split(' ')
                random.shuffle(words)
                shuffled_prompts.append(' '.join(words))
            prompts = shuffled_prompts
            batch['txt'] = prompts

        cond_frame = batch["png"]
        target_frame = batch["tif"]
        video = torch.stack([cond_frame, target_frame], dim=2)
        video = (video * 255.0).to(torch.uint8)
        
        # Ensure tokenizer is on the same device as the video data
        self._ensure_models_on_device(video.device, move_text_encoder=False, move_tokenizer=True)
        B, C, T, H, W = video.shape
        # Ensure text embeddings are on the same device as the video tensor
        text_embeddings = self.pipe.encode_prompt(prompts)
        text_embeddings = text_embeddings.to(dtype=self.precision, device=video.device)
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
        # Use the device of the input tensor instead of self.device for DDP compatibility
        device = next(self.parameters()).device
        epsilon = torch.randn(x0_size, device=device)
        sigma_B = self.pipe.scheduler.sample_sigma(batch_size).to(device=device)
        sigma_B_1 = rearrange(sigma_B, "b -> b 1")
        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B_1 = sigma_B_1 * multiplier
        return sigma_B_1, epsilon

    def get_per_sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.pipe.sigma_data**2) / (sigma * self.pipe.sigma_data) ** 2

    def convert_to_rgb_range(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor from [-1, 1] range to [0, 1] range for HSV conversion.
        """
        return (x + 1.0) / 2.0

    def compute_loss(self, x0, condition, epsilon, sigma) -> tuple[dict, torch.Tensor]:
        mean, std = x0, sigma
        xt = mean + epsilon * rearrange(std, "b t -> b 1 t 1 1")
        out_pred = self.pipe.denoise(xt, sigma, condition)
        weights = self.get_per_sigma_loss_weights(sigma=sigma)
        
        # Standard RGB MSE loss
        mse_pred = (x0 - out_pred.x0) ** 2
        edm_loss = mse_pred * rearrange(weights, "b t -> b 1 t 1 1")
        
        # HSV domain loss
        hsv_loss = torch.tensor(0.0, device=x0.device, dtype=x0.dtype)
        if self.hsv_weight > 0:
            try:
                # Convert from model space to RGB [0, 1] range
                rgb_true = self.convert_to_rgb_range(x0)
                rgb_pred = self.convert_to_rgb_range(out_pred.x0)
                
                # Convert to HSV space
                if KORNIA_AVAILABLE:
                    hsv_true = kornia.color.rgb_to_hsv(rgb_true)
                    hsv_pred = kornia.color.rgb_to_hsv(rgb_pred)
                else:
                    hsv_true = rgb_to_hsv_pytorch_simple(rgb_true)
                    hsv_pred = rgb_to_hsv_pytorch_simple(rgb_pred)
                
                # Simple MSE loss in HSV space
                hsv_loss = ((hsv_true - hsv_pred) ** 2)
            except Exception as e:
                print(f"Warning: HSV loss computation failed: {e}")
                hsv_loss = torch.tensor(0.0, device=x0.device, dtype=x0.dtype)
        
        output_batch = {
            "out_pred": out_pred,
            "mse_loss": mse_pred.mean(),
            "edm_loss": edm_loss.mean(),
            "hsv_loss": hsv_loss.mean(),
        }
        
        # Combine losses - ensure both are on the same device
        hsv_loss = hsv_loss.to(device=edm_loss.device)
        ret_loss = edm_loss + self.hsv_weight * hsv_loss
        
        return output_batch, ret_loss

    def core_step(self, data_batch):
        _, x0, condition = self.pipe.get_data_and_condition(data_batch)
        sigma, epsilon = self.draw_training_sigma_and_epsilon(x0.size(), condition.data_type == DataType.VIDEO)
        x0, condition, epsilon, sigma = self.pipe.broadcast_split_for_model_parallelsim(x0, condition, epsilon, sigma)
        output, loss = self.compute_loss(x0, condition, epsilon, sigma)
        
        if self.loss_reduce == "mean":
            loss = loss.mean() * self.loss_scale
        else:
            loss = loss.sum(dim=1).mean() * self.loss_scale
        return output, loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        data_batch = self.process_batch(batch)
        output_batch, loss = self.core_step(data_batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mse_loss', output_batch['mse_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_edm_loss', output_batch['edm_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_hsv_loss', output_batch['hsv_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if batch_idx == 0:
            self._generate_denoised_image(data_batch)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        data_batch = self.process_batch(batch)
        output_batch, loss = self.core_step(data_batch)
        # Log the validation/test loss.
        self.log(f'val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'val_hsv_loss', output_batch['hsv_loss'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        if batch_idx == 0:
            self._generate_denoised_image(data_batch)
        return loss
    
    def test_step(self, batch: dict, batch_idx: int) -> None:
        data_batch = self.process_batch(batch)
        _, loss = self.core_step(data_batch)
        self.log(f'test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def _generate_denoised_image(self, data_batch: dict):
        with torch.no_grad():
            # Use the device of model parameters for DDP compatibility
            device = next(self.parameters()).device
            # Ensure tokenizer is on the correct device
            self._ensure_models_on_device(device, move_text_encoder=False, move_tokenizer=True)
            
            x0_fn = self.pipe.get_x0_fn_from_batch(data_batch, guidance=7.0, is_negative_prompt=False)
            _T, _H, _W = data_batch["video"].shape[-3:]
            state_shape = [
                self.pipe.config.state_ch,
                self.pipe.tokenizer.get_latent_num_frames(_T),
                _H // self.pipe.tokenizer.spatial_compression_factor,
                _W // self.pipe.tokenizer.spatial_compression_factor,
            ]
            x_sigma_max = misc.arch_invariant_rand(
                (data_batch["video"].shape[0],) + tuple(state_shape), torch.float32, device, 0
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
    
    