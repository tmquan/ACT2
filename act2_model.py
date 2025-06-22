import torch
import torch.nn.functional as F
from lightning import LightningModule
import math
from einops import rearrange
from typing import Any, Tuple

from imaginaire.lazy_config import LazyCall as L
from imaginaire.utils import misc

from cosmos_predict2.configs.base.defaults.ema import EMAConfig
from cosmos_predict2.configs.base.config_video2world import (
    PREDICT2_VIDEO2WORLD_PIPELINE_2B,
    PREDICT2_VIDEO2WORLD_NET_2B,
    ConditioningStrategy,
    Video2WorldPipelineConfig,
    Vid2VidConditioner,
    TextAttr,
    CosmosReason1Config,
    CosmosGuardrailConfig,
)
from cosmos_predict2.conditioner import T2VCondition


from cosmos_predict2.conditioner import BooleanFlag, ReMapkey, TextAttr
from cosmos_predict2.configs.base.defaults.ema import EMAConfig
from cosmos_predict2.configs.vid2vid.defaults.conditioner import Vid2VidConditioner
from cosmos_predict2.models.text2image_dit import SACConfig
from cosmos_predict2.models.video2world_dit import MinimalV1LVGDiT
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.tokenizers.tokenizer import TokenizerInterface
from imaginaire.config import make_freezable
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict

# Cosmos Predict2 Image2Image 2B
PREDICT2_IMAGE2IMAGE_NET_2B = L(MinimalV1LVGDiT)(
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
    sac_config=L(SACConfig)(
        every_n_blocks=1,
        mode="predict2_2b_720",
    ),
)

PREDICT2_IMAGE2IMAGE_PIPELINE_2B = Video2WorldPipelineConfig(
    adjust_video_noise=True,
    conditioner=L(Vid2VidConditioner)(
        fps=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="fps",
            output_key="fps",
        ),
        padding_mask=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="padding_mask",
            output_key="padding_mask",
        ),
        text=L(TextAttr)(
            dropout_rate=0.2,
            input_key=["t5_text_embeddings"],
        ),
        use_video_condition=L(BooleanFlag)(
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
    ema=L(EMAConfig)(enabled=True),  # Enable EMA
    sigma_conditional=0.0001,
    sigma_data=1.0,
    state_ch=16,
    state_t=2, #24,
    text_encoder_class="T5",
    tokenizer=L(TokenizerInterface)(
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

class CosmosVideoPredictionModel(LightningModule):
    """
    LightningModule to wrap the Cosmos-Predict2-2B-Video2World model for post-training.
    This class reuses training logic from cosmos_predict2.models.video2world_model.py for consistency.
    """
    def __init__(self, dit_path: str, text_encoder_path: str, learning_rate: float):
        super().__init__()
        self.save_hyperparameters()
        self.last_prediction = None
        self.loss_reduce = "mean"
        self.loss_scale = 1e2 # From Predict2Video2WorldModelConfig defaults
        if PREDICT2_IMAGE2IMAGE_PIPELINE_2B.adjust_video_noise:
            self.video_noise_multiplier = math.sqrt(PREDICT2_IMAGE2IMAGE_PIPELINE_2B.state_t)
        else:
            self.video_noise_multiplier = 1.0
        
        # Load the pre-trained pipeline from config
        self.pipeline = Video2WorldPipeline.from_config(
            config=PREDICT2_IMAGE2IMAGE_PIPELINE_2B,
            dit_path=dit_path,
            text_encoder_path=text_encoder_path,
            torch_dtype=torch.bfloat16,
        )

        # Freeze the text encoder's parameters and set it to evaluation mode
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.text_encoder.eval()

        # Freeze the tokenizer's parameters and set it to evaluation mode
        self.pipeline.tokenizer.model.model.requires_grad_(False)
        self.pipeline.tokenizer.model.model.eval()

        # Unfreeze the DiT
        self.pipeline.denoising_model().requires_grad_(True)
        self.pipeline.denoising_model().train()

    @torch.no_grad()
    def _full_denoise_step(self, batch: dict) -> torch.Tensor:
        """
        Performs a full, guided denoising loop to generate a video prediction for evaluation.
        This method simulates the complete inference process, including classifier-free guidance.

        Args:
            batch (dict): A dictionary containing the input data, including 'png' for the
                          conditional frame, 'tif' for the target frame (used for shape),
                          and 'txt' for the text prompts.

        Returns:
            torch.Tensor: The predicted target frame as a tensor.
        """
        prompts = batch['txt']
        cond_frame = batch['png']
        target_frame = batch['tif']  # Used as a placeholder for shape
        B, _, H, W = cond_frame.shape

        # --- 1. Prepare data for the pipeline ---
        # The VAE tokenizer requires a temporal dimension of at least 3.
        # We pad the 2-frame input with an additional zero frame.
        zero_frame = torch.zeros_like(target_frame)
        video_placeholder = torch.stack([cond_frame, zero_frame], dim=2)
        video = (video_placeholder * 255.0).to(torch.uint8)
        
        # --- 2. Encode prompts for Classifier-Free Guidance (CFG) ---
        cond_embeddings = self.pipeline.encode_prompt(prompts).to(dtype=self.pipeline.torch_dtype)
        uncond_embeddings = self.pipeline.encode_prompt([""] * B).to(dtype=self.pipeline.torch_dtype)

        # --- 3. Assemble the full data batch ---
        data_batch = {
            'video': video,
            't5_text_embeddings': cond_embeddings,
            'uncond_t5_text_embeddings': uncond_embeddings,
            'dataset_name': 'video_data',
            'num_conditional_frames': 1,
            'fps': torch.ones((B,), device=video.device, dtype=torch.long),
            'padding_mask': torch.zeros(B, 1, H, W, device=video.device, dtype=self.pipeline.torch_dtype),
        }

        # --- 4. Perform the full denoising process ---
        x0_fn = self.pipeline.get_x0_fn_from_batch(data_batch, guidance=7.0, is_negative_prompt=True)

        _T = video.shape[2]
        latent_T = self.pipeline.tokenizer.get_latent_num_frames(_T)
        latent_H = H // self.pipeline.tokenizer.spatial_compression_factor
        latent_W = W // self.pipeline.tokenizer.spatial_compression_factor
        state_shape = [self.pipeline.config.state_ch, latent_T, latent_H, latent_W]

        x_sigma_max = (
            misc.arch_invariant_rand(
                (B,) + tuple(state_shape),
                torch.float32,
                self.device,
                seed=0,
            )
            * self.pipeline.scheduler.config.sigma_max
        )

        scheduler = self.pipeline.scheduler
        num_sampling_step = 35
        scheduler.set_timesteps(num_sampling_step, device=x_sigma_max.device)
        sample = x_sigma_max.to(dtype=torch.float32)
        x0_prev = None

        for i, _ in enumerate(scheduler.timesteps):
            sigma_t = scheduler.sigmas[i].to(sample.device, dtype=torch.float32)
            sigma_in = sigma_t.repeat(sample.shape[0])
            x0_pred = x0_fn(sample, sigma_in)
            sample, x0_prev = scheduler.step(
                x0_pred=x0_pred,
                i=i,
                sample=sample,
                x0_prev=x0_prev,
            )

        sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
        sigma_in = sigma_min.repeat(sample.shape[0])
        output_latents = x0_fn(sample, sigma_in)

        output_video = self.pipeline.decode(output_latents)

        # The output_video contains 3 frames; we return the second one, which is our prediction.
        output_frame = output_video[:, :, 1, :, :]
        return output_frame

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step.
        This involves running one step of the diffusion denoising process and
        calculating the loss between the predicted noise and the actual noise.
        """
        # --- 1. Data Preparation ---
        prompts = batch['txt']
        cond_frame = batch['png']
        target_frame = batch['tif']

        # Create a 2-frame video tensor.
        video = torch.stack([cond_frame, target_frame], dim=2)
        video = (video * 255.0).to(torch.uint8)

        B, C, T, H, W = video.shape
        
        # Encode the text prompt.
        text_embeddings = self.pipeline.encode_prompt(prompts).to(dtype=self.pipeline.torch_dtype)

        # Assemble the batch dictionary for the pipeline.
        data_batch = {
            'video': video,
            't5_text_embeddings': text_embeddings,
            'dataset_name': 'video_data',
            'num_conditional_frames': 1,
            'fps': torch.ones((B,), device=video.device, dtype=torch.long),
            'padding_mask': torch.zeros(B, 1, H, W, device=video.device, dtype=self.pipeline.torch_dtype),
        }
        
        # --- 2. Core Training Logic (reused from video2world_model.py) ---
        # Get the input data to noise and denoise and the corresponding conditioner.
        _, x0_B_C_T_H_W, condition = self.pipeline.get_data_and_condition(data_batch)

        # Sample pertubation noise levels and N(0, 1) noises
        sigma_B_T, epsilon_B_C_T_H_W = self.draw_training_sigma_and_epsilon(x0_B_C_T_H_W.size(), condition)

        output_batch, kendall_loss, _, _ = self.compute_loss_with_epsilon_and_sigma(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
        )

        if self.loss_reduce == "mean":
            loss = kendall_loss.mean() * self.loss_scale
        elif self.loss_reduce == "sum":
            loss = kendall_loss.sum(dim=1).mean() * self.loss_scale
        else:
            raise ValueError(f"Invalid loss_reduce: {self.loss_reduce}")
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/mse_loss', output_batch['mse_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train/edm_loss', output_batch['edm_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        return loss

   
    def draw_training_sigma_and_epsilon(self, x0_size: torch.Size, condition: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        from cosmos_predict2.conditioner import DataType
        batch_size = x0_size[0]
        epsilon = torch.randn(x0_size, device=self.device)
        sigma_B = self.pipeline.scheduler.sample_sigma(batch_size).to(device=self.device)
        sigma_B_1 = rearrange(sigma_B, "b -> b 1")  # add a dimension for T, all frames share the same sigma
        is_video_batch = condition.data_type == DataType.VIDEO

        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B_1 = sigma_B_1 * multiplier
        return sigma_B_1, epsilon

    def get_per_sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.pipeline.sigma_data**2) / (sigma * self.pipeline.sigma_data) ** 2

    def compute_loss_with_epsilon_and_sigma(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: T2VCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
    ) -> Tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss givee epsilon and sigma
        This method is reused from cosmos_predict2/models/video2world_model.py for consistency.
        """
        # Get the mean and stand deviation of the marginal probability distribution.
        mean_B_C_T_H_W, std_B_T = x0_B_C_T_H_W, sigma_B_T
        # Generate noisy observations
        xt_B_C_T_H_W = mean_B_C_T_H_W + epsilon_B_C_T_H_W * rearrange(std_B_T, "b t -> b 1 t 1 1")
        # make prediction
        model_pred = self.pipeline.denoise(xt_B_C_T_H_W, sigma_B_T, condition)
        # loss weights for different noise levels
        weights_per_sigma_B_T = self.get_per_sigma_loss_weights(sigma=sigma_B_T)
        # extra loss mask for each sample, for example, human faces, hands
        pred_mse_B_C_T_H_W = (x0_B_C_T_H_W - model_pred.x0) ** 2
        edm_loss_B_C_T_H_W = pred_mse_B_C_T_H_W * rearrange(weights_per_sigma_B_T, "b t -> b 1 t 1 1")
        kendall_loss = edm_loss_B_C_T_H_W
        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "sigma": sigma_B_T,
            "weights_per_sigma": weights_per_sigma_B_T,
            "condition": condition,
            "model_pred": model_pred,
            "mse_loss": pred_mse_B_C_T_H_W.mean(),
            "edm_loss": edm_loss_B_C_T_H_W.mean(),
            "edm_loss_per_frame": torch.mean(edm_loss_B_C_T_H_W, dim=[1, 3, 4]),
        }
        output_batch["loss"] = kendall_loss.mean()

        return output_batch, kendall_loss, pred_mse_B_C_T_H_W, edm_loss_B_C_T_H_W

    def eval_step(self, batch: dict, batch_idx: int, stage: str) -> None:
        """
        Performs a single evaluation step (for validation or testing).
        This involves a full denoising loop to generate a predicted image,
        followed by calculating a loss against the ground truth.
        """
        # Generate the predicted video by running the full inference loop.
        predicted_frame = self._full_denoise_step(batch)
        if batch_idx == 0:
            with torch.no_grad():
                self.last_prediction =  predicted_frame

        target_frame = batch['tif']
        
        # Calculate the L1 (Mean Absolute Error) loss between the generated
        # image and the ground truth target image.
        eval_loss = F.l1_loss(predicted_frame, target_frame)
        
        # Log the evalidation/test loss.
        self.log(f'{stage}/loss', eval_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        return self.eval_step(batch, batch_idx, "test")
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        # We are fine-tuning the DiT part of the pipeline
        optimizer = torch.optim.AdamW(
            self.pipeline.dit.parameters(), 
            lr=self.hparams.learning_rate
        )
        return optimizer

    # def on_before_optimizer_step(self, optimizer):
    #     if self.pipeline.config.ema.enabled:
    #         beta = self.ema_beta(self.global_step)
    #         self.pipeline.dit_ema_worker.update_average(self.pipeline.dit, self.pipeline.dit_ema, beta=beta)

    # def ema_beta(self, iteration: int) -> float:
    #     """
    #     Calculate the beta value for EMA update.
    #     weights = weights * beta + (1 - beta) * new_weights
    #     """
    #     iteration = iteration + self.pipeline.config.ema.iteration_shift
    #     if iteration < 1:
    #         return 0.0
    #     return (1 - 1 / (iteration + 1)) ** (self.pipeline.ema_exp_coefficient + 1) 