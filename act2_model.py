import torch
import torch.nn.functional as F
import lightning as L
from cosmos_predict2.configs.base.config_video2world import PREDICT2_VIDEO2WORLD_PIPELINE_2B
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from imaginaire.utils import misc


class CosmosVideoPredictionModel(L.LightningModule):
    """
    LightningModule to wrap the Cosmos-Predict2-2B-Video2World model for post-training.
    """
    def __init__(self, dit_path: str, text_encoder_path: str, learning_rate: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.last_prediction = None
        
        # Load the pre-trained pipeline from config
        self.pipeline = Video2WorldPipeline.from_config(
            config=PREDICT2_VIDEO2WORLD_PIPELINE_2B,
            dit_path=dit_path,
            text_encoder_path=text_encoder_path,
            torch_dtype=torch.bfloat16
        )
        
        # Override config to force 2-frame video processing
        self.pipeline.config.state_t = self.pipeline.tokenizer.get_latent_num_frames(2)

        # Freeze the text encoder's parameters and set it to evaluation mode
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.text_encoder.eval()

        # # Freeze the tokenizer's parameters and set it to evaluation mode
        # self.pipeline.tokenizer.model.requires_grad_(False)
        # self.pipeline.tokenizer.model.eval()

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
        # prompts = batch['txt']
        cond_frame = batch['png']
        target_frame = batch['tif']  # Used as a placeholder for shape
        B, _, H, W = cond_frame.shape

        # --- 1. Prepare data for the pipeline ---
        video_placeholder = torch.stack([cond_frame, torch.zeros_like(target_frame)], dim=2)
        video = (video_placeholder * 255.0).to(torch.uint8)
        
        # --- 2. Encode prompts for Classifier-Free Guidance (CFG) ---
        cond_embeddings = self.pipeline.encode_prompt(batch['txt']).to(dtype=self.pipeline.torch_dtype)
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
        output_frame = output_video[:, :, -1, :, :]
        return output_frame

    def _iter_denoise_step(self, batch: dict):
        """
        A common step for training, performing a single-step denoising process.
        This is the core of the diffusion model training.

        Returns:
            torch.Tensor: The calculated loss for this training step.
        """
        cond_frame = batch['png']
        target_frame = batch['tif']

        # Create a 2-frame video tensor [cond_frame, target_frame] and prepare for pipeline.
        video = torch.stack([cond_frame, target_frame], dim=2)
        video = (video * 255.0).to(torch.uint8)

        B, C, T, H, W = video.shape
        
        # Encode the text prompt.
        text_embeddings = self.pipeline.encode_prompt(batch['txt']).to(dtype=self.pipeline.torch_dtype)

        # Assemble the batch dictionary.
        data_batch = {
            'video': video,
            't5_text_embeddings': text_embeddings,
            'dataset_name': 'video_data',
            'num_conditional_frames': 1,
            'fps': torch.ones((B,), device=video.device, dtype=torch.long),
            'padding_mask': torch.zeros(B, 1, H, W, device=video.device, dtype=self.pipeline.torch_dtype),
        }

        # 1. Get clean latents (x0) for the loss target.
        _, x0, _ = self.pipeline.get_data_and_condition(data_batch)

        # 2. Sample random timesteps (t) and noise (epsilon).
        t = torch.rand(B, device=self.device).to(x0.dtype)
        epsilon = torch.randn_like(x0)

        # 3. Create noisy latents (xt) using the Rectified Flow formula.
        t_reshaped = t.view(B, *([1] * (x0.dim() - 1)))
        xt = (1 - t_reshaped) * x0 + t_reshaped * epsilon
        
        # 4. Get the prediction function from the pipeline (with no guidance for training).
        x0_fn = self.pipeline.get_x0_fn_from_batch(data_batch, guidance=0.0, is_negative_prompt=False)

        # 5. Predict the clean latent from the noisy latent.
        # The timestep `t` is used as the noise level `sigma`.
        sigma = t.view(B, 1)
        x0_pred = x0_fn(xt, sigma)
        
        # # 6. Calculate the Mean Squared Error loss against the original clean latent.
        # loss = F.mse_loss(x0_pred, x0)

        # 6. Derive the predicted noise (`eps_pred`) from the predicted clean latent (`x0_pred`).
        # The relationship is: x0_pred = xt - t * eps_pred  =>  eps_pred = (xt - x0_pred) / t
        eps_pred = (xt - x0_pred) / t_reshaped

        # 7. Calculate the Mean Squared Error loss against the original noise.
        loss = F.mse_loss(eps_pred, epsilon)
        
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step.
        This involves running one step of the diffusion denoising process and
        calculating the loss between the predicted noise and the actual noise.
        """
        predicted_frame = self._full_denoise_step(batch)
        if batch_idx == 0:
            with torch.no_grad():
                self.last_prediction =  predicted_frame

        # The _iter_denoise_step method encapsulates the core training logic.
        loss = self._iter_denoise_step(batch)
        
        # Log the training loss for monitoring.
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
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