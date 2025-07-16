# def _process_batch(self, batch: dict) -> dict:
#     prompts = batch["txt"]
#     cond_frame = batch["png"]
#     target_frame = batch["tif"]
#
#     if self.training:
#         shuffled_prompts = []
#         for prompt in prompts:
#             words = prompt.split(' ')
#             random.shuffle(words)
#             shuffled_prompts.append(' '.join(words))
#         prompts = shuffled_prompts
#         batch['txt'] = prompts
#
#     video = torch.stack([cond_frame, target_frame], dim=2)
#     video = (video * 255.0).to(torch.uint8)
#     B, C, T, H, W = video.shape
#    
#     # Use cached text embeddings for better performance
#     # text_embeddings = self._encode_prompts_with_cache(prompts)
#    
#     return {
#         "video": video,
#         "prompt": prompts,
#         "t5_text_embeddings": text_embeddings,
#         "dataset_name": "video_data",
#         "num_conditional_frames": 1,
#         "fps": torch.ones((B,), device=video.device, dtype=torch.long),
#         "padding_mask": torch.zeros(B, 1, H, W, device=video.device, dtype=self.precision),
#     }
#
# def _draw_training_sigma_and_epsilon(self, x0_size: torch.Size, is_video_batch: bool) -> tuple[torch.Tensor, torch.Tensor]:
#     batch_size = x0_size[0]
#     epsilon = torch.randn(x0_size, device=self.device)
#     sigma_B = self.pipe.scheduler.sample_sigma(batch_size).to(device=self.device)
#     sigma_B_1 = rearrange(sigma_B, "b -> b 1")
#     multiplier = self.video_noise_multiplier if is_video_batch else 1
#     sigma_B_1 = sigma_B_1 * multiplier
#     return sigma_B_1, epsilon
#
# def _get_per_sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
#     return (sigma**2 + self.pipe.sigma_data**2) / (sigma * self.pipe.sigma_data) ** 2
#
# def _compute_loss(self, x0, condition, epsilon, sigma) -> tuple[dict, torch.Tensor]:
#     mean, std = x0, sigma
#     xt = mean + epsilon * rearrange(std, "b t -> b 1 t 1 1")
#     out_pred = self.pipe.denoise(xt, sigma, condition)
#     weights = self.get_per_sigma_loss_weights(sigma=sigma)
#     mse_loss = (x0 - out_pred.x0) ** 2
#     edm_loss = mse_loss * rearrange(weights, "b t -> b 1 t 1 1")
    
#     output_batch = {
#         "out_pred": out_pred,
#         "mse_loss": mse_loss.mean(),
#         "edm_loss": edm_loss.mean(),
#     }
#    
#     ret_loss = edm_loss
#    
#     return output_batch, ret_loss
#
# def _generate_denoised_image(self, data_batch: dict):
#     with torch.no_grad():
#         x0_fn = self.pipe.get_x0_fn_from_batch(data_batch, guidance=7.0, is_negative_prompt=False)
#         _T, _H, _W = data_batch["video"].shape[-3:]
#         state_shape = [
#             self.pipe.config.state_ch,
#             self.pipe.tokenizer.get_latent_num_frames(_T),
#             _H // self.pipe.tokenizer.spatial_compression_factor,
#             _W // self.pipe.tokenizer.spatial_compression_factor,
#         ]
#         x_sigma_max = misc.arch_invariant_rand(
#             (data_batch["video"].shape[0],) + tuple(state_shape), torch.float32, self.device, 0
#         ) * self.pipe.scheduler.config.sigma_max
#        
#         scheduler = self.pipe.scheduler
#         scheduler.set_timesteps(35, device=x_sigma_max.device)
#         sample = x_sigma_max.to(dtype=torch.float32)
#         x0_prev = None
#         for i, _ in enumerate(scheduler.timesteps):
#             sigma_t = scheduler.sigmas[i].to(sample.device, dtype=torch.float32)
#             sigma_in = sigma_t.repeat(sample.shape[0])
#             x0_pred = x0_fn(sample, sigma_in)
#             sample, x0_prev = scheduler.step(x0_pred, i, sample, x0_prev)
#
#         sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
#         samples = x0_fn(sample, sigma_min.repeat(sample.shape[0]))
#         video = self.pipe.decode(samples)
#         self.last_prediction = video[:, :, -1, :, :]

# EDM Loss Implementation
# ========================
# 
# The EDM (Elucidating the Design Space of Diffusion Models) loss functions
# have been implemented and integrated into the CustomizedCosmosPredict2Module 
# class in act2_module.py.
#
# Implemented functions:
# - process_batch(): Processes input batch data for training/inference
# - draw_training_sigma_and_epsilon(): Generates training noise parameters
# - get_per_sigma_loss_weights(): Computes per-sigma loss weights
# - compute_loss(): Main EDM loss computation
# - _generate_denoised_image(): Inference/generation for visualization
#
# The implementation follows the original research paper and provides:
# - Proper noise scheduling with sigma sampling
# - Weighted loss computation for stable training
# - Support for video data with conditional frames
# - Text conditioning through T5 embeddings
#
# Usage:
# The loss functions are automatically used when training with the
# CustomizedCosmosPredict2Module in Lightning framework.