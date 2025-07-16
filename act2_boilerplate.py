# Placeholder classes for demonstration purposes if the actual modules are not available
class PlaceholderDiT(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dummy_layer = nn.Linear(10, 10)
    def forward(self, x, t, context):
        return x

class PlaceholderVAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dummy_layer = nn.Linear(10, 10)
    def encode(self, x):
        return self.dummy_layer(x),
    def decode(self, x):
        return self.dummy_layer(x)

class PlaceholderTextEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dummy_layer = nn.Linear(10, 10)
    def forward(self, x):
        return self.dummy_layer(x)

class NVLightningModule(LightningModule):
    """
    A PyTorch Lightning module for the ACT2 model, designed for fine-tuning
    and scalable distributed training.

    This module encapsulates the entire training, validation, and testing logic,
    adhering to Lightning best practices. It implements a specific fine-tuning
    strategy where the Diffusion Transformer (DiT) is trained, while the VAE and
    text encoder components are frozen.
    """
    def __init__(self, model_cfg: DictConfig, train_cfg: DictConfig):
        super().__init__()
        # Save hyperparameters, making them accessible via self.hparams
        # This is useful for logging and checkpointing.
        self.save_hyperparameters()

        # Store configurations for easy access
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

        # --- 1. Instantiate Model Components ---
        # Instantiate the core components from the configuration.
        # NOTE: Replace placeholder classes with actual model imports.
        # self.vae = AutoencoderKL(**model_cfg.vae)
        # self.text_encoder = TextEncoder(**model_cfg.text_encoder)
        # self.model = DiT(**model_cfg.dit)
        self.vae = PlaceholderVAE()
        self.text_encoder = PlaceholderTextEncoder()
        self.model = PlaceholderDiT() # In a real scenario, this would be the DiT

        # Instantiate loss function
        self.criterion = nn.MSELoss()

        # --- 2. Implement Fine-Tuning Strategy: Parameter Freezing ---
        # A robust "freeze-all, then unfreeze-selected" approach is used.
        # This prevents accidental training of unintended layers.

        # Step 2a: Freeze all parameters in the entire module by default.
        for param in self.parameters():
            param.requires_grad = False

        # Step 2b: Selectively unfreeze the parameters of the DiT model.
        # This ensures only the DiT component will be updated during training.
        for param in self.model.parameters(): # Assuming self.model is the DiT
            param.requires_grad = True

        # Note: The original implementation's manual EMA logic is removed.
        # EMA should be handled by a dedicated Lightning Callback for clean separation of concerns.

    def forward(self, z: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for inference. This method should contain only
        the core model logic.

        Args:
            z (torch.Tensor): The latent representation of the input.
            t (torch.Tensor): The timestep tensor.
            context (torch.Tensor): The conditioning context (e.g., text embeddings).

        Returns:
            torch.Tensor: The output of the DiT model (predicted noise).
        """
        return self.model(z, t, context=context)

    def _shared_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        A shared logic block for training, validation, and test steps to avoid code duplication.
        """
        images = batch['image']
        text_tokens = batch['text_tokens']

        # Encode images into the latent space using the frozen VAE.
        # The operation is wrapped in no_grad() as VAE is frozen.
        with torch.no_grad():
            # The VAE output is a tuple, we take the first element (the latent representation)
            z = self.vae.encode(images)

        # Generate text embeddings using the frozen text encoder.
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_tokens)

        # Prepare for DiT forward pass
        # Sample random timesteps
        t = torch.randint(0, self.model_cfg.dit.num_timesteps, (z.shape,), device=self.device).long()

        # Sample noise and create the noisy latent z_t
        noise = torch.randn_like(z)
        z_t = z * (1. - self.train_cfg.noise_schedule[t]).view(-1, 1, 1, 1) + \
              noise * self.train_cfg.noise_schedule[t].view(-1, 1, 1, 1)

        # Predict the noise using the DiT model
        predicted_noise = self.forward(z_t, t, context=text_embeddings)

        # Calculate the loss
        loss = self.criterion(predicted_noise, noise)
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step.
        """
        loss = self._shared_step(batch, batch_idx)
        
        # Log the training loss. `prog_bar=True` shows it in the progress bar.
        # `on_step=True` logs it at every step, `on_epoch=True` logs the average at epoch end.
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Performs a single validation step.
        """
        loss = self._shared_step(batch, batch_idx)
        
        # Log the validation loss.
        # `sync_dist=True` is crucial for correct metric aggregation in DDP mode.
        # It ensures that the loss from all GPUs is gathered and averaged.
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True, logger=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Performs a single test step.
        """
        loss = self._shared_step(batch, batch_idx)
        
        # Log the test loss, also ensuring synchronization across devices.
        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True, logger=True)

    def on_train_epoch_end(self):
        """
        Hook called at the end of the training epoch. Can be used for custom logic.
        """
        pass

    def on_validation_epoch_end(self):
        """
        Hook called at the end of the validation epoch.
        """
        pass

    def on_test_epoch_end(self):
        """
        Hook called at the end of the test epoch.
        """
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures the optimizer and learning rate scheduler. This is a critical
        method for ensuring the fine-tuning strategy is correctly implemented.
        """
        # --- The Linchpin of Fine-Tuning and DDP Compatibility ---
        # We must filter the parameters to ensure the optimizer only receives
        # the parameters that are meant to be trained (i.e., requires_grad=True).
        # This is not just an optimization; it is a requirement for DDP to work
        # correctly without `find_unused_parameters=True`, as DDP would otherwise
        # wait for gradients from frozen parameters that are never computed.
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())

        optimizer = AdamW(
            trainable_params,
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay
        )

        # Example of a simple learning rate scheduler (e.g., linear warmup and decay)
        def lr_lambda(current_step: int):
            if current_step < self.train_cfg.warmup_steps:
                return float(current_step) / float(max(1, self.train_cfg.warmup_steps))
            # Implement decay logic here if needed, otherwise constant
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Call the scheduler at every step
                "frequency": 1,
            },
        }
