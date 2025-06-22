import lightning as L
from lightning.pytorch.callbacks import Callback
import torch
import torchvision

class TensorBoardImageCallback(Callback):
    """
    A callback to log input images, prompts, ground truth, and predictions to TensorBoard
    at the end of training and validation epochs.
    """

    def __init__(self):
        super().__init__()
        self.train_batch_data = None
        self.val_batch_data = None

    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: dict, batch: dict, batch_idx: int
    ) -> None:
        """Stores the first batch and prediction of the training epoch."""
        # Only store data for the first batch
        if batch_idx == 0:
            batch_for_storage = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_for_storage[k] = v.cpu()
                elif isinstance(v, list):
                    batch_for_storage[k] = v
            
            self.train_batch_data = {
                'batch': batch_for_storage,
                'prediction': pl_module.last_prediction.cpu()
            }


    def on_validation_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: dict, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Stores the first batch and prediction of the validation epoch."""
        # Only store data for the first batch
        if batch_idx == 0:
            batch_for_storage = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_for_storage[k] = v.cpu()
                elif isinstance(v, list):
                    batch_for_storage[k] = v

            self.val_batch_data = {
                'batch': batch_for_storage,
                'prediction': pl_module.last_prediction.cpu()
            }

    def _log_images(self, trainer: L.Trainer, batch_data: dict, stage: str) -> None:
        """Helper function to log images to TensorBoard."""
        if not batch_data:
            return

        try:
            batch = batch_data['batch']
            # The prediction is the predicted frame, with shape (B, C, H, W).
            pred_vid = batch_data['prediction']
            logger = trainer.logger.experiment
            
            # Get full batches for logging
            input_batch = batch['png']  # Shape: (B, C, H, W)
            gt_batch = batch['tif']     # Shape: (B, C, H, W)
            
            # The model prediction is already the single predicted frame.
            pred_tif = pred_vid

            # For TensorBoard, we'll log the first sample of the batch for clarity.
            input_img = input_batch
            gt_img = gt_batch
            pred_img = (pred_tif.clamp(-1, 1) + 1) / 2
            prompt = batch['txt']

            # Create a grid for comparison
            # We want to show input, gt, and prediction for each item in the batch
            # Concatenate along the batch dimension
            grid_tensor = torch.cat([input_img, gt_img, pred_img], dim=3)
            grid = torchvision.utils.make_grid(grid_tensor, nrow=1, padding=0)
            
            logger.add_image(
                f"{stage}/comparison (Input_GT_Pred)", 
                grid, 
                global_step=trainer.current_epoch
            )
            
            logger.add_text(
                f"{stage}/prompt", 
                "\n".join(prompt), 
                global_step=trainer.current_epoch
            )

        except Exception as e:
            print(f"Error in TensorBoardImageCallback ({stage}): {e}")

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Logs images at the end of the training epoch."""
        self._log_images(trainer, self.train_batch_data, "train")
        self.train_batch_data = None # Clear after use

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Logs images at the end of the validation epoch."""
        self._log_images(trainer, self.val_batch_data, "val")
        self.val_batch_data = None # Clear after use 