import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint

from act2_model import CosmosVideoPredictionModel
from act2_datamodule import ACT2DataModule
from act2_callbacks import TensorBoardImageCallback

if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Change this path to the location of your ACT2 dataset files.
    data_root = "data/ACT2_raw" 
    
    # --- DataModule ---
    datamodule = ACT2DataModule(
        root_folder=data_root,
        image_H=512,
        image_W=512,
        micro_batch_size=4,
        num_workers=8,
    )

    # --- Model ---
    model = CosmosVideoPredictionModel(
        dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt",
        text_encoder_path="checkpoints/google-t5/t5-11b",
        learning_rate=1e-5
    )
    
    # --- Callbacks ---
    image_callback = TensorBoardImageCallback()
    progress_bar = RichProgressBar()
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/act2_train",
        filename="{epoch}-{val/loss:.2f}",
        save_top_k=3,
        monitor="val/loss",
        mode="min",
    )

    # --- Trainer ---
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=2000,
        callbacks=[image_callback, progress_bar, checkpoint_callback],
        logger=L.pytorch.loggers.TensorBoardLogger("lightning_logs", name="act2_train"),
    )
    
    # --- Start Training ---
    trainer.fit(model, datamodule=datamodule)