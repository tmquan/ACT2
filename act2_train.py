import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch import seed_everything
from argparse import ArgumentParser

from act2_model import ACT2CosmosPredict2Model
from act2_callbacks import TensorBoardImageCallback
from act2_datamodule import ACT2DataModule

def main(hparams):
    # --- Configuration ---
    seed_everything(42, workers=True)
    
    # --- DataModule ---
    datamodule = ACT2DataModule(
        root_folder=hparams.data_root,
        image_H=512,
        image_W=512,
        micro_batch_size=4,
        train_samples=10000,
        val_samples=400,
        test_samples=400,
    )

    # --- Model ---
    model = ACT2CosmosPredict2Model(
        dit_path=hparams.dit_path,
        text_encoder_path=hparams.text_encoder_path,
        learning_rate=hparams.learning_rate
    )
    
    # --- Callbacks ---
    image_callback = TensorBoardImageCallback()
    progress_bar = RichProgressBar()
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/act2_train",
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=5,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    # --- Trainer ---
    trainer = L.Trainer(
        accelerator="gpu",
        devices=-1,  # Use all available GPUs
        strategy="ddp",  # Standard Distributed Data Parallel
        max_epochs=500,
        precision="bf16-mixed",
        deterministic=True,
        sync_batchnorm=True,  # Synchronize batch normalization across GPUs
        callbacks=[image_callback, progress_bar, checkpoint_callback],
        logger=L.pytorch.loggers.TensorBoardLogger("lightning_logs", name="act2_train"),
    )
    
    # --- Start Training ---
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Add model-specific arguments
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--dit_path", type=str, help="Path to the DiT model weights.",
                        default="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt")
    parser.add_argument("--text_encoder_path", type=str, help="Path to the text encoder model.",
                        default="checkpoints/google-t5/t5-11b")

    # Add datamodule-specific arguments
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data_root", type=str, help="Root directory of the dataset.",
                        default="data/ACT2_raw")

    hparams = parser.parse_args()
    main(hparams)