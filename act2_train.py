import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch import seed_everything
from argparse import ArgumentParser

from act2_module import ACT2CosmosPredict2Module
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
        train_samples=4000,
        val_samples=800,
        test_samples=800,
        num_workers=64,
        pin_memory=True,
        persistent_workers=True,
        use_thread_dataloader=True,
        cache_rate=1.0,
        prefetch_factor=16,
        cache_num_workers=16,
        copy_cache=False,
        cache_strategy="memory",
        cache_dir="./cache_test",
    )

    # --- Model ---
    model = ACT2CosmosPredict2Module(
        dit_path=hparams.dit_path,
        tokenizer_path=hparams.tokenizer_path,
        text_encoder_path=hparams.text_encoder_path,
        learning_rate=hparams.learning_rate,
        hsv_weight=hparams.hsv_weight,
        img_weight=hparams.img_weight,
        cache_size=hparams.cache_size,
        enable_cache=hparams.enable_cache,
    )
    
    # --- Callbacks ---
    image_callback = TensorBoardImageCallback()
    progress_bar = RichProgressBar()
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/act2_train",
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=3,
        save_last=True,
        monitor="val_loss",
        mode="min",
    )

    # --- Trainer ---
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=300,
        precision="bf16-mixed",
        deterministic=True,
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
    parser.add_argument("--tokenizer_path", type=str, help="Path to the tokenizer model.",
                        default="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth")
    parser.add_argument("--text_encoder_path", type=str, help="Path to the text encoder model.",
                        default="checkpoints/google-t5/t5-11b")
    parser.add_argument("--hsv_weight", type=float, default=0.1, help="Weight for HSV loss.")
    parser.add_argument("--img_weight", type=float, default=0.1, help="Weight for IMG loss.")
    
    # Add cache-specific arguments
    parser.add_argument("--cache_size", type=int, default=10000, help="Size of text embedding cache.")
    parser.add_argument("--enable_cache", action="store_true", default=True, help="Enable text embedding caching.")
    parser.add_argument("--disable_cache", action="store_true", default=False, help="Disable text embedding caching.")

    # Add datamodule-specific arguments
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data_root", type=str, help="Root directory of the dataset.",
                        default="data/ACT2_raw")

    hparams = parser.parse_args()
    
    # Handle cache enable/disable logic
    if hparams.disable_cache:
        hparams.enable_cache = False
    
    main(hparams)