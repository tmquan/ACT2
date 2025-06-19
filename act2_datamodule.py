# act2_datamodule.py
import os
import glob
from typing import List, Optional, Dict

import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from monai.transforms import (
    apply_transform,
    Compose,
    RandAxisFlipDict,
    RandRotate90Dict,
    EnsureChannelFirstDict,
    ScaleIntensityRangeDict,
    ToTensorDict,
    LoadImageDict,
    RandSpatialCropDict,
)

# Set a higher limit for image pixels if dealing with large TIFs
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


class ACT2Dataset(Dataset):
    """A PyTorch Dataset for ACT2 data, handling TIF/PNG/TXT triplets."""
    def __init__(self, data: List[Dict], transform: Compose, image_key: str = 'images', hint_key: str = 'hint', txt_key: str = 'txt'):
        self.data = data
        self.transform = transform
        self.image_key = image_key
        self.hint_key = hint_key
        self.txt_key = txt_key


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, any]:
        """
        Retrieves a sample, applies transforms, and returns it.
        The sample dictionary keys are determined by the MONAI pipeline,
        and we map them to what the model expects ('hint', 'images', 'txt').
        """
        sample = self.data[index]
        
        # Read text content from the txt file
        with open(sample["txt"], 'r') as f:
            txt_content = f.read().strip()
        
        # This dictionary, with keys matching the file types, goes into the transform pipeline.
        item_to_transform = {
            'tif': sample['tif'],
            'png': sample['png'],
        }
        
        # The MONAI transform is applied first
        transformed_item = apply_transform(self.transform, item_to_transform)
        
        # The model expects 'hint', 'images', and 'txt' keys.
        # Here we map our dataset keys ('tif', 'png') to the model's expected keys.
        final_sample = {
            self.hint_key: transformed_item['tif'],
            self.image_key: transformed_item['png'],
            self.txt_key: txt_content,
        }

        return final_sample

class ACT2DataModule(LightningDataModule):
    """
    A PyTorch Lightning DataModule for the ACT2 dataset, loading data directly from a folder.
    """
    def __init__(
        self,
        root_folder: str,
        image_H: int = 512,
        image_W: int = 512,
        micro_batch_size: int = 1,
        global_batch_size: int = 8,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        """
        Initializes the DataModule.

        Args:
            root_folder (str): The root directory containing TIF, PNG, and TXT files.
            image_H (int): The target height for images after cropping.
            image_W (int): The target width for images after cropping.
            micro_batch_size (int): Batch size per GPU.
            global_batch_size (int): Total batch size across all GPUs.
            num_workers (int): Number of workers for the DataLoader.
            pin_memory (bool): Whether to use pinned memory.
            persistent_workers (bool): Whether to keep worker processes alive.
        """
        super().__init__()
        self.root_folder = root_folder
        self.image_H = image_H
        self.image_W = image_W
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        # Define MONAI transforms
        self.train_transforms = Compose([
            LoadImageDict(keys=["tif", "png"], image_only=True),
            EnsureChannelFirstDict(keys=["tif", "png"]),
            RandSpatialCropDict(keys=["tif", "png"], roi_size=(self.image_H, self.image_W), random_size=False),
            RandAxisFlipDict(keys=["tif", "png"], prob=0.75),
            RandRotate90Dict(keys=["tif", "png"], prob=0.75),
            ScaleIntensityRangeDict(keys=["tif", "png"], clip=True, a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
            ToTensorDict(keys=["tif", "png"]),
        ])

        self.val_transforms = Compose([
            LoadImageDict(keys=["tif", "png"], image_only=True),
            EnsureChannelFirstDict(keys=["tif", "png"]),
            RandSpatialCropDict(keys=["tif", "png"], roi_size=(self.image_H, self.image_W), random_size=False),
            RandAxisFlipDict(keys=["tif", "png"], prob=0.00),
            RandRotate90Dict(keys=["tif", "png"], prob=0.00),
            ScaleIntensityRangeDict(keys=["tif", "png"], clip=True, a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
            ToTensorDict(keys=["tif", "png"]),
        ])


        # Lists to hold file path triplets
        self.train_tuples: List[Dict] = []
        self.val_tuples: List[Dict] = []
        self.test_tuples: List[Dict] = []

    def _load_samples(self):
        """
        Loads image samples (tif, png, txt) from the root folder by finding matching basenames.
        Splits data based on 'Subject' in the filename.
        """
        print(f"Loading image samples from: {self.root_folder}")
        
        if not os.path.isdir(self.root_folder):
            print(f"Warning: Data directory not found at {self.root_folder}")
            return
            
        all_files = glob.glob(os.path.join(self.root_folder, "*"))
        
        # Group files by their base name (e.g., 'Subject2_2_1_1')
        file_groups: Dict[str, Dict[str, str]] = {}
        for file_path in all_files:
            if os.path.isfile(file_path):
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                ext = os.path.splitext(file_path)[1].lower()
                
                if base_name not in file_groups:
                    file_groups[base_name] = {}
                file_groups[base_name][ext] = file_path
        
        # Create samples where we have a complete triplet of (tif, png, txt)
        for base_name, files in file_groups.items():
            if '.tif' in files and '.png' in files and '.txt' in files:
                sample = {
                    'png': files['.png'],
                    'txt': files['.txt'],
                    'tif': files['.tif'],
                }
                # print(sample)
                # Distribute samples based on the subject number in the filename
                if 'Subject1' in base_name:
                    self.val_tuples.append(sample)
                    self.test_tuples.append(sample)  # Use same data for test for now
                elif 'Subject2' in base_name or 'Subject4' in base_name:
                    self.train_tuples.append(sample)
        
        print(f"Found {len(file_groups)} unique file base names.")
        print(f"Created {len(self.train_tuples)} training, {len(self.val_tuples)} val, and {len(self.test_tuples)} test samples.")

    def setup(self, stage: Optional[str] = None):
        """Instantiates the training, val, and test datasets."""
        self._load_samples()

        if stage == 'fit' or stage is None:
            self._train_ds = ACT2Dataset(
                self.train_tuples, 
                self.train_transforms, 
                image_key='tif', 
                hint_key='png', 
                txt_key='txt'
            )
            self._val_ds = ACT2Dataset(
                self.val_tuples, 
                self.val_transforms, 
                image_key='tif', 
                hint_key='png', 
                txt_key='txt'
            )
        
        if stage == 'test' or stage is None:
            self._test_ds = ACT2Dataset(
                self.test_tuples, 
                self.val_transforms,
                image_key='tif', 
                hint_key='png', 
                txt_key='txt'
            )

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Creates a DataLoader for a given Dataset instance."""
        return DataLoader(
            dataset,
            batch_size=self.micro_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        """Returns the training DataLoader."""
        return self._create_dataloader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Returns the val DataLoader."""
        return self._create_dataloader(self._val_ds)

    def test_dataloader(self) -> DataLoader:
        """Returns the testing DataLoader."""
        return self._create_dataloader(self._test_ds)

def main():
    """Main function to test and verify the ACT2DataModule."""
    
    # --- Configuration ---
    # IMPORTANT: Change this path to the location of your ACT2 dataset files.
    # The directory should contain the .tif, .png, and .txt files.
    data_root = "data/ACT2_raw" 
    
    print("="*60)
    print("ACT2 DataModule Verification")
    print("="*60)
    
    if not os.path.exists(data_root):
        print(f"Error: The data directory '{data_root}' was not found.")
        print("Please create a dummy directory and some files to run this test:")
        print(f"  mkdir -p {data_root}")
        print(f"  touch {data_root}/Subject1_1.tif {data_root}/Subject1_1.png {data_root}/Subject1_1.txt")
        print(f"  touch {data_root}/Subject2_1.tif {data_root}/Subject2_1.png {data_root}/Subject2_1.txt")
        return

    # Instantiate the datamodule
    datamodule = ACT2DataModule(
        root_folder=data_root,
        image_H=512,
        image_W=512,
        micro_batch_size=2,
        global_batch_size=4,
        num_workers=2,
    )
    
    # Setup datasets
    datamodule.setup('fit')
    
    # Check if data was loaded
    if not datamodule.train_tuples or not datamodule.val_tuples:
        print("\nWarning: No training or val samples were loaded.")
        print("Please check the 'root_folder' path and the contents of the directory.")
        return
        
    print("\n--- Verifying Training Dataloader ---")
    try:
        train_loader = datamodule.train_dataloader()
        print(f"Train dataloader created with batch size {datamodule.micro_batch_size}.")
        
        # Get one batch to inspect
        batch = next(iter(train_loader))
        
        print("\nSample batch from training data:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - Key: '{key}', Shape: {value.shape}, DType: {value.dtype}, Min: {value.min():.2f}, Max: {value.max():.2f}")
            elif isinstance(value, list): # For text prompts
                 print(f"  - Key: '{key}', Type: list, Length: {len(value)}, First element: '{value[0][:]}'")
    except Exception as e:
        print(f"An error occurred while testing the train dataloader: {e}")

    print("\n--- Verifying Val Dataloader ---")
    try:
        val_loader = datamodule.val_dataloader()
        print(f"Val dataloader created with batch size {datamodule.micro_batch_size}.")

        batch = next(iter(val_loader))
        
        print("\nSample batch from val data:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - Key: '{key}', Shape: {value.shape}, DType: {value.dtype}, Min: {value.min():.2f}, Max: {value.max():.2f}")
            elif isinstance(value, list):
                 print(f"  - Key: '{key}', Type: list, Length: {len(value)}, First element: '{value[0][:]}'")
    except Exception as e:
        print(f"An error occurred while testing the val dataloader: {e}")
        
    print("\n" + "="*60)
    print("Verification complete.")
    print("If shapes and dtypes look correct, the datamodule is likely working.")
    print("="*60)


if __name__ == "__main__":
    main()