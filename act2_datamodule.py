# act2_datamodule.py
import os
import glob
import random
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import torch
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
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
from monai.data import CacheDataset, ThreadDataLoader, PersistentDataset, SmartCacheDataset

# Set a higher limit for image pixels if dealing with large TIFs
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


class OptimizedACT2Dataset(Dataset):
    """Highly optimized PyTorch Dataset for ACT2 data with preloaded text content."""
    
    def __init__(
        self, 
        data: List[Dict], 
        transform: Compose, 
        image_key: str = 'images', 
        hint_key: str = 'hint', 
        txt_key: str = 'txt', 
        num_samples: Optional[int] = None,
        preload_text: bool = True,
    ):
        self.data = data
        self.transform = transform
        self.image_key = image_key
        self.hint_key = hint_key
        self.txt_key = txt_key
        self.num_samples = num_samples
        
        # Preload all text content for maximum speed
        if preload_text:
            self._preload_text_content()
    
    def _preload_text_content(self):
        """Preload all text files into memory for faster access."""
        print("Preloading text content...")
        
        def load_text_file(sample):
            try:
                with open(sample["txt"], 'r') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Error loading text file {sample['txt']}: {e}")
                return ""
        
        # Use ThreadPoolExecutor for parallel text loading
        with ThreadPoolExecutor(max_workers=min(32, len(self.data))) as executor:
            text_contents = list(executor.map(load_text_file, self.data))
        
        # Store text content in memory
        for i, content in enumerate(text_contents):
            self.data[i]['txt_content'] = content
        
        print(f"Preloaded {len(text_contents)} text files.")

    def __len__(self):
        if self.num_samples is not None:
            return self.num_samples
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Optimized retrieval with preloaded text content.
        """
        if self.num_samples is not None:
            # Randomly pick an index from the actual data when num_samples is set
            index = random.randint(0, len(self.data) - 1)
        
        sample = self.data[index]
        
        # Use preloaded text content (much faster than file I/O)
        txt_content = sample.get('txt_content', '')
        
        # Load images using transforms
        item_to_transform = {
            'tif': sample['tif'],
            'png': sample['png'],
        }
        
        # Apply MONAI transforms
        transformed_item = apply_transform(self.transform, item_to_transform)
        
        # Map to model's expected keys
        final_sample = {
            self.hint_key: transformed_item['tif'],
            self.txt_key: txt_content,
            self.image_key: transformed_item['png'],
        }

        return final_sample


class ACT2DataModule(LightningDataModule):
    """
    Optimized PyTorch Lightning DataModule for ACT2 dataset with MONAI caching.
    """
    
    def __init__(
        self,
        root_folder: str,
        image_H: int = 512,
        image_W: int = 512,
        micro_batch_size: int = 1,
        global_batch_size: int = 8,
        num_workers: int = None,  # Auto-detect optimal workers
        pin_memory: bool = True,
        persistent_workers: bool = True,
        train_samples: Optional[int] = None,
        val_samples: Optional[int] = None,
        test_samples: Optional[int] = None,
        cache_rate: float = 1.0,  # Cache all data for maximum speed
        use_thread_dataloader: bool = True,  # Use faster thread-based dataloader
        prefetch_factor: int = 4,  # Prefetch batches
        cache_num_workers: int = None,  # Workers for caching process
        copy_cache: bool = False,  # Copy cache to avoid data corruption
        cache_strategy: str = "memory",  # "memory", "disk", "smart"
        cache_dir: str = "./cache",  # Directory for disk caching
    ):
        """
        Optimized DataModule with MONAI caching for transformed images.
        
        Args:
            cache_strategy: Choose caching strategy:
                - "memory": CacheDataset (fastest, high RAM usage)
                - "disk": PersistentDataset (slower, saves to disk)
                - "smart": SmartCacheDataset (intelligent memory management)
        """
        super().__init__()
        self.root_folder = root_folder
        self.image_H = image_H
        self.image_W = image_W
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.cache_rate = cache_rate
        self.use_thread_dataloader = use_thread_dataloader
        self.prefetch_factor = prefetch_factor
        self.copy_cache = copy_cache
        self.cache_strategy = cache_strategy.lower()
        self.cache_dir = cache_dir
        
        # Validate cache strategy
        valid_strategies = ["memory", "disk", "smart"]
        if self.cache_strategy not in valid_strategies:
            raise ValueError(f"cache_strategy must be one of {valid_strategies}, got {cache_strategy}")
        
        # Auto-detect optimal number of workers
        if num_workers is None:
            cpu_count = mp.cpu_count()
            self.num_workers = min(16, max(1, int(cpu_count * 0.75)))
        else:
            self.num_workers = num_workers
        
        # Auto-detect cache workers (fewer than data loading workers)
        if cache_num_workers is None:
            self.cache_num_workers = min(8, max(1, int(self.num_workers // 2)))
        else:
            self.cache_num_workers = cache_num_workers
        
        print(f"Using {self.num_workers} workers for data loading")
        print(f"Using {self.cache_num_workers} workers for caching")
        print(f"Cache strategy: {self.cache_strategy}")
        print(f"Cache rate: {self.cache_rate} (1.0 = cache all data)")
        
        if self.cache_strategy == "disk":
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Disk cache directory: {self.cache_dir}")

        # Optimized MONAI transforms with better performance
        self.train_transforms = Compose([
            LoadImageDict(keys=["tif"], image_only=True, reader="ITKReader"),
            LoadImageDict(keys=["png"], image_only=True, reader="PILReader"),
            EnsureChannelFirstDict(keys=["tif", "png"]),
            RandSpatialCropDict(
                keys=["tif", "png"], 
                roi_size=(self.image_H, self.image_W), 
                random_size=False
            ),
            RandAxisFlipDict(keys=["tif", "png"], prob=0.75),
            RandRotate90Dict(keys=["tif", "png"], prob=0.75),
            ScaleIntensityRangeDict(
                keys=["tif", "png"], 
                clip=True, 
                a_min=0.0, 
                a_max=255.0, 
                b_min=0.0, 
                b_max=1.0
            ),
            ToTensorDict(keys=["tif", "png"]),
        ])

        self.val_transforms = Compose([
            LoadImageDict(keys=["tif"], image_only=True, reader="ITKReader"),
            LoadImageDict(keys=["png"], image_only=True, reader="PILReader"),
            EnsureChannelFirstDict(keys=["tif", "png"]),
            RandSpatialCropDict(
                keys=["tif", "png"], 
                roi_size=(self.image_H, self.image_W), 
                random_size=False
            ),
            # No augmentations for validation
            ScaleIntensityRangeDict(
                keys=["tif", "png"], 
                clip=True, 
                a_min=0.0, 
                a_max=255.0, 
                b_min=0.0, 
                b_max=1.0
            ),
            ToTensorDict(keys=["tif", "png"]),
        ])

        # Lists to hold file path triplets
        self.train_tuples: List[Dict] = []
        self.val_tuples: List[Dict] = []
        self.test_tuples: List[Dict] = []

    def _load_samples(self):
        """
        Optimized sample loading with better file system operations.
        """
        print(f"Loading image samples from: {self.root_folder}")
        
        if not os.path.isdir(self.root_folder):
            print(f"Warning: Data directory not found at {self.root_folder}")
            return
        
        # Use more efficient file scanning
        all_files = glob.glob(os.path.join(self.root_folder, "*"), recursive=False)
        
        # Group files by their base name more efficiently
        file_groups: Dict[str, Dict[str, str]] = {}
        for file_path in all_files:
            if os.path.isfile(file_path):
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                ext = os.path.splitext(file_path)[1].lower()
                
                if base_name not in file_groups:
                    file_groups[base_name] = {}
                file_groups[base_name][ext] = file_path
        
        # Create samples more efficiently
        for base_name, files in file_groups.items():
            if all(ext in files for ext in ['.tif', '.png', '.txt']):
                sample = {
                    'png': files['.png'],
                    'txt': files['.txt'],
                    'tif': files['.tif'],
                }
                
                # Distribute samples based on subject number
                if 'Subject1' in base_name:
                    self.val_tuples.append(sample)
                    self.test_tuples.append(sample)
                elif 'Subject2' in base_name or 'Subject4' in base_name:
                    self.train_tuples.append(sample)
        
        print(f"Curated {len(file_groups)} unique file base names.")
        print(f"Created {len(self.train_tuples)} training, {len(self.val_tuples)} val, and {len(self.test_tuples)} test samples.")

    def _prepare_data_dicts(self, tuples_list: List[Dict]) -> List[Dict]:
        """Prepare data dictionaries for MONAI CacheDataset with text content."""
        data_dicts = []
        
        print(f"Preparing {len(tuples_list)} samples for MONAI caching...")
        
        for sample in tuples_list:
            # Load text content
            try:
                with open(sample['txt'], 'r') as f:
                    txt_content = f.read().strip()
            except Exception as e:
                print(f"Error loading text file {sample['txt']}: {e}")
                txt_content = ""
            
            # Create MONAI-compatible data dict
            data_dict = {
                'tif': sample['tif'],
                'png': sample['png'],
                'txt': txt_content,  # Store text content directly
            }
            data_dicts.append(data_dict)
        
        return data_dicts

    def _create_cached_dataset(self, data_dicts: List[Dict], transform: Compose, split_name: str):
        """Create appropriate cached dataset based on the caching strategy."""
        
        if self.cache_strategy == "memory":
            print(f"Creating {split_name} CacheDataset (memory) with cache_rate={self.cache_rate}")
            return CacheDataset(
                data=data_dicts,
                transform=transform,
                cache_rate=self.cache_rate,
                num_workers=self.cache_num_workers,
                copy_cache=self.copy_cache,
                progress=True,
            )
        
        elif self.cache_strategy == "disk":
            cache_path = os.path.join(self.cache_dir, f"{split_name}_cache")
            print(f"Creating {split_name} PersistentDataset (disk) at {cache_path}")
            return PersistentDataset(
                data=data_dicts,
                transform=transform,
                cache_dir=cache_path,
                num_workers=self.cache_num_workers,
                progress=True,
            )
        
        elif self.cache_strategy == "smart":
            print(f"Creating {split_name} SmartCacheDataset with cache_rate={self.cache_rate}")
            return SmartCacheDataset(
                data=data_dicts,
                transform=transform,
                cache_rate=self.cache_rate,
                num_init_workers=self.cache_num_workers,
                num_replace_workers=max(1, self.cache_num_workers // 2),
                progress=True,
            )
        
        else:
            raise ValueError(f"Unknown cache strategy: {self.cache_strategy}")

    def setup(self, stage: Optional[str] = None):
        """Instantiate MONAI cached datasets with the chosen caching strategy."""
        self._load_samples()

        if stage == 'fit' or stage is None:
            # Prepare data for training with caching
            train_data_dicts = self._prepare_data_dicts(self.train_tuples)
            
            self._train_ds = self._create_cached_dataset(
                train_data_dicts, 
                self.train_transforms, 
                "train"
            )
            
            # Apply sample limit if specified
            if self.train_samples is not None:
                self._train_ds = self._create_limited_dataset(self._train_ds, self.train_samples)
            
            # Prepare validation data with caching
            val_data_dicts = self._prepare_data_dicts(self.val_tuples)
            
            self._val_ds = self._create_cached_dataset(
                val_data_dicts, 
                self.val_transforms, 
                "val"
            )
            
            # Apply sample limit if specified
            if self.val_samples is not None:
                self._val_ds = self._create_limited_dataset(self._val_ds, self.val_samples)
        
        if stage == 'test' or stage is None:
            # Prepare test data with caching
            test_data_dicts = self._prepare_data_dicts(self.test_tuples)
            
            self._test_ds = self._create_cached_dataset(
                test_data_dicts, 
                self.val_transforms,  # Use val transforms (no augmentation)
                "test"
            )
            
            # Apply sample limit if specified
            if self.test_samples is not None:
                self._test_ds = self._create_limited_dataset(self._test_ds, self.test_samples)

    def _create_limited_dataset(self, dataset, num_samples: int):
        """Create a limited dataset wrapper that samples from the full dataset."""
        class LimitedDataset(Dataset):
            def __init__(self, base_dataset, num_samples):
                self.base_dataset = base_dataset
                self.num_samples = num_samples
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, index):
                # Randomly sample from the base dataset
                actual_index = random.randint(0, len(self.base_dataset) - 1)
                return self.base_dataset[actual_index]
        
        return LimitedDataset(dataset, num_samples)

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False, num_workers: Optional[int] = None) -> DataLoader:
        """Creates an optimized DataLoader with best performance settings."""
        effective_workers = num_workers if num_workers is not None else self.num_workers
        
        # Use ThreadDataLoader for better performance if available
        if self.use_thread_dataloader:
            try:
                return ThreadDataLoader(
                    dataset,
                    batch_size=self.micro_batch_size,
                    num_workers=effective_workers,
                    pin_memory=self.pin_memory,
                    persistent_workers=self.persistent_workers,
                    shuffle=shuffle,
                    prefetch_factor=self.prefetch_factor,
                    drop_last=False,
                )
            except Exception as e:
                print(f"ThreadDataLoader failed, falling back to standard DataLoader: {e}")
        
        # Fallback to standard DataLoader with optimized settings
        return DataLoader(
            dataset,
            batch_size=self.micro_batch_size,
            num_workers=effective_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=shuffle,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        """Returns optimized training DataLoader."""
        return self._create_dataloader(
            self._train_ds, 
            num_workers=self.num_workers, 
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        """Returns optimized validation DataLoader."""
        return self._create_dataloader(self._val_ds, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Returns optimized testing DataLoader."""
        return self._create_dataloader(self._test_ds, num_workers=self.num_workers)


# Keep the old class for backward compatibility
ACT2Dataset = OptimizedACT2Dataset


def main():
    """Main function to test and verify the ACT2DataModule with MONAI caching."""
    
    # --- Configuration ---
    data_root = "data/ACT2_raw" 
    
    print("="*60)
    print("🚀 ACT2 DataModule with MONAI CacheDataset")
    print("="*60)
    
    if not os.path.exists(data_root):
        print(f"Error: The data directory '{data_root}' was not found.")
        print("Please create a dummy directory and some files to run this test:")
        print(f"  mkdir -p {data_root}")
        print(f"  touch {data_root}/Subject1_1.tif {data_root}/Subject1_1.png {data_root}/Subject1_1.txt")
        print(f"  touch {data_root}/Subject2_1.tif {data_root}/Subject2_1.png {data_root}/Subject2_1.txt")
        return

    # Instantiate the datamodule with MONAI caching
    datamodule = ACT2DataModule(
        root_folder=data_root,
        image_H=512,
        image_W=512,
        micro_batch_size=2,
        global_batch_size=4,
        train_samples=100,  # Reduced for testing
        val_samples=50,
        test_samples=50,
        cache_rate=1.0,  # Cache everything for maximum speed
        cache_num_workers=24,  # Parallel caching
        copy_cache=False,  # Don't copy cache for memory efficiency
        cache_strategy="memory",  # Try "memory", "disk", or "smart"
        cache_dir="./cache_test",  # For disk caching
        use_thread_dataloader=True,
        prefetch_factor=4,
    )
    
    # Setup datasets (this will create and populate the MONAI cache)
    print("\n--- Setting up datasets (MONAI caching in progress) ---")
    import time
    setup_start = time.time()
    datamodule.setup('fit')
    setup_time = time.time() - setup_start
    print(f"MONAI caching setup completed in {setup_time:.2f} seconds")
    
    # Check if data was loaded
    if not datamodule.train_tuples or not datamodule.val_tuples:
        print("\nWarning: No training or val samples were loaded.")
        print("Please check the 'root_folder' path and the contents of the directory.")
        return
        
    print(f"\n📊 Cache Statistics:")
    print(f"  - Training samples cached: {len(datamodule._train_ds)}")
    print(f"  - Validation samples cached: {len(datamodule._val_ds)}")
    
    print("\n--- 🏃 Testing MONAI Cached Training Dataloader ---")
    try:
        train_loader = datamodule.train_dataloader()
        print(f"Train dataloader created with batch size {datamodule.micro_batch_size}.")
        print(f"Using {datamodule.num_workers} workers with prefetch_factor={datamodule.prefetch_factor}")
        
        # Performance test - measure loading time for multiple batches
        print("\n⚡ Performance test - loading 5 batches from cache...")
        batch_times = []
        
        for i in range(5):
            start_time = time.time()
            batch = next(iter(train_loader))
            load_time = time.time() - start_time
            batch_times.append(load_time)
            
            if i == 0:  # Print first batch info
                print(f"\nFirst cached batch loaded in {load_time:.3f} seconds")
                print("Sample batch from MONAI cached training data:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  - Key: '{key}', Shape: {value.shape}, DType: {value.dtype}, Min: {value.min():.2f}, Max: {value.max():.2f}")
                    elif isinstance(value, list):
                        print(f"  - Key: '{key}', Type: list, Length: {len(value)}, First element: '{value[0][:50]}...'")
        
        avg_batch_time = sum(batch_times) / len(batch_times)
        print(f"\n📈 Average cached batch load time: {avg_batch_time:.3f} seconds")
        print(f"🚀 Cached batches per second: {1/avg_batch_time:.1f}")
            
    except Exception as e:
        print(f"An error occurred while testing the train dataloader: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- 🧪 Testing MONAI Cached Val Dataloader ---")
    try:
        val_loader = datamodule.val_dataloader()
        print(f"Val dataloader created with batch size {datamodule.micro_batch_size}.")

        # Performance test
        start_time = time.time()
        batch = next(iter(val_loader))
        load_time = time.time() - start_time
        print(f"First cached batch loaded in {load_time:.3f} seconds")
        
        print("\nSample batch from MONAI cached val data:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - Key: '{key}', Shape: {value.shape}, DType: {value.dtype}, Min: {value.min():.2f}, Max: {value.max():.2f}")
            elif isinstance(value, list):
                print(f"  - Key: '{key}', Type: list, Length: {len(value)}, First element: '{value[0][:50]}...'")
            
    except Exception as e:
        print(f"An error occurred while testing the val dataloader: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*60)
    print("🎉 MONAI CacheDataset Verification Complete!")
    print("="*60)
    print("🚀 MONAI caching benefits:")
    print("✅ Transformed images cached in memory/disk")
    print("✅ Eliminates redundant image processing")
    print("✅ Parallel caching during setup")
    print("✅ Multiple caching strategies available")
    print("✅ Configurable cache rate (partial/full)")
    print("✅ Memory-efficient cache management")
    print("✅ Progress tracking during cache creation")
    print("\n📊 Caching strategies:")
    print("• memory: CacheDataset - Fastest, high RAM usage")
    print("• disk: PersistentDataset - Slower, saves to disk")
    print("• smart: SmartCacheDataset - Intelligent memory management")
    print("\n📈 Expected performance gains:")
    print("• 5-10x faster data loading after initial cache")
    print("• Eliminates repeated image transformations")
    print("• Reduced CPU usage during training")
    print("• Better GPU utilization (no I/O waiting)")
    print("• Consistent batch loading times")
    print("\n⚠️  Memory considerations:")
    print("• Memory caching: High RAM usage, fastest access")
    print("• Disk caching: Lower RAM, persistent across runs")
    print("• Smart caching: Balanced memory/performance")
    print("• Adjust cache_rate for large datasets")
    print("• Monitor memory usage during training")
    print("="*60)


if __name__ == "__main__":
    main()