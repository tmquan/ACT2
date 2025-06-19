# verify_webdataset.py
import os
import webdataset as wds
from PIL import Image
import io

# Add this line to disable the decompression bomb check for large images
Image.MAX_IMAGE_PIXELS = None

def verify_act2_webdataset(webdataset_dir: str, split: str = 'train', num_samples_to_check: int = 5):
    """
    Reads a few samples from a WebDataset to verify its integrity.
    """
    split_dir = os.path.join(webdataset_dir, split)
    if not os.path.isdir(split_dir):
        print(f"Error: WebDataset split directory not found at {split_dir}")
        return

    # Construct the path to the shards using brace notation
    shard_pattern = os.path.join(split_dir, f"{split}-{{000000..999999}}.tar")
    
    try:
        dataset = wds.WebDataset(shard_pattern).decode("pil")
    except Exception as e:
        print(f"Could not open dataset at {shard_pattern}. Error: {e}")
        print("Please ensure the shard filenames match the pattern.")
        return

    print(f"Successfully opened WebDataset for split '{split}'. Verifying samples...")

    for i, sample in enumerate(dataset):
        if i >= num_samples_to_check:
            break
            
        print(f"\n--- Sample {i+1} ---")
        print(f"Key: {sample['__key__']}")
        
        # Check for all required components
        if "tif" in sample and "png" in sample and "txt" in sample:
            print("Status: OK - All components (tif, png, txt) are present.")
            
            # Check if images can be loaded
            try:
                tif_image = sample["tif"]
                png_image = sample["png"]
                print(f"TIF image loaded: mode={tif_image.mode}, size={tif_image.size}")
                print(f"PNG image loaded: mode={png_image.mode}, size={png_image.size}")
            except Exception as e:
                print(f"Status: ERROR - Could not decode images. Error: {e}")
                
            # Check text content
            try:
                text_content = sample["txt"].strip()
                print(f"Text content: '{text_content}'")
            except Exception as e:
                print(f"Status: ERROR - Could not decode text. Error: {e}")

        else:
            print(f"Status: ERROR - Missing components. Found keys: {list(sample.keys())}")

if __name__ == '__main__':
    # Configuration
    WDS_DIR = "data/ACT2_wds"
    
    verify_act2_webdataset(WDS_DIR, split='train', num_samples_to_check=3)
    verify_act2_webdataset(WDS_DIR, split='val', num_samples_to_check=3)