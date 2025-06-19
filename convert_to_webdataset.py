# convert_to_webdataset.py
import os
import json
import webdataset as wds
import shutil
from rich.progress import Progress

def convert_to_webdataset(manifest_path: str, output_dir: str, samples_per_shard: int = 1000):
    """
    Converts the dataset to WebDataset format based on a manifest file.

    Args:
        manifest_path (str): Path to the manifest.json file.
        output_dir (str): Directory to save the WebDataset shards.
        samples_per_shard (int): Number of samples to store in each.tar shard.
    """
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found at {manifest_path}")
        return

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"Loaded manifest with splits: {list(manifest.keys())}")

    for split in manifest.keys():
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            print(f"Output directory for split '{split}' already exists. Removing it.")
            shutil.rmtree(split_dir)
        os.makedirs(split_dir)

        print(f"\nProcessing split: {split}...")
        
        # Pattern for shard filenames, e.g., train-000000.tar
        pattern = os.path.join(split_dir, f"{split}-%06d.tar")
        
        with wds.ShardWriter(pattern, maxcount=samples_per_shard) as sink:
            with Progress() as progress:
                task = progress.add_task(f"Writing {split} shards", total=len(manifest[split]))
                for sample in manifest[split]:
                    base_name = sample["base_name"]
                    
                    # Read the content of each file in binary mode
                    with open(sample["tif_path"], "rb") as stream:
                        tif_data = stream.read()
                    with open(sample["png_path"], "rb") as stream:
                        png_data = stream.read()
                    with open(sample["txt_path"], "rb") as stream:
                        txt_data = stream.read()
                    
                    # Create the sample dictionary for WebDataset
                    # The keys are the file extensions that will be used inside the tarball
                    wds_sample = {
                        "__key__": base_name,
                        "tif": tif_data,
                        "png": png_data,
                        "txt": txt_data,
                    }
                    
                    # Write the sample to the current shard
                    sink.write(wds_sample)
                    progress.update(task, advance=1)
                
        print(f"Finished processing split: {split}. Shards saved to {split_dir}")

if __name__ == '__main__':
    # Configuration
    INPUT_MANIFEST_PATH = "data/act2_manifest.json"
    OUTPUT_WDS_DIR = "data/ACT2_wds"
    SAMPLES_PER_SHARD = 1 # Adjust based on average file size and preference
    
    convert_to_webdataset(INPUT_MANIFEST_PATH, OUTPUT_WDS_DIR, SAMPLES_PER_SHARD)