# generate_manifest.py
import os
import glob
from collections import defaultdict
import json
import re # Import re for regex matching

def create_act2_manifest(root_folder: str, output_path: str):
    """
    Scans the ACT2 root folder, groups tif, png, and txt files by their
    base name, and creates a JSON manifest separating train and val splits.

    Args:
        root_folder (str): The path to the raw ACT2 dataset directory.
        output_path (str): The path to save the output manifest.json file.
    """
    print(f"Scanning for data in: {root_folder}")
    if not os.path.isdir(root_folder):
        print(f"Error: Directory not found at {root_folder}")
        return

    # Use a dictionary to group files by their base name
    file_groups = defaultdict(dict)
    
    # Search for all relevant file types recursively
    search_pattern = os.path.join(root_folder, '**', '*')
    all_files = glob.glob(search_pattern, recursive=True)

    for file_path in all_files:
        if os.path.isfile(file_path):
            # We only care about the basename for grouping, not the full path
            base_name_with_ext = os.path.basename(file_path)
            base_name, ext = os.path.splitext(base_name_with_ext)
            ext = ext.lower()
            if ext in ['.tif', '.png', '.txt']:
                # Use the full path for the value to locate the file later
                file_groups[base_name][ext] = file_path

    print(f"Found {len(file_groups)} unique base names.")

    # Define regex patterns for splits based on subject ID in the filename
    # Train: Subject 2 or Subject 4
    # Val: Subject 1
    train_pattern = re.compile(r'Subject(2|4)')
    val_pattern = re.compile(r'Subject1')
    
    # The problem description also mentions a 'cropped folder 512x512 for test/predict'.
    # This script focuses on train/val for WebDataset conversion. The test set
    # will be handled separately for batch inference.
    
    manifest = {
        "train": [],
        "val": []
    }

    processed_samples = 0
    for base_name, files in file_groups.items():
        # A complete sample must have all three file types
        if '.tif' in files and '.png' in files and '.txt' in files:
            sample_record = {
                "base_name": base_name,
                "tif_path": files['.tif'],
                "png_path": files['.png'],
                "txt_path": files['.txt']
            }

            # Assign to a split using regex search on the base filename
            if train_pattern.search(base_name):
                manifest["train"].append(sample_record)
                processed_samples += 1
            elif val_pattern.search(base_name):
                manifest["val"].append(sample_record)
                processed_samples += 1

    print(f"Successfully processed {processed_samples} complete samples.")
    print(f"Found {len(manifest['train'])} training samples.")
    print(f"Found {len(manifest['val'])} validation samples.")

    # Save the manifest to a JSON file
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    print(f"Manifest saved to {output_path}")

if __name__ == '__main__':
    # Configuration
    RAW_DATA_ROOT = "data/ACT2_raw"  # Assumes slides 1, 2, 4 are here
    OUTPUT_MANIFEST_PATH = "data/act2_manifest.json"
    
    os.makedirs(os.path.dirname(OUTPUT_MANIFEST_PATH), exist_ok=True)
    create_act2_manifest(RAW_DATA_ROOT, OUTPUT_MANIFEST_PATH)