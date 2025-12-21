#!/usr/bin/env python3
"""
Inspect parquet file structure to see why layer images are None
"""
import glob
import os
from datasets import load_dataset

data_dir = "/workspace/dataset/cld_dataset/snapshots/snapshot_1"
local_parquet_dir = os.path.join(data_dir, "data")
local_parquet_files = sorted(glob.glob(os.path.join(local_parquet_dir, "*.parquet")))

if len(local_parquet_files) == 0:
    print(f"No parquet files found in {local_parquet_dir}")
    exit(1)

print(f"Found {len(local_parquet_files)} parquet files")
print(f"Checking first file: {local_parquet_files[0]}")

# Load first parquet file
ds = load_dataset("parquet", data_files=local_parquet_files[0])["train"]

print(f"\nDataset size: {len(ds)}")
print(f"Dataset columns: {ds.column_names}")

# Check first sample
sample = ds[0]
print("\n" + "="*60)
print("First sample structure:")
print("="*60)

for key, value in sample.items():
    if key == "image":
        print(f"\n'{key}': {type(value)}")
        if isinstance(value, list):
            print(f"  Length: {len(value)}")
            for i, img in enumerate(value[:3]):  # Check first 3
                print(f"  image[{i}]: {type(img)} = {img}")
        else:
            print(f"  Value: {value}")
    elif key == "preview":
        print(f"\n'{key}': {type(value)}")
        print(f"  Value: {value}")
    elif isinstance(value, list):
        print(f"'{key}': {type(value)}, length={len(value)}, first_item={value[0] if len(value) > 0 else 'empty'}")
    else:
        print(f"'{key}': {type(value)}, value={value}")

# Check if it's storing paths vs actual images
print("\n" + "="*60)
print("Checking if 'image' field stores paths or Image objects")
print("="*60)

if 'image' in sample:
    images = sample['image']
    if isinstance(images, list) and len(images) > 0:
        first_img = images[0]
        print(f"First image type: {type(first_img)}")
        print(f"First image value: {first_img}")
        
        # Check if it's a path
        if isinstance(first_img, str):
            print("\n⚠️  PROBLEM: 'image' field stores paths (strings), not Image objects!")
            print("The dataset loader is not automatically loading the images.")
            
            # Try to check if path exists
            if os.path.exists(first_img):
                print(f"✓ Path exists: {first_img}")
            else:
                print(f"✗ Path does NOT exist: {first_img}")


