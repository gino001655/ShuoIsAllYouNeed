#!/usr/bin/env python3
"""
檢查 parquet 文件的實際結構，找出為什麼 layer images 都是 None
"""
import glob
import os
from datasets import load_dataset
from PIL import Image

data_dir = "/workspace/dataset/cld_dataset/snapshots/snapshot_1"
local_parquet_dir = os.path.join(data_dir, "data")
local_parquet_files = sorted(glob.glob(os.path.join(local_parquet_dir, "*.parquet")))

if len(local_parquet_files) == 0:
    print(f"❌ No parquet files found in {local_parquet_dir}")
    exit(1)

print(f"Found {len(local_parquet_files)} parquet files")
print(f"Checking first file: {local_parquet_files[0]}\n")

# Load first parquet file
ds = load_dataset("parquet", data_files=local_parquet_files[0])["train"]

print("=" * 80)
print("DATASET INFO")
print("=" * 80)
print(f"Total samples: {len(ds)}")
print(f"Columns: {ds.column_names}\n")

# Check first sample
sample = ds[0]
print("=" * 80)
print("FIRST SAMPLE - ALL FIELDS")
print("=" * 80)

for key, value in sample.items():
    print(f"\n'{key}':")
    print(f"  Type: {type(value)}")
    
    if isinstance(value, list):
        print(f"  Length: {len(value)}")
        if len(value) > 0:
            print(f"  First item type: {type(value[0])}")
            print(f"  First item: {value[0]}")
            if len(value) > 1:
                print(f"  Second item: {value[1]}")
            if len(value) > 2:
                print(f"  Third item: {value[2]}")
    elif isinstance(value, Image.Image):
        print(f"  Image size: {value.size}")
    elif isinstance(value, str) and len(value) > 100:
        print(f"  String (truncated): {value[:100]}...")
    else:
        print(f"  Value: {value}")

print("\n" + "=" * 80)
print("CHECKING 'image' FIELD IN DETAIL")
print("=" * 80)

if 'image' in sample:
    images = sample['image']
    print(f"Type of 'image' field: {type(images)}")
    
    if isinstance(images, list):
        print(f"Number of items in 'image': {len(images)}")
        print(f"\nChecking each item:")
        for i, img in enumerate(images):
            print(f"\n  image[{i}]:")
            print(f"    Type: {type(img)}")
            print(f"    Value: {img}")
            
            if isinstance(img, str):
                print(f"    -> This is a PATH (string)")
                if os.path.exists(img):
                    print(f"    -> Path EXISTS ✓")
                else:
                    print(f"    -> Path DOES NOT EXIST ✗")
            elif isinstance(img, Image.Image):
                print(f"    -> This is an Image object, size: {img.size}")
            elif img is None:
                print(f"    -> This is None ⚠️")
else:
    print("❌ 'image' field not found in sample!")

# Check if there are other fields that might contain layer images
print("\n" + "=" * 80)
print("LOOKING FOR OTHER POSSIBLE IMAGE FIELDS")
print("=" * 80)

possible_image_fields = [k for k in sample.keys() if 'layer' in k.lower() or 'img' in k.lower()]
print(f"Fields with 'layer' or 'img' in name: {possible_image_fields}")

for field in possible_image_fields:
    print(f"\n'{field}':")
    value = sample[field]
    print(f"  Type: {type(value)}")
    if isinstance(value, list):
        print(f"  Length: {len(value)}")
        if len(value) > 0:
            print(f"  First item: {value[0]}")

print("\n" + "=" * 80)
print("CHECKING RAW ARROW TABLE")
print("=" * 80)

# Check raw Arrow table
if hasattr(ds, '_data'):
    arrow_table = ds._data
    print(f"Arrow schema: {arrow_table.schema}")
    
    # Check 'image' column in Arrow table
    if 'image' in arrow_table.column_names:
        image_col = arrow_table.column('image')
        print(f"\nArrow 'image' column:")
        print(f"  Type: {image_col.type}")
        print(f"  First value: {image_col[0].as_py()}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if 'image' in sample:
    images = sample['image']
    if isinstance(images, list) and len(images) > 0:
        if all(img is None for img in images):
            print("⚠️  ALL layer images are None!")
            print("❌ Problem: The parquet file does not contain layer image data.")
            print("\n💡 Possible solutions:")
            print("   1. Regenerate the parquet files with layer images included")
            print("   2. Check if layer images are stored in separate files")
            print("   3. Verify the dataset generation script")
        elif isinstance(images[0], str):
            print("✓ Layer images are stored as paths (strings)")
            print("💡 The path fixing logic should handle this")
        elif isinstance(images[0], Image.Image):
            print("✓ Layer images are stored as Image objects")
            print("💡 Should work directly")


