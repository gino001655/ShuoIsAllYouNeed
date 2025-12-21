#!/usr/bin/env python3
"""
檢查 TAData/DLCV_dataset 的 parquet 文件結構
"""
from datasets import load_dataset
from PIL import Image

parquet_file = "/tmp2/b12902041/Gino/TAData/DLCV_dataset/data/train-00000-of-00031.parquet"
print(f"Checking: {parquet_file}\n")

ds = load_dataset("parquet", data_files=parquet_file)["train"]

print("=" * 80)
print("DATASET INFO")
print("=" * 80)
print(f"Total samples: {len(ds)}")
print(f"Columns: {ds.column_names}\n")

# Check first sample
sample = ds[0]
print("=" * 80)
print("FIRST SAMPLE")
print("=" * 80)

for key, value in sample.items():
    print(f"\n'{key}':")
    print(f"  Type: {type(value)}")
    
    if isinstance(value, list):
        print(f"  Length: {len(value)}")
        if len(value) > 0:
            print(f"  First item type: {type(value[0])}")
            if isinstance(value[0], str):
                print(f"  First item: {value[0][:100] if len(value[0]) > 100 else value[0]}")
            elif isinstance(value[0], Image.Image):
                print(f"  First item: Image({value[0].size})")
            else:
                print(f"  First item: {value[0]}")
    elif isinstance(value, Image.Image):
        print(f"  Image size: {value.size}")
        print(f"  Image mode: {value.mode}")
    elif isinstance(value, str) and len(value) > 100:
        print(f"  String (truncated): {value[:100]}...")
    else:
        print(f"  Value: {value}")

print("\n" + "=" * 80)
print("CHECKING 'image' FIELD")
print("=" * 80)

if 'image' in sample:
    images = sample['image']
    print(f"Type: {type(images)}")
    
    if isinstance(images, list):
        print(f"Length: {len(images)}")
        print("\nFirst 3 items:")
        for i in range(min(3, len(images))):
            img = images[i]
            print(f"  image[{i}]: {type(img)}")
            if isinstance(img, Image.Image):
                print(f"    -> Image size: {img.size}, mode: {img.mode}")
            elif isinstance(img, str):
                print(f"    -> Path: {img}")
            elif img is None:
                print(f"    -> None ⚠️")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if 'image' in sample:
    images = sample['image']
    if isinstance(images, list) and len(images) > 0:
        first_img = images[0]
        if isinstance(first_img, Image.Image):
            print("✅ TAData HAS layer images stored as Image objects!")
            print("💡 This dataset can be used directly for training!")
        elif isinstance(first_img, str):
            print("✅ TAData HAS layer images stored as paths!")
            print("💡 Need to load images from paths")
        elif first_img is None:
            print("❌ TAData DOES NOT have layer images (all None)")
            print("💡 Need to crop from preview like cld_dataset")


