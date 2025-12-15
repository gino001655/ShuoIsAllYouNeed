
import glob
from datasets import load_dataset
import pandas as pd
from PIL import Image
import io

parquet_files = glob.glob("/tmp2/b12902041/Gino/DLCV_project/ShuoIsAllYouNeed/Yao/final/dataset/**/*.parquet", recursive=True)
ds = load_dataset("parquet", data_files=parquet_files[:1], split="train")
sample = ds[0]

# Check sizes
if isinstance(sample['preview'], dict):
    img = Image.open(io.BytesIO(sample['preview']['bytes']))
elif isinstance(sample['preview'], bytes):
    img = Image.open(io.BytesIO(sample['preview']))
else:
    img = sample['preview']

print(f"Image Size: {img.size}")
print(f"Canvas Width: {sample.get('canvas_width')}")
print(f"Canvas Height: {sample.get('canvas_height')}")
print(f"First Layer Left: {sample.get('left')[0]}")
print(f"First Layer Width: {sample.get('width')[0]}")
