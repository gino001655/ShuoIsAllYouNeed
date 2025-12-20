#!/usr/bin/env python3
"""
Quick script to check how many layers are in the dataset
"""
import sys
sys.path.insert(0, '/workspace/ShuoIsAllYouNeed/Yao/final/CLD')

from tools.dlcv_dataset import DLCVLayoutDataset

# Check validation set
data_dir = "/workspace/dataset/cld_dataset/snapshots/snapshot_1"
caption_mapping = "/workspace/ShuoIsAllYouNeed/Yao/final/CLD/caption_formal.json"

print("=" * 60)
print("Checking DLCV Dataset (used for INFERENCE)")
print("=" * 60)

try:
    dataset = DLCVLayoutDataset(data_dir, split="val", caption_mapping_path=caption_mapping)
    
    print(f"\nTotal samples: {len(dataset)}")
    print("\nChecking first 10 samples:")
    print("-" * 60)
    
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        num_layers = len(sample['layout'])
        # layout includes: whole_img + background + foreground layers
        num_foreground = num_layers - 2
        print(f"Sample {i}: {num_layers} total layers (2 base + {num_foreground} foreground)")
        print(f"  Layout boxes: {sample['layout']}")
        print(f"  Caption: {sample['caption'][:60]}...")
        print()
        
except Exception as e:
    print(f"Error loading dataset: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
print("Now checking with LayoutTrainDataset (used for TRAINING)")
print("=" * 60)

try:
    from tools.dataset import LayoutTrainDataset
    train_dataset = LayoutTrainDataset(data_dir, split="train")
    
    print(f"\nTotal train samples: {len(train_dataset)}")
    print("\nChecking first 10 samples:")
    print("-" * 60)
    
    for i in range(min(10, len(train_dataset))):
        sample = train_dataset[i]
        num_layers = len(sample['layout'])
        num_foreground = num_layers - 2
        print(f"Sample {i}: {num_layers} total layers (2 base + {num_foreground} foreground)")
        
except Exception as e:
    print(f"Error loading train dataset: {e}")
    import traceback
    traceback.print_exc()
