#!/usr/bin/env python3
"""
驗證轉換後的 dataset 是否可以被 train/inference 正確讀取
"""

import sys
from pathlib import Path
from datasets import load_dataset

# Add CLD tools to path
sys.path.insert(0, str(Path(__file__).parent))

from tools.dlcv_dataset import DLCVLayoutDataset, collate_fn
from torch.utils.data import DataLoader

def verify_converted_dataset(converted_data_dir: str, num_samples: int = 3):
    """
    驗證轉換後的 dataset
    
    Args:
        converted_data_dir: 轉換後的 dataset 根目錄
        num_samples: 要測試的樣本數
    """
    print("="*60)
    print("驗證轉換後的 Dataset")
    print("="*60)
    
    # === 1. 檢查 parquet 文件格式 ===
    print("\n[步驟 1] 檢查 Parquet 文件格式...")
    
    data_dir = Path(converted_data_dir)
    parquet_dir = data_dir / "snapshots" / "snapshot_1" / "data"
    
    if not parquet_dir.exists():
        parquet_dir = data_dir / "data"
    
    parquet_files = list(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"❌ 找不到 parquet 文件在 {parquet_dir}")
        return False
    
    print(f"✓ 找到 {len(parquet_files)} 個 parquet 文件")
    
    # 載入第一個文件檢查
    first_parquet = parquet_files[0]
    print(f"  檢查: {first_parquet.name}")
    
    ds = load_dataset('parquet', data_files=str(first_parquet))['train']
    print(f"  ✓ 載入成功: {len(ds)} 個樣本")
    
    # === 2. 檢查必要欄位 ===
    print("\n[步驟 2] 檢查必要欄位...")
    
    sample = ds[0]
    required_fields = {
        'preview': str,
        'title': str,
        'left': list,
        'top': list,
        'width': list,
        'height': list,
        'length': int,
        'image': list,
        'canvas_width': (int, type(None)),
        'canvas_height': (int, type(None)),
        'type': (list, type(None)),
    }
    
    all_good = True
    for field, expected_type in required_fields.items():
        if field not in sample:
            print(f"  ❌ 缺少欄位: {field}")
            all_good = False
        else:
            value = sample[field]
            if isinstance(expected_type, tuple):
                if not isinstance(value, expected_type):
                    print(f"  ⚠️  {field}: {type(value)} (期望 {expected_type})")
            else:
                if not isinstance(value, expected_type):
                    print(f"  ❌ {field}: {type(value)} (期望 {expected_type})")
                    all_good = False
                else:
                    print(f"  ✓ {field}: {type(value).__name__}")
    
    if not all_good:
        print("\n❌ 欄位檢查失敗！")
        return False
    
    # === 3. 檢查資料內容 ===
    print("\n[步驟 3] 檢查資料內容...")
    
    print(f"  preview: {sample['preview']}")
    print(f"  title: {sample['title'][:80]}...")
    print(f"  length: {sample['length']}")
    print(f"  left: {sample['left'][:3]}... (length={len(sample['left'])})")
    print(f"  image: {len(sample['image'])} layers")
    
    # 檢查圖片路徑是否存在
    import os
    preview_path = sample['preview']
    if isinstance(preview_path, str):
        if os.path.exists(preview_path):
            print(f"  ✓ preview 圖片存在")
        else:
            print(f"  ❌ preview 圖片不存在: {preview_path}")
            all_good = False
    
    # 檢查 layer 圖片
    layer_paths = sample['image']
    existing_layers = 0
    for i, layer_path in enumerate(layer_paths):
        if layer_path and isinstance(layer_path, str) and os.path.exists(layer_path):
            existing_layers += 1
    
    print(f"  ✓ {existing_layers}/{len(layer_paths)} layer 圖片存在")
    
    # === 4. 測試用 DLCVLayoutDataset 載入 ===
    print("\n[步驟 4] 測試用 DLCVLayoutDataset 載入...")
    
    try:
        dataset = DLCVLayoutDataset(
            data_dir=str(converted_data_dir),
            split="train",
            caption_mapping_path=None,  # captions 已經在 parquet 中
            enable_debug=False
        )
        print(f"  ✓ DLCVLayoutDataset 載入成功: {len(dataset)} 個樣本")
    except Exception as e:
        print(f"  ❌ DLCVLayoutDataset 載入失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # === 5. 測試 DataLoader ===
    print("\n[步驟 5] 測試 DataLoader...")
    
    try:
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        # 測試讀取幾個樣本
        for i, batch in enumerate(loader):
            if i >= num_samples:
                break
            
            print(f"\n  樣本 {i}:")
            print(f"    caption: {batch['caption'][:60]}...")
            print(f"    pixel_RGBA shape: {batch['pixel_RGBA'].shape}")
            print(f"    pixel_RGB shape: {batch['pixel_RGB'].shape}")
            print(f"    layout: {len(batch['layout'])} layers")
            print(f"    height: {batch['height']}, width: {batch['width']}")
        
        print(f"\n  ✓ DataLoader 測試成功！")
        
    except Exception as e:
        print(f"  ❌ DataLoader 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # === 6. 最終結果 ===
    print("\n" + "="*60)
    print("✅ 所有驗證通過！")
    print("="*60)
    print("\n💡 這個 dataset 可以直接用於:")
    print(f"  - Training: data_dir=\"{converted_data_dir}\"")
    print(f"  - Inference: data_dir=\"{converted_data_dir}\"")
    print(f"  - 不需要額外的 caption_mapping（captions 已在 parquet 中）")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="驗證轉換後的 dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="轉換後的 dataset 根目錄"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="要測試的樣本數（預設: 3）"
    )
    
    args = parser.parse_args()
    
    success = verify_converted_dataset(
        converted_data_dir=args.data_dir,
        num_samples=args.num_samples,
    )
    
    sys.exit(0 if success else 1)
