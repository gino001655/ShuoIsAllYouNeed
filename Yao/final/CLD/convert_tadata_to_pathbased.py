#!/usr/bin/env python3
"""
將 TAData 的 parquet 轉換為基於路徑的版本
- 把 preview Image 對象保存為文件
- 把 image list (layer images) 保存為文件
- 生成新的 parquet，preview 和 image 都是路徑字符串

這樣就可以用 caption_mapping.json 了！
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset, Dataset
from PIL import Image
import pandas as pd
from tqdm import tqdm

def convert_tadata_to_path_based(
    input_parquet_dir: str,
    output_dir: str,
    images_output_dir: str,
):
    """
    轉換 TAData parquet 為基於路徑的版本
    
    Args:
        input_parquet_dir: TAData parquet 文件目錄
        output_dir: 輸出 parquet 文件目錄
        images_output_dir: 輸出圖片文件目錄
    """
    input_dir = Path(input_parquet_dir)
    output_dir = Path(output_dir)
    images_dir = Path(images_output_dir)
    
    # 建立輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 建立子目錄
    preview_dir = images_dir / "previews"
    layers_dir = images_dir / "layers"
    preview_dir.mkdir(exist_ok=True)
    layers_dir.mkdir(exist_ok=True)
    
    # 找到所有 parquet 文件
    parquet_files = sorted(list(input_dir.glob("*.parquet")))
    print(f"找到 {len(parquet_files)} 個 parquet 文件")
    
    for pf in parquet_files:
        print(f"\n處理 {pf.name}...")
        
        # 載入 dataset
        ds = load_dataset('parquet', data_files=str(pf))['train']
        print(f"  載入 {len(ds)} 個樣本")
        
        # 轉換每個樣本
        records = []
        
        for i in tqdm(range(len(ds)), desc="  轉換樣本"):
            item = ds[i]
            
            # 生成唯一 ID（使用原始 id 或 index）
            sample_id = item.get('id', f"sample_{i:08d}")
            
            # 1. 保存 preview 圖片
            preview_img = item['preview']
            if isinstance(preview_img, Image.Image):
                preview_filename = f"{sample_id}_preview.png"
                preview_path = preview_dir / preview_filename
                preview_img.save(preview_path)
                preview_path_str = str(preview_path.absolute())
            else:
                # 已經是路徑
                preview_path_str = preview_img
            
            # 2. 保存每個 layer 圖片
            layer_images = item['image']
            layer_paths = []
            
            if isinstance(layer_images, list):
                for layer_idx, layer_img in enumerate(layer_images):
                    if layer_img is None:
                        layer_paths.append(None)
                    elif isinstance(layer_img, Image.Image):
                        layer_filename = f"{sample_id}_layer_{layer_idx:02d}.png"
                        layer_path = layers_dir / layer_filename
                        layer_img.save(layer_path)
                        layer_paths.append(str(layer_path.absolute()))
                    else:
                        # 已經是路徑
                        layer_paths.append(layer_img)
            else:
                layer_paths = layer_images
            
            # 3. 建立新記錄（保留所有原始欄位，只修改 preview 和 image）
            record = dict(item)
            record['preview'] = preview_path_str
            record['image'] = layer_paths
            
            records.append(record)
        
        # 保存為新的 parquet
        df = pd.DataFrame(records)
        output_parquet = output_dir / pf.name
        df.to_parquet(output_parquet, index=False, engine='pyarrow')
        print(f"  ✓ 保存到 {output_parquet}")
    
    print(f"\n✅ 轉換完成！")
    print(f"  新 parquet: {output_dir}")
    print(f"  圖片文件: {images_dir}")
    print(f"\n💡 現在可以用這個 dataset 配合 caption_mapping.json！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="轉換 TAData 為基於路徑的版本")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/tmp2/b12902041/Gino/TAData/DLCV_dataset/data",
        help="TAData parquet 目錄"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp2/b12902041/Gino/TAData_with_paths/data",
        help="輸出 parquet 目錄"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/tmp2/b12902041/Gino/TAData_with_paths/images",
        help="輸出圖片目錄"
    )
    
    args = parser.parse_args()
    
    convert_tadata_to_path_based(
        input_parquet_dir=args.input_dir,
        output_dir=args.output_dir,
        images_output_dir=args.images_dir,
    )
