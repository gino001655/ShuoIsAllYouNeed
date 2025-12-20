#!/usr/bin/env python3
"""
將 TAData 轉換為基於路徑的 dataset，並加入 LLaVA captions

輸入：
1. TAData parquet 目錄（有完整 layer 圖片，但是 Image 對象格式）
2. LLaVA caption JSON（可選）

輸出：
1. 新的 parquet 文件（所有圖片都是路徑字符串）
2. 所有圖片保存為文件（preview + layers）
3. 加入 LLaVA captions（如果提供）

用法：
    python convert_tadata_with_captions.py \
        --input_dir /path/to/TAData/DLCV_dataset/data \
        --output_dir /path/to/output_dataset \
        --caption_json /path/to/caption_llava16_final.json
"""

import os
import sys
import json
from pathlib import Path
from datasets import load_dataset, Dataset
from PIL import Image
import pandas as pd
from tqdm import tqdm
import argparse


def convert_tadata_with_captions(
    input_parquet_dir: str,
    output_dir: str,
    caption_json: str = None,
    create_snapshot_structure: bool = True,
):
    """
    轉換 TAData 為基於路徑的版本，並加入 captions
    
    Args:
        input_parquet_dir: TAData parquet 文件目錄
        output_dir: 輸出根目錄
        caption_json: Caption JSON 文件（可選）
        create_snapshot_structure: 是否創建 snapshots/snapshot_1 結構
    """
    input_dir = Path(input_parquet_dir)
    output_root = Path(output_dir)
    
    # 建立輸出目錄結構
    if create_snapshot_structure:
        output_parquet_dir = output_root / "snapshots" / "snapshot_1" / "data"
        output_images_dir = output_root / "images"
    else:
        output_parquet_dir = output_root / "data"
        output_images_dir = output_root / "images"
    
    output_parquet_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    # 建立圖片子目錄
    preview_dir = output_images_dir / "previews"
    layers_dir = output_images_dir / "layers"
    preview_dir.mkdir(exist_ok=True)
    layers_dir.mkdir(exist_ok=True)
    
    print(f"[INFO] 輸出結構:")
    print(f"  Parquet: {output_parquet_dir}")
    print(f"  Images: {output_images_dir}")
    
    # 載入 caption mapping（如果提供）
    caption_mapping = {}
    caption_by_index = {}
    
    if caption_json and os.path.exists(caption_json):
        print(f"\n[INFO] 載入 caption mapping: {caption_json}")
        with open(caption_json, 'r', encoding='utf-8') as f:
            raw_captions = json.load(f)
        
        print(f"[INFO] 原始 captions: {len(raw_captions)} 個")
        
        # 分析 caption key 格式
        sample_keys = list(raw_captions.keys())[:5]
        print(f"[INFO] Caption key 範例:")
        for sk in sample_keys:
            print(f"  - {sk}")
        
        # 嘗試建立映射：從原始路徑提取檔名
        for path, caption in raw_captions.items():
            # 提取檔名（去掉路徑和副檔名）
            # /workspace/dataset/preprocessed_data/images/train/00000000.png -> 00000000
            filename = Path(path).stem
            caption_mapping[filename] = caption
        
        print(f"[INFO] Caption mapping 建立完成: {len(caption_mapping)} 個")
    
    # 找到所有 parquet 文件
    parquet_files = sorted(list(input_dir.glob("*.parquet")))
    if not parquet_files:
        print(f"[ERROR] 找不到 parquet 文件在 {input_dir}")
        return
    
    print(f"\n[INFO] 找到 {len(parquet_files)} 個 parquet 文件")
    
    # 全局計數器
    global_idx = 0
    total_with_caption = 0
    total_without_caption = 0
    
    # 處理每個 parquet 文件
    for pf_idx, pf in enumerate(parquet_files):
        print(f"\n{'='*60}")
        print(f"處理 {pf.name} ({pf_idx+1}/{len(parquet_files)})")
        print(f"{'='*60}")
        
        # 載入 dataset（使用 pyarrow 避免 metadata 問題）
        try:
            ds = load_dataset('parquet', data_files=str(pf))['train']
            print(f"  載入 {len(ds)} 個樣本")
        except (TypeError, KeyError) as e:
            # Fallback: 直接用 pyarrow 讀取，避免 metadata 解析問題
            print(f"  ⚠️  load_dataset 失敗，使用 pyarrow 直接讀取...")
            import pyarrow.parquet as pq
            table = pq.read_table(str(pf))
            # 轉換為 dict format
            ds = []
            for i in range(len(table)):
                row = {col: table[col][i].as_py() for col in table.column_names}
                ds.append(row)
            print(f"  ✓ 載入 {len(ds)} 個樣本")
        
        # 轉換每個樣本
        records = []
        
        # 處理兩種格式：Dataset 對象或 list
        ds_len = len(ds)
        
        for i in tqdm(range(ds_len), desc=f"  轉換樣本"):
            # 支援 Dataset 對象和 list
            if isinstance(ds, list):
                item = ds[i]
            else:
                item = ds[i]
            
            # 生成唯一 ID
            sample_id = item.get('id', f"sample_{global_idx:08d}")
            
            # === 1. 保存 preview 圖片 ===
            preview_img = item['preview']
            
            # 處理不同格式的 preview
            if isinstance(preview_img, Image.Image):
                # PIL Image 對象
                preview_filename = f"{global_idx:08d}_preview.png"
                preview_path = preview_dir / preview_filename
                preview_img.save(preview_path)
                preview_path_str = str(preview_path.absolute())
            elif isinstance(preview_img, bytes):
                # 二進制數據（pyarrow 格式）
                from io import BytesIO
                preview_img = Image.open(BytesIO(preview_img))
                preview_filename = f"{global_idx:08d}_preview.png"
                preview_path = preview_dir / preview_filename
                preview_img.save(preview_path)
                preview_path_str = str(preview_path.absolute())
            elif isinstance(preview_img, dict) and 'bytes' in preview_img:
                # HuggingFace Image feature format
                from io import BytesIO
                preview_img = Image.open(BytesIO(preview_img['bytes']))
                preview_filename = f"{global_idx:08d}_preview.png"
                preview_path = preview_dir / preview_filename
                preview_img.save(preview_path)
                preview_path_str = str(preview_path.absolute())
            else:
                # 已經是路徑
                preview_path_str = str(preview_img)
            
            # === 2. 保存每個 layer 圖片 ===
            layer_images = item['image']
            layer_paths = []
            
            if isinstance(layer_images, list):
                for layer_idx, layer_img in enumerate(layer_images):
                    if layer_img is None:
                        layer_paths.append(None)
                    elif isinstance(layer_img, Image.Image):
                        # PIL Image 對象
                        layer_filename = f"{global_idx:08d}_layer_{layer_idx:02d}.png"
                        layer_path = layers_dir / layer_filename
                        layer_img.save(layer_path)
                        layer_paths.append(str(layer_path.absolute()))
                    elif isinstance(layer_img, bytes):
                        # 二進制數據
                        from io import BytesIO
                        img = Image.open(BytesIO(layer_img))
                        layer_filename = f"{global_idx:08d}_layer_{layer_idx:02d}.png"
                        layer_path = layers_dir / layer_filename
                        img.save(layer_path)
                        layer_paths.append(str(layer_path.absolute()))
                    elif isinstance(layer_img, dict) and 'bytes' in layer_img:
                        # HuggingFace Image feature format
                        from io import BytesIO
                        img = Image.open(BytesIO(layer_img['bytes']))
                        layer_filename = f"{global_idx:08d}_layer_{layer_idx:02d}.png"
                        layer_path = layers_dir / layer_filename
                        img.save(layer_path)
                        layer_paths.append(str(layer_path.absolute()))
                    else:
                        # 已經是路徑
                        layer_paths.append(str(layer_img) if layer_img else None)
            else:
                layer_paths = layer_images
            
            # === 3. 查找 caption ===
            # 嘗試多種方式匹配 caption
            caption = None
            
            # 方法 1: 用 global_idx（檔名格式：00000000）
            filename_key = f"{global_idx:08d}"
            if filename_key in caption_mapping:
                caption = caption_mapping[filename_key]
                total_with_caption += 1
            # 方法 2: 用 item ID
            elif 'id' in item and item['id'] in caption_mapping:
                caption = caption_mapping[item['id']]
                total_with_caption += 1
            # 方法 3: 直接用 index
            elif str(global_idx) in caption_mapping:
                caption = caption_mapping[str(global_idx)]
                total_with_caption += 1
            else:
                # Fallback: 用內建 title
                caption = item.get('title', 'A design image')
                total_without_caption += 1
            
            # === 4. 建立新記錄 ===
            # 取得 canvas 尺寸
            canvas_width = item.get('canvas_width', 0)
            canvas_height = item.get('canvas_height', 0)
            
            # 如果沒有 canvas 尺寸，從保存的圖片獲取
            if (canvas_width == 0 or canvas_height == 0) and os.path.exists(preview_path_str):
                try:
                    with Image.open(preview_path_str) as img:
                        canvas_width = img.width
                        canvas_height = img.height
                except:
                    pass
            
            record = {
                'preview': preview_path_str,
                'title': caption,  # 使用找到的 caption
                'left': item['left'],
                'top': item['top'],
                'width': item['width'],
                'height': item['height'],
                'length': item['length'],
                'image': layer_paths,
                # 保留其他欄位（如果需要）
                'canvas_width': canvas_width,
                'canvas_height': canvas_height,
                'type': item.get('type', []),
            }
            
            records.append(record)
            global_idx += 1
        
        # === 5. 保存為新的 parquet ===
        df = pd.DataFrame(records)
        output_parquet = output_parquet_dir / pf.name
        df.to_parquet(output_parquet, index=False, engine='pyarrow')
        print(f"  ✓ 保存到 {output_parquet}")
    
    # 最終統計
    print(f"\n{'='*60}")
    print(f"轉換完成！")
    print(f"{'='*60}")
    print(f"總樣本數: {global_idx}")
    print(f"  有 caption: {total_with_caption} ({total_with_caption/global_idx*100:.1f}%)")
    print(f"  無 caption (用內建title): {total_without_caption} ({total_without_caption/global_idx*100:.1f}%)")
    print(f"\n輸出位置:")
    print(f"  Parquet: {output_parquet_dir}")
    print(f"  Images: {output_images_dir}")
    print(f"\n✅ 新 dataset 可以直接用於 training/inference！")


def main():
    parser = argparse.ArgumentParser(
        description="將 TAData 轉換為基於路徑的 dataset，並加入 LLaVA captions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：

1. 基本轉換（不加 captions）：
    python convert_tadata_with_captions.py \\
        --input_dir /tmp2/b12902041/Gino/TAData/DLCV_dataset/data \\
        --output_dir /tmp2/b12902041/Gino/TAData_converted

2. 加入 LLaVA captions：
    python convert_tadata_with_captions.py \\
        --input_dir /tmp2/b12902041/Gino/TAData/DLCV_dataset/data \\
        --output_dir /tmp2/b12902041/Gino/TAData_with_llava_captions \\
        --caption_json /path/to/caption_llava16_final.json

3. 不使用 snapshot 結構：
    python convert_tadata_with_captions.py \\
        --input_dir /tmp2/b12902041/Gino/TAData/DLCV_dataset/data \\
        --output_dir /tmp2/b12902041/Gino/TAData_converted \\
        --no_snapshot_structure
        """
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="TAData parquet 文件目錄（例如：/path/to/TAData/DLCV_dataset/data）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="輸出根目錄"
    )
    parser.add_argument(
        "--caption_json",
        type=str,
        default=None,
        help="LLaVA caption JSON 文件（可選）"
    )
    parser.add_argument(
        "--no_snapshot_structure",
        action="store_true",
        help="不使用 snapshots/snapshot_1 目錄結構"
    )
    
    args = parser.parse_args()
    
    convert_tadata_with_captions(
        input_parquet_dir=args.input_dir,
        output_dir=args.output_dir,
        caption_json=args.caption_json,
        create_snapshot_structure=not args.no_snapshot_structure,
    )


if __name__ == "__main__":
    main()

