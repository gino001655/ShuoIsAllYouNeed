#!/usr/bin/env python3
"""
將 COCO 格式的資料轉換為 CLD 可用的 Parquet 格式

輸入格式（COCO）:
- annotations/train.json, annotations/val.json
- images/train/, images/val/

輸出格式（CLD Parquet）:
- snapshots/snapshot_1/data/train-*.parquet
- snapshots/snapshot_1/data/val-*.parquet
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import pandas as pd
from PIL import Image
import numpy as np

def validate_bbox(bbox: List[float], img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """
    驗證並修正 bbox，確保在圖片範圍內
    
    Args:
        bbox: COCO 格式 [x, y, w, h]
        img_width: 圖片寬度
        img_height: 圖片高度
    
    Returns:
        (x, y, w, h) 修正後的整數座標
    """
    x, y, w, h = bbox
    
    # 轉換為整數
    x = int(round(x))
    y = int(round(y))
    w = int(round(w))
    h = int(round(h))
    
    # 確保座標在有效範圍內
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    
    # 確保寬度和高度有效
    w = max(1, min(w, img_width - x))
    h = max(1, min(h, img_height - y))
    
    # 確保不超出邊界
    if x + w > img_width:
        w = img_width - x
    if y + h > img_height:
        h = img_height - y
    
    return x, y, w, h

def convert_coco_bbox_to_cld(bbox: List[float], img_width: int, img_height: int) -> List[int]:
    """
    將 COCO 格式的 bbox [x, y, w, h] 轉換為 CLD 格式 [x, y, x+w, y+h]
    
    Args:
        bbox: COCO 格式 [x, y, w, h]
        img_width: 圖片寬度
        img_height: 圖片高度
    
    Returns:
        CLD 格式 [x, y, x+w, y+h]（整數）
    """
    x, y, w, h = validate_bbox(bbox, img_width, img_height)
    
    # 轉換為 CLD 格式：[x, y, x+w, y+h]
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    
    # 確保不超出邊界
    x2 = min(x2, img_width)
    y2 = min(y2, img_height)
    
    return [x1, y1, x2, y2]

def load_coco_data(annotation_path: str) -> Dict[str, Any]:
    """載入 COCO 格式的 JSON 檔案"""
    print(f"[INFO] 載入 COCO 註解檔案: {annotation_path}")
    with open(annotation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[INFO] 載入完成: {len(data.get('images', []))} 張圖片, {len(data.get('annotations', []))} 個註解")
    return data

def process_coco_to_cld(
    coco_data: Dict[str, Any],
    images_dir: str,
    output_dir: str,
    split_name: str,
    max_samples_per_shard: int = 1000
) -> None:
    """
    將 COCO 資料轉換為 CLD Parquet 格式
    
    Args:
        coco_data: COCO 格式的資料字典
        images_dir: 圖片目錄路徑
        output_dir: 輸出目錄（會建立 snapshots/snapshot_1/data/）
        split_name: 'train' 或 'val'
        max_samples_per_shard: 每個 Parquet shard 的最大樣本數
    """
    # 建立輸出目錄
    snapshot_dir = os.path.join(output_dir, "snapshots", "snapshot_1", "data")
    os.makedirs(snapshot_dir, exist_ok=True)
    print(f"[INFO] 輸出目錄: {snapshot_dir}")
    
    # 建立圖片 ID 到圖片資訊的映射
    image_dict = {img['id']: img for img in coco_data['images']}
    
    # 按 image_id 分組 annotations
    image_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id in image_dict:
            image_to_anns[image_id].append(ann)
    
    print(f"[INFO] 找到 {len(image_to_anns)} 張有註解的圖片")
    
    # 處理每張圖片
    records = []
    errors = []
    warnings = []
    
    for img_id, img_info in image_dict.items():
        try:
            # 取得該圖片的 annotations
            anns = image_to_anns.get(img_id, [])
            
            # 檢查是否有 annotations
            if len(anns) == 0:
                warnings.append(f"圖片 {img_id} ({img_info['file_name']}) 沒有註解，跳過")
                continue
            
            # 按 layer_order 排序（從背景到前景）
            anns_sorted = sorted(anns, key=lambda x: x.get('layer_order', 0.0))
            
            # 驗證 layer_order 是否有效
            layer_orders = [ann.get('layer_order', 0.0) for ann in anns_sorted]
            if not all(0.0 <= lo <= 1.0 for lo in layer_orders):
                warnings.append(f"圖片 {img_id} 的 layer_order 不在 [0.0, 1.0] 範圍內")
            
            # 檢查是否有重複的 layer_order
            if len(set(layer_orders)) != len(layer_orders):
                warnings.append(f"圖片 {img_id} 有重複的 layer_order，將使用排序後的順序")
            
            # 取得圖片尺寸
            img_width = img_info['width']
            img_height = img_info['height']
            
            # 驗證圖片檔案是否存在
            img_path = os.path.join(images_dir, split_name, img_info['file_name'])
            if not os.path.exists(img_path):
                errors.append(f"圖片檔案不存在: {img_path}")
                continue
            
            # 驗證圖片尺寸（可選：實際讀取圖片驗證）
            try:
                with Image.open(img_path) as img:
                    actual_width, actual_height = img.size
                    if actual_width != img_width or actual_height != img_height:
                        warnings.append(
                            f"圖片 {img_id} 尺寸不匹配: "
                            f"JSON 說 {img_width}x{img_height}, 實際 {actual_width}x{actual_height}"
                        )
                        # 使用實際尺寸
                        img_width, img_height = actual_width, actual_height
            except Exception as e:
                warnings.append(f"無法讀取圖片 {img_path}: {e}")
            
            # 轉換 bbox 格式並建立列表
            left_list = []
            top_list = []
            width_list = []
            height_list = []
            
            for ann in anns_sorted:
                bbox = ann['bbox']  # COCO 格式: [x, y, w, h]
                
                # 驗證 bbox 格式
                if len(bbox) != 4:
                    errors.append(f"圖片 {img_id} 的 bbox 格式錯誤: {bbox}")
                    continue
                
                # 轉換並驗證 bbox
                x, y, w, h = validate_bbox(bbox, img_width, img_height)
                
                # 檢查 bbox 是否有效
                if w <= 0 or h <= 0:
                    warnings.append(f"圖片 {img_id} 有無效的 bbox: {bbox} -> ({x}, {y}, {w}, {h})")
                    continue
                
                left_list.append(x)
                top_list.append(y)
                width_list.append(w)
                height_list.append(h)
            
            # 檢查是否有有效的圖層
            if len(left_list) == 0:
                warnings.append(f"圖片 {img_id} 沒有有效的圖層，跳過")
                continue
            
            # 建立記錄
            # 注意：preview 使用絕對路徑（因為 custom_dataset.py 的 get_img 會直接 Image.open(x)）
            # 這樣可以確保無論從哪裡執行都能找到圖片
            preview_path = os.path.abspath(img_path)
            
            record = {
                "preview": preview_path,  # 絕對路徑（確保能找到圖片）
                "title": "A design image",  # 預設 caption（如果 COCO 沒有 caption）
                "left": left_list,
                "top": top_list,
                "width": width_list,
                "height": height_list,
                "length": len(left_list),  # 圖層數量
                "image": [None] * len(left_list),  # 沒有個別圖層圖片（使用 None）
            }
            
            records.append(record)
            
        except Exception as e:
            errors.append(f"處理圖片 {img_id} 時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
    
    # 輸出統計資訊
    print(f"\n[INFO] 轉換完成統計:")
    print(f"  - 成功轉換: {len(records)} 筆")
    print(f"  - 錯誤: {len(errors)} 筆")
    print(f"  - 警告: {len(warnings)} 筆")
    
    if warnings:
        print(f"\n[WARNING] 前 10 個警告:")
        for w in warnings[:10]:
            print(f"  - {w}")
        if len(warnings) > 10:
            print(f"  ... 還有 {len(warnings) - 10} 個警告")
    
    if errors:
        print(f"\n[ERROR] 錯誤列表:")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... 還有 {len(errors) - 20} 個錯誤")
    
    if len(records) == 0:
        print("[ERROR] 沒有成功轉換任何資料！")
        return
    
    # 儲存為 Parquet（可能需要分 shard）
    num_shards = (len(records) + max_samples_per_shard - 1) // max_samples_per_shard
    
    for shard_idx in range(num_shards):
        start_idx = shard_idx * max_samples_per_shard
        end_idx = min(start_idx + max_samples_per_shard, len(records))
        shard_records = records[start_idx:end_idx]
        
        # 建立 DataFrame
        df = pd.DataFrame(shard_records)
        
        # 儲存為 Parquet
        if num_shards == 1:
            parquet_path = os.path.join(snapshot_dir, f"{split_name}-00000-of-00001.parquet")
        else:
            parquet_path = os.path.join(
                snapshot_dir,
                f"{split_name}-{shard_idx:05d}-of-{num_shards:05d}.parquet"
            )
        
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        print(f"[INFO] 儲存 shard {shard_idx + 1}/{num_shards}: {parquet_path} ({len(shard_records)} 筆)")

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="將 COCO 格式轉換為 CLD Parquet 格式")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="輸入目錄（包含 annotations/ 和 images/）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="輸出目錄（會建立 snapshots/snapshot_1/data/）"
    )
    parser.add_argument(
        "--max_samples_per_shard",
        type=int,
        default=1000,
        help="每個 Parquet shard 的最大樣本數（預設: 1000）"
    )
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # 檢查輸入目錄
    if not os.path.exists(input_dir):
        print(f"[ERROR] 輸入目錄不存在: {input_dir}")
        sys.exit(1)
    
    annotations_dir = os.path.join(input_dir, "annotations")
    images_dir = os.path.join(input_dir, "images")
    
    if not os.path.exists(annotations_dir):
        print(f"[ERROR] annotations 目錄不存在: {annotations_dir}")
        sys.exit(1)
    
    if not os.path.exists(images_dir):
        print(f"[ERROR] images 目錄不存在: {images_dir}")
        sys.exit(1)
    
    # 處理 train 和 val
    for split in ['train', 'val']:
        annotation_path = os.path.join(annotations_dir, f"{split}.json")
        
        if not os.path.exists(annotation_path):
            print(f"[WARNING] {split}.json 不存在，跳過")
            continue
        
        print(f"\n{'='*60}")
        print(f"[INFO] 處理 {split} split")
        print(f"{'='*60}")
        
        # 載入 COCO 資料
        coco_data = load_coco_data(annotation_path)
        
        # 轉換為 CLD 格式
        process_coco_to_cld(
            coco_data=coco_data,
            images_dir=images_dir,
            output_dir=output_dir,
            split_name=split,
            max_samples_per_shard=args.max_samples_per_shard
        )
    
    print(f"\n{'='*60}")
    print(f"[INFO] 轉換完成！輸出目錄: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
