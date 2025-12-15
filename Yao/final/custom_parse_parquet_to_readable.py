"""
CUSTOM FILE: Parse Parquet Dataset to Human-Readable Format

將 PrismLayersPro Parquet 檔案轉換為人類可讀的格式：
- 提取所有圖片並保存為 PNG
- 提取文字資訊並保存為 JSON
- 組織成清晰的目錄結構
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
import numpy as np

try:
    from datasets import load_dataset, concatenate_datasets
except ImportError:
    print("錯誤：請安裝 datasets 套件: pip install datasets")
    exit(1)


def save_image(image_data, output_path: Path):
    """
    保存圖片
    
    Args:
    ----
        image_data: PIL Image 或 numpy array
        output_path: 輸出路徑
    """
    if isinstance(image_data, Image.Image):
        image_data.save(output_path)
    elif isinstance(image_data, np.ndarray):
        Image.fromarray(image_data).save(output_path)
    else:
        # 嘗試轉換
        try:
            img = Image.fromarray(np.array(image_data))
            img.save(output_path)
        except Exception as e:
            print(f"  警告：無法保存圖片到 {output_path}: {e}")


def parse_single_parquet_file(
    parquet_path: str,
    output_base_dir: Path,
    max_samples: int = None,
    category_name: str = None,
):
    """
    解析單個 Parquet 檔案
    
    Args:
    ----
        parquet_path: Parquet 檔案路徑
        output_base_dir: 輸出基礎目錄
        max_samples: 最大解析樣本數（None 表示全部）
        category_name: 類別名稱（用於組織輸出）
    """
    print(f"\n{'='*60}")
    print(f"解析檔案: {parquet_path}")
    print(f"{'='*60}")
    
    # 載入資料集
    print("載入 Parquet 檔案...")
    dataset = load_dataset('parquet', data_files=parquet_path)
    train_dataset = dataset['train']
    
    total_samples = len(train_dataset)
    print(f"總共 {total_samples} 筆資料")
    
    if max_samples:
        total_samples = min(total_samples, max_samples)
        print(f"將解析前 {total_samples} 筆資料")
    
    # 取得類別名稱（從檔案名或參數）
    if category_name is None:
        category_name = Path(parquet_path).stem.split('-')[0]
    
    # 創建輸出目錄
    category_dir = output_base_dir / category_name
    category_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析每一筆資料
    for idx in range(total_samples):
        try:
            item = train_dataset[idx]
            sample_id = item.get('id', f'sample_{idx:06d}')
            
            print(f"\n處理樣本 {idx+1}/{total_samples}: {sample_id}")
            
            # 創建樣本目錄
            sample_dir = category_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            # 提取文字資訊
            text_info = {}
            
            # 基本資訊
            text_info['id'] = item.get('id', '')
            text_info['style_category'] = item.get('style_category', '')
            text_info['whole_caption'] = item.get('whole_caption', '')
            text_info['base_caption'] = item.get('base_caption', '')
            text_info['layer_count'] = item.get('layer_count', 0)
            
            # 保存完整圖片
            if 'whole_image' in item and item['whole_image'] is not None:
                whole_img_path = sample_dir / "00_whole_image.png"
                save_image(item['whole_image'], whole_img_path)
                print(f"  ✓ 保存完整圖片: {whole_img_path.name}")
            
            # 保存背景圖片
            if 'base_image' in item and item['base_image'] is not None:
                base_img_path = sample_dir / "01_base_image.png"
                save_image(item['base_image'], base_img_path)
                print(f"  ✓ 保存背景圖片: {base_img_path.name}")
            
            # 提取各層資訊
            layer_count = item.get('layer_count', 0)
            layers_info = []
            
            for layer_idx in range(layer_count):
                layer_key = f"layer_{layer_idx:02d}"
                
                # 檢查是否有這個圖層
                if layer_key not in item or item[layer_key] is None:
                    continue
                
                layer_data = {
                    'layer_index': layer_idx,
                    'caption': item.get(f'{layer_key}_caption', ''),
                    'box': item.get(f'{layer_key}_box', []),
                    'width_dst': item.get(f'{layer_key}_width_dst', 0),
                    'height_dst': item.get(f'{layer_key}_height_dst', 0),
                }
                
                # 保存圖層圖片
                layer_img = item[layer_key]
                if layer_img is not None:
                    layer_img_path = sample_dir / f"{layer_idx+2:02d}_layer_{layer_idx:02d}.png"
                    save_image(layer_img, layer_img_path)
                    layer_data['image_path'] = layer_img_path.name
                    print(f"  ✓ 保存圖層 {layer_idx}: {layer_img_path.name}")
                
                layers_info.append(layer_data)
            
            text_info['layers'] = layers_info
            
            # 保存文字資訊為 JSON
            json_path = sample_dir / "metadata.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(text_info, f, indent=2, ensure_ascii=False)
            print(f"  ✓ 保存元資料: {json_path.name}")
            
        except Exception as e:
            print(f"  ✗ 處理樣本 {idx} 時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n完成！類別 '{category_name}' 的資料已保存到: {category_dir}")


def parse_all_parquet_files(
    data_dir: str,
    output_dir: str,
    max_samples_per_file: int = None,
    max_files: int = None,
):
    """
    解析所有 Parquet 檔案
    
    Args:
    ----
        data_dir: Parquet 檔案所在目錄
        output_dir: 輸出目錄
        max_samples_per_file: 每個檔案最大解析樣本數
        max_files: 最大解析檔案數
    """
    # 尋找所有 Parquet 檔案
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    parquet_files.sort()
    
    if not parquet_files:
        print(f"錯誤：在 {data_dir} 找不到 Parquet 檔案")
        return
    
    print(f"找到 {len(parquet_files)} 個 Parquet 檔案")
    
    if max_files:
        parquet_files = parquet_files[:max_files]
        print(f"將解析前 {max_files} 個檔案")
    
    # 創建輸出目錄
    output_base_dir = Path(output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析每個檔案
    for parquet_file in parquet_files:
        category_name = Path(parquet_file).stem.split('-')[0]
        parse_single_parquet_file(
            parquet_path=parquet_file,
            output_base_dir=output_base_dir,
            max_samples=max_samples_per_file,
            category_name=category_name,
        )
    
    print(f"\n{'='*60}")
    print(f"所有檔案解析完成！")
    print(f"輸出目錄: {output_base_dir}")
    print(f"{'='*60}")


def create_summary(output_dir: str):
    """
    創建資料集摘要
    
    Args:
    ----
        output_dir: 輸出目錄
    """
    output_base_dir = Path(output_dir)
    
    summary = {
        'categories': {},
        'total_samples': 0,
        'total_categories': 0,
    }
    
    # 統計每個類別
    for category_dir in output_base_dir.iterdir():
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name
        sample_dirs = [d for d in category_dir.iterdir() if d.is_dir()]
        
        summary['categories'][category_name] = {
            'sample_count': len(sample_dirs),
            'samples': [d.name for d in sample_dirs],
        }
        summary['total_samples'] += len(sample_dirs)
        summary['total_categories'] += 1
    
    # 保存摘要
    summary_path = output_base_dir / "dataset_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n資料集摘要:")
    print(f"  總類別數: {summary['total_categories']}")
    print(f"  總樣本數: {summary['total_samples']}")
    print(f"  摘要已保存到: {summary_path}")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="解析 Parquet 資料集為可讀格式")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="my_download/data",
        help="Parquet 檔案所在目錄"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="parsed_dataset",
        help="輸出目錄"
    )
    parser.add_argument(
        "--max-samples-per-file",
        type=int,
        default=None,
        help="每個檔案最大解析樣本數（None 表示全部）"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="最大解析檔案數（None 表示全部）"
    )
    parser.add_argument(
        "--create-summary",
        action="store_true",
        help="創建資料集摘要"
    )
    
    args = parser.parse_args()
    
    # 轉換為絕對路徑
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    print(f"資料目錄: {data_dir}")
    print(f"輸出目錄: {output_dir}")
    
    # 解析所有檔案
    parse_all_parquet_files(
        data_dir=data_dir,
        output_dir=output_dir,
        max_samples_per_file=args.max_samples_per_file,
        max_files=args.max_files,
    )
    
    # 創建摘要
    if args.create_summary:
        create_summary(output_dir)





