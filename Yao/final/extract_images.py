#!/usr/bin/env python3
"""
從 Hugging Face 資料集中提取圖片並保存為 PNG 檔案
"""
import os
import sys
from pathlib import Path

try:
    from datasets import load_from_disk
    from PIL import Image
except ImportError as e:
    print(f"錯誤：缺少必要的套件。請執行: pip install datasets pillow")
    print(f"詳細錯誤: {e}")
    sys.exit(1)

def extract_images(dataset_path, output_dir, num_samples=5):
    """
    從資料集中提取圖片並保存為 PNG
    
    Args:
        dataset_path: 資料集路徑（Parquet 檔案所在目錄）
        output_dir: 輸出目錄
        num_samples: 要提取的樣本數量
    """
    print(f"正在載入資料集: {dataset_path}")
    
    try:
        # 嘗試從 Parquet 檔案載入
        import glob
        
        # 尋找所有 Parquet 檔案
        parquet_files = glob.glob(os.path.join(dataset_path, "snapshots/*/data/train-*.parquet"))
        if not parquet_files:
            # 如果找不到，嘗試直接從 dataset_path 載入
            parquet_files = glob.glob(os.path.join(dataset_path, "**/train-*.parquet"), recursive=True)
        
        if not parquet_files:
            # 嘗試使用 load_from_disk
            try:
                dataset = load_from_disk(dataset_path)
                if 'train' in dataset:
                    train_dataset = dataset['train']
                else:
                    train_dataset = dataset
            except:
                raise ValueError(f"無法在 {dataset_path} 找到 Parquet 檔案")
        else:
            print(f"找到 {len(parquet_files)} 個 Parquet 檔案")
            # 載入所有 Parquet 檔案
            from datasets import load_dataset, concatenate_datasets
            datasets_list = []
            for pf in sorted(parquet_files)[:1]:  # 先只載入第一個檔案以加快速度
                print(f"  載入: {pf}")
                ds = load_dataset('parquet', data_files=pf)
                datasets_list.append(ds['train'])
            
            if len(datasets_list) > 1:
                train_dataset = concatenate_datasets(datasets_list)
            else:
                train_dataset = datasets_list[0]
        
        print(f"資料集大小: {len(train_dataset)} 筆")
        print(f"將提取前 {num_samples} 筆資料的圖片")
        
        # 建立輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取圖片
        for i in range(min(num_samples, len(train_dataset))):
            print(f"\n處理第 {i+1} 筆資料...")
            item = train_dataset[i]
            
            # 建立該筆資料的目錄
            item_dir = os.path.join(output_dir, f"sample_{i:04d}")
            os.makedirs(item_dir, exist_ok=True)
            
            # 保存預覽圖 (preview)
            if 'preview' in item and item['preview'] is not None:
                preview = item['preview']
                if isinstance(preview, Image.Image):
                    preview_path = os.path.join(item_dir, "preview.png")
                    preview.save(preview_path)
                    print(f"  已保存預覽圖: {preview_path}")
                else:
                    print(f"  預覽圖格式不正確: {type(preview)}")
            
            # 保存圖層圖片 (image list)
            if 'image' in item and item['image'] is not None:
                images = item['image']
                if isinstance(images, list) and len(images) > 0:
                    for j, img in enumerate(images):
                        if isinstance(img, Image.Image):
                            layer_path = os.path.join(item_dir, f"layer_{j:03d}.png")
                            img.save(layer_path)
                            print(f"  已保存圖層 {j}: {layer_path}")
                        else:
                            print(f"  圖層 {j} 格式不正確: {type(img)}")
                    print(f"  總共保存了 {len(images)} 個圖層")
                else:
                    print(f"  沒有圖層圖片")
            
            # 顯示其他資訊
            if 'id' in item:
                print(f"  ID: {item['id']}")
            if 'title' in item:
                print(f"  標題: {item['title']}")
            if 'length' in item:
                print(f"  圖層數量: {item['length']}")
        
        print(f"\n完成！圖片已保存到: {output_dir}")
        
    except Exception as e:
        print(f"錯誤：無法載入資料集")
        print(f"詳細錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # 設定路徑
    dataset_path = "dataset"
    output_dir = "extracted_images"
    num_samples = 5  # 提取 5 筆資料
    
    # 執行提取
    extract_images(dataset_path, output_dir, num_samples)

