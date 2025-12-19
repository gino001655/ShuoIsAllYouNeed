#!/usr/bin/env python3
"""
檢查 preprocessed_data 中所有圖片的尺寸
找出最大寬度、最大高度、以及尺寸分佈
"""

import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def check_image_sizes(base_dir):
    """檢查指定目錄下所有圖片的尺寸"""
    
    base_path = Path(base_dir)
    
    # 找出所有圖片檔案
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    image_files = []
    
    print("[INFO] 搜尋圖片檔案...")
    for split in ['train', 'val']:
        img_dir = base_path / 'images' / split
        if img_dir.exists():
            for img_file in img_dir.glob('*'):
                if img_file.suffix in image_extensions:
                    image_files.append(img_file)
    
    print(f"[INFO] 找到 {len(image_files)} 張圖片")
    
    if len(image_files) == 0:
        print("[ERROR] 沒有找到任何圖片！")
        return
    
    # 統計資訊
    max_width = 0
    max_height = 0
    min_width = float('inf')
    min_height = float('inf')
    
    width_list = []
    height_list = []
    
    size_distribution = defaultdict(int)
    not_divisible_by_8 = []
    not_divisible_by_16 = []
    
    # 遍歷所有圖片
    print("[INFO] 檢查圖片尺寸...")
    for img_path in tqdm(image_files, desc="處理圖片", unit="張"):
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                
                # 更新統計
                max_width = max(max_width, width)
                max_height = max(max_height, height)
                min_width = min(min_width, width)
                min_height = min(min_height, height)
                
                width_list.append(width)
                height_list.append(height)
                
                size_distribution[(width, height)] += 1
                
                # 檢查是否是 8 的倍數
                if width % 8 != 0 or height % 8 != 0:
                    not_divisible_by_8.append((img_path.name, width, height))
                
                # 檢查是否是 16 的倍數
                if width % 16 != 0 or height % 16 != 0:
                    not_divisible_by_16.append((img_path.name, width, height))
                
        except Exception as e:
            print(f"[ERROR] 無法讀取圖片 {img_path}: {e}")
    
    # 輸出統計結果
    print("\n" + "="*70)
    print("圖片尺寸統計")
    print("="*70)
    
    print(f"\n總圖片數量: {len(image_files)}")
    
    print(f"\n尺寸範圍:")
    print(f"  最大寬度: {max_width} px")
    print(f"  最大高度: {max_height} px")
    print(f"  最小寬度: {min_width} px")
    print(f"  最小高度: {min_height} px")
    
    print(f"\n平均尺寸:")
    avg_width = sum(width_list) / len(width_list)
    avg_height = sum(height_list) / len(height_list)
    print(f"  平均寬度: {avg_width:.1f} px")
    print(f"  平均高度: {avg_height:.1f} px")
    
    print(f"\n最常見的尺寸 (前 10 名):")
    sorted_sizes = sorted(size_distribution.items(), key=lambda x: x[1], reverse=True)
    for i, ((w, h), count) in enumerate(sorted_sizes[:10], 1):
        percentage = (count / len(image_files)) * 100
        print(f"  {i}. {w}x{h}: {count} 張 ({percentage:.1f}%)")
    
    print(f"\n不是 8 的倍數的圖片:")
    print(f"  數量: {len(not_divisible_by_8)} / {len(image_files)} ({len(not_divisible_by_8)/len(image_files)*100:.1f}%)")
    if len(not_divisible_by_8) > 0 and len(not_divisible_by_8) <= 10:
        for name, w, h in not_divisible_by_8:
            print(f"    - {name}: {w}x{h}")
    elif len(not_divisible_by_8) > 10:
        print(f"  (前 10 張)")
        for name, w, h in not_divisible_by_8[:10]:
            print(f"    - {name}: {w}x{h}")
    
    print(f"\n不是 16 的倍數的圖片:")
    print(f"  數量: {len(not_divisible_by_16)} / {len(image_files)} ({len(not_divisible_by_16)/len(image_files)*100:.1f}%)")
    if len(not_divisible_by_16) > 0 and len(not_divisible_by_16) <= 10:
        for name, w, h in not_divisible_by_16:
            print(f"    - {name}: {w}x{h}")
    elif len(not_divisible_by_16) > 10:
        print(f"  (前 10 張)")
        for name, w, h in not_divisible_by_16[:10]:
            print(f"    - {name}: {w}x{h}")
    
    print("\n" + "="*70)
    print("檢查完成")
    print("="*70)
    
    # 返回統計資訊
    return {
        'total': len(image_files),
        'max_width': max_width,
        'max_height': max_height,
        'min_width': min_width,
        'min_height': min_height,
        'avg_width': avg_width,
        'avg_height': avg_height,
        'not_divisible_by_8': len(not_divisible_by_8),
        'not_divisible_by_16': len(not_divisible_by_16),
        'size_distribution': dict(sorted_sizes[:10])
    }

if __name__ == "__main__":
    import sys
    
    # 預設路徑
    default_path = "/tmp2/b12902041/Gino/preprocessed_data"
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = default_path
    
    if not os.path.exists(data_dir):
        print(f"[ERROR] 目錄不存在: {data_dir}")
        sys.exit(1)
    
    print(f"[INFO] 檢查目錄: {data_dir}")
    stats = check_image_sizes(data_dir)

