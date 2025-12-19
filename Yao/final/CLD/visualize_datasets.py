#!/usr/bin/env python3
"""
可視化 PrismLayersPro 和 DLCV Dataset 的處理結果
比較兩種格式轉換後的圖層結構
"""

import os
import sys
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.path.insert(0, '.')

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image"""
    # tensor shape: [C, H, W]
    if tensor.shape[0] == 4:  # RGBA
        mode = "RGBA"
    elif tensor.shape[0] == 3:  # RGB
        mode = "RGB"
    else:
        raise ValueError(f"Unsupported channel count: {tensor.shape[0]}")
    
    # Convert to numpy and scale to 0-255
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode)


def create_layer_visualization(item, sample_idx, output_dir, dataset_name):
    """
    創建圖層可視化
    
    Args:
        item: dataset[idx] 的返回值
        sample_idx: 樣本索引
        output_dir: 輸出目錄
        dataset_name: 數據集名稱（用於命名）
    """
    sample_dir = os.path.join(output_dir, f"{dataset_name}_sample_{sample_idx:02d}")
    os.makedirs(sample_dir, exist_ok=True)
    
    # 提取數據
    pixel_RGBA = item["pixel_RGBA"]  # [L, 4, H, W]
    pixel_RGB = item["pixel_RGB"]    # [L, 3, H, W]
    caption = item["caption"]
    layout = item["layout"]
    height = item["height"]
    width = item["width"]
    whole_img = item["whole_img"]
    
    num_layers = pixel_RGBA.shape[0]
    
    # 創建說明文件
    info_path = os.path.join(sample_dir, "info.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Sample Index: {sample_idx}\n")
        f.write(f"Caption: {caption}\n")
        f.write(f"Image Size: {width} x {height}\n")
        f.write(f"Total Layers: {num_layers}\n")
        f.write(f"\n")
        f.write(f"Layer Structure:\n")
        f.write(f"  Layer 0: Whole Image (完整圖片)\n")
        f.write(f"  Layer 1: Base/Background (背景)\n")
        f.write(f"  Layer 2+: Foreground Layers (前景圖層)\n")
        f.write(f"\n")
        f.write(f"Layout (Bounding Boxes):\n")
        for i, bbox in enumerate(layout):
            f.write(f"  Layer {i}: {bbox}\n")
    
    # 保存每一層的 RGBA 和 RGB 圖片
    for layer_idx in range(num_layers):
        # RGBA
        rgba_img = tensor_to_pil(pixel_RGBA[layer_idx])
        rgba_path = os.path.join(sample_dir, f"layer_{layer_idx:02d}_RGBA.png")
        rgba_img.save(rgba_path)
        
        # RGB
        rgb_img = tensor_to_pil(pixel_RGB[layer_idx])
        rgb_path = os.path.join(sample_dir, f"layer_{layer_idx:02d}_RGB.png")
        rgb_img.save(rgb_path)
    
    # 保存原始 whole_img
    whole_img_path = os.path.join(sample_dir, "whole_image_original.png")
    whole_img.save(whole_img_path)
    
    # 創建帶標註的可視化
    create_annotated_visualization(pixel_RGBA, layout, caption, sample_dir, num_layers)
    
    print(f"✅ {dataset_name} Sample {sample_idx} saved to {sample_dir}")


def create_annotated_visualization(pixel_RGBA, layout, caption, output_dir, num_layers):
    """創建帶標註的整體可視化"""
    # 創建一個大圖，顯示所有圖層
    layer_imgs = []
    
    for i in range(num_layers):
        img = tensor_to_pil(pixel_RGBA[i])
        # 縮小圖片以便並排顯示
        img.thumbnail((300, 300), Image.LANCZOS)
        
        # 添加文字標籤
        draw = ImageDraw.Draw(img)
        label = f"Layer {i}"
        if i == 0:
            label += " (Whole)"
        elif i == 1:
            label += " (Base)"
        else:
            label += f" (FG{i-1})"
        
        # 繪製標籤背景
        bbox = draw.textbbox((10, 10), label)
        draw.rectangle(bbox, fill='black')
        draw.text((10, 10), label, fill='white')
        
        layer_imgs.append(img)
    
    # 創建並排圖片
    # 計算網格大小
    cols = min(4, num_layers)
    rows = (num_layers + cols - 1) // cols
    
    tile_width = 300
    tile_height = 300
    margin = 10
    
    total_width = cols * tile_width + (cols + 1) * margin
    total_height = rows * tile_height + (rows + 1) * margin + 50  # 額外空間給標題
    
    overview = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(overview)
    
    # 添加標題
    title = f"Layers Overview: {caption[:50]}..."
    draw.text((margin, margin), title, fill='black')
    
    # 貼上圖層
    y_offset = 50 + margin
    for i, img in enumerate(layer_imgs):
        row = i // cols
        col = i % cols
        
        x = col * (tile_width + margin) + margin
        y = row * (tile_height + margin) + y_offset
        
        overview.paste(img, (x, y))
    
    overview_path = os.path.join(output_dir, "overview.png")
    overview.save(overview_path)


def main():
    output_base = "/tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD/dataset_comparison"
    os.makedirs(output_base, exist_ok=True)
    
    print("=" * 70)
    print("數據集可視化工具")
    print("=" * 70)
    
    # 1. 嘗試載入 DLCV Dataset
    print("\n[1] 載入 DLCV Dataset...")
    try:
        from tools.dlcv_dataset import DLCVLayoutDataset
        dlcv_dataset = DLCVLayoutDataset(
            data_dir='/tmp2/b12902041/Gino/TAData/DLCV_dataset',
            split='train'
        )
        print(f"✅ DLCV Dataset 載入成功！總共 {len(dlcv_dataset)} 筆資料")
        
        # 可視化前 3 筆
        print("\n可視化 DLCV Dataset 前 3 筆...")
        for i in range(min(3, len(dlcv_dataset))):
            try:
                item = dlcv_dataset[i]
                create_layer_visualization(item, i, output_base, "DLCV")
            except Exception as e:
                print(f"⚠️ Sample {i} 失敗: {e}")
        
    except Exception as e:
        print(f"❌ DLCV Dataset 載入失敗: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. 嘗試載入 PrismLayersPro (LayoutTrainDataset)
    print("\n[2] 嘗試載入 PrismLayersPro Dataset...")
    try:
        from tools.dataset import LayoutTrainDataset
        
        # 嘗試幾個可能的路徑
        possible_paths = [
            "/tmp2/b12902041/Gino/TAData/DLCV_dataset",  # 可能也支援這個格式
            # 添加其他可能的 PrismLayersPro 路徑
        ]
        
        prism_dataset = None
        for path in possible_paths:
            try:
                prism_dataset = LayoutTrainDataset(data_dir=path, split='train')
                print(f"✅ PrismLayersPro Dataset 載入成功！總共 {len(prism_dataset)} 筆資料")
                break
            except:
                continue
        
        if prism_dataset is not None:
            # 可視化前 3 筆
            print("\n可視化 PrismLayersPro Dataset 前 3 筆...")
            for i in range(min(3, len(prism_dataset))):
                try:
                    item = prism_dataset[i]
                    create_layer_visualization(item, i, output_base, "PrismLayersPro")
                except Exception as e:
                    print(f"⚠️ Sample {i} 失敗: {e}")
        else:
            print("⚠️ 找不到 PrismLayersPro 格式的數據集")
            
    except Exception as e:
        print(f"❌ PrismLayersPro Dataset 載入失敗: {e}")
    
    # 創建總體說明文件
    readme_path = os.path.join(output_base, "README.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("數據集對比說明\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("目錄結構:\n")
        f.write("  DLCV_sample_XX/          - DLCV 數據集的樣本\n")
        f.write("  PrismLayersPro_sample_XX/ - PrismLayersPro 數據集的樣本\n\n")
        
        f.write("每個樣本目錄包含:\n")
        f.write("  - info.txt               - 樣本資訊\n")
        f.write("  - overview.png           - 所有圖層的總覽\n")
        f.write("  - layer_XX_RGBA.png      - 各圖層的 RGBA 版本（帶透明度）\n")
        f.write("  - layer_XX_RGB.png       - 各圖層的 RGB 版本（無透明度）\n")
        f.write("  - whole_image_original.png - 原始完整圖片\n\n")
        
        f.write("圖層結構 (CLD 訓練格式):\n")
        f.write("  Layer 0: Whole Image     - 完整圖片（所有圖層疊加的結果）\n")
        f.write("  Layer 1: Base/Background - 純背景（沒有前景物件）\n")
        f.write("  Layer 2+: Foreground     - 各個前景圖層（物件、文字等）\n\n")
        
        f.write("檢查重點:\n")
        f.write("  1. Layer 0 是否是完整的合成圖片？\n")
        f.write("  2. Layer 1 是否是純背景（沒有前景物件）？\n")
        f.write("  3. Layer 2+ 是否包含正確的前景圖層？\n")
        f.write("  4. 每個圖層的透明度處理是否正確？\n")
        f.write("  5. Bounding box 是否與圖層對應？\n\n")
    
    print("\n" + "=" * 70)
    print(f"✅ 完成！檢查資料夾: {output_base}")
    print("=" * 70)
    print(f"\n請查看 {readme_path} 了解目錄結構")
    print(f"使用圖片查看器打開 overview.png 快速查看每個樣本的圖層結構")


if __name__ == "__main__":
    main()

