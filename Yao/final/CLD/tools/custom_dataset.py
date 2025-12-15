"""
自訂資料集適配器，將 DLCV final project 資料集轉換為 CLD 所需的格式
"""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
import torchvision.transforms as T
from collections import defaultdict
import glob
import os

def collate_fn(batch):
    """與原始 collate_fn 相同"""
    pixels_RGBA = [torch.stack(item["pixel_RGBA"]) for item in batch]  # [L, C, H, W]
    pixels_RGB  = [torch.stack(item["pixel_RGB"])  for item in batch]  # [L, C, H, W]
    pixels_RGBA = torch.stack(pixels_RGBA)  # [B, L, C, H, W]
    pixels_RGB  = torch.stack(pixels_RGB)   # [B, L, C, H, W]

    return {
        "pixel_RGBA": pixels_RGBA,
        "pixel_RGB": pixels_RGB,
        "whole_img": [item["whole_img"] for item in batch],
        "caption": [item["caption"] for item in batch],
        "height": [item["height"] for item in batch],
        "width": [item["width"] for item in batch],
        "layout": [item["layout"] for item in batch],
    }

class CustomLayoutDataset(Dataset):
    """
    自訂資料集類別，適配 DLCV final project 資料集格式
    將 Parquet 格式的資料集轉換為 CLD 所需的格式
    """
    def __init__(self, data_dir, split="train"):
        """
        Args:
            data_dir: 資料集目錄路徑（包含 snapshots 的目錄）
            split: "train", "val", 或 "test"
        """
        # 載入所有 Parquet 檔案
        parquet_files = glob.glob(os.path.join(data_dir, "snapshots/*/data/train-*.parquet"))
        if not parquet_files:
            parquet_files = glob.glob(os.path.join(data_dir, "**/train-*.parquet"), recursive=True)
        
        if not parquet_files:
            raise ValueError(f"無法在 {data_dir} 找到 Parquet 檔案")
        
        print(f"找到 {len(parquet_files)} 個 Parquet 檔案")
        
        # 載入所有 Parquet 檔案
        datasets_list = []
        for pf in sorted(parquet_files):
            print(f"  載入: {pf}")
            ds = load_dataset('parquet', data_files=pf)
            datasets_list.append(ds['train'])
        
        # 合併所有資料集
        full_dataset = concatenate_datasets(datasets_list)
        print(f"總共載入 {len(full_dataset)} 筆資料")
        
        # 根據 split 分割資料集
        # 使用簡單的 90/5/5 分割（train/val/test）
        total_len = len(full_dataset)
        idx_90 = int(total_len * 0.9)
        idx_95 = int(total_len * 0.95)
        
        if split == "train":
            selected_idx = list(range(0, idx_90))
        elif split == "val":
            selected_idx = list(range(idx_90, idx_95))
        elif split == "test":
            selected_idx = list(range(idx_95, total_len))
        else:
            raise ValueError("split 必須是 'train', 'val', 或 'test'")
        
        self.dataset = full_dataset.select(selected_idx)
        print(f"{split} split: {len(self.dataset)} 筆資料")
        
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        def rgba2rgb(img_RGBA):
            """將 RGBA 轉換為 RGB，使用灰色背景"""
            img_RGB = Image.new("RGB", img_RGBA.size, (128, 128, 128))
            img_RGB.paste(img_RGBA, mask=img_RGBA.split()[3])
            return img_RGB

        def get_img(x):
            """取得圖片，支援 PIL Image 或檔案路徑"""
            if isinstance(x, str):
                img_RGBA = Image.open(x).convert("RGBA")
                img_RGB = rgba2rgb(img_RGBA)
            else:
                img_RGBA = x.convert("RGBA")
                img_RGB = rgba2rgb(img_RGBA)
            return img_RGBA, img_RGB

        # 取得完整圖片（preview）
        whole_img_RGBA, whole_img_RGB = get_img(item["preview"])
        W, H = whole_img_RGBA.size
        
        # 使用 title 作為 caption，如果沒有則使用預設值
        whole_cap = item.get("title", "A design image")
        if not whole_cap or whole_cap == "":
            whole_cap = "A design image"
        
        # 基礎 layout（整個畫布）
        base_layout = [0, 0, W - 1, H - 1]

        # 初始化列表
        layer_image_RGBA = [self.to_tensor(whole_img_RGBA)]
        layer_image_RGB  = [self.to_tensor(whole_img_RGB)]
        layout = [base_layout]

        # 取得圖層數量
        layer_count = item.get("length", 0)
        
        # 取得圖層圖片和位置資訊
        layer_images = item.get("image", [])
        left_list = item.get("left", [])
        top_list = item.get("top", [])
        width_list = item.get("width", [])
        height_list = item.get("height", [])
        
        # 確保所有列表長度一致
        min_len = min(len(layer_images), len(left_list), len(top_list), 
                     len(width_list), len(height_list), layer_count)
        
        # 處理每個圖層
        for i in range(min_len):
            # 取得圖層圖片
            if i < len(layer_images) and layer_images[i] is not None:
                img_RGBA, img_RGB = get_img(layer_images[i])
            else:
                # 如果沒有圖層圖片，建立透明圖層
                img_RGBA = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                img_RGB = rgba2rgb(img_RGBA)
            
            # 取得圖層位置（轉換為整數）
            w0 = int(left_list[i]) if i < len(left_list) else 0
            h0 = int(top_list[i]) if i < len(top_list) else 0
            w1 = w0 + int(width_list[i]) if i < len(width_list) else W
            h1 = h0 + int(height_list[i]) if i < len(height_list) else H
            
            # 確保邊界在畫布範圍內
            w0, h0 = max(0, w0), max(0, h0)
            w1, h1 = min(W, w1), min(H, h1)
            
            # 建立畫布並貼上圖層
            canvas_RGBA = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            canvas_RGB = Image.new("RGB", (W, H), (128, 128, 128))

            W_img, H_img = w1 - w0, h1 - h0
            if img_RGBA.size != (W_img, H_img) and W_img > 0 and H_img > 0:
                img_RGBA = img_RGBA.resize((W_img, H_img), Image.BILINEAR)
                img_RGB  = img_RGB.resize((W_img, H_img), Image.BILINEAR)

            if W_img > 0 and H_img > 0:
                canvas_RGBA.paste(img_RGBA, (w0, h0), img_RGBA)
                canvas_RGB.paste(img_RGB, (w0, h0))

            layer_image_RGBA.append(self.to_tensor(canvas_RGBA))
            layer_image_RGB.append(self.to_tensor(canvas_RGB))
            layout.append([w0, h0, w1, h1])

        # 如果沒有圖層，至少要有背景
        if layer_count == 0:
            # 使用完整圖片作為背景
            base_img_RGBA, base_img_RGB = whole_img_RGBA, whole_img_RGB
            layer_image_RGBA.append(self.to_tensor(base_img_RGBA))
            layer_image_RGB.append(self.to_tensor(base_img_RGB))
            layout.append(base_layout)

        return {
            "pixel_RGBA": layer_image_RGBA,
            "pixel_RGB": layer_image_RGB,
            "whole_img": whole_img_RGB,
            "caption": whole_cap,
            "height": H,
            "width": W,
            "layout": layout,
        }

# 為了向後相容，也提供 LayoutTrainDataset 別名
LayoutTrainDataset = CustomLayoutDataset







