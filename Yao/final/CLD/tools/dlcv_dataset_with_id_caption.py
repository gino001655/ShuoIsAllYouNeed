"""
支持 ID-based caption mapping 的 Dataset
這樣 TAData 可以直接用，不需要轉換！

用法：
1. 為 TAData 生成 ID-based caption_mapping.json:
   {
     "sample_0": "caption for sample 0",
     "sample_1": "caption for sample 1",
     "589b107395a7a863ddcc47d8": "caption for this ID",
     ...
   }

2. 或者用 index-based:
   {
     "0": "caption for index 0",
     "1": "caption for index 1",
     ...
   }
"""

# 複製原有的 dlcv_dataset.py 並修改 caption 查找邏輯
import os
import sys
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
import torchvision.transforms as T
import torch


def rgba2rgb(rgba_image, background=(128, 128, 128)):
    """Convert RGBA to RGB with specified background color."""
    rgb_image = Image.new("RGB", rgba_image.size, background)
    rgb_image.paste(rgba_image, mask=rgba_image.split()[3])
    return rgb_image


class DLCVLayoutDatasetWithIDCaption(Dataset):
    """
    支持 ID-based 或 index-based caption mapping
    """
    
    def __init__(self, data_dir, split="train", caption_mapping_path=None, enable_debug=True):
        self.data_dir = data_dir
        self.enable_debug = enable_debug
        
        local_parquet_dir = os.path.join(data_dir, "data")
        local_parquet_files = sorted(glob.glob(os.path.join(local_parquet_dir, "*.parquet")))
        
        if len(local_parquet_files) == 0:
            raise FileNotFoundError(f"找不到 Parquet 檔案在 {local_parquet_dir}")
        
        print(f"[INFO] 載入 {len(local_parquet_files)} 個 parquet 檔案...")
        
        # Load caption mapping
        self.caption_mapping = {}
        if caption_mapping_path and os.path.exists(caption_mapping_path):
            import json
            with open(caption_mapping_path, 'r', encoding='utf-8') as f:
                self.caption_mapping = json.load(f)
            print(f"[INFO] 載入 caption mapping: {len(self.caption_mapping)} 個 captions")
        
        # Load parquet files
        datasets_list = []
        for pf in local_parquet_files:
            ds = load_dataset("parquet", data_files=pf)["train"]
            datasets_list.append(ds)
        
        full_dataset = concatenate_datasets(datasets_list)
        
        # Split train/val (90/10)
        total_len = len(full_dataset)
        idx_90 = int(total_len * 0.9)
        
        if split == "train":
            self.dataset = full_dataset.select(range(0, idx_90))
        else:
            self.dataset = full_dataset.select(range(idx_90, total_len))
        
        self.to_tensor = T.ToTensor()
        
        print(f"[INFO] DLCVLayoutDataset loaded: {len(self.dataset)} samples for {split}")
    
    def __len__(self):
        return len(self.dataset)
    
    def _get_caption_for_sample(self, idx, item):
        """
        查找 caption，支持多種方式：
        1. ID-based: 用 item['id']
        2. Index-based: 用 str(idx)
        3. Fallback: 用 item.get('title')
        """
        # 嘗試用 ID
        if 'id' in item and item['id'] in self.caption_mapping:
            return self.caption_mapping[item['id']]
        
        # 嘗試用 index
        if str(idx) in self.caption_mapping:
            return self.caption_mapping[str(idx)]
        
        # Fallback: 用內建 title
        return item.get('title', 'A design image')
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get caption
        caption = self._get_caption_for_sample(idx, item)
        
        # Get preview image
        preview_value = item["preview"]
        if isinstance(preview_value, Image.Image):
            whole_img_RGBA = preview_value.convert("RGBA")
        elif isinstance(preview_value, str):
            whole_img_RGBA = Image.open(preview_value).convert("RGBA")
        else:
            raise ValueError(f"preview 欄位格式不支持: {type(preview_value)}")
        
        # Canvas size
        canvas_W = item.get("canvas_width", whole_img_RGBA.width)
        canvas_H = item.get("canvas_height", whole_img_RGBA.height)
        W = ((canvas_W + 15) // 16) * 16
        H = ((canvas_H + 15) // 16) * 16
        
        if whole_img_RGBA.size != (W, H):
            whole_img_RGBA = whole_img_RGBA.resize((W, H), Image.LANCZOS)
        whole_img_RGB = rgba2rgb(whole_img_RGBA)
        
        # Base layout
        base_layout = [0, 0, W - 1, H - 1]
        
        # Initialize with whole image
        layer_image_RGBA = [self.to_tensor(whole_img_RGBA)]
        layer_image_RGB = [self.to_tensor(whole_img_RGB)]
        layout = [base_layout]
        
        # Find background layer
        layer_types = item.get("type", [])
        background_indices = [i for i, t in enumerate(layer_types) if t == 3]
        
        if len(background_indices) > 0:
            bg_idx = background_indices[0]
            bg_img = item["image"][bg_idx]
            
            if isinstance(bg_img, Image.Image):
                bg_img_RGBA = bg_img.convert("RGBA")
            elif isinstance(bg_img, str):
                bg_img_RGBA = Image.open(bg_img).convert("RGBA")
            else:
                bg_img_RGBA = whole_img_RGBA
            
            if bg_img_RGBA.size != (W, H):
                bg_img_RGBA = bg_img_RGBA.resize((W, H), Image.LANCZOS)
            bg_img_RGB = rgba2rgb(bg_img_RGBA)
        else:
            bg_img_RGBA = Image.new("RGBA", (W, H), (255, 255, 255, 0))
            bg_img_RGB = Image.new("RGB", (W, H), (255, 255, 255))
        
        layer_image_RGBA.append(self.to_tensor(bg_img_RGBA))
        layer_image_RGB.append(self.to_tensor(bg_img_RGB))
        layout.append(base_layout)
        
        # Process individual layers
        layer_count = item["length"]
        layer_images = item["image"]
        left_list = item["left"]
        top_list = item["top"]
        width_list = item["width"]
        height_list = item["height"]
        
        for i in range(layer_count):
            # Skip background layers
            if layer_types and i < len(layer_types) and layer_types[i] == 3:
                continue
            
            # Get layer image
            layer_img = layer_images[i]
            
            # Handle different types
            if layer_img is None:
                continue
            elif isinstance(layer_img, Image.Image):
                layer_img_RGBA = layer_img.convert("RGBA")
            elif isinstance(layer_img, str):
                layer_img_RGBA = Image.open(layer_img).convert("RGBA")
            else:
                continue
            
            layer_img_RGB = rgba2rgb(layer_img_RGBA)
            
            # Get bbox
            x = int(left_list[i])
            y = int(top_list[i])
            w = int(width_list[i])
            h = int(height_list[i])
            
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            
            # Clamp to canvas bounds
            x1 = max(0, min(x1, W))
            y1 = max(0, min(y1, H))
            x2 = max(0, min(x2, W))
            y2 = max(0, min(y2, H))
            
            # Create canvas
            canvas_RGBA = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            canvas_RGB = Image.new("RGB", (W, H), (128, 128, 128))
            
            # Resize and paste
            target_w, target_h = x2 - x1, y2 - y1
            if target_w > 0 and target_h > 0:
                if layer_img_RGBA.size != (target_w, target_h):
                    layer_img_RGBA = layer_img_RGBA.resize((target_w, target_h), Image.LANCZOS)
                    layer_img_RGB = layer_img_RGB.resize((target_w, target_h), Image.LANCZOS)
                
                canvas_RGBA.paste(layer_img_RGBA, (x1, y1), layer_img_RGBA)
                canvas_RGB.paste(layer_img_RGB, (x1, y1))
            
            layer_image_RGBA.append(self.to_tensor(canvas_RGBA))
            layer_image_RGB.append(self.to_tensor(canvas_RGB))
            layout.append([x1, y1, x2, y2])
        
        # Stack tensors
        pixel_RGBA = torch.stack(layer_image_RGBA, dim=0)
        pixel_RGB = torch.stack(layer_image_RGB, dim=0)
        
        return {
            "pixel_RGBA": pixel_RGBA,
            "pixel_RGB": pixel_RGB,
            "whole_img": whole_img_RGB,
            "caption": caption,
            "layout": layout,
            "height": H,
            "width": W,
        }


def collate_fn(batch):
    """Simple collate function."""
    return batch[0] if len(batch) == 1 else batch

