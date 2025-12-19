"""
Custom Dataset for COCO-format data with only bounding boxes (no separate layer images).
"""

import os
import glob
import numpy as np
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
import torchvision.transforms as T


def rgba2rgb(rgba_image, background=(128, 128, 128)):
    """Convert RGBA to RGB with specified background color."""
    rgb_image = Image.new("RGB", rgba_image.size, background)
    rgb_image.paste(rgba_image, mask=rgba_image.split()[3])  # Alpha channel as mask
    return rgb_image


class CustomLayoutTrainDataset(Dataset):
    """
    Dataset for layout-based training where we only have:
    - Full image
    - Bounding boxes for objects (no separate layer images)
    
    This is suitable for COCO-format data converted by convert_coco_to_cld.py
    """
    
    def __init__(self, data_dir, split="train"):
        """
        Args:
            data_dir: Directory containing parquet files in {data_dir}/data/*.parquet
            split: "train" or "val"
        """
        local_parquet_dir = os.path.join(data_dir, "data")
        local_parquet_files = sorted(glob.glob(os.path.join(local_parquet_dir, "*.parquet")))
        
        if len(local_parquet_files) == 0:
            raise FileNotFoundError(
                f"找不到 Parquet 檔案在 {local_parquet_dir}\n"
                f"請確認數據已經轉換完成，且路徑正確。"
            )
        
        # Load all parquet files
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
        
        print(f"[INFO] CustomLayoutTrainDataset loaded: {len(self.dataset)} samples for {split}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Load the full image
        img_path = item["preview"]
        
        try:
            whole_img = Image.open(img_path).convert("RGBA")
        except Exception as e:
            raise RuntimeError(f"無法載入圖片: {img_path}, 錯誤: {e}")
        
        whole_img_RGB = rgba2rgb(whole_img)
        W, H = whole_img.size
        
        # Caption
        caption = item.get("title", "A design image")
        
        # Base layout (entire image)
        base_layout = [0, 0, W - 1, H - 1]

        # Initialize with whole image as first layer
        layer_image_RGBA = [self.to_tensor(whole_img)]
        layer_image_RGB = [self.to_tensor(whole_img_RGB)]
        layout = [base_layout]

        # Add base image (same as whole image) as second layer
        layer_image_RGBA.append(self.to_tensor(whole_img))
        layer_image_RGB.append(self.to_tensor(whole_img_RGB))
        layout.append(base_layout)
        
        # Process bounding boxes and create layers
        layer_count = item["length"]
        left_list = item["left"]
        top_list = item["top"]
        width_list = item["width"]
        height_list = item["height"]
        
        for i in range(layer_count):
            # Get bbox in [x, y, w, h] format
            x = left_list[i]
            y = top_list[i]
            w = width_list[i]
            h = height_list[i]
            
            # Convert to [x1, y1, x2, y2]
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            
            # Ensure bbox is within image bounds
            x1 = max(0, min(x1, W))
            y1 = max(0, min(y1, H))
            x2 = max(0, min(x2, W))
            y2 = max(0, min(y2, H))
            
            # Crop the region from the whole image
            try:
                cropped_RGBA = whole_img.crop((x1, y1, x2, y2))
                cropped_RGB = whole_img_RGB.crop((x1, y1, x2, y2))
            except Exception as e:
                print(f"[WARNING] 無法裁剪圖層 {i} (bbox: {x1},{y1},{x2},{y2}): {e}, 使用空白圖層")
                cropped_RGBA = Image.new("RGBA", (max(1, x2-x1), max(1, y2-y1)), (0, 0, 0, 0))
                cropped_RGB = Image.new("RGB", (max(1, x2-x1), max(1, y2-y1)), (128, 128, 128))
            
            # Create canvas and paste cropped region
            canvas_RGBA = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            canvas_RGB = Image.new("RGB", (W, H), (128, 128, 128))

            # Resize if needed
            target_w, target_h = x2 - x1, y2 - y1
            if target_w > 0 and target_h > 0:
                if cropped_RGBA.size != (target_w, target_h):
                    cropped_RGBA = cropped_RGBA.resize((target_w, target_h), Image.BILINEAR)
                    cropped_RGB = cropped_RGB.resize((target_w, target_h), Image.BILINEAR)

                canvas_RGBA.paste(cropped_RGBA, (x1, y1), cropped_RGBA)
                canvas_RGB.paste(cropped_RGB, (x1, y1))

            layer_image_RGBA.append(self.to_tensor(canvas_RGBA))
            layer_image_RGB.append(self.to_tensor(canvas_RGB))
            layout.append([x1, y1, x2, y2])

        # Stack tensors
        import torch
        pixel_RGBA = torch.stack(layer_image_RGBA, dim=0)  # [L+2, 4, H, W]
        pixel_RGB = torch.stack(layer_image_RGB, dim=0)    # [L+2, 3, H, W]

        return {
            "pixel_RGBA": pixel_RGBA,
            "pixel_RGB": pixel_RGB,
            "whole_img": whole_img,
            "caption": caption,
            "layout": layout,
            "height": H,
            "width": W,
        }


def collate_fn(batch):
    """Simple collate function that returns the first item (batch_size=1)."""
    return batch[0] if len(batch) == 1 else batch
