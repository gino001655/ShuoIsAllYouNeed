"""
Dataset adapter for DLCV dataset format (PrismLayersPro-like structure)
Converts DLCV format to CLD training format
"""

import os
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
    rgb_image.paste(rgba_image, mask=rgba_image.split()[3])  # Alpha channel as mask
    return rgb_image


class DLCVLayoutDataset(Dataset):
    """
    Dataset adapter for DLCV dataset format.
    
    DLCV format:
    - preview: full image
    - title: caption
    - image: list of layer images (PNG with transparency)
    - left, top, width, height: lists of bbox coordinates
    - length: number of layers
    
    Converts to CLD training format:
    - pixel_RGBA, pixel_RGB: stacked layer tensors
    - whole_img: full image
    - caption: text description
    - layout: list of [x1, y1, x2, y2] bboxes
    - height, width: image dimensions
    """
    
    def __init__(self, data_dir, split="train", caption_mapping_path=None):
        """
        Args:
            data_dir: Directory containing parquet files in {data_dir}/data/*.parquet
            split: "train" or "val"
            caption_mapping_path: Optional path to caption mapping JSON file
        """
        self.data_dir = data_dir  # Save for path remapping
        local_parquet_dir = os.path.join(data_dir, "data")
        local_parquet_files = sorted(glob.glob(os.path.join(local_parquet_dir, "*.parquet")))
        
        if len(local_parquet_files) == 0:
            raise FileNotFoundError(
                f"找不到 Parquet 檔案在 {local_parquet_dir}\n"
                f"請確認數據路徑正確。"
            )
        
        print(f"[INFO] 載入 {len(local_parquet_files)} 個 parquet 檔案...")
        
        # Load caption mapping if provided
        self.caption_mapping = {}
        if caption_mapping_path and os.path.exists(caption_mapping_path):
            import json
            with open(caption_mapping_path, 'r', encoding='utf-8') as f:
                self.caption_mapping = json.load(f)
            print(f"[INFO] 載入 caption mapping: {len(self.caption_mapping)} 個 captions")
        
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
        
        print(f"[INFO] DLCVLayoutDataset loaded: {len(self.dataset)} samples for {split}")
    
    def __len__(self):
        return len(self.dataset)
    
    def _fix_image_path(self, image_or_path):
        """
        Fix hardcoded image paths in parquet files.
        If image_or_path is a string path pointing to preprocessed_data,
        remap it to the actual data_dir location.
        """
        if not isinstance(image_or_path, str):
            return image_or_path  # Already an Image object or None
        
        # Check if it's a hardcoded path that needs remapping
        if "/preprocessed_data/" in image_or_path:
            # Extract the relative part after "preprocessed_data/"
            # Example: /workspace/dataset/preprocessed_data/images/train/00017531.png
            #       -> images/train/00017531.png
            parts = image_or_path.split("/preprocessed_data/")
            if len(parts) == 2:
                relative_path = parts[1]
                # Construct new path: data_dir + relative_path
                new_path = os.path.join(self.data_dir, relative_path)
                
                if os.path.exists(new_path):
                    return new_path
                else:
                    # Try without the images/ prefix if that doesn't work
                    # Some datasets might have: data_dir/train/xxx.png
                    if relative_path.startswith("images/"):
                        alt_path = os.path.join(self.data_dir, relative_path.replace("images/", "", 1))
                        if os.path.exists(alt_path):
                            return alt_path
        
        # Return as-is if no remapping needed
        return image_or_path
    
    def __getitem__(self, idx):
        # Get preview path and image
        # Strategy: datasets library may convert 'preview' to Image automatically,
        # but we can get the original path from Arrow table
        preview_path = None
        preview_img = None
        
        # First, try to get raw path from Arrow table
        try:
            if hasattr(self.dataset, '_data'):
                preview_col = self.dataset._data.column('preview')
                if preview_col and idx < len(preview_col):
                    preview_value = preview_col[idx].as_py()
                    if isinstance(preview_value, str):
                        preview_path = preview_value
        except Exception as e:
            pass
        
        # Now get the processed item
        item = self.dataset[idx]
        
        # Get full preview image (支持两种格式：Image 对象或字符串路径)
        preview_value = item["preview"]
        if isinstance(preview_value, Image.Image):
            # 格式 1: Image 对象（如 TAData/DLCV_dataset）
            preview_img = preview_value
        elif isinstance(preview_value, str):
            # 格式 2: 字符串路径（如自己生成的 cld_dataset）
            preview_path = preview_value
            # Fix path if needed
            fixed_path = self._fix_image_path(preview_path)
            try:
                preview_img = Image.open(fixed_path)
            except Exception as e:
                raise ValueError(f"无法打开图片: {fixed_path} (原路径: {preview_path}), 错误: {e}")
        else:
            raise ValueError(f"preview 欄位格式不支持: {type(preview_value)}")
        
        # Canvas size (from metadata, not from image size)
        # Canvas size (from metadata, not from image size)
        canvas_W = item.get("canvas_width", preview_img.width)
        canvas_H = item.get("canvas_height", preview_img.height)

        # Force dimensions to be multiples of 16 to avoid mismatches between VAE grid and quantized boxes
        W = ((canvas_W + 15) // 16) * 16
        H = ((canvas_H + 15) // 16) * 16
        
        # Convert to RGBA and resize to canvas size if needed
        whole_img_RGBA = preview_img.convert("RGBA")
        if whole_img_RGBA.size != (W, H):
            whole_img_RGBA = whole_img_RGBA.resize((W, H), Image.LANCZOS)
        whole_img_RGB = rgba2rgb(whole_img_RGBA)
        
        # Caption: prioritize caption_mapping, fallback to title
        caption_found = False
        if self.caption_mapping and preview_path:
            # Try original path first
            if preview_path in self.caption_mapping:
                caption = self.caption_mapping[preview_path]
                caption_found = True
            else:
                # Try fixed path
                fixed_preview_path = self._fix_image_path(preview_path)
                if fixed_preview_path in self.caption_mapping:
                    caption = self.caption_mapping[fixed_preview_path]
                    caption_found = True
            
            if caption_found:
                # Debug: print for first few samples
                if not hasattr(self, '_caption_debug_count'):
                    self._caption_debug_count = 0
                if self._caption_debug_count < 3:
                    print(f"[DEBUG] Sample {idx}: Using caption mapping")
                    print(f"  Path: {preview_path}")
                    print(f"  Caption: {caption[:80]}...")
                    self._caption_debug_count += 1
        
        if not caption_found:
            caption = item.get("title", "A design image")
            # Debug
            if not hasattr(self, '_caption_debug_count'):
                self._caption_debug_count = 0
            if self._caption_debug_count < 3:
                print(f"[DEBUG] Sample {idx}: Using default title")
                print(f"  Path from arrow: {preview_path}")
                print(f"  Path in mapping keys: {preview_path in self.caption_mapping if self.caption_mapping else 'No mapping'}")
                if self.caption_mapping and preview_path:
                    # 检查是否路径格式不同
                    sample_keys = list(self.caption_mapping.keys())[:3]
                    print(f"  Sample mapping keys: {sample_keys}")
                self._caption_debug_count += 1
        
        # Base layout (entire canvas)
        base_layout = [0, 0, W - 1, H - 1]
        
        # Initialize with whole image as first layer
        layer_image_RGBA = [self.to_tensor(whole_img_RGBA)]
        layer_image_RGB = [self.to_tensor(whole_img_RGB)]
        layout = [base_layout]
        
        # Find background layer (type == 3: ColoredBackground)
        layer_types = item.get("type", [])
        background_indices = [i for i, t in enumerate(layer_types) if t == 3]
        
        if len(background_indices) > 0:
            # Use the first ColoredBackground as base image
            bg_idx = background_indices[0]
            bg_img = item["image"][bg_idx]
            
            # Fix path if it's a string
            if isinstance(bg_img, str):
                bg_img = self._fix_image_path(bg_img)
                try:
                    bg_img = Image.open(bg_img)
                except Exception as e:
                    print(f"[WARNING] Cannot load background image: {e}")
                    bg_img = None
            
            if bg_img is not None and isinstance(bg_img, Image.Image):
                bg_img_RGBA = bg_img.convert("RGBA")
                # Resize background to canvas size
                if bg_img_RGBA.size != (W, H):
                    bg_img_RGBA = bg_img_RGBA.resize((W, H), Image.LANCZOS)
                bg_img_RGB = rgba2rgb(bg_img_RGBA)
            else:
                # Fallback: use whole image
                bg_img_RGBA = whole_img_RGBA
                bg_img_RGB = whole_img_RGB
        else:
            # No ColoredBackground found: create blank background or use whole image
            # Option 1: Blank background (more correct for training)
            bg_img_RGBA = Image.new("RGBA", (W, H), (255, 255, 255, 0))  # Transparent
            bg_img_RGB = Image.new("RGB", (W, H), (255, 255, 255))       # White
            
            # Option 2: Use whole image as background (less ideal)
            # bg_img_RGBA = whole_img_RGBA
            # bg_img_RGB = whole_img_RGB
        
        # Add base image as second layer
        layer_image_RGBA.append(self.to_tensor(bg_img_RGBA))
        layer_image_RGB.append(self.to_tensor(bg_img_RGB))
        layout.append(base_layout)
        
        # Process individual layers
        layer_count = item["length"]
        layer_images = item["image"]  # List of PIL Images
        left_list = item["left"]
        top_list = item["top"]
        width_list = item["width"]
        height_list = item["height"]
        
        # Debug: print layer info for first few samples
        if not hasattr(self, '_layer_debug_count'):
            self._layer_debug_count = 0
        if self._layer_debug_count < 3:
            print(f"[DEBUG] Sample {idx}: Total layers in dataset: {layer_count}")
            if layer_types:
                print(f"  Layer types: {layer_types}")
        
        for i in range(layer_count):
            # Skip background layers (already added as base)
            if layer_types and i < len(layer_types) and layer_types[i] == 3:
                if self._layer_debug_count < 3:
                    print(f"  [SKIP] Layer {i}: ColoredBackground (type=3)")
                continue  # Skip ColoredBackground
            
            # Get layer image
            layer_img = layer_images[i]
            
            if layer_img is None:
                # If no layer image, skip
                if self._layer_debug_count < 3:
                    print(f"  [SKIP] Layer {i}: layer_img is None")
                continue
            
            # Fix path if it's a string (hardcoded path in parquet)
            if isinstance(layer_img, str):
                original_path = layer_img
                layer_img = self._fix_image_path(layer_img)
                
                if self._layer_debug_count < 3:
                    print(f"  [PATH FIX] Layer {i}: {original_path}")
                    print(f"           -> {layer_img}")
                
                # Try to load the image
                try:
                    layer_img = Image.open(layer_img)
                    if self._layer_debug_count < 3:
                        print(f"           ✓ Successfully loaded image: {layer_img.size}")
                except Exception as e:
                    if self._layer_debug_count < 3:
                        print(f"  [SKIP] Layer {i}: Cannot load image: {e}")
                    continue
            
            # Convert to RGBA
            if not isinstance(layer_img, Image.Image):
                if self._layer_debug_count < 3:
                    print(f"  [SKIP] Layer {i}: Not an Image, type={type(layer_img)}")
                continue
            
            if self._layer_debug_count < 3:
                print(f"  [ADDED] Layer {i}: Image size={layer_img.size}, bbox=({left_list[i]}, {top_list[i]}, {width_list[i]}, {height_list[i]})")
            
            layer_img_RGBA = layer_img.convert("RGBA")
            layer_img_RGB = rgba2rgb(layer_img_RGBA)
            
            # Get bbox (convert from left, top, width, height to x1, y1, x2, y2)
            x = int(left_list[i])
            y = int(top_list[i])
            w = int(width_list[i])
            h = int(height_list[i])
            
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            
            # Ensure bbox is within canvas bounds
            x1 = max(0, min(x1, W))
            y1 = max(0, min(y1, H))
            x2 = max(0, min(x2, W))
            y2 = max(0, min(y2, H))
            
            # Create canvas and place layer image
            canvas_RGBA = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            canvas_RGB = Image.new("RGB", (W, H), (128, 128, 128))
            
            # Resize layer image to match bbox size
            target_w, target_h = x2 - x1, y2 - y1
            if target_w > 0 and target_h > 0:
                if layer_img_RGBA.size != (target_w, target_h):
                    layer_img_RGBA = layer_img_RGBA.resize((target_w, target_h), Image.LANCZOS)
                    layer_img_RGB = layer_img_RGB.resize((target_w, target_h), Image.LANCZOS)
                
                # Paste at the specified location
                canvas_RGBA.paste(layer_img_RGBA, (x1, y1), layer_img_RGBA)
                canvas_RGB.paste(layer_img_RGB, (x1, y1))
            
            layer_image_RGBA.append(self.to_tensor(canvas_RGBA))
            layer_image_RGB.append(self.to_tensor(canvas_RGB))
            layout.append([x1, y1, x2, y2])
        
        # Stack tensors
        pixel_RGBA = torch.stack(layer_image_RGBA, dim=0)  # [L+2, 4, H, W]
        pixel_RGB = torch.stack(layer_image_RGB, dim=0)    # [L+2, 3, H, W]
        
        # Debug: print final layer count
        if self._layer_debug_count < 3:
            print(f"  [RESULT] Total layers added: {len(layout)} (including whole_img + background + {len(layout)-2} foreground layers)")
            print(f"  Layout boxes: {layout}")
            self._layer_debug_count += 1
        
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
    """Simple collate function that returns the first item (batch_size=1)."""
    return batch[0] if len(batch) == 1 else batch


