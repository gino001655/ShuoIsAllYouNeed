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
    
    def __init__(self, data_dir, split="train", caption_mapping_path=None, enable_debug=True):
        """
        Args:
            data_dir: Directory containing parquet files in {data_dir}/data/*.parquet
            split: "train" or "val"
            caption_mapping_path: Optional path to caption mapping JSON file
            enable_debug: If True, print debug info for first few samples (default: True for backwards compatibility)
        """
        self.data_dir = data_dir  # Save for path remapping
        self.enable_debug = enable_debug
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
        self.caption_mapping_indexed = {}  # Index-based mapping
        if caption_mapping_path and os.path.exists(caption_mapping_path):
            import json
            import re
            with open(caption_mapping_path, 'r', encoding='utf-8') as f:
                path_based_mapping = json.load(f)
            
            # Keep path-based mapping
            self.caption_mapping = path_based_mapping
            
            # Also create index-based mapping for TAData (Image objects without paths)
            # Extract index from paths like "/workspace/.../00000123.png"
            pattern = re.compile(r'(\d{8})\.png$')
            for path, caption in path_based_mapping.items():
                match = pattern.search(path)
                if match:
                    idx = int(match.group(1))  # Extract number, remove leading zeros
                    self.caption_mapping_indexed[idx] = caption
            
            print(f"[INFO] 載入 caption mapping: {len(self.caption_mapping)} 個 captions (path-based)")
            print(f"[INFO] 建立 index-based mapping: {len(self.caption_mapping_indexed)} 個 captions")
        
        # Load all parquet files (with pyarrow fallback for metadata issues)
        datasets_list = []
        use_pyarrow_fallback = False
        
        for pf in local_parquet_files:
            try:
                ds = load_dataset("parquet", data_files=pf)["train"]
                datasets_list.append(ds)
            except (TypeError, KeyError) as e:
                # Metadata error - use pyarrow fallback
                if not use_pyarrow_fallback:
                    print(f"[INFO] ⚠️  load_dataset 失敗，使用 pyarrow 直接讀取...")
                    use_pyarrow_fallback = True
                
                import pyarrow.parquet as pq
                table = pq.read_table(str(pf))
                
                # Convert to list of dicts
                rows = []
                for i in range(len(table)):
                    row = {col: table[col][i].as_py() for col in table.column_names}
                    rows.append(row)
                
                # Create a simple Dataset-like object
                class SimpleDataset:
                    def __init__(self, data):
                        self.data = data
                    def __len__(self):
                        return len(self.data)
                    def __getitem__(self, idx):
                        return self.data[idx]
                
                datasets_list.append(SimpleDataset(rows))
        
        # Concatenate datasets
        if use_pyarrow_fallback:
            # Simple concatenation for list-based datasets
            all_rows = []
            for ds in datasets_list:
                if hasattr(ds, 'data'):
                    all_rows.extend(ds.data)
                else:
                    all_rows.extend([ds[i] for i in range(len(ds))])
            
            class SimpleDataset:
                def __init__(self, data):
                    self.data = data
                def __len__(self):
                    return len(self.data)
                def __getitem__(self, idx):
                    return self.data[idx]
            
            full_dataset = SimpleDataset(all_rows)
        else:
            full_dataset = concatenate_datasets(datasets_list)
        
        # Split train/val (90/10)
        total_len = len(full_dataset)
        idx_90 = int(total_len * 0.9)
        
        if hasattr(full_dataset, 'select'):
            # HuggingFace Dataset
            if split == "train":
                self.dataset = full_dataset.select(range(0, idx_90))
            else:
                self.dataset = full_dataset.select(range(idx_90, total_len))
        else:
            # SimpleDataset (pyarrow fallback)
            if split == "train":
                selected_data = full_dataset.data[0:idx_90]
            else:
                selected_data = full_dataset.data[idx_90:total_len]
            
            class SimpleDataset:
                def __init__(self, data):
                    self.data = data
                def __len__(self):
                    return len(self.data)
                def __getitem__(self, idx):
                    return self.data[idx]
            
            self.dataset = SimpleDataset(selected_data)
        
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
        
        # Get full preview image (支持多种格式)
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
        elif isinstance(preview_value, dict):
            # 格式 3: HuggingFace Image feature format (dict with 'bytes' or 'path')
            if 'bytes' in preview_value and preview_value['bytes']:
                from io import BytesIO
                preview_img = Image.open(BytesIO(preview_value['bytes']))
            elif 'path' in preview_value:
                preview_path = preview_value['path']
                fixed_path = self._fix_image_path(preview_path)
                preview_img = Image.open(fixed_path)
            else:
                raise ValueError(f"preview dict 格式不支持，keys: {preview_value.keys()}")
        elif isinstance(preview_value, bytes):
            # 格式 4: 直接的 bytes
            from io import BytesIO
            preview_img = Image.open(BytesIO(preview_value))
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
        
        # If preview is Image object (TAData), use index-based matching
        if isinstance(preview, Image.Image) and self.caption_mapping_indexed:
            if idx in self.caption_mapping_indexed:
                caption = self.caption_mapping_indexed[idx]
                caption_found = True
                if show_debug:
                    print(f"[CAPTION] From index {idx}: {caption[:100]}...")
        
        # Otherwise use path-based matching
        if not caption_found and self.caption_mapping and preview_path:
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
                else:
                    # Try converting path prefix for cross-machine compatibility
                    # /tmp2/b12902041/Gino/preprocessed_data -> /workspace/dataset/preprocessed_data
                    # /workspace/dataset/preprocessed_data -> /tmp2/b12902041/Gino/preprocessed_data
                    alt_paths = []
                    if '/tmp2/b12902041/Gino/' in preview_path:
                        alt_paths.append(preview_path.replace('/tmp2/b12902041/Gino/', '/workspace/dataset/'))
                    if '/workspace/dataset/' in preview_path:
                        alt_paths.append(preview_path.replace('/workspace/dataset/', '/tmp2/b12902041/Gino/'))
                    
                    for alt_path in alt_paths:
                        if alt_path in self.caption_mapping:
                            caption = self.caption_mapping[alt_path]
                            caption_found = True
                            break
            
            if caption_found:
                # Debug: print for first few samples
                if not hasattr(self, '_caption_debug_count'):
                    self._caption_debug_count = 0
                if self.enable_debug and self._caption_debug_count < 3:
                    print(f"[DEBUG] Sample {idx}: Using caption mapping")
                    print(f"  Path: {preview_path}")
                    print(f"  Caption: {caption[:80]}...")
                    self._caption_debug_count += 1
        
        if not caption_found:
            caption = item.get("title", "A design image")
            # Debug
            if not hasattr(self, '_caption_debug_count'):
                self._caption_debug_count = 0
            if self.enable_debug and self._caption_debug_count < 3:
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
        show_debug = self.enable_debug and self._layer_debug_count < 3
        if show_debug:
            print(f"[DEBUG] Sample {idx}: Total layers in dataset: {layer_count}")
            if layer_types:
                print(f"  Layer types: {layer_types}")
        
        for i in range(layer_count):
            # Skip background layers (already added as base)
            if layer_types and i < len(layer_types) and layer_types[i] == 3:
                if show_debug:
                    print(f"  [SKIP] Layer {i}: ColoredBackground (type=3)")
                continue  # Skip ColoredBackground
            
            # Get layer image
            layer_img = layer_images[i]
            
            # If layer_img is None, crop from preview image using bbox
            if layer_img is None:
                if show_debug:
                    print(f"  [CROP] Layer {i}: No separate image, will crop from preview")
                
                # Get bbox for this layer
                x = int(left_list[i])
                y = int(top_list[i])
                w = int(width_list[i])
                h = int(height_list[i])
                
                # Validate bbox
                if w <= 0 or h <= 0:
                    if show_debug:
                        print(f"  [SKIP] Layer {i}: Invalid bbox size (w={w}, h={h})")
                    continue
                
                # Crop from whole image
                try:
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h
                    
                    # Ensure bbox is within image bounds
                    x1 = max(0, min(x1, whole_img_RGBA.width))
                    y1 = max(0, min(y1, whole_img_RGBA.height))
                    x2 = max(0, min(x2, whole_img_RGBA.width))
                    y2 = max(0, min(y2, whole_img_RGBA.height))
                    
                    if x2 <= x1 or y2 <= y1:
                        if show_debug:
                            print(f"  [SKIP] Layer {i}: Invalid crop region ({x1},{y1},{x2},{y2})")
                        continue
                    
                    # Crop the region
                    layer_img = whole_img_RGBA.crop((x1, y1, x2, y2))
                    
                    if show_debug:
                        print(f"  [CROP] Layer {i}: Cropped {layer_img.size} from preview at bbox=({x},{y},{w},{h})")
                
                except Exception as e:
                    if show_debug:
                        print(f"  [SKIP] Layer {i}: Cannot crop from preview: {e}")
                    continue
            
            # 處理不同格式的 layer_img
            if isinstance(layer_img, str):
                # 格式 1: 字符串路徑
                original_path = layer_img
                layer_img = self._fix_image_path(layer_img)
                
                if show_debug:
                    print(f"  [PATH FIX] Layer {i}: {original_path}")
                    print(f"           -> {layer_img}")
                
                try:
                    layer_img = Image.open(layer_img)
                    if show_debug:
                        print(f"           ✓ Successfully loaded image: {layer_img.size}")
                except Exception as e:
                    if show_debug:
                        print(f"  [SKIP] Layer {i}: Cannot load image: {e}")
                    continue
            
            elif isinstance(layer_img, dict):
                # 格式 2: HuggingFace Image feature (dict with 'bytes')
                if 'bytes' in layer_img and layer_img['bytes']:
                    from io import BytesIO
                    layer_img = Image.open(BytesIO(layer_img['bytes']))
                    if show_debug:
                        print(f"  [CONVERT] Layer {i}: dict → PIL Image {layer_img.size}")
                elif 'path' in layer_img:
                    layer_path = self._fix_image_path(layer_img['path'])
                    layer_img = Image.open(layer_path)
                    if show_debug:
                        print(f"  [LOAD] Layer {i}: from path in dict {layer_img.size}")
                else:
                    if show_debug:
                        print(f"  [SKIP] Layer {i}: dict format not supported, keys: {layer_img.keys()}")
                    continue
            
            elif isinstance(layer_img, bytes):
                # 格式 3: 直接的 bytes
                from io import BytesIO
                layer_img = Image.open(BytesIO(layer_img))
                if show_debug:
                    print(f"  [CONVERT] Layer {i}: bytes → PIL Image {layer_img.size}")
            
            # 檢查是否成功轉換為 Image
            if not isinstance(layer_img, Image.Image):
                if show_debug:
                    print(f"  [SKIP] Layer {i}: Not an Image, type={type(layer_img)}")
                continue
            
            if show_debug:
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
        if show_debug:
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


