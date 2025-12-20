import numpy as np
import torch
import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
import torchvision.transforms as T
from collections import defaultdict

def collate_fn(batch):
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

class LayoutTrainDataset(Dataset):
    def __init__(self, data_dir, split="train", enable_debug=False):
        """
        Dataset loader for PrismLayersPro.

        Supported `data_dir` formats:
        1) HuggingFace datasets cache directory (original behavior):
           - We will call `load_dataset("artplus/PrismLayersPro", cache_dir=data_dir)`
        2) Local parquet shards directory (Method A):
           - Expect files under:  {data_dir}/data/*.parquet
           - Each file is typically a style shard, e.g. "anime-00000-of-00007.parquet".
           - We will load each parquet file and concatenate them into one dataset.
        
        Args:
            data_dir: Directory containing parquet files or HF cache
            split: "train", "val", or "test"
            enable_debug: If True, print debug info for first few samples
        """
        self.data_dir = data_dir
        self.enable_debug = enable_debug
        
        local_parquet_dir = os.path.join(data_dir, "data")
        local_parquet_files = sorted(glob.glob(os.path.join(local_parquet_dir, "*.parquet")))

        if len(local_parquet_files) > 0:
            # Load local parquet shards directly (no internet needed).
            datasets_list = []
            for pf in local_parquet_files:
                style = os.path.basename(pf).split("-000")[0]
                ds = load_dataset("parquet", data_files=pf)["train"]

                # Some local exports may not contain `style_category`; add it from filename.
                if "style_category" not in ds.column_names:
                    ds = ds.add_column("style_category", [style] * len(ds))

                datasets_list.append(ds)

            full_dataset = concatenate_datasets(datasets_list)
        else:
            # Fallback to HuggingFace dataset (may download if cache missing).
            full_dataset = load_dataset(
                "artplus/PrismLayersPro",
                cache_dir=data_dir,
            )
            full_dataset = concatenate_datasets(list(full_dataset.values()))

        if "style_category" not in full_dataset.column_names:
            raise ValueError("Dataset must contain a 'style_category' field to split by class.")

        categories = np.array(full_dataset["style_category"])
        category_to_indices = defaultdict(list)
        for i, cat in enumerate(categories):
            category_to_indices[cat].append(i)

        subsets = []
        for cat, indices in category_to_indices.items():
            total_len = len(indices)
            idx_90 = int(total_len * 0.9)
            idx_95 = int(total_len * 0.95)

            if split == "train":
                selected_idx = indices[:idx_90]
            elif split == "test":
                selected_idx = indices[idx_90:idx_95]
            elif split == "val":
                selected_idx = indices[idx_95:]
            else:
                raise ValueError("split must be 'train', 'val', or 'test'")

            subsets.append(full_dataset.select(selected_idx))

        self.dataset = concatenate_datasets(subsets)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.dataset)
    
    def _fix_image_path(self, image_or_path):
        """
        Fix hardcoded image paths in parquet files.
        Same logic as DLCVLayoutDataset.
        """
        if not isinstance(image_or_path, str):
            return image_or_path
        
        if "/preprocessed_data/" in image_or_path:
            parts = image_or_path.split("/preprocessed_data/")
            if len(parts) == 2:
                relative_path = parts[1]
                new_path = os.path.join(self.data_dir, relative_path)
                
                if os.path.exists(new_path):
                    return new_path
                else:
                    if relative_path.startswith("images/"):
                        alt_path = os.path.join(self.data_dir, relative_path.replace("images/", "", 1))
                        if os.path.exists(alt_path):
                            return alt_path
        
        return image_or_path

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Initialize debug counter
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        
        show_debug = self.enable_debug and self._debug_count < 5

        def rgba2rgb(img_RGBA):
            img_RGB = Image.new("RGB", img_RGBA.size, (128, 128, 128))
            img_RGB.paste(img_RGBA, mask=img_RGBA.split()[3])
            return img_RGB

        def get_img(x, img_name="image"):
            original_x = x
            is_path = isinstance(x, str)
            
            if is_path:
                # Fix path if needed
                x = self._fix_image_path(x)
                if show_debug and x != original_x:
                    print(f"    [PATH FIX] {img_name}: {original_x}")
                    print(f"             -> {x}")
                
                try:
                    img_RGBA = Image.open(x).convert("RGBA")
                    img_RGB = rgba2rgb(img_RGBA)
                    if show_debug:
                        print(f"    [LOADED] {img_name}: path -> Image {img_RGBA.size}")
                except Exception as e:
                    if show_debug:
                        print(f"    [ERROR] {img_name}: Cannot load from {x}: {e}")
                    raise
            else:
                img_RGBA = x.convert("RGBA")
                img_RGB = rgba2rgb(img_RGBA)
                if show_debug:
                    print(f"    [LOADED] {img_name}: Image object {img_RGBA.size}")
            
            return img_RGBA, img_RGB

        if show_debug:
            print(f"\n[TRAIN DEBUG] Sample {idx}:")
            print(f"  Dataset: PrismLayersPro format")
            print(f"  Layer count: {item['layer_count']}")
        
        whole_img_RGBA, whole_img_RGB = get_img(item["whole_image"], "whole_image")
        whole_cap = item["whole_caption"]
        W, H = whole_img_RGBA.size
        base_layout = [0, 0, W - 1, H - 1]

        layer_image_RGBA = [self.to_tensor(whole_img_RGBA)]
        layer_image_RGB  = [self.to_tensor(whole_img_RGB)]
        layout = [base_layout]

        base_img_RGBA, base_img_RGB = get_img(item["base_image"], "base_image")
        layer_image_RGBA.append(self.to_tensor(base_img_RGBA))
        layer_image_RGB.append(self.to_tensor(base_img_RGB))
        layout.append(base_layout)

        layer_count = item["layer_count"]
        loaded_layers = 0
        skipped_layers = 0
        
        for i in range(layer_count):
            key = f"layer_{i:02d}"
            try:
                img_RGBA, img_RGB = get_img(item[key], key)
                
                w0, h0, w1, h1 = item[f"{key}_box"]

                canvas_RGBA = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                canvas_RGB = Image.new("RGB", (W, H), (128, 128, 128))

                W_img, H_img = w1 - w0, h1 - h0
                if img_RGBA.size != (W_img, H_img):
                    img_RGBA = img_RGBA.resize((W_img, H_img), Image.BILINEAR)
                    img_RGB  = img_RGB.resize((W_img, H_img), Image.BILINEAR)

                canvas_RGBA.paste(img_RGBA, (w0, h0), img_RGBA)
                canvas_RGB.paste(img_RGB, (w0, h0))

                layer_image_RGBA.append(self.to_tensor(canvas_RGBA))
                layer_image_RGB.append(self.to_tensor(canvas_RGB))
                layout.append([w0, h0, w1, h1])
                loaded_layers += 1
                
                if show_debug:
                    print(f"    [ADDED] {key}: bbox=({w0}, {h0}, {w1}, {h1})")
            except Exception as e:
                skipped_layers += 1
                if show_debug:
                    print(f"    [SKIP] {key}: {e}")
        
        if show_debug:
            total_layers = len(layout)
            print(f"  [RESULT] Total layers: {total_layers} (2 base + {loaded_layers} foreground)")
            if skipped_layers > 0:
                print(f"  [WARNING] Skipped {skipped_layers} layers due to errors")
            self._debug_count += 1

        return {
            "pixel_RGBA": layer_image_RGBA,
            "pixel_RGB": layer_image_RGB,
            "whole_img": whole_img_RGB,
            "caption": whole_cap,
            "height": H,
            "width": W,
            "layout": layout,
        }