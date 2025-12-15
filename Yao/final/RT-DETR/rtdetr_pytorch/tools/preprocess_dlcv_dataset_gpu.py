
import os
import math
import random
import shutil
import json
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import io
import torch
import torchvision.transforms.functional as TF
from datasets import load_dataset

# --- CONFIG ---
SEED = 42
TRAIN_RATIO = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- GEOMETRY UTILS (CPU is fine for simple math) ---
def calculate_rotated_aabb(left, top, w, h, angle_deg):
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    cx = left + w / 2
    cy = top + h / 2
    corners = [
        (-w / 2, -h / 2),
        ( w / 2, -h / 2),
        ( w / 2,  h / 2),
        (-w / 2,  h / 2)
    ]
    rotated_xs = []
    rotated_ys = []
    for dx, dy in corners:
        nx = cx + (dx * cos_t - dy * sin_t)
        ny = cy + (dx * sin_t + dy * cos_t)
        rotated_xs.append(nx)
        rotated_ys.append(ny)
    min_x = min(rotated_xs)
    max_x = max(rotated_xs)
    min_y = min(rotated_ys)
    max_y = max(rotated_ys)
    return min_x, min_y, max_x, max_y

def get_tight_box_with_scale_pil(layer_asset, meta_left, meta_top, meta_w, meta_h):
    # This needs PIL image access to getbbox.
    # If we use GPU, we convert to tensor. 
    # But for BBox, we need alpha box.
    # Operations on small alpha mask on CPU are fast.
    # We can keep using PIL for this specific check if needed, OR implement on GPU?
    # Keeping PIL for getbbox is safer for legacy match.
    if layer_asset is None: return None
    alpha_bbox = layer_asset.getbbox()
    if alpha_bbox is None: return None
    a_left, a_top, a_right, a_bottom = alpha_bbox
    asset_w, asset_h = layer_asset.size
    scale_x = meta_w / asset_w
    scale_y = meta_h / asset_h
    final_x1 = meta_left + (a_left * scale_x)
    final_y1 = meta_top + (a_top * scale_y)
    final_x2 = meta_left + (a_right * scale_x)
    final_y2 = meta_top + (a_bottom * scale_y)
    return final_x1, final_y1, final_x2, final_y2

def start_processing(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if output_dir.exists(): shutil.rmtree(output_dir)
    
    (output_dir / "images" / "train").mkdir(parents=True)
    (output_dir / "images" / "val").mkdir(parents=True)
    (output_dir / "depths" / "train").mkdir(parents=True)
    (output_dir / "depths" / "val").mkdir(parents=True)
    (output_dir / "annotations").mkdir(parents=True)
    
    print(f"Loading Parquet from {input_dir}...")
    parquet_files = glob.glob(str(input_dir / "**/*.parquet"), recursive=True)
    ds = load_dataset("parquet", data_files=parquet_files, split="train")
    ds = ds.shuffle(seed=SEED)
    
    coco_data = {
        "train": {"images": [], "annotations": [], "categories": [{"id": 1, "name": "object"}]},
        "val": {"images": [], "annotations": [], "categories": [{"id": 1, "name": "object"}]}
    }
    ann_counters = {"train": 1, "val": 1}
    
    total = len(ds)
    print(f"Processing {total} samples on {DEVICE}...") # 2 CPUs but GPU available
    
    for idx, sample in enumerate(tqdm(ds)):
        split = "train" if idx < total * TRAIN_RATIO else "val"
        
        try:
            # 1. Image
            if 'preview' not in sample or sample['preview'] is None: continue
            preview = sample['preview']
            if isinstance(preview, dict) and 'bytes' in preview:
                 main_img_pil = Image.open(io.BytesIO(preview['bytes'])).convert("RGB")
            elif isinstance(preview, Image.Image):
                 main_img_pil = preview.convert("RGB")
            else: 
                 if isinstance(preview, bytes):
                     main_img_pil = Image.open(io.BytesIO(preview)).convert("RGB")
                 else:
                     main_img_pil = Image.fromarray(preview).convert("RGB")

            img_id = idx
            img_w, img_h = main_img_pil.size
            canvas_w = float(sample.get('canvas_width', img_w))
            canvas_h = float(sample.get('canvas_height', img_h))
            
            # Save RGB (IO bound, unavoidable)
            img_filename = f"{img_id:08d}.png"
            main_img_pil.save(output_dir / "images" / split / img_filename, compress_level=1)
            
            # Setup Depth Tensor on GPU
            # Use float32 for accumulation
            depth_map_tensor = torch.zeros((img_h, img_w), dtype=torch.float32, device=DEVICE)
            
            # Pre-calculate scale
            scale_x = img_w / canvas_w if canvas_w > 0 else 1.0
            scale_y = img_h / canvas_h if canvas_h > 0 else 1.0
            
            l_imgs = sample.get('image', [])
            l_left = sample.get('left', [])
            l_top = sample.get('top', [])
            l_width = sample.get('width', [])
            l_height = sample.get('height', [])
            l_angle = sample.get('angle', [])
            num_layers = len(l_left)
            
            valid_anns = []
            
            for i in range(num_layers):
                meta_left = float(l_left[i]) * scale_x
                meta_top = float(l_top[i]) * scale_y
                meta_w = float(l_width[i]) * scale_x
                meta_h = float(l_height[i]) * scale_y
                meta_angle = float(l_angle[i]) if i < len(l_angle) else 0.0
                
                asset_data = l_imgs[i] 
                asset_pil = None
                
                # Load Asset (CPU)
                if isinstance(asset_data, dict) and 'bytes' in asset_data:
                    asset_pil = Image.open(io.BytesIO(asset_data['bytes']))
                elif isinstance(asset_data, Image.Image):
                    asset_pil = asset_data
                elif isinstance(asset_data, bytes):
                    asset_pil = Image.open(io.BytesIO(asset_data))
                
                # --- A. BBox (CPU) ---
                if meta_w * meta_h > (img_w * img_h * 0.95): pass 

                final_box = None
                if abs(meta_angle) > 1.0:
                    min_x, min_y, max_x, max_y = calculate_rotated_aabb(meta_left, meta_top, meta_w, meta_h, meta_angle)
                    final_box = (min_x, min_y, max_x, max_y)
                else:
                    if asset_pil is not None:
                        try:
                            # Use PIL for alpha check (fast on CPU for single asset)
                            final_box = get_tight_box_with_scale_pil(asset_pil, meta_left, meta_top, meta_w, meta_h)
                        except: pass
                    if final_box is None:
                        final_box = (meta_left, meta_top, meta_left + meta_w, meta_top + meta_h)
                
                if final_box:
                    x1, y1, x2, y2 = final_box
                    x1 = max(0, min(x1, img_w))
                    y1 = max(0, min(y1, img_h))
                    x2 = max(0, min(x2, img_w))
                    y2 = max(0, min(y2, img_h))
                    if (x2 - x1) > 2 and (y2 - y1) > 2:
                        norm_layer_idx = (i + 1) / num_layers if num_layers > 0 else 0
                        ann = {
                            "id": ann_counters[split],
                            "image_id": img_id,
                            "category_id": 1,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "area": (x2 - x1) * (y2 - y1),
                            "iscrowd": 0,
                            "layer_order": norm_layer_idx
                        }
                        valid_anns.append(ann)
                        ann_counters[split] += 1
                
                # --- B. Depth Map (GPU) ---
                if asset_pil is not None:
                    try:
                        # 1. Convert to Tensor (CPU -> GPU)
                        # TF.to_tensor scales to [0, 1]. We want to keep alpha.
                        # asset_pil might be RGBA or RGB.
                        asset_tensor = TF.to_tensor(asset_pil).to(DEVICE) # (C, H, W)
                        
                        # 2. Resize
                        target_h, target_w = int(meta_h), int(meta_w)
                        if target_h <= 0 or target_w <= 0: continue
                        
                        resized_tensor = TF.resize(asset_tensor, [target_h, target_w], antialias=True)
                        
                        # 3. Rotate
                        if abs(meta_angle) > 0.1:
                            # rotate receives (angle) in degrees.
                            # PIL rotate is counter-clockwise for positive? 
                            # TF.rotate is same logic.
                            # Chang's script used -meta_angle for PIL rotate.
                            # So we use -meta_angle here too.
                            rotated_tensor = TF.rotate(resized_tensor, -meta_angle, expand=True)
                        else:
                            rotated_tensor = resized_tensor
                            
                        # 4. Paste Logic (Masking)
                        # We need to place rotated_tensor into depth_map_tensor
                        cx = meta_left + meta_w / 2
                        cy = meta_top + meta_h / 2
                        
                        _, rh, rw = rotated_tensor.shape
                        paste_x = int(cx - rw / 2)
                        paste_y = int(cy - rh / 2)
                        
                        # Clip coords
                        # We need to slice both source and destination to handle out-of-bounds
                        sy_start, sx_start = 0, 0
                        sy_end, sx_end = rh, rw
                        
                        dy_start, dx_start = paste_y, paste_x
                        dy_end, dx_end = paste_y + rh, paste_x + rw
                        
                        # Clip Top/Left
                        if dy_start < 0:
                            sy_start -= dy_start
                            dy_start = 0
                        if dx_start < 0:
                            sx_start -= dx_start
                            dx_start = 0
                            
                        # Clip Bottom/Right
                        if dy_end > img_h:
                            diff = dy_end - img_h
                            dy_end = img_h
                            sy_end -= diff
                        if dx_end > img_w:
                            diff = dx_end - img_w
                            dx_end = img_w
                            sx_end -= diff
                        
                        # Check valid region
                        if dy_end > dy_start and dx_end > dx_start:
                             # Extract alpha mask
                             # If RGBA (4 channels)
                             if rotated_tensor.shape[0] == 4:
                                 alpha = rotated_tensor[3, sy_start:sy_end, sx_start:sx_end]
                                 mask = alpha > 0.1 # Threshold
                             else:
                                 # Opaque
                                 mask = torch.ones((dy_end-dy_start, dx_end-dx_start), device=DEVICE, dtype=torch.bool)
                             
                             norm_val = (i + 1) / num_layers if num_layers > 0 else 0
                             
                             # Update Depth Map (In-place)
                             # We operate on a slice
                             target_slice = depth_map_tensor[dy_start:dy_end, dx_start:dx_end]
                             target_slice[mask] = norm_val
                             
                    except Exception as e:
                        pass
                        
            # Save Depth Map (GPU -> CPU -> PNG)
            depth_map_cpu = (depth_map_tensor * 255.0).byte().cpu() # (H, W)
            # numpy conversion for PIL
            depth_np = depth_map_cpu.numpy()
            depth_pil = Image.fromarray(depth_np, mode='L')
            
            depth_filename = f"{img_id:08d}.png"
            depth_pil.save(output_dir / "depths" / split / img_filename, compress_level=1)
            
            coco_data[split]["images"].append({
                "id": img_id,
                "file_name": img_filename, 
                "width": img_w,
                "height": img_h
            })
            coco_data[split]["annotations"].extend(valid_anns)
        
        except Exception as e:
            print(f"Error {idx}: {e}")
            continue

    for split in ["train", "val"]:
        with open(output_dir / "annotations" / f"{split}.json", "w") as f:
            json.dump(coco_data[split], f)
    
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    start_processing(args.input_dir, args.output_dir)
