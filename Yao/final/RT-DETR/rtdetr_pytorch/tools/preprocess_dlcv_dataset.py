
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
from datasets import load_dataset, concatenate_datasets

# --- CONFIG ---
SEED = 42
TRAIN_RATIO = 0.9
PADDING_RATIO = 0.05 

# --- GEOMETRY UTILS (From Chang/prepare_dlcv_training_data.py) ---
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

def get_tight_box_with_scale(layer_asset, meta_left, meta_top, meta_w, meta_h):
    if layer_asset is None: return None
    asset_w, asset_h = layer_asset.size
    if asset_w == 0 or asset_h == 0: return None
    alpha_bbox = layer_asset.getbbox() # Tight box of non-zero alpha
    if alpha_bbox is None: return None
    a_left, a_top, a_right, a_bottom = alpha_bbox
    scale_x = meta_w / asset_w
    scale_y = meta_h / asset_h
    final_x1 = meta_left + (a_left * scale_x)
    final_y1 = meta_top + (a_top * scale_y)
    final_x2 = meta_left + (a_right * scale_x)
    final_y2 = meta_top + (a_bottom * scale_y)
    return final_x1, final_y1, final_x2, final_y2

# --- DEPTH MAP UTILS (Adapted from ml-depth-pro) ---
def create_layer_index_map(l_imgs, l_left, l_top, l_width, l_height, canvas_w, canvas_h):
    layer_index_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    num_layers = len(l_imgs)
    if num_layers == 0: return layer_index_map
    
    # Iterate layers from background (0) to foreground
    for i, (img, left, top, w, h) in enumerate(zip(l_imgs, l_left, l_top, l_width, l_height)):
        normalized_index = (i + 1) / num_layers
        pass # Placeholder for loop structure
    return layer_index_map

def start_processing(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if output_dir.exists(): shutil.rmtree(output_dir)
    
    # Setup Dirs
    (output_dir / "images" / "train").mkdir(parents=True)
    (output_dir / "images" / "val").mkdir(parents=True)
    (output_dir / "depths" / "train").mkdir(parents=True)
    (output_dir / "depths" / "val").mkdir(parents=True)
    (output_dir / "annotations").mkdir(parents=True)
    
    # Load Parquet
    print(f"Loading Parquet from {input_dir}...")
    parquet_files = glob.glob(str(input_dir / "**/*.parquet"), recursive=True)
    if not parquet_files:
        raise ValueError("No parquet files found")
    
    # Load dataset
    ds = load_dataset("parquet", data_files=parquet_files, split="train") # 'train' is usually the default split name in parquet files
    ds = ds.shuffle(seed=SEED)
    
    # Init COCO buffers
    coco_data = {
        "train": {"images": [], "annotations": [], "categories": [{"id": 1, "name": "object"}]},
        "val": {"images": [], "annotations": [], "categories": [{"id": 1, "name": "object"}]}
    }
    ann_counters = {"train": 1, "val": 1}
    
    # Process
    total = len(ds)
    print(f"Processing {total} samples...")
    
    for idx, sample in enumerate(tqdm(ds)):
        # Split
        split = "train" if idx < total * TRAIN_RATIO else "val"
        
        try:
            # 1. Image
            if 'preview' not in sample or sample['preview'] is None: continue
            # Preview might be bytes or object
            preview = sample['preview']
            if isinstance(preview, dict) and 'bytes' in preview: # Huggingface Image feature sometimes
                 main_img = Image.open(io.BytesIO(preview['bytes'])).convert("RGB")
            elif isinstance(preview, Image.Image):
                 main_img = preview.convert("RGB")
            else: # Bytes or ndarray
                 # If bytes
                 import io
                 if isinstance(preview, bytes):
                     main_img = Image.open(io.BytesIO(preview)).convert("RGB")
                 else:
                     main_img = Image.fromarray(preview).convert("RGB")

            img_id = idx
            img_w, img_h = main_img.size
            canvas_w = float(sample.get('canvas_width', img_w))
            canvas_h = float(sample.get('canvas_height', img_h))
            
            # --- FIX: Calculate Scale Factor (Preview / Canvas) ---
            # Use X scale for horizontal, Y for vertical to be safe
            scale_x = img_w / canvas_w if canvas_w > 0 else 1.0
            scale_y = img_h / canvas_h if canvas_h > 0 else 1.0

            # Save Image
            img_filename = f"{img_id:08d}.png"
            main_img.save(output_dir / "images" / split / img_filename)
            
            # 2. Extract Layers Info
            l_imgs = sample.get('image', [])
            l_left = sample.get('left', [])
            l_top = sample.get('top', [])
            l_width = sample.get('width', [])
            l_height = sample.get('height', [])
            l_angle = sample.get('angle', [])
            
            # Prepare Depth Map Buffer (Use IMAGE size)
            depth_map = np.zeros((img_h, img_w), dtype=np.float32)
            num_layers = len(l_left)
            
            # 3. Process Layers (BBox + Depth)
            valid_anns = []
            
            for i in range(len(l_left)):
                # --- FIX: Adjust metadata to Image Space ---
                meta_left = float(l_left[i]) * scale_x
                meta_top = float(l_top[i]) * scale_y
                meta_w = float(l_width[i]) * scale_x
                meta_h = float(l_height[i]) * scale_y
                meta_angle = float(l_angle[i]) if i < len(l_angle) else 0.0
                
                asset_img = l_imgs[i] 
                if isinstance(asset_img, dict) and 'bytes' in asset_img:
                    asset_img = Image.open(io.BytesIO(asset_img['bytes']))
                
                # --- A. BBox Calculation (Bifurcation Logic) ---
                # NOTE: We work in Image Space now.
                # Threshold check also in Image Space.
                if meta_w * meta_h > (img_w * img_h * 0.95): 
                    pass 

                final_box = None
                if abs(meta_angle) > 1.0:
                    min_x, min_y, max_x, max_y = calculate_rotated_aabb(meta_left, meta_top, meta_w, meta_h, meta_angle)
                    final_box = (min_x, min_y, max_x, max_y)
                else:
                    if asset_img is not None:
                        try:
                            final_box = get_tight_box_with_scale(asset_img, meta_left, meta_top, meta_w, meta_h)
                        except: pass
                    if final_box is None:
                        final_box = (meta_left, meta_top, meta_left + meta_w, meta_top + meta_h)
                
                if final_box:
                    x1, y1, x2, y2 = final_box
                    # Clip
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
                
                # --- B. Depth Map (Masking) ---
                if asset_img is not None:
                    try:
                        # 1. Resize Asset to Scaled Meta Size
                        target_size = (int(meta_w), int(meta_h))
                        if target_size[0] > 0 and target_size[1] > 0:
                            resized_asset = asset_img.resize(target_size, Image.BILINEAR)
                            
                            # 2. Rotate
                            if abs(meta_angle) > 0.1:
                                rotated_asset = resized_asset.rotate(-meta_angle, expand=True, resample=Image.BICUBIC)
                            else:
                                rotated_asset = resized_asset
                                
                            # 3. Paste Position (Calculated in Image Space)
                            cx = meta_left + meta_w / 2
                            cy = meta_top + meta_h / 2
                            
                            rw, rh = rotated_asset.size
                            paste_x = int(cx - rw / 2)
                            paste_y = int(cy - rh / 2)
                            
                            # 4. Create Mask
                            layer_mask = Image.new('L', (img_w, img_h), 0)
                            if rotated_asset.mode == 'RGBA':
                                alpha = rotated_asset.split()[3]
                                layer_mask.paste(alpha, (paste_x, paste_y))
                            else:
                                layer_mask.paste(255, (paste_x, paste_y, paste_x+rw, paste_y+rh))
                                
                            mask_np = np.array(layer_mask) > 20 # Threshold
                            
                            # 5. Update Depth Map
                            norm_val = (i + 1) / num_layers if num_layers > 0 else 0
                            depth_map[mask_np] = norm_val
                    except Exception as e:
                        # print(f"Depth Error: {e}")
                        pass
            
            # Save Depth Map
            depth_img = Image.fromarray((depth_map * 255).astype(np.uint8), mode='L')
            depth_filename = f"{img_id:08d}.png" 
            depth_img.save(output_dir / "depths" / split / img_filename)
            
            # Add to COCO
            coco_data[split]["images"].append({
                "id": img_id,
                "file_name": img_filename, 
                "width": img_w,
                "height": img_h
            })
            coco_data[split]["annotations"].extend(valid_anns)
            
        except Exception as e:
            print(f"Error processing {idx}: {e}")
            continue

    # Save JSONs
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
