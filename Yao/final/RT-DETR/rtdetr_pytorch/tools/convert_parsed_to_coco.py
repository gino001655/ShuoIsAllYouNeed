
import json
import os
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

def convert_to_coco(data_dir: str, output_dir: str):
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Collect Samples (Replicating LayerOrderDataset logic)
    print("Collecting samples...")
    all_samples = []
    for category_dir in data_path.iterdir():
        if not category_dir.is_dir() or category_dir.name == "__pycache__":
            continue
        
        for sample_dir in category_dir.iterdir():
            if not sample_dir.is_dir():
                continue
            
            metadata_path = sample_dir / "metadata.json"
            whole_image_path = sample_dir / "00_whole_image.png"
            
            if metadata_path.exists() and whole_image_path.exists():
                all_samples.append({
                    'sample_dir': sample_dir,
                    'category': category_dir.name,
                    'sample_id': sample_dir.name,
                    'file_name': f"{category_dir.name}/{sample_dir.name}/00_whole_image.png",
                    'metadata_path': metadata_path
                })
                
    category_to_samples = defaultdict(list)
    for sample in all_samples:
        category_to_samples[sample['category']].append(sample)
        
    # 2. Split (Replicating Logic)
    train_samples = []
    val_samples = []
    test_samples = []
    
    train_ratio = 0.9
    val_ratio = 0.05
    # test_ratio = 0.05
    
    print(f"Splitting {len(all_samples)} samples...")
    for category, samples in category_to_samples.items():
        random.Random(42).shuffle(samples)
        total_len = len(samples)
        if total_len == 0: continue
        
        idx_train = int(total_len * train_ratio)
        idx_val = int(total_len * (train_ratio + val_ratio))
        
        train_samples.extend(samples[:idx_train])
        val_samples.extend(samples[idx_train:idx_val])
        test_samples.extend(samples[idx_val:])
        
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # 3. Create COCO JSONs
    categories = [{"id": 1, "name": "object"}]
    
    def create_json(samples, split_name):
        images = []
        annotations = []
        ann_id_counter = 1
        
        for img_id, sample in enumerate(samples, 1):
            # Read image size
            # We can peek metadata or open image. Metadata often lacks image size.
            # LayerOrderDataset opens image. We should try to read metadata "width_dst"? 
            # metadata layer has 'width_dst' but that's for layer.
            # Let's open one image per sample? It's slow.
            # Metadata has "layers" -> "box". Box is within canvas.
            # Usually strict parsing requires opening. But we can assume 1536?
            # User codebase said "image_size=1536" for model input, but "original_size" comes from image.
            # Let's open the image.
            
            # Optimization: Try to infer from first layer box or just open '00_whole_image.png'
            # To be safe and correct for COCO, we should read w,h.
            try:
                from PIL import Image
                with Image.open(sample['sample_dir'] / "00_whole_image.png") as img:
                    w, h = img.size
            except Exception:
                print(f"Warning: Could not open {sample['sample_dir']}")
                w, h = 1024, 1024 # Fallback
            
            images.append({
                "id": img_id,
                "file_name": sample['file_name'],
                "width": w,
                "height": h
            })
            
            # Read Annotations
            with open(sample['metadata_path'], 'r') as f:
                meta = json.load(f)
            
            layers = meta.get('layers', [])
            for layer in layers:
                # box: [x1, y1, x2, y2] (from my cat output earlier)
                box = layer.get('box', [])
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    # COCO bbox is [x, y, w, h]
                    coco_box = [x1, y1, x2 - x1, y2 - y1]
                    
                    # Layer Order
                    l_idx = layer.get('layer_index', 0)
                    # Normalize? 
                    # User asked: "is it normalize 0-1 or normalized over >1?"
                    # User's previous dataset code: `(i+1)/num_layers`.
                    # Here we store raw index or normalized?
                    # RT-DETR prediction is regression.
                    # Standard practice: Store RAW in annotation, Normalize in Dataset.
                    # Or Store Normalized here.
                    # Let's Store RAW and let Dataset normalize derived from 'layer_count'?
                    # But finding 'layer_count' in dataset __getitem__ is hard if we just have target dict.
                    # So better to Store NORMALIZED here.
                    # User formula: (i + 1) / num_layers
                    # or (layer_index - median) / std? User asked about this in previous turn.
                    # My previous summary said: "ground truth d is normalized ... in dataset".
                    # Let's recreate the logic from LayerOrderDataset.
                    # "if num_layers > 0: normalized_index = (i + 1) / num_layers"
                    
                    num_layers_total = len(layers) # or meta['layer_count']
                    if num_layers_total > 0:
                        norm_l = (l_idx + 1) / num_layers_total
                    else:
                        norm_l = 0.0
                        
                    annotations.append({
                        "id": ann_id_counter,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": coco_box,
                        "area": coco_box[2] * coco_box[3],
                        "iscrowd": 0,
                        "layer_order": norm_l # Custom field
                    })
                    ann_id_counter += 1
                    
        coco_output = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        
        out_file = output_path / f"{split_name}.json"
        with open(out_file, 'w') as f:
            json.dump(coco_output, f)
        print(f"Saved {out_file}")

    create_json(train_samples, "train")
    create_json(val_samples, "val")
    # Test set might not have annotations or we don't need it for training
    # create_json(test_samples, "test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    convert_to_coco(args.data_dir, args.output_dir)
