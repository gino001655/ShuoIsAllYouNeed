"""
Batch inference script for RT-DETR.
Reads all images in a directory, runs inference, and saves results in CLD-compatible Parquet format.
"""
import torch
import torch.nn as nn 
import torchvision.transforms as T
import numpy as np 
from PIL import Image, ImageDraw
import os 
import sys 
import glob
import pandas as pd
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig 

def load_rgbd_image(rgb_path, depth_path=None, use_zero_depth=False):
    """
    Load RGB image and depth map.
    """
    try:
        rgb_pil = Image.open(rgb_path).convert('RGB')
    except Exception as e:
        print(f"Error loading {rgb_path}: {e}")
        return None, None, 0, 0

    w, h = rgb_pil.size
    
    if use_zero_depth:
        depth_pil = Image.new('L', (w, h), 0)
    elif depth_path and os.path.exists(depth_path):
        depth_pil = Image.open(depth_path).convert('L')
        if depth_pil.size != (w, h):
            depth_pil = depth_pil.resize((w, h), Image.NEAREST)
    else:
        # Default to zero depth if not provided or found
        depth_pil = Image.new('L', (w, h), 0)
    
    return rgb_pil, depth_pil, w, h

def prepare_input(rgb_pil, depth_pil, target_size=640):
    rgb_resized = rgb_pil.resize((target_size, target_size), Image.BILINEAR)
    depth_resized = depth_pil.resize((target_size, target_size), Image.NEAREST)
    
    rgb_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    rgb_tensor = rgb_transform(rgb_resized)
    depth_array = np.array(depth_resized).astype(np.float32) / 255.0
    depth_tensor = torch.from_numpy(depth_array).unsqueeze(0)
    
    rgbd_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)
    return rgbd_tensor

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def custom_layer_nms(boxes, scores, labels, layers, iou_threshold=0.9, layer_diff_threshold=0.1):
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    indices = np.argsort(scores)[::-1]
    keep_indices = []
    
    while len(indices) > 0:
        current_idx = indices[0]
        keep_indices.append(current_idx)
        
        if len(indices) == 1:
            break
            
        remaining_indices = indices[1:]
        current_box = boxes[current_idx]
        current_layer = layers[current_idx]
        
        valid_mask = []
        for idx in remaining_indices:
            other_box = boxes[idx]
            other_layer = layers[idx]
            
            iou = calculate_iou(current_box, other_box)
            layer_diff = abs(current_layer - other_layer)
            
            should_suppress = (iou > iou_threshold) and (layer_diff < layer_diff_threshold)
            valid_mask.append(not should_suppress)
            
        indices = remaining_indices[np.array(valid_mask, dtype=bool)]
    
    keep_indices = np.array(keep_indices)
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices], layers[keep_indices]

def main(args):
    # Load config and model
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    else:
        raise AttributeError('Must provide --resume checkpoint path')
        
    cfg.model.load_state_dict(state)
    
    class Model(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.eval()
        def forward(self, rgbd_images):
            with torch.no_grad():
                return self.model(rgbd_images)
                
    device = torch.device(args.device)
    model = Model(cfg.model).to(device)
    print(f"Model loaded on {device}")
    
    # Get images
    formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for fmt in formats:
        image_files.extend(glob.glob(os.path.join(args.input_dir, fmt)))
        # Case insensitive check
        image_files.extend(glob.glob(os.path.join(args.input_dir, fmt.upper())))
    
    if not image_files:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Found {len(image_files)} images")
    
    records = []
    
    for img_path in tqdm(image_files):
        rgb_pil, depth_pil, orig_w, orig_h = load_rgbd_image(img_path, use_zero_depth=args.use_zero_depth)
        if rgb_pil is None:
            continue
            
        rgbd_tensor = prepare_input(rgb_pil, depth_pil)
        rgbd_batch = rgbd_tensor.unsqueeze(0).to(device)
        
        outputs = model(rgbd_batch)
        
        pred_logits = outputs['pred_logits'][0]
        pred_boxes = outputs['pred_boxes'][0]
        pred_layers = outputs['pred_layers'][0].squeeze(-1) if 'pred_layers' in outputs else torch.zeros(300)
        
        scores = pred_logits.sigmoid().max(dim=-1)[0]
        labels = pred_logits.sigmoid().argmax(dim=-1)
        
        keep_mask = scores > args.threshold
        
        boxes_np = pred_boxes[keep_mask].cpu().numpy()
        scores_np = scores[keep_mask].cpu().numpy()
        labels_np = labels[keep_mask].cpu().numpy()
        layers_np = pred_layers[keep_mask].cpu().numpy()
        
        # Convert to xyxy pixel
        boxes_xyxy = []
        for box in boxes_np:
            cx, cy, w, h = box
            x1 = (cx - w/2) * orig_w
            y1 = (cy - h/2) * orig_h
            x2 = (cx + w/2) * orig_w
            y2 = (cy + h/2) * orig_h
            boxes_xyxy.append([x1, y1, x2, y2])
        boxes_xyxy = np.array(boxes_xyxy)
        
        # NMS
        boxes_xyxy, scores_np, labels_np, layers_np = custom_layer_nms(
            boxes_xyxy, scores_np, labels_np, layers_np
        )
        
        # Expand Boxes
        if args.expand_ratio != 1.0 and len(boxes_xyxy) > 0:
            expanded_boxes = []
            for box in boxes_xyxy:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2
                
                new_w = w * args.expand_ratio
                new_h = h * args.expand_ratio
                
                new_x1 = max(0, cx - new_w / 2)
                new_y1 = max(0, cy - new_h / 2)
                new_x2 = min(orig_w, cx + new_w / 2)
                new_y2 = min(orig_h, cy + new_h / 2)
                
                expanded_boxes.append([new_x1, new_y1, new_x2, new_y2])
            boxes_xyxy = np.array(expanded_boxes)

        # Sort by layer order (background first -> foreground last)
        # CLD expects layers to be ordered or not?
        # The COCO converter sorts by layer_order.
        # Here we have predicted layers. Let's sort them ascending (0.0=bg, 1.0=fg)
        if len(boxes_xyxy) > 0:
            sort_idx = np.argsort(layers_np)
            boxes_xyxy = boxes_xyxy[sort_idx]
            layers_np = layers_np[sort_idx]
        
        # Prepare record for Parquet
        left_list = []
        top_list = []
        width_list = []
        height_list = []
        
        for box in boxes_xyxy:
            x1, y1, x2, y2 = box
            w_box = x2 - x1
            h_box = y2 - y1
            
            left_list.append(float(x1))
            top_list.append(float(y1))
            width_list.append(float(w_box))
            height_list.append(float(h_box))
            
        record = {
            "preview": os.path.abspath(img_path),
            "title": "A design image",
            "left": left_list,
            "top": top_list,
            "width": width_list,
            "height": height_list,
            "length": len(left_list),
            "image": [None] * len(left_list),
            # Add canvas size
            "canvas_width": orig_w,
            "canvas_height": orig_h
        }
        records.append(record)
        
    # Save to Parquet
    if not records:
        print("No records to save.")
        return

    output_dir = os.path.join(args.output_dir, "snapshots", "snapshot_1", "data")
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)
    
    # Save Parquet
    df = pd.DataFrame(records)
    parquet_path = os.path.join(output_dir, "inference-00000-of-00001.parquet")
    df.to_parquet(parquet_path, index=False, engine='pyarrow')
    print(f"✅ Saved parquet results to {parquet_path}")

    # Save Visualizations
    if args.save_vis:
        print(f"Saving visualizations to {vis_dir}...")
        for img_path, record in tqdm(zip(image_files, records), total=len(records), desc="Visualizing"):
            try:
                # Load original image
                img = Image.open(img_path).convert('RGB')
                draw = ImageDraw.Draw(img)
                
                # Draw boxes
                # Layer order is typically from background to foreground.
                # box is [x, y, w, h] in the record
                for i in range(record['length']):
                    x, y = record['left'][i], record['top'][i]
                    w, h = record['width'][i], record['height'][i]
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    
                    # Random color or based on index
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    draw.text((x1, y1), f"L{i}", fill=color)
                
                # Save
                basename = os.path.basename(img_path)
                img.save(os.path.join(vis_dir, basename))
            except Exception as e:
                print(f"Error visualizing {img_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Config path')
    parser.add_argument('-r', '--resume', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for parquet')
    parser.add_argument('--use-zero-depth', action='store_true', help='Use zero depth')
    parser.add_argument('--expand-ratio', type=float, default=1.0, help='Expansion ratio')
    parser.add_argument('-t', '--threshold', type=float, default=0.3, help='Confidence threshold (default: 0.3)')
    parser.add_argument('--save-vis', action='store_true', help='Save visualized results (images with boxes)')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='Device')
    
    args = parser.parse_args()
    main(args)
