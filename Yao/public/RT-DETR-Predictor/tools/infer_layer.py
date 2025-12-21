"""
Inference script for 4-channel RT-DETR with layer prediction.
Supports RGB + Depth input and visualizes layer order predictions.
"""
import torch
import torch.nn as nn 
import torchvision.transforms as T
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.core import YAMLConfig 


def load_rgbd_image(rgb_path, depth_path=None):
    """
    Load RGB image and depth map (if provided).
    If depth_path is None, creates a zero depth map.
    
    Returns:
        rgb_pil: PIL Image (RGB)
        depth_pil: PIL Image (grayscale)
        rgbd_tensor: torch.Tensor [4, H, W]
    """
    # Load RGB
    rgb_pil = Image.open(rgb_path).convert('RGB')
    w, h = rgb_pil.size
    
    # Load or create depth
    if depth_path and os.path.exists(depth_path):
        depth_pil = Image.open(depth_path).convert('L')  # Grayscale
        if depth_pil.size != (w, h):
            depth_pil = depth_pil.resize((w, h), Image.NEAREST)
    else:
        print(f"Warning: No depth map at {depth_path}, using zero depth")
        depth_pil = Image.new('L', (w, h), 0)
    
    return rgb_pil, depth_pil, w, h


def prepare_input(rgb_pil, depth_pil, target_size=640):
    """
    Prepare 4-channel input tensor.
    
    Args:
        rgb_pil: PIL Image (RGB)
        depth_pil: PIL Image (L)
        target_size: int, resize to this size
    
    Returns:
        rgbd_tensor: [4, H, W]
        original_size: (W, H)
    """
    # Resize
    rgb_resized = rgb_pil.resize((target_size, target_size), Image.BILINEAR)
    depth_resized = depth_pil.resize((target_size, target_size), Image.NEAREST)
    
    # Convert to tensors
    # Convert to tensors
    # RGB Normalization (ImageNet)
    rgb_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    rgb_tensor = rgb_transform(rgb_resized)  # [3, H, W] normalized
    depth_array = np.array(depth_resized).astype(np.float32) / 255.0
    depth_tensor = torch.from_numpy(depth_array).unsqueeze(0)  # [1, H, W]
    
    # Concatenate
    rgbd_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)  # [4, H, W]
    
    return rgbd_tensor


def visualize_predictions(rgb_pil, boxes, labels, scores, layers, 
                         threshold=0.3, save_path='result.jpg'):
    """
    Visualize predictions with boxes colored by layer values.
    
    Args:
        rgb_pil: PIL Image
        boxes: np.array [N, 4] in xyxy format
        labels: np.array [N]
        scores: np.array [N]
        layers: np.array [N] layer values in [0, 1]
        threshold: confidence threshold
        save_path: output path
    """
    # Filter by threshold
    valid_mask = scores > threshold
    boxes = boxes[valid_mask]
    labels = labels[valid_mask]
    scores = scores[valid_mask]
    layers = layers[valid_mask] if layers is not None else np.zeros(len(boxes))
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(rgb_pil)
    
    # Draw boxes
    for box, label, score, layer in zip(boxes, labels, scores, layers):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Color based on layer (using plasma colormap)
        color = plt.cm.plasma(layer)
        
        # Draw rectangle
        rect = patches.Rectangle((x1, y1), width, height, 
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add text
        text = f'C{int(label)} {score:.2f}\nL:{layer:.2f}'
        ax.text(x1, y1-5, text, color=color, fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    ax.axis('off')
    ax.set_title(f'RT-DETR Predictions (Color=Layer Order)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Result saved to {save_path}")



def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes (x1, y1, x2, y2).
    """
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
    """
    Custom NMS:
    If IoU > 0.9 AND abs(L1 - L2) < 0.1:
        Keep the one with higher confidence score.
    """
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    # Sort by score descending
    indices = np.argsort(scores)[::-1]
    keep_indices = []
    
    while len(indices) > 0:
        current_idx = indices[0]
        keep_indices.append(current_idx)
        
        if len(indices) == 1:
            break
            
        # Compare current box with all remaining boxes
        remaining_indices = indices[1:]
        current_box = boxes[current_idx]
        current_layer = layers[current_idx]
        
        valid_mask = []
        for idx in remaining_indices:
            other_box = boxes[idx]
            other_layer = layers[idx]
            
            iou = calculate_iou(current_box, other_box)
            layer_diff = abs(current_layer - other_layer)
            
            # Condition to SUPPRESS:
            # Overlap high AND Layer close -> Suppress (since current has higher score)
            should_suppress = (iou > iou_threshold) and (layer_diff < layer_diff_threshold)
            
            # Keep if NOT strictly suppressed
            valid_mask.append(not should_suppress)
            
        # Update indices to only those we kept
        indices = remaining_indices[np.array(valid_mask, dtype=bool)]
    
    keep_indices = np.array(keep_indices)
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices], layers[keep_indices]


def main(args):
    """Main inference function"""
    
    # Load config
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    # Load checkpoint
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu') 
        
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
            print("Using EMA weights")
        elif 'model' in checkpoint:
            state = checkpoint['model']
            print("Using model weights")
        else:
            state = checkpoint
            print("Using raw checkpoint")
    else:
        raise AttributeError('Must provide --resume checkpoint path')
    
    # Load weights into model
    cfg.model.load_state_dict(state)
    
    # Create inference model
    class Model(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.eval()
            
        def forward(self, rgbd_images):
            """
            Args:
                rgbd_images: [1, 4, H, W]
            
            Returns:
                outputs: dict with 'pred_logits', 'pred_boxes', 'pred_layers'
            """
            with torch.no_grad():
                outputs = self.model(rgbd_images)
            return outputs
    
    model = Model(cfg.model).to(args.device)
    print(f"Model loaded on {args.device}")
    
    # Load RGBD image
    print(f"Loading image: {args.rgb_file}")
    rgb_pil, depth_pil, orig_w, orig_h = load_rgbd_image(args.rgb_file, args.depth_file)
    
    if args.use_zero_depth:
        print("⚠️ Forcing zero depth input (ignoring loaded depth map)")
        depth_pil = Image.new('L', (orig_w, orig_h), 0)

    # Prepare input
    rgbd_tensor = prepare_input(rgb_pil, depth_pil, target_size=640)
    rgbd_batch = rgbd_tensor.unsqueeze(0).to(args.device)  # [1, 4, 640, 640]
    
    print("Running inference...")
    outputs = model(rgbd_batch)
    
    # Extract predictions
    pred_logits = outputs['pred_logits'][0]  # [300, num_classes]
    pred_boxes = outputs['pred_boxes'][0]    # [300, 4] cxcywh normalized
    pred_layers = outputs.get('pred_layers', None)
    
    if pred_layers is not None:
        pred_layers = pred_layers[0].squeeze(-1)  # [300]
    else:
        pred_layers = torch.zeros(300)
    
    # Get scores and labels
    scores = pred_logits.sigmoid().max(dim=-1)[0]  # [300]
    labels = pred_logits.sigmoid().argmax(dim=-1)  # [300]
    
    # Filter by confidence
    keep_mask = scores > args.threshold
    
    # Convert to numpy
    boxes_np = pred_boxes[keep_mask].cpu().numpy()
    scores_np = scores[keep_mask].cpu().numpy()
    labels_np = labels[keep_mask].cpu().numpy()
    layers_np = pred_layers[keep_mask].cpu().numpy()
    
    # Convert boxes from cxcywh normalized to xyxy pixel
    boxes_xyxy = []
    for box in boxes_np:
        cx, cy, w, h = box
        x1 = (cx - w/2) * orig_w
        y1 = (cy - h/2) * orig_h
        x2 = (cx + w/2) * orig_w
        y2 = (cy + h/2) * orig_h
        boxes_xyxy.append([x1, y1, x2, y2])
    boxes_xyxy = np.array(boxes_xyxy)
    
    print(f"Before Custom NMS: {len(boxes_xyxy)} boxes")

    # Apply Custom NMS
    boxes_xyxy, scores_np, labels_np, layers_np = custom_layer_nms(
        boxes_xyxy, scores_np, labels_np, layers_np, 
        iou_threshold=0.9, layer_diff_threshold=0.1
    )
    
    print(f"After Custom NMS: {len(boxes_xyxy)} detections (threshold={args.threshold})")

    # Expand boxes by 1.1x
    if len(boxes_xyxy) > 0:
        expanded_boxes = []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            
            new_w = w * 1.1
            new_h = h * 1.1
            
            new_x1 = max(0, cx - new_w / 2)
            new_y1 = max(0, cy - new_h / 2)
            new_x2 = min(orig_w, cx + new_w / 2)
            new_y2 = min(orig_h, cy + new_h / 2)
            
            expanded_boxes.append([new_x1, new_y1, new_x2, new_y2])
        boxes_xyxy = np.array(expanded_boxes)
        print("Expanded boxes by 1.1x (clipped to image boundaries)")


def main(args):
    """Main inference function"""
    
    # Load config
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    # Load checkpoint
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu') 
        
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
            print("Using EMA weights")
        elif 'model' in checkpoint:
            state = checkpoint['model']
            print("Using model weights")
        else:
            state = checkpoint
            print("Using raw checkpoint")
    else:
        raise AttributeError('Must provide --resume checkpoint path')
    
    # Load weights into model
    cfg.model.load_state_dict(state)
    
    # Create inference model
    class Model(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.eval()
            
        def forward(self, rgbd_images):
            """
            Args:
                rgbd_images: [1, 4, H, W]
            
            Returns:
                outputs: dict with 'pred_logits', 'pred_boxes', 'pred_layers'
            """
            with torch.no_grad():
                outputs = self.model(rgbd_images)
            return outputs
    
    model = Model(cfg.model).to(args.device)
    print(f"Model loaded on {args.device}")
    
    # Load RGBD image
    print(f"Loading image: {args.rgb_file}")
    rgb_pil, depth_pil, orig_w, orig_h = load_rgbd_image(args.rgb_file, args.depth_file)
    
    if args.use_zero_depth:
        print("⚠️ Forcing zero depth input (ignoring loaded depth map)")
        depth_pil = Image.new('L', (orig_w, orig_h), 0)

    # Prepare input
    rgbd_tensor = prepare_input(rgb_pil, depth_pil, target_size=640)
    rgbd_batch = rgbd_tensor.unsqueeze(0).to(args.device)  # [1, 4, 640, 640]
    
    print("Running inference...")
    outputs = model(rgbd_batch)
    
    # Extract predictions
    pred_logits = outputs['pred_logits'][0]  # [300, num_classes]
    pred_boxes = outputs['pred_boxes'][0]    # [300, 4] cxcywh normalized
    pred_layers = outputs.get('pred_layers', None)
    
    if pred_layers is not None:
        pred_layers = pred_layers[0].squeeze(-1)  # [300]
    else:
        pred_layers = torch.zeros(300)
    
    # Get scores and labels
    scores = pred_logits.sigmoid().max(dim=-1)[0]  # [300]
    labels = pred_logits.sigmoid().argmax(dim=-1)  # [300]
    
    # Filter by confidence
    keep_mask = scores > args.threshold
    
    # Convert to numpy
    boxes_np = pred_boxes[keep_mask].cpu().numpy()
    scores_np = scores[keep_mask].cpu().numpy()
    labels_np = labels[keep_mask].cpu().numpy()
    layers_np = pred_layers[keep_mask].cpu().numpy()
    
    # Convert boxes from cxcywh normalized to xyxy pixel
    boxes_xyxy = []
    for box in boxes_np:
        cx, cy, w, h = box
        x1 = (cx - w/2) * orig_w
        y1 = (cy - h/2) * orig_h
        x2 = (cx + w/2) * orig_w
        y2 = (cy + h/2) * orig_h
        boxes_xyxy.append([x1, y1, x2, y2])
    boxes_xyxy = np.array(boxes_xyxy)
    
    print(f"Before Custom NMS: {len(boxes_xyxy)} boxes")

    # Apply Custom NMS
    boxes_xyxy, scores_np, labels_np, layers_np = custom_layer_nms(
        boxes_xyxy, scores_np, labels_np, layers_np
    )
    
    print(f"After Custom NMS: {len(boxes_xyxy)} detections (threshold={args.threshold})")

    # Expand boxes logic
    if args.expand_ratio != 1.0 and len(boxes_xyxy) > 0:
        print(f"Expanding boxes by {args.expand_ratio}x")
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
        print("Expanded boxes clipped to image boundaries")
    
    # Print predictions
    for i, (box, label, score, layer) in enumerate(zip(boxes_xyxy, labels_np, scores_np, layers_np)):
        print(f"  [{i}] Class={label}, Score={score:.3f}, Layer={layer:.3f}, Box={box}")
    
    # Visualize
    visualize_predictions(rgb_pil, boxes_xyxy, labels_np, scores_np, layers_np,
                         threshold=args.threshold, save_path=args.output)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RT-DETR 4-channel inference with layer prediction')
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('-r', '--resume', type=str, required=True,
                       help='Path to checkpoint (.pth)')
    parser.add_argument('--rgb-file', type=str, required=True,
                       help='Path to input RGB image')
    parser.add_argument('--depth-file', type=str, default=None,
                       help='Path to depth map (grayscale PNG). If not provided, uses zero depth.')
    parser.add_argument('--use-zero-depth', action='store_true',
                       help='Force depth input to be zero (ignore depth file)')
    parser.add_argument('--expand-ratio', type=float, default=1.0,
                       help='Box expansion ratio (default 1.0 = no expansion)')
    parser.add_argument('-o', '--output', type=str, default='result.jpg',
                       help='Output visualization path')
    parser.add_argument('-d', '--device', type=str, default='cuda:4',
                       help='Device to run on')
    parser.add_argument('-t', '--threshold', type=float, default=0.3,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    main(args)
