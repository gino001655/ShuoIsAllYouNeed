
import os
import json
import random
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import math

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.dlcv_dataset import DLCVLayoutDataset

def get_rotated_bbox(bbox, angle, center, img_w, img_h):
    """
    Calculate the bounding box of a rotated rectangle.
    bbox: [x, y, w, h] (top-left x, top-left y, width, height)
    angle: rotation angle in degrees (counter-clockwise)
    center: center of rotation (cx, cy)
    """
    x, y, w, h = bbox
    cx, cy = center
    
    # 4 corners of the rectangle relative to center
    corners = np.array([
        [x - cx, y - cy],
        [x + w - cx, y - cy],
        [x + w - cx, y + h - cy],
        [x - cx, y + h - cy]
    ])
    
    # Rotation matrix
    theta = math.radians(-angle) # PIL rotates clockwise for negative? No, mathematically counter-clockwise
    c, s = math.cos(theta), math.sin(theta)
    R = np.array(((c, -s), (s, c)))
    
    # Rotate corners
    rotated_corners = np.dot(corners, R.T)
    
    # Shift back
    rotated_corners[:, 0] += cx
    rotated_corners[:, 1] += cy
    
    # New bounding box
    x_min = np.min(rotated_corners[:, 0])
    x_max = np.max(rotated_corners[:, 0])
    y_min = np.min(rotated_corners[:, 1])
    y_max = np.max(rotated_corners[:, 1])
    
    # Clip to image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)
    
    if x_max <= x_min or y_max <= y_min:
        return None
        
    return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

class ComplexLayoutAugmentor:
    def __init__(self, data_dir, output_dir="debug_augment", canvas_size=(1024, 1024)):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.canvas_size = canvas_size
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset to access layers
        print("Dataset loading...")
        # We use DLCVDataset to access paths easily
        self.dataset = DLCVLayoutDataset(
            data_dir=data_dir, 
            split="train"
        )
        print(f"Loaded {len(self.dataset)} base images.")

    def generate_random_layer(self):
        """Pick a random layer from the dataset"""
        idx = random.randint(0, len(self.dataset) - 1)
        
        # Access raw data manually to get crops
        json_path = self.dataset.data_paths[idx]
        img_dir = self.dataset.img_dir_map[json_path]
        
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except:
            return None
            
        if "assets" not in data or not data["assets"]:
             # Try 'elements' if assets distinct
             if "elements" in data:
                 # Logic for standard format
                 pass
             return None

        # DLCV format logic: 'assets' usually contains layer image paths?
        # Let's inspect data[0] if it list? No data is dict.
        # Based on dlcv_dataset.py, data structure is analyzed there.
        # Actually dlcv_dataset loads 'image' key which contains list of paths.
        
        if "image" not in data:
            return None
            
        layer_idx = random.randint(0, len(data["image"]) - 1)
        layer_rel_path = data["image"][layer_idx]
        
        layer_path = os.path.join(img_dir, layer_rel_path)
        
        if not os.path.exists(layer_path):
            return None
            
        try:
            layer_img = Image.open(layer_path).convert("RGBA")
        except:
            return None
            
        return layer_img

    def create_super_complex_image(self, num_layers=20, complexity_factor=1.0):
        """
        Generate one super complex image.
        """
        canvas = Image.new("RGBA", self.canvas_size, (255, 255, 255, 255))
        annotations = []
        
        # Create a randomized background color occasionally
        if random.random() < 0.3:
            bg_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255), 255)
            canvas = Image.new("RGBA", self.canvas_size, bg_color)
        
        pbar = tqdm(range(num_layers), desc="Mixing Soup", leave=False)
        for _ in pbar:
            layer = self.generate_random_layer()
            if layer is None: continue
            
            # --- Geometric Jittering ---
            
            # 1. Scaling
            scale = random.uniform(0.3, 1.5) * complexity_factor
            new_w = int(layer.width * scale)
            new_h = int(layer.height * scale)
            if new_w <= 10 or new_h <= 10: continue
            layer = layer.resize((new_w, new_h), resample=Image.BILINEAR)
            
            # 2. Rotation
            angle = random.uniform(-30, 30) * complexity_factor
            if random.random() < 0.2 * complexity_factor: # Occasional full rotation
                angle = random.uniform(0, 360)
                
            layer_rotated = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
            
            # 3. Translation
            cx = random.randint(0, self.canvas_size[0])
            cy = random.randint(0, self.canvas_size[1])
            
            # Top-left from centroid
            tl_x = cx - layer_rotated.width // 2
            tl_y = cy - layer_rotated.height // 2
            
            # Paste (Alpha composite)
            canvas.alpha_composite(layer_rotated, dest=(tl_x, tl_y))
            # Note: PIL alpha_composite requires same size, or we typically use paste with mask
            # Safe way for arbitrary position:
            temp_canvas = Image.new("RGBA", self.canvas_size, (0,0,0,0))
            temp_canvas.paste(layer_rotated, (tl_x, tl_y))
            canvas = Image.alpha_composite(canvas, temp_canvas)
            
            # BBox Calculation
            final_bbox = [
                max(0, tl_x), 
                max(0, tl_y), 
                min(self.canvas_size[0], tl_x + layer_rotated.width) - max(0, tl_x),
                min(self.canvas_size[1], tl_y + layer_rotated.height) - max(0, tl_y)
            ]
            
            if final_bbox[2] > 0 and final_bbox[3] > 0:
                annotations.append({
                    "bbox": final_bbox,
                    "rotation": angle
                })
                
        return canvas.convert("RGB"), annotations

    def visualize(self, count=5):
        print(f"Generating {count} super complex samples...")
        for i in range(count):
            img, annos = self.create_super_complex_image(num_layers=random.randint(15, 40))
            
            # Draw bboxes
            vis_img = img.copy()
            draw = ImageDraw.Draw(vis_img)
            for ann in annos:
                x, y, w, h = ann['bbox']
                draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
                
            save_path = os.path.join(self.output_dir, f"complex_augment_{i}.jpg")
            vis_img.save(save_path)
            print(f"Saved: {save_path}")

if __name__ == "__main__":
    DATA_DIR = "/workspace/dataset/cld_dataset/snapshots/snapshot_1"
    
    augmentor = ComplexLayoutAugmentor(DATA_DIR)
    augmentor.visualize(count=5)
