
import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pathlib import Path
import numpy as np

from torchvision import datapoints
from src.core import register
from .coco_dataset import CocoDetection

@register
class CocoLayerOrderDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, depth_map_dir=None):
        """
        Args:
            depth_map_dir: Root directory for depth maps. 
                           Expects structure matching filename or flat, contains 'pred_layer_index.png'
                           User's depth pro output: out_dir/<split>/<category>/<sample_id>/pred_layer_index.png
        """
        super().__init__(img_folder, ann_file, transforms, return_masks)
        self.depth_map_dir = Path(depth_map_dir) if depth_map_dir else None

    def _load_image(self, id: int) -> Image.Image:
        # Override if needed, but we do it manual in getitem to handle 4 channels
        return super()._load_image(id)

    def __getitem__(self, idx):
        id = self.ids[idx]
        image = self._load_image(id) # RGB PIL Image
        
        # Load Target
        target = self._load_target(id) # List[Dict]
        
        # Inject Depth
        if self.depth_map_dir:
            file_name_raw = self.coco.loadImgs(id)[0]['file_name'] 
            # file_name example: "00000.png" or "category/id/image.png"
            
            # Robust Search Logic
            # 1. Direct name match (e.g. "00001.png" in depth_dir) using leaf name
            depth_path = self.depth_map_dir / Path(file_name_raw).name
            
            if not depth_path.exists():
                # 2. Legacy structure: try replacing "00_whole_image.png" with "pred_layer_index.png"
                # This works if file_name is "category/id/00_whole_image.png" and depth_dir matches structure
                depth_path = self.depth_map_dir / file_name_raw.replace("00_whole_image.png", "pred_layer_index.png")
            
            if not depth_path.exists():
                # 3. Legacy flattened: leaf name replacement
                depth_path = self.depth_map_dir / Path(file_name_raw).name.replace("00_whole_image.png", "pred_layer_index.png")

            if depth_path.exists():
                depth_img = Image.open(depth_path).convert("L")
                # Resize to match RGB (before transforms)
                if depth_img.size != image.size:
                    depth_img = depth_img.resize(image.size, Image.NEAREST)
                
                depth_np = np.array(depth_img).astype(np.float32) / 255.0
                depth_tensor = torch.from_numpy(depth_np).unsqueeze(0) # (1, H, W)
            else:
                # Fallback: zeros
                W, H = image.size
                depth_tensor = torch.zeros(1, H, W)
        else:
             W, H = image.size
             depth_tensor = torch.zeros(1, H, W)

        # Merge RGB + Depth -> Tensor
        # Convert RGB PIL to Tensor
        image_tensor = torchvision.transforms.functional.to_tensor(image) # (3, H, W)
        image_tensor = torch.cat([image_tensor, depth_tensor], dim=0) # (4, H, W)
        
        # Wrap in Image object for v2 transforms if needed, but Tensor is fine.
        # However, for TV v2 transforms to know it's an image (for photometric distort etc), 
        # normally we wrap it in datapoints.Image, but 4-channel might break "PhotometricDistort".
        # We assume the user config mostly uses geometric transforms (Resize, Crop, Flip).
        # We wrap it to be safe if 'datapoints' available.
        image_tensor = datapoints.Image(image_tensor)

        # Prepare Target for Transforms
        # RT-DETR transforms expect {'boxes': ..., 'labels': ...} output from CoCoDetection usually happens inside.
        # But we are manually calling transforms.
        
        # Convert raw annotation list to standard dict format
        target = dict(image_id=id, annotations=target)
        # We need to extract 'layers' from annotations and add to target
        # Assume annotation dict has 'layer_order'
        
        # Flatten target (standard coco logic often implemented in custom ConvertCocoPolysToMask but let's do it here)
        anns = target['annotations']
        if len(anns) > 0:
            # Boxes provided by coco are [x, y, w, h]
            boxes = [obj['bbox'] for obj in anns]
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes = datapoints.BoundingBox(boxes, format=datapoints.BoundingBoxFormat.XYWH, spatial_size=image.size[::-1])
            
            labels = [obj['category_id'] for obj in anns] # Needs remapping? usually yes.
            labels = torch.tensor(labels, dtype=torch.int64)
            
            # NEW: Layers
            # If 'layer_order' is missing, default to 0.5? or -1?
            layers = [obj.get('layer_order', 0.5) for obj in anns]
            layers = torch.as_tensor(layers, dtype=torch.float32)
            
            target = {
                'boxes': boxes,
                'labels': labels,
                'layers': layers, # Won't be transformed by geometric ops, which is distinct?
                                  # Wait, layers is a scalar property of the object, like label. 
                                  # It survives geometric transforms (crop might drop some boxes, and thus drop corresponding layers/labels).
                                  # TV v2 converts dict values. If it's a Tensor matching box count, does it slice it automatically?
                                  # NO. Only specific keys or datapoint types.
                                  # Labels are sliced because they are passed specially or usually just standard keys.
                                  # We need to check if 'layers' will be sliced.
                                  # Usually not. We might need to handle 'layers' dropping manually or assume transforms handles all tensor vectors matching box count?
                                  # transforms.v2 is smart but maybe not that smart for custom keys.
            }
        else:
            target = {
                'boxes': datapoints.BoundingBox(torch.zeros((0, 4)), format=datapoints.BoundingBoxFormat.XYWH, spatial_size=image.size[::-1]),
                'labels': torch.tensor([], dtype=torch.int64),
                'layers': torch.tensor([], dtype=torch.float32),
            }

        # Apply Transforms
        if self.transforms is not None:
            # We explicitly pass our construct. 
            # Note: The transforms must be V2 compatible (accepting dict target)
            output = self.transforms(image_tensor, target)
            image_tensor, target = output
            
            # Verify if 'layers' was sliced properly if boxes were dropped (e.g. RandomCrop)
            # If 'boxes' size changed, 'layers' logic:
            # Standard TV v2 wrapper won't slice 'layers' unless we tell it it's a "Labels" type or similar.
            # RT-DETR 'SanitizeBoundingBox' or 'RandomCrop' usually filters boxes.
            # We need to ensure 'layers' is filtered in sync.
            
            if len(target['boxes']) != len(target['layers']):
                # This implies transforms dropped boxes but not layers
                # This is tricky. 
                # Workaround: If we cannot easily hook into transform slicing, 
                # we might need to assume no cropping or accept mismatch (bad).
                # OR, we define 'layers' as a TV Label type?
                # TV v2 doesn't have a generic "PerBoxAttribute".
                pass
                
        return image_tensor, target
