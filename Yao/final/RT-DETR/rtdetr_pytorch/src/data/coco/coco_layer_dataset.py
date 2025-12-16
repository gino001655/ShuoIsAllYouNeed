
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
        
        # Load Depth Map (PIL Image, not tensor yet)
        depth_img = None
        if self.depth_map_dir:
            file_name_raw = self.coco.loadImgs(id)[0]['file_name'] 
            depth_path = self.depth_map_dir / Path(file_name_raw).name
            
            if not depth_path.exists():
                depth_path = self.depth_map_dir / file_name_raw.replace("00_whole_image.png", "pred_layer_index.png")
            
            if not depth_path.exists():
                depth_path = self.depth_map_dir / Path(file_name_raw).name.replace("00_whole_image.png", "pred_layer_index.png")

            if depth_path.exists():
                depth_img = Image.open(depth_path).convert("L")
                if depth_img.size != image.size:
                    depth_img = depth_img.resize(image.size, Image.NEAREST)
        
        # Convert annotations to target dict
        target = dict(image_id=id, annotations=target)
        anns = target['annotations']
        
        if len(anns) > 0:
            boxes = torch.as_tensor([obj['bbox'] for obj in anns], dtype=torch.float32).reshape(-1, 4)
            boxes = datapoints.BoundingBox(boxes, format=datapoints.BoundingBoxFormat.XYWH, spatial_size=image.size[::-1])
            labels = torch.tensor([obj['category_id'] for obj in anns], dtype=torch.int64)
            layers = torch.as_tensor([obj.get('layer_order', 0.5) for obj in anns], dtype=torch.float32)
            
            target = {'boxes': boxes, 'labels': labels, 'layers': layers}
        else:
            target = {
                'boxes': datapoints.BoundingBox(torch.zeros((0, 4)), format=datapoints.BoundingBoxFormat.XYWH, spatial_size=image.size[::-1]),
                'labels': torch.tensor([], dtype=torch.int64),
                'layers': torch.tensor([], dtype=torch.float32),
            }

        # Apply Transforms (on PIL Image)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
            # Handle layer syncing if boxes were filtered
            if 'boxes' in target and 'layers' in target:
                num_boxes = len(target['boxes'])
                if len(target['layers']) > num_boxes:
                    # Assume boxes were filtered, need to sync layers
                    # This is a workaround - ideally transforms would handle this
                    target['layers'] = target['layers'][:num_boxes]
        
        # NOW convert to tensor and merge depth
        # After transforms, image should be a tensor from ToImageTensor transform
        if not isinstance(image, torch.Tensor):
            image = torchvision.transforms.functional.to_tensor(image)
        
        # Resize depth map to match transformed image size
        # IMPORTANT: image tensor shape is (C, H, W), PIL size is (W, H)
        _, img_h, img_w = image.shape
        if depth_img is not None:
            # PIL Image.size returns (width, height), we need (width, height) for resize
            if depth_img.size != (img_w, img_h):
                depth_img =depth_img.resize((img_w, img_h), Image.NEAREST)
            depth_tensor = torch.from_numpy(np.array(depth_img).astype(np.float32) / 255.0).unsqueeze(0)
        else:
            depth_tensor = torch.zeros(1, img_h, img_w)
        
        # Merge RGB + Depth
        image = torch.cat([image, depth_tensor], dim=0) # (4, H, W)
                
        return image, target
