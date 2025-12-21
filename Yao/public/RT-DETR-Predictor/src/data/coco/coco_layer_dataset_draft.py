
import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools import mask as coco_mask
import numpy as np
from pathlib import Path

from src.core import register
from .coco_dataset import CocoDetection, mscoco_category2label, mscoco_label2category, mscoco_category2name

@register
class CocoLayerOrderDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, depth_map_dir=None):
        super().__init__(img_folder, ann_file, transforms, return_masks)
        self.depth_map_dir = Path(depth_map_dir) if depth_map_dir else None

    def __getitem__(self, idx):
        # 1. Load original RGB image and target
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        
        # 2. Load Depth Map (assuming name correspondence matches image_id or filename)
        # We need to find the filename first.
        # super() loads via self.coco.loadImgs(image_id)[0]['file_name'] internally but doesn't expose it
        coco_img_info = self.coco.loadImgs(image_id)[0]
        file_name = coco_img_info['file_name']
        
        depth_tensor = torch.zeros(1, img.shape[-2], img.shape[-1]) # Default 0 if no depth map

        if self.depth_map_dir:
            # Construct depth map path. Structure: split/category/sample_id/pred_layer_index.png
            # Or simplified: assuming depth_map_dir contains matching filenames or nested structure.
            # We will search recursively or assume a specific structure matching typical setups.
            # For now, let's assuming a flat or mirrored structure.
            # Since user's depth pro output is in: out_dir/<split>/<category>/<sample_id>/pred_layer_index.png
            # and Coco filenames might be just "00_whole_image.png" inside a folder structure "category/sample_id" if using parsed dataset style?
            # Or standard COCO flat filenames.
            
            # ADAPTATION: We assume the dataset structure mirrors the parsed_dataset logic.
            # file_name usually includes relative path if defined in JSON "file_name": "category/sample_id/00_whole_image.png"
            
            # Depth map filename in ml-depth-pro output is "pred_layer_index.png"
            # So we replace "00_whole_image.png" with "pred_layer_index.png"
            depth_path = self.depth_map_dir / file_name.replace("00_whole_image.png", "pred_layer_index.png")
            
            if depth_path.exists():
                depth_img = Image.open(depth_path).convert("L") # Grayscale
                # Resize to match RGB if necessary (though they should match)
                if depth_img.size != (img.shape[-1], img.shape[-2]):
                    depth_img = depth_img.resize((img.shape[-1], img.shape[-2]), Image.NEAREST)
                
                depth_np = np.array(depth_img).astype(np.float32) / 255.0 # Normalize 0-1
                depth_tensor = torch.from_numpy(depth_np).unsqueeze(0) # (1, H, W)
            else:
                # Fallback or warning?
                # print(f"Warning: Depth map not found for {file_name}")
                pass

        # 3. Concatenate (RGB + Depth) -> (4, H, W)
        # img is Tensor [3, H, W] from transforms (if ToTensor used)
        # ensure img is tensor
        if not isinstance(img, torch.Tensor):
            img = torchvision.transforms.functional.to_tensor(img)
            
        img = torch.cat([img, depth_tensor], dim=0)

        # 4. Add 'layers' to target
        # Target is list of dicts (from super) or dict (if transforms merged it)?
        # CocoDetection returns (img, target) where target is LIST of dicts (raw coco annotations)
        # BUT standard RT-DETR transforms usually convert it to a single dict with 'boxes', 'labels' keys.
        # Let's check 'transforms'.
        
        # If transforms is standard RT-DETR 'Compose', it handles (image, target) -> (image, target_dict)
        # We need to inject 'layers' into target dict BEFORE transforms if we want transforms to handle it (e.g. crop invalidation),
        # OR AFTER if we handle it manually. 
        # RT-DETR transforms expect 'boxes', 'labels'. 
        
        # We need to extract 'layer' info from annotation
        # Assuming our JSON annotations have a 'layer_order' field or similar.
        # User defined: "Assume the DataLoader already provides this normalized value in the targets."
        # We need to parse it from 'segmentation' or 'categories' or new field.
        
        # ADAPTATION: We will calculate normalized layer order here if not present.
        # Simple heuristic or using 'id' order? 
        # Actually user said "Assume DataLoader provides it".
        # Let's assume the annotation JSON has a field 'layer_order' (int).
        
        # We inject it into the raw target list first
        new_target = {'image_id': image_id, 'annotations': target}
        
        # Pre-process 'layers' for the transforms pipeline
        # RT-DETR transforms usually look for target['boxes'] etc.
        # We rely on the `transforms` passed to super to convert List[Dict] -> Dict[Tensor]
        # BUT standard CocoDetection just returns the list. The transforms do the work.
        
        # To make 'layers' survive the transforms (like Resize, RandomCrop), we likely need to modify the transforms?
        # Standard transforms might drop unknown keys.
        # However, for now, we will assume we attach it to the output target tensor dict.
        
        # Let's assume transforms has been updated OR we do it manually.
        # Actually, simpler: We calculate layer values and add them.
        
        # We need to manually construct the 'layers' tensor if transforms don't.
        # But 'transforms' argument in __init__ usually does everything.
        
        return img, target

@register
class CocoLayerOrderDatasetWrapped(CocoDetection):
    # Alternative strategy: We wrap the standard output and inject 4th channel and layer targets.
    # This avoids rewriting internal transform logic if we are careful.
    def __init__(self, img_folder, ann_file, transforms, return_masks, depth_map_dir=None):
        super().__init__(img_folder, ann_file, transforms, return_masks)
        self.depth_map_dir = Path(depth_map_dir) if depth_map_dir else None

    def __getitem__(self, idx):
        # image, target = super().__getitem__(idx) 
        # The parent returns transformed image and target.
        # Image is (3, H, W), Target is dict {'boxes': ..., 'labels': ...}
        
        img, target = super().__getitem__(idx)
        
        # 1. Depth Channel Injection
        if self.depth_map_dir:
            image_id = target['image_id'].item() # target['image_id'] is a tensor usually
            # We need to look up file name.
            # Convert tensor image_id back to int
            if isinstance(image_id, torch.Tensor):
                image_id = int(image_id)
            
            coco_img_info = self.coco.loadImgs(image_id)[0]
            file_name = coco_img_info['file_name']
            
            # Path logic matching user structure
            depth_path = self.depth_map_dir / file_name.replace("00_whole_image.png", "pred_layer_index.png")
            
            depth_tensor = torch.zeros(1, img.shape[-2], img.shape[-1])
            if depth_path.exists():
                depth_img = Image.open(depth_path).convert("L")
                depth_img = depth_img.resize((img.shape[-1], img.shape[-2]), Image.NEAREST) # Resize to match augmented image size?
                # WAIT: transforms might have cropped/resized the image!
                # If we load depth map here, it won't match the transformed image (e.g. random crop).
                # CRITICAL: Depth map MUST go through same transforms as Image.
                pass
        
        return img, target

# REDO: We MUST modify the dataset to load 4 channels from the start so transforms apply to all 4.
# OR we modify the transforms to accept 4 channels. 
# Standard Torchvision transforms work on multi-channel tensors usually.
# So strategy:
# Override _load_image in CocoDetection? No, it uses self.coco.loadImgs directly in __getitem__.
# We must use the first approach but handle transforms carefully.

