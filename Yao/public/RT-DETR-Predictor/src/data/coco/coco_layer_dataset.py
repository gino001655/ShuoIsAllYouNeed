
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
    """
    COCO Detection Dataset with Layer Order prediction and 4-channel input (RGB + Depth).
    
    DATA VALIDATION:
    - Automatically skips samples without depth maps
    - Automatically skips samples with missing 'layer_order' in annotations
    - Automatically skips samples without any annotations
    """
    __inject__ = ['transforms']

    def __init__(self, img_folder, ann_file, transforms, return_masks, depth_map_dir=None):
        """
        Args:
            depth_map_dir: Root directory for depth maps. 
                           Expects structure matching filename or flat, contains 'pred_layer_index.png'
                           User's depth pro output: out_dir/<split>/<category>/<sample_id>/pred_layer_index.png
                           REQUIRED: Dataset will skip samples without valid depth maps.
        """
        super().__init__(img_folder, ann_file, transforms, return_masks)
        self.depth_map_dir = Path(depth_map_dir) if depth_map_dir else None

    def _load_image(self, id: int) -> Image.Image:
        # Override if needed, but we do it manual in getitem to handle 4 channels
        return super()._load_image(id)

    def __getitem__(self, idx):
        """
        Get item with validation: skip samples without depth maps or layer_order.
        If validation fails, try the next sample (with recursion limit).
        """
        max_attempts = 10  # Prevent infinite recursion
        attempt = 0
        
        while attempt < max_attempts:
            # Handle wrap-around
            current_idx = (idx + attempt) % len(self.ids)
            id = self.ids[current_idx]
            
            try:
                image = self._load_image(id) # RGB PIL Image
                
                # Load Target
                target = self._load_target(id) # List[Dict]
                
                # VALIDATION 1: Check if depth map exists
                depth_img = None
                if self.depth_map_dir:
                    file_name_raw = self.coco.loadImgs(id)[0]['file_name'] 
                    depth_path = self.depth_map_dir / Path(file_name_raw).name
                    
                    if not depth_path.exists():
                        depth_path = self.depth_map_dir / file_name_raw.replace("00_whole_image.png", "pred_layer_index.png")
                    
                    if not depth_path.exists():
                        depth_path = self.depth_map_dir / Path(file_name_raw).name.replace("00_whole_image.png", "pred_layer_index.png")

                    if not depth_path.exists():
                        # SKIP: No depth map found
                        if attempt == 0:  # Only print on first attempt
                            print(f"⚠️  Skipping sample {current_idx} (id={id}): Depth map not found")
                        attempt += 1
                        continue
                    
                    depth_img = Image.open(depth_path).convert("L")
                    if depth_img.size != image.size:
                        depth_img = depth_img.resize(image.size, Image.NEAREST)
                else:
                    # SKIP: No depth_map_dir specified
                    if attempt == 0:
                        print(f"⚠️  Skipping sample {current_idx} (id={id}): No depth_map_dir specified")
                    attempt += 1
                    continue
                
                # Convert annotations to target dict
                target = dict(image_id=id, annotations=target)
                anns = target['annotations']
                
                # VALIDATION 2: Check if all annotations have layer_order
                if len(anns) > 0:
                    # Check if any annotation is missing layer_order
                    missing_layer_order = [i for i, obj in enumerate(anns) if 'layer_order' not in obj]
                    if missing_layer_order:
                        if attempt == 0:
                            print(f"⚠️  Skipping sample {current_idx} (id={id}): {len(missing_layer_order)}/{len(anns)} annotations missing 'layer_order'")
                        attempt += 1
                        continue
                    
                    boxes = torch.as_tensor([obj['bbox'] for obj in anns], dtype=torch.float32).reshape(-1, 4)
                    boxes = datapoints.BoundingBox(boxes, format=datapoints.BoundingBoxFormat.XYWH, spatial_size=image.size[::-1])
                    # Force label to 0 for single-class training
                    labels = torch.zeros(len(anns), dtype=torch.int64)
                    layers = torch.as_tensor([obj['layer_order'] for obj in anns], dtype=torch.float32)
                    
                    target = {
                        'boxes': boxes, 
                        'labels': labels, 
                        'layers': layers,
                        'image_id': torch.tensor([id]),
                        'orig_size': torch.as_tensor([int(image.size[0]), int(image.size[1])])
                    }
                else:
                    # SKIP: No annotations (empty image)
                    if attempt == 0:
                        print(f"⚠️  Skipping sample {current_idx} (id={id}): No annotations")
                    attempt += 1
                    continue

                # Apply Transforms (on PIL Image)
                if self._transforms is not None:
                    image, target = self._transforms(image, target)
                    
                    # Handle layer syncing if boxes were filtered
                    if 'boxes' in target and 'layers' in target:
                        num_boxes = len(target['boxes'])
                        if len(target['layers']) > num_boxes:
                            # Assume boxes were filtered, need to sync layers
                            target['layers'] = target['layers'][:num_boxes]
                
                # NOW convert to tensor and merge depth
                # After transforms, image should be a tensor from ToImageTensor transform
                if not isinstance(image, torch.Tensor):
                    image = torchvision.transforms.functional.to_tensor(image)
                
                # Check for mismatched sizes explicit check
                if image.shape[1] != 640 or image.shape[2] != 640:
                    print(f"⚠️  WARNING: Image shape mismatch at idx {idx}: {image.shape}. Expected 640x640. Transform might be failing.")
                
                # Resize depth map to match transformed image size
                # IMPORTANT: image tensor shape is (C, H, W), PIL size is (W, H)
                _, img_h, img_w = image.shape
                if depth_img is not None:
                    # PIL Image.size returns (width, height), we need (width, height) for resize
                    if depth_img.size != (img_w, img_h):
                        depth_img = depth_img.resize((img_w, img_h), Image.NEAREST)
                    depth_tensor = torch.from_numpy(np.array(depth_img).astype(np.float32) / 255.0).unsqueeze(0)
                else:
                    # This should not happen due to earlier validation
                    depth_tensor = torch.zeros(1, img_h, img_w)
                
                # Merge RGB + Depth
                image = torch.cat([image, depth_tensor], dim=0) # (4, H, W)
                        
                return image, target
                
            except Exception as e:
                # SKIP: Any other error (corrupted image, etc.)
                print(f"⚠️  Skipping sample {idx} (id={id}) due to error: {e}")
                import traceback
                traceback.print_exc()
                attempt += 1
                continue
        
        # If we've tried max_attempts times and all failed, raise error
        raise RuntimeError(f"Failed to load valid sample after {max_attempts} attempts starting from idx {idx}")
