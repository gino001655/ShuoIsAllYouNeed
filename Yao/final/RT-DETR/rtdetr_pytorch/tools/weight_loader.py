
import torch
import torch.nn as nn

def load_rtdetr_weights(model, checkpoint_path):
    """
    Load 3-channel COCO weights into 4-channel RT-DETR model.
    1. Loads original state dict.
    2. Adapts backbone.conv1 to 4 channels (zero-init 4th channel).
    3. Ignores shape mismatches (heads).
    """
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'ema' in checkpoint and checkpoint['ema'] is not None:
        state_dict = checkpoint['ema']['module']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model_state_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in model_state_dict:
            # Check shape mismatch
            if v.shape != model_state_dict[k].shape:
                # Handle Backbone Conv1 
                # PresNet structure: backbone.conv1.0.conv.weight (if sequential) or similar
                # We look for 'conv1' and 4D tensor with input channel mismatch
                if 'conv1' in k and v.dim() == 4 and model_state_dict[k].dim() == 4:
                    if v.shape[1] == 3 and model_state_dict[k].shape[1] == 4:
                        print(f"Adapting {k} from {v.shape} to {model_state_dict[k].shape} (RGB -> RGBD)")
                        new_weight = torch.zeros_like(model_state_dict[k])
                        new_weight[:, :3, :, :] = v
                        # 4th channel initialized to 0.0 by torch.zeros_like
                        new_state_dict[k] = new_weight
                        continue
                
                print(f"Skipping {k} due to shape mismatch: checkpoint {v.shape} vs model {model_state_dict[k].shape}")
                continue
            
            new_state_dict[k] = v
        else:
            # Key not in model (e.g. maybe some aux head stuff if architecture changed significantly, or just unused)
            pass
            
    # Load strict=False to ignore missing keys (new heads) and unexpected keys
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    print("Weight loading completed.")
    print(f"Missing keys (initialized randomly/default): {len(missing)}")
    # Expected missing: *layer_head*, *score_head* (if class num diff), *bbox_head* (if using one shared head logic changed?? No usually same)
    # Actually score_head will be missing/skipped if num_classes changed.
    
    return model
