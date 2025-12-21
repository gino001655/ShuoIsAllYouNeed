import os
import argparse
import torch
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import torch.nn.functional as F

# Import dataset loader
try:
    from tools.dlcv_dataset import DLCVLayoutDataset
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from tools.dlcv_dataset import DLCVLayoutDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CLD Inference results using CLIP Similarity")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the Ground Truth dataset (TAData directory)")
    parser.add_argument("--infer_dir", type=str, required=True, help="Path to the inference output directory (containing case_X folders)")
    parser.add_argument("--caption_mapping", type=str, default=None, help="Path to caption mapping JSON")
    parser.add_argument("--model_id", type=str, default="openai/clip-vit-base-patch32", help="HuggingFace CLIP model ID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def load_clip_model(model_id, device):
    print(f"[INFO] Loading CLIP model: {model_id}...")
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor

def get_image_features(model, processor, images, device):
    """Compute normalized image features"""
    # Ensure images are RGB
    images = [img.convert("RGB") for img in images]
    
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    
    # Normalize features
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features

def main():
    args = parse_args()
    
    # 1. Load Dataset (Ground Truth)
    print(f"[INFO] Loading GT Dataset from {args.data_dir}...")
    dataset = DLCVLayoutDataset(
        data_dir=args.data_dir,
        split="train", # Use all data
        caption_mapping_path=args.caption_mapping,
        enable_debug=False
    )
    print(f"[INFO] Dataset loaded with {len(dataset)} samples.")

    # 2. Load Model
    model, processor = load_clip_model(args.model_id, args.device)
    
    # 3. Iterate through inference results
    # Find all case directories: infer_dir/case_0, infer_dir/case_1 ...
    case_dirs = sorted(glob.glob(os.path.join(args.infer_dir, "case_*")))
    
    if not case_dirs:
        print(f"[WARN] No case directories found in {args.infer_dir}")
        return

    print(f"[INFO] Found {len(case_dirs)} inference cases to evaluate.")
    
    total_score = 0.0
    total_layers_count = 0
    case_scores = {}
    
    for case_dir in tqdm(case_dirs):
        dirname = os.path.basename(case_dir)
        # Extract index from "case_123"
        try:
            case_idx = int(dirname.split('_')[1])
        except (IndexError, ValueError):
            print(f"[WARN] Skipping invalid directory name: {dirname}")
            continue
            
        # Get Ground Truth Item
        if case_idx >= len(dataset):
            print(f"[WARN] Index {case_idx} out of bounds for dataset (size {len(dataset)}). Skipping.")
            continue
            
        gt_item = dataset[case_idx]
        # gt_layout: list of [x1, y1, x2, y2]
        # But we need images. 
        # DLCVLayoutDataset.__getitem__ returns specialized dict.
        # It puts layers into 'pixel_RGBA' tensor, but also potentially raw images?
        # Let's verify DLCVLayoutDataset.__getitem__ return value:
        # returns: "whole_img" (RGB Tensor), "pixel_RGBA" (Tensor Stack), "pixel_RGB" (Tensor Stack)
        # It does NOT return raw PIL images for layers directly in the final dict, 
        # BUT it reconstructs them from 'preview' and 'image' inside getitem.
        
        # To get PIL images for CLIP, we can convert the tensors back or modify dataset to return PIL.
        # Re-converting tensors [C, H, W] to PIL is easier without modifying dataset code.
        
        gt_whole_img_rgb = gt_item['whole_img'] # Tensor [3, H, W]
        gt_layers_rgba = gt_item['pixel_RGBA']  # Tensor [L, 4, H, W]
        # Layer 0: Whole (Redundant with whole_img usually, but pixel_RGBA[0] is whole_img)
        # Layer 1: Background
        # Layer 2+: Foregrounds
        
        # Load Prediction Images
        # Mapping:
        # GT Layer 0 -> whole_image_rgba.png
        # GT Layer 1 -> background_rgba.png
        # GT Layer k (k>=2) -> layer_{k-2}_rgba.png
        
        pred_images = {}
        
        # 3.1 Load Pred Whole Image
        path_whole = os.path.join(case_dir, "whole_image_rgba.png")
        if os.path.exists(path_whole):
            pred_images[0] = Image.open(path_whole)
            
        # 3.2 Load Pred Background
        path_bg = os.path.join(case_dir, "background_rgba.png")
        if os.path.exists(path_bg):
            pred_images[1] = Image.open(path_bg)
            
        # 3.3 Load Pred Foregrounds
        # GT layers count
        num_gt_layers = gt_layers_rgba.shape[0]
        for k in range(2, num_gt_layers):
            pred_idx = k - 2
            path_fg = os.path.join(case_dir, f"layer_{pred_idx}_rgba.png")
            if os.path.exists(path_fg):
                pred_images[k] = Image.open(path_fg)
        
        # Compare
        layer_scores = []
        
        for k in range(num_gt_layers):
            if k not in pred_images:
                # Missing prediction layer
                # print(f"  [Simple] Case {case_idx} Layer {k} missing prediction.")
                continue
                
                
            # Get GT Image (Tensor -> PIL)
            # gt_layers_rgba[k] is [4, H, W], value 0-1
            gt_tensor = gt_layers_rgba[k]
            gt_array = (gt_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gt_pil = Image.fromarray(gt_array, "RGBA").convert("RGB")
            
            # Get Pred Image (PIL RGBA -> RGB)
            pred_pil = pred_images[k].convert("RGB")
            
            # --- Compute Similarity (CLIP) ---
            feats = get_image_features(model, processor, [gt_pil, pred_pil], args.device)
            # feats shape: [2, 512]
            similarity = F.cosine_similarity(feats[0].unsqueeze(0), feats[1].unsqueeze(0)).item()
            
            # --- Compute MSE ---
            # Convert to numpy float arrays [0, 1]
            gt_np = np.array(gt_pil).astype(np.float32) / 255.0
            pred_np = np.array(pred_pil).astype(np.float32) / 255.0
            
            # Ensure same size (CLIP loads handle resize, but for MSE we need exact match)
            if gt_np.shape != pred_np.shape:
                # Resize pred to match GT
                pred_pil_resized = pred_pil.resize(gt_pil.size, Image.LANCZOS)
                pred_np = np.array(pred_pil_resized).astype(np.float32) / 255.0
            
            mse = np.mean((gt_np - pred_np) ** 2)
            
            layer_name = "Whole" if k == 0 else "Background" if k == 1 else f"Layer {k-2}"
            
            layer_scores.append({
                "idx": k,
                "name": layer_name,
                "clip": similarity,
                "mse": mse
            })
            
        if layer_scores:
            avg_clip = sum(s["clip"] for s in layer_scores) / len(layer_scores)
            avg_mse = sum(s["mse"] for s in layer_scores) / len(layer_scores)
            
            case_scores[case_idx] = {"clip": avg_clip, "mse": avg_mse, "count": len(layer_scores)}
            
            total_score += sum(s["clip"] for s in layer_scores)
            total_layers_count += len(layer_scores)
            
            # Detailed reporting per case
            print(f"Case {case_idx}: Avg CLIP {avg_clip:.4f}, MSE {avg_mse:.6f} (Layers: {len(layer_scores)})")
            for s in layer_scores:
                 print(f"  - {s['name']:<10}: CLIP={s['clip']:.4f}, MSE={s['mse']:.6f}")
        else:
            print(f"Case {case_idx}: No valid layers compared.")
            
    if total_layers_count > 0:
        macro_avg_clip = sum(c["clip"] for c in case_scores.values()) / len(case_scores)
        macro_avg_mse = sum(c["mse"] for c in case_scores.values()) / len(case_scores)
        
        # Micro is average over all layers
        # Since I didn't verify total MSE accumulation, let's recalculate accurately if needed.
        # But for CLIP I kept total_score.
        
        print("\n" + "="*40)
        print(f"EVALUATION RESULTS ({len(case_scores)} cases)")
        print("="*40)
        print(f"Macro Avg CLIP: {macro_avg_clip:.4f}")
        print(f"Macro Avg MSE : {macro_avg_mse:.4f}")
        print("="*40)

if __name__ == "__main__":
    main()
