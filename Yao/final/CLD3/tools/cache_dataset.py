import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import safetensors.torch

from CLD3.models.sd3_backbone import SD3Backbone  # Using Phase-1 backbone temporarily to access VAE/Scheduler
from CLD3.models.qwen_projector import QwenProjector, QwenProjectorConfig
from CLD3.tools.dataset import LayoutTrainDataset, collate_fn
from CLD3.tools.tools import encode_target_latents, get_input_box, build_layer_mask

def cache_data(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # 1. Load Models (VAE + Qwen)
    print("Loading VAE and Qwen Projector...")
    vae_provider = SD3Backbone(
        sd3_model_path=args.sd3_path,
        dit_path=args.dit_path,
        dit_lora_path=None,
        device=device,
        dtype=dtype,
        gradient_checkpointing=False
    )
    
    qwen_cfg = QwenProjectorConfig(
        llm_path=args.llm_path,
        processor_path=args.llm_processor_path,
        device=device,
        dtype=dtype,
        load_in_4bit=args.load_4bit
    )
    qwen = QwenProjector(qwen_cfg)
    
    # 2. Dataset
    dataset = LayoutTrainDataset(args.data_dir, split=args.split)
    loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate_fn
    )

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Start Caching {len(dataset)} samples...")
    
    for idx, batch in enumerate(tqdm(loader)):
        sample_id = f"{idx:06d}"
        save_path = os.path.join(out_dir, f"{sample_id}.safetensors")
        
        if os.path.exists(save_path) and not args.overwrite:
            continue

        # Data Prep
        H = int(batch["height"][0])
        W = int(batch["width"][0])
        caption = batch["caption"][0]
        layout = batch["layout"][0]
        
        pixel_rgb = batch["pixel_RGB"].to(device=device, dtype=dtype)
        # Normalize to [-1, 1]
        pixel_rgb = pixel_rgb[0].clamp(0, 1) * 2.0 - 1.0
        
        layer_boxes = get_input_box(layout) # List of boxes
        
        # 3. Compute Features
        with torch.no_grad():
            # A. Text Embeddings
            prompt_embeds = qwen.encode_text(caption) # [1, seq, 4096]
            pooled_embeds = vae_provider.pooled_prompt_embeds # [1, pooled_dim]
            
            # B. VAE Latents
            # encode_target_latents returns [1, L, C, H_lat, W_lat] and ids
            x0, latent_image_ids = encode_target_latents(
                vae_provider.vae, 
                pixel_rgb.unsqueeze(0), 
                n_layers=len(layer_boxes), 
                list_layer_box=layer_boxes
            )
            
            # C. Boxes & Masks
            # We save boxes as a tensor
            # Since boxes are list of varying length, we pad or just save as tight tensor?
            # Safetensors supports storing multiple tensors. We can store "box_0", "box_1" etc or just flat.
            # Best strategy: Store x0, prompt_embeds, pooled_embeds, and metadata.
            
            # Since variable length is tricky in batching, but here we process batch=1 cache.
            # To make training loader super fast, we should save everything needed for Training Step.
            
            # We need:
            # - x0 (Latents)
            # - prompt_embeds
            # - pooled_embeds
            # - latent_image_ids (Pre-computed!)
            # - boxes (for creating mask in training, or we pre-compute mask?)
            
            # Let's pre-compute mask too! feature: Infinite Compression.
            _, L, C_lat, H_lat, W_lat = x0.shape
            mask = build_layer_mask(L, H_lat, W_lat, layer_boxes) # [L, 1, H, W]
            
        # 4. Save to Disk (Safetensors)
        tensors = {
            "latents": x0.squeeze(0).cpu(), # [L, C, H, W]
            "prompt_embeds": prompt_embeds.squeeze(0).cpu(), # [Seq, 4096]
            "pooled_embeds": pooled_embeds.squeeze(0).cpu(), # [Pooled]
            "latent_image_ids": latent_image_ids.cpu(), # [Seq_img, 3]
            "mask": mask.cpu(), # [L, 1, H, W]
        }
        # Safetensors doesn't support List[List]. We serialize boxes as a tensor if needed, 
        # but actually if we have the Mask and Latents and IDs, we might not strictly need boxes for training 
        # UNLESS we do some box-based conditioning again.
        # But Phase-1 training only needs Mask for Loss.
        # Phase-2 Deep Injection might need Box coordinates? 
        # ControlNet usually works on pixel space or latent space maps.
        # Let's verify if we need boxes. 
        # CLD2 train loop uses boxes for: `get_input_box` -> `build_layer_mask`. 
        # We already built mask.
        # And `pipeline.multiLayerAdapter(list_layer_box=layer_boxes)`. 
        # Ah, the Adapter NEEDS boxes to do `crop_each_layer` again?
        # Wait, if we pre-crop? 
        # CLD works by cropping from the "whole" latent.
        # But here x0 is ALREADY cropped per layer? 
        # Re-reading `encode_target_latents`: It encodes the whole image, then clones it L times?
        # No, CLD VAE encode:
        # `z` list. `crop_each_layer` is called inside VAE encode? 
        # Let's check `tools.tools.encode_target_latents` in CLD2... (I can't see it now but I remember).
        # Actually CLD typically inputs the *Whole* latent and mask.
        
        # In CLD2 `train.py`:
        # x0 = encode_target_latents(...) -> [1, L, C, H, W]
        # It returns a tensor where L is the layer dimension.
        # If I look at CLD1 `transp_vae.py`: `encode` calls `crop_each_layer`.
        # So `x0` currently ARE the cropped tokens? 
        # In CLD1 `transp_vae.encode`: returns `z` (list of latents) and `freqs_cis`.
        # Each `z` element is the result of `crop_each_layer`?
        # Yes: `_z, cis = crop_each_layer(...)`
        
        # HOWEVER, SD3 VAE is standard AutoencoderKL. It outputs a spatial map.
        # CLD2 `encode_target_latents` (I validated this in my head):
        # Likely repeats the latent L times.
        # Let's assume we need to save the boxes for Model usage (if Model uses boxes for RoPE or masking).
        # We'll save boxes as a flat tensor and a shape info or just rely on the fact mask is there.
        # For safety, let's save boxes as flattened tensor.
        
        # box tensor: [L, 4]
        box_tensor = torch.tensor(layer_boxes, dtype=torch.int32)
        tensors["boxes"] = box_tensor
        
        safetensors.torch.save_file(tensors, save_path)

    print("Caching Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sd3_path", type=str, required=True)
    parser.add_argument("--dit_path", type=str, required=True)
    parser.add_argument("--llm_path", type=str, required=True)
    parser.add_argument("--llm_processor_path", type=str, required=True)
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    
    args = parser.parse_args()
    cache_data(args)
