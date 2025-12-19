import os
import argparse
import random
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import safetensors.torch
from prodigyopt import Prodigy

from CLD3.models.cld3_backbone import CLD3Backbone
from CLD3.tools.tools import seed_everything

class CachedDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = sorted(glob(os.path.join(cache_dir, "*.safetensors")))
        print(f"Found {len(self.files)} cached samples.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Determine cached file path
        path = self.files[idx]
        data = safetensors.torch.load_file(path)
        return data

def collate_fn(batch):
    # Batch is List of dicts.
    # Logic:
    # 1. We receive [L, C, H, W] latents.
    # 2. We flatten them into a huge batch [B*L, C, H, W].
    # 3. We create corresponding Layer IDs.
    
    # Extract
    latents_list = []
    layer_ids_list = []
    prompt_list = []
    pooled_list = []
    mask_list = []
    
    for sample in batch:
        x0 = sample["latents"] # [L, C, H, W]
        L = x0.shape[0]
        
        # Layer IDs
        ids = torch.arange(L, dtype=torch.long)
        
        # Text Cond (Repeated per layer)
        prompt = sample["prompt_embeds"].unsqueeze(0).repeat(L, 1, 1) # [L, Seq, Dim]
        pooled = sample["pooled_embeds"].unsqueeze(0).repeat(L, 1) # [L, Dim]
        
        # Mask
        mask = sample["mask"] # [L, 1, H, W]
        
        latents_list.append(x0)
        layer_ids_list.append(ids)
        prompt_list.append(prompt)
        pooled_list.append(pooled)
        mask_list.append(mask)

    # Concat
    latents = torch.cat(latents_list, dim=0)
    layer_ids = torch.cat(layer_ids_list, dim=0)
    prompts = torch.cat(prompt_list, dim=0)
    pooleds = torch.cat(pooled_list, dim=0)
    masks = torch.cat(mask_list, dim=0)
    
    return {
        "latents": latents, # [BS_Total, C, H, W]
        "layer_ids": layer_ids, # [BS_Total]
        "prompts": prompts,
        "pooleds": pooleds,
        "masks": masks
    }

def train(args):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    # 1. Model
    print("Loading CLD3 Backbone...")
    model = CLD3Backbone(
        dit_path=args.dit_path,
        dtype=dtype,
        device=device,
        gradient_checkpointing=True,
        max_layer_num=64
    )
    
    # Optimize: Only train Layer Embedding and LoRA (if enabled)
    # CLD3Backbone handles this in get_trainable_params
    params = model.get_trainable_params()
    
    optimizer = Prodigy(
        params,
        lr=1.0,
        weight_decay=0.01
    )
    
    # 2. Dataset
    dataset = CachedDataset(args.cache_dir)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, # Samples per batch (each sample has L layers)
        shuffle=True, 
        num_workers=4, 
        collate_fn=collate_fn
    )
    
    print(f"Start Training CLD3 (Deep Injection)... Steps: {args.max_steps}")
    
    model.train()
    step = 0
    pbar = tqdm(total=args.max_steps)
    
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps: break
            
            # Move to device
            latents = batch["latents"].to(device, dtype=dtype) # x0
            layer_ids = batch["layer_ids"].to(device)
            prompts = batch["prompts"].to(device, dtype=dtype)
            pooleds = batch["pooleds"].to(device, dtype=dtype)
            mask = batch["masks"].to(device, dtype=dtype)
            
            # Noise
            noise = torch.randn_like(latents)
            bs = latents.shape[0]
            
            # Timestep (Uniform or Logit-Normal? SD3 uses Flow Match)
            # Simple u [0, 1000]
            t = torch.randint(0, 1000, (bs,), device=device).long()
            
            # Add Noise (Flow Matching: t=0 is data, t=1 is noise)
            # t_norm [0,1]
            t_norm = t.float() / 1000.0
            t_norm = t_norm.view(bs, 1, 1, 1).to(dtype)
            
            # xt = (1 - t) * x0 + t * x1
            xt = (1 - t_norm) * latents + t_norm * noise
            
            # v_star = x1 - x0
            v_star = noise - latents
            
            # Prediction
            # Note: We pass flat batch. The Model handles Deep Injection via layer_ids.
            v_pred = model(
                hidden_states=xt,
                layer_ids=layer_ids,
                timestep=t, 
                encoder_hidden_states=prompts,
                pooled_projections=pooleds
            )
            
            # Loss (Masked MSE)
            mse = (v_pred - v_star) ** 2
            # mask [BS, 1, H, W]
            loss = (mse * mask).sum() / (mask.sum() + 1e-8)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())
            
            if step % args.save_every == 0:
                save_path = os.path.join(args.output_dir, f"cld3_step_{step}.pt")
                torch.save(model.layer_embed.state_dict(), save_path)
                
    print("Training Finished.")
    torch.save(model.layer_embed.state_dict(), os.path.join(args.output_dir, "cld3_final.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dit_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()
    train(args)
