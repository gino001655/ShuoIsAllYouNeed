import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import SD3Transformer2DModel
from diffusers.models.attention_processor import AttentionProcessor

class CLD3Backbone(nn.Module):
    """
    CLD3 Backbone: SD3 Transformer with Deep Layer Injection.
    """
    def __init__(
        self,
        dit_path: str,
        dtype: torch.dtype,
        device: torch.device,
        max_layer_num: int = 64,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # 1. Load SD3 Transformer
        self.transformer = SD3Transformer2DModel.from_pretrained(dit_path, torch_dtype=dtype).to(device)
        
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            
        # 2. Deep Layer Injection System
        # We create a learnable embedding for Layers that will be injected into every block
        self.max_layer_num = max_layer_num
        
        # Hidden size of SD3 is usually 1536 (small/medium?) or something else.
        # Let's dynamically get it.
        # config.patch_size is 2. config.sample_size is 128 (latent).
        # We need `joint_attention_dim` or similar.
        # SD3 has `transformer.config.joint_attention_dim` (4096) and `inner_dim`?
        # Actually `hidden_size` is the main dim.
        
        # We assume standard SD3.5 Medium/Large
        hidden_dim = self.transformer.config.in_channels * (self.transformer.config.patch_size ** 2) # Just heuristics? No.
        # Better: check the first block.
        # But we can read from config.
        # SD3 config usually has `joint_attention_dim` (context) and `caption_projection_dim`.
        # The main hidden state dim is often implicit in `patch_embed`. 
        # But actually for DiT, `sample_size` * `patch_size`...
        # Wait, Diffusers SD3 config has `num_attention_heads` * `attention_head_dim`.
        self.hidden_size = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        
        self.layer_embed = nn.Embedding(max_layer_num, self.hidden_size)
        nn.init.normal_(self.layer_embed.weight, std=0.02)
        
        # 3. Inject into Blocks
        # We will wrap the `forward` of blocks OR use the `joint_attention_kwargs` trick provided by Diffusers?
        # Diffusers pipelines support `joint_attention_kwargs`.
        # However, we want to Add to Hidden States (ControlNet style), not just cross-attention.
        # SD3 blocks are `SD3Transformer2DModel` -> `transformer_blocks`.
        # We can implement a "Processor" or just wrap the blocks.
        # Monkey-patching `forward` is risky but effective for "Deep Injection".
        
        # Approach: "Side-car" or "Direct Add"?
        # Direct Add: hidden_states = hidden_states + layer_emb
        # We need to broadcast layer_emb [Batch, 1, Dim] to [Batch, Seq, Dim].
        
        # We will attach a hook or wrapper to each block.
        for i, block in enumerate(self.transformer.transformer_blocks):
            block.original_forward = block.forward
            block.forward = self._make_injected_forward(block)
            
        # Also single blocks?
        # Usually joint blocks are the main ones. Single blocks refine.
        # Let's inject into single blocks too for max control.
        # But single blocks signature might differ.
        if hasattr(self.transformer, "single_transformer_blocks"):
             for i, block in enumerate(self.transformer.single_transformer_blocks):
                block.original_forward = block.forward
                block.forward = self._make_injected_forward(block)

    def _make_injected_forward(self, block):
        def forward(*args, **kwargs):
            # We expect `layer_emb` to be passed in `joint_attention_kwargs` or checking a global context?
            # Diffusers `forward` passes `joint_attention_kwargs`.
            
            # The signature of SD3 block forward:
            # (hidden_states, encoder_hidden_states, temb, ...)
            
            hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            
            # Extract layer_emb from kwargs or a temporary storage
            # Since we can't easily change the signature called by the main Transformer loop without copying the WHOLE loop,
            # We have 2 options:
            # A) Copy the main `SD3Transformer2DModel.forward` loop (Robust but verbose)
            # B) Store `current_layer_embed` in `self` and access it (Stateful, not thread-safe but ok for training).
            
            if hasattr(self, "current_layer_embed") and self.current_layer_embed is not None:
                # Add Layer Embedding
                # hidden_states: [B, L, D]
                # current_layer_embed: [B, 1, D]
                hidden_states = hidden_states + self.current_layer_embed
                
                # Update args/kwargs
                if len(args) > 0:
                    args = (hidden_states,) + args[1:]
                else:
                    kwargs["hidden_states"] = hidden_states
            
            return block.original_forward(*args, **kwargs)
        return forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_ids: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
    ):
        """
        Forward pass with Deep Layer Injection.
        """
        batch_size = hidden_states.shape[0]
        
        # 1. Compute Layer Embeddings
        # layer_ids: [B] or [B, L] ?
        # In our case, flattening [B*Layers]
        # layer_ids input should align with hidden_states batch dim.
        
        emb = self.layer_embed(layer_ids) # [B, Dim]
        self.current_layer_embed = emb.unsqueeze(1) # [B, 1, Dim]
        
        # 2. Forward SD3
        output = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            return_dict=False
        )[0]
        
        # 3. Cleanup
        self.current_layer_embed = None
        
        return output

    def get_trainable_params(self):
        # Only Train Layer Embed (and maybe LoRA if we load it)
        return list(self.layer_embed.parameters()) 
        # Note: If user wants LoRA, they inject it into `self.transformer` standard way
        # and we include those params.
