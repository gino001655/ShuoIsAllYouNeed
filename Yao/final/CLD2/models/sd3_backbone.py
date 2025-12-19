import os
from typing import List, Optional

import torch
import torch.nn as nn

from diffusers import SD3Transformer2DModel, AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

from uae.utils.denoiser_prompt_embeds import encode_prompt as sd3_encode_prompt


class SD3Backbone(nn.Module):
    """
    SD3/SD3.5 backbone wrapper for CLD2 training.

    What it provides:
    - SD3Transformer2DModel denoiser (optionally with LoRA)
    - SD3 VAE for encoding target images -> latents
    - SD3 scheduler (FlowMatch Euler) for timesteps sampling
    - UAE-style `pooled_prompt_embeds` computed once from empty prompt, then CLIP/T5 freed (saves VRAM)
    - Trainable `layer_embed` (Phase-1 layer-id embedding in latent channel space)
    """

    def __init__(
        self,
        sd3_model_path: str,
        dit_path: str,
        dit_lora_path: Optional[str],
        device: torch.device,
        dtype: torch.dtype,
        gradient_checkpointing: bool = True,
        max_layer_num: int = 64,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # denoiser (DiT)
        self.transformer = SD3Transformer2DModel.from_pretrained(dit_path, torch_dtype=dtype).to(device)
        if gradient_checkpointing and hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()
        elif gradient_checkpointing and hasattr(self.transformer, "gradient_checkpointing"):
            self.transformer.gradient_checkpointing = True

        # optionally load LoRA weights into denoiser
        if dit_lora_path is not None:
            if PeftModel is None:
                raise RuntimeError("peft is required to load LoRA weights for SD3 transformer.")
            self.transformer = PeftModel.from_pretrained(self.transformer, dit_lora_path).to(device)
            self.transformer.set_adapter("default")

        # VAE from official SD3 model folder
        self.vae = AutoencoderKL.from_pretrained(sd3_model_path, subfolder="vae", torch_dtype=dtype).to(device)
        self.vae.requires_grad_(False)
        self.vae.eval()

        # Scheduler (fallback if subfolder missing)
        try:
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(sd3_model_path, subfolder="scheduler")
        except Exception:
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(sd3_model_path)

        # Compute pooled prompt embeds once (then free CLIP/T5 to save VRAM)
        pooled = self._compute_pooled_prompt_embeds(sd3_model_path, device=device, dtype=dtype)
        self.register_buffer("pooled_prompt_embeds", pooled, persistent=False)

        # Layer id embedding in latent channel space (Phase-1)
        latent_ch = int(getattr(self.vae.config, "latent_channels", 16))
        self.layer_embed = nn.Embedding(max_layer_num, latent_ch)
        nn.init.normal_(self.layer_embed.weight, std=0.02)

        self.transformer.train()
        self.layer_embed.train()

    @staticmethod
    @torch.no_grad()
    def _compute_pooled_prompt_embeds(sd3_model_path: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        UAE-style pooled_prompt_embeds computed from empty prompt.
        This requires loading SD3 text encoders briefly; we free them right away to save VRAM.
        """
        te1 = CLIPTextModelWithProjection.from_pretrained(sd3_model_path, subfolder="text_encoder", torch_dtype=dtype).to(device)
        te2 = CLIPTextModelWithProjection.from_pretrained(sd3_model_path, subfolder="text_encoder_2", torch_dtype=dtype).to(device)
        te3 = T5EncoderModel.from_pretrained(sd3_model_path, subfolder="text_encoder_3", torch_dtype=dtype).to(device)

        tok1 = CLIPTokenizer.from_pretrained(sd3_model_path, subfolder="tokenizer")
        tok2 = CLIPTokenizer.from_pretrained(sd3_model_path, subfolder="tokenizer_2")
        tok3 = T5TokenizerFast.from_pretrained(sd3_model_path, subfolder="tokenizer_3")

        _, _, pooled = sd3_encode_prompt(
            text_encoders=[te1, te2, te3],
            tokenizers=[tok1, tok2, tok3],
            prompt="",
            max_sequence_length=512,
            device=device,
            num_images_per_prompt=1,
        )
        pooled = pooled.to(device=device, dtype=dtype)

        # Free heavy text encoders
        del te1, te2, te3, tok1, tok2, tok3
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pooled

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        for n, p in self.transformer.named_parameters():
            if "lora" in n.lower():
                p.requires_grad_(True)
                params.append(p)
            else:
                p.requires_grad_(False)
        for p in self.layer_embed.parameters():
            p.requires_grad_(True)
            params.append(p)
        return params

    def add_layer_embedding(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Add layer id embedding directly to latent channels.
        x0: [B, L, C, H, W]
        """
        _, l, c, _, _ = x0.shape
        layer_ids = torch.arange(l, device=x0.device)
        emb = self.layer_embed(layer_ids).to(dtype=x0.dtype)  # [L, C_emb]
        if emb.shape[-1] != c:
            if emb.shape[-1] < c:
                emb = torch.nn.functional.pad(emb, (0, c - emb.shape[-1]))
            else:
                emb = emb[:, :c]
        return x0 + emb.view(1, l, c, 1, 1)

    def denoise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

    def save_lora_and_layer_embed(self, out_dir: str, step: int):
        os.makedirs(out_dir, exist_ok=True)
        if hasattr(self.transformer, "save_pretrained"):
            lora_dir = os.path.join(out_dir, f"sd3_lora_step_{step}")
            self.transformer.save_pretrained(lora_dir)

        torch.save(
            {"layer_embed": self.layer_embed.state_dict(), "step": step},
            os.path.join(out_dir, f"layer_embed_step_{step}.pth"),
        )
