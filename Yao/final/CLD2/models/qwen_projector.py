import os
from typing import Optional

import torch

from transformers import AutoProcessor

# UAE provides Qwen2_5_VLWithLongContext with a projector to 4096 dims.
from uae.models.modeling_longcontext import Qwen2_5_VLWithLongContext


class QwenProjector:
    """
    Text -> projected embeddings for SD3 conditioning, following UAE's design.

    We keep the model frozen by default. For VRAM saving, you can load in 4-bit.
    """

    def __init__(
        self,
        llm_path: str,
        llm_processor_path: str,
        llm_lora_path: Optional[str],
        load_in_4bit: bool,
        attn_implementation: str,
        device: torch.device,
        dtype: torch.dtype,
        gradient_checkpointing: bool = True,
    ):
        self.device = device
        self.dtype = dtype

        quant_cfg = None
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
            except Exception as e:
                raise RuntimeError("Requested 4-bit load, but transformers/bitsandbytes config is unavailable.") from e

        kwargs = dict(torch_dtype=dtype)
        if quant_cfg is not None:
            kwargs["quantization_config"] = quant_cfg
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = None

        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        self.model = Qwen2_5_VLWithLongContext.from_pretrained(llm_path, **kwargs)
        if quant_cfg is None:
            self.model = self.model.to(device)

        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        # For embedding extraction we do NOT want KV cache (saves VRAM).
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        # Load LoRA if provided (UAE uses safetensors shards, but PeftModel path also works)
        if llm_lora_path is not None:
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, llm_lora_path)
                self.model.set_adapter("default")
            except Exception:
                # If LoRA dir isn't a peft adapter folder, we leave it to user to adapt.
                pass

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.processor = AutoProcessor.from_pretrained(llm_processor_path)

    @torch.no_grad()
    def caption_to_prompt_embeds(self, caption: str) -> torch.Tensor:
        """
        Convert a plain caption string into projected embeddings [1, seq, 4096].

        UAE uses a chat template; we follow a minimal version that is stable.
        """
        messages = [
            {
                "role": "generate",
                "content": [{"type": "text", "text": caption}],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.processor.tokenizer.encode(text, return_tensors="pt").to(self.device)
        attn_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

        embeds = self.model.get_projected_embeddings(input_ids=input_ids, attention_mask=attn_mask)
        return embeds.to(dtype=self.dtype)

"""
Qwen2.5-VL projector wrapper for CLD2.

We follow UAE's idea:
- Use Qwen2.5-VL as a "semantic encoder" and project its hidden states to SD3's
  cross-attention dimension (4096).

This module focuses on *text -> projected embeddings* for training CLD2.
We intentionally do NOT run image -> caption in the training loop (too expensive).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class QwenProjectorConfig:
    llm_path: str
    llm_lora_path: Optional[str] = None
    processor_path: Optional[str] = None

    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    # Memory switches
    gradient_checkpointing: bool = True
    use_flash_attention_2: bool = False  # requires flash-attn2 + transformers support
    load_in_4bit: bool = False  # bitsandbytes


class QwenProjector(nn.Module):
    """
    Produce projected embeddings from a text prompt:
      encoder_hidden_states: [B, seq, 4096]
    """

    def __init__(self, cfg: QwenProjectorConfig):
        super().__init__()
        self.cfg = cfg

        try:
            from transformers import AutoProcessor
        except Exception as e:  # pragma: no cover
            raise RuntimeError("transformers is required for QwenProjector.") from e

        # Reuse UAE's model implementation (keeps the projector definition consistent).
        # In CLD2, we vendor UAE's `uae/` package under `CLD2/uae/`.
        from uae.models.modeling_longcontext import Qwen2_5_VLWithLongContext

        attn_impl = "flash_attention_2" if cfg.use_flash_attention_2 else "sdpa"

        model_kwargs = {
            "torch_dtype": cfg.dtype,
        }

        if cfg.load_in_4bit:
            # Optional 4-bit quant for VRAM saving (text encoder only).
            model_kwargs.update(
                {
                    "load_in_4bit": True,
                    "device_map": "auto",
                    "attn_implementation": attn_impl,
                }
            )
            self.model = Qwen2_5_VLWithLongContext.from_pretrained(cfg.llm_path, **model_kwargs)
        else:
            model_kwargs.update({"attn_implementation": attn_impl})
            self.model = Qwen2_5_VLWithLongContext.from_pretrained(cfg.llm_path, **model_kwargs).to(cfg.device)

        if cfg.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            # For decoder-only LLMs, turning off cache is important when checkpointing.
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False

        # Load LoRA if provided (UAE style: weight shards or a PEFT adapter).
        if cfg.llm_lora_path:
            # Prefer PEFT adapter path if it exists; otherwise allow users to load shards manually.
            try:
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(self.model, cfg.llm_lora_path)
                self.model.set_adapter("default")
            except Exception:
                # If LoRA isn't in PEFT format, users can load state_dict in their own script.
                pass

        # Processor provides tokenizer + chat template; UAE uses Qwen's processor.
        if cfg.processor_path is None:
            raise ValueError("processor_path is required (e.g., UAE Checkpoints/llm_model_preprocessor).")
        self.processor = AutoProcessor.from_pretrained(cfg.processor_path)

        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def encode_text(self, prompt: str, max_new_tokens: int = 0) -> torch.Tensor:
        """
        Encode a plain text prompt into SD3-compatible projected embeddings.

        Returns:
          encoder_hidden_states: [1, seq, 4096]
        """
        # UAE uses a chat template with role "generate" to produce prompt embeddings.
        messages = [
            {
                "role": "generate",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.processor.tokenizer.encode(text, return_tensors="pt").to(self.cfg.device)
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

        embeds = self.model.get_projected_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return embeds


