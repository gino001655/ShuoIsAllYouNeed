import os
from typing import Optional

import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import AutoProcessor

# CLD3: Use vendored UAE
from CLD3.uae.models.modeling_longcontext import Qwen2_5_VLWithLongContext


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
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, cfg.llm_lora_path)
                self.model.set_adapter("default")
            except Exception:
                pass

        # Processor provides tokenizer + chat template; UAE uses Qwen's processor.
        if cfg.processor_path is None:
            raise ValueError("processor_path is required.")
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
