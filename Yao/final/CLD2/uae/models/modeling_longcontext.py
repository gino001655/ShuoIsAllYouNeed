from torch._tensor import Tensor


from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration


class ComposedModelLongContext(nn.Module):
    def __init__(self, llm_model, denoiser):
        super().__init__()
        self.llm_model = llm_model
        self.denoiser = denoiser

    def forward(
        self,
        # semantic encoder
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        # denoiser
        target: torch.Tensor,
        weighting: torch.Tensor,
        timesteps: torch.LongTensor,
        pooled_prompt_embeds: torch.Tensor,
        noisy_model_input: torch.Tensor,
        # image inputs (optional)
        image_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_vit: Optional[torch.Tensor] = None,
    ):
        encoder_hidden_states = self.llm_model.get_projected_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            pixel_values=pixel_values_vit,
        )

        model_pred = self.denoiser(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

        loss_diff = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                target.shape[0], -1
            ),
            1,
        ).mean()

        return loss_diff


class Qwen2_5_VLWithLongContext(Qwen2_5_VLForConditionalGeneration):
    """
    Qwen2.5-VL with projector for SD3 integration
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projector = nn.Sequential(
            nn.Linear(self.config.hidden_size, 4096 * 2),
            nn.GELU(),
            nn.Linear(4096 * 2, 4096),
        )
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        output_last_hidden_state: bool = False,
        **kwargs
    ):
        """
        Forward pass for the composed model.
        
        IMPORTANT: This method should NOT apply projection by default.
        The default behavior is normal Qwen text generation.
        For SD3 image generation, use get_projected_embeddings() method.
        """
        
        # Handle inputs_embeds case - never apply projection by default
        if inputs_embeds is not None:
            # print(f"Processing inputs_embeds directly with shape: {inputs_embeds.shape}")
            # print("⚠️ Using original embeddings (no projection applied for text generation)")
            return inputs_embeds
        
        # Filter out our custom parameters and potential conflicts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['use_projection', 'use_projection_for_sd3', 'apply_projection']}
        
        # Always call parent Qwen model normally
        try:
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **filtered_kwargs
            )

            # print("✅ Successfully got outputs from Qwen model (normal text generation)")
            return outputs
            
        except Exception as e:
            print(f"❌ Error in Qwen forward: {e}")
            raise e
    
    def get_projected_embeddings(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
        Get embeddings WITH projection applied - specifically for SD3 image generation.
        This is separate from the forward method to avoid confusion.
        """
        # Get raw outputs from Qwen model
        try:
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                return_dict=True,
                **kwargs
            )
            # print("✅ Got raw outputs from Qwen model")
        except Exception as e:
            print(f"❌ Error in Qwen forward: {e}")
            raise e
    
        # Extract hidden states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            # For Qwen2.5-VL, use the last layer from hidden_states tuple
            raw_hidden_states = outputs.hidden_states[-1]
            # print(f"✅ Using hidden_states[-1] with shape: {raw_hidden_states.shape}")
        elif hasattr(outputs, 'last_hidden_state'):
            # Fallback for other models that have last_hidden_state
            raw_hidden_states = outputs.last_hidden_state
            # print(f"✅ Using last_hidden_state with shape: {raw_hidden_states.shape}")
        else:
            raise ValueError("No hidden_states or last_hidden_state found in outputs")
        
        # print(f"Raw hidden states shape: {raw_hidden_states.shape}")
        
        # Apply projection for SD3
        projected_states = self.projector(raw_hidden_states)
        # print(f"✅ Applied projection for SD3: {raw_hidden_states.shape} -> {projected_states.shape}")
        
        return projected_states