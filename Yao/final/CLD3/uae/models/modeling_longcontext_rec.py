from torch._tensor import Tensor


from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration


class ComposedModelLongContext(nn.Module):
    def __init__(self, llm_model, denoiser, processor, sd_pipe=None, clip_score=True):
        super().__init__()
        self.llm_model = llm_model
        self.denoiser = denoiser
        from .sd_model import SdWrapper
        self.sd_pipe = SdWrapper(sd_pipe)
        
        # 
        self.processor = processor
        self.pooled_prompt_embeds = None
        self.negative_pooled_prompt_embeds = None

        # clip_score
        if clip_score:
            print("Using ClipScore for image quality evaluation. Current Model: clip-vit-large-patch14")
            from .clip_score import ClipScore
            self.clip_score = ClipScore(clip_name='clip-vit-large-patch14')
        else:
            print("Not using ClipScore for image quality evaluation.")
            self.clip_score = None

    def get_clip_transform(self):
        return self.clip_score.get_clip_transform()

    # NOTE:
    def set_pooled_prompt(self, pooled_prompt_embeds, negative_pooled_prompt_embeds=None):
        """
        Set the pooled prompt embeddings for the model.
        This is used to provide a fixed context for the model.
        """
        self.pooled_prompt_embeds = pooled_prompt_embeds
        if negative_pooled_prompt_embeds is not None:
            self.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        else:
            self.negative_pooled_prompt_embeds = None
        print("Pooled prompt embeddings set.")

    # NOTE:
    def get_neg_prompt(self, neg_prompt="Generate a random, low quality, ugly, blur, bad and anime, cartoon image."):
        processor = self.processor
        device = self.llm_model.device
        # Get the negative input ids
        def get_input_ids(prompt: str):
            messages = [
                {
                    "role": "generate",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = processor.tokenizer.encode(
                text,
                return_tensors="pt",
            )
            input_ids = input_ids.to(device)
            return input_ids[0]
        # Get the negative input ids
        negative_input_ids = get_input_ids(neg_prompt)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [negative_input_ids],
            batch_first=True,
            padding_value=processor.tokenizer.pad_token_id,
            # padding_side="right",
        )
        inputs = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(processor.tokenizer.pad_token_id),
        }
        prompt_embeds, _ = self.llm_model.get_projected_embeddings(
            inputs=inputs,
        )
        negative_prompt_embeds = prompt_embeds
        
        return negative_prompt_embeds

    # NOTE:
    def sd_gen(self, prompt_embeds, negative_prompt_embeds, 
               pooled_prompt_embeds=None, negative_pooled_prompt_embeds=None,
               max_sequence_length=256, height=512, width=512, num_inference_steps=30, guidance_scale=5.0,):
        
        self.sd_pipe.sd_pipe.set_progress_bar_config(disable=True)
        
        # negative_prompt_embeds = self.get_neg_prompt()
        assert self.pooled_prompt_embeds is not None, "Pooled prompt embeddings must be set before generating images."
        assert self.negative_pooled_prompt_embeds is not None, "Negative prompt embeddings must be set before generating images."
        
        if pooled_prompt_embeds is None:
            pooled_prompt_embeds = self.pooled_prompt_embeds.clone()
        if negative_pooled_prompt_embeds is None:
            negative_pooled_prompt_embeds = self.negative_pooled_prompt_embeds.clone()

        # Generate the image using the SD pipeline
        sd_output = self.sd_pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="pt",
            return_dict=False,
        )[0]
        return sd_output

    def forward(
        self,
        # NOTE:
        inputs,
        labels = None,
        pixel_values=None,
        naive_images=None,
    ):  
    
        encoder_hidden_states, _ = self.llm_model.get_projected_embeddings(
            # NOTE:
            inputs=inputs,
            labels=labels,
        )
        
        # NOTE: 取出 answer部分对应的hidden states (仅支持image与question对应token总和在batch中是一致的情况，即图像尺寸一致+问题长度一致)
        mask = labels != -100
        pos_encoder_hidden_states = encoder_hidden_states[mask]
        pos_encoder_hidden_states = pos_encoder_hidden_states.reshape(
            (encoder_hidden_states.shape[0]) - 1, -1, encoder_hidden_states.shape[-1]
        )
        
        # NOTE: batch中最后一个张量是负样本的encoder_hidden_states
        neg_encoder_hidden_states = encoder_hidden_states[-1:, :pos_encoder_hidden_states.shape[1], :]
        
        # NOTE:
        # reconstruct the image
        if self.sd_pipe is not None and pixel_values is not None:
            rec_img = self.sd_gen(
                    prompt_embeds=pos_encoder_hidden_states,
                    negative_prompt_embeds=neg_encoder_hidden_states.repeat(pos_encoder_hidden_states.shape[0], 1, 1),
                    pooled_prompt_embeds=self.pooled_prompt_embeds.clone(),
                    negative_pooled_prompt_embeds=self.negative_pooled_prompt_embeds.clone(),
                    height=512,
                    width=512,
                    num_inference_steps=10,
                    guidance_scale=5.0,
                )
            
            if self.clip_score is not None:
                clip_score = self.clip_score(images=rec_img, gt_images=naive_images)
        else:
            mse_loss = torch.tensor(0.0, device=self.llm_model.device)

        # NOTE:
        loss = clip_score
        print(f"clip_score: {clip_score}, total_loss: {loss}")

        return loss



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
        # NOTE:
        inputs,
        # input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        # NOTE:
        labels = None,
        **kwargs
    ):
        """
        Get embeddings WITH projection applied - specifically for SD3 image generation.
        This is separate from the forward method to avoid confusion.
        """
        # Get raw outputs from Qwen model
        try:
            outputs = super().forward(
                **inputs,
                labels=None,
                # input_ids=input_ids,
                # attention_mask=attention_mask,
                # pixel_values=pixel_values,
                # image_grid_thw=image_grid_thw,
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
        # loss_vqa = outputs.loss
        # print(f"✅ Applied projection for SD3: {raw_hidden_states.shape} -> {projected_states.shape}")
        
        return projected_states, None

        