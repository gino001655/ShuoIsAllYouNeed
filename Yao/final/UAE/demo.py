# %%
import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from uae.models.modeling_longcontext import Qwen2_5_VLWithLongContext
from transformers import (
    AutoProcessor,
    CLIPTextModelWithProjection,
    T5EncoderModel,
    CLIPTokenizer,
    T5TokenizerFast,
)
import os
import torch.nn as nn
from diffusers import SD3Transformer2DModel, AutoencoderKL
from omegaconf import OmegaConf
from uae.utils.denoiser_prompt_embeds import encode_prompt
from qwen_vl_utils import process_vision_info
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel


def set_lora(target_model, target_modules, rank, alpha, lora_path=None):
    """
    Configure LoRA (Low-Rank Adaptation) for a target model.
    
    Args:
        target_model: The model to apply LoRA to
        target_modules: Target modules to apply LoRA (e.g., 'all-linear')
        rank: LoRA rank parameter
        alpha: LoRA alpha parameter
        lora_path: Optional path to pre-trained LoRA weights
        
    Returns:
        Model with LoRA configuration applied
    """
    from peft import get_peft_model, PeftModel, LoraConfig
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    if lora_path is not None:
        target_model = PeftModel.from_pretrained(target_model, lora_path)
        target_model.set_adapter("default")
    else:
        target_model = get_peft_model(target_model, lora_config)

    return target_model


def load_models(model_cfg, device="cuda"):
    """
    Load and initialize all required models for the pipeline.
    
    Args:
        model_cfg: Path of model
        checkpoint_path: Path to model checkpoint
        device: Device to load models on (default: "cuda")
        
    Returns:
        tuple: (uae_model, processor, pipe, pooled_prompt_embeds)
    """
    uae_model = Qwen2_5_VLWithLongContext.from_pretrained(
        model_cfg["llm"],
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    uae_model.model = set_lora(
        target_model=uae_model.model,
        target_modules='all-linear',
        rank=32,
        alpha=64,
    )

    state_dict = {}
    from safetensors.torch import load_file
    base_path = model_cfg["llm_lora"]
    
    for i in range(1, 4):
        safe_tensor_path = os.path.join(base_path, f'model-0000{i}-of-00003.safetensors')
        weights = load_file(safe_tensor_path)
        state_dict.update(weights)

    uae_model.load_state_dict(state_dict, strict=False)

    processor = AutoProcessor.from_pretrained(
        model_cfg["llm_processor"],
    )

    transformer = SD3Transformer2DModel.from_pretrained(
        model_cfg["dit"],
        torch_dtype=torch.bfloat16,
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        model_cfg['SD3'],
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to(device)

    text_encoder_cls_one = CLIPTextModelWithProjection.from_pretrained(
        model_cfg['SD3'],
        torch_dtype=torch.bfloat16,
        subfolder="text_encoder",
    )
    text_encoder_cls_two = CLIPTextModelWithProjection.from_pretrained(
        model_cfg['SD3'],
        torch_dtype=torch.bfloat16,
        subfolder="text_encoder_2",
    )
    text_encoder_cls_three = T5EncoderModel.from_pretrained(
        model_cfg['SD3'],
        torch_dtype=torch.bfloat16,
        subfolder="text_encoder_3",
    )
    text_encoder_cls_one.to(device)
    text_encoder_cls_two.to(device)
    text_encoder_cls_three.to(device)

    tokenizer_one = CLIPTokenizer.from_pretrained(
        model_cfg['SD3'],
        subfolder="tokenizer",
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        model_cfg['SD3'],
        subfolder="tokenizer_2",
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        model_cfg['SD3'],
        subfolder="tokenizer_3",
    )

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_cfg['SD3'],
        vae=vae,
        transformer=transformer,
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        torch_dtype=torch.bfloat16,
    ).to(device)

    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, model_cfg['dit_lora'])

    text_encoders = [
        text_encoder_cls_one,
        text_encoder_cls_two,
        text_encoder_cls_three,
    ]
    tokenizers = [
        tokenizer_one,
        tokenizer_two,
        tokenizer_three,
    ]
    _, _, pooled_prompt_embeds = encode_prompt(
        text_encoders,
        tokenizers,
        prompt="",
        max_sequence_length=512,
        device=device,
        num_images_per_prompt=1,
    )

    return uae_model, processor, pipe, pooled_prompt_embeds


def get_prompt_embeds(prompt, uae_model, processor, device):
    """
    Generate prompt embeddings for both positive and negative prompts.
    
    Args:
        prompt: Text prompt for image generation
        uae_model: The vision-language model
        processor: Text processor
        device: Computing device
        
    Returns:
        tuple: (negative_prompt_embeds, positive_prompt_embeds)
    """
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

    negative_input_ids = get_input_ids("Generate a random, low quality, ugly, blur, bad and anime, cartoon image.")
    positive_input_ids = get_input_ids(prompt)

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [negative_input_ids, positive_input_ids],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id,
    )
    attention_mask = input_ids.ne(processor.tokenizer.pad_token_id)
    
    prompt_embeds = uae_model.get_projected_embeddings(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    negative_prompt_embeds, positive_prompt_embeds = prompt_embeds.chunk(2, dim=0)
    return negative_prompt_embeds, positive_prompt_embeds


@torch.no_grad()
def gen_prompt(input_img_dir, processor, llm_model, device):
    """
    Generate a textual description from an input image using the vision-language model.
    
    Args:
        input_img_dir: Path to the input image
        processor: Image and text processor
        llm_model: Vision-language model for caption generation
        device: Computing device
        
    Returns:
        str: Generated text description of the image
    """
    question = """You are an expert vision-language model.  
Your task is: Given an input image, generate a **textual description** of the image. If there is text in the image, transcribe it inside double quotes "".  
Now, carefully analyze the input image and output the full description. """
    
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": question},
                {"type": "image", "image": input_img_dir},
            ]
        },
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    
    model_inputs = processor(
        text=[text], 
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt", 
        padding=True
    ).to(device)

    generated_ids = llm_model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
        )
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(generated_text)
    return generated_text


@torch.no_grad()
def gen_image(text, processor, llm_model, pipe, pooled_prompt_embeds, device='cuda', seed=None):
    """
    Generate an image from text description using the diffusion pipeline.
    
    Args:
        text: Text prompt for image generation
        processor: Text processor
        llm_model: Vision-language model for embedding generation
        pipe: Stable Diffusion pipeline
        pooled_prompt_embeds: Pre-computed pooled prompt embeddings
        device: Computing device (default: 'cuda')
        seed: Random seed for generation (default: None, random)
        
    Returns:
        PIL.Image: Generated image
    """
    if seed is None:
        seed = np.random.randint(0, 10000)
    generator = torch.Generator(device=device).manual_seed(seed)
    negative_prompt_embeds, prompt_embeds = get_prompt_embeds(prompt=text, uae_model=llm_model, processor=processor, device=device)
    image = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=pooled_prompt_embeds,
            height=1024,
            width=1024,
            num_inference_steps=40,
            guidance_scale=5.0,
            generator=generator,
        ).images[0]
    return image


def generate_from_input(input_data, output_path=None, seed=None):
    """
    Unified generation function that accepts either image path or text input and generates final image.
    
    Args:
        input_data: Image path (str) or text description (str)
        output_path: Output image path, if None then don't save (default: None)
        seed: Random seed, if None then randomly generate (default: None)
    
    Returns:
        PIL.Image: Generated image
    """
    if seed is None:
        seed = np.random.randint(0, 10000)
    
    if os.path.exists(input_data):
        text = gen_prompt(input_img_dir=input_data, processor=processor, llm_model=uae_model, device=device)
    else:
        text = input_data
    
    generated_image = gen_image(text, processor, uae_model, pipe, pooled_prompt_embeds, device=device, seed=seed)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        generated_image.save(output_path)
        print(f"Image saved to: {output_path}")
    
    return generated_image


# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_text', type=str, default=None, help='Input text for image generation')
parser.add_argument('--input_img', type=str, default=None, help='Input image path for image-to-text or image-to-image generation')
parser.add_argument('--output_path', type=str, default=None, help='Output path for generated image')
parser.add_argument('--prompt_only', action='store_true', help='Only generate prompt from input image without generating new image')
args = parser.parse_args()

# Validation logic
if args.prompt_only:
    assert args.input_img is not None, "When using --prompt_only, --input_img must be provided."
    assert args.input_text is None, "When using --prompt_only, --input_text should not be provided."
else:
    assert args.input_text is not None or args.input_img is not None, "Either --input_text or --input_img must be provided."
    assert not (args.input_text and args.input_img), "Only one of --input_text or --input_img should be provided."

device = "cuda:0"

model_cfg = {
    "pretrained_denoiser_name_or_path": "./Checkpoints/SD3_model",
    "dit": "./Checkpoints/dit",
    "dit_lora": "./Checkpoints/dit_lora",
    "llm": "./Checkpoints/llm",
    "llm_lora": "./Checkpoints/llm_lora",
    "llm_processor": "./Checkpoints/llm_model_preprocessor"
}

uae_model, processor, pipe, pooled_prompt_embeds = load_models(model_cfg, device)

# Handle prompt_only mode
if args.prompt_only:
    print("=== Prompt Generation Mode ===")
    print(f"Input image: {args.input_img}")
    generated_prompt = gen_prompt(input_img_dir=args.input_img, processor=processor, llm_model=uae_model, device=device)
    print(f"Generated prompt: {generated_prompt}")
else:
    # Original functionality for image generation
    if args.input_text:
        input_data = args.input_text
    else:
        input_data = args.input_img
    
    result = generate_from_input(input_data, args.output_path, seed=42)
