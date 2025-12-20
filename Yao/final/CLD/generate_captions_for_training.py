#!/usr/bin/env python3
"""
为训练数据生成 LLaVA captions
基于 parquet 文件中的图片路径，生成 caption 映射文件
支持批处理和多 GPU 并行
"""

import argparse
import json
import sys
import os
from pathlib import Path
from tqdm import tqdm
from tqdm import tqdm
# import torch  <-- MOVED INSIDE FUNCTIONS TO PREVENT EARLY CUDA INIT
from PIL import Image
from PIL import Image
from datasets import load_dataset
import multiprocessing as mp
from typing import List, Tuple, Dict

# Add LLaVA directory to path
LLAVA_DIR = Path("/tmp2/b12902041/Gino/dlcv-fall-2025-final-project/third_party/llava")
if str(LLAVA_DIR) not in sys.path:
    sys.path.insert(0, str(LLAVA_DIR))

# CRITICAL: DO NOT import LLaVA or torch at module level!
# These imports trigger CUDA initialization before worker processes can set CUDA_VISIBLE_DEVICES
# All LLaVA imports are now done inside LLaVACaptioner.__init__()
LLAVA_AVAILABLE = True  # Assume available, will fail gracefully if not
# try:
#     from llava.constants import self.IMAGE_TOKEN_INDEX, self.DEFAULT_IMAGE_TOKEN, self.DEFAULT_IM_START_TOKEN, self.DEFAULT_IM_END_TOKEN, self.IMAGE_PLACEHOLDER
#     from llava.conversation import self.conv_templates, SeparatorStyle
#     from llava.model.builder import load_pretrained_model
#     from llava.utils import disable_torch_init
#     from llava.mm_utils import (
#         self.process_images,
#         self.tokenizer_image_token,
#         get_model_name_from_path,
#     )
#     LLAVA_AVAILABLE = True
# except ImportError as e:
#     print(f"❌ Failed to import LLaVA modules: {e}")
#     print(f"   Make sure you're in the correct conda environment (e.g., llava15)")
#     print(f"   and that LLaVA codebase is available at: {LLAVA_DIR}")
#     LLAVA_AVAILABLE = False


class LLaVACaptioner:
    """LLaVA captioner using the official LLaVA codebase."""
    
    def __init__(
        self,
        model_path: str = "liuhaotian/llava-v1.5-7b",
        model_base: str = None,
        device: str = "cuda",
        load_4bit: bool = True,
        load_8bit: bool = False,
        prompt: str = "Describe style, main subject, and especially the background of the whole image in one short sentence.",
        # prompt: str = "Precisely describe style, subjects, text, and the background of the whole image in simple sentences.",
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_p: float = None,
        num_beams: int = 1,
        use_device_map_auto: bool = True,
    ):
        # Import LLaVA and torch INSIDE __init__ to allow CUDA_VISIBLE_DEVICES to be set first
        import torch
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
        
        # Store as instance variables for use in other methods
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.IMAGE_PLACEHOLDER = IMAGE_PLACEHOLDER
        self.conv_templates = conv_templates
        self.SeparatorStyle = SeparatorStyle
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        
        self.device = torch.device(device) if isinstance(device, str) else device
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        
        # Disable torch init for faster loading
        disable_torch_init()
        
        # Get model name from path
        model_name = get_model_name_from_path(model_path)
        
        # Load model
        print(f"📥 Loading LLaVA model from {model_path}...")
        print(f"   Using prompt: \"{prompt}\"")
        
        # For multiprocessing worker, explicitly map to target device
        if use_device_map_auto:
            device_map = "auto" if device == "cuda" else {"": device}
        else:
            device_map = None
        
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=model_name,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            device_map=device_map,
            device=device,
        )
        
        # Check model device (4-bit models are already on correct device)
        try:
            model_device = next(self.model.parameters()).device
            print(f"   Model is on device: {model_device}")
        except:
            print(f"   Could not determine model device")
        
        # Determine conversation mode
        if "llama-2" in model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            self.conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"
        
        print(f"✅ LLaVA model loaded successfully (conv_mode: {self.conv_mode})")
    
    # @torch.inference_mode()  <-- REMOVED DECORATOR because torch is not imported at module level
    def generate_batch(self, images: List[Image.Image]) -> List[str]:
        """Generate captions for a batch of images - TRUE BATCHING."""
        import torch
        import re
        
        if not images:
            return []
        
        # Single image case
        if len(images) == 1:
            return [self.generate_single(images[0])]
        
        batch_size = len(images)
        
        # Prepare prompts for all images
        qs = self.prompt
        image_token_se = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN
        
        if self.IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(self.IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(self.IMAGE_PLACEHOLDER, self.DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = self.DEFAULT_IMAGE_TOKEN + "\n" + qs
        
        # Create conversation for each image
        convs = []
        prompt_texts = []
        for _ in images:
            conv = self.conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt_texts.append(conv.get_prompt())
            convs.append(conv)
        
        # Process all images
        image_sizes = [img.size for img in images]
        images_tensor = self.process_images(
            images,
            self.image_processor,
            self.model.config
        )
        
        # Move to correct device with correct dtype
        if isinstance(images_tensor, list):
            images_tensor = [image.to(device=self.device, dtype=torch.float16) for image in images_tensor]
        else:
            images_tensor = images_tensor.to(device=self.device, dtype=torch.float16)
        
        # Tokenize all prompts
        input_ids_list = []
        for prompt_text in prompt_texts:
            input_ids = self.tokenizer_image_token(
                prompt_text, 
                self.tokenizer, 
                self.IMAGE_TOKEN_INDEX, 
                return_tensors="pt"
            )
            # Ensure it's 2D
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids_list.append(input_ids)
        
        # Pad sequences to same length
        max_len = max(ids.shape[-1] for ids in input_ids_list)
        padded_input_ids = []
        attention_masks = []
        
        for input_ids in input_ids_list:
            current_len = input_ids.shape[-1]
            padding_length = max_len - current_len
            if padding_length > 0:
                # Pad with tokenizer pad token
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padded = torch.cat([
                    torch.full((1, padding_length), pad_token_id, dtype=input_ids.dtype),
                    input_ids
                ], dim=-1)
                # Attention mask: 0 for padding, 1 for actual tokens
                attention_mask = torch.cat([
                    torch.zeros((1, padding_length), dtype=torch.long),
                    torch.ones((1, current_len), dtype=torch.long)
                ], dim=-1)
            else:
                padded = input_ids
                attention_mask = torch.ones((1, current_len), dtype=torch.long)
            
            padded_input_ids.append(padded)
            attention_masks.append(attention_mask)
        
        # Stack into batch
        input_ids_batch = torch.cat(padded_input_ids, dim=0).to(self.device)
        attention_mask_batch = torch.cat(attention_masks, dim=0).to(self.device)
        
        # Generate for batch
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids_batch,
                attention_mask=attention_mask_batch,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0,
            )
        
        # Decode all outputs
        captions = []
        for i, output_id in enumerate(output_ids):
            decoded_full = self.tokenizer.decode(output_id, skip_special_tokens=True).strip()
            
            if not decoded_full:
                decoded_raw = self.tokenizer.decode(output_id, skip_special_tokens=False).strip()
                decoded_full = decoded_raw.replace("</s>", "").replace("<s>", "").strip()
            
            # Parse answer from conversation
            text = self._parse_answer(decoded_full, convs[i])
            captions.append(text)
        
        return captions
    
    def generate_single(self, image: Image.Image) -> str:
        """Generate caption for a single image."""
        import torch
        import re
        
        # Prepare query with image token
        qs = self.prompt
        image_token_se = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN
        
        if self.IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(self.IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(self.IMAGE_PLACEHOLDER, self.DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = self.DEFAULT_IMAGE_TOKEN + "\n" + qs
        
        # Get conversation template
        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        # Process image (ensure it's on the correct device)
        image_sizes = [image.size]
        images_tensor = self.process_images(
            [image],
            self.image_processor,
            self.model.config
        )
        
        # Move to correct device with correct dtype
        if isinstance(images_tensor, list):
            images_tensor = [image.to(device=self.device, dtype=torch.float16) for image in images_tensor]
        else:
            images_tensor = images_tensor.to(device=self.device, dtype=torch.float16)
        
        # Tokenize
        input_ids = (
            self.tokenizer_image_token(prompt_text, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )
        
        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

        # Decode
        #
        # IMPORTANT:
        # Some LLaVA checkpoints/wrappers may NOT return output_ids as [prompt + answer].
        # In that case, slicing by `input_ids.shape[1]` will incorrectly chop off the
        # beginning of the answer (this is exactly the "truncated fragment" symptom).
        # Therefore we always decode the FULL sequence and then parse out the assistant
        # answer by conversation markers.
        decoded_full = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # If empty, try decoding without skipping special tokens (diagnostic fallback)
        if not decoded_full:
            decoded_raw = self.tokenizer.decode(output_ids[0], skip_special_tokens=False).strip()
            decoded_full = decoded_raw.replace("</s>", "").replace("<s>", "").strip()

        # Debug: show first few raw decodes for diagnosis
        if not hasattr(self, "_debug_count"):
            self._debug_count = 0
        if self._debug_count < 3:
            print(f"\n[DEBUG {self._debug_count}] Decoded FULL text:")
            print(f"  '{decoded_full[:250]}...'")
            self._debug_count += 1

        text = decoded_full

        # Best-effort: remove the prompt portion if it appears verbatim in decoded text
        # (may not always match due to special image tokens).
        if prompt_text in text:
            text = text.split(prompt_text, 1)[-1].strip()

        # Extract assistant answer using common markers (keep the LAST occurrence)
        markers = [
            "### Assistant:",
            "###Assistant:",
            "ASSISTANT:",
            "Assistant:",
        ]
        for m in markers:
            if m in text:
                text = text.split(m)[-1].strip()

        # If we still have both USER and ASSISTANT blocks, prefer content after assistant role.
        # (Some templates use roles like "USER:" / "ASSISTANT:".)
        if ("USER:" in text or "User:" in text) and ("ASSISTANT:" in text or "Assistant:" in text):
            if "ASSISTANT:" in text:
                text = text.split("ASSISTANT:", 1)[-1].strip()
            elif "Assistant:" in text:
                text = text.split("Assistant:", 1)[-1].strip()

        # Remove leading punctuation/whitespace artifacts
        text = text.lstrip(".:,;!? \n\t")

        # Remove trailing separators if present
        if conv.sep_style == self.SeparatorStyle.TWO and conv.sep2 and text.endswith(conv.sep2):
            text = text[: -len(conv.sep2)].strip()
        if conv.sep and text.endswith(conv.sep):
            text = text[: -len(conv.sep)].strip()

        text = text.strip()

        if hasattr(self, "_debug_count") and self._debug_count <= 3:
            print("[DEBUG] Parsed answer:")
            print(f"  '{text[:250]}...'")

        return text
    
    def _parse_answer(self, decoded_text: str, conv) -> str:
        """Extract the assistant's answer from decoded text."""
        import re
        
        text = decoded_text
        
        # Try to find assistant's response after conversation markers
        patterns = [
            r'### Assistant:\s*(.*?)(?:###|$)',
            r'ASSISTANT:\s*(.*?)(?:USER:|$)',
            r'Assistant:\s*(.*?)(?:User:|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                break
        
        # Remove prompt text if it's still there
        if self.prompt in text:
            text = text.replace(self.prompt, "").strip()
        
        # Remove leading punctuation/whitespace
        text = text.lstrip(".:,;!? \n\t")
        
        # Remove trailing separators
        if conv.sep_style == self.SeparatorStyle.TWO and conv.sep2 and text.endswith(conv.sep2):
            text = text[: -len(conv.sep2)].strip()
        if conv.sep and text.endswith(conv.sep):
            text = text[: -len(conv.sep)].strip()
        
        return text.strip()
    
    def generate(self, image: Image.Image) -> str:
        """Backward compatibility: generate caption for single image."""
        return self.generate_single(image)


def worker_process(
    gpu_id: int,
    image_paths: List[str],
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    load_4bit: bool,
    output_queue: mp.Queue,
    worker_idx: int,
    batch_size: int = 8,
):
    """Worker process for a single GPU."""
    try:
        # Import torch first
        import os
        # Set CUDA_VISIBLE_DEVICES to the specific physical GPU ID
        # This isolates the process to only see this GPU, effectively making it "cuda:0"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        import torch
        
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA not available")
        
        # Since we isolated the GPU, we always use cuda:0 inside this process
        device = "cuda:0"
        torch.cuda.set_device(0)
        
        print(f"[DEBUG] Worker {gpu_id}: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
        print(f"[DEBUG] Worker {gpu_id}: torch.cuda.device_count()={torch.cuda.device_count()}", flush=True)
        print(f"[GPU {gpu_id}] Initializing LLaVA model on device '{device}' (Physical: {gpu_id})...", flush=True)
        print(f"[GPU {gpu_id}] Current CUDA device: {torch.cuda.current_device()}", flush=True)
        
        # Initialize captioner (must disable device_map="auto" for multi-GPU)
        captioner = LLaVACaptioner(
            model_path=model_path,
            device=device,
            load_4bit=load_4bit,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_device_map_auto=False,  # Must be False for multi-GPU!
        )
        
        print(f"[GPU {gpu_id}] Processing {len(image_paths)} images with batch_size={batch_size}...", flush=True)
        
        # Process images in batches
        results = []
        batch_images = []
        batch_indices = []  # To track which image path belongs to which image in batch
        
        # Helper to process a batch
        def process_batch(current_batch_images, current_batch_indices):
            nonlocal results
            if not current_batch_images:
                return
            
            try:
                # Generate captions for the batch
                captions = captioner.generate_batch(current_batch_images)
                
                # Store results
                for i, caption in enumerate(captions):
                    original_idx = current_batch_indices[i]
                    img_path = image_paths[original_idx]
                    results.append((img_path, caption, None))
                    
            except Exception as e_batch:
                # If batch fails, try one by one as fallback or just fail all
                # Fallback to single processing to isolate the error
                print(f"[GPU {gpu_id}] Batch failed with error: {type(e_batch).__name__}: {str(e_batch)}", flush=True)
                print(f"[GPU {gpu_id}] Falling back to single processing for this batch...", flush=True)
                import traceback
                traceback.print_exc()
                for i, img in enumerate(current_batch_images):
                    original_idx = current_batch_indices[i]
                    img_path = image_paths[original_idx]
                    try:
                        single_caption = captioner.generate(img)
                        results.append((img_path, single_caption, None))
                    except Exception as e_single:
                        print(f"[GPU {gpu_id}] Single processing also failed for {img_path}: {str(e_single)}", flush=True)
                        results.append((img_path, None, str(e_single)))

        # Main loop
        pbar = tqdm(total=len(image_paths), desc=f"GPU {gpu_id}", position=worker_idx, leave=True)
        
        for idx, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
                batch_indices.append(idx)
                
                # If batch is full, process it
                if len(batch_images) >= batch_size:
                    process_batch(batch_images, batch_indices)
                    pbar.update(len(batch_images))
                    batch_images = []
                    batch_indices = []
                    
            except Exception as e:
                # If image loading fails, record error immediately
                results.append((img_path, None, f"Load Error: {str(e)}"))
                pbar.update(1)
        
        # Process remaining images in last batch
        if batch_images:
            process_batch(batch_images, batch_indices)
            pbar.update(len(batch_images))
            
        pbar.close()
        
        # Send results back
        output_queue.put((worker_idx, results))
        print(f"[GPU {gpu_id}] Completed! Processed {len([r for r in results if r[2] is None])} images.", flush=True)
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        output_queue.put((worker_idx, []))


def run_multi_gpu(args, gpu_ids, parquet_files, output_path, caption_mapping, mp_context):
    """Run caption generation on multiple GPUs in parallel."""
    
    # Collect all image paths that need processing
    print("\n📊 Collecting image paths...")
    all_image_paths = []
    
    for parquet_file in parquet_files:
        try:
            ds = load_dataset('parquet', data_files=str(parquet_file))['train']
            for i in range(len(ds)):
                image_path = ds[i]['preview']
                
                # Skip if already processed (unless force)
                if not args.force and image_path in caption_mapping:
                    continue
                
                all_image_paths.append(image_path)
                
                # Stop if reached max_samples
                if args.max_samples and len(all_image_paths) >= args.max_samples:
                    break
            
            if args.max_samples and len(all_image_paths) >= args.max_samples:
                break
        except Exception as e:
            print(f"⚠️  Failed to load {parquet_file}: {e}")
            continue
    
    print(f"📋 Total images to process: {len(all_image_paths)}")
    
    if not all_image_paths:
        print("✅ No images to process!")
        return
    
    # Split image paths among GPUs
    num_gpus = len(gpu_ids)
    chunk_size = (len(all_image_paths) + num_gpus - 1) // num_gpus
    image_chunks = [
        all_image_paths[i * chunk_size:(i + 1) * chunk_size]
        for i in range(num_gpus)
    ]
    
    print(f"\n🔀 Split into {num_gpus} chunks:")
    for i, chunk in enumerate(image_chunks):
        print(f"   GPU {gpu_ids[i]}: {len(chunk)} images")
    
    # Create output queue using spawn context
    output_queue = mp_context.Queue()
    
    # Start worker processes
    print(f"\n🚀 Starting {num_gpus} worker processes...")
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = mp_context.Process(
            target=worker_process,
            args=(
                gpu_id,
                image_chunks[i],
                args.model,
                args.prompt,
                args.max_new_tokens,
                args.temperature,
                args.load_4bit,
                output_queue,
                i,  # worker_idx
                args.batch_size,
            )
        )
        p.start()
        processes.append(p)
    
    # Collect results
    print("\n📥 Collecting results...")
    all_results_dict = {}
    for _ in range(num_gpus):
        worker_idx, results = output_queue.get()
        all_results_dict[worker_idx] = results
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Merge results in order
    all_results = []
    for i in range(num_gpus):
        all_results.extend(all_results_dict.get(i, []))
    
    # Process results
    total_processed = 0
    total_failed = 0
    
    for img_path, caption, error in all_results:
        if error:
            print(f"❌ Failed: {img_path} - {error}")
            total_failed += 1
        else:
            caption_mapping[img_path] = caption
            total_processed += 1
    
    # Save final results
    print(f"\n💾 Saving caption mapping to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(caption_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Completed!")
    print(f"   Processed: {total_processed}")
    print(f"   Failed: {total_failed}")
    print(f"   Total captions in mapping: {len(caption_mapping)}")
    print(f"\n📂 Caption mapping saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate LLaVA captions for training data")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to parquet files directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output JSON file for caption mapping")
    parser.add_argument("--model", type=str, default="liuhaotian/llava-v1.5-7b", help="LLaVA model path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--load_4bit", action="store_true", default=True, help="Load model in 4-bit")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process (for testing)")
    parser.add_argument("--force", "-f", action="store_true", help="Force regenerate all captions")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing multiple images")
    parser.add_argument("--save_images_dir", type=str, default=None, help="Directory to save image copies for inspection")
    parser.add_argument("--file_indices", type=str, default=None, help="Process specific parquet file indices (e.g., '0,1,2' or '0-5')")
    parser.add_argument("--prompt", type=str, 
                        default="Describe style, main subject, and especially the background of the whole image in one short sentence.",
                        help="Custom prompt for LLaVA caption generation")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens for caption generation")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for generation (higher = more random)")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for parallel processing")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated GPU IDs (e.g., '0,1,2'). If not set, uses 0 to num_gpus-1")
    args = parser.parse_args()
    
    import torch  # Re-added local import because top-level was removed
    
    if not LLAVA_AVAILABLE:
        print("❌ LLaVA is not available. Please check your environment.")
        sys.exit(1)
    
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    
    # Find all parquet files
    all_parquet_files = sorted(list(data_dir.glob("*.parquet")))
    if not all_parquet_files:
        print(f"❌ No parquet files found in {data_dir}")
        sys.exit(1)
    
    # Filter by file indices if specified
    if args.file_indices:
        if '-' in args.file_indices:
            # Range format: "0-5"
            start, end = map(int, args.file_indices.split('-'))
            file_indices = list(range(start, end + 1))
        else:
            # Comma-separated: "0,1,2"
            file_indices = [int(x.strip()) for x in args.file_indices.split(',')]
        
        parquet_files = [all_parquet_files[i] for i in file_indices if i < len(all_parquet_files)]
        print(f"📂 Processing {len(parquet_files)} parquet files (indices {args.file_indices}):")
        for f in parquet_files:
            print(f"   - {f.name}")
    else:
        parquet_files = all_parquet_files
        print(f"📂 Found {len(parquet_files)} parquet files")
    
    # Load existing caption mapping if exists
    caption_mapping = {}
    if output_path.exists() and not args.force:
        print(f"📖 Loading existing caption mapping from {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            caption_mapping = json.load(f)
        print(f"   Found {len(caption_mapping)} existing captions")
    
    # Determine GPU configuration
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        num_gpus = len(gpu_ids)
    else:
        num_gpus = args.num_gpus
        gpu_ids = list(range(num_gpus))
    
    print(f"\n🎮 Using {num_gpus} GPU(s): {gpu_ids}")
    
    # Multi-GPU mode
    if num_gpus > 1:
        print("🚀 Multi-GPU parallel processing mode enabled!")
        # Use spawn context for CUDA compatibility
        mp_context = mp.get_context('spawn')
        return run_multi_gpu(
            args=args,
            gpu_ids=gpu_ids,
            parquet_files=parquet_files,
            output_path=output_path,
            caption_mapping=caption_mapping,
            mp_context=mp_context,
        )
    
    # Single GPU mode (original logic)
    print("🚀 Single GPU sequential processing mode")
    
    # Initialize LLaVA captioner
    print("\n📥 Initializing LLaVA captioner...")
    captioner = LLaVACaptioner(
        model_path=args.model,
        device=args.device,
        load_4bit=args.load_4bit,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    
    # Create image save directory if specified
    save_images_dir = None
    if args.save_images_dir:
        save_images_dir = Path(args.save_images_dir)
        save_images_dir.mkdir(parents=True, exist_ok=True)
        print(f"📸 Will save image copies to: {save_images_dir}")
    
    # Process each parquet file
    total_processed = 0
    total_skipped = 0
    total_failed = 0
    batch_size = args.batch_size
    print(f"📦 Using batch size: {batch_size}")
    
    for pf_idx, parquet_file in enumerate(parquet_files):
        print(f"\n📄 Processing {parquet_file.name} ({pf_idx+1}/{len(parquet_files)})...")
        
        try:
            # Load dataset
            ds = load_dataset('parquet', data_files=str(parquet_file))['train']
            print(f"   Loaded {len(ds)} samples")
            
            # Determine how many to process
            n_samples = len(ds) if args.max_samples is None else min(args.max_samples - total_processed, len(ds))
            
            # Collect batch
            batch_images = []
            batch_paths = []
            
            # Process in batches
            for i in tqdm(range(n_samples), desc=f"   Generating captions (batch={batch_size})"):
                item = ds[i]
                image_path = item['preview']
                
                # Skip if caption already exists (unless force)
                if not args.force and image_path in caption_mapping:
                    total_skipped += 1
                    continue
                
                try:
                    # Load image and add to batch
                    img = Image.open(image_path).convert("RGB")
                    batch_images.append(img)
                    batch_paths.append(image_path)
                    
                    # Process batch when full or at end
                    if len(batch_images) >= batch_size or i == n_samples - 1:
                        if batch_images:  # Only if there are images
                            # Generate captions for batch
                            captions = captioner.generate_batch(batch_images)
                            
                            # Save all captions in batch
                            for idx, (path, caption) in enumerate(zip(batch_paths, captions)):
                                caption_mapping[path] = caption
                                total_processed += 1
                                
                                # Save image copy if directory specified (first 10 only)
                                if save_images_dir and total_processed <= 10:
                                    img_filename = f"sample_{total_processed:05d}.png"
                                    img_save_path = save_images_dir / img_filename
                                    batch_images[idx].save(img_save_path)
                                    
                                    # Also save caption as txt
                                    txt_filename = f"sample_{total_processed:05d}.txt"
                                    txt_save_path = save_images_dir / txt_filename
                                    with open(txt_save_path, 'w', encoding='utf-8') as f:
                                        f.write(f"Image: {Path(path).name}\n")
                                        f.write(f"Path: {path}\n")
                                        f.write(f"\nCaption:\n{caption}\n")
                                
                                # Show first few captions
                                if total_processed <= 5:
                                    print(f"\n   📝 Sample caption for {Path(path).name}:")
                                    print(f"      {caption}")
                        
                        # Clear batch
                        batch_images = []
                        batch_paths = []
                        
                        # Save periodically (every 100 samples)
                        if total_processed % 100 == 0:
                            print(f"\n   💾 Saving checkpoint at {total_processed} captions...")
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(output_path, 'w', encoding='utf-8') as f:
                                json.dump(caption_mapping, f, indent=2, ensure_ascii=False)
                        
                        # Check if we've reached max_samples
                        if args.max_samples and total_processed >= args.max_samples:
                            print(f"\n   ⏹️  Reached max_samples limit ({args.max_samples})")
                            break
                
                except Exception as e:
                    print(f"\n   ❌ Failed to process {image_path}: {e}")
                    total_failed += 1
                    # Don't break batch, just skip this image
                    continue
            
            # Check if we've reached max_samples
            if args.max_samples and total_processed >= args.max_samples:
                break
        
        except Exception as e:
            print(f"   ❌ Failed to load {parquet_file}: {e}")
            continue
    
    # Final save
    print(f"\n💾 Saving final caption mapping to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(caption_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Completed!")
    print(f"   Processed: {total_processed}")
    print(f"   Skipped: {total_skipped}")
    print(f"   Failed: {total_failed}")
    print(f"   Total captions in mapping: {len(caption_mapping)}")
    print(f"\n📂 Caption mapping saved to: {output_path}")


if __name__ == "__main__":
    main()

