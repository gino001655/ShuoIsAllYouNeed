import os
import random
import yaml
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T

# Reuse UAE's namespace package by adding sibling path at runtime
_HERE = os.path.dirname(__file__)
_CLD2_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_UAE_ROOT = os.path.abspath(os.path.join(_CLD2_ROOT, "..", "UAE"))
if _UAE_ROOT not in os.sys.path:
    os.sys.path.insert(0, _UAE_ROOT)

from tools.dataset import LayoutTrainDataset, collate_fn  # noqa: E402
from tools.tools import (  # noqa: E402
    seed_everything,
    build_layer_mask,
    get_input_box,
    encode_target_latents,
    get_timesteps_sd3,
    set_sdpa_kernel,
)
from models.sd3_backbone import SD3Backbone  # noqa: E402
from models.qwen_projector import QwenProjector  # noqa: E402
from models.image_conditioning import ImageConditioning  # noqa: E402


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def preprocess_pixel_rgb(pixel_rgb: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Input:  pixel_rgb in [0,1], shape [L,3,H,W] or [B,L,3,H,W]
    Output: normalized to [-1,1]
    """
    if pixel_rgb.dim() == 4:
        x = pixel_rgb
    else:
        x = pixel_rgb[0]
    x = x.to(dtype=dtype)
    x = x.clamp(0, 1) * 2.0 - 1.0
    return x


def maybe_subsample_layers(
    pixel_rgb_lchw: torch.Tensor,
    layout: List[List[int]],
    max_layers_per_step: int,
) -> Tuple[torch.Tensor, List[List[int]]]:
    """
    Keep layer 0 (whole) and 1 (background) always. Subsample foreground layers if needed.
    """
    if max_layers_per_step is None or max_layers_per_step <= 0:
        return pixel_rgb_lchw, layout

    L = pixel_rgb_lchw.shape[0]
    if L <= 2:
        return pixel_rgb_lchw, layout

    # foreground indices: [2..L-1]
    fg_indices = list(range(2, L))
    keep_fg = min(max_layers_per_step, len(fg_indices))
    chosen = sorted(random.sample(fg_indices, keep_fg))

    keep = [0, 1] + chosen
    pixel_rgb_lchw = pixel_rgb_lchw[keep]
    layout = [layout[i] for i in keep]
    return pixel_rgb_lchw, layout


def train(config_path: str):
    cfg = load_config(config_path)
    seed_everything(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if bool(cfg.get("enable_tf32", True)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    mp = (cfg.get("mixed_precision") or "bf16").lower()
    if mp == "bf16":
        dtype = torch.bfloat16
    elif mp == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    set_sdpa_kernel(cfg.get("sdpa_kernel", "auto"))

    # Models
    sd3 = SD3Backbone(
        sd3_model_path=cfg["sd3_model_path"],
        dit_path=cfg["dit_path"],
        dit_lora_path=cfg.get("dit_lora_path"),
        device=device,
        dtype=dtype,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        max_layer_num=int(cfg.get("max_layer_num", 64)),
    )

    if bool(cfg.get("torch_compile", False)) and hasattr(torch, "compile"):
        try:
            sd3.transformer = torch.compile(sd3.transformer)  # type: ignore[attr-defined]
        except Exception:
            pass

    qwen = QwenProjector(
        llm_path=cfg["llm_path"],
        llm_processor_path=cfg["llm_processor_path"],
        llm_lora_path=cfg.get("llm_lora_path"),
        load_in_4bit=bool(cfg.get("llm_load_in_4bit", False)),
        attn_implementation=str(cfg.get("llm_attn_implementation", "sdpa")),
        device=device,
        dtype=dtype,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
    )

    img_cond = ImageConditioning(enabled=bool(cfg.get("use_image_conditioning", False)))
    to_tensor = T.ToTensor()

    # Dataset
    dataset = LayoutTrainDataset(cfg["data_dir"], split=str(cfg.get("split", "train")))
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

    # Trainable params: LoRA on SD3 + layer embedding
    params = sd3.get_trainable_params()
    optimizer = torch.optim.AdamW(params, lr=float(cfg.get("lr", 1e-4)), weight_decay=float(cfg.get("weight_decay", 1e-3)))

    max_steps = int(cfg.get("max_steps", 1000))
    log_every = int(cfg.get("log_every", 50))
    save_every = int(cfg.get("save_every", 200))
    accum_steps = int(cfg.get("accum_steps", 1))
    grad_clip = float(cfg.get("grad_clip", 1.0))

    out_dir = cfg.get("output_dir", "./cld2_out")
    os.makedirs(out_dir, exist_ok=True)

    num_inference_steps = int(cfg.get("num_inference_steps", 28))
    max_layers_per_step = int(cfg.get("max_layers_per_step", 0))

    step = 0
    pbar = tqdm(total=max_steps, desc="train")
    optimizer.zero_grad(set_to_none=True)

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            H = int(batch["height"][0])
            W = int(batch["width"][0])
            caption = batch["caption"][0]
            layout = batch["layout"][0]
            adapter_img = batch["whole_img"][0]

            pixel_rgb = preprocess_pixel_rgb(batch["pixel_RGB"].to(device=device), dtype=dtype)  # [L,3,H,W]
            pixel_rgb, layout = maybe_subsample_layers(pixel_rgb, layout, max_layers_per_step=max_layers_per_step)

            # Quantize boxes to align with model grid
            layer_boxes = get_input_box(layout)

            # Text embeddings (UAE projector)
            with torch.no_grad():
                prompt_embeds = qwen.caption_to_prompt_embeds(caption)  # [1, seq, 4096]
                pooled_prompt_embeds = sd3.pooled_prompt_embeds  # [1, pooled_dim]

            # Encode targets to latents per layer
            x0 = encode_target_latents(sd3.vae, pixel_rgb.unsqueeze(0), dtype=dtype)  # [1,L,C,H_lat,W_lat]
            bs, L, C_lat, H_lat, W_lat = x0.shape

            # Layer embedding (Phase-1): add layer id embedding in latent space
            x0 = sd3.add_layer_embedding(x0)

            x1 = torch.randn_like(x0)

            # Timesteps (use SD3 scheduler to sample t; normalize for RF mix)
            timesteps = get_timesteps_sd3(sd3.scheduler, H_lat, W_lat, num_inference_steps=num_inference_steps, device=device)
            t = timesteps[random.randint(0, len(timesteps) - 1)]
            t_batch = t.expand(bs).to(dtype=torch.float32)
            t_norm = (t_batch / 1000.0).to(dtype=dtype).view(bs, 1, 1, 1, 1)

            xt = (1.0 - t_norm) * x0 + t_norm * x1
            v_star = x1 - x0

            # Optional image conditioning (Phase-1 minimal): add masked whole-image latent residuals
            if img_cond.enabled:
                whole_bchw = to_tensor(adapter_img).unsqueeze(0).to(device=device, dtype=dtype)  # [1,3,H,W] in [0,1]
                whole_bchw = whole_bchw.clamp(0, 1) * 2.0 - 1.0
                residual = img_cond.build_residual(
                    vae=sd3.vae,
                    whole_img_bchw=whole_bchw,
                    list_layer_box=layer_boxes,
                    n_layers=L,
                    dtype=dtype,
                    device=device,
                    scale=float(cfg.get("image_conditioning_scale", 0.5)),
                )
                xt = xt + residual

            # bbox mask in latent space
            mask = build_layer_mask(L, H_lat, W_lat, layer_boxes).to(device=device, dtype=dtype)  # [L,1,H_lat,W_lat]
            mask = mask.unsqueeze(0)  # [1,L,1,H_lat,W_lat]

            # Phase-1: flatten layers into batch dimension
            xt_flat = xt.view(bs * L, C_lat, H_lat, W_lat)
            v_star_flat = v_star.view(bs * L, C_lat, H_lat, W_lat)
            mask_flat = mask.view(bs * L, 1, H_lat, W_lat)

            # repeat text cond per layer
            prompt_flat = prompt_embeds.repeat_interleave(L, dim=0).to(device=device, dtype=dtype)
            pooled_flat = pooled_prompt_embeds.repeat_interleave(L, dim=0).to(device=device, dtype=dtype)

            # SD3 transformer expects raw timesteps (not normalized)
            t_model = t.expand(bs * L).to(device=device)

            v_pred_flat = sd3.denoise(
                latents=xt_flat,
                timestep=t_model,
                prompt_embeds=prompt_flat,
                pooled_prompt_embeds=pooled_flat,
            )  # [bs*L,C_lat,H_lat,W_lat]

            mse = (v_pred_flat - v_star_flat) ** 2
            mse = mse.mean(dim=1, keepdim=True)  # [bs*L,1,H,W]
            loss = (mse * mask_flat).sum() / (mask_flat.sum() + 1e-8)

            (loss / accum_steps).backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0:
                pbar.set_postfix(loss=float(loss.detach().cpu()))
            if (step + 1) % log_every == 0:
                pbar.update(log_every)

            if (step + 1) % save_every == 0:
                sd3.save_lora_and_layer_embed(out_dir, step + 1)

            step += 1

    pbar.close()
    sd3.save_lora_and_layer_embed(out_dir, step)
    print("[DONE] CLD2 training finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    args = parser.parse_args()
    train(args.config_path)

import os, random
# Set CUDA_VISIBLE_DEVICES before importing torch
# You can modify this or set it via environment variable
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm
from prodigyopt import Prodigy
from diffusers import FluxTransformer2DModel
from diffusers.configuration_utils import FrozenDict

from models.mmdit import CustomFluxTransformer2DModel
from models.pipeline import CustomFluxPipelineCfgLayer, CustomFluxPipeline
from models.multiLayer_adapter import MultiLayerAdapter
from tools.tools import save_checkpoint, load_checkpoint, load_config, seed_everything, get_input_box, set_lora_into_transformer, build_layer_mask, encode_target_latents, get_timesteps
from tools.dataset import LayoutTrainDataset, collate_fn


def train(config_path):
    config = load_config(config_path)
    seed_everything(config.get("seed", 1234))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"[INFO] CUDA available. GPU count: {torch.cuda.device_count()}, Current GPU: {torch.cuda.current_device()}, GPU name: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("[WARNING] CUDA not available! Training will run on CPU (very slow).", flush=True)

    print("[INFO] Loading pretrained Transformer...", flush=True)
    transformer_orig = FluxTransformer2DModel.from_pretrained(
        config.get('transformer_varient', config['pretrained_model_name_or_path']),
        subfolder="" if 'transformer_varient' in config else "transformer",
        revision=config.get('revision', None),
        variant=config.get('variant', None),
        torch_dtype=torch.bfloat16,
        cache_dir=config.get('cache_dir', None),
    )
    mmdit_config = dict(transformer_orig.config)
    mmdit_config["_class_name"] = "CustomSD3Transformer2DModel"
    mmdit_config["max_layer_num"] = config['max_layer_num']
    mmdit_config = FrozenDict(mmdit_config)

    transformer = CustomFluxTransformer2DModel.from_config(mmdit_config).to(dtype=torch.bfloat16).to(device)
    missing_keys, unexpected_keys = transformer.load_state_dict(transformer_orig.state_dict(), strict=False)
    if missing_keys: print(f"[WARN] Missing keys: {missing_keys}")
    if unexpected_keys: print(f"[WARN] Unexpected keys: {unexpected_keys}")
    # Free memory from transformer_orig
    del transformer_orig
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if 'pretrained_lora_dir' in config:
        print("[INFO] Loading LoRA weights...", flush=True)
        lora_state_dict = CustomFluxPipeline.lora_state_dict(config['pretrained_lora_dir'])
        CustomFluxPipeline.load_lora_into_transformer(lora_state_dict, None, transformer)
        transformer.fuse_lora(safe_fusing=True)
        transformer.unload_lora()
        print("[INFO] Successfully loaded and fused LoRA weights.", flush=True)

    if 'artplus_lora_dir' in config:
        print("[INFO] Loading artplus LoRA weights...", flush=True)
        lora_state_dict = CustomFluxPipeline.lora_state_dict(config['artplus_lora_dir'])
        CustomFluxPipeline.load_lora_into_transformer(lora_state_dict, None, transformer)
        transformer.fuse_lora(safe_fusing=True)
        transformer.unload_lora()
        print("[INFO] Successfully loaded and fused artplus LoRA weights.", flush=True)

    # Load MultiLayer-Adapter
    print("[INFO] Loading MultiLayer-Adapter weights...", flush=True)
    multiLayer_adapter = MultiLayerAdapter.from_pretrained(config['pretrained_adapter_path']).to(torch.bfloat16).to(device)
    multiLayer_adapter.set_layerPE(transformer.layer_pe, transformer.max_layer_num)
    print("[INFO] Successfully loaded MultiLayer-Adapter weights.", flush=True)

    pipeline = CustomFluxPipelineCfgLayer.from_pretrained(
        config['pretrained_model_name_or_path'],
        transformer=transformer,
        revision=config.get('revision', None),
        variant=config.get('variant', None),
        torch_dtype=torch.bfloat16,
        cache_dir=config.get('cache_dir', None),
    ).to(device)
    pipeline.set_multiLayerAdapter(multiLayer_adapter)
    
    # Verify models are on correct device
    if torch.cuda.is_available():
        transformer_device = next(pipeline.transformer.parameters()).device
        adapter_device = next(pipeline.multiLayerAdapter.parameters()).device
        print(f"[INFO] Transformer device: {transformer_device}, Adapter device: {adapter_device}", flush=True)
        if transformer_device.type != 'cuda' or adapter_device.type != 'cuda':
            print(f"[WARNING] Models are not on GPU! Transformer: {transformer_device}, Adapter: {adapter_device}", flush=True)
    pipeline.transformer.gradient_checkpointing = True
    pipeline.multiLayerAdapter.gradient_checkpointing = True

    lora_rank = int(config.get("lora_rank", 16))
    lora_alpha = float(config.get("lora_alpha", 16))
    lora_dropout = float(config.get("lora_dropout", 0.0))
    set_lora_into_transformer(pipeline.transformer, lora_rank, lora_alpha, lora_dropout)
    set_lora_into_transformer(pipeline.multiLayerAdapter, lora_rank, lora_alpha, lora_dropout)
    pipeline.transformer.requires_grad_(False)
    pipeline.multiLayerAdapter.requires_grad_(False)
    pipeline.transformer.train()
    pipeline.multiLayerAdapter.train()
    for n, param in pipeline.transformer.named_parameters():
        if 'lora' in n or 'layer_pe' in n:
            param.requires_grad = True
        else:
            param.requires_grad = False
    for n, param in pipeline.multiLayerAdapter.named_parameters():
        if 'lora' in n or 'layer_pe' in n:
            param.requires_grad = True
        else:
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in pipeline.transformer.parameters() if p.requires_grad)
    n_trainable_adapter = sum(p.numel() for p in pipeline.multiLayerAdapter.parameters() if p.requires_grad)
    print(f"[INFO] LoRA injected. Transformer Trainable params: {n_trainable/1e6:.2f}M; MultiLayer-Adapter Trainable params: {n_trainable_adapter/1e6:.2f}M", flush=True)

    print("[INFO] Using Prodigy optimizer.", flush=True)
    params = [p for p in pipeline.transformer.parameters() if p.requires_grad]
    params_adapter = [p for p in pipeline.multiLayerAdapter.parameters() if p.requires_grad]
    optimizer = Prodigy(
        params,
        lr=1.0,
        betas=(0.9, 0.999),
        weight_decay=0.001,
        decouple=True,
        safeguard_warmup=True,
        use_bias_correction=True,
    )
    optimizer_adapter = Prodigy(
        params_adapter,
        lr=1.0,
        betas=(0.9, 0.999),
        weight_decay=0.001,
        decouple=True,
        safeguard_warmup=True,
        use_bias_correction=True,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0
    )
    scheduler_adapter = LambdaLR(
        optimizer_adapter,
        lr_lambda=lambda step: 1.0
    )

    dataset = LayoutTrainDataset(data_dir = config['data_dir'], split="train")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

    max_steps = int(config.get("max_steps", 1000))
    log_every = int(config.get("log_every", 50))
    save_every = int(config.get("save_every", 500))
    accum_steps = int(config.get("accum_steps", 1))
    out_dir = config.get("output_dir", "./rf_lora_out")
    os.makedirs(out_dir, exist_ok=True)
    tb_writer = SummaryWriter(out_dir)

    num_inference_steps = config.get("num_inference_steps", 28)

    start_step = 0
    if "resume_from" in config and config["resume_from"] is not None:
        ckpt_dir = config["resume_from"]
        start_step = load_checkpoint(pipeline.transformer, pipeline.multiLayerAdapter, optimizer, optimizer_adapter, scheduler, scheduler_adapter, ckpt_dir, device)
    pbar = tqdm(total=max_steps, desc="train", initial=start_step)
    step = start_step

    while step < max_steps:
        for batch in loader:
            if step >= max_steps: break

            pixel_RGB = batch["pixel_RGB"].to(device=device, dtype=torch.bfloat16)
            pixel_RGB = pipeline.image_processor.preprocess(pixel_RGB[0])
            H = int(batch["height"][0])     # By default, only a single sample per batch is allowed (because later the data will be concatenated based on bounding boxes, which have varying lengths)
            W = int(batch["width"][0])
            adapter_img = batch["whole_img"][0]
            caption = batch["caption"][0]
            layer_boxes = get_input_box(batch["layout"][0])

            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                    prompt=caption,
                    prompt_2=None,
                    num_images_per_prompt=1,
                    max_sequence_length=int(config.get("max_sequence_length", 512)),
                )

                prompt_embeds = prompt_embeds.to(device=device, dtype=torch.bfloat16)   # (1, 512, 4096)
                pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=torch.bfloat16)     # (1, 768)
                text_ids = text_ids.to(device=device, dtype=torch.bfloat16)  # (512, 3)

                adapter_image, _, _ = pipeline.prepare_image(
                    image=adapter_img,
                    width=W,
                    height=H,
                    batch_size=1,
                    num_images_per_prompt=1,
                    device=device,
                    dtype=pipeline.transformer.dtype,
                )

            x0, latent_image_ids = encode_target_latents(pipeline, pixel_RGB.unsqueeze(0), n_layers=len(layer_boxes), list_layer_box=layer_boxes)
            _, L, C_lat, H_lat, W_lat = x0.shape

            x1 = torch.randn_like(x0)
            image_seq_len = latent_image_ids.shape[0]
            timesteps = get_timesteps(pipeline, image_seq_len=image_seq_len, num_inference_steps=num_inference_steps, device=device)
            t = timesteps[random.randint(0, len(timesteps)-1)].to(device=device, dtype=torch.float32)
            t = t.expand(x0.shape[0]).to(x0.dtype)
            t_b = t.view(1, 1, 1, 1, 1).to(x0.dtype)
            t_b = t_b / 1000.0  # [0,1]
            xt = (1.0 - t_b) * x0 + t_b * x1
            v_star = x1 - x0

            mask = build_layer_mask(L, H_lat, W_lat, layer_boxes).to(device=device, dtype=x0.dtype)  # [L,1,H_lat,W_lat]
            mask = mask.unsqueeze(0)  # [1,L,1,H_lat,W_lat]

            # classifier-free guidance
            guidance_scale=config.get('cfg', 4.0)
            if pipeline.transformer.config.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                guidance = guidance.expand(x0.shape[0])
            else:
                guidance = None

            pipeline.transformer.train()
            pipeline.multiLayerAdapter.train()

            (
                adapter_block_samples,
                adapter_single_block_samples,
            ) = pipeline.multiLayerAdapter(
                hidden_states=xt,
                list_layer_box=layer_boxes,
                adapter_cond=adapter_image,
                conditioning_scale=config.get("adapter_scale", 1.0),
                timestep=t / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )

            v_pred = pipeline.transformer(
                hidden_states=xt,
                adapter_block_samples=[
                    sample.to(dtype=pipeline.transformer.dtype)
                    for sample in adapter_block_samples
                ],
                adapter_single_block_samples=[
                    sample.to(dtype=pipeline.transformer.dtype)
                    for sample in adapter_single_block_samples
                ] if adapter_single_block_samples is not None else adapter_single_block_samples,
                list_layer_box=layer_boxes,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=t / 1000,                 # [0,1]
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                guidance=guidance,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]  # [1, L, C, H_lat, W_lat]

            # MSE（masked）
            mse = (v_pred - v_star) ** 2
            mse = mse.mean(dim=2, keepdim=True)  # [1,L,1,H_lat,W_lat]
            loss = (mse * mask).sum() / (mask.sum() + 1e-8)

            loss = loss / accum_steps
            loss.float().backward()
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(config.get("grad_clip", 1.0)))
                optimizer.step()
                optimizer_adapter.step()
                scheduler.step()
                scheduler_adapter.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_adapter.zero_grad(set_to_none=True)

                tb_writer.add_scalar("loss", loss.item(), step)

            step += 1
            if step % log_every == 0:
                pbar.set_postfix(loss=float(loss.detach().cpu()))
                pbar.update(log_every)

            if step % save_every == 0 or step == max_steps:
                save_checkpoint(pipeline.transformer, pipeline.multiLayerAdapter, optimizer, optimizer_adapter, scheduler, scheduler_adapter, step, out_dir)

    pbar.close()
    print("[DONE] Training finished.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    args = parser.parse_args()
    train(args.config_path)