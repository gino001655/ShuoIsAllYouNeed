import os, random
import warnings
import logging
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Set CUDA_VISIBLE_DEVICES before importing torch
# You can modify this or set it via environment variable
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Suppress verbose warnings and truncation messages
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*truncated.*")
warnings.filterwarnings("ignore", message=".*CLIP.*")
warnings.filterwarnings("ignore", message=".*torch.cpu.amp.autocast.*")

# Suppress PIL and diffusers verbose output
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

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

    # Multi-GPU support with DataParallel
    use_multi_gpu = config.get("use_multi_gpu", False)
    # Save dtype and config before wrapping (needed for prepare_image and other operations)
    transformer_dtype = pipeline.transformer.dtype
    transformer_config = pipeline.transformer.config
    if use_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        print(f"[INFO] Using {num_gpus} GPUs with DataParallel", flush=True)
        # Wrap models with DataParallel
        pipeline.transformer = torch.nn.DataParallel(pipeline.transformer)
        pipeline.multiLayerAdapter = torch.nn.DataParallel(pipeline.multiLayerAdapter)
        # Update device to cuda:0 (DataParallel will handle distribution)
        device = torch.device("cuda:0")
        print(f"[INFO] Models wrapped with DataParallel on {num_gpus} GPUs", flush=True)
    else:
        if torch.cuda.is_available():
            print(f"[INFO] Using single GPU: {device}", flush=True)

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

    # 載入 dataset，支持三種模式：
    # 1. Indexed dataset (TAData + caption.json，用 index 匹配)
    # 2. DLCVLayoutDataset (path-based dataset + caption mapping)
    # 3. LayoutTrainDataset (PrismLayersPro format)
    
    caption_mapping_path = config.get('caption_mapping', None)
    enable_dataset_debug = config.get('enable_dataset_debug', True)  # 默認啟用，顯示前幾個樣本的載入情況
    use_indexed_dataset = config.get('use_indexed_dataset', False)
    
    print("\n" + "="*60)
    print("載入訓練數據集")
    print("="*60)
    
    if use_indexed_dataset:
        # 方案 B: 使用 indexed dataset (TAData + caption.json)
        print(f"[INFO] 使用 DLCVLayoutDatasetIndexed (Index-based caption matching)", flush=True)
        print(f"[INFO] Data dir: {config['data_dir']}", flush=True)
        print(f"[INFO] Caption JSON: {caption_mapping_path}", flush=True)
        
        from tools.dlcv_dataset_indexed import DLCVLayoutDatasetIndexed, collate_fn as indexed_collate_fn
        
        if enable_dataset_debug:
            print(f"[INFO] 🔍 Dataset debug enabled: 將顯示前 3 個樣本的詳細資訊", flush=True)
        
        dataset = DLCVLayoutDatasetIndexed(
            data_dir=config['data_dir'],
            caption_json_path=caption_mapping_path,
            enable_debug=enable_dataset_debug,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=indexed_collate_fn)
        print(f"[INFO] ✓ 載入 {len(dataset)} 個訓練樣本", flush=True)
    
    else:
        # 方案 A 或原始方案
        try:
            from tools.dlcv_dataset import DLCVLayoutDataset, collate_fn as dlcv_collate_fn
            print(f"[INFO] 使用 DLCVLayoutDataset（DLCV 格式，path-based）", flush=True)
            
            if enable_dataset_debug:
                print(f"[INFO] 🔍 Dataset debug enabled: 將顯示前幾個樣本的詳細資訊", flush=True)
            
            dataset = DLCVLayoutDataset(
                data_dir=config['data_dir'],
                split="train",
                caption_mapping_path=caption_mapping_path,
                enable_debug=enable_dataset_debug
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=dlcv_collate_fn)
            
            if caption_mapping_path:
                print(f"[INFO] 使用 LLaVA 生成的 captions: {caption_mapping_path}", flush=True)
            
            print(f"[INFO] ✓ 載入 {len(dataset)} 個訓練樣本", flush=True)
            
        except Exception as e:
            print(f"[INFO] DLCVLayoutDataset 失敗，回退到 LayoutTrainDataset: {e}", flush=True)
            print(f"[INFO] 使用 LayoutTrainDataset（PrismLayersPro 格式）", flush=True)
            
            if enable_dataset_debug:
                print(f"[INFO] 🔍 Dataset debug enabled: 將顯示前幾個樣本的詳細資訊", flush=True)
            
            dataset = LayoutTrainDataset(
                data_dir=config['data_dir'],
                split="train",
                enable_debug=enable_dataset_debug
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
            print(f"[INFO] ✓ 載入 {len(dataset)} 個訓練樣本", flush=True)
    
    print("="*60 + "\n", flush=True)

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

    print(f"[INFO] 開始訓練循環，目標步數: {max_steps}", flush=True)
    print(f"[INFO] 每 10 步顯示詳細資訊，每 {log_every} 步記錄 loss\n", flush=True)
    
    # Optional: skip samples with <unk> in caption
    skip_unk_captions = bool(config.get("skip_unk_captions", False))
    unk_token = str(config.get("unk_token", "<unk>"))
    unk_skip_min_count = int(config.get("unk_skip_min_count", 1))
    
    while step < max_steps:
        for batch in loader:
            if step >= max_steps: break

            # 提取 batch 數據
            pixel_RGB = batch["pixel_RGB"].to(device=device, dtype=torch.bfloat16)
            pixel_RGB = pipeline.image_processor.preprocess(pixel_RGB)
            H = int(batch["height"])     # By default, only a single sample per batch is allowed (because later the data will be concatenated based on bounding boxes, which have varying lengths)
            W = int(batch["width"])
            adapter_img = batch["whole_img"]
            caption = batch["caption"]
            # Build layer boxes and filter out invalid (zero-area) boxes to prevent model crash
            raw_layer_boxes = get_input_box(batch["layout"])
            valid_indices = []
            layer_boxes = []
            for bi, box in enumerate(raw_layer_boxes):
                if box is None or len(box) < 4:
                    continue
                x1, y1, x2, y2 = box[:4]
                if (x2 - x1) > 0 and (y2 - y1) > 0:
                    valid_indices.append(bi)
                    layer_boxes.append(box)
            if len(layer_boxes) == 0:
                sample_idx = batch.get("idx", None)
                print(f"[SKIP] step={step} sample_idx={sample_idx} no valid layer boxes -> skip", flush=True)
                continue
            if len(layer_boxes) != len(raw_layer_boxes):
                sample_idx = batch.get("idx", None)
                print(
                    f"[WARN] step={step} sample_idx={sample_idx} "
                    f"filtered invalid boxes: {len(raw_layer_boxes)} -> {len(layer_boxes)}",
                    flush=True,
                )
                # Keep pixel layers aligned with boxes (dim 0 is layer dimension)
                pixel_RGB = pixel_RGB[valid_indices]

            # Skip samples with <unk> in caption (data quality filter)
            if skip_unk_captions:
                caption_str = caption if isinstance(caption, str) else str(caption)
                unk_count = caption_str.count(unk_token) if unk_token else 0
                if unk_token and unk_count >= unk_skip_min_count:
                    sample_idx = batch.get("idx", None)
                    print(f"[SKIP] step={step} sample_idx={sample_idx} unk_count={unk_count} -> skip training this sample", flush=True)
                    continue
            
            # 顯示詳細資訊（每 10 步或第 0 步）
            if step == 0 or step % 10 == 0:
                print(f"\n{'='*60}")
                print(f"[STEP {step}] 訓練數據詳情")
                print(f"{'='*60}")
                print(f"  📊 Canvas 尺寸: {W} x {H}")
                print(f"  🎨 圖層數量: {len(layer_boxes)}")
                
                # 顯示每個圖層的資訊
                for i, layer in enumerate(batch["layout"][:5]):  # 只顯示前 5 個圖層
                    # 支援兩種 layout 格式：
                    # - dict: {'left','top','width','height','type'} (indexed dataset)
                    # - list/tuple: [x1, y1, x2, y2] (DLCVLayoutDataset)
                    if isinstance(layer, dict):
                        print(
                            f"    Layer {i}: "
                            f"bbox=({layer['left']:.0f}, {layer['top']:.0f}, {layer['width']:.0f}, {layer['height']:.0f}), "
                            f"type={layer.get('type', 'unknown')}"
                        )
                    elif isinstance(layer, (list, tuple)) and len(layer) >= 4:
                        x1, y1, x2, y2 = layer[:4]
                        w = x2 - x1
                        h = y2 - y1
                        print(
                            f"    Layer {i}: "
                            f"bbox=({x1:.0f}, {y1:.0f}, {w:.0f}, {h:.0f}) "
                            f"[x1,y1,x2,y2]=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
                        )
                    else:
                        print(f"    Layer {i}: (unknown format) type={type(layer)} value={layer}")
                if len(batch["layout"]) > 5:
                    print(f"    ... 還有 {len(batch['layout']) - 5} 個圖層")
                
                # 顯示 caption（截斷顯示）
                caption_preview = caption[:150] + '...' if len(caption) > 150 else caption
                print(f"  📝 Caption: {caption_preview}")
                print(f"  📏 Caption 長度: {len(caption)} 字元")
                print(f"{'='*60}")
                print(f"[STEP {step}] 開始文本編碼...", flush=True)

            with torch.no_grad():
                # Suppress CLIP truncation warnings - only show simplified message
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Temporarily suppress stdout/stderr to catch truncation messages
                    f = StringIO()
                    with redirect_stdout(f), redirect_stderr(f):
                        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                            prompt=caption,
                            prompt_2=None,
                            num_images_per_prompt=1,
                            max_sequence_length=int(config.get("max_sequence_length", 512)),
                            device=device,  # Explicitly pass device to avoid _execution_device access issues
                        )
                    
                    # Check if truncation occurred and show simplified message
                    output = f.getvalue()
                    if "truncated" in output.lower() or ("clip" in output.lower() and "token" in output.lower()):
                        # Only show once every 10 steps to avoid spam
                        if step % 10 == 0:
                            print(f"[STEP {step}] 注意: 部分文本因長度限制已截斷", flush=True)

                prompt_embeds = prompt_embeds.to(device=device, dtype=torch.bfloat16)   # (1, 512, 4096)
                pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=torch.bfloat16)     # (1, 768)
                text_ids = text_ids.to(device=device, dtype=torch.bfloat16)  # (512, 3)

                if step == 0 or step % 10 == 0:
                    print(f"[STEP {step}] 文本編碼完成", flush=True)
                    print(f"[STEP {step}] 開始 Adapter 圖像編碼...", flush=True)

                # Get dtype: if DataParallel wrapped, use saved dtype; otherwise use model.dtype
                model_dtype = transformer_dtype if use_multi_gpu else pipeline.transformer.dtype
                adapter_image, _, _ = pipeline.prepare_image(
                    image=adapter_img,
                    width=W,
                    height=H,
                    batch_size=1,
                    num_images_per_prompt=1,
                    device=device,
                    dtype=model_dtype,
                )

            if step == 0 or step % 10 == 0:
                print(f"[STEP {step}] Adapter 圖像編碼完成", flush=True)
                print(f"[STEP {step}] 開始 VAE 編碼目標圖層 (共 {len(layer_boxes)} 層)...", flush=True)

            x0, latent_image_ids = encode_target_latents(pipeline, pixel_RGB.unsqueeze(0), n_layers=len(layer_boxes), list_layer_box=layer_boxes)
            _, L, C_lat, H_lat, W_lat = x0.shape

            if step == 0 or step % 10 == 0:
                print(f"[STEP {step}] VAE 編碼完成 (latent shape: {x0.shape})", flush=True)
                print(f"[STEP {step}] 準備噪聲和 timestep...", flush=True)

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
            # Use saved config if DataParallel wrapped, otherwise access directly
            model_config = transformer_config if use_multi_gpu else pipeline.transformer.config
            if model_config.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                guidance = guidance.expand(x0.shape[0])
            else:
                guidance = None

            pipeline.transformer.train()
            pipeline.multiLayerAdapter.train()

            if step == 0 or step % 10 == 0:
                print(f"[STEP {step}] 開始 MultiLayer Adapter 前向傳播...", flush=True)

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

            if step == 0 or step % 10 == 0:
                print(f"[STEP {step}] MultiLayer Adapter 完成，開始 Transformer (DiT) 前向傳播...", flush=True)

            # Get dtype: if DataParallel wrapped, use saved dtype; otherwise use model.dtype
            model_dtype = transformer_dtype if use_multi_gpu else pipeline.transformer.dtype
            v_pred = pipeline.transformer(
                hidden_states=xt,
                adapter_block_samples=[
                    sample.to(dtype=model_dtype)
                    for sample in adapter_block_samples
                ],
                adapter_single_block_samples=[
                    sample.to(dtype=model_dtype)
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

            if step == 0 or step % 10 == 0:
                print(f"[STEP {step}] Transformer 前向傳播完成，計算 loss...", flush=True)

            # MSE（masked）
            mse = (v_pred - v_star) ** 2
            mse = mse.mean(dim=2, keepdim=True)  # [1,L,1,H_lat,W_lat]
            loss = (mse * mask).sum() / (mask.sum() + 1e-8)

            loss = loss / accum_steps
            
            if step == 0 or step % 10 == 0:
                print(f"[STEP {step}] Loss: {loss.item():.6f}，開始反向傳播...", flush=True)
            
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
                
                if step == 0 or step % 10 == 0:
                    print(f"[STEP {step}] 權重更新完成！", flush=True)

            step += 1
            if step % log_every == 0:
                pbar.set_postfix(loss=float(loss.detach().cpu()))
                pbar.update(log_every)
                print(f"[INFO] Step {step}/{max_steps} 完成，Loss: {loss.item():.6f}", flush=True)
            elif step % 10 == 0:
                # 每 10 步也顯示簡短進度
                print(f"[INFO] Step {step}/{max_steps}，Loss: {loss.item():.6f}", flush=True)

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