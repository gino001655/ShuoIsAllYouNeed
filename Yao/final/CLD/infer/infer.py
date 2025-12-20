import os
# Set CUDA_VISIBLE_DEVICES before importing torch
# You can modify this or set it via environment variable
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch
import argparse
from PIL import Image
from diffusers import FluxTransformer2DModel
from diffusers.configuration_utils import FrozenDict
from torch.utils.data import DataLoader, Subset

from models.multiLayer_adapter import MultiLayerAdapter
from models.mmdit import CustomFluxTransformer2DModel
from models.pipeline import CustomFluxPipeline, CustomFluxPipelineCfgLayer
from models.transp_vae import AutoencoderKLTransformerTraining as CustomVAE
from tools.tools import load_config, seed_everything
# 嘗試載入自訂資料集，如果失敗則使用原始資料集
_use_custom_dataset = False
try:
    from tools.custom_dataset import LayoutTrainDataset as CustomLayoutTrainDataset, collate_fn as custom_collate_fn
    _use_custom_dataset = True
    print("[INFO] 自訂資料集模組載入成功，將嘗試使用 (custom_dataset.py)")
except ImportError:
    from tools.dataset import LayoutTrainDataset, collate_fn
    print("[INFO] 自訂資料集模組不存在，使用原始資料集 (dataset.py)")


def get_lora_path(lora_dir_or_file):
    """
    將 LoRA 目錄路徑或文件路徑轉換為文件路徑。
    如果輸入是目錄，自動拼接 'pytorch_lora_weights.safetensors'。
    如果輸入是文件，直接返回。
    """
    if os.path.isfile(lora_dir_or_file):
        return lora_dir_or_file
    elif os.path.isdir(lora_dir_or_file):
        lora_file = os.path.join(lora_dir_or_file, "pytorch_lora_weights.safetensors")
        if os.path.exists(lora_file):
            return lora_file
        else:
            raise FileNotFoundError(
                f"LoRA directory '{lora_dir_or_file}' does not contain 'pytorch_lora_weights.safetensors'. "
                f"Please check the path or provide the full file path instead."
            )
    else:
        raise FileNotFoundError(f"LoRA path does not exist: {lora_dir_or_file}")


# Initialize pipeline
def initialize_pipeline(config):
    # Determine device based on CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"[INFO] CUDA available. GPU count: {torch.cuda.device_count()}, Current GPU: {torch.cuda.current_device()}, GPU name: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("[WARNING] CUDA not available! Inference will run on CPU (very slow).", flush=True)
    
    print("[INFO] Loading pretrained Transformer model...", flush=True)
    transformer_orig = FluxTransformer2DModel.from_pretrained(
        config.get('transformer_varient', config['pretrained_model_name_or_path']),
        subfolder="" if 'transformer_varient' in config else "transformer",
        revision=config.get('revision', None),
        variant=config.get('variant', None),
        torch_dtype=torch.bfloat16,
        cache_dir=config.get('cache_dir', None),
    )
    print("[INFO] Successfully loaded pretrained Transformer model.", flush=True)

    print("[INFO] Loading custom Transformer configuration...", flush=True)
    mmdit_config = dict(transformer_orig.config)
    mmdit_config["_class_name"] = "CustomSD3Transformer2DModel"
    mmdit_config["max_layer_num"] = config['max_layer_num']
    mmdit_config = FrozenDict(mmdit_config)

    print("[INFO] Initializing custom Transformer model...", flush=True)
    transformer = CustomFluxTransformer2DModel.from_config(mmdit_config).to(dtype=torch.bfloat16)
    print("[INFO] Successfully initialized custom Transformer model.", flush=True)

    print("[INFO] Loading Transformer weights...", flush=True)
    missing_keys, unexpected_keys = transformer.load_state_dict(transformer_orig.state_dict(), strict=False)
    if missing_keys:
        print(f"[WARNING] Missing keys: {missing_keys}", flush=True)
    if unexpected_keys:
        print(f"[WARNING] Unexpected keys: {unexpected_keys}", flush=True)
    print("[INFO] Successfully loaded Transformer weights.", flush=True)

    # Load LoRA weights
    if 'pretrained_lora_dir' in config and config['pretrained_lora_dir']:
        print("[INFO] Loading pretrained LoRA weights...", flush=True)
        lora_path = get_lora_path(config['pretrained_lora_dir'])
        lora_state_dict = CustomFluxPipeline.lora_state_dict(lora_path)
        CustomFluxPipeline.load_lora_into_transformer(lora_state_dict, None, transformer)
        transformer.fuse_lora(safe_fusing=True)
        transformer.unload_lora()
        print("[INFO] Successfully loaded and fused pretrained LoRA weights.", flush=True)

    if 'artplus_lora_dir' in config and config['artplus_lora_dir']:
        print("[INFO] Loading artplus LoRA weights...", flush=True)
        lora_path = get_lora_path(config['artplus_lora_dir'])
        lora_state_dict = CustomFluxPipeline.lora_state_dict(lora_path)
        CustomFluxPipeline.load_lora_into_transformer(lora_state_dict, None, transformer)
        transformer.fuse_lora(safe_fusing=True)
        transformer.unload_lora()
        print("[INFO] Successfully loaded and fused artplus LoRA weights.", flush=True)

    # Load layer_pe weights
    layer_pe_path = os.path.join(config['layer_ckpt'], "layer_pe.pth")
    if os.path.exists(layer_pe_path):
        print("[INFO] Loading layer_pe weights...", flush=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer_pe = torch.load(layer_pe_path, map_location=device)
        missing_keys, unexpected_keys = transformer.load_state_dict(layer_pe, strict=False)
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys in layer_pe: {unexpected_keys}", flush=True)
        print("[INFO] Successfully loaded layer_pe weights.", flush=True)
    else:
        print(f"[WARNING] Could not find layer_pe weights file: {layer_pe_path}", flush=True)

    # Load MultiLayer-Adapter
    print("[INFO] Loading MultiLayer-Adapter weights...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multiLayer_adapter = MultiLayerAdapter.from_pretrained(config['pretrained_adapter_path']).to(torch.bfloat16).to(device)
    print("[INFO] Successfully loaded MultiLayer-Adapter weights.", flush=True)
    if 'adapter_lora_dir' in config and config['adapter_lora_dir']:
        print("[INFO] Loading MultiLayer-Adapter LoRA weights...", flush=True)
        lora_path = get_lora_path(config['adapter_lora_dir'])
        lora_state_dict = CustomFluxPipeline.lora_state_dict(lora_path)
        CustomFluxPipeline.load_lora_into_transformer(lora_state_dict, None, multiLayer_adapter)
        multiLayer_adapter.fuse_lora(safe_fusing=True)
        multiLayer_adapter.unload_lora()
        print("[INFO] Successfully loaded and fused MultiLayer-Adapter LoRA weights.", flush=True)        
    multiLayer_adapter.set_layerPE(transformer.layer_pe, transformer.max_layer_num)

    print("[INFO] Initializing CustomFluxPipeline...", flush=True)
    pipeline_type = CustomFluxPipelineCfgLayer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline_type.from_pretrained(
        config['pretrained_model_name_or_path'],
        transformer=transformer,
        revision=config.get('revision', None),
        variant=config.get('variant', None),
        torch_dtype=torch.bfloat16,
        cache_dir=config.get('cache_dir', None),
    ).to(device)
    pipeline.set_multiLayerAdapter(multiLayer_adapter)
    print("[INFO] Successfully initialized CustomFluxPipeline.", flush=True)

    print("[INFO] Loading pipeline LoRA weights...", flush=True)
    pipeline.load_lora_weights(config['lora_ckpt'], adapter_name="layer")
    print("[INFO] Successfully loaded pipeline LoRA weights.", flush=True)

    return pipeline

# Calculate layer boxes
def get_input_box(layer_boxes):
    """
    Convert layer boxes to quantized format (aligned to 16-pixel boundaries).
    Uses correct ceiling division: ((val + 15) // 16) * 16
    """
    list_layer_box = []
    for layer_box in layer_boxes:
        if layer_box is None or len(layer_box) < 4:
            list_layer_box.append(None)
            continue
        min_row, max_row = layer_box[1], layer_box[3]
        min_col, max_col = layer_box[0], layer_box[2]
        # Downward quantization (floor to 16)
        quantized_min_row = (min_row // 16) * 16
        quantized_min_col = (min_col // 16) * 16
        # Upward quantization (ceiling to 16) - correct way
        quantized_max_row = ((max_row + 15) // 16) * 16
        quantized_max_col = ((max_col + 15) // 16) * 16
        
        # Ensure valid box after quantization
        if quantized_max_col <= quantized_min_col or quantized_max_row <= quantized_min_row:
            list_layer_box.append(None)
            continue

        list_layer_box.append((quantized_min_col, quantized_min_row, quantized_max_col, quantized_max_row))
    return list_layer_box

@torch.no_grad()
def inference_layout(config):
    if config['seed'] is not None:
        seed_everything(config['seed'])
    
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], "merged"), exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], "merged_rgba"), exist_ok=True)

    # Load transparent VAE
    print("[INFO] Loading Transparent VAE...", flush=True)
    
    vae_args = argparse.Namespace(
        max_layers=config.get('max_layers', 48),
        decoder_arch=config.get('decoder_arch', 'vit'),
        pos_embedding=config.get('pos_embedding', 'rope'),
        layer_embedding=config.get('layer_embedding', 'rope'),
        single_layer_decoder=config.get('single_layer_decoder', None)
    )
    transp_vae = CustomVAE(vae_args)
    transp_vae_path = config.get('transp_vae_path')
    # Determine device based on CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading Transparent VAE on device: {device}", flush=True)
    transp_vae_weights = torch.load(transp_vae_path, map_location=device)
    missing_keys, unexpected_keys = transp_vae.load_state_dict(transp_vae_weights['model'], strict=False)
    if missing_keys or unexpected_keys:
        print(f"[WARNING] Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
    transp_vae.eval()
    transp_vae = transp_vae.to(device)
    print("[INFO] Transparent VAE loaded successfully.", flush=True)

    pipeline = initialize_pipeline(config)

    # 載入 dataset
    use_indexed_dataset = config.get('use_indexed_dataset', False)
    enable_dataset_debug = config.get('enable_dataset_debug', False)
    max_samples = config.get('max_samples', None)
    
    print("\n" + "="*60)
    print("載入推理數據集")
    print("="*60)
    
    if use_indexed_dataset:
        # 方案 B: 使用 indexed dataset (TAData + caption.json)
        print(f"[INFO] 使用 DLCVLayoutDatasetIndexed (Index-based caption matching)", flush=True)
        print(f"[INFO] Data dir: {config['data_dir']}", flush=True)
        print(f"[INFO] Caption JSON: {config.get('caption_json', 'Not specified')}", flush=True)
        
        from tools.dlcv_dataset_indexed import DLCVLayoutDatasetIndexed, collate_fn as indexed_collate_fn
        
        if enable_dataset_debug:
            print(f"[INFO] 🔍 Dataset debug enabled: 將顯示每個樣本的詳細資訊", flush=True)
        
        dataset = DLCVLayoutDatasetIndexed(
            data_dir=config['data_dir'],
            caption_json_path=config.get('caption_json', None),
            enable_debug=enable_dataset_debug,
        )
        
        # 限制樣本數量（如果指定）
        if max_samples is not None and max_samples > 0:
            print(f"[INFO] 限制樣本數量: {max_samples}", flush=True)
            # 創建一個子集
            dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=indexed_collate_fn)
        print(f"[INFO] ✓ 載入 {len(dataset) if not isinstance(dataset, Subset) else len(dataset.indices)} 個推理樣本", flush=True)
    else:
        # 方案 A 或原始方案：嘗試載入 dataset，如果 custom_dataset 失敗則 fallback 到原始 dataset
        if _use_custom_dataset:
            try:
                dataset = CustomLayoutTrainDataset(config['data_dir'], split="test")
                collate_fn = custom_collate_fn
                print("[INFO] 成功使用自訂資料集 (custom_dataset.py)", flush=True)
            except (FileNotFoundError, ValueError) as e:
                print(f"[WARNING] 自訂資料集載入失敗: {e}", flush=True)
                print("[INFO] 切換到原始資料集 (dataset.py)", flush=True)
                from tools.dataset import LayoutTrainDataset, collate_fn
                dataset = LayoutTrainDataset(config['data_dir'], split="test")
        else:
            from tools.dataset import LayoutTrainDataset, collate_fn
            dataset = LayoutTrainDataset(config['data_dir'], split="test")
        
        # 限制樣本數量（如果指定）
        if max_samples is not None and max_samples > 0:
            print(f"[INFO] 限制樣本數量: {max_samples}", flush=True)
            dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
        print(f"[INFO] ✓ 載入 {len(dataset) if not isinstance(dataset, Subset) else len(dataset.indices)} 個推理樣本", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(config['seed'])

    # Helper function to safely extract batch values
    # Handles both formats: single sample dict (from indexed/standard collate_fn) or dict with lists
    def get_batch_value(batch_dict, key):
        """Get value from batch, handling both list and direct value formats."""
        if key not in batch_dict:
            raise KeyError(f"Batch does not contain key '{key}'. Available keys: {list(batch_dict.keys())}")
        value = batch_dict[key]
        # If value is a list/tuple with elements, return first element
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return value[0]
        # Otherwise return value directly (already a single value)
        return value

    idx = 0
    for batch in loader:
        print(f"Processing case {idx}", flush=True)

        # Extract batch data - automatically handles different collate_fn formats
        try:
            height = int(get_batch_value(batch, "height"))
            width = int(get_batch_value(batch, "width"))
            adapter_img = get_batch_value(batch, "whole_img")
            caption = get_batch_value(batch, "caption")
            layout = get_batch_value(batch, "layout")
        except (KeyError, TypeError, ValueError) as e:
            print(f"[ERROR] Failed to extract batch data: {e}", flush=True)
            print(f"[DEBUG] Batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'Not a dict'}", flush=True)
            print(f"[DEBUG] Batch type: {type(batch)}", flush=True)
            idx += 1
            continue
        
        raw_layer_boxes = get_input_box(layout)
        layer_boxes = []
        for box in raw_layer_boxes:
            if box is None or len(box) < 4:
                continue
            x1, y1, x2, y2 = box[:4]
            # Check: box must have positive area
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue
            # Check: after quantization (divide by 16 in model), box must still have positive area
            # Model does: x1, y1, x2, y2 = x1 // 16, y1 // 16, x2 // 16, y2 // 16
            quantized_x1, quantized_y1 = x1 // 16, y1 // 16
            quantized_x2, quantized_y2 = x2 // 16, y2 // 16
            if (quantized_x2 - quantized_x1) <= 0 or (quantized_y2 - quantized_y1) <= 0:
                continue
            # Check: box must be within canvas bounds (with some tolerance for quantization)
            # Note: get_input_box already quantized, so boxes should be roughly in bounds
            # But we check anyway for safety
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                # Clamp to canvas bounds
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(x1, min(x2, width))
                y2 = max(y1, min(y2, height))
                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue
                # Re-check quantization after clamping (box is already quantized, but ensure it's still valid)
                quantized_x1_after, quantized_y1_after = x1 // 16, y1 // 16
                quantized_x2_after, quantized_y2_after = x2 // 16, y2 // 16
                if (quantized_x2_after - quantized_x1_after) <= 0 or (quantized_y2_after - quantized_y1_after) <= 0:
                    continue
                box = (x1, y1, x2, y2)
            layer_boxes.append(box)
        
        if len(layer_boxes) == 0:
            print(f"[WARN] Sample {idx}: No valid layer boxes after filtering; skip", flush=True)
            idx += 1
            continue
        
        if len(layer_boxes) != len(raw_layer_boxes):
            print(
                f"[WARN] Sample {idx}: Filtered invalid boxes: {len(raw_layer_boxes)} -> {len(layer_boxes)}",
                flush=True,
            )

        # Generate layers using pipeline
        x_hat, image, latents = pipeline(
            prompt=caption,
            adapter_image=adapter_img,
            adapter_conditioning_scale=0.9,
            validation_box=layer_boxes,
            generator=generator,
            height=height,
            width=width,
            guidance_scale=config.get('cfg', 4.0),
            num_layers=len(layer_boxes),
            sdxl_vae=transp_vae,  # Use transparent VAE
        )

        # Adjust x_hat range from [-1, 1] to [0, 1]
        x_hat = (x_hat + 1) / 2

        # Remove batch dimension and ensure float32 dtype
        x_hat = x_hat.squeeze(0).permute(1, 0, 2, 3).to(torch.float32)
        
        this_index = f"case_{idx}"
        case_dir = os.path.join(config['save_dir'], this_index)
        os.makedirs(case_dir, exist_ok=True)
        
        # Save whole image_RGBA (X_hat[0]) and background_RGBA (X_hat[1])
        whole_image_layer = (x_hat[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        whole_image_rgba_image = Image.fromarray(whole_image_layer, "RGBA")
        whole_image_rgba_image.save(os.path.join(case_dir, "whole_image_rgba.png"))

        adapter_img.save(os.path.join(case_dir, "origin.png"))

        background_layer = (x_hat[1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        background_rgba_image = Image.fromarray(background_layer, "RGBA")
        background_rgba_image.save(os.path.join(case_dir, "background_rgba.png"))

        x_hat = x_hat[2:]
        merged_image = image[1]
        image = image[2:]

        # Save transparent VAE decoded results
        for layer_idx in range(x_hat.shape[0]):
            layer = x_hat[layer_idx]
            rgba_layer = (layer.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rgba_image = Image.fromarray(rgba_layer, "RGBA")
            rgba_image.save(os.path.join(case_dir, f"layer_{layer_idx}_rgba.png"))

        # Composite background and foreground layers
        for layer_idx in range(x_hat.shape[0]):
            rgba_layer = (x_hat[layer_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            layer_image = Image.fromarray(rgba_layer, "RGBA")
            merged_image = Image.alpha_composite(merged_image.convert('RGBA'), layer_image)
        
        # Save final composite images
        merged_image.convert('RGB').save(os.path.join(config['save_dir'], "merged", f"{this_index}.png"))
        merged_image.convert('RGB').save(os.path.join(case_dir, f"{this_index}.png"))
        # Save final composite RGBA image
        merged_image.save(os.path.join(config['save_dir'], "merged_rgba", f"{this_index}.png"))

        print(f"Saved case {idx} to {case_dir}")
        idx += 1

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config = load_config(args.config_path)

    inference_layout(config)
