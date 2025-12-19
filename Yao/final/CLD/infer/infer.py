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
from torch.utils.data import DataLoader

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
    if 'pretrained_lora_dir' in config:
        print("[INFO] Loading LoRA weights...", flush=True)
        lora_state_dict = CustomFluxPipeline.lora_state_dict(config['pretrained_lora_dir'])
        CustomFluxPipeline.load_lora_into_transformer(lora_state_dict, None, transformer)
        transformer.fuse_lora(safe_fusing=True)
        transformer.unload_lora()
        print("[INFO] Successfully loaded and fused LoRA weights.", flush=True)

    if 'artplus_lora_dir' in config:
        print("[INFO] Loading LoRA weights...", flush=True)
        lora_state_dict = CustomFluxPipeline.lora_state_dict(config['artplus_lora_dir'])
        CustomFluxPipeline.load_lora_into_transformer(lora_state_dict, None, transformer)
        transformer.fuse_lora(safe_fusing=True)
        transformer.unload_lora()
        print("[INFO] Successfully loaded and fused LoRA weights.", flush=True)

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
    if 'adapter_lora_dir' in config:
        print("[INFO] Loading MultiLayer-Adapter LoRA weights...", flush=True)
        lora_state_dict = CustomFluxPipeline.lora_state_dict(config['adapter_lora_dir'])
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
    list_layer_box = []
    for layer_box in layer_boxes:
        min_row, max_row = layer_box[1], layer_box[3]
        min_col, max_col = layer_box[0], layer_box[2]
        quantized_min_row = (min_row // 16) * 16
        quantized_min_col = (min_col // 16) * 16
        quantized_max_row = ((max_row // 16) + 1) * 16
        quantized_max_col = ((max_col // 16) + 1) * 16

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

    # 嘗試載入 dataset，如果 custom_dataset 失敗則 fallback 到原始 dataset
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
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(config['seed'])

    idx = 0
    for batch in loader:
        print(f"Processing case {idx}", flush=True)

        height = int(batch["height"][0])
        width = int(batch["width"][0])
        
        # 調整為 16 的倍數（向上取整）- 因為 latent space 需要 /16
        original_height = height
        original_width = width
        height = ((height + 15) // 16) * 16  # 向上取整到最近的 16 的倍數
        width = ((width + 15) // 16) * 16     # 向上取整到最近的 16 的倍數
        
        if height != original_height or width != original_width:
            print(f"[INFO] 調整圖片尺寸: {original_height}x{original_width} -> {height}x{width} (必須是 16 的倍數)", flush=True)
        
        adapter_img = batch["whole_img"][0]
        caption = batch["caption"][0]
        
        # 計算尺寸調整比例
        scale_h = height / original_height if original_height > 0 else 1.0
        scale_w = width / original_width if original_width > 0 else 1.0
        
        # 調整 layout 座標以匹配調整後的尺寸
        original_layout = batch["layout"][0]
        adjusted_layout = []
        for layer_box in original_layout:
            # layer_box 格式: [x1, y1, x2, y2]
            x1, y1, x2, y2 = layer_box
            # 按比例調整座標（四捨五入以保持精度）
            adjusted_x1 = round(x1 * scale_w)
            adjusted_y1 = round(y1 * scale_h)
            adjusted_x2 = round(x2 * scale_w)
            adjusted_y2 = round(y2 * scale_h)
            
            adjusted_layout.append([adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2])
        
        # 使用調整後的 layout 計算 layer_boxes
        layer_boxes = get_input_box(adjusted_layout) 

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
        # Resize back to original size if needed
        if height != original_height or width != original_width:
            whole_image_rgba_image = whole_image_rgba_image.resize((original_width, original_height), Image.LANCZOS)
        whole_image_rgba_image.save(os.path.join(case_dir, "whole_image_rgba.png"))

        adapter_img.save(os.path.join(case_dir, "origin.png"))

        background_layer = (x_hat[1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        background_rgba_image = Image.fromarray(background_layer, "RGBA")
        # Resize back to original size if needed
        if height != original_height or width != original_width:
            background_rgba_image = background_rgba_image.resize((original_width, original_height), Image.LANCZOS)
        background_rgba_image.save(os.path.join(case_dir, "background_rgba.png"))

        x_hat = x_hat[2:]
        merged_image = image[1]
        image = image[2:]
        
        # Resize merged_image if needed (before compositing layers)
        if height != original_height or width != original_width:
            merged_image = merged_image.resize((original_width, original_height), Image.LANCZOS)

        # Save transparent VAE decoded results
        for layer_idx in range(x_hat.shape[0]):
            layer = x_hat[layer_idx]
            rgba_layer = (layer.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rgba_image = Image.fromarray(rgba_layer, "RGBA")
            # Resize back to original size if needed
            if height != original_height or width != original_width:
                rgba_image = rgba_image.resize((original_width, original_height), Image.LANCZOS)
            rgba_image.save(os.path.join(case_dir, f"layer_{layer_idx}_rgba.png"))

        # Composite background and foreground layers
        for layer_idx in range(x_hat.shape[0]):
            layer = x_hat[layer_idx]
            rgba_layer = (layer.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            layer_image = Image.fromarray(rgba_layer, "RGBA")
            # Resize back to original size if needed
            if height != original_height or width != original_width:
                layer_image = layer_image.resize((original_width, original_height), Image.LANCZOS)
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
