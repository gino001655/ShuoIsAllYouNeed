#!/usr/bin/env python3
"""
單張圖片 CLD Inference 範例
展示如何準備輸入格式並執行 inference
"""
import os
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
from tools.tools import load_config, seed_everything, get_input_box
from infer.infer import initialize_pipeline


def prepare_single_image_input(image_path, caption, layout_boxes):
    """
    準備單張圖片的輸入格式
    
    Args:
        image_path: 圖片路徑
        caption: 圖片描述文字
        layout_boxes: 圖層邊界框列表 [[w0, h0, w1, h1], ...]
    
    Returns:
        dict: 包含所有必要輸入的字典
    """
    # 載入圖片
    whole_img = Image.open(image_path).convert("RGB")
    width, height = whole_img.size
    
    # 確保 layout 包含整個畫布和背景
    if len(layout_boxes) == 0:
        # 如果沒有提供，預設為整個畫布
        layout_boxes = [
            [0, 0, width-1, height-1],  # 整個畫布
            [0, 0, width-1, height-1],  # 背景
        ]
    elif len(layout_boxes) == 1:
        # 如果只有一個，添加背景
        layout_boxes = [
            [0, 0, width-1, height-1],  # 整個畫布
        ] + layout_boxes
    
    return {
        "whole_img": whole_img,
        "caption": caption,
        "width": width,
        "height": height,
        "layout": layout_boxes,
    }


@torch.no_grad()
def inference_single_image(config, image_path, caption, layout_boxes=None, output_dir=None):
    """
    對單張圖片執行 CLD inference
    
    Args:
        config: 配置字典
        image_path: 輸入圖片路徑
        caption: 圖片描述
        layout_boxes: 圖層邊界框列表（可選）
        output_dir: 輸出目錄（可選）
    """
    if config['seed'] is not None:
        seed_everything(config['seed'])
    
    # 設定輸出目錄
    if output_dir is None:
        output_dir = config.get('save_dir', './output_single')
    os.makedirs(output_dir, exist_ok=True)
    
    # 準備輸入
    print("[INFO] 準備輸入資料...")
    inputs = prepare_single_image_input(image_path, caption, layout_boxes or [])
    
    width = inputs["width"]
    height = inputs["height"]
    adapter_img = inputs["whole_img"]
    caption = inputs["caption"]
    layout = inputs["layout"]
    
    # 轉換 layout 格式（量化邊界框）
    layer_boxes = get_input_box(layout)
    
    print(f"[INFO] 圖片尺寸: {width} x {height}")
    print(f"[INFO] 描述: {caption}")
    print(f"[INFO] 圖層數: {len(layer_boxes)}")
    
    # 載入 Transparent VAE
    print("[INFO] 載入 Transparent VAE...")
    vae_args = argparse.Namespace(
        max_layers=config.get('max_layers', 48),
        decoder_arch=config.get('decoder_arch', 'vit'),
        pos_embedding=config.get('pos_embedding', 'rope'),
        layer_embedding=config.get('layer_embedding', 'rope'),
        single_layer_decoder=config.get('single_layer_decoder', None)
    )
    transp_vae = CustomVAE(vae_args)
    transp_vae_path = config.get('transp_vae_path')
    transp_vae_weights = torch.load(transp_vae_path, map_location=torch.device("cuda"))
    transp_vae.load_state_dict(transp_vae_weights['model'], strict=False)
    transp_vae.eval()
    transp_vae = transp_vae.to(torch.device("cuda"))
    print("[INFO] Transparent VAE 載入完成")
    
    # 初始化 pipeline
    print("[INFO] 初始化 pipeline...")
    pipeline = initialize_pipeline(config)
    
    # 建立隨機生成器
    generator = torch.Generator(device=torch.device("cuda")).manual_seed(config['seed'])
    
    # 執行 inference
    print("[INFO] 開始生成圖層...")
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
        sdxl_vae=transp_vae,
    )
    
    # 處理輸出
    print("[INFO] 處理輸出結果...")
    x_hat = (x_hat + 1) / 2  # 調整範圍從 [-1, 1] 到 [0, 1]
    x_hat = x_hat.squeeze(0).permute(1, 0, 2, 3).to(torch.float32)
    
    # 保存結果
    # 完整圖片
    whole_image_layer = (x_hat[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    whole_image_rgba = Image.fromarray(whole_image_layer, "RGBA")
    whole_image_rgba.save(os.path.join(output_dir, "whole_image_rgba.png"))
    
    # 原始圖片
    adapter_img.save(os.path.join(output_dir, "origin.png"))
    
    # 背景
    background_layer = (x_hat[1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    background_rgba = Image.fromarray(background_layer, "RGBA")
    background_rgba.save(os.path.join(output_dir, "background_rgba.png"))
    
    # 各個圖層
    x_hat = x_hat[2:]
    merged_image = image[1]
    image = image[2:]
    
    for layer_idx in range(x_hat.shape[0]):
        layer = x_hat[layer_idx]
        rgba_layer = (layer.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rgba_image = Image.fromarray(rgba_layer, "RGBA")
        rgba_image.save(os.path.join(output_dir, f"layer_{layer_idx}_rgba.png"))
    
    # 合成最終結果
    for layer_idx in range(x_hat.shape[0]):
        rgba_layer = (x_hat[layer_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        layer_image = Image.fromarray(rgba_layer, "RGBA")
        merged_image = Image.alpha_composite(merged_image.convert('RGBA'), layer_image)
    
    # 保存最終結果
    merged_image.convert('RGB').save(os.path.join(output_dir, "merged.png"))
    merged_image.save(os.path.join(output_dir, "merged_rgba.png"))
    
    print(f"[INFO] 結果已保存到: {output_dir}")
    
    # 清理
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLD 單張圖片 Inference")
    parser.add_argument("--config_path", "-c", type=str, required=True, 
                       help="配置檔案路徑 (YAML)")
    parser.add_argument("--image_path", "-i", type=str, required=True,
                       help="輸入圖片路徑")
    parser.add_argument("--caption", "-p", type=str, required=True,
                       help="圖片描述文字")
    parser.add_argument("--layout", "-l", type=str, default=None,
                       help="圖層邊界框 (JSON 格式: [[w0,h0,w1,h1],...])")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                       help="輸出目錄")
    
    args = parser.parse_args()
    
    # 載入配置
    config = load_config(args.config_path)
    
    # 解析 layout（如果提供）
    layout_boxes = None
    if args.layout:
        import json
        layout_boxes = json.loads(args.layout)
    
    # 執行 inference
    inference_single_image(
        config=config,
        image_path=args.image_path,
        caption=args.caption,
        layout_boxes=layout_boxes,
        output_dir=args.output_dir,
    )







