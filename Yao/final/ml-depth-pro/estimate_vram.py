"""
VRAM 使用量估算腳本

計算訓練 Layer Order Model 所需的 VRAM
"""

import torch
import torch.nn as nn
from custom_layer_order_model import (
    create_layer_order_model_and_transforms,
    LayerOrderConfig,
    DEFAULT_LAYER_ORDER_CONFIG,
)


def estimate_model_size(model: nn.Module) -> dict:
    """估算模型大小"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # FP32: 4 bytes per parameter
    # FP16: 2 bytes per parameter
    model_size_fp32 = total_params * 4 / (1024 ** 3)  # GB
    model_size_fp16 = total_params * 2 / (1024 ** 3)  # GB
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_fp32_gb": model_size_fp32,
        "model_size_fp16_gb": model_size_fp16,
    }


def estimate_vram_usage(
    model: nn.Module,
    batch_size: int = 4,
    image_size: int = 1536,
    use_amp: bool = True,
    use_gradient_checkpointing: bool = False,
) -> dict:
    """
    估算 VRAM 使用量
    
    Args:
    ----
        model: 模型
        batch_size: 批次大小
        image_size: 輸入圖像尺寸
        use_amp: 是否使用混合精度訓練
        use_gradient_checkpointing: 是否使用梯度檢查點
    """
    # 1. 模型參數大小
    model_info = estimate_model_size(model)
    total_params = model_info["total_params"]
    trainable_params = model_info["trainable_params"]
    
    # 根據精度計算模型參數佔用
    if use_amp:
        # 混合精度：模型參數用 FP16，但優化器狀態用 FP32
        model_params_gb = total_params * 2 / (1024 ** 3)  # FP16
    else:
        model_params_gb = total_params * 4 / (1024 ** 3)  # FP32
    
    # 2. 梯度大小（與可訓練參數相同）
    if use_amp:
        gradients_gb = trainable_params * 2 / (1024 ** 3)  # FP16
    else:
        gradients_gb = trainable_params * 4 / (1024 ** 3)  # FP32
    
    # 3. 優化器狀態（AdamW 需要 2 個狀態：momentum 和 variance，都是 FP32）
    optimizer_states_gb = trainable_params * 2 * 4 / (1024 ** 3)  # 2 states * FP32
    
    # 4. 激活值估算（粗略估算）
    # 輸入圖像: [B, 3, H, W]
    input_size = batch_size * 3 * image_size * image_size * 4 / (1024 ** 3)  # FP32
    
    # Encoder 激活值（DINOv2 Large）
    # Patch size: 16, 所以有 (1536/16)^2 = 96^2 = 9216 patches
    # Hidden dim: 1024 (DINOv2 Large)
    # 多層激活值，粗略估算為輸入的 10-20 倍
    encoder_activations_gb = input_size * 15  # 粗略估算
    
    # Decoder 激活值（多尺度特徵）
    # 假設 decoder 有 4 個尺度，每個尺度約為輸入的 1/4, 1/8, 1/16, 1/32
    decoder_activations_gb = input_size * 2  # 粗略估算
    
    # Head 激活值
    head_activations_gb = input_size * 0.5
    
    total_activations_gb = encoder_activations_gb + decoder_activations_gb + head_activations_gb
    
    # 如果使用梯度檢查點，激活值可以減少 50-70%
    if use_gradient_checkpointing:
        total_activations_gb *= 0.4
    
    # 5. 其他開銷（CUDA 核心、框架開銷等）
    overhead_gb = 2.0
    
    # 總計
    total_vram_gb = (
        model_params_gb +
        gradients_gb +
        optimizer_states_gb +
        total_activations_gb +
        overhead_gb
    )
    
    return {
        "model_params_gb": model_params_gb,
        "gradients_gb": gradients_gb,
        "optimizer_states_gb": optimizer_states_gb,
        "activations_gb": total_activations_gb,
        "overhead_gb": overhead_gb,
        "total_vram_gb": total_vram_gb,
        "model_info": model_info,
    }


def print_vram_estimate(info: dict):
    """打印 VRAM 估算結果"""
    print("=" * 80)
    print("VRAM 使用量估算")
    print("=" * 80)
    print(f"\n模型資訊:")
    print(f"  總參數數: {info['model_info']['total_params']:,}")
    print(f"  可訓練參數: {info['model_info']['trainable_params']:,}")
    print(f"  模型大小 (FP32): {info['model_info']['model_size_fp32_gb']:.2f} GB")
    print(f"  模型大小 (FP16): {info['model_info']['model_size_fp16_gb']:.2f} GB")
    
    print(f"\nVRAM 使用量分解:")
    print(f"  模型參數: {info['model_params_gb']:.2f} GB")
    print(f"  梯度: {info['gradients_gb']:.2f} GB")
    print(f"  優化器狀態 (AdamW): {info['optimizer_states_gb']:.2f} GB")
    print(f"  激活值: {info['activations_gb']:.2f} GB")
    print(f"  其他開銷: {info['overhead_gb']:.2f} GB")
    print(f"\n  總計: {info['total_vram_gb']:.2f} GB")
    print("=" * 80)


def main():
    """主函數"""
    print("正在載入模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 創建模型
    model, _ = create_layer_order_model_and_transforms(
        config=DEFAULT_LAYER_ORDER_CONFIG,
        device=device,
        precision="fp16",  # 使用 FP16 進行估算
    )
    
    print("\n" + "=" * 80)
    print("場景 1: 標準訓練（Batch Size = 4, FP16, 無梯度檢查點）")
    print("=" * 80)
    info1 = estimate_vram_usage(
        model=model,
        batch_size=4,
        image_size=1536,
        use_amp=True,
        use_gradient_checkpointing=False,
    )
    print_vram_estimate(info1)
    
    print("\n" + "=" * 80)
    print("場景 2: 小批次訓練（Batch Size = 2, FP16, 無梯度檢查點）")
    print("=" * 80)
    info2 = estimate_vram_usage(
        model=model,
        batch_size=2,
        image_size=1536,
        use_amp=True,
        use_gradient_checkpointing=False,
    )
    print_vram_estimate(info2)
    
    print("\n" + "=" * 80)
    print("場景 3: 使用梯度檢查點（Batch Size = 4, FP16, 有梯度檢查點）")
    print("=" * 80)
    info3 = estimate_vram_usage(
        model=model,
        batch_size=4,
        image_size=1536,
        use_amp=True,
        use_gradient_checkpointing=True,
    )
    print_vram_estimate(info3)
    
    print("\n" + "=" * 80)
    print("場景 4: FP32 訓練（Batch Size = 2, FP32, 無梯度檢查點）")
    print("=" * 80)
    info4 = estimate_vram_usage(
        model=model,
        batch_size=2,
        image_size=1536,
        use_amp=False,
        use_gradient_checkpointing=False,
    )
    print_vram_estimate(info4)
    
    print("\n" + "=" * 80)
    print("建議:")
    print("=" * 80)
    print("1. 推薦使用場景 1 或場景 3（FP16 + 梯度檢查點可節省約 40% 激活值 VRAM）")
    print("2. 如果 VRAM 不足，可以:")
    print("   - 減少 batch size（從 4 降到 2）")
    print("   - 啟用梯度檢查點（--use-gradient-checkpointing，如果實現了）")
    print("   - 使用更小的圖像尺寸（但需要修改模型配置）")
    print("3. 最低建議 VRAM: 24 GB (RTX 3090 / A4000)")
    print("4. 推薦 VRAM: 40 GB (A100 40GB / RTX 6000 Ada)")
    print("=" * 80)


if __name__ == "__main__":
    main()




