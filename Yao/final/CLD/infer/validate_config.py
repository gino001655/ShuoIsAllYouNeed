"""
驗證 inference 配置和數據集，不載入模型權重
用於快速檢查配置錯誤和數據集問題
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.tools import load_config


def validate_config(config_path):
    """驗證配置文件和數據集，不載入模型"""
    print("="*60)
    print("Inference 配置驗證")
    print("="*60)
    
    # 1. 載入配置
    print("\n[1] 載入配置文件...")
    try:
        config = load_config(config_path)
        print(f"✓ 配置文件載入成功: {config_path}")
    except Exception as e:
        print(f"✗ 配置文件載入失敗: {e}")
        return False
    
    # 2. 檢查必需參數
    print("\n[2] 檢查必需參數...")
    required_keys = [
        'data_dir',
        'pretrained_model_name_or_path',
        'pretrained_adapter_path',
        'transp_vae_path',
        'layer_ckpt',
        'lora_ckpt',
        'adapter_lora_dir',
        'save_dir',
    ]
    
    missing_keys = []
    for key in required_keys:
        if key not in config or not config[key]:
            missing_keys.append(key)
        else:
            print(f"  ✓ {key}: {config[key]}")
    
    if missing_keys:
        print(f"  ✗ 缺少必需參數: {missing_keys}")
        return False
    
    # 3. 檢查路徑是否存在
    print("\n[3] 檢查路徑...")
    
    # 檢查數據目錄
    data_dir = config.get('data_dir')
    if data_dir and os.path.exists(data_dir):
        print(f"  ✓ 數據目錄存在: {data_dir}")
    else:
        print(f"  ✗ 數據目錄不存在: {data_dir}")
        return False
    
    # 檢查 checkpoint 路徑
    layer_ckpt = config.get('layer_ckpt')
    if layer_ckpt:
        layer_pe_path = os.path.join(layer_ckpt, "layer_pe.pth")
        if os.path.exists(layer_pe_path):
            print(f"  ✓ layer_pe.pth 存在: {layer_pe_path}")
        else:
            print(f"  ⚠ layer_pe.pth 不存在: {layer_pe_path} (訓練時會創建)")
    
    lora_ckpt = config.get('lora_ckpt')
    if lora_ckpt:
        # 檢查是目錄還是文件
        if os.path.isdir(lora_ckpt):
            lora_file = os.path.join(lora_ckpt, "pytorch_lora_weights.safetensors")
            if os.path.exists(lora_file):
                print(f"  ✓ LoRA checkpoint 存在: {lora_file}")
            else:
                print(f"  ✗ LoRA checkpoint 不存在: {lora_file}")
                return False
        elif os.path.isfile(lora_ckpt):
            print(f"  ✓ LoRA checkpoint 存在: {lora_ckpt}")
        else:
            print(f"  ✗ LoRA checkpoint 不存在: {lora_ckpt}")
            return False
    
    adapter_lora_dir = config.get('adapter_lora_dir')
    if adapter_lora_dir:
        if os.path.isdir(adapter_lora_dir):
            adapter_lora_file = os.path.join(adapter_lora_dir, "pytorch_lora_weights.safetensors")
            if os.path.exists(adapter_lora_file):
                print(f"  ✓ Adapter LoRA 存在: {adapter_lora_file}")
            else:
                print(f"  ✗ Adapter LoRA 不存在: {adapter_lora_file}")
                return False
        elif os.path.isfile(adapter_lora_dir):
            print(f"  ✓ Adapter LoRA 存在: {adapter_lora_dir}")
        else:
            print(f"  ✗ Adapter LoRA 不存在: {adapter_lora_dir}")
            return False
    
    # 檢查 VAE 路徑
    transp_vae_path = config.get('transp_vae_path')
    if transp_vae_path and os.path.exists(transp_vae_path):
        print(f"  ✓ Transparent VAE 存在: {transp_vae_path}")
    else:
        print(f"  ✗ Transparent VAE 不存在: {transp_vae_path}")
        return False
    
    # 4. 檢查數據集
    print("\n[4] 檢查數據集...")
    use_indexed_dataset = config.get('use_indexed_dataset', False)
    
    if use_indexed_dataset:
        print("  使用 Indexed Dataset (TAData + caption.json)")
        caption_json = config.get('caption_json')
        if caption_json and os.path.exists(caption_json):
            print(f"  ✓ Caption JSON 存在: {caption_json}")
            # 檢查 JSON 格式
            try:
                import json
                with open(caption_json, 'r') as f:
                    caption_data = json.load(f)
                if isinstance(caption_data, dict):
                    print(f"  ✓ Caption JSON 格式正確 (包含 {len(caption_data)} 個條目)")
                else:
                    print(f"  ⚠ Caption JSON 格式異常: 期望 dict，得到 {type(caption_data)}")
            except json.JSONDecodeError as e:
                print(f"  ✗ Caption JSON 格式錯誤: {e}")
                return False
        else:
            print(f"  ✗ Caption JSON 不存在: {caption_json}")
            return False
        
        # 嘗試載入數據集（不載入模型）
        try:
            from tools.dlcv_dataset_indexed import DLCVLayoutDatasetIndexed
            dataset = DLCVLayoutDatasetIndexed(
                data_dir=data_dir,
                caption_json_path=caption_json,
                enable_debug=False,
            )
            print(f"  ✓ 數據集載入成功: {len(dataset)} 個樣本")
            
            # 測試載入第一個樣本
            if len(dataset) > 0:
                try:
                    sample = dataset[0]
                    print(f"  ✓ 成功載入第一個樣本")
                    print(f"    - Height: {sample.get('height', 'N/A')}")
                    print(f"    - Width: {sample.get('width', 'N/A')}")
                    print(f"    - Layers: {len(sample.get('layout', []))}")
                    print(f"    - Caption length: {len(sample.get('caption', ''))}")
                except Exception as e:
                    print(f"  ✗ 載入樣本失敗: {e}")
                    return False
        except Exception as e:
            print(f"  ✗ 數據集載入失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("  使用 Standard Dataset")
        try:
            from tools.dataset import LayoutTrainDataset
            dataset = LayoutTrainDataset(data_dir, split="test")
            print(f"  ✓ 數據集載入成功: {len(dataset)} 個樣本")
            
            # 測試載入第一個樣本
            if len(dataset) > 0:
                try:
                    sample = dataset[0]
                    print(f"  ✓ 成功載入第一個樣本")
                except Exception as e:
                    print(f"  ✗ 載入樣本失敗: {e}")
                    return False
        except Exception as e:
            print(f"  ✗ 數據集載入失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 5. 檢查輸出目錄
    print("\n[5] 檢查輸出目錄...")
    save_dir = config.get('save_dir')
    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
            print(f"  ✓ 輸出目錄可寫入: {save_dir}")
        except Exception as e:
            print(f"  ✗ 輸出目錄無法創建: {e}")
            return False
    
    # 6. 檢查其他配置
    print("\n[6] 檢查其他配置...")
    max_samples = config.get('max_samples')
    if max_samples is not None:
        print(f"  ✓ max_samples: {max_samples}")
    else:
        print(f"  ✓ max_samples: null (處理全部樣本)")
    
    enable_dataset_debug = config.get('enable_dataset_debug', False)
    print(f"  ✓ enable_dataset_debug: {enable_dataset_debug}")
    
    seed = config.get('seed')
    print(f"  ✓ seed: {seed}")
    
    cfg = config.get('cfg', 4.0)
    print(f"  ✓ cfg (guidance_scale): {cfg}")
    
    max_layer_num = config.get('max_layer_num', 52)
    print(f"  ✓ max_layer_num: {max_layer_num}")
    
    # 7. 總結
    print("\n" + "="*60)
    print("✓ 配置驗證通過！可以開始 inference")
    print("="*60)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="驗證 inference 配置和數據集，不載入模型權重"
    )
    parser.add_argument(
        "--config_path", "-c",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    
    success = validate_config(args.config_path)
    sys.exit(0 if success else 1)

