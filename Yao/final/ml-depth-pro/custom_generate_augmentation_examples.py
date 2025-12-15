"""
CUSTOM FILE: 生成資料增強範例

用於視覺化資料增強效果，確認 ground truth 是否正確處理
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
import random
from custom_layer_order_dataset import LayerOrderDataset

def visualize_augmentation_examples(
    data_dir: str,
    output_dir: str = "./augmentation_examples",
    num_examples: int = 3,
):
    """
    生成資料增強範例，展示原始和增強後的圖像及 ground truth
    
    Args:
    ----
        data_dir: 資料集目錄
        output_dir: 輸出目錄
        num_examples: 生成幾個範例
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 創建不帶增強的資料集（用於取得原始資料）
    dataset_no_aug = LayerOrderDataset(
        data_dir=data_dir,
        split="train",
        use_parsed_format=True,
        use_augmentation=False,
    )
    
    # 創建帶增強的資料集
    dataset_with_aug = LayerOrderDataset(
        data_dir=data_dir,
        split="train",
        use_parsed_format=True,
        use_augmentation=True,
        augmentation_config={
            "horizontal_flip_prob": 0.6,  # 強制應用以便展示
            "vertical_flip_prob": 0.6,
            "rotation_prob": 0.6,
            "rotation_degrees": 90,
            "color_jitter_prob": 0.6,
            "color_jitter_brightness": 0.2,
            "color_jitter_contrast": 0.5,
            "color_jitter_saturation": 0.5,
        },
    )
    
    # 隨機選擇幾個樣本
    indices = random.sample(range(len(dataset_no_aug)), min(num_examples, len(dataset_no_aug)))
    
    for example_idx, idx in enumerate(indices):
        print(f"\n{'='*80}")
        print(f"處理範例 {example_idx + 1}/{len(indices)}")
        print(f"{'='*80}")
        
        # 取得樣本資訊（檔案路徑）
        sample_dir = None
        category = "unknown"
        sample_id = "unknown"
        metadata = None
        
        if hasattr(dataset_no_aug, 'samples'):
            sample_info = dataset_no_aug.samples[idx]
            sample_dir = sample_info['sample_dir']
            category = sample_info.get('category', 'unknown')
            sample_id = sample_info.get('sample_id', 'unknown')
            
            print(f"樣本索引: {idx}")
            print(f"分類: {category}")
            print(f"樣本 ID: {sample_id}")
            print(f"樣本目錄: {sample_dir}")
            
            # 讀取 metadata 來顯示所有檔案
            metadata_path = sample_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                print(f"\n檔案路徑:")
                whole_img_path = sample_dir / "00_whole_image.png"
                base_img_path = sample_dir / "01_base_image.png"
                print(f"  - 完整圖片: {whole_img_path}")
                if whole_img_path.exists():
                    print(f"    ✓ 檔案存在")
                else:
                    print(f"    ✗ 檔案不存在！")
                
                print(f"  - 背景圖片: {base_img_path}")
                if base_img_path.exists():
                    print(f"    ✓ 檔案存在")
                else:
                    print(f"    ✗ 檔案不存在（可選）")
                
                print(f"  - Metadata: {metadata_path}")
                
                num_layers = metadata.get('layer_count', 0)
                print(f"  - Metadata 中的圖層數量: {num_layers}")
                
                print(f"\n各圖層檔案:")
                missing_layers = []
                existing_layers = []
                for i, layer_info in enumerate(metadata.get('layers', [])):
                    layer_img_path = sample_dir / layer_info.get('image_path', '')
                    box = layer_info.get('box', [0, 0, 0, 0])
                    if layer_img_path.exists():
                        print(f"    圖層 {i}: ✓ {layer_img_path}")
                        print(f"            Box: {box}")
                        existing_layers.append(i)
                    else:
                        print(f"    圖層 {i}: ✗ {layer_img_path} (檔案不存在！)")
                        print(f"            Box: {box}")
                        missing_layers.append(i)
                
                if missing_layers:
                    print(f"\n⚠️  警告: 有 {len(missing_layers)} 個圖層檔案缺失: {missing_layers}")
                print(f"✓ 存在的圖層: {len(existing_layers)} 個")
        else:
            print(f"樣本索引: {idx}")
            print(f"注意: 無法取得檔案路徑（使用 Parquet 格式）")
        
        print(f"{'='*80}\n")
        
        # 取得原始資料（不帶增強）
        sample_original = dataset_no_aug[idx]
        image_original = sample_original["image"]
        layer_map_original = sample_original["layer_index_map"]
        num_layers_actual = sample_original.get("num_layers", 0)
        
        print(f"實際載入的圖層數量: {num_layers_actual}")
        
        # 取得增強後的資料
        sample_aug = dataset_with_aug[idx]
        image_aug = sample_aug["image"]
        layer_map_aug = sample_aug["layer_index_map"]
        
        # 轉換 tensor 為 numpy 以便顯示
        # image: [3, H, W] -> [H, W, 3]
        image_orig_np = image_original.permute(1, 2, 0).numpy()
        image_orig_np = (image_orig_np + 1) / 2  # 反標準化 [-1, 1] -> [0, 1]
        image_orig_np = np.clip(image_orig_np, 0, 1)
        
        image_aug_np = image_aug.permute(1, 2, 0).numpy()
        image_aug_np = (image_aug_np + 1) / 2  # 反標準化
        image_aug_np = np.clip(image_aug_np, 0, 1)
        
        # layer_map: [H, W] -> [H, W]
        layer_map_orig_np = layer_map_original.numpy()
        layer_map_aug_np = layer_map_aug.numpy()
        
        # 準備標題資訊
        if hasattr(dataset_no_aug, 'samples'):
            sample_info = dataset_no_aug.samples[idx]
            title_info = f"{sample_info.get('category', 'unknown')}/{sample_info.get('sample_id', 'unknown')}"
        else:
            title_info = f"sample_{idx}"
        
        # 創建視覺化
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle(f"Sample: {title_info} | Index: {idx} | Layers: {num_layers_actual}", 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 原始圖像
        axes[0, 0].imshow(image_orig_np)
        axes[0, 0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 原始 ground truth
        im1 = axes[0, 1].imshow(layer_map_orig_np, cmap='viridis', vmin=0, vmax=1)
        axes[0, 1].set_title(f"Original Ground Truth (Layers: {num_layers_actual})", fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 增強後的圖像
        axes[1, 0].imshow(image_aug_np)
        axes[1, 0].set_title("Augmented Image", fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 增強後的 ground truth
        im2 = axes[1, 1].imshow(layer_map_aug_np, cmap='viridis', vmin=0, vmax=1)
        axes[1, 1].set_title("Augmented Ground Truth", fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # 保存圖片
        output_file = output_path / f"augmentation_example_{example_idx + 1}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  已保存: {output_file}")
        
        plt.close()
        
        # 也保存個別對比圖
        fig2, axes2 = plt.subplots(1, 2, figsize=(20, 10))
        
        # 並排顯示原始和增強後的圖像
        axes2[0].imshow(image_orig_np)
        axes2[0].set_title("原始圖像", fontsize=16, fontweight='bold')
        axes2[0].axis('off')
        
        axes2[1].imshow(image_aug_np)
        axes2[1].set_title("增強後的圖像", fontsize=16, fontweight='bold')
        axes2[1].axis('off')
        
        plt.tight_layout()
        output_file2 = output_path / f"augmentation_example_{example_idx + 1}_images.png"
        plt.savefig(output_file2, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 並排顯示原始和增強後的 ground truth
        fig3, axes3 = plt.subplots(1, 2, figsize=(20, 10))
        
        im3 = axes3[0].imshow(layer_map_orig_np, cmap='viridis', vmin=0, vmax=1)
        axes3[0].set_title("原始 Ground Truth", fontsize=16, fontweight='bold')
        axes3[0].axis('off')
        plt.colorbar(im3, ax=axes3[0], fraction=0.046, pad=0.04)
        
        im4 = axes3[1].imshow(layer_map_aug_np, cmap='viridis', vmin=0, vmax=1)
        axes3[1].set_title("增強後的 Ground Truth", fontsize=16, fontweight='bold')
        axes3[1].axis('off')
        plt.colorbar(im4, ax=axes3[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        output_file3 = output_path / f"augmentation_example_{example_idx + 1}_gt.png"
        plt.savefig(output_file3, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\n完成！所有範例已保存到: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成資料增強範例")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../parsed_dataset",
        help="資料集目錄路徑"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./augmentation_examples",
        help="輸出目錄"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="生成幾個範例"
    )
    
    args = parser.parse_args()
    
    visualize_augmentation_examples(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_examples=args.num_examples,
    )

