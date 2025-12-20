#!/usr/bin/env python3
"""
為 TAData 生成 ID-based caption mapping

兩種模式：
1. 從現有的 path-based caption_mapping.json 轉換（如果圖片對應得上）
2. 直接為 TAData 保存 preview 圖片並生成新的 captions
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def create_id_based_caption_from_tadata(
    tadata_dir: str,
    output_json: str,
    save_preview_images: bool = True,
    preview_output_dir: str = "/tmp2/b12902041/Gino/TAData_previews"
):
    """
    為 TAData 創建 ID-based caption mapping
    
    Args:
        tadata_dir: TAData parquet 目錄
        output_json: 輸出的 JSON 文件
        save_preview_images: 是否保存 preview 圖片（用於後續生成 LLaVA captions）
        preview_output_dir: Preview 圖片輸出目錄
    """
    tadata_dir = Path(tadata_dir)
    preview_dir = Path(preview_output_dir)
    
    if save_preview_images:
        preview_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 將保存 preview 圖片到: {preview_dir}")
    
    # Load all parquet files
    parquet_files = sorted(list(tadata_dir.glob("*.parquet")))
    print(f"[INFO] 找到 {len(parquet_files)} 個 parquet 文件")
    
    # ID-based caption mapping
    id_caption_mapping = {}
    index_caption_mapping = {}
    preview_path_mapping = {}  # id -> preview_path (用於生成 captions)
    
    global_idx = 0
    
    for pf in tqdm(parquet_files, desc="處理 parquet 文件"):
        ds = load_dataset('parquet', data_files=str(pf))['train']
        
        for i in tqdm(range(len(ds)), desc=f"  {pf.name}", leave=False):
            item = ds[i]
            
            # Get ID
            sample_id = item.get('id', f'sample_{global_idx:08d}')
            
            # Get title (現有 caption)
            title = item.get('title', 'A design image')
            
            # Save ID-based mapping
            id_caption_mapping[sample_id] = title
            index_caption_mapping[str(global_idx)] = title
            
            # Save preview image if needed
            if save_preview_images:
                preview_img = item['preview']
                if isinstance(preview_img, Image.Image):
                    preview_filename = f"{sample_id}.png"
                    preview_path = preview_dir / preview_filename
                    preview_img.save(preview_path)
                    preview_path_mapping[sample_id] = str(preview_path)
            
            global_idx += 1
    
    print(f"\n[INFO] 總共處理 {global_idx} 個樣本")
    print(f"[INFO] ID-based captions: {len(id_caption_mapping)}")
    print(f"[INFO] Index-based captions: {len(index_caption_mapping)}")
    
    # Save mappings
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save ID-based
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(id_caption_mapping, f, ensure_ascii=False, indent=2)
    print(f"[INFO] ✓ 保存 ID-based mapping: {output_path}")
    
    # Save index-based (備用)
    index_output = output_path.parent / (output_path.stem + "_index.json")
    with open(index_output, 'w', encoding='utf-8') as f:
        json.dump(index_caption_mapping, f, ensure_ascii=False, indent=2)
    print(f"[INFO] ✓ 保存 index-based mapping: {index_output}")
    
    # Save preview path mapping (用於生成 LLaVA captions)
    if save_preview_images:
        path_output = output_path.parent / (output_path.stem + "_paths.json")
        with open(path_output, 'w', encoding='utf-8') as f:
            json.dump(preview_path_mapping, f, ensure_ascii=False, indent=2)
        print(f"[INFO] ✓ 保存 preview paths mapping: {path_output}")
        print(f"\n💡 接下來可以為這些圖片生成 LLaVA captions:")
        print(f"   preview 圖片在: {preview_dir}")
    
    print(f"\n✅ 完成！現在可以使用 TAData + ID-based caption mapping!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="為 TAData 生成 ID-based caption mapping")
    parser.add_argument(
        "--tadata_dir",
        type=str,
        default="/tmp2/b12902041/Gino/TAData/DLCV_dataset/data",
        help="TAData parquet 目錄"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD/tadata_id_caption_mapping.json",
        help="輸出 JSON 文件"
    )
    parser.add_argument(
        "--save_previews",
        action="store_true",
        help="保存 preview 圖片（用於後續生成 LLaVA captions）"
    )
    parser.add_argument(
        "--preview_dir",
        type=str,
        default="/tmp2/b12902041/Gino/TAData_previews",
        help="Preview 圖片輸出目錄"
    )
    
    args = parser.parse_args()
    
    create_id_based_caption_from_tadata(
        tadata_dir=args.tadata_dir,
        output_json=args.output,
        save_preview_images=args.save_previews,
        preview_output_dir=args.preview_dir,
    )
