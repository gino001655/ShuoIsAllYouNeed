#!/usr/bin/env python3
"""
檢查單個樣本的詳細信息
"""
import os
import sys
import glob
from PIL import Image
import io

# 添加 tools 目錄到路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

try:
    from datasets import load_dataset
    import pyarrow.parquet as pq
except ImportError as e:
    print(f"Error importing libraries: {e}")
    sys.exit(1)


def load_parquet_with_fallback(parquet_file):
    """Load parquet file with fallback to pyarrow if datasets fails."""
    try:
        ds = load_dataset("parquet", data_files=parquet_file)["train"]
        return ds
    except (TypeError, KeyError) as e:
        # Fallback to pyarrow
        table = pq.read_table(parquet_file)
        # Convert to simple list of dicts
        data = []
        for i in range(len(table)):
            row = {}
            for col in table.column_names:
                val = table[col][i].as_py()
                row[col] = val
            data.append(row)
        return data


def get_image_size(preview_value):
    """Get image size from various formats."""
    if isinstance(preview_value, Image.Image):
        return preview_value.size  # (width, height)
    elif isinstance(preview_value, dict):
        if 'bytes' in preview_value:
            img = Image.open(io.BytesIO(preview_value['bytes']))
            return img.size
        elif 'path' in preview_value:
            img = Image.open(preview_value['path'])
            return img.size
    elif isinstance(preview_value, bytes):
        img = Image.open(io.BytesIO(preview_value))
        return img.size
    elif isinstance(preview_value, str):
        img = Image.open(preview_value)
        return img.size
    return None


def get_layer_count(item):
    """Get number of layers from item."""
    # Try different possible field names
    if 'length' in item:
        return int(item['length'])
    elif 'layout' in item:
        if isinstance(item['layout'], list):
            return len(item['layout'])
    elif 'left' in item and isinstance(item['left'], list):
        return len(item['left'])
    return 0


def get_canvas_size(item):
    """Get canvas size from item."""
    if 'canvas_width' in item and 'canvas_height' in item:
        return (int(item['canvas_width']), int(item['canvas_height']))
    elif 'width' in item and 'height' in item:
        return (int(item['width']), int(item['height']))
    else:
        # Try to get from preview
        preview = item.get('preview', None)
        if preview is not None:
            size = get_image_size(preview)
            if size:
                return size
    return None


def find_sample_by_index(data_dir, target_index):
    """Find sample by index (e.g., 2016 for 2016.png)"""
    parquet_dir = os.path.join(data_dir, "data")
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    
    if len(parquet_files) == 0:
        print(f"❌ 找不到 Parquet 檔案在 {parquet_dir}")
        return None
    
    print(f"🔍 搜尋 index {target_index} (對應 {target_index:08d}.png)...")
    print(f"📁 檢查 {len(parquet_files)} 個 parquet 檔案\n")
    
    current_idx = 0
    for pf_idx, pf in enumerate(parquet_files):
        try:
            dataset = load_parquet_with_fallback(pf)
            dataset_len = len(dataset)
            
            # Check if target index is in this file
            if current_idx <= target_index < current_idx + dataset_len:
                local_idx = target_index - current_idx
                item = dataset[local_idx]
                
                print("="*60)
                print(f"✅ 找到樣本 index {target_index}")
                print("="*60)
                
                # Get layer count
                layer_count = get_layer_count(item)
                print(f"\n📊 Layer 數量: {layer_count}")
                
                # Get image size
                canvas_size = get_canvas_size(item)
                if canvas_size:
                    width, height = canvas_size
                    total_pixels = width * height
                    max_dim = max(width, height)
                    print(f"\n📐 圖像尺寸:")
                    print(f"  寬度: {width} px")
                    print(f"  高度: {height} px")
                    print(f"  總像素數: {total_pixels:,} px²")
                    print(f"  最大邊: {max_dim} px")
                    print(f"  寬高比: {width/height:.2f}")
                
                # Get preview info
                preview = item.get('preview', None)
                if preview:
                    preview_size = get_image_size(preview)
                    if preview_size:
                        print(f"\n🖼️  Preview 尺寸: {preview_size[0]} x {preview_size[1]} px")
                
                # Get layer details if available
                if 'left' in item and isinstance(item['left'], list):
                    print(f"\n📋 Layer 詳情 (前 5 個):")
                    for i in range(min(5, len(item['left']))):
                        left = item.get('left', [])[i] if i < len(item.get('left', [])) else None
                        top = item.get('top', [])[i] if i < len(item.get('top', [])) else None
                        w = item.get('width', [])[i] if i < len(item.get('width', [])) else None
                        h = item.get('height', [])[i] if i < len(item.get('height', [])) else None
                        if left is not None:
                            print(f"  Layer {i}: bbox=({left}, {top}, {w}, {h})")
                
                # Check caption
                if 'title' in item:
                    caption = item['title']
                    print(f"\n📝 Caption: {caption[:100]}..." if len(caption) > 100 else f"\n📝 Caption: {caption}")
                
                return item
            else:
                current_idx += dataset_len
        except Exception as e:
            print(f"⚠️  無法讀取檔案 {pf}: {e}")
            continue
    
    print(f"\n❌ 找不到 index {target_index} 的樣本")
    return None


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python check_single_sample.py <data_dir> <index>")
        print("範例: python check_single_sample.py /workspace/dataset/DLCV_dataset 2016")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    try:
        target_index = int(sys.argv[2])
    except ValueError:
        print(f"❌ 無效的 index: {sys.argv[2]}")
        sys.exit(1)
    
    if not os.path.exists(data_dir):
        print(f"❌ 路徑不存在: {data_dir}")
        sys.exit(1)
    
    find_sample_by_index(data_dir, target_index)

