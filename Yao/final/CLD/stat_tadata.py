#!/usr/bin/env python3
"""
統計 TAData 中圖片的 layer 數量和圖像大小
"""
import os
import sys
import glob
import json
import numpy as np
from collections import defaultdict
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


def statistics_tadata(data_dir):
    """統計 TAData 的 layer 數量和圖像大小"""
    print("="*60)
    print("TAData 統計分析")
    print("="*60)
    
    parquet_dir = os.path.join(data_dir, "data")
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    
    if len(parquet_files) == 0:
        print(f"❌ 找不到 Parquet 檔案在 {parquet_dir}")
        return None
    
    print(f"📁 找到 {len(parquet_files)} 個 parquet 檔案")
    print()
    
    # Statistics
    layer_counts = []
    image_sizes = []
    total_pixels = []
    valid_samples = 0
    invalid_samples = 0
    
    # Detailed stats
    layer_count_dist = defaultdict(int)
    size_dist = defaultdict(int)
    
    print("正在分析樣本...")
    for pf_idx, pf in enumerate(parquet_files):
        print(f"  處理檔案 {pf_idx+1}/{len(parquet_files)}: {os.path.basename(pf)}", end='\r')
        
        try:
            dataset = load_parquet_with_fallback(pf)
            
            for item in dataset:
                try:
                    # Get layer count
                    layer_count = get_layer_count(item)
                    if layer_count > 0:
                        layer_counts.append(layer_count)
                        layer_count_dist[layer_count] += 1
                    
                    # Get image size
                    canvas_size = get_canvas_size(item)
                    if canvas_size:
                        width, height = canvas_size
                        image_sizes.append((width, height))
                        total_pixels.append(width * height)
                        # Round to nearest 100 for distribution
                        size_key = f"{width//100*100}x{height//100*100}"
                        size_dist[size_key] += 1
                    
                    valid_samples += 1
                except Exception as e:
                    invalid_samples += 1
                    continue
        except Exception as e:
            print(f"\n⚠️  無法讀取檔案 {pf}: {e}")
            continue
    
    print("\n" + "="*60)
    print("統計結果")
    print("="*60)
    
    if len(layer_counts) == 0:
        print("❌ 沒有找到有效的樣本")
        return None
    
    # Layer count statistics
    print("\n📊 Layer 數量統計:")
    print(f"  總樣本數: {valid_samples}")
    print(f"  無效樣本數: {invalid_samples}")
    print(f"  平均 Layer 數: {np.mean(layer_counts):.2f}")
    print(f"  中位數 Layer 數: {np.median(layer_counts):.2f}")
    print(f"  最小 Layer 數: {min(layer_counts)}")
    print(f"  最大 Layer 數: {max(layer_counts)}")
    print(f"  標準差: {np.std(layer_counts):.2f}")
    
    # Percentiles
    percentiles = [50, 75, 90, 95, 99]
    print(f"\n  Layer 數百分位數:")
    for p in percentiles:
        val = np.percentile(layer_counts, p)
        print(f"    {p}%: {val:.0f} layers")
    
    # Layer count distribution (top 10)
    print(f"\n  Layer 數分布 (前 10 名):")
    sorted_layers = sorted(layer_count_dist.items(), key=lambda x: x[1], reverse=True)[:10]
    for layer_count, count in sorted_layers:
        percentage = count / len(layer_counts) * 100
        print(f"    {layer_count} layers: {count} 個樣本 ({percentage:.1f}%)")
    
    # Image size statistics
    if len(image_sizes) > 0:
        print("\n📐 圖像大小統計:")
        widths = [s[0] for s in image_sizes]
        heights = [s[1] for s in image_sizes]
        pixels = total_pixels
        
        print(f"  平均寬度: {np.mean(widths):.0f} px")
        print(f"  平均高度: {np.mean(heights):.0f} px")
        print(f"  平均像素數: {np.mean(pixels):.0f} px²")
        print(f"  最大寬度: {max(widths)} px")
        print(f"  最大高度: {max(heights)} px")
        print(f"  最大像素數: {max(pixels):,} px²")
        
        # Size percentiles
        print(f"\n  像素數百分位數:")
        for p in percentiles:
            val = np.percentile(pixels, p)
            print(f"    {p}%: {val:,.0f} px²")
        
        # Size distribution (top 10)
        print(f"\n  圖像大小分布 (前 10 名):")
        sorted_sizes = sorted(size_dist.items(), key=lambda x: x[1], reverse=True)[:10]
        for size_key, count in sorted_sizes:
            percentage = count / len(image_sizes) * 100
            print(f"    {size_key}: {count} 個樣本 ({percentage:.1f}%)")
    
    # Recommendations
    print("\n💡 建議的過濾閾值:")
    layer_95th = np.percentile(layer_counts, 95)
    layer_99th = np.percentile(layer_counts, 99)
    print(f"  Layer 數閾值:")
    print(f"    保守 (95%): {layer_95th:.0f} layers")
    print(f"    嚴格 (99%): {layer_99th:.0f} layers")
    
    if len(pixels) > 0:
        pixel_95th = np.percentile(pixels, 95)
        pixel_99th = np.percentile(pixels, 99)
        print(f"  像素數閾值:")
        print(f"    保守 (95%): {pixel_95th:,.0f} px²")
        print(f"    嚴格 (99%): {pixel_99th:,.0f} px²")
        
        # Calculate approximate dimensions
        import math
        size_95th = int(math.sqrt(pixel_95th))
        size_99th = int(math.sqrt(pixel_99th))
        print(f"    對應尺寸 (95%): ~{size_95th}x{size_95th} px")
        print(f"    對應尺寸 (99%): ~{size_99th}x{size_99th} px")
    
    # Save statistics to JSON
    stats = {
        "total_samples": valid_samples,
        "invalid_samples": invalid_samples,
        "layer_stats": {
            "mean": float(np.mean(layer_counts)),
            "median": float(np.median(layer_counts)),
            "min": int(min(layer_counts)),
            "max": int(max(layer_counts)),
            "std": float(np.std(layer_counts)),
            "percentiles": {str(p): float(np.percentile(layer_counts, p)) for p in percentiles}
        },
        "size_stats": {
            "mean_width": float(np.mean(widths)) if len(image_sizes) > 0 else None,
            "mean_height": float(np.mean(heights)) if len(image_sizes) > 0 else None,
            "mean_pixels": float(np.mean(pixels)) if len(pixels) > 0 else None,
            "max_width": int(max(widths)) if len(image_sizes) > 0 else None,
            "max_height": int(max(heights)) if len(image_sizes) > 0 else None,
            "max_pixels": int(max(pixels)) if len(pixels) > 0 else None,
            "percentiles": {str(p): float(np.percentile(pixels, p)) for p in percentiles} if len(pixels) > 0 else None
        },
        "recommendations": {
            "max_layers_95th": float(layer_95th),
            "max_layers_99th": float(layer_99th),
            "max_pixels_95th": float(pixel_95th) if len(pixels) > 0 else None,
            "max_pixels_99th": float(pixel_99th) if len(pixels) > 0 else None,
        }
    }
    
    output_file = os.path.join(os.path.dirname(__file__), "tadata_stats.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n💾 統計結果已保存到: {output_file}")
    
    return stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python stat_tadata.py <data_dir>")
        print("範例: python stat_tadata.py /workspace/dataset/DLCV_dataset")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    if not os.path.exists(data_dir):
        print(f"❌ 路徑不存在: {data_dir}")
        sys.exit(1)
    
    statistics_tadata(data_dir)

