#!/usr/bin/env python3
"""
驗證 train.py 中的 tensor 尺寸是否正確匹配
測試圖片尺寸調整和座標轉換邏輯
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.tools import get_input_box, build_layer_mask

def test_get_input_box():
    """測試 get_input_box 的量化邏輯"""
    print("\n" + "="*70)
    print("測試 1: get_input_box 量化邏輯")
    print("="*70)
    
    test_cases = [
        # (input_layout, expected_description)
        ([[0, 0, 704, 704]], "704x704 (已是16的倍數)"),
        ([[0, 0, 700, 1000]], "700x1000 (需要調整)"),
        ([[0, 0, 1024, 1024]], "1024x1024 (已是16的倍數)"),
        ([[0, 0, 300, 250]], "300x250 (小尺寸)"),
    ]
    
    all_passed = True
    for layout, desc in test_cases:
        result = get_input_box(layout)
        x1, y1, x2, y2 = result[0]
        
        # 檢查是否是 16 的倍數
        is_valid = (x1 % 16 == 0 and y1 % 16 == 0 and 
                   x2 % 16 == 0 and y2 % 16 == 0)
        
        # 檢查是否沒有超出太多（最多多 15 像素）
        orig_x2, orig_y2 = layout[0][2], layout[0][3]
        is_reasonable = (x2 - orig_x2 <= 15 and y2 - orig_y2 <= 15)
        
        status = "✅ PASS" if (is_valid and is_reasonable) else "❌ FAIL"
        print(f"\n{desc}:")
        print(f"  輸入: {layout[0]}")
        print(f"  輸出: [{x1}, {y1}, {x2}, {y2}]")
        print(f"  是否16的倍數: {is_valid}")
        print(f"  是否合理擴展: {is_reasonable} (最多擴展 {x2-orig_x2}x{y2-orig_y2} 像素)")
        print(f"  狀態: {status}")
        
        if not (is_valid and is_reasonable):
            all_passed = False
    
    return all_passed

def test_size_adjustment():
    """測試尺寸調整邏輯"""
    print("\n" + "="*70)
    print("測試 2: 尺寸調整邏輯")
    print("="*70)
    
    test_sizes = [
        (700, 1000),
        (1024, 1024),
        (300, 250),
        (704, 1008),
        (1023, 1024),
    ]
    
    all_passed = True
    for original_H, original_W in test_sizes:
        # 模擬 train.py 中的調整邏輯
        H = ((original_H + 15) // 16) * 16
        W = ((original_W + 15) // 16) * 16
        
        # 檢查
        is_valid = (H % 16 == 0 and W % 16 == 0)
        diff_H = H - original_H
        diff_W = W - original_W
        
        status = "✅ PASS" if is_valid else "❌ FAIL"
        print(f"\n原始尺寸: {original_H}x{original_W}")
        print(f"調整後: {H}x{W}")
        print(f"差異: +{diff_H}x+{diff_W} 像素")
        print(f"是否16的倍數: {is_valid}")
        print(f"狀態: {status}")
        
        if not is_valid:
            all_passed = False
    
    return all_passed

def test_tensor_size_matching():
    """測試 tensor 尺寸匹配"""
    print("\n" + "="*70)
    print("測試 3: Tensor 尺寸匹配")
    print("="*70)
    
    # 模擬一個完整的流程
    original_H, original_W = 700, 1000
    print(f"\n原始圖片尺寸: {original_H}x{original_W}")
    
    # 步驟 1: 調整尺寸
    H = ((original_H + 15) // 16) * 16
    W = ((original_W + 15) // 16) * 16
    print(f"調整後尺寸: {H}x{W}")
    
    # 步驟 2: 模擬 pixel_RGB
    L = 5  # 假設 5 層
    C = 3  # RGB
    pixel_RGB = torch.randn(L, C, original_H, original_W)
    print(f"\n原始 pixel_RGB shape: {list(pixel_RGB.shape)}")
    
    # 步驟 3: Resize pixel_RGB
    if H != original_H or W != original_W:
        pixel_RGB = torch.nn.functional.interpolate(
            pixel_RGB,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
    print(f"Resize 後 pixel_RGB shape: {list(pixel_RGB.shape)}")
    
    # 步驟 4: 模擬 VAE 編碼後的 latent
    # VAE 的 scale factor 是 8
    H_lat = H // 8
    W_lat = W // 8
    print(f"\n期望的 latent 尺寸: {H_lat}x{W_lat} (H/8 × W/8)")
    
    # 步驟 5: 調整座標
    scale_h = H / original_H
    scale_w = W / original_W
    print(f"\n座標調整比例: scale_h={scale_h:.4f}, scale_w={scale_w:.4f}")
    
    # 假設有一個 layout
    original_layout = [[0, 0, original_W-1, original_H-1]]
    adjusted_layout = []
    for layer_box in original_layout:
        x1, y1, x2, y2 = layer_box
        adjusted_x1 = round(x1 * scale_w)
        adjusted_y1 = round(y1 * scale_h)
        adjusted_x2 = round(x2 * scale_w)
        adjusted_y2 = round(y2 * scale_h)
        adjusted_layout.append([adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2])
    
    print(f"原始 layout: {original_layout[0]}")
    print(f"調整後 layout: {adjusted_layout[0]}")
    
    # 步驟 6: 計算 layer_boxes
    layer_boxes = get_input_box(adjusted_layout)
    print(f"量化後 layer_boxes: {layer_boxes[0]}")
    
    # 步驟 7: 建立 mask
    try:
        mask = build_layer_mask(L, H_lat, W_lat, layer_boxes)
        print(f"\nmask shape: {list(mask.shape)}")
        print(f"期望 mask shape: [{L}, 1, {H_lat}, {W_lat}]")
        
        is_valid = (mask.shape == (L, 1, H_lat, W_lat))
        status = "✅ PASS" if is_valid else "❌ FAIL"
        print(f"Mask 尺寸匹配: {status}")
        
        return is_valid
    except Exception as e:
        print(f"\n❌ FAIL: 建立 mask 時發生錯誤: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("Train.py Tensor 尺寸驗證")
    print("="*70)
    
    results = []
    
    # 測試 1
    results.append(("get_input_box 量化邏輯", test_get_input_box()))
    
    # 測試 2
    results.append(("尺寸調整邏輯", test_size_adjustment()))
    
    # 測試 3
    results.append(("Tensor 尺寸匹配", test_tensor_size_matching()))
    
    # 總結
    print("\n" + "="*70)
    print("測試總結")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 所有測試通過！train.py 的 tensor 尺寸應該正確匹配。")
        return 0
    else:
        print("\n⚠️ 部分測試失敗，請檢查上述錯誤。")
        return 1

if __name__ == "__main__":
    sys.exit(main())



