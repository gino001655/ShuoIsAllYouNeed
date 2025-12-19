#!/usr/bin/env python3
"""
測試資料集是否能正確載入
"""
import sys
import os

# 添加 tools 目錄到路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

try:
    from custom_dataset import CustomLayoutDataset, collate_fn
    print("✓ 成功載入 custom_dataset")
except ImportError as e:
    print(f"✗ 無法載入 custom_dataset: {e}")
    sys.exit(1)

# 測試資料集載入
data_dir = "../dataset"  # 相對於 CLD 目錄

print(f"\n正在測試資料集載入: {data_dir}")
print("=" * 50)

try:
    # 測試 test split
    print("\n[測試] 載入 test split...")
    test_dataset = CustomLayoutDataset(data_dir, split="test")
    print(f"✓ Test split: {len(test_dataset)} 筆資料")
    
    if len(test_dataset) > 0:
        # 檢查第一筆資料
        print("\n[測試] 檢查第一筆資料...")
        item = test_dataset[0]
        
        print(f"✓ Caption: {item['caption'][:50]}...")
        print(f"✓ 尺寸: {item['width']} x {item['height']}")
        print(f"✓ 圖層數: {len(item['layout'])}")
        print(f"✓ whole_img 類型: {type(item['whole_img'])}")
        print(f"✓ pixel_RGBA 數量: {len(item['pixel_RGBA'])}")
        print(f"✓ pixel_RGB 數量: {len(item['pixel_RGB'])}")
        
        # 檢查 layout 格式
        print(f"\n✓ Layout 範例 (前3個):")
        for i, layout in enumerate(item['layout'][:3]):
            print(f"  圖層 {i}: {layout}")
        
        print("\n✓ 資料集載入成功！")
    else:
        print("⚠ 資料集為空")
        
except Exception as e:
    print(f"✗ 載入資料集時發生錯誤: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("所有測試通過！資料集已準備好用於 inference。")







