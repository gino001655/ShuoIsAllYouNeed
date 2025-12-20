#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

import json
from tools.dlcv_dataset import DLCVLayoutDataset

print("="*70)
print("測試 Caption Mapping")
print("="*70)

print("\n1. 載入 dataset...")
try:
    dataset = DLCVLayoutDataset(
        data_dir='/tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1',
        split='train',
        caption_mapping_path='/tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD/caption_mapping_test.json'
    )
    print(f"✅ 成功！{len(dataset)} 筆資料")
except Exception as e:
    print(f"❌ 失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n2. 測試讀取第 0 筆...")
try:
    item = dataset[0]
    print(f"✅ 讀取成功")
    print(f"  Keys: {list(item.keys())}")
    print(f"  Caption: '{item['caption']}'")
    print(f"  Caption length: {len(item['caption'])}")
    print(f"  Height: {item['height']}, Width: {item['width']}")
except Exception as e:
    print(f"❌ 失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. 測試前 3 筆 caption...")
for i in range(3):
    try:
        item = dataset[i]
        caption = item['caption']
        print(f"[{i}] {caption[:100]}...")
    except Exception as e:
        print(f"[{i}] ❌ 錯誤: {e}")

print("\n✅ 測試完成！")


