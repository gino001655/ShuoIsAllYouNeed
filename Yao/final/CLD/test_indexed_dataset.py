"""
測試 index-based caption matching 方案
"""

import sys
sys.path.insert(0, '/tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD/tools')

from dlcv_dataset_indexed import DLCVLayoutDatasetIndexed, collate_fn
from torch.utils.data import DataLoader


def test_indexed_dataset():
    """測試 indexed dataset"""
    
    print("="*60)
    print("測試方案 B: Index-based Caption Matching")
    print("="*60)
    
    # 設定路徑
    data_dir = "/tmp2/b12902041/Gino/TAData/DLCV_dataset/data"
    caption_json = "/tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD/caption_llava16_final.json"
    
    print(f"\n1. 載入 dataset...")
    print(f"   Data dir: {data_dir}")
    print(f"   Caption JSON: {caption_json}")
    
    # 創建 dataset
    dataset = DLCVLayoutDatasetIndexed(
        data_dir=data_dir,
        caption_json_path=caption_json,
        enable_debug=True,  # 顯示前 3 個樣本的 debug 資訊
    )
    
    print(f"\n2. Dataset 資訊:")
    print(f"   總樣本數: {len(dataset)}")
    
    # 測試前 3 個樣本
    print(f"\n3. 測試前 3 個樣本...")
    
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        
        print(f"\n--- Sample {i} ---")
        print(f"Preview size: {sample['whole_img'].size}")
        print(f"Canvas size: {sample['width']} x {sample['height']}")
        print(f"Number of layers: {len(sample['layout'])}")
        print(f"Caption (前 150 字): {sample['caption'][:150]}...")
        
        # 檢查 layers
        for j, layer in enumerate(sample['layout'][:5]):  # 只顯示前 5 個 layer
            print(f"  Layer {j}: {layer['layer_img'].size}, bbox=({layer['left']}, {layer['top']}, {layer['width']}, {layer['height']})")
    
    # 測試 DataLoader
    print(f"\n4. 測試 DataLoader...")
    
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    print(f"   創建 DataLoader 成功 (batch_size=1)")
    
    # 測試迭代
    print(f"\n5. 測試迭代前 2 個 batch...")
    
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        
        print(f"\nBatch {i}:")
        print(f"  Type: {type(batch)}")
        print(f"  Keys: {batch.keys() if isinstance(batch, dict) else 'N/A'}")
        print(f"  Preview type: {type(batch['whole_img'])}")
        print(f"  Caption length: {len(batch['caption'])}")
        print(f"  Layers: {len(batch['layout'])}")
    
    print("\n" + "="*60)
    print("✓ 測試完成！")
    print("="*60)
    
    # 統計資訊
    print(f"\n📊 統計資訊:")
    
    num_samples_to_check = min(10, len(dataset))
    layer_counts = []
    caption_lengths = []
    
    for i in range(num_samples_to_check):
        sample = dataset[i]
        layer_counts.append(len(sample['layout']))
        caption_lengths.append(len(sample['caption']))
    
    print(f"   檢查前 {num_samples_to_check} 個樣本:")
    print(f"   - 平均 layer 數: {sum(layer_counts) / len(layer_counts):.1f}")
    print(f"   - Layer 數範圍: {min(layer_counts)} ~ {max(layer_counts)}")
    print(f"   - 平均 caption 長度: {sum(caption_lengths) / len(caption_lengths):.0f} 字元")
    print(f"   - Caption 長度範圍: {min(caption_lengths)} ~ {max(caption_lengths)} 字元")
    
    print("\n🎉 方案 B 可行！不需要轉換數據集！")


if __name__ == '__main__':
    test_indexed_dataset()

