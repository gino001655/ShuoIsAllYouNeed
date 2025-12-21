# 方案 B：直接使用 TAData + Index-based Caption Matching

## 🎯 概述

**不需要轉換數據集！** 直接讀取 TAData (Image 對象) + 用 index 匹配 caption.json

## 📊 方案對比

### 方案 A（轉換數據集）
- ✅ 數據自包含（caption 在 parquet 內）
- ✅ 跨機器兼容（所有路徑相對）
- ❌ 需要轉換時間（~45-50 分鐘）
- ❌ 需要額外磁碟空間（圖片重複存儲）

### 方案 B（Index-based matching）⭐
- ✅ **不需要轉換！直接使用！**
- ✅ **不需要額外磁碟空間**
- ✅ **立即可用**
- ⚠️ 需要兩個文件：TAData + caption.json

## 🚀 使用步驟

### 1. 複製新文件到你的機器

```bash
# 在 meow1 機器上
cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD

# 方法 1: scp 複製
scp tools/dlcv_dataset_indexed.py user@workspace:/workspace/ShuoIsAllYouNeed/Yao/final/CLD/tools/
scp test_indexed_dataset.py user@workspace:/workspace/ShuoIsAllYouNeed/Yao/final/CLD/
scp infer/infer.py user@workspace:/workspace/ShuoIsAllYouNeed/Yao/final/CLD/infer/
scp configs/infer_tadata_indexed.json user@workspace:/workspace/ShuoIsAllYouNeed/Yao/final/CLD/configs/

# 方法 2: git (如果有 commit)
# 在 workspace 機器：git pull
```

### 2. 測試 Dataset（可選）

```bash
# 在 workspace 機器
cd /workspace/ShuoIsAllYouNeed/Yao/final/CLD

# 測試前 3 個樣本（會顯示詳細 debug 資訊）
python test_indexed_dataset.py
```

**預期輸出：**
```
============================================================
測試方案 B: Index-based Caption Matching
============================================================

1. 載入 dataset...
   Data dir: /workspace/dataset/TAData/DLCV_dataset/data
   Caption JSON: /workspace/ShuoIsAllYouNeed/Yao/final/CLD/caption_llava16_final.json

載入 dataset from /workspace/dataset/TAData/DLCV_dataset/data...
✓ 載入 19480 個樣本
載入 caption mapping from /workspace/ShuoIsAllYouNeed/Yao/final/CLD/caption_llava16_final.json...
✓ 載入 19480 個 captions

============================================================
[LOAD] Sample 0
============================================================
[IMG] Preview: PIL Image (1024, 1024)
[CAPTION] From index 0: - The style is cartoonish and colorful...
[CANVAS] 1024 x 1024
[LAYERS] Total: 5
  [IMG] Layer 0: PIL Image (256, 256)
  [IMG] Layer 1: PIL Image (512, 512)
  ...
[RESULT] Loaded 5 layers

--- Sample 0 ---
Preview size: (1024, 1024)
Canvas size: 1024 x 1024
Number of layers: 5
Caption (前 150 字): - The style is cartoonish and colorful, with a playful and celebratory theme...

✓ 測試完成！
🎉 方案 B 可行！不需要轉換數據集！
```

### 3. 執行 Inference

#### 方法 A: 使用配置文件

```bash
cd /workspace/ShuoIsAllYouNeed/Yao/final/CLD

# 修改 configs/infer_tadata_indexed.json 中的模型路徑
# 然後執行：
python infer/infer.py --config configs/infer_tadata_indexed.json
```

#### 方法 B: 直接命令行

```bash
cd /workspace/ShuoIsAllYouNeed/Yao/final/CLD

python infer/infer.py \
    --config configs/infer_tadata_indexed.json \
    --data_dir /workspace/dataset/TAData/DLCV_dataset/data \
    --caption_json /workspace/ShuoIsAllYouNeed/Yao/final/CLD/caption_llava16_final.json \
    --use_indexed_dataset true \
    --output_dir output/tadata_indexed_test \
    --max_samples 5
```

### 4. 執行 Training

修改 `train/train.py` 使用 indexed dataset：

```python
# 在配置中添加：
config = {
    'data_dir': '/workspace/dataset/TAData/DLCV_dataset/data',
    'caption_json': '/workspace/ShuoIsAllYouNeed/Yao/final/CLD/caption_llava16_final.json',
    'use_indexed_dataset': True,
    'enable_dataset_debug': True,  # 查看載入資訊
    # ... 其他配置
}
```

## 🔑 核心原理

### Caption 匹配邏輯

1. **Caption JSON 格式：**
   ```json
   {
     "/workspace/dataset/.../00000000.png": "caption for sample 0",
     "/workspace/dataset/.../00000001.png": "caption for sample 1",
     "/workspace/dataset/.../00000123.png": "caption for sample 123"
   }
   ```

2. **提取 Index：**
   - 從路徑 `00000123.png` 提取數字 → `123`
   - 建立映射：`{123: "caption for sample 123"}`

3. **查找 Caption：**
   ```python
   # TAData 的 sample index = 123
   caption = caption_mapping[123]  # 直接查找！
   ```

### 數據格式處理

支持多種 Image 格式：
- ✅ PIL Image 對象（TAData 原生格式）
- ✅ bytes（pyarrow 格式）
- ✅ dict with 'bytes'（HuggingFace Image feature）
- ✅ 文件路徑（字符串）

支持 Layer 處理：
- ✅ 直接讀取 PIL Image layers
- ✅ 如果 layer 是 None，自動從 preview crop

## 📝 配置文件說明

### `configs/infer_tadata_indexed.json`

```json
{
  "model_path": "path/to/your/model",
  "vae_path": "path/to/vae",
  "t5_path": "path/to/t5",
  
  // 關鍵設定
  "data_dir": "/workspace/dataset/TAData/DLCV_dataset/data",
  "caption_json": "/workspace/ShuoIsAllYouNeed/Yao/final/CLD/caption_llava16_final.json",
  "use_indexed_dataset": true,  // 啟用方案 B
  
  "output_dir": "output/tadata_indexed",
  "max_samples": 10,  // 測試時用少量樣本
  
  "enable_dataset_debug": true  // 顯示載入資訊
}
```

## ⚠️ 注意事項

1. **Caption JSON 必須存在：**
   - 確保 `caption_llava16_final.json` 在指定路徑
   - 檔案格式必須正確（`{path: caption}` 格式）

2. **Index 必須匹配：**
   - Caption JSON 的檔名數字必須對應 TAData 的 index
   - 如果你的 caption 是按順序生成的，這應該沒問題

3. **TAData 格式：**
   - 確保 TAData 是標準的 HuggingFace Dataset（parquet 格式）
   - 包含 `preview`, `image`, `left`, `top`, `width`, `height` 等欄位

## 🎉 優勢總結

方案 B 優勢：
1. ⚡ **立即可用** - 不需要等待轉換
2. 💾 **節省空間** - 不需要重複存儲圖片
3. 🔄 **易於更新** - 修改 caption.json 就能更新 captions
4. 🐛 **易於調試** - 直接看到原始數據

適合：
- ✅ 快速測試
- ✅ 開發階段
- ✅ Caption 經常變動
- ✅ 磁碟空間有限

## 📚 相關文件

- `tools/dlcv_dataset_indexed.py` - Index-based dataset 實現
- `test_indexed_dataset.py` - 測試腳本
- `infer/infer.py` - 支持 indexed dataset 的 inference
- `configs/infer_tadata_indexed.json` - 配置範例

## 🆚 何時用方案 A？

如果你需要：
- 跨機器共享數據（不想帶著 caption.json）
- 最終發布版本（數據自包含）
- Caption 不會再變動

那就用方案 A（轉換數據集）。

## ❓ 常見問題

### Q: 我的 caption.json 順序對嗎？
A: 執行 `test_indexed_dataset.py`，它會顯示前 3 個樣本的 caption，你可以檢查是否正確。

### Q: 可以同時用兩種方案嗎？
A: 可以！開發時用方案 B，最終發布用方案 A。

### Q: 轉換還要繼續嗎？
A: 如果你只想測試，可以先用方案 B。如果想要最終版本，讓轉換繼續跑完。


