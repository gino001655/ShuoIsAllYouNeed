# 詳細輸出指南：訓練和推理時的資訊顯示

## 📊 概述

現在 **training** 和 **inference** 都會顯示詳細資訊：
- ✅ 使用的 Caption（LLaVA 生成）
- ✅ 抓取的圖片尺寸
- ✅ 所有 Layer 的資訊（位置、尺寸、類型）
- ✅ Dataset 載入過程

---

## 🎯 Training 時的輸出

### 1️⃣ Dataset 載入時

```
============================================================
載入訓練數據集
============================================================
[INFO] 使用 DLCVLayoutDatasetIndexed (Index-based caption matching)
[INFO] Data dir: /workspace/dataset/TAData/DLCV_dataset/data
[INFO] Caption JSON: /workspace/.../caption_llava16_final.json
[INFO] 🔍 Dataset debug enabled: 將顯示前 3 個樣本的詳細資訊

載入 dataset from /workspace/dataset/TAData/DLCV_dataset/data...
✓ 載入 19479 個樣本
載入 caption mapping from /workspace/.../caption_llava16_final.json...
✓ 載入 19479 個 captions

============================================================
[LOAD] Sample 0
============================================================
[IMG] Preview: PIL Image (1024, 1024)
[CAPTION] From index 0: - The style is cartoonish and colorful...
[CANVAS] 1080 x 1080
[LAYERS] Total: 8
  [IMG] Layer 0: PIL Image (1024, 1024)
  [IMG] Layer 1: PIL Image (982, 828)
  [IMG] Layer 2: PIL Image (651, 370)
  ...
[RESULT] Loaded 8 layers

============================================================
✓ 載入 19479 個訓練樣本
============================================================
```

### 2️⃣ 訓練循環中（每 10 步顯示）

```
============================================================
[STEP 0] 訓練數據詳情
============================================================
  📊 Canvas 尺寸: 1080 x 1080
  🎨 圖層數量: 8
    Layer 0: bbox=(0, 0, 1080, 1080), type=image
    Layer 1: bbox=(52, 122, 982, 828), type=image
    Layer 2: bbox=(214, 355, 651, 370), type=text
    Layer 3: bbox=(232, 374, 616, 331), type=text
    Layer 4: bbox=(470, 416, 139, 18), type=text
    ... 還有 3 個圖層
  📝 Caption: - The style is cartoonish and colorful, with a playful and celebratory theme.
- The main subject is a group of hands reaching upwards...
  📏 Caption 長度: 368 字元
============================================================
[STEP 0] 開始文本編碼...
[STEP 0] 文本編碼完成
[STEP 0] 開始 Adapter 圖像編碼...
[STEP 0] Adapter 圖像編碼完成
[STEP 0] 開始 VAE 編碼目標圖層 (共 8 層)...
[STEP 0] VAE 編碼完成 (latent shape: torch.Size([1, 8, 16, 67, 67]))
[STEP 0] 準備噪聲和 timestep...
[STEP 0] 開始 MultiLayer Adapter 前向傳播...
[STEP 0] MultiLayer Adapter 完成，開始 Transformer (DiT) 前向傳播...
[STEP 0] Transformer 前向傳播完成，計算 loss...
[STEP 0] Loss: 0.123456，開始反向傳播...
[STEP 0] 權重更新完成！

[INFO] Step 10/200000，Loss: 0.098765
```

### 3️⃣ 定期記錄（每 1000 步）

```
[INFO] Step 1000/200000 完成，Loss: 0.045678
```

---

## 🚀 Inference 時的輸出

### 1️⃣ Dataset 載入時

```
============================================================
載入推理數據集
============================================================
[INFO] 使用 DLCVLayoutDatasetIndexed (Index-based caption matching)
[INFO] Data dir: /workspace/dataset/TAData/DLCV_dataset/data
[INFO] Caption JSON: /workspace/.../caption_llava16_final.json
[INFO] 🔍 Dataset debug enabled: 將顯示前 3 個樣本的詳細資訊

載入 dataset...
✓ 載入 19479 個樣本
✓ 載入 19479 個 captions
```

### 2️⃣ 處理每個樣本時

```
============================================================
處理樣本 0
============================================================
  📊 Canvas 尺寸: 1080 x 1080
  🎨 圖層數量: 8
  📝 Caption: - The style is cartoonish and colorful, with a playful and celebratory theme.
- The main subject is a group of hands reaching upwards...
  📏 Caption 長度: 368 字元
  🖼️  圖層詳情:
    Layer 0: bbox=(0, 0, 1080, 1080), type=image
    Layer 1: bbox=(52, 122, 982, 828), type=image
    Layer 2: bbox=(214, 355, 651, 370), type=text
    Layer 3: bbox=(232, 374, 616, 331), type=text
    Layer 4: bbox=(470, 416, 139, 18), type=text
    ... 還有 3 個圖層
============================================================

[INFO] 開始推理...
[INFO] Encoding prompt...
[INFO] Encoding adapter image...
[INFO] Running diffusion...
[INFO] Decoding latents...
[INFO] Saving results...

Saved case 0 to output/case_0000/
```

---

## ⚙️ 配置選項

### Training 配置 (`train/train_tadata_indexed.yaml`)

```yaml
# 方案 B：使用 TAData + caption.json
use_indexed_dataset: true  # 啟用 index-based matching
data_dir: "/workspace/dataset/TAData/DLCV_dataset/data"
caption_mapping: "/workspace/.../caption_llava16_final.json"

# Debug 設定
enable_dataset_debug: true  # 顯示前 3 個樣本的詳細資訊
```

### Inference 配置 (`configs/infer_tadata_indexed.json`)

```json
{
  "use_indexed_dataset": true,
  "data_dir": "/workspace/dataset/TAData/DLCV_dataset/data",
  "caption_json": "/workspace/.../caption_llava16_final.json",
  "enable_dataset_debug": true,
  "max_samples": 10  // 限制樣本數量（測試用）
}
```

---

## 🔍 Debug 模式詳解

### `enable_dataset_debug: true` 時

**顯示內容：**
1. **前 3 個樣本的完整載入過程：**
   - Preview 圖片來源和尺寸
   - Caption 來源（從 index X 讀取）
   - Canvas 尺寸
   - 每個 Layer 的詳細資訊：
     - 圖片來源（PIL Image / bytes / 從 preview crop）
     - 尺寸
     - Bounding box
   - 最終載入的 Layer 數量

2. **訓練/推理過程中（每 10 步）：**
   - Canvas 尺寸
   - 圖層數量
   - 前 5 個圖層的 bbox 和類型
   - Caption 預覽（前 150 字元）
   - Caption 總長度

### `enable_dataset_debug: false` 時

只顯示基本進度資訊：
```
[INFO] 載入 19479 個訓練樣本
[INFO] Step 10/200000，Loss: 0.098765
[INFO] Step 1000/200000 完成，Loss: 0.045678
```

---

## 📝 輸出內容說明

### 1. Caption 資訊

```
📝 Caption: - The style is cartoonish and colorful...
📏 Caption 長度: 368 字元
```

- **完整的 LLaVA 生成描述**（不是簡單的 title）
- 顯示前 150 字元（避免輸出過長）
- 顯示總長度

### 2. Canvas 資訊

```
📊 Canvas 尺寸: 1080 x 1080
```

- 圖片的總尺寸

### 3. Layer 資訊

```
🎨 圖層數量: 8
🖼️  圖層詳情:
  Layer 0: bbox=(0, 0, 1080, 1080), type=image
  Layer 1: bbox=(52, 122, 982, 828), type=image
  Layer 2: bbox=(214, 355, 651, 370), type=text
  ...
```

- 總圖層數
- 每個圖層的：
  - **bbox**: (left, top, width, height)
  - **type**: image / text / shape / etc.
- 只顯示前 5 個圖層（避免輸出過長）

### 4. 圖片來源資訊（Debug 模式）

```
[IMG] Layer 0: PIL Image (1024, 1024)
[IMG] Layer 1: bytes → PIL Image (982, 828)
[CROP] Layer 2: No image, cropping from preview
```

- **PIL Image**: 直接從 TAData 讀取的 Image 對象
- **bytes → PIL Image**: 從二進制數據轉換
- **cropping from preview**: 沒有單獨的 layer 圖片，從 preview 裁切

---

## 🎯 使用場景

### 場景 1：檢查數據集是否正確載入

```bash
# 只測試前 3 個樣本
python infer/infer.py \
    --config configs/infer_tadata_indexed.json \
    --max_samples 3 \
    --enable_dataset_debug true
```

**檢查：**
- ✅ Caption 是否是完整的 LLaVA 描述？
- ✅ 每個樣本有多少 layers？
- ✅ Layers 的尺寸和位置是否合理？

### 場景 2：訓練時監控數據

```bash
# 開始訓練，每 10 步顯示詳細資訊
python train/train.py train/train_tadata_indexed.yaml
```

**監控：**
- 第 0, 10, 20, ... 步會顯示完整資訊
- 確認每個 batch 的數據正確

### 場景 3：關閉 Debug（生產環境）

```yaml
# train.yaml
enable_dataset_debug: false
```

只顯示基本進度，減少日誌量。

---

## 🆘 故障排除

### 問題 1：Caption 顯示為 ""（空字串）

**可能原因：**
- `caption_json` 路徑錯誤
- Index 匹配失敗

**檢查：**
```bash
# 確認 caption_json 存在且格式正確
cat /path/to/caption_llava16_final.json | head -20

# 檢查 index 匹配
python test_indexed_dataset.py
```

### 問題 2：Layer 數量太少（只有 1-2 個）

**可能原因：**
- TAData 中的 `image` 欄位為 `None`
- 需要從 preview crop

**解決：**
- 方案 B 已自動處理（會從 preview crop）
- 檢查輸出是否有 `[CROP] Layer X: ...`

### 問題 3：輸出太多，難以閱讀

**解決：**
```yaml
# 關閉 debug
enable_dataset_debug: false
```

或者：
```bash
# 只看特定步數
python train/train.py ... 2>&1 | grep "STEP [0-9]*0]"
```

---

## 📊 輸出範例總結

### Training 簡潔模式 (`enable_dataset_debug: false`)

```
[INFO] 載入 19479 個訓練樣本
[INFO] 開始訓練循環，目標步數: 200000
[INFO] Step 1000/200000 完成，Loss: 0.045678
[INFO] Step 2000/200000 完成，Loss: 0.034567
...
```

### Training 詳細模式 (`enable_dataset_debug: true`)

```
[INFO] 載入 19479 個訓練樣本
[INFO] 開始訓練循環，目標步數: 200000

============================================================
[STEP 0] 訓練數據詳情
============================================================
  📊 Canvas 尺寸: 1080 x 1080
  🎨 圖層數量: 8
  📝 Caption: - The style is cartoonish...
  📏 Caption 長度: 368 字元
============================================================
[STEP 0] 開始文本編碼...
...
[INFO] Step 10/200000，Loss: 0.098765

============================================================
[STEP 10] 訓練數據詳情
============================================================
  📊 Canvas 尺寸: 1920 x 1080
  🎨 圖層數量: 7
  📝 Caption: The image is a graphic design...
  📏 Caption 長度: 511 字元
============================================================
...
```

---

## ✅ 總結

現在你可以：

1. ✅ **看到每個樣本使用的 Caption**（完整的 LLaVA 描述）
2. ✅ **看到抓取的圖片資訊**（尺寸、來源）
3. ✅ **看到所有 Layer 的詳細資訊**（位置、尺寸、類型）
4. ✅ **控制輸出詳細程度**（`enable_dataset_debug`）

**推薦設定：**
- 開發/測試：`enable_dataset_debug: true`
- 正式訓練：`enable_dataset_debug: false`（或每 100 步顯示一次）
