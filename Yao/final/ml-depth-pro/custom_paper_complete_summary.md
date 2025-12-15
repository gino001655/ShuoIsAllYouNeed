# CUSTOM FILE: 完整實現總結 - 符合論文方法

參考論文：[Illustrator's Depth: Monocular Layer Index Prediction for Image Decomposition](https://www.alphaxiv.org/abs/2511.17454)

## ✅ 實現完全符合論文的三個核心要求

### 1. ✅ 模型初始化（Initialization）

**論文方法**：
- 載入 Depth-Pro 預訓練權重
- 保留 Encoder (DINO-v2) 和 Decoder 架構
- 移除 FOV head

**實現位置**：`custom_layer_order_model.py`
- 第 150-167 行：載入預訓練權重
- 第 131 行：`use_fov_head=False`
- 第 243 行：確認不使用 FOV head

**狀態**：✅ **完全符合**

---

### 2. ✅ 差異化學習率（Differential Learning Rates）

**論文方法**：
- Encoder: 較低學習率（保留預訓練特徵）
- Decoder + Head: 較高學習率（快速適應新任務）

**實現位置**：`custom_layer_order_train.py`
- 第 70-142 行：`setup_optimizer()` 函數
- Encoder 學習率：1e-5
- Decoder 學習率：1e-4
- 比例：1:10

**狀態**：✅ **完全符合**

---

### 3. ✅ Scale-Invariant Loss

**論文方法**：
- 直接預測圖層索引（不預測 inverse depth）
- 使用 Scale-Invariant MAE Loss
- 標準化公式：$\hat{d} = \frac{d - m}{s}$

**實現位置**：`custom_layer_order_loss.py`
- 第 14-99 行：`scale_shift_invariant_loss()` 函數
- 使用中位數（median）和平均絕對偏差（MAD）
- 計算標準化後的 MAE

**狀態**：✅ **完全符合**

---

## 📁 檔案結構

```
ml-depth-pro/
├── custom_layer_order_model.py          # 模型定義（符合論文）
├── custom_layer_order_dataset.py        # 資料集（支援兩種格式）
├── custom_layer_order_loss.py          # 損失函數（符合論文）
├── custom_layer_order_train.py         # 訓練腳本（符合論文）
├── custom_layer_order_config.py        # 配置文件
├── custom_TODO_explanations.md         # TODO 詳細說明
├── custom_paper_alignment.md           # 論文對齊檢查
├── custom_paper_implementation_guide.md # 實現指南
└── custom_paper_complete_summary.md    # 本文件
```

---

## 🚀 完整訓練命令

### 使用已解析的資料格式（推薦）

```bash
cd /tmp2/b12902041/Gino/DLCV/final/ml-depth-pro

python custom_layer_order_train.py \
    --data-dir ../parsed_dataset \
    --checkpoint-path ./checkpoints/depth_pro.pt \
    --batch-size 4 \
    --num-epochs 100 \
    --learning-rate 1e-4 \
    --encoder-lr 1e-5 \
    --weight-decay 1e-4 \
    --use-edge-loss \
    --edge-loss-weight 0.1 \
    --save-dir ./checkpoints/layer_order \
    --num-workers 4 \
    --use-parsed-format
```

### 使用原始 Parquet 格式

```bash
python custom_layer_order_train.py \
    --data-dir ../dataset \
    --checkpoint-path ./checkpoints/depth_pro.pt \
    --batch-size 4 \
    --num-epochs 100 \
    --learning-rate 1e-4 \
    --encoder-lr 1e-5 \
    --weight-decay 1e-4 \
    --use-edge-loss \
    --edge-loss-weight 0.1 \
    --save-dir ./checkpoints/layer_order \
    --num-workers 4
    # 不加 --use-parsed-format 表示使用原始格式
```

---

## 📊 關鍵實現細節對比

| 論文要求 | 我們的實現 | 檔案位置 | 狀態 |
|---------|----------|---------|------|
| 載入預訓練權重 | ✅ | `custom_layer_order_model.py:150-167` | ✅ |
| 移除 FOV head | ✅ | `custom_layer_order_model.py:131, 243` | ✅ |
| 差異化學習率 | ✅ | `custom_layer_order_train.py:70-142` | ✅ |
| Scale-Invariant Loss | ✅ | `custom_layer_order_loss.py:14-99` | ✅ |
| 輸出 [0, 1] 圖層索引 | ✅ | `custom_layer_order_model.py:233` | ✅ |
| 中位數標準化 | ✅ | `custom_layer_order_loss.py:71-72` | ✅ |
| MAD 標準化 | ✅ | `custom_layer_order_loss.py:78-79` | ✅ |
| MAE Loss | ✅ | `custom_layer_order_loss.py:93` | ✅ |

---

## 🎯 與論文完全一致的關鍵點

### 1. 模型架構
- ✅ 保留 Depth-Pro 的 Encoder (DINO-v2)
- ✅ 保留 Depth-Pro 的 Decoder
- ✅ 移除 FOV head（不需要真實世界尺度）
- ✅ 輸出改為單通道圖層索引
- ✅ 使用 Sigmoid 確保輸出在 [0, 1]

### 2. 訓練策略
- ✅ Encoder 學習率：1e-5（較低）
- ✅ Decoder 學習率：1e-4（較高，10倍）
- ✅ 使用 AdamW 優化器
- ✅ 使用 CosineAnnealingLR 調度器
- ✅ 實現梯度裁剪

### 3. 損失函數
- ✅ 直接預測圖層索引（不預測 inverse depth）
- ✅ 使用中位數進行標準化（shift-invariant）
- ✅ 使用 MAD 進行標準化（scale-invariant）
- ✅ 計算標準化後的 MAE
- ✅ 對每張圖分別標準化（處理不同圖層數量）

---

## 📝 使用步驟

### 步驟 1: 準備資料集

如果使用已解析格式：
```bash
# 解析 Parquet 檔案為可讀格式
cd /tmp2/b12902041/Gino/DLCV/final
python custom_parse_parquet_to_readable.py \
    --data-dir my_download/data \
    --output-dir parsed_dataset \
    --create-summary
```

### 步驟 2: 下載預訓練模型

```bash
cd ml-depth-pro
source get_pretrained_models.sh  # 下載 Depth-Pro 權重
```

### 步驟 3: 開始訓練

```bash
python custom_layer_order_train.py \
    --data-dir ../parsed_dataset \
    --checkpoint-path ./checkpoints/depth_pro.pt \
    --batch-size 4 \
    --num-epochs 100 \
    --learning-rate 1e-4 \
    --encoder-lr 1e-5 \
    --use-parsed-format
```

---

## ⚠️ 重要注意事項

1. **資料集格式**：
   - 已解析格式（`parsed_dataset`）：PNG 圖片 + JSON 元資料
   - 原始格式（`dataset`）：Parquet 檔案
   - 使用 `--use-parsed-format` 參數切換

2. **GPU 記憶體**：
   - 1536x1536 輸入需要較大 GPU 記憶體
   - 建議 batch_size=4，如果記憶體不足可以減小

3. **學習率**：
   - Encoder: 1e-5（固定，不要改）
   - Decoder: 1e-4（可以根據訓練情況微調）

4. **損失函數**：
   - 主要使用 Scale-Invariant Loss
   - 可選：Edge Preserving Loss（權重 0.1）

---

## ✅ 總結

**我們的實現完全符合論文 [Illustrator's Depth](https://www.alphaxiv.org/abs/2511.17454) 的方法**：

1. ✅ **模型初始化**：載入 Depth-Pro 預訓練權重，保留架構，移除 FOV
2. ✅ **差異化學習率**：Encoder 1e-5，Decoder 1e-4（比例 1:10）
3. ✅ **Scale-Invariant Loss**：使用中位數和 MAD 標準化，計算 MAE

**可以直接開始訓練！**




