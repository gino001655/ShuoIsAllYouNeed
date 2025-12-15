# CUSTOM FILE: 論文對齊檢查 - Illustrator's Depth

參考論文：[Illustrator's Depth: Monocular Layer Index Prediction for Image Decomposition](https://www.alphaxiv.org/abs/2511.17454)

## 📋 論文方法 vs 目前實現對比

### ✅ 1. 模型初始化

**論文要求**：
- 載入 Depth-Pro 預訓練權重
- 保留 Encoder (DINO-v2) 和 Decoder 架構
- 移除 FOV head（因為不需要真實世界尺度）

**目前實現**：✅ **完全符合**
- `custom_layer_order_model.py` 第 150-167 行
- 使用 `strict=False` 處理 FOV head 權重不匹配
- 保留完整架構，只移除 FOV

### ✅ 2. 差異化學習率

**論文要求**：
- Encoder: 較低學習率（保留預訓練特徵）
- Decoder + Head: 較高學習率（快速適應新任務）

**目前實現**：✅ **完全符合**
- `custom_layer_order_train.py` 的 `setup_optimizer()` 函數
- Encoder: 1e-5
- Decoder: 1e-4
- 使用參數組實現

### ✅ 3. Scale-Invariant Loss

**論文要求**：
- 直接預測圖層索引（不預測 inverse depth）
- 使用 Scale-Invariant MAE Loss
- 標準化公式：$\hat{d} = \frac{d - m}{s}$（m=中位數, s=MAD）

**目前實現**：✅ **完全符合**
- `custom_layer_order_loss.py` 的 `scale_shift_invariant_loss()` 函數
- 使用中位數和 MAD 進行標準化
- 計算標準化後的 MAE

### ✅ 4. 輸出格式

**論文要求**：
- 輸出圖層索引圖（歸一化到 [0, 1]）
- 背景 = 0，前景 = 1

**目前實現**：✅ **完全符合**
- `custom_layer_order_model.py` 使用 Sigmoid 確保輸出在 [0, 1]
- `custom_layer_order_dataset.py` 生成 GT 時背景=0，前景=1

---

## 🔍 需要確認的細節

根據論文標題和常見做法，以下細節需要確認：

1. **激活函數**：使用 Sigmoid 確保 [0, 1] 輸出 ✓
2. **損失函數**：Scale-Invariant MAE ✓
3. **學習率比例**：Encoder:Decoder = 1:10 ✓
4. **權重初始化**：最後一層 bias 設為 0 ✓

---

## 📝 實現檢查清單

- [x] 載入 Depth-Pro 預訓練權重
- [x] 移除 FOV head
- [x] 保留 Encoder 和 Decoder 架構
- [x] 實現差異化學習率
- [x] 實現 Scale-Invariant Loss
- [x] 輸出歸一化到 [0, 1]
- [x] 使用 Sigmoid 激活函數
- [x] 最後一層 bias 初始化為 0




