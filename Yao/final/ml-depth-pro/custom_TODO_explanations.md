# CUSTOM FILE: TODO 詳細說明

本文檔詳細解釋所有 TODO 項目的意義和如何完成。

## 📋 架構符合度檢查

### ✅ 1. 模型初始化 (Initialization)

**要求**：
- 載入 Depth-Pro 預訓練權重
- 保留 Encoder (DINO-v2) 和 Decoder 架構
- 只移除 FOV head

**目前實現**：✅ **符合**
- `custom_layer_order_model.py` 第 150-167 行載入預訓練權重
- 使用 `strict=False` 處理 FOV head 的權重不匹配
- 保留完整的 Encoder 和 Decoder 架構

### ✅ 2. 差異化學習率 (Differential Learning Rates)

**要求**：
- Encoder 使用較低學習率（保留預訓練特徵）
- Decoder 使用較高學習率（快速適應新任務）

**目前實現**：✅ **已修正**
- `custom_layer_order_train.py` 的 `setup_optimizer()` 函數已實現差異化學習率
- Encoder: 1e-5 (較低)
- Decoder: 1e-4 (較高)

### ✅ 3. Scale-Invariant Loss

**要求**：
- 直接預測圖層索引（不預測 inverse depth）
- 使用 Scale-Invariant MAE Loss
- 標準化公式：$\hat{d} = \frac{d - m}{s}$

**目前實現**：✅ **符合**
- `custom_layer_order_loss.py` 實現了 Scale-Shift Invariant Loss
- 使用中位數 (median) 和平均絕對偏差 (MAD) 進行標準化
- 計算標準化後的 MAE

---

## 📚 詳細 TODO 說明

### 模型架構相關 (`custom_layer_order_model.py`)

#### TODO 1: 理解 backbone 模型創建
**位置**：`create_backbone_model()` 函數

**問題**：
1. 它創建了什麼類型的模型？
2. `use_pretrained=False` 意味著什麼？
3. 為什麼需要返回 config？

**答案**：
1. **模型類型**：創建 Vision Transformer (ViT) 模型，具體是 DINO-v2
   - DINO-v2 是自監督學習的 ViT，在大量無標籤資料上訓練
   - 提供了強大的視覺特徵提取能力

2. **`use_pretrained=False`**：
   - 這裡設為 False 是因為我們會從 Depth-Pro checkpoint 載入權重
   - 如果設為 True，會載入 DINO-v2 的原始權重（但我們不需要）

3. **返回 config**：
   - config 包含模型的配置資訊（維度、層數等）
   - 這些資訊用於構建 Encoder 和 Decoder

**如何完成**：
- 閱讀 `network/vit_factory.py` 了解 ViT 的創建過程
- 理解 DINO-v2 的架構和預訓練方式

---

#### TODO 2: 理解編碼器的結構
**位置**：`create_layer_order_model_and_transforms()` 函數

**問題**：
1. `patch_encoder` 和 `image_encoder` 有什麼不同？
2. 為什麼需要兩個編碼器？
3. `dims_encoder` 是什麼？

**答案**：
1. **patch_encoder vs image_encoder**：
   - **patch_encoder**：處理圖像的局部 patches（滑動窗口）
   - **image_encoder**：處理完整的圖像（全局視圖）
   - 兩者結合提供多尺度特徵

2. **為什麼需要兩個**：
   - **多尺度理解**：patches 提供局部細節，完整圖像提供全局上下文
   - **Depth-Pro 的設計**：結合局部和全局特徵能更好地理解深度/圖層關係

3. **`dims_encoder`**：
   - 編碼器在不同層的特徵維度列表
   - 例如：[768, 768, 768, ...] 表示每層都是 768 維
   - 用於構建 Decoder 的輸入維度

**如何完成**：
- 閱讀 `network/encoder.py` 了解 DepthProEncoder 的實現
- 理解多尺度特徵提取的機制

---

#### TODO 3: 理解解碼器的結構
**位置**：`create_layer_order_model_and_transforms()` 函數

**問題**：
1. `dims_encoder` 列表的順序有什麼意義？
2. `dim_decoder` 控制什麼？

**答案**：
1. **順序意義**：
   - 列表順序對應編碼器的不同層級
   - 從高解析度到低解析度：[高解析度特徵, ..., 低解析度特徵]
   - Decoder 會融合這些多尺度特徵

2. **`dim_decoder`**：
   - Decoder 內部特徵的維度
   - 控制 Decoder 的容量（越大越強，但也越慢）
   - 預設是 256

**如何完成**：
- 閱讀 `network/decoder.py` 了解 MultiresConvDecoder
- 理解多解析度特徵融合的過程

---

#### TODO 4: 理解 head 的結構
**位置**：`LayerOrderModel.__init__()` 函數

**問題**：
1. 為什麼第一層是 `dim_decoder -> dim_decoder // 2`？
2. `ConvTranspose2d` 的作用是什麼？
3. 最後一層為什麼是 1 通道？

**答案**：
1. **維度減半**：
   - 逐步減少特徵維度，降低計算量
   - 256 -> 128 -> 32 -> 1 的設計是常見的漸進式降維

2. **ConvTranspose2d**：
   - 轉置卷積，用於上採樣（upsampling）
   - `kernel_size=2, stride=2` 表示將空間尺寸放大 2 倍
   - 從 Encoder 的較小特徵圖恢復到原始圖像尺寸

3. **1 通道輸出**：
   - 我們只需要預測圖層索引（單一數值）
   - 不像 RGB 需要 3 通道
   - 值在 [0, 1] 區間（通過 Sigmoid 保證）

**如何完成**：
- 理解卷積和轉置卷積的區別
- 理解上採樣和下採樣的過程

---

#### TODO 5: 理解權重載入
**位置**：`create_layer_order_model_and_transforms()` 函數

**問題**：
1. 為什麼要載入預訓練的 depth_pro 權重？
2. `strict=False` 意味著什麼？
3. 如果某些層不匹配會發生什麼？

**答案**：
1. **載入預訓練權重**：
   - Depth-Pro 已經在大量深度資料上訓練
   - Encoder 學到了通用的視覺特徵（邊緣、紋理、幾何）
   - 這些特徵對圖層順序預測也很有用

2. **`strict=False`**：
   - `strict=True`：所有權重必須完全匹配，否則報錯
   - `strict=False`：允許部分權重不匹配（如 FOV head）
   - 我們移除了 FOV head，所以必須用 `strict=False`

3. **不匹配的處理**：
   - `missing_keys`：模型需要但 checkpoint 沒有的權重（新層）
   - `unexpected_keys`：checkpoint 有但模型不需要的權重（FOV head）
   - 我們會忽略 `unexpected_keys`（FOV 相關）

**如何完成**：
- 理解 PyTorch 的 `load_state_dict()` 機制
- 理解權重遷移學習（transfer learning）的概念

---

### 資料集相關 (`custom_layer_order_dataset.py`)

#### TODO 6: 理解圖層索引圖的生成
**位置**：`create_layer_index_map()` 函數

**問題**：
1. 為什麼要初始化為 0？
2. 應該從背景到前景疊加，還是從前景到背景？
3. 如果圖層有重疊，應該使用哪個圖層的索引？

**答案**：
1. **初始化為 0**：
   - 0 代表背景（最底層）
   - 如果某個像素沒有被任何圖層覆蓋，保持為 0

2. **疊加順序**：
   - **從背景到前景**（layer 0, 1, 2, ...）
   - 後面的圖層會覆蓋前面的
   - 這樣最後的圖層索引就是最前景的

3. **重疊處理**：
   - 使用最後一個圖層的索引（最前景）
   - 這符合「後面的圖層在前面的圖層之上」的邏輯

**如何完成**：
- 理解圖層合成的原理
- 理解 alpha 通道和遮罩的使用

---

#### TODO 7: 理解歸一化公式
**位置**：`create_layer_index_map()` 函數

**問題**：
1. 為什麼要用 `i / (num_layers - 1)`？
2. 這樣歸一化後，背景是 0 還是 1？
3. 如果希望背景=0，前景=1，應該怎麼改？

**答案**：
1. **歸一化公式**：
   - `i` 是圖層索引（0, 1, 2, ..., num_layers-1）
   - `(num_layers - 1)` 是最大索引
   - 這樣可以將索引映射到 [0, 1] 區間

2. **背景和前景**：
   - 背景（i=0）：`0 / (num_layers-1) = 0`
   - 前景（i=num_layers-1）：`(num_layers-1) / (num_layers-1) = 1`
   - 所以背景=0，前景=1 ✓

3. **已經是正確的**：
   - 目前的實現已經符合要求
   - 背景=0，前景=1

**如何完成**：
- 理解線性歸一化的數學原理
- 驗證不同 num_layers 的情況

---

### 損失函數相關 (`custom_layer_order_loss.py`)

#### TODO 8: 理解中位數的作用
**位置**：`scale_shift_invariant_loss()` 函數

**問題**：
1. 為什麼用中位數而不是平均值？
2. 中位數對異常值更魯棒嗎？

**答案**：
1. **中位數 vs 平均值**：
   - **平均值**：受極端值影響大
   - **中位數**：不受極端值影響
   - 圖層索引可能有異常值（錯誤標記），中位數更穩定

2. **魯棒性**：
   - 是的，中位數對異常值更魯棒
   - 例如：[1, 2, 3, 100]，平均值=26.5，中位數=2.5
   - 中位數更能代表「典型值」

**如何完成**：
- 理解統計學中的中位數和平均值
- 理解異常值對統計量的影響

---

#### TODO 9: 理解平均絕對偏差（MAD）
**位置**：`scale_shift_invariant_loss()` 函數

**問題**：
1. MAD 和標準差有什麼不同？
2. 為什麼用 MAD 而不是標準差？

**答案**：
1. **MAD vs 標準差**：
   - **標準差**：$\sigma = \sqrt{\frac{1}{N}\sum (x_i - \mu)^2}$
   - **MAD**：$MAD = \frac{1}{N}\sum |x_i - m|$（m 是中位數）
   - MAD 使用絕對值，標準差使用平方

2. **為什麼用 MAD**：
   - MAD 對異常值更魯棒（因為用絕對值而非平方）
   - 與中位數配合使用，形成魯棒的標準化方法
   - 適合處理可能有錯誤標記的圖層索引

**如何完成**：
- 理解統計學中的離散度度量
- 理解魯棒統計學的概念

---

#### TODO 10: 理解標準化公式
**位置**：`scale_shift_invariant_loss()` 函數

**問題**：
1. `(pred_b - m_pred) / s_pred` 做了什麼？
2. 標準化後的值的範圍是什麼？
3. 為什麼這樣可以實現 scale-invariant？

**答案**：
1. **標準化過程**：
   - 減去中位數：消除位移（shift）
   - 除以 MAD：消除尺度（scale）
   - 結果：標準化後的相對值

2. **值的範圍**：
   - 沒有固定範圍（不像 [0, 1]）
   - 但分佈是標準化的（中位數=0，MAD=1）
   - 值通常在 [-3, 3] 左右（3 個標準差）

3. **Scale-Invariant 的證明**：
   - 如果所有值乘以 k：分子和分母都乘以 k，結果不變
   - 如果所有值加上 c：中位數也加 c，相減後抵消
   - 因此對尺度和位移都不敏感

**如何完成**：
- 理解標準化的數學原理
- 驗證 scale-invariant 的性質

---

#### TODO 11: 理解 MAE Loss
**位置**：`scale_shift_invariant_loss()` 函數

**問題**：
1. 為什麼用 L1（MAE）而不是 L2（MSE）？
2. L1 對邊緣更友好嗎？

**答案**：
1. **L1 vs L2**：
   - **L1 (MAE)**：$|x - y|$，對異常值不敏感
   - **L2 (MSE)**：$(x - y)^2$，對異常值敏感
   - 圖層索引是階梯狀的（離散值），L1 更適合

2. **邊緣友好性**：
   - 是的，L1 對邊緣更友好
   - L2 會「懲罰」大的誤差，導致模型傾向於產生平滑的輸出
   - L1 允許大的誤差，鼓勵產生銳利的邊界（適合圖層索引）

**如何完成**：
- 理解不同損失函數的特性
- 理解為什麼 L1 適合離散值預測

---

### 訓練相關 (`custom_layer_order_train.py`)

#### TODO 12: 理解差異化學習率
**位置**：`setup_optimizer()` 函數

**問題**：
1. 為什麼 Encoder 要用較低學習率？
2. 為什麼 Decoder 要用較高學習率？
3. 如何分離 Encoder 和 Decoder 的參數？

**答案**：
1. **Encoder 低學習率**：
   - Encoder 已經在大量資料上預訓練
   - 學到了通用的視覺特徵（邊緣、紋理、幾何）
   - 我們希望保留這些特徵，只做微調
   - 低學習率（1e-5）可以緩慢調整，不破壞預訓練特徵

2. **Decoder 高學習率**：
   - Decoder 負責將特徵轉換為最終輸出
   - 從深度預測改為圖層順序預測，需要較大的調整
   - 高學習率（1e-4）可以快速適應新任務

3. **參數分離**：
   - 通過 `model.named_parameters()` 遍歷所有參數
   - 檢查參數名稱中是否包含 "encoder"
   - 分別加入不同的參數組

**如何完成**：
- 理解遷移學習（transfer learning）的概念
- 理解學習率對訓練的影響
- 理解 PyTorch 的參數組機制

---

#### TODO 13: 理解訓練循環
**位置**：`train_epoch()` 函數

**問題**：
1. 為什麼要用 `model.train()`？
2. `optimizer.zero_grad()` 的作用是什麼？
3. `loss.backward()` 做了什麼？
4. `optimizer.step()` 做了什麼？

**答案**：
1. **`model.train()`**：
   - 設定模型為訓練模式
   - 啟用 Dropout、BatchNorm 等訓練時的行為
   - 與 `model.eval()` 對應（驗證時使用）

2. **`optimizer.zero_grad()`**：
   - 清除之前的梯度
   - PyTorch 會累積梯度，所以每次迭代前要清零
   - 如果不清零，梯度會累積，導致訓練不穩定

3. **`loss.backward()`**：
   - 反向傳播，計算所有參數的梯度
   - 使用鏈式法則從損失函數往回計算
   - 梯度存儲在 `param.grad` 中

4. **`optimizer.step()`**：
   - 根據梯度更新參數
   - 使用優化器算法（如 AdamW）計算更新量
   - 執行：`param = param - lr * grad`

**如何完成**：
- 理解深度學習的訓練流程
- 理解反向傳播算法
- 理解優化器的工作原理

---

#### TODO 14: 理解梯度裁剪
**位置**：`train_epoch()` 函數

**問題**：
1. 為什麼要裁剪梯度？
2. `max_norm` 應該設多少？

**答案**：
1. **梯度裁剪的原因**：
   - 防止梯度爆炸（gradient explosion）
   - 當梯度過大時，參數更新會過大，導致訓練不穩定
   - 裁剪後可以穩定訓練過程

2. **`max_norm` 的設定**：
   - 常見值：0.5, 1.0, 5.0
   - 1.0 是一個安全的預設值
   - 可以根據訓練情況調整（如果梯度經常被裁剪，可以增大）

**如何完成**：
- 理解梯度爆炸和梯度消失的問題
- 理解梯度裁剪的數學原理

---

## 🎯 完成 TODO 的建議順序

1. **先理解模型架構**（TODO 1-5）
   - 理解 Encoder、Decoder、Head 的作用
   - 理解權重載入的機制

2. **再理解資料處理**（TODO 6-7）
   - 理解圖層索引圖的生成
   - 理解歸一化公式

3. **然後理解損失函數**（TODO 8-11）
   - 理解 Scale-Invariant Loss 的原理
   - 理解為什麼用中位數和 MAD

4. **最後理解訓練流程**（TODO 12-14）
   - 理解差異化學習率
   - 理解訓練循環的每個步驟

---

## 📝 檢查清單

完成每個 TODO 後，請確認：

- [ ] 理解了該部分的原理
- [ ] 能夠解釋給別人聽
- [ ] 知道如何調整相關參數
- [ ] 知道如果出錯該如何調試

---

## 🔗 參考資源

- **Depth-Pro 論文**：了解原始架構
- **DINO-v2 論文**：了解 Encoder 的預訓練方式
- **MiDaS 論文**：了解 Scale-Invariant Loss 的來源
- **PyTorch 文檔**：了解具體 API 的使用





