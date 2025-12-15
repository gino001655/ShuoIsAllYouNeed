# CUSTOM FILE: 論文實現指南 - Illustrator's Depth

參考論文：[Illustrator's Depth: Monocular Layer Index Prediction for Image Decomposition](https://www.alphaxiv.org/abs/2511.17454)

## 🎯 實現對齊確認

### ✅ 完全符合論文的三個核心要求

#### 1. 模型初始化（Initialization）

**論文方法**：
- 載入 Depth-Pro 預訓練權重
- 保留 Encoder (DINO-v2) 和 Decoder 架構
- 移除 FOV head（不需要真實世界尺度）

**我們的實現**：
```python
# custom_layer_order_model.py 第 150-167 行
if config.checkpoint_uri is not None:
    state_dict = torch.load(config.checkpoint_uri, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict=state_dict, strict=False  # 允許 FOV head 權重不匹配
    )
```

**對齊狀態**：✅ **完全符合**

---

#### 2. 差異化學習率（Differential Learning Rates）

**論文方法**：
- Encoder (DINO-v2): 較低學習率，保留預訓練特徵
- Decoder + Head: 較高學習率，快速適應新任務

**我們的實現**：
```python
# custom_layer_order_train.py 第 70-142 行
encoder_params = []  # Encoder 參數
decoder_params = []  # Decoder + Head 參數

param_groups = [
    {'params': encoder_params, 'lr': encoder_lr},  # 1e-5
    {'params': decoder_params, 'lr': decoder_lr},  # 1e-4
]
```

**對齊狀態**：✅ **完全符合**

---

#### 3. Scale-Invariant Loss

**論文方法**：
- 直接預測圖層索引（不預測 inverse depth）
- 使用 Scale-Invariant MAE Loss
- 標準化：$\hat{d} = \frac{d - m}{s}$（m=中位數, s=MAD）

**我們的實現**：
```python
# custom_layer_order_loss.py 第 14-99 行
def scale_shift_invariant_loss(pred, target, eps=1e-6):
    # 對每張圖分別標準化
    m_pred = torch.median(pred_b)
    s_pred = torch.mean(torch.abs(pred_b - m_pred)) + eps
    pred_norm = (pred_b - m_pred) / s_pred
    
    # 同樣處理 target
    m_target = torch.median(target_b)
    s_target = torch.mean(torch.abs(target_b - m_target)) + eps
    target_norm = (target_b - m_target) / s_target
    
    # MAE Loss
    loss = torch.mean(torch.abs(pred_norm - target_norm))
```

**對齊狀態**：✅ **完全符合**

---

## 📐 模型架構細節

### Head 結構（符合論文）

```python
self.head = nn.Sequential(
    nn.Conv2d(dim_decoder, dim_decoder // 2, ...),      # 256 -> 128
    nn.ConvTranspose2d(..., kernel_size=2, stride=2),  # 上採樣 2x
    nn.Conv2d(128, 32, ...),                           # 128 -> 32
    nn.ReLU(True),
    nn.Conv2d(32, 1, kernel_size=1, ...),              # 32 -> 1
    nn.Sigmoid(),  # 確保輸出在 [0, 1]
)
```

**關鍵點**：
- ✅ 最後一層是 1 通道（圖層索引）
- ✅ 使用 Sigmoid 確保輸出在 [0, 1]
- ✅ 最後一層 bias 初始化為 0（深度估計的常見技巧）

---

## 🔧 訓練配置（符合論文）

### 學習率設置

```python
encoder_lr = 1e-5   # Encoder: 較低學習率
decoder_lr = 1e-4   # Decoder: 較高學習率（10倍）
```

**比例**：Encoder:Decoder = 1:10 ✅

### 優化器

```python
optimizer = AdamW(
    param_groups,  # 兩個參數組
    weight_decay=1e-4,
)
```

### 學習率調度器

```python
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,  # 根據實際 epoch 數調整
)
```

---

## 📊 損失函數（完全符合論文）

### Scale-Invariant MAE Loss

**數學公式**（論文中的公式）：
$$
\hat{d} = \frac{d - m}{s}
$$

$$
L_{MAE} = |\hat{D}_{pred} - \hat{D}_{gt}|
$$

其中：
- $m$ = 中位數 (median)
- $s$ = 平均絕對偏差 (Mean Absolute Deviation, MAD)

**我們的實現**：
```python
# 對每張圖分別計算
m_pred = torch.median(pred_b)
s_pred = torch.mean(torch.abs(pred_b - m_pred)) + eps
pred_norm = (pred_b - m_pred) / s_pred

# 同樣處理 target
m_target = torch.median(target_b)
s_target = torch.mean(torch.abs(target_b - m_target)) + eps
target_norm = (target_b - m_target) / s_target

# MAE Loss
loss = torch.mean(torch.abs(pred_norm - target_norm))
```

**對齊狀態**：✅ **完全符合論文公式**

---

## 🚀 完整訓練命令

### 基本訓練

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
    --num-workers 4
```

### 參數說明

| 參數 | 值 | 說明 |
|------|-----|------|
| `--data-dir` | `../parsed_dataset` | 解析後的資料集目錄 |
| `--checkpoint-path` | `./checkpoints/depth_pro.pt` | Depth-Pro 預訓練權重 |
| `--batch-size` | `4` | 批次大小（根據 GPU 記憶體調整） |
| `--learning-rate` | `1e-4` | Decoder 學習率 |
| `--encoder-lr` | `1e-5` | Encoder 學習率（較低） |
| `--use-edge-loss` | - | 使用邊緣保持損失（可選） |
| `--edge-loss-weight` | `0.1` | 邊緣損失權重 |

---

## 📋 實現檢查清單

### 模型架構
- [x] 載入 Depth-Pro 預訓練權重
- [x] 保留 Encoder (DINO-v2) 架構
- [x] 保留 Decoder 架構
- [x] 移除 FOV head
- [x] 輸出改為單通道圖層索引
- [x] 使用 Sigmoid 確保輸出在 [0, 1]
- [x] 最後一層 bias 初始化為 0

### 訓練策略
- [x] 實現差異化學習率
- [x] Encoder 學習率：1e-5
- [x] Decoder 學習率：1e-4
- [x] 使用 AdamW 優化器
- [x] 使用 CosineAnnealingLR 調度器
- [x] 實現梯度裁剪

### 損失函數
- [x] 實現 Scale-Invariant Loss
- [x] 使用中位數進行標準化
- [x] 使用 MAD 進行標準化
- [x] 計算標準化後的 MAE
- [x] 可選：邊緣保持損失

### 資料處理
- [x] 生成圖層索引圖（GT）
- [x] 歸一化到 [0, 1]（背景=0，前景=1）
- [x] 處理圖層疊加順序
- [x] 處理透明區域

---

## 🎓 與論文的對應關係

| 論文要求 | 我們的實現 | 檔案位置 |
|---------|----------|---------|
| 載入預訓練權重 | ✅ | `custom_layer_order_model.py:150-167` |
| 差異化學習率 | ✅ | `custom_layer_order_train.py:70-142` |
| Scale-Invariant Loss | ✅ | `custom_layer_order_loss.py:14-99` |
| 移除 FOV head | ✅ | `custom_layer_order_model.py:131, 243` |
| 輸出 [0, 1] 圖層索引 | ✅ | `custom_layer_order_model.py:233` |
| 資料集處理 | ✅ | `custom_layer_order_dataset.py` |

---

## ⚠️ 注意事項

1. **資料集格式**：
   - 確保使用 `parsed_dataset` 目錄（已解析的資料）
   - 或修改 `custom_layer_order_dataset.py` 以適配您的資料格式

2. **GPU 記憶體**：
   - 1536x1536 輸入需要較大 GPU 記憶體
   - 如果記憶體不足，可以：
     - 減小 batch_size
     - 使用混合精度訓練（`--use-amp`）

3. **學習率調整**：
   - 如果訓練不穩定，可以降低學習率
   - 如果收斂太慢，可以適當提高學習率

4. **損失權重**：
   - 邊緣損失權重（`edge_loss_weight`）可以根據效果調整
   - 如果邊緣不夠銳利，可以增加權重

---

## 📚 參考資料

- **論文**：[Illustrator's Depth: Monocular Layer Index Prediction for Image Decomposition](https://www.alphaxiv.org/abs/2511.17454)
- **Depth-Pro 論文**：Sharp Monocular Metric Depth in Less Than a Second
- **MiDaS 論文**：Towards Robust Monocular Depth Estimation

---

## ✅ 總結

**我們的實現完全符合論文的方法**：

1. ✅ **模型初始化**：載入 Depth-Pro 預訓練權重，保留架構，移除 FOV
2. ✅ **差異化學習率**：Encoder 1e-5，Decoder 1e-4
3. ✅ **Scale-Invariant Loss**：使用中位數和 MAD 進行標準化，計算 MAE

可以直接使用提供的訓練命令開始訓練！




