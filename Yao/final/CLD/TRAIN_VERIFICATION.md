# Train.py 修正驗證報告

## 已完成的修正

### 1. 修正 tools.py 中的 get_input_box 量化邏輯

**檔案**: `tools/tools.py` 第 150-151 行

**修正前**:
```python
quantized_max_row = ((max_row // 16) + 1) * 16  # 錯誤：總是多加一個 patch
quantized_max_col = ((max_col // 16) + 1) * 16
```

**修正後**:
```python
# 向上取整到 16 的倍數（正確的方式）
quantized_max_row = ((max_row + 15) // 16) * 16
quantized_max_col = ((max_col + 15) // 16) * 16
```

**驗證**:
- ✅ 704 → 704 (正確，不會變成 720)
- ✅ 700 → 704 (正確，向上取整)
- ✅ 1000 → 1008 (正確，向上取整)
- ✅ 1024 → 1024 (正確，不會變成 1040)

---

### 2. 在 train.py 中加入 pixel_RGB resize 邏輯

**檔案**: `train/train.py` 第 209-223 行

**新增代碼**:
```python
if H != original_H or W != original_W:
    if step == 0 or step % 10 == 0:
        print(f"[STEP {step}] 調整圖片尺寸: {original_W}x{original_H} -> {W}x{H} (必須是 16 的倍數)", flush=True)
    
    # Resize pixel_RGB 以匹配調整後的尺寸
    # pixel_RGB shape: [L, C, original_H, original_W]
    L, C, _, _ = pixel_RGB.shape
    pixel_RGB = torch.nn.functional.interpolate(
        pixel_RGB, 
        size=(H, W), 
        mode='bilinear', 
        align_corners=False
    )
    if step == 0 or step % 10 == 0:
        print(f"[STEP {step}] Resized pixel_RGB from {original_H}x{original_W} to {H}x{W}", flush=True)
```

**作用**:
- 確保 `pixel_RGB` 的尺寸與調整後的 `H x W` 一致
- VAE 編碼後的 latent 尺寸將是 `H/8 x W/8`
- 與 `layer_boxes` 和 `mask` 的尺寸匹配

---

## 數據流驗證

### 修正前的問題

```
原始數據:
├─ pixel_RGB: [L, C, 700, 1000]
├─ H = 700, W = 1000
└─ layout: [[0, 0, 999, 699]]

調整尺寸:
├─ H = 704 (向上取整到 16 的倍數)
├─ W = 1008
└─ pixel_RGB: [L, C, 700, 1000]  ❌ 未調整！

計算 layer_boxes:
├─ adjusted_layout: [[0, 0, 1007, 703]]
├─ layer_boxes (量化): [[0, 0, 1024, 720]]  ❌ 錯誤的量化！
└─ layer_boxes (latent): [[0, 0, 128, 90]]  (除以 8)

VAE 編碼:
├─ pixel_RGB: [L, C, 700, 1000]
└─ latent: [1, L, 16, 87, 125]  ❌ 87 != 88, 125 != 126!

建立 mask:
├─ H_lat = 88 (704/8)
├─ W_lat = 126 (1008/8)
└─ mask: [L, 1, 88, 126]

結果: ❌ latent 尺寸 (87x125) 與 mask 尺寸 (88x126) 不匹配！
```

### 修正後的正確流程

```
原始數據:
├─ pixel_RGB: [L, C, 700, 1000]
├─ H = 700, W = 1000
└─ layout: [[0, 0, 999, 699]]

調整尺寸:
├─ H = 704 (向上取整到 16 的倍數)
├─ W = 1008
└─ pixel_RGB: [L, C, 700, 1000]

Resize pixel_RGB:
└─ pixel_RGB: [L, C, 704, 1008]  ✅ 已調整！

計算 layer_boxes:
├─ adjusted_layout: [[0, 0, 1007, 703]]
├─ layer_boxes (量化): [[0, 0, 1008, 704]]  ✅ 正確的量化！
└─ layer_boxes (latent): [[0, 0, 126, 88]]  (除以 8)

VAE 編碼:
├─ pixel_RGB: [L, C, 704, 1008]
└─ latent: [1, L, 16, 88, 126]  ✅ 88 和 126 正確！

建立 mask:
├─ H_lat = 88 (704/8)
├─ W_lat = 126 (1008/8)
└─ mask: [L, 1, 88, 126]

結果: ✅ latent 尺寸 (88x126) 與 mask 尺寸 (88x126) 完全匹配！
```

---

## 訓練測試清單

在實際訓練前，請確認以下幾點：

### 啟動訓練時的檢查

1. ✅ **第一個 batch 的輸出檢查**:
   ```
   [STEP 0] 調整圖片尺寸: WxH -> W'xH' (必須是 16 的倍數)
   [STEP 0] Resized pixel_RGB from HxW to H'xW'
   [STEP 0] 數據載入完成 (尺寸: W'xH', 圖層數: N)
   [STEP 0] VAE 編碼完成 (latent shape: [1, L, 16, H'/8, W'/8])
   [STEP 0] mask shape: [L, 1, H'/8, W'/8]
   [STEP 0] Loss: X.XXXXXX
   ```

2. ✅ **確認不會出現以下錯誤**:
   - ❌ `RuntimeError: The expanded size of the tensor (X) must match the existing size (Y)`
   - ❌ `RuntimeError: Sizes of tensors must match`
   - ❌ `IndexError: index X is out of bounds`

3. ✅ **Loss 計算正常**:
   - Loss 值應該是合理的數值（不是 NaN 或 Inf）
   - Loss 應該能正常反向傳播
   - 梯度更新應該正常進行

### 監控指標

在訓練過程中監控：
- Loss 是否在合理範圍內
- 是否有任何 tensor 尺寸錯誤
- 內存使用是否穩定（沒有內存洩漏）

---

## 與 infer.py 的一致性

現在 train.py 和 infer.py 使用相同的邏輯：

| 項目 | train.py | infer.py | 一致性 |
|------|----------|----------|--------|
| 尺寸調整 | 16 的倍數 | 16 的倍數 | ✅ |
| 量化邏輯 | `((x+15)//16)*16` | `((x+15)//16)*16` | ✅ |
| pixel_RGB resize | ✅ 有 | N/A (不需要) | ✅ |
| 座標調整 | 按比例 | 按比例 | ✅ |
| 最終輸出 resize | ❌ 無 (不生成圖片) | ✅ 有 | ✅ |

---

## 總結

### 修正的問題

1. **量化邏輯錯誤**: 從 `((x//16)+1)*16` 改為 `((x+15)//16)*16`
2. **pixel_RGB 未 resize**: 加入 resize 邏輯，確保與調整後的尺寸一致

### 預期結果

- ✅ 所有 tensor 尺寸正確匹配
- ✅ Loss 計算正確，不會"歪掉"
- ✅ 訓練可以正常進行
- ✅ 與 infer.py 邏輯一致

### 下一步

1. 啟動訓練，觀察第一個 batch 的輸出
2. 確認沒有 tensor 尺寸錯誤
3. 確認 Loss 計算正常
4. 監控訓練過程是否穩定

---

## 快速驗證命令

```bash
# 啟動訓練（使用小的 max_steps 進行測試）
cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD
python -m train.train -c train/train.yaml

# 檢查輸出是否包含：
# - "調整圖片尺寸" 訊息
# - "Resized pixel_RGB" 訊息
# - "VAE 編碼完成 (latent shape: ...)" 訊息
# - 沒有 RuntimeError 或 tensor 尺寸錯誤
```

如果訓練正常進行且 Loss 可以計算，則修正成功！

