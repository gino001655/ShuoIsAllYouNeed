# CLD Inference 使用指南

## 📋 概述

CLD (Controllable Layer Decomposition) 的 inference 需要以下組件：

1. **配置檔案** (YAML)
2. **資料集** (符合格式要求)
3. **模型權重檔案**
4. **執行腳本**

## 🔧 需要的資料格式

CLD 的 inference 需要資料集提供以下資訊：

### 每個樣本需要的欄位：

1. **whole_image** (PIL Image): 完整的合成圖片
2. **whole_caption** (str): 圖片描述文字（用於 prompt）
3. **base_image** (PIL Image): 背景圖片
4. **layer_count** (int): 圖層數量
5. **layer_XX** (PIL Image): 各個圖層的圖片（layer_00, layer_01, ...）
6. **layer_XX_box** (list): 各個圖層的邊界框 `[w0, h0, w1, h1]`
7. **style_category** (str, 可選): 風格類別（用於資料集分割）

### 資料集轉換

您的資料集格式（DLCV final project）已經透過 `tools/custom_dataset.py` 適配為 CLD 格式：

- `preview` → `whole_image`
- `title` → `whole_caption`
- `image` (list) → `layer_XX`
- `left`, `top`, `width`, `height` → `layer_XX_box`

## 📝 配置檔案設定

建立或修改 `infer/infer.yaml`：

```yaml
seed: 42                    # 隨機種子
max_layer_num: 52           # 最大圖層數

# 資料集路徑（指向包含 snapshots 的目錄）
data_dir: "../dataset"

# 模型路徑
pretrained_model_name_or_path: "Path_to_pretrained_FLUX_model"
pretrained_adapter_path: "Path_to_pretrained_FLUX_adapter"
transp_vae_path: "Path_to_transparent_vae"
pretrained_lora_dir: "Path_to_pretrained_lora"
lora_ckpt: "Path_to_trained_lora"
layer_ckpt: "Path_to_layer_pe_ckpt"
adapter_lora_dir: "Path_to_adapter_lora"

# 輸出目錄
save_dir: "Path_to_save_results"

# 其他參數（可選）
cfg: 4.0                    # Guidance scale
max_layers: 48              # 最大圖層數（VAE）
decoder_arch: "vit"          # VAE decoder 架構
pos_embedding: "rope"        # 位置編碼
layer_embedding: "rope"      # 圖層編碼
```

## 🚀 執行 Inference

### 方法 1: 使用原始命令

```bash
cd CLD
python -m infer.infer -c infer/infer.yaml
```

### 方法 2: 直接執行

```bash
cd CLD/infer
python infer.py --config_path infer.yaml
```

## 📊 輸出結果

Inference 會產生以下輸出：

```
save_dir/
├── case_0/
│   ├── origin.png              # 原始輸入圖片
│   ├── whole_image_rgba.png   # 完整圖片（RGBA）
│   ├── background_rgba.png    # 背景圖層（RGBA）
│   ├── layer_0_rgba.png       # 圖層 0（RGBA）
│   ├── layer_1_rgba.png       # 圖層 1（RGBA）
│   ├── ...
│   └── case_0.png             # 最終合成結果
├── case_1/
│   └── ...
├── merged/                    # 所有合成結果（RGB）
│   ├── case_0.png
│   └── ...
└── merged_rgba/              # 所有合成結果（RGBA）
    ├── case_0.png
    └── ...
```

## ⚙️ 資料集處理流程

1. **載入 Parquet 檔案**: 從 `data_dir/snapshots/*/data/train-*.parquet` 載入
2. **資料分割**: 自動分割為 train/val/test (90/5/5)
3. **格式轉換**: 
   - 將 `preview` 轉為 `whole_image`
   - 將 `image` 列表轉為個別圖層
   - 將 `left`, `top`, `width`, `height` 轉為邊界框
4. **批次處理**: 透過 DataLoader 批次載入

## 🔍 檢查資料集

在執行 inference 前，可以先檢查資料集是否正確載入：

```python
from tools.custom_dataset import CustomLayoutDataset

dataset = CustomLayoutDataset("../dataset", split="test")
print(f"資料集大小: {len(dataset)}")

# 檢查第一筆資料
item = dataset[0]
print(f"Caption: {item['caption']}")
print(f"尺寸: {item['width']}x{item['height']}")
print(f"圖層數: {len(item['layout'])}")
```

## ⚠️ 注意事項

1. **CUDA 設定**: `infer.py` 中硬編碼了 `CUDA_VISIBLE_DEVICES = "1"`，如需修改請編輯檔案
2. **記憶體需求**: CLD 需要大量 GPU 記憶體，建議使用至少 24GB 的 GPU
3. **模型檔案**: 確保所有模型權重檔案路徑正確
4. **資料集路徑**: `data_dir` 應指向包含 `snapshots` 目錄的資料集根目錄

## 🐛 常見問題

### Q: 找不到資料集？
A: 確認 `data_dir` 路徑正確，且包含 `snapshots` 目錄

### Q: 記憶體不足？
A: 減少批次大小或使用較小的模型

### Q: 資料格式錯誤？
A: 檢查 `tools/custom_dataset.py` 中的資料轉換邏輯







