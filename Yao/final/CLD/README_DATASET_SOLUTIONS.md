# TAData + LLaVA Captions 使用方案

## 📁 文件總覽

### 核心文件

```
Yao/final/CLD/
├── tools/
│   ├── dlcv_dataset_indexed.py    ⭐ 方案 B：Index-based dataset
│   ├── dlcv_dataset.py             原始 dataset（支持 path-based）
│   └── dataset.py                  PrismLayersPro 格式 dataset
│
├── convert_tadata_with_captions.py ⭐ 方案 A：轉換腳本
├── verify_converted_dataset.py     ⭐ 方案 A：驗證腳本
├── test_indexed_dataset.py         ⭐ 方案 B：測試腳本
│
├── configs/
│   ├── infer_tadata_indexed.json   ⭐ 方案 B 配置
│   └── infer.json                  方案 A 配置
│
├── infer/
│   └── infer.py                    ⭐ 支持兩種方案的 inference
│
├── quick_test_plan_b.sh            ⭐ 方案 B 快速測試腳本
├── CHOOSE_PLAN.md                  ⭐ 選擇指南
├── PLAN_B_GUIDE.md                 方案 B 詳細說明
└── README_DATASET_SOLUTIONS.md     本文件
```

### 數據文件

```
/workspace/dataset/
├── TAData/
│   └── DLCV_dataset/
│       └── data/
│           ├── train-00000-of-00031.parquet
│           ├── train-00001-of-00031.parquet
│           └── ...                 ⭐ 原始 TAData（Image 對象）
│
└── TAData_with_llava_captions/     ⭐ 方案 A 轉換結果
    ├── preview/                    所有 preview 圖片
    ├── layers/                     所有 layer 圖片
    └── train-xxxxx.parquet         包含 caption 的 parquet

/workspace/ShuoIsAllYouNeed/Yao/final/CLD/
└── caption_llava16_final.json      ⭐ LLaVA captions（19480 個）
```

---

## 🚀 快速開始

### 方案 B（立即可用）⚡

**1 分鐘內開始使用！**

```bash
# 在 workspace 機器
cd /workspace/ShuoIsAllYouNeed/Yao/final/CLD

# 快速測試
chmod +x quick_test_plan_b.sh
./quick_test_plan_b.sh

# 開始 inference（測試 5 個樣本）
python infer/infer.py \
    --config configs/infer_tadata_indexed.json \
    --max_samples 5
```

**核心設定：**
```json
{
  "data_dir": "/workspace/dataset/TAData/DLCV_dataset/data",
  "caption_json": "/workspace/.../caption_llava16_final.json",
  "use_indexed_dataset": true
}
```

---

### 方案 A（轉換後使用）📦

**需要先轉換（約 45 分鐘）：**

```bash
# 1. 轉換數據集
python convert_tadata_with_captions.py \
    --input_dir /workspace/dataset/TAData/DLCV_dataset/data \
    --output_dir /workspace/dataset/TAData_with_llava_captions \
    --caption_json /workspace/.../caption_llava16_final.json

# 2. 驗證結果
python verify_converted_dataset.py \
    --data_dir /workspace/dataset/TAData_with_llava_captions \
    --num_samples 5

# 3. 使用（不需要 caption_json！）
python infer/infer.py \
    --config configs/infer.json \
    --data_dir /workspace/dataset/TAData_with_llava_captions
```

**核心設定：**
```json
{
  "data_dir": "/workspace/dataset/TAData_with_llava_captions"
  // 不需要 caption_json！
  // 不需要 use_indexed_dataset！
}
```

---

## 🎯 使用場景

### 場景 1：快速測試 → 方案 B

**你想：** 馬上看到 inference 結果

**操作：**
```bash
./quick_test_plan_b.sh
python infer/infer.py --config configs/infer_tadata_indexed.json --max_samples 10
```

**時間：** 5 分鐘內看到結果

---

### 場景 2：開發調試 → 方案 B

**你想：** 快速迭代，經常修改 caption

**操作：**
1. 修改 `caption_llava16_final.json`
2. 直接重新執行（不需要重新轉換）

**優點：** 修改成本低

---

### 場景 3：正式訓練 → 方案 A 或 B

**你想：** 開始完整的訓練

**方案 B（如果轉換還沒完成）：**
```bash
python train/train.py \
    --data_dir /workspace/dataset/TAData/DLCV_dataset/data \
    --caption_json /workspace/.../caption_llava16_final.json \
    --use_indexed_dataset true
```

**方案 A（如果轉換已完成）：**
```bash
python train/train.py \
    --data_dir /workspace/dataset/TAData_with_llava_captions
```

---

### 場景 4：數據共享/發布 → 方案 A

**你想：** 給別人使用或備份數據

**操作：**
```bash
# 打包
cd /workspace/dataset
tar -czf TAData_with_llava_captions.tar.gz TAData_with_llava_captions/

# 對方使用
tar -xzf TAData_with_llava_captions.tar.gz
python train/train.py --data_dir TAData_with_llava_captions/
```

**優點：** 自包含，不需要額外文件

---

## 🔍 技術細節

### 方案 B 的核心原理

**問題：** TAData 的 `preview` 是 Image 對象，怎麼匹配 caption？

**解決：** 提取 caption 路徑中的數字作為 index

```python
# Caption JSON:
{
  "/workspace/.../00000000.png": "caption for sample 0",
  "/workspace/.../00000123.png": "caption for sample 123"
}

# 轉換為 index-based:
caption_mapping = {
  0: "caption for sample 0",
  123: "caption for sample 123"
}

# 查找:
sample = dataset[123]  # TAData 的第 123 個樣本
caption = caption_mapping[123]  # 直接用 index 匹配！
```

### 方案 A 的核心原理

**做什麼：** 把 Image 對象存成文件，caption 內嵌到 parquet

```python
# 原始 TAData:
{
  'preview': <PIL.Image object>,
  'image': [<PIL.Image>, <PIL.Image>, ...],
  'title': 'simple title'
}

# 轉換後:
{
  'preview': '/path/to/00000123_preview.png',
  'image': ['/path/to/00000123_layer_00.png', ...],
  'title': 'Complete LLaVA caption from JSON'  # ⭐ 已更新
}
```

---

## 📊 對比表

| 項目 | 方案 A | 方案 B |
|------|--------|--------|
| **等待時間** | 45-50 分鐘 | 0 秒 ⚡ |
| **磁碟空間** | ~2x 原始大小 | 原始大小 |
| **caption 位置** | parquet 內 | 外部 JSON |
| **修改 caption** | 需重新轉換 | 直接改 JSON |
| **跨機器** | 簡單（一個目錄） | 需要兩個文件 |
| **適合場景** | 最終訓練、發布 | 開發、測試 |

---

## 💡 最佳實踐

### 推薦策略：兩個都用！

```
階段 1（現在）
    ↓
  方案 B ⚡
  - 立即開始開發
  - 測試 inference
  - 調試 captions
    ↓
階段 2（背景）
    ↓
  方案 A 🔄
  - 轉換在背景執行
  - 45 分鐘完成
    ↓
階段 3（之後）
    ↓
  選擇使用
  - 開發：繼續用方案 B
  - 訓練：用方案 A
  - 發布：用方案 A
```

---

## 🆘 故障排除

### 問題 1：`TypeError: must be called with a dataclass type`

**原因：** `datasets` 版本不兼容 TAData metadata

**解決：** 已在 `convert_tadata_with_captions.py` 中自動處理（fallback 到 pyarrow）

---

### 問題 2：Caption 匹配不上

**檢查：**
```bash
# 測試 dataset
python test_indexed_dataset.py

# 會顯示前 3 個樣本的 caption
# 檢查是否正確
```

**常見原因：**
- Caption JSON 路徑錯誤
- Caption JSON 格式不正確
- Index 不匹配（不太可能，因為你是按順序生成的）

---

### 問題 3：找不到模組

**錯誤：** `ModuleNotFoundError: No module named 'dlcv_dataset_indexed'`

**解決：**
```bash
# 確認文件存在
ls tools/dlcv_dataset_indexed.py

# 如果不存在，從 meow1 複製
scp meow1:/tmp2/.../dlcv_dataset_indexed.py tools/
```

---

## 📚 更多文檔

- **[CHOOSE_PLAN.md](CHOOSE_PLAN.md)** - 詳細的方案選擇指南
- **[PLAN_B_GUIDE.md](PLAN_B_GUIDE.md)** - 方案 B 完整說明
- **快速測試：** `./quick_test_plan_b.sh`

---

## 🎉 總結

你現在有兩個可用的方案：

1. **方案 B（Index-based）：** 
   - ⚡ 立即可用
   - 💾 節省空間
   - 🔄 易於更新

2. **方案 A（轉換）：**
   - 📦 自包含
   - 🚀 更穩定
   - 🌐 易於分享

**建議：現在用方案 B 開始工作，同時讓方案 A 的轉換在背景執行！**

## 🚀 下一步

```bash
# 在 workspace 機器執行
cd /workspace/ShuoIsAllYouNeed/Yao/final/CLD
./quick_test_plan_b.sh
```

**5 分鐘內開始工作！** 🎯
