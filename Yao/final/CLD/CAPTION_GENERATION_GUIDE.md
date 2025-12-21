# Caption 生成使用指南

## 📋 概述

這個指南說明如何為訓練數據生成高質量的 LLaVA captions。

---

## 🎯 使用的 Prompt

```
Precisely describe style, subjects, text, and the background of the whole image in simple sentences.
```

**特點**：
- ✅ Precisely (精確描述)
- ✅ Style (風格)
- ✅ Subjects (主體物件)
- ✅ Text (文字內容) ← 新增！對設計圖很重要
- ✅ Background (背景)
- ✅ Simple sentences (簡單句子，易於理解)

---

## 🚀 使用步驟

### Step 1: 準備環境

**需要的 conda 環境**: `llava15`

```bash
# 確認環境存在
conda env list | grep llava15

# 如果不存在，需要創建（參考 LLaVA 官方文檔）
```

**需要的 LLaVA codebase**: `/tmp2/b12902041/Gino/dlcv-fall-2025-final-project/third_party/llava`

---

### Step 2: 測試生成（推薦先測試！）

**先用 10 個樣本測試**：

```bash
# 進入 CLD 目錄
cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD

# 啟動 llava15 環境並測試
conda activate llava15

python generate_captions_for_training.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output caption_mapping_test.json \
    --max_samples 10 \
    --device cuda
```

**檢查輸出**：
```bash
# 查看生成的 caption
cat caption_mapping_test.json | head -50

# 確認 caption 質量
```

**預期輸出範例**：
```json
{
  "/tmp2/b12902041/Gino/preprocessed_data/images/train/00000000.png": "A modern poster design with a blue gradient background, featuring bold white text 'SUMMER SALE' at the center, minimalist style with geometric shapes.",
  "/tmp2/b12902041/Gino/preprocessed_data/images/train/00000001.png": "An Instagram post template with a pink floral background, displaying the text 'Happy Birthday' in elegant script font, surrounded by watercolor flowers.",
  ...
}
```

---

### Step 3: 全量生成 Caption

**處理所有 18,000 個樣本**：

```bash
conda activate llava15

python generate_captions_for_training.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output caption_mapping_full.json \
    --device cuda \
    --load_4bit
```

**參數說明**：
- `--data_dir`: parquet 文件目錄
- `--output`: 輸出的 caption 映射文件
- `--device`: 使用的設備（cuda/cpu）
- `--load_4bit`: 使用 4-bit 量化（節省顯存）

**預計時間**：
- 每個樣本約 2-3 秒
- 18,000 個樣本約需 **10-15 小時**

**注意事項**：
- ✅ 腳本每 100 個樣本自動保存一次
- ✅ 如果中斷，可以重新運行（會跳過已生成的）
- ✅ 使用 `--force` 可以強制重新生成

---

### Step 4: 斷點續傳（如果中斷）

如果生成過程中斷：

```bash
# 直接重新運行同樣的命令
# 腳本會自動跳過已經生成的 caption
python generate_captions_for_training.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output caption_mapping_full.json \
    --device cuda \
    --load_4bit
```

**不會重複生成**！腳本會：
1. 讀取現有的 `caption_mapping_full.json`
2. 跳過已經有 caption 的圖片
3. 只處理缺少的部分

---

### Step 5: 訓練時使用新 Caption

**修改 `train.yaml`**：

```yaml
# 原來的配置
data_dir: "/tmp2/b12902041/Gino/TAData/DLCV_dataset"

# 改為你自己的數據 + caption mapping
data_dir: "/tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1"
caption_mapping: "/tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD/caption_mapping_full.json"
```

**或者，在訓練腳本中手動指定**：

```python
# 在 train.py 中
from tools.dlcv_dataset import DLCVLayoutDataset

dataset = DLCVLayoutDataset(
    data_dir=config['data_dir'],
    split="train",
    caption_mapping_path="/path/to/caption_mapping_full.json"  # ← 新增！
)
```

---

## 📊 命令行參數完整說明

```bash
python generate_captions_for_training.py \
    --data_dir PATH              # 必需：parquet 文件目錄
    --output PATH                # 必需：輸出 JSON 文件
    --model MODEL_ID             # 可選：LLaVA 模型（默認 llava-v1.5-7b）
    --device cuda/cpu            # 可選：設備（默認 cuda）
    --load_4bit                  # 可選：使用 4-bit 量化（默認開啟）
    --max_samples N              # 可選：最多處理 N 個樣本（測試用）
    --force                      # 可選：強制重新生成所有 caption
```

---

## 🔍 檢查生成質量

### 查看生成的 Caption

```bash
# 查看前 10 個
python3 -c "
import json
with open('caption_mapping_full.json', 'r') as f:
    data = json.load(f)
for i, (path, caption) in enumerate(list(data.items())[:10]):
    print(f'{i+1}. {caption}')
"
```

### 統計信息

```bash
# Caption 數量
python3 -c "
import json
with open('caption_mapping_full.json', 'r') as f:
    data = json.load(f)
print(f'Total captions: {len(data)}')
"

# Caption 平均長度
python3 -c "
import json
with open('caption_mapping_full.json', 'r') as f:
    data = json.load(f)
lengths = [len(caption.split()) for caption in data.values()]
print(f'Average words: {sum(lengths)/len(lengths):.1f}')
print(f'Min words: {min(lengths)}')
print(f'Max words: {max(lengths)}')
"
```

---

## ⚠️ 常見問題

### Q1: 顯存不足

**解決方案**：
```bash
# 使用 4-bit 量化（已默認開啟）
--load_4bit

# 或使用 CPU（會很慢）
--device cpu
```

### Q2: LLaVA 導入失敗

**檢查**：
```bash
# 確認 LLaVA 目錄存在
ls /tmp2/b12902041/Gino/dlcv-fall-2025-final-project/third_party/llava

# 確認在正確的環境
conda activate llava15
```

### Q3: 生成的 Caption 太短

**原因**: `max_new_tokens=128` 可能不夠

**解決方案**：修改腳本第 84 行
```python
max_new_tokens=256,  # 增加到 256
```

### Q4: 想要更改 Prompt

**修改位置**：`generate_captions_for_training.py` 第 229 行

```python
prompt="Your custom prompt here.",
```

---

## 📈 預期結果

### 原來的 Caption（太簡單）
```
"A design image"
"A design image"
"A design image"
...
```

### 新的 Caption（詳細描述）
```
"A vibrant summer sale poster with a gradient blue background, featuring bold yellow text 'SUMMER SALE 50% OFF' in the center, modern flat design style with geometric patterns."

"An Instagram story template with a pastel pink background, displaying 'Happy Monday' in elegant handwritten font, decorated with small floral illustrations at the corners."

"A minimalist business card design on white background, showing company name 'Tech Solutions' in sans-serif font, with contact information and a simple geometric logo."
```

---

## ✅ 完整流程總結

```bash
# 1. 測試（10 個樣本）
conda activate llava15
cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD
python generate_captions_for_training.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output caption_mapping_test.json \
    --max_samples 10

# 2. 檢查質量
cat caption_mapping_test.json

# 3. 全量生成（約 10-15 小時）
python generate_captions_for_training.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output caption_mapping_full.json

# 4. 訓練時使用
# 修改 train.yaml 或在代碼中指定 caption_mapping_path
```

---

## 🎉 完成後

生成完成後，你會得到：
- ✅ `caption_mapping_full.json` - 包含 18,000 個高質量 captions
- ✅ 每個 caption 精確描述圖片的風格、主體、文字、背景
- ✅ 訓練時自動使用新 caption（比 "A design image" 好太多！）

**現在可以開始訓練了！訓練品質會大幅提升！** 🚀



