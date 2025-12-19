# LLaVA 環境安裝指南

## 📋 方案選擇

根據你的情況，有兩個方案：

### 方案 A: 使用現有的 LLaVA 環境（如果已存在）
如果 `/tmp2/b12902041/Gino/dlcv-fall-2025-final-project/` 中已經有 LLaVA，可以直接使用。

### 方案 B: 全新安裝 LLaVA（推薦）
在你自己的目錄下安裝 LLaVA。

---

## 🚀 方案 B: 全新安裝 LLaVA（推薦）

### Step 1: 創建 conda 環境

```bash
# 創建新環境（Python 3.10）
conda create -n llava15 python=3.10 -y

# 啟動環境
conda activate llava15
```

### Step 2: 安裝 PyTorch

```bash
# 安裝 PyTorch（CUDA 11.8 版本，根據你的 CUDA 版本調整）
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 或者 CUDA 12.1 版本
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 驗證 PyTorch 安裝
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: Clone LLaVA Repository

```bash
# 進入你的工作目錄
cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD

# Clone LLaVA
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

# 或者使用特定版本（更穩定）
# git clone -b v1.2.0 https://github.com/haotian-liu/LLaVA.git
```

### Step 4: 安裝 LLaVA 依賴

```bash
# 確保在 llava15 環境中
conda activate llava15

# 進入 LLaVA 目錄
cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD/LLaVA

# 安裝依賴
pip install --upgrade pip
pip install -e .

# 如果上面失敗，嘗試：
# pip install -e ".[train]"
```

### Step 5: 安裝額外依賴

```bash
# 安裝其他需要的包
pip install datasets
pip install pyarrow
pip install fastparquet
pip install transformers
pip install accelerate
pip install bitsandbytes  # 用於 4-bit/8-bit 量化
pip install sentencepiece
pip install protobuf
```

### Step 6: 驗證安裝

```bash
python -c "
from llava.model.builder import load_pretrained_model
print('✅ LLaVA 模組可以導入！')
"
```

### Step 7: 修改 Caption 生成腳本

**修改 `generate_captions_for_training.py` 第 20 行**：

```python
# 原來
LLAVA_DIR = Path("/tmp2/b12902041/Gino/dlcv-fall-2025-final-project/third_party/llava")

# 改為你的 LLaVA 路徑
LLAVA_DIR = Path("/tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD/LLaVA")
```

---

## 🔧 方案 A: 使用現有的 LLaVA 環境

如果專案中已經有 LLaVA 環境：

```bash
# 檢查環境
conda env list | grep llava

# 啟動環境（名稱可能是 llava, llava15, 或其他）
conda activate llava15  # 或 conda activate llava

# 測試
python -c "
import sys
sys.path.insert(0, '/tmp2/b12902041/Gino/dlcv-fall-2025-final-project/third_party/llava')
from llava.model.builder import load_pretrained_model
print('✅ LLaVA 可用！')
"
```

---

## 📦 簡化版安裝（最小依賴）

如果只是要生成 caption，可以用更簡單的方式：

```bash
# 創建環境
conda create -n llava_simple python=3.10 -y
conda activate llava_simple

# 安裝 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Clone 並安裝 LLaVA
cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .

# 安裝必要的包
pip install datasets transformers accelerate bitsandbytes protobuf sentencepiece
```

---

## 🐛 常見問題排查

### Q1: `import llava` 失敗

**檢查**：
```bash
# 確認環境
conda activate llava15

# 檢查 LLAVA_DIR 路徑
ls /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD/LLaVA

# 手動添加路徑測試
python -c "
import sys
sys.path.insert(0, '/path/to/LLaVA')
from llava.model.builder import load_pretrained_model
print('成功！')
"
```

### Q2: CUDA 不可用

**檢查**：
```bash
python -c "import torch; print(torch.cuda.is_available())"

# 如果是 False，檢查 CUDA 版本
nvcc --version

# 重新安裝對應版本的 PyTorch
```

### Q3: 顯存不足

**解決方案**：
```bash
# 在 generate_captions_for_training.py 中使用 4-bit 量化
python generate_captions_for_training.py \
    --load_4bit \
    ...
```

### Q4: `bitsandbytes` 安裝失敗

**Linux**：
```bash
pip install bitsandbytes
```

**如果失敗**：
```bash
# 從源碼安裝
pip install git+https://github.com/TimDettmers/bitsandbytes.git
```

### Q5: 模型下載很慢

**使用 HuggingFace 鏡像**：
```bash
# 設置環境變量
export HF_ENDPOINT=https://hf-mirror.com

# 或在 Python 中
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

---

## 🎯 快速驗證清單

安裝完成後，運行這些測試：

```bash
# 1. 啟動環境
conda activate llava15

# 2. 測試 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 3. 測試 LLaVA 導入
python -c "
import sys
sys.path.insert(0, '/tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD/LLaVA')
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
print('✅ LLaVA 模組正常！')
"

# 4. 測試其他依賴
python -c "
from datasets import load_dataset
from PIL import Image
from transformers import AutoTokenizer
print('✅ 所有依賴正常！')
"

# 5. 測試 Caption 腳本（10 個樣本）
cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD
python generate_captions_for_training.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output test_caption.json \
    --max_samples 10
```

---

## 📊 系統需求

### 硬體需求
- **GPU**: NVIDIA GPU with ≥ 16GB VRAM（推薦 24GB+）
- **RAM**: ≥ 32GB
- **Storage**: ≥ 50GB（for model cache）

### 軟體需求
- **OS**: Linux（推薦 Ubuntu 20.04+）
- **CUDA**: 11.7+ 或 12.1+
- **Python**: 3.9 或 3.10
- **Conda**: Miniconda 或 Anaconda

---

## 🎉 安裝完成後

完成後你應該有：

```bash
# 環境
llava15 (Python 3.10)

# LLaVA codebase
/tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD/LLaVA/

# 可以運行
python generate_captions_for_training.py --help
```

**現在可以開始生成 captions 了！** 🚀

---

## 💡 推薦的完整流程

```bash
# 1. 安裝環境（一次性）
conda create -n llava15 python=3.10 -y
conda activate llava15
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Clone LLaVA
cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
pip install datasets transformers accelerate bitsandbytes protobuf sentencepiece

# 3. 修改腳本中的 LLAVA_DIR 路徑

# 4. 測試
cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD
python generate_captions_for_training.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output test.json \
    --max_samples 2

# 5. 全量生成
python generate_captions_for_training.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output caption_mapping_full.json
```

**祝安裝順利！** 🎊

