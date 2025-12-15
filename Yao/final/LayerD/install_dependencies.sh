#!/bin/bash
# 分批安裝依賴的腳本，避免連線中斷

set -e  # 遇到錯誤就停止

export PATH="$HOME/.local/bin:$PATH"

cd /tmp2/b12902041/Gino/DLCV/final/LayerD

echo "開始安裝依賴..."

# 方法1: 使用 pip 分批安裝（更穩定）
echo "使用 pip 分批安裝依賴..."

# 先安裝基礎套件
pip install numpy>=1.26 tqdm>=4.62.0

# 安裝 PyTorch 相關（最大型的套件）
echo "安裝 PyTorch..."
pip install torch>=2.5.1 torchvision>=0.20.1

# 安裝 OpenCV 和影像處理
echo "安裝影像處理套件..."
pip install opencv-python>=4.5.3 scikit-image>=0.25.0

# 安裝 Hugging Face 相關
echo "安裝 Hugging Face 相關套件..."
pip install huggingface-hub>=0.25.0 accelerate>=1.1.1 transformers==4.55.4 datasets>=2.14.4

# 安裝其他套件
echo "安裝其他套件..."
pip install hydra-core>=1.3.2 simple-lama-inpainting>=0.1.1 einops>=0.8.0 timm>=1.0.12 kornia>=0.5.0

# 安裝從 Git 的套件
echo "安裝 cr-renderer..."
pip install git+https://github.com/CyberAgentAILab/cr-renderer.git@a17e1fb

echo "安裝完成！"









