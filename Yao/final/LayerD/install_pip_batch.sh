#!/bin/bash
# 使用 pip 分批安裝，避免連線中斷

set -e

cd /tmp2/b12902041/Gino/DLCV/final/LayerD

echo "開始分批安裝依賴..."

# 第一批：基礎套件
echo "[1/5] 安裝基礎套件..."
pip install "numpy>=1.26" "tqdm>=4.62.0" "einops>=0.8.0" || true

# 第二批：PyTorch（最大型，可能需要較長時間）
echo "[2/5] 安裝 PyTorch（這可能需要較長時間）..."
pip install "torch>=2.5.1" "torchvision>=0.20.1" || true

# 第三批：影像處理
echo "[3/5] 安裝影像處理套件..."
pip install "opencv-python>=4.5.3" "scikit-image>=0.25.0" "kornia>=0.5.0" || true

# 第四批：Hugging Face 相關
echo "[4/5] 安裝 Hugging Face 相關套件..."
pip install "huggingface-hub>=0.25.0" "accelerate>=1.1.1" "transformers==4.55.4" "datasets>=2.14.4" || true

# 第五批：其他套件
echo "[5/5] 安裝其他套件..."
pip install "hydra-core>=1.3.2" "simple-lama-inpainting>=0.1.1" "timm>=1.0.12" || true

# Git 套件
echo "安裝 cr-renderer（從 Git）..."
pip install "git+https://github.com/CyberAgentAILab/cr-renderer.git@a17e1fb" || true

echo "安裝完成！"









