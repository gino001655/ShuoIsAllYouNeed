#!/bin/bash
# 使用 conda 創建環境，然後分批安裝依賴（避免連線中斷）

set -e

cd /tmp2/b12902041/Gino/DLCV/final/LayerD

ENV_NAME="layerd"

echo "=== 步驟 1: 創建 conda 環境 ==="
# 如果環境已存在，先刪除
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "環境 ${ENV_NAME} 已存在，是否要刪除並重建？(y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        conda env remove -n ${ENV_NAME} -y
    else
        echo "使用現有環境"
        conda activate ${ENV_NAME}
        exit 0
    fi
fi

# 創建新的 conda 環境
conda create -n ${ENV_NAME} python=3.10 -y

echo "=== 步驟 2: 啟動環境並分批安裝依賴 ==="
# 啟動環境並安裝
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo "[1/6] 安裝基礎套件..."
pip install "numpy>=1.26" "tqdm>=4.62.0" "einops>=0.8.0" || echo "部分套件安裝失敗，繼續..."

echo "[2/6] 安裝 PyTorch（這可能需要較長時間，請耐心等待）..."
pip install "torch>=2.5.1" "torchvision>=0.20.1" || echo "PyTorch 安裝失敗，請檢查網路連線"

echo "[3/6] 安裝影像處理套件..."
pip install "opencv-python>=4.5.3" "scikit-image>=0.25.0" "kornia>=0.5.0" || echo "部分套件安裝失敗，繼續..."

echo "[4/6] 安裝 Hugging Face 相關套件..."
pip install "huggingface-hub>=0.25.0" "accelerate>=1.1.1" || echo "部分套件安裝失敗，繼續..."

echo "[5/6] 安裝 transformers 和其他套件..."
pip install "transformers==4.55.4" "datasets>=2.14.4" "hydra-core>=1.3.2" "simple-lama-inpainting>=0.1.1" "timm>=1.0.12" || echo "部分套件安裝失敗，繼續..."

echo "[6/6] 安裝 cr-renderer（從 Git）..."
pip install "git+https://github.com/CyberAgentAILab/cr-renderer.git@a17e1fb" || echo "cr-renderer 安裝失敗，請檢查網路連線"

echo ""
echo "=== 安裝完成！ ==="
echo "使用以下命令啟動環境："
echo "  conda activate ${ENV_NAME}"
echo ""
echo "如果某些套件安裝失敗，可以手動重新安裝："
echo "  conda activate ${ENV_NAME}"
echo "  pip install <套件名稱>"








