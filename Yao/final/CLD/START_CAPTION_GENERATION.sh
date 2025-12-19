#!/bin/bash
# 一键启动 Caption 生成

set -e

echo "========================================"
echo "  并行 Caption 生成 - 快速启动脚本"
echo "========================================"
echo ""

# 检查环境
echo "📋 检查环境..."
if ! command -v conda &> /dev/null; then
    echo "❌ Conda 未找到"
    exit 1
fi

# 激活环境
echo "🔧 激活 llava 环境..."
source /tmp2/b12902041/miniconda3/etc/profile.d/conda.sh
conda activate llava

# 进入目录
cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD

# 检查 GPU
echo "🎮 检查 GPU..."
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader | head -5

echo ""
read -p "使用多少个 GPU? (1-6, 推荐 5): " NUM_GPUS
NUM_GPUS=${NUM_GPUS:-5}

echo ""
echo "🚀 生成并行脚本..."
python generate_captions_parallel.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output_dir . \
    --num_gpus $NUM_GPUS

echo ""
read -p "立即启动所有进程? (y/n): " START_NOW

if [[ "$START_NOW" == "y" || "$START_NOW" == "Y" ]]; then
    echo ""
    echo "🚀 启动所有进程..."
    bash run_parallel_caption_generation.sh
    
    echo ""
    echo "✅ 所有进程已启动！"
    echo ""
    echo "监控进度:"
    echo "  tail -f caption_gpu*.log"
    echo ""
    echo "查看 GPU 使用:"
    echo "  watch -n 1 nvidia-smi"
else
    echo ""
    echo "📝 手动启动:"
    echo "  bash run_parallel_caption_generation.sh"
fi

echo ""
echo "✅ 完成！"
