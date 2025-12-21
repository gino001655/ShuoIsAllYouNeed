#!/bin/bash
#
# 快速測試方案 B：Index-based Caption Matching
# 不需要轉換數據集，直接使用 TAData + caption.json
#

set -e

echo "=========================================="
echo "方案 B：快速測試"
echo "=========================================="
echo ""

# 設定路徑（根據你的機器調整）
DATA_DIR="/workspace/dataset/TAData/DLCV_dataset/data"
CAPTION_JSON="/workspace/ShuoIsAllYouNeed/Yao/final/CLD/caption_llava16_final.json"
CLD_DIR="/workspace/ShuoIsAllYouNeed/Yao/final/CLD"

cd "$CLD_DIR"

echo "1️⃣  檢查文件..."
echo ""

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ TAData 目錄不存在: $DATA_DIR"
    exit 1
fi
echo "✓ TAData 目錄存在"

if [ ! -f "$CAPTION_JSON" ]; then
    echo "❌ Caption JSON 不存在: $CAPTION_JSON"
    exit 1
fi
echo "✓ Caption JSON 存在"

if [ ! -f "tools/dlcv_dataset_indexed.py" ]; then
    echo "❌ indexed dataset 模組不存在"
    echo "   請先從 meow1 複製: tools/dlcv_dataset_indexed.py"
    exit 1
fi
echo "✓ Indexed dataset 模組存在"

echo ""
echo "2️⃣  測試 Dataset（前 3 個樣本）..."
echo ""

python test_indexed_dataset.py

echo ""
echo "=========================================="
echo "✅ 測試完成！"
echo "=========================================="
echo ""
echo "接下來你可以："
echo ""
echo "📌 方法 1: 快速測試 inference（5 個樣本）"
echo "   python infer/infer.py \\"
echo "     --config configs/infer_tadata_indexed.json \\"
echo "     --max_samples 5"
echo ""
echo "📌 方法 2: 完整 inference"
echo "   修改 configs/infer_tadata_indexed.json 中的模型路徑"
echo "   然後執行: python infer/infer.py --config configs/infer_tadata_indexed.json"
echo ""
echo "📌 方法 3: 如果想要轉換數據集（方案 A）"
echo "   讓之前的轉換繼續跑完即可"
echo ""


