#!/bin/bash
#
# 部署所有修改到 workspace 機器
# 在 meow1 機器執行
#

set -e

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================="
echo "部署方案 B 到 workspace 機器"
echo -e "==========================================${NC}\n"

# 設定目標機器（根據實際情況修改）
TARGET_USER="your_username"
TARGET_HOST="workspace_machine"
TARGET_DIR="/workspace/ShuoIsAllYouNeed/Yao/final/CLD"

echo -e "${YELLOW}請確認目標機器資訊：${NC}"
echo "  User: $TARGET_USER"
echo "  Host: $TARGET_HOST"
echo "  Directory: $TARGET_DIR"
echo ""
read -p "確認無誤？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}已取消${NC}"
    exit 1
fi

# 目標位置
REMOTE="${TARGET_USER}@${TARGET_HOST}:${TARGET_DIR}"

echo -e "\n${GREEN}開始複製文件...${NC}\n"

# 1. 核心功能文件
echo -e "${YELLOW}1. 複製核心功能文件${NC}"

echo "  → tools/dlcv_dataset_indexed.py"
scp tools/dlcv_dataset_indexed.py "${REMOTE}/tools/"

echo "  → train/train.py"
scp train/train.py "${REMOTE}/train/"

echo "  → infer/infer.py"
scp infer/infer.py "${REMOTE}/infer/"

# 2. 配置文件
echo -e "\n${YELLOW}2. 複製配置文件${NC}"

echo "  → train/train_tadata_indexed.yaml"
scp train/train_tadata_indexed.yaml "${REMOTE}/train/"

echo "  → configs/infer_tadata_indexed.json"
scp configs/infer_tadata_indexed.json "${REMOTE}/configs/"

# 3. 測試和輔助腳本
echo -e "\n${YELLOW}3. 複製測試和輔助腳本${NC}"

echo "  → test_indexed_dataset.py"
scp test_indexed_dataset.py "${REMOTE}/"

echo "  → quick_test_plan_b.sh"
scp quick_test_plan_b.sh "${REMOTE}/"

# 4. 文檔（可選）
echo -e "\n${YELLOW}4. 複製文檔${NC}"

echo "  → DETAILED_OUTPUT_GUIDE.md"
scp DETAILED_OUTPUT_GUIDE.md "${REMOTE}/"

echo "  → SUMMARY_OF_CHANGES.md"
scp SUMMARY_OF_CHANGES.md "${REMOTE}/"

echo "  → README_DATASET_SOLUTIONS.md"
scp README_DATASET_SOLUTIONS.md "${REMOTE}/"

echo "  → CHOOSE_PLAN.md"
scp CHOOSE_PLAN.md "${REMOTE}/"

echo "  → PLAN_B_GUIDE.md"
scp PLAN_B_GUIDE.md "${REMOTE}/"

echo -e "\n${GREEN}=========================================="
echo "✓ 所有文件複製完成！"
echo -e "==========================================${NC}\n"

echo -e "${YELLOW}接下來在 workspace 機器執行：${NC}\n"
echo "  cd ${TARGET_DIR}"
echo "  chmod +x quick_test_plan_b.sh"
echo "  ./quick_test_plan_b.sh"
echo ""
echo -e "${GREEN}開始使用！${NC} 🚀"


