# 文件清單：需要部署的所有文件

## ✅ 必需文件（核心功能）

### 1. Dataset 類
- [ ] `tools/dlcv_dataset_indexed.py` ⭐ **核心** - Index-based dataset

### 2. 主程序
- [ ] `train/train.py` - 訓練腳本（已修改，支持 indexed dataset）
- [ ] `infer/infer.py` - 推理腳本（已修改，支持 indexed dataset）

### 3. 配置文件
- [ ] `train/train_tadata_indexed.yaml` - 訓練配置範例
- [ ] `configs/infer_tadata_indexed.json` - 推理配置範例

## 🔧 輔助文件（推薦）

### 4. 測試腳本
- [ ] `test_indexed_dataset.py` - 測試 dataset 功能
- [ ] `quick_test_plan_b.sh` - 一鍵測試腳本

### 5. 部署腳本
- [ ] `DEPLOY_TO_WORKSPACE.sh` - 自動部署腳本

## 📚 文檔（可選但推薦）

- [ ] `SUMMARY_OF_CHANGES.md` - 修改總結 ⭐
- [ ] `DETAILED_OUTPUT_GUIDE.md` - 詳細輸出指南 ⭐
- [ ] `README_DATASET_SOLUTIONS.md` - 完整方案說明
- [ ] `CHOOSE_PLAN.md` - 方案選擇指南
- [ ] `PLAN_B_GUIDE.md` - 方案 B 詳細說明

---

## 📦 快速部署（方法 1：自動）

```bash
# 在 meow1 機器
cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD

# 修改 DEPLOY_TO_WORKSPACE.sh 中的目標機器資訊
# 然後執行：
./DEPLOY_TO_WORKSPACE.sh
```

---

## 📦 快速部署（方法 2：手動）

### 在 meow1 機器執行：

```bash
cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD

# 設定目標（根據實際情況修改）
REMOTE="user@workspace:/workspace/ShuoIsAllYouNeed/Yao/final/CLD"

# 複製核心文件
scp tools/dlcv_dataset_indexed.py ${REMOTE}/tools/
scp train/train.py ${REMOTE}/train/
scp infer/infer.py ${REMOTE}/infer/
scp train/train_tadata_indexed.yaml ${REMOTE}/train/
scp configs/infer_tadata_indexed.json ${REMOTE}/configs/

# 複製輔助文件
scp test_indexed_dataset.py ${REMOTE}/
scp quick_test_plan_b.sh ${REMOTE}/

# 複製文檔（可選）
scp SUMMARY_OF_CHANGES.md ${REMOTE}/
scp DETAILED_OUTPUT_GUIDE.md ${REMOTE}/
scp README_DATASET_SOLUTIONS.md ${REMOTE}/
```

---

## ✅ 驗證部署

### 在 workspace 機器執行：

```bash
cd /workspace/ShuoIsAllYouNeed/Yao/final/CLD

# 檢查文件是否存在
echo "檢查核心文件..."
ls -lh tools/dlcv_dataset_indexed.py
ls -lh train/train.py
ls -lh infer/infer.py
ls -lh train/train_tadata_indexed.yaml
ls -lh configs/infer_tadata_indexed.json

echo "檢查輔助文件..."
ls -lh test_indexed_dataset.py
ls -lh quick_test_plan_b.sh

echo ""
echo "✓ 文件檢查完成！"
```

---

## 🧪 測試部署

```bash
# 在 workspace 機器
cd /workspace/ShuoIsAllYouNeed/Yao/final/CLD

# 1. 快速測試
chmod +x quick_test_plan_b.sh
./quick_test_plan_b.sh

# 2. 測試 inference（5 個樣本）
python infer/infer.py \
    --config configs/infer_tadata_indexed.json \
    --max_samples 5

# 如果測試成功，你會看到：
# ✓ 載入 19479 個樣本
# ✓ 載入 19479 個 captions
# ============================================================
# 處理樣本 0
# ============================================================
#   📊 Canvas 尺寸: 1080 x 1080
#   🎨 圖層數量: 8
#   📝 Caption: ...
```

---

## 📝 修改配置

### 訓練配置 (`train/train_tadata_indexed.yaml`)

**必須修改的路徑：**
```yaml
# 數據路徑（根據你的機器調整）
data_dir: "/workspace/dataset/TAData/DLCV_dataset/data"
caption_mapping: "/workspace/.../caption_llava16_final.json"

# 模型路徑（根據你的機器調整）
artplus_lora_dir: "/workspace/.../ckpt/prism_ft_LoRA"
resume_from: "/workspace/.../ckpt/decouple_LoRA"
pretrained_model_name_or_path: "flux_model"
pretrained_adapter_path: "Path_to_pretrained_FLUX_adapter"
output_dir: "FT_on_TAData_ckpt"
```

### 推理配置 (`configs/infer_tadata_indexed.json`)

**必須修改的路徑：**
```json
{
  "model_path": "/workspace/your/model/path",
  "vae_path": "/workspace/your/vae/path",
  "t5_path": "/workspace/your/t5/path",
  "data_dir": "/workspace/dataset/TAData/DLCV_dataset/data",
  "caption_json": "/workspace/.../caption_llava16_final.json"
}
```

---

## 🎯 使用檢查表

### 部署前
- [ ] 確認 caption_llava16_final.json 已在 workspace 機器
- [ ] 確認 TAData 目錄存在
- [ ] 修改 DEPLOY_TO_WORKSPACE.sh 中的目標機器資訊

### 部署中
- [ ] 執行 DEPLOY_TO_WORKSPACE.sh 或手動複製
- [ ] 檢查所有文件都已複製

### 部署後
- [ ] 執行 quick_test_plan_b.sh 驗證
- [ ] 修改配置文件中的路徑
- [ ] 測試 inference（5 個樣本）
- [ ] 測試 training（幾個 steps）

### 確認功能
- [ ] Dataset 載入成功（19,479 個樣本）
- [ ] Caption 匹配成功
- [ ] 每個樣本顯示詳細資訊（Canvas、Layers、Caption）
- [ ] Training 每 10 步顯示詳情
- [ ] Inference 每個樣本顯示詳情

---

## 🆘 常見問題

### Q: 文件傳輸失敗
```bash
# 檢查連接
ssh user@workspace "ls /workspace/ShuoIsAllYouNeed/Yao/final/CLD"

# 手動複製單個文件
scp tools/dlcv_dataset_indexed.py user@workspace:/workspace/.../tools/
```

### Q: caption_llava16_final.json 不存在
```bash
# 在 meow1
scp /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD/caption_llava16_final.json \
    user@workspace:/workspace/ShuoIsAllYouNeed/Yao/final/CLD/
```

### Q: 測試失敗
```bash
# 查看詳細錯誤
python test_indexed_dataset.py 2>&1 | tee test_output.log

# 檢查路徑
cat configs/infer_tadata_indexed.json
cat train/train_tadata_indexed.yaml
```

---

## ✅ 完成標記

部署完成後，在 workspace 機器上：

```bash
# 創建一個標記文件
cat > DEPLOYMENT_INFO.txt << EOF
部署日期: $(date)
部署來源: meow1:/tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD
方案: B (Index-based caption matching)
數據集: TAData (19,479 個樣本)
Caption: caption_llava16_final.json (19,479 個)
狀態: ✅ 已驗證
EOF

echo "✓ 部署完成！"
cat DEPLOYMENT_INFO.txt
```

---

## 🎉 成功！

如果所有檢查都通過，你現在可以：
1. ✅ 開始 training
2. ✅ 開始 inference
3. ✅ 看到所有詳細資訊（Caption、Canvas、Layers）

**開始使用！** 🚀
