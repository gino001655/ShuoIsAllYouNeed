# RT-DETR Layout Predictor

這是一個封裝好的 RT-DETR 模型預測工具，專門用於預測圖片的圖層 Layout，並支援輸出為 CLD 訓練可用的 Parquet 格式。

## 目錄結構
```
RT-DETR-Predictor/
├── configs/   # 設定檔
├── weights/   # 模型權重 (Checkpoint)
├── src/       # 核心程式碼
└── tools/     # 執行腳本
    ├── infer_layer.py        # 單張圖片推論
    └── batch_infer_cld.py    # 批次推論 (輸出 Parquet)
```

## 環境需求
請確保環境中安裝了以下套件：
- `torch`, `torchvision`
- `PIL` (Pillow)
- `numpy`
- `pandas` (用於批次輸出)
- `pyarrow` (用於 Parquet 格式)
- `gdown` (用於下載權重)

## 下載模型權重 (Download Weights)
在使用之前，請先執行以下腳本下載模型權重：

```bash
chmod +x download_ckpt.sh
./download_ckpt.sh
```
這將會下載 `checkpoint000.pth` 到 `weights/` 資料夾。

## 1. 單張圖片推論 (Single Image)

如果您只想測試單張圖片並查看可視化結果 (`.jpg`)：

```bash
python tools/infer_layer.py \
    -c configs/rtdetr/rtdetr_r50vd_layer_prediction.yml \
    -r weights/checkpoint000.pth \
    --rgb-file <輸入圖片路徑> \
    --use-zero-depth \
    --expand-ratio 1.1 \
    -o result.jpg \
    -d cuda:0 \
    -t 0.4
```
以上皆為推薦參數
- `--rgb-file`: 輸入圖片路徑。
- `--use-zero-depth`: **重要**，採用不吃深度圖的模式（我之後會拿一個數據比較說這個沒差多少，所以選擇用流程簡單的這個模式　）。
- `--expand-ratio`: Box 擴充倍率 (例如 `1.1` 代表擴充 10%)。
- `-o`: 輸出的可視化圖片檔名。
- `-t`: 信心閾值 (Threshold)。

---

## 2. 批次推論 (Batch Inference) → CLD 格式

如果您有一整個資料夾的圖片，並希望轉換為 CLD (Co-Speech Layout Diffusion) 模型可直接讀取的格式 (Parquet)：

```bash
python tools/batch_infer_cld.py \
    -c configs/rtdetr/rtdetr_r50vd_layer_prediction.yml \
    -r weights/checkpoint0005.pth \
    --input-dir <圖片資料夾路徑> \
    --output-dir <輸出資料夾路徑> \
    --use-zero-depth \
    --expand-ratio 1.1 \
    --save-vis \
    -t 0.4 \
    -d cuda:0
```

### 參數說明
- `-t`: 信心閾值 (Threshold)，預設 0.3。如果覺得框太多或太少可調整此值。
- `--save-vis`: 若加上此參數，會在 `output-dir/visualizations/` 下產生畫上框的圖片，方便人類檢視預測結果。

### 輸出說明
執行完畢後，`output-dir` 內會產生符合 CLD 要求的目錄結構：

```
output-dir/
├── snapshots/
│   └── snapshot_1/
│       └── data/
│           └── inference-00000-of-00001.parquet  <-- CLD 用
└── visualizations/
    └── piccollage_xxx.png                        <-- 人類用 (若有加 --save-vis)
```

這個 `inference-*.parquet` 檔案可以直接被 CLD 的 `DLCVLayoutDataset` 讀取並用於生成。
Parquet 內部包含了圖片絕對路徑、預測的 Layout (Box 座標) 以及其他必要資訊。

### 注意事項
- **圖片路徑**：Parquet 檔案中儲存的是圖片的 **絕對路徑**。如果你把 output 移動到別的電腦，請確保圖片原始路徑依然有效，或自行修改 Parquet 內容。
