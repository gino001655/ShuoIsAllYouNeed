Layout Analysis Visualization Tool (DLCV Final Project)本工具用於視覺化 RT-DETR 模型在海報佈局分析（Layout Analysis）中的偵測結果。不同於傳統物件偵測，我們採用了**階層保留（Hierarchy Preservation）**策略，透過信心分級來評估模型表現，而非強制過濾重疊框。🚀 快速開始 (Quick Start)1. 環境安裝確保你的環境中已安裝 OpenCV 與 NumPy：Bashpip install opencv-python numpy tqdm
2. 單張圖片視覺化 (Single Image Mode)適用於針對特定 Case（如 00017532.png）進行深度檢查：Bashpython draw_bbox.py \
  configs/exp001/outputs/pipeline_outputs/rtdetr/00017532.json \
  configs/exp001/inputs/00017532.png \
  --output evaluation_result_00017532.png
3. 批次處理資料夾 (Batch Processing Mode)將整個資料夾的偵測結果一次性轉換為可視化圖片：Bashpython visualize_batch.py \
  --json_dir configs/exp001/outputs/pipeline_outputs/rtdetr \
  --img_dir configs/exp001/inputs \
  --output_dir visual_results
🧠 技術原理 (Technical Methodology)1. 信心分數計算 (Confidence Calculation)我們利用 PyTorch 框架實現兩階段的判定流程：$$C_{score} = \sigma(\text{Objectness}) \cdot \max P(\text{Class} | \text{Object})$$第一關：物件性判定 ($\sigma$)：判斷該區域是否存在具備特徵的設計元素（非背景）。第二關：類別機率 ($P$)：判定該物件屬於「文字」、「裝飾」或其他預定義類別。最終分數：兩者相乘確保只有「存在且類別明確」的物件能獲得高分。2. 信心分級視覺化 (Three-Tiered Visualization)為了呈現模型原始預測能力，我們設定了三色分級機制：🟢 綠色 (High Confidence): $C_{score} > 0.75$ —— 模型高度確信的可靠預測。🟡 黃色 (Moderate): $0.5 \le C_{score} \le 0.75$ —— 模型存在猶豫，需人工核對。🔴 紅色 (Low Confidence): $C_{score} < 0.5$ —— 可能為雜訊或誤判。3. 為什麼我們不使用 NMS (Non-Maximum Suppression)？在海報設計中，「大框包小框」（例如文字出現在背景裝飾之上）是標準的設計階層（Hierarchy）。傳統 NMS 會因 IoU（重疊度）過高而刪除內部的文字層，造成資訊遺失。本工具選擇**全量保留（Full Retention）**所有偵測框，以驗證模型是否成功學習到設計元素的階層結構，確保後續 CLD (Canvas Layout Decomposition) 流程的完整性。📂 檔案結構 (File Structure)draw_bbox.py: 基礎繪圖腳本，支援自定義輸出路徑。visualize_batch.py: 批次處理工具，支援自動檔案配對（JSON & PNG）。visual_results/: 預設的批次輸出目錄。
