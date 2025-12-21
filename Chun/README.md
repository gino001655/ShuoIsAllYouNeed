
### 1. 安裝環境（Setting Environments）

```markdown
pip install opencv-python numpy tqdm

```

### 2. 單張圖片視覺化 (Single Image Mode)

針對特定案例進行深度檢查，自動標註信心分數與判定狀態：

```bash
python draw_bbox.py \
  configs/exp001/outputs/pipeline_outputs/rtdetr/00017532.json \
  configs/exp001/inputs/00017532.png \
  --output evaluation_result_00017532.png

```

### 3. 批次處理資料夾 (Batch Processing Mode)

自動配對 JSON 與圖片資料夾，將結果統一輸出至指定目錄：

```bash
python visualize_batch.py \
  --json_dir configs/exp001/outputs/pipeline_outputs/rtdetr \
  --img_dir configs/exp001/inputs \
  --output_dir visual_results

```

---

## 🧠 技術原理 (Technical Methodology)

### 1. 信心分數計算機制 (Confidence Scoring)

本研究利用 **PyTorch** 框架實現了基於雙階段判定的聯合機率模型。最終的信心分數  並非單一指標，而是由兩個關鍵維度相乘而成：

* **物件性判定 ()**：模型首先透過 Sigmoid 函數評估該區域是否包含具備顯著特徵的「物件實體」，而非無意義的背景。
* **類別機率 ()**：在確認物件存在後，進一步比對該特徵與「文字」或「裝飾」等類別的符合程度。

**決策邏輯**：此乘積機制確保了只有當模型「確信有東西」且「看清是什麼」時，才會給予高分，有效過濾背景雜訊。

### 2. 三階層視覺化評估 (Three-Tiered Evaluation)

為了直觀量化模型表現，我們制定了基於信心分數的「紅綠燈」評估標準：

* 🟢 **綠色 (High Confidence)**: 
模型高度確信，屬於穩定且可靠的圖層偵測。
* 🟡 **黃色 (Moderate Confidence)**: 
模型偵測到特徵但存在猶豫，標記為潛在警告。
* 🔴 **紅色 (Low Confidence)**: 
模型信心不足，通常為誤判或品質不佳的候選框。

### 3. 階層保留策略 (Hierarchy Preservation vs. NMS)

在傳統物件偵測中，**NMS (非極大值抑制)** 用於刪除重疊的框。然而，在海報設計中，「大框包小框」（例如：背景底板包含多行文字）是標準的階層化佈局。

若使用 NMS，內部的文字框會因為與背景框 IoU (Intersection over Union) 重疊過高而被強制刪除，導致資訊遺失。

**本專案採用的策略：**

* **不以 IoU 作為過濾指標**：承認重疊框在佈局分析中的語義價值。
* **保留父子關係**：只要大框與小框皆具備高信心分數（綠色），即判定為正確的階層辨識。這對於後續 **CLD (Canvas Layout Decomposition)** 的圖層生成至關重要。

---

## 📂 檔案架構 (File Structure)

| 檔案名稱 | 用途說明 |
| --- | --- |
| `draw_bbox.py` | 核心繪圖腳本，處理單張圖片的 JSON 解析與可視化。 |
| `visualize_batch.py` | 批次處理工具，支援自動搜尋資料夾內的 JSON 與 PNG 進行配對。 |
| `visual_results/` | 程式自動生成的結果輸出目錄。 |

---

## ⚠️ 限制與挑戰 (Limitations & Future Work)

### 過度分割問題 (Over-Granularity / Lack of Semantic Grouping)
本工具忠實呈現了 RT-DETR 的原始偵測結果。雖然我們成功保留了「階層結構」，但在某些設計複雜的案例中，會暴露模型傾向於偵測**原子級元素（Atomic Elements）**而非**語義區塊（Semantic Blocks）**的特性。

**案例分析：**
如下圖的 "BLACK FRIDAY" 海報所示：
* **現象**：模型對每一個單獨的字母（B, L, A, C, K...）都給出了極高的信心分數，因此本工具將其全部標示為 **[GOOD]** 並予以保留。
* **問題**：雖然技術上偵測正確（這些確實是文字），但在佈局生成的應用場景中，我們通常更希望得到「單字級（Word-level）」或「行級（Line-level）」的框選，而非零散的字母框。
* **結論**：這顯示了單純依賴「信心分數」與「移除 NMS」雖然能救回階層資訊，但對於過度細碎的偵測結果，仍需要引入後處理演算法（如基於距離的 **Layout Clustering**）來進行語義分組，而非單純視為壞框刪除。

<img width="472" height="818" alt="598558139_1578974490196697_4718800710661783201_n" src="https://github.com/user-attachments/assets/9ecf6773-c625-4232-ba05-37bf4fa253ab" />

```

```

