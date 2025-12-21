

### 1.1 資料結構說明

**輸入資料**

* `json_dir/`：模型輸出的 bbox JSON（一張圖一個檔案）
* `img_dir/`：對應的原始圖片（PNG 或 JPG）

**輸出結果**

* `output_dir/`

  * `*_vis.png`：畫好 bbox、merge 結果與 IQS 分數的圖片
  * `summary.json`：每張圖片的分數與統計資訊

---

### 1.2 完整執行指令

```bash
python evaluate_bbox.py \
  --json_dir configs/exp001/outputs/pipeline_outputs/rtdetr \
  --img_dir configs/exp001/inputs \
  --output_dir visual_results \
  --merge_mode word \
  --draw_scores \
  --draw_score_details
```

常用參數說明：

* `--merge_mode word`：將字母級 bbox 合併成單字
* `--draw_scores`：顯示每個 bbox 的 confidence
* `--draw_score_details`：顯示 IQS 的各個子分數（H / E / C / L）

---


## 2. IQS 分數怎麼算？

IQS 由四個直覺化的指標組成，每一項都對應一種「人眼會在意的問題」。

最後分數會落在 **0 ~ 100 分**。

---

### 2.1 總分公式（概念版）

```text
IQS = 高信心程度
    + 合併是否有效
    + 覆蓋是否合理
    - 低品質框比例
```

實際上是加權後再正規化，確保「完美結果」可以接近 100 分。

---

### 2.2 H — High-confidence Strength（高信心程度）

**問題在問：**

> 這張圖的 bbox，是不是真的很有把握？

設計方式：

* 只有 confidence ≥ 0.75 的 bbox 才會加分
* confidence 越接近 1，加分越多

直覺：

* 很多 0.99 的框 → H 很高
* 只有 0.6~0.7 的框 → H 偏低

---

### 2.3 E — Merge Effectiveness（碎度控制）

**問題在問：**

> bbox 是不是「合理但太碎」？

在設計海報或版面時：

* 字母級 bbox 本來就會很多
* 但它們其實屬於同一個單字或句子

因此：

* 我們會先把 **位置與尺寸合理的小框合併**
* 再比較：

  * 合併前有多少框（N）
  * 合併後剩多少框（K）

設計邏輯：

* 框很少 → 不需要懲罰
* 框很多但能順利合併 → E 高
* 框很多又亂，合不起來 → E 低

---

### 2.4 C — Coverage Sanity（覆蓋合理性）

**問題在問：**

> 模型有沒有抓到「主要內容」？

我們只看：

* **最大的 bbox 佔整張圖的比例**

原因是：

* 海報通常本來就有一個很大的背景或主圖框
* 大框不等於錯

合理情況：

* 最大框太小 → 幾乎沒抓到東西（扣分）
* 最大框太大（接近 100%）→ 不合理（扣分）
* 介於中間 → C 高

---

### 2.5 L — Noise Rate（雜訊比例）

**問題在問：**

> 有沒有很多低品質的亂框？

定義方式：

* 計算 confidence < 0.75 的 bbox 比例

直覺：

* 全部都是綠框 → L = 0
* 一堆黃框、紅框 → L 上升 → 扣分

---

## 3. 分數該怎麼看？

建議解讀方式：

| IQS 分數  | 解讀                |
| ------- | ----------------- |
| ≥ 80    | GOOD：乾淨、可直接使用     |
| 55 ~ 80 | OK：可用，但可能偏碎或有少量雜訊 |
| < 55    | BAD：亂框多，不適合後續分析   |

---

## 4. 限制與挑戰

### 5.1 沒有 Ground Truth

* IQS 不是標準 benchmark
* 不能與其他模型直接比較
* 目的是 **結果品質評估，而非模型排名**

---

### 4.2 Merge 不會修正錯誤

語義合併的用途是：

* 避免「合理但過碎」被誤判為差

它 **不會** 把錯誤 bbox 變成正確：

* 真正的錯誤框通常：

  * confidence 低
  * 無法形成語義結構
  * 破壞覆蓋合理性

因此仍會在 H、E、C、L 多個面向被扣分。

---

### 4.3 複雜版面仍具挑戰性

例如：

* 斜體或曲線文字
* 高度藝術化排版
* 裝飾元素與文字混合

這些情況未來可透過：

* OCR
* 文字方向估計
* 更進階的 layout clustering 改善

---

## 總結

IQS 提供了一種 **在沒有 Ground Truth 的情況下，仍能判斷 bbox 結果是否乾淨、合理、可用的方法**。

它特別適合用於：

* 模型輸出檢查
* 結果篩選
* 後續 Canvas Layout Decomposition（CLD）前的品質控管
