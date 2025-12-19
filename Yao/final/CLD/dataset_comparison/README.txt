======================================================================
數據集對比說明
======================================================================

目錄結構:
  DLCV_sample_XX/          - DLCV 數據集的樣本
  PrismLayersPro_sample_XX/ - PrismLayersPro 數據集的樣本

每個樣本目錄包含:
  - info.txt               - 樣本資訊
  - overview.png           - 所有圖層的總覽
  - layer_XX_RGBA.png      - 各圖層的 RGBA 版本（帶透明度）
  - layer_XX_RGB.png       - 各圖層的 RGB 版本（無透明度）
  - whole_image_original.png - 原始完整圖片

圖層結構 (CLD 訓練格式):
  Layer 0: Whole Image     - 完整圖片（所有圖層疊加的結果）
  Layer 1: Base/Background - 純背景（沒有前景物件）
  Layer 2+: Foreground     - 各個前景圖層（物件、文字等）

檢查重點:
  1. Layer 0 是否是完整的合成圖片？
  2. Layer 1 是否是純背景（沒有前景物件）？
  3. Layer 2+ 是否包含正確的前景圖層？
  4. 每個圖層的透明度處理是否正確？
  5. Bounding box 是否與圖層對應？

