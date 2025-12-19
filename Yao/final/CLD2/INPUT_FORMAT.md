# CLD Inference 輸入格式說明

## 📋 核心輸入需求

CLD inference 需要以下**5個必要資訊**：

### 1. **whole_img** (PIL Image)
- **類型**: `PIL.Image.Image`
- **說明**: 完整的合成圖片（作為 adapter image）
- **範例**: 
```python
from PIL import Image
whole_img = Image.open("path/to/image.png").convert("RGB")
```

### 2. **caption** (str)
- **類型**: `str`
- **說明**: 圖片描述文字（用於 text prompt）
- **範例**: 
```python
caption = "A beautiful design with colorful elements"
```

### 3. **layout** (list of lists)
- **類型**: `list[[w0, h0, w1, h1], ...]`
- **說明**: 圖層邊界框列表，每個元素是 `[w0, h0, w1, h1]`
  - `w0, h0`: 左上角座標
  - `w1, h1`: 右下角座標
- **範例**:
```python
layout = [
    [0, 0, 1024, 1024],      # 圖層 0: 整個畫布
    [0, 0, 1024, 1024],      # 圖層 1: 背景
    [100, 100, 500, 400],    # 圖層 2: 某個元素
    [600, 200, 900, 600],    # 圖層 3: 另一個元素
]
```

### 4. **width** (int)
- **類型**: `int`
- **說明**: 圖片寬度（像素）
- **範例**: `width = 1024`

### 5. **height** (int)
- **類型**: `int`
- **說明**: 圖片高度（像素）
- **範例**: `height = 1024`

## 🔄 從您的資料集格式轉換

### 您的資料集欄位 → CLD 需要的格式

| 您的資料集 | CLD 需要 | 轉換方式 |
|-----------|---------|---------|
| `preview` | `whole_img` | 直接使用 PIL Image |
| `title` | `caption` | 直接使用字串 |
| `canvas_width` | `width` | 直接使用整數 |
| `canvas_height` | `height` | 直接使用整數 |
| `left`, `top`, `width`, `height` | `layout` | 轉換為 `[left, top, left+width, top+height]` |
| `length` | `num_layers` | 圖層數量 |

### 轉換範例

```python
# 從您的資料集格式
item = {
    "preview": PIL_Image,
    "title": "My Design",
    "canvas_width": 1024,
    "canvas_height": 1024,
    "left": [0, 100, 600],
    "top": [0, 100, 200],
    "width": [1024, 500, 300],
    "height": [1024, 400, 200],
    "length": 3
}

# 轉換為 CLD 格式
whole_img = item["preview"]
caption = item["title"]
width = item["canvas_width"]
height = item["canvas_height"]

# 轉換 layout
layout = [[0, 0, width-1, height-1]]  # 第一個是整個畫布
for i in range(item["length"]):
    w0 = item["left"][i]
    h0 = item["top"][i]
    w1 = w0 + item["width"][i]
    h1 = h0 + item["height"][i]
    layout.append([w0, h0, w1, h1])
```

## 📝 完整輸入範例

```python
from PIL import Image

# 1. 載入圖片
whole_img = Image.open("my_image.png").convert("RGB")
width, height = whole_img.size

# 2. 準備描述
caption = "A modern design with geometric shapes and vibrant colors"

# 3. 定義圖層邊界框
# 格式: [w0, h0, w1, h1] 每個圖層一個
layout = [
    [0, 0, width-1, height-1],      # 圖層 0: 整個畫布（必須）
    [0, 0, width-1, height-1],      # 圖層 1: 背景（必須）
    [100, 100, 500, 400],           # 圖層 2: 前景元素 1
    [600, 200, 900, 600],           # 圖層 3: 前景元素 2
]

# 4. 準備給 pipeline 的參數
num_layers = len(layout)  # 圖層數量
```

## 🚀 Pipeline 呼叫格式

```python
x_hat, image, latents = pipeline(
    prompt=caption,                    # 文字描述
    adapter_image=whole_img,           # 完整圖片
    adapter_conditioning_scale=0.9,    # Adapter 強度
    validation_box=layout,            # 圖層邊界框列表
    generator=generator,               # 隨機生成器
    height=height,                     # 圖片高度
    width=width,                       # 圖片寬度
    guidance_scale=4.0,                # Guidance scale
    num_layers=num_layers,             # 圖層數量
    sdxl_vae=transp_vae,              # Transparent VAE
)
```

## ⚠️ 重要注意事項

1. **Layout 格式**: 
   - 必須是 `[w0, h0, w1, h1]` 格式（不是 `[x, y, w, h]`）
   - 第一個元素通常是整個畫布 `[0, 0, width-1, height-1]`
   - 第二個元素通常是背景（也是整個畫布）

2. **邊界框量化**: 
   - CLD 會自動將邊界框量化到 16 的倍數
   - 所以 `[100, 100, 500, 400]` 會被量化為 `[96, 96, 512, 416]`

3. **圖層順序**: 
   - 圖層順序很重要，從背景到前景
   - 第一個圖層通常是整個畫布
   - 第二個圖層通常是背景

4. **圖片格式**: 
   - 必須是 PIL Image 物件
   - 建議轉換為 RGB 模式
   - 尺寸會自動調整，但建議使用模型訓練時的尺寸

## 📦 資料集格式（用於批次處理）

如果要使用資料集批次處理，資料集需要返回以下格式的字典：

```python
{
    "whole_img": PIL.Image,           # 完整圖片
    "caption": str,                  # 文字描述
    "height": int,                    # 高度
    "width": int,                     # 寬度
    "layout": list[[w0, h0, w1, h1]], # 邊界框列表
    "pixel_RGBA": list[Tensor],      # 圖層 RGBA（訓練用）
    "pixel_RGB": list[Tensor],       # 圖層 RGB（訓練用）
}
```

這個格式已經在 `tools/custom_dataset.py` 中實現了。







