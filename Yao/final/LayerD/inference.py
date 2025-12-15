from PIL import Image
from layerd import LayerD
import torch
import os

# 輸入圖片路徑
input_image_path = "../extracted_images/sample_0001/preview.png"
image = Image.open(input_image_path)

# 使用 CPU 或 cuda（注意：PyTorch 需要小寫的 "cuda"）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用設備: {device}")

# 初始化 LayerD 模型
layerd = LayerD(matting_hf_card="cyberagent/layerd-birefnet").to(device) 

# 執行分解
print("開始分解圖像...")
layers = layerd.decompose(image)
print(f"分解完成！共生成 {len(layers)} 個圖層")

# 創建輸出目錄
output_dir = "./output2"
os.makedirs(output_dir, exist_ok=True)

# 保存每個圖層
# layers 的順序：從背景到前景（最後一個是最前面的圖層）
for i, layer in enumerate(layers):
    # 圖層編號：0 是背景，1, 2, 3... 是前景（從後到前）
    output_path = os.path.join(output_dir, f"layer_{i:03d}.png")
    layer.save(output_path)
    print(f"已保存圖層 {i}: {output_path}")

print(f"\n所有圖層已保存到: {os.path.abspath(output_dir)}")
# output layers 是一個 RGBA 格式的 PIL Image 物件列表