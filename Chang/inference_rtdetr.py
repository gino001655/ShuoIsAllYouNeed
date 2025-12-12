import os
from ultralytics import RTDETR 
from PIL import Image, ImageDraw

# === Settings ===
# Please confirm that the path here points to the best.pt trained by RT-DETR
# Assuming your project name is dlcv_v100_project, name is rtdetr_19k_no_mosaic
MODEL_PATH = "dlcv_v100_project/rtdetr_19k_no_mosaic/weights/best.pt"

# Input folder
INPUT_SOURCE = "inputs" 
OUTPUT_DIR = "inference_results_rtdetr" # <--- Suggest to change the name and separate from YOLO's results

def run_inference():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("💡 Tip: Please confirm that train_rtdetr.py has finished running, or the path is correct.")
        return

    # If the inputs folder doesn't exist, automatically create it and prompt the user
    if not os.path.exists(INPUT_SOURCE):
        os.makedirs(INPUT_SOURCE, exist_ok=True)
        print(f"⚠️ '{INPUT_SOURCE}' 資料夾是空的。")
        # Here points to the latest V7 dataset validation set
        print(f"➡️  Switching to using Dataset validation images for testing...")
        source = "datasets/dlcv_19k_v7_geo/images/val" # <--- Confirm this is your latest dataset path
    else:
        source = INPUT_SOURCE

    print(f"🚀 Load RT-DETR model: {MODEL_PATH}")
    print(f"📂 Read image source: {source}")
    
    # <--- Key modification 2: Instantiate RTDETR
    model = RTDETR(MODEL_PATH)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run inference
    # RT-DETR doesn't need NMS, but still has conf (confidence threshold)
    results = model.predict(source=source, conf=0.25, save=False)

    print(f"🎨 Start drawing prediction results (Clean Style)...")
    
    count = 0
    for result in results:
        # Convert to PIL
        img = Image.fromarray(result.orig_img[..., ::-1])
        draw = ImageDraw.Draw(img)
        
        has_box = False
        for box in result.boxes:
            # RT-DETR's output format is exactly the same as YOLO (xyxy)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Draw box (here we use red box, to distinguish from YOLO's green box)
            draw.rectangle([x1, y1, x2, y2], outline="#FF0000", width=4)
            has_box = True
            
        # Save
        fname = os.path.basename(result.path)
        save_path = os.path.join(OUTPUT_DIR, fname)
        img.save(save_path)
        count += 1
        
        # Limit the number of outputs
        if count >= 50: break

    print(f"\n✅ RT-DETR inference completed! Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()