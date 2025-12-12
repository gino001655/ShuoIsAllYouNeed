import os
from ultralytics import YOLO
from PIL import Image, ImageDraw

# === Settings ===
# Correspond to the project/name set in train_scratch.py
MODEL_PATH = "dlcv_v100_project/yolo_v7_fast_check/weights/best.pt"

# Input folder (you can put a few posters from the internet to test)
# If this folder doesn't exist, I will default to using the validation set images for testing
INPUT_SOURCE = "inputs" 
OUTPUT_DIR = "inference_results_yolo"

def run_inference():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("💡 Tip: Please confirm that train_scratch.py has finished running, or the path is correct.")
        return

    # If the inputs folder doesn't exist, automatically create it and prompt the user
    if not os.path.exists(INPUT_SOURCE):
        os.makedirs(INPUT_SOURCE, exist_ok=True)
        print(f"⚠️ '{INPUT_SOURCE}' folder is empty.")
        print(f"➡️  Switching to using Dataset validation images for testing...")
        source = "datasets/dlcv_19k/images/val" # Backup path
    else:
        source = INPUT_SOURCE

    print(f"🚀 Load model: {MODEL_PATH}")
    print(f"📂 Read image source: {source}")
    
    model = YOLO(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run inference (save=False because we want to draw clean boxes ourselves)
    results = model.predict(source=source, conf=0.25, save=False)

    print(f"🎨 Start drawing prediction results (Clean Style)...")
    
    count = 0
    for result in results:
        # Convert to PIL
        img = Image.fromarray(result.orig_img[..., ::-1])
        draw = ImageDraw.Draw(img)
        
        has_box = False
        for box in result.boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Draw box (green, no text label)
            draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=4)
            has_box = True
            
        # Save
        fname = os.path.basename(result.path)
        save_path = os.path.join(OUTPUT_DIR, fname)
        img.save(save_path)
        count += 1
        
        # To avoid the test set running too long, here we limit only to output the first 50 images
        if count >= 50: break

    print(f"\n✅ Inference completed! Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()