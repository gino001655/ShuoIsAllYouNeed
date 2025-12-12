import os
import random
import glob
from PIL import Image, ImageDraw

# Correspond to the output path of prepare_19k.py
DATASET_DIR = "datasets/dlcv_19k"

def sanity_check_v100(num_samples=5):
    # 1. Search for images
    search_pattern = os.path.join(DATASET_DIR, "images", "train", "*.jpg")
    image_files = glob.glob(search_pattern)
    
    if not image_files:
        print(f"❌ No images found in {DATASET_DIR}, please run prepare_19k.py first")
        return

    # 2. Random sampling
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    output_dir = "sanity_check_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"🔍 Checking {len(samples)} images...")

    for img_path in samples:
        # Derive Label path
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        
        if not os.path.exists(label_path):
            print(f"⚠️ Missing Label: {label_path}")
            continue

        # 3. Draw
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w_img, h_img = img.size
        
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        box_count = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5: continue
            
            # YOLO Format: class, cx, cy, w, h (Normalized)
            _, cx, cy, nw, nh = map(float, parts)
            
            # 轉回絕對座標
            w = nw * w_img
            h = nh * h_img
            x_center = cx * w_img
            y_center = cy * h_img
            
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
            
            # Draw green box (width 3)
            draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=3)
            box_count += 1
            
        # Save
        fname = os.path.basename(img_path)
        save_path = os.path.join(output_dir, f"check_{fname}")
        img.save(save_path)
        print(f"✅ Output: {save_path} (contains {box_count} boxes)")

    print(f"\n🎉 Check completed! Please check the '{output_dir}' folder.")
    print("Confirm that the boxes tightly enclose the text and are not shifted.")

if __name__ == "__main__":
    sanity_check_v100()