from ultralytics import YOLO

def train_yolo_fast():
    # Load pre-trained YOLOv8m weights (utilize COCO's edge features to accelerate convergence)
    model = YOLO("yolov8m.pt") 

    print("🚀 Start YOLOv8 fast training (using V7 repaired version data)...")
    
    results = model.train(
        # Ensure pointing to V7 version folder (please confirm your output_dir of the prepare script)
        data="datasets/dlcv_19k/data.yaml",
        
        # === Speed-first settings ===
        epochs=50,          # Run 50 epochs first, stop if good enough, no need to run to 300
        patience=10,        # Stop if no improvement for 10 epochs
        batch=8,           # V100 32GB can be opened to 64 or even 128 (fill up GPU)
        imgsz=640,
        device=0,
        workers=16,         # Open bigger, ensure GPU doesn't wait for disk
        
        # === Project settings ===
        project="dlcv_v100_project",
        name="yolo_v7_fast_check",
        exist_ok=True,
        
        # === Adjustments for Layout enhancement ===
        mosaic=0.0,         # Close mosaic (to avoid destroying the layout)
        mixup=0.0,          # Close Mixup
        degrees=0.0,        # Close rotation enhancement (because our labels are already AABB)
        scale=0.5,          # Keep a little bit of scale enhancement
        
        # === System optimization ===
        cache=True,         # [Key] Cache 19K images to RAM, this can make speed double!
    )

if __name__ == '__main__':
    train_yolo_fast()