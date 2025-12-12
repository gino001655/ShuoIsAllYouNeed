from ultralytics import RTDETR
import torch

def train_rtdetr_final():
    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠️ Warning: No GPU found, RT-DETR will run forever on CPU!")
    
    # Load pre-trained RT-DETR model
    # rtdetr-l.pt (Large) is the sweet spot for accuracy and speed
    model = RTDETR("rtdetr-l.pt") 

    print("🚀 Start RT-DETR training (Layout Analysis specific settings)...")
    
    results = model.train(
        data="datasets/dlcv_19k/data.yaml",
        
        # === Training strategy ===
        epochs=100,         # 100 epochs is enough for Fine-tuning
        patience=15,        # Slightly stricter, stop if no improvement for 15 epochs (save compute)
        batch=16,           # [Important] If V100 OOM, please降到 8
        imgsz=640,
        device=0,
        workers=8,          # Accelerate data loading
        
        # === Project settings ===
        project="dlcv_v100_project",
        name="rtdetr_19k_no_mosaic",
        exist_ok=True,
        
        # === Optimizer and hyperparameters ===
        optimizer='AdamW',  # Transformer 
        lr0=0.0001,         # Fine-tune smaller learning rate
        
        # === Key: Layout enhancement strategy ===
        mosaic=0.0,         # ❌ Close mosaic (to avoid destroying the layout logic)
        mixup=0.0,          # ❌ Close Mixup (to avoid transparency confusion)
        degrees=0.0,        # ❌ No rotation (layout is usually upright)
        
        # ✅ Keep safe enhancements
        scale=0.5,          # Random scaling (0.5 means +/- 50%) -> Make model adapt to different canvas sizes
        fliplr=0.5,         # Left-right flip (Layout is usually symmetric left-right, unless very concerned about text direction)
        hsv_h=0.015,        # Mild color tone variation
        hsv_s=0.7,          # Saturation variation (Layout color variation is large, can be kept)
        hsv_v=0.4,          # Brightness variation
        
        # === System optimization ===
        cache=True,         # Cache images to RAM, accelerate training of 19K images
        amp=True,           # Automatic mixed precision (save memory, must open)
    )
    
    print(f"🎉 Training completed! Model saved at: {results.save_dir}")

if __name__ == '__main__':
    train_rtdetr_final()