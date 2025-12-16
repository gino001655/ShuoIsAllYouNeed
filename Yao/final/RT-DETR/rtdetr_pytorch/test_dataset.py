import sys
sys.path.insert(0, '.')
from src.data.coco.coco_layer_dataset import CocoLayerOrderDataset
import torchvision.transforms.v2 as T
from pathlib import Path

# Create dataset
dataset = CocoLayerOrderDataset(
    img_folder="/tmp2/b12902041/Gino/preprocessed_data/images/train",
    ann_file="/tmp2/b12902041/Gino/preprocessed_data/annotations/train.json",
    depth_map_dir="/tmp2/b12902041/Gino/preprocessed_data/depths/train",
    return_masks=False,
    transforms=T.Compose([
        T.RandomHorizontalFlip(),
        T.Resize((640, 640)),
        T.ToImageTensor(),
        T.ConvertDtype(),
        T.SanitizeBoundingBox(min_size=1),
        T.ConvertPILToTensor() if hasattr(T, 'ConvertPILToTensor') else T.ToTensor()
    ])
)

# Load first few samples
for i in range(5):
    try:
        img, target = dataset[i]
        print(f"Sample {i}: img shape={img.shape}, num_boxes={len(target['boxes'])}, num_layers={len(target['layers'])}")
    except Exception as e:
        print(f"Sample {i}: ERROR - {e}")
        import traceback
        traceback.print_exc()
        break
