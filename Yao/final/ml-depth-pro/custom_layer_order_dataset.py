"""
CUSTOM FILE: Layer Order Dataset
從 PrismLayersPro 資料集生成圖層索引圖作為 Ground Truth

主要功能：
1. 載入 PrismLayersPro 資料集
2. 從每層的圖片和位置資訊生成圖層索引圖
3. 將圖層索引歸一化到 [0, 1] 區間
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets
import glob
import os
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class LayerOrderDataset(Dataset):
    """
    圖層順序預測資料集
    
    支援兩種資料格式：
    1. PrismLayersPro Parquet 格式（原始格式）
    2. Parsed dataset 格式（已解析為 PNG + JSON）
    
    從資料集生成：
    - 輸入：RGB 合成圖像
    - 輸出：圖層索引圖（每個像素值 = 該像素所在圖層的歸一化索引）
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 1536,  # Depth-Pro 的固定輸入尺寸
        use_parsed_format: bool = True,  # 是否使用已解析的格式
        use_augmentation: bool = False,  # 是否使用資料增強（僅在 train split 時有效）
        augmentation_config: Optional[Dict] = None,  # 資料增強配置
    ):
        """
        Args:
        ----
            data_dir: 資料集目錄路徑
                - 如果 use_parsed_format=True: 指向 parsed_dataset 目錄
                - 如果 use_parsed_format=False: 指向包含 snapshots 的目錄
            split: "train", "val", 或 "test"
            image_size: 模型輸入尺寸（Depth-Pro 使用 1536x1536）
            use_parsed_format: 是否使用已解析的格式（PNG + JSON）
        """
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.use_parsed_format = use_parsed_format
        self.use_augmentation = use_augmentation and (split == "train")  # 只在訓練時使用增強
        
        # 資料增強配置
        if augmentation_config is None:
            augmentation_config = {
                "horizontal_flip_prob": 0.5,
                "vertical_flip_prob": 0.5,
                "rotation_prob": 0.5,
                "rotation_degrees": 90,  # 90, 180, 270 度旋轉
                "color_jitter_prob": 0.5,
                "color_jitter_brightness": 0.2,
                "color_jitter_contrast": 0.2,
                "color_jitter_saturation": 0.2,
            }
        self.augmentation_config = augmentation_config
        
        # 基本轉換（不包含資料增強）- 需要在載入資料集之前設置
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        # 完整的 transform（包含資料增強，如果啟用）
        self.transform = self.base_transform
        
        # Detect COCO format (processed_dlcv_dataset)
        coco_train_json = os.path.join(self.data_dir, "annotations", "train.json")
        self.use_coco_format = os.path.exists(coco_train_json)

        if self.use_coco_format:
            print(f"[Dataset] Detected COCO format at {self.data_dir}")
            self._load_coco_dataset()
        elif use_parsed_format:
            # TODO: 理解已解析格式的載入
            # 1. 如何從 parsed_dataset 目錄載入資料？
            # 2. 如何組織 train/val/test 分割？
            # 提示：parsed_dataset 目錄結構是 category/sample_id/
            self._load_parsed_dataset()
        else:
            # TODO: 理解 Parquet 格式的載入
            # 1. 為什麼要載入所有 Parquet 檔案？
            # 2. snapshots 目錄的作用是什麼？
            self._load_parquet_dataset()
    
    def _load_parsed_dataset(self):
        """載入已解析的資料集格式（PNG + JSON）"""
        import json
        from pathlib import Path
        
        data_path = Path(self.data_dir)
        if not data_path.exists():
            raise ValueError(f"資料集目錄不存在: {data_path}")
        
        # 收集所有樣本
        all_samples = []
        for category_dir in data_path.iterdir():
            if not category_dir.is_dir() or category_dir.name == "__pycache__":
                continue
            
            for sample_dir in category_dir.iterdir():
                if not sample_dir.is_dir():
                    continue
                
                metadata_path = sample_dir / "metadata.json"
                whole_image_path = sample_dir / "00_whole_image.png"
                
                if metadata_path.exists() and whole_image_path.exists():
                    all_samples.append({
                        'sample_dir': sample_dir,
                        'category': category_dir.name,
                        'sample_id': sample_dir.name,
                    })
        
        print(f"[Dataset] 找到 {len(all_samples)} 個樣本")
        
        # 按分類分組樣本
        category_to_samples = defaultdict(list)
        for sample in all_samples:
            category_to_samples[sample['category']].append(sample)
        
        print(f"[Dataset] 找到 {len(category_to_samples)} 個分類")
        
        # 對每個分類分別進行分割，確保每個分類都有 train/val/test
        selected_samples = []
        train_ratio = 0.9
        val_ratio = 0.05
        test_ratio = 0.05
        
        for category, samples in category_to_samples.items():
            # 打亂順序（使用固定種子以確保可重現性）
            random.Random(42).shuffle(samples)
            
            total_len = len(samples)
            if total_len == 0:
                continue
            
            idx_train = int(total_len * train_ratio)
            idx_val = int(total_len * (train_ratio + val_ratio))
            
            if self.split == "train":
                selected_samples.extend(samples[:idx_train])
            elif self.split == "val":
                selected_samples.extend(samples[idx_train:idx_val])
            elif self.split == "test":
                selected_samples.extend(samples[idx_val:])
            else:
                raise ValueError("split 必須是 'train', 'val', 或 'test'")
        
        # 打亂最終的樣本順序
        random.Random(42).shuffle(selected_samples)
        
        self.samples = selected_samples
        print(f"[Dataset] {self.split} split: {len(self.samples)} 筆資料")
        
        # 統計每個分類的樣本數
        category_counts = defaultdict(int)
        for sample in selected_samples:
            category_counts[sample['category']] += 1
        print(f"[Dataset] 各分類樣本數: {dict(category_counts)}")
    
    def _load_parquet_dataset(self):
        """載入 Parquet 格式的資料集（原始格式）"""
        parquet_files = glob.glob(os.path.join(self.data_dir, "snapshots/*/data/train-*.parquet"))
        if not parquet_files:
            parquet_files = glob.glob(os.path.join(self.data_dir, "**/train-*.parquet"), recursive=True)
        
        if not parquet_files:
            raise ValueError(f"無法在 {self.data_dir} 找到 Parquet 檔案")
        
        print(f"[Dataset] 找到 {len(parquet_files)} 個 Parquet 檔案")
        
        # 載入所有 Parquet 檔案
        datasets_list = []
        for pf in sorted(parquet_files):
            print(f"[Dataset] 載入: {pf}")
            ds = load_dataset('parquet', data_files=pf)
            datasets_list.append(ds['train'])
        
        # 合併所有資料集
        full_dataset = concatenate_datasets(datasets_list)
        print(f"[Dataset] 總共載入 {len(full_dataset)} 筆資料")
        
        # 資料分割
        total_len = len(full_dataset)
        idx_90 = int(total_len * 0.9)
        idx_95 = int(total_len * 0.95)
        
        if self.split == "train":
            selected_idx = list(range(0, idx_90))
        elif self.split == "val":
            selected_idx = list(range(idx_90, idx_95))
        elif self.split == "test":
            selected_idx = list(range(idx_95, total_len))
        else:
            raise ValueError("split 必須是 'train', 'val', 或 'test'")
        
        self.dataset = full_dataset.select(selected_idx)
        print(f"[Dataset] {self.split} split: {len(self.dataset)} 筆資料")
        
        # 基本轉換（不包含資料增強）
        self.base_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        # 完整的 transform（包含資料增強，如果啟用）
        self.transform = self.base_transform
    
    def __len__(self) -> int:
        if hasattr(self, 'use_coco_format') and self.use_coco_format:
            return len(self.samples)
        elif self.use_parsed_format:
            return len(self.samples)
        else:
            return len(self.dataset)
    
    def create_layer_index_map(
        self,
        layer_images: List[Image.Image],
        left_list: List[float],
        top_list: List[float],
        width_list: List[float],
        height_list: List[float],
        canvas_width: int,
        canvas_height: int,
    ) -> np.ndarray:
        """
        創建圖層索引圖
        
        Args:
        ----
            layer_images: 各層的圖片列表
            left_list, top_list, width_list, height_list: 各層的位置資訊
            canvas_width, canvas_height: 畫布尺寸
        
        Returns:
        -------
            layer_index_map: [H, W] numpy array，值在 [0, 1]
        """
        # TODO: 理解圖層索引圖的生成邏輯
        # 1. 為什麼要初始化為 0？
        # 2. 0 代表什麼（背景還是最前景）？
        layer_index_map = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        
        num_layers = len(layer_images)
        if num_layers == 0:
            return layer_index_map
        
        # TODO: 理解圖層疊加順序
        # 1. 應該從背景到前景疊加，還是從前景到背景？
        # 2. 如果圖層有重疊，應該使用哪個圖層的索引？
        # 提示：通常背景層索引較小，前景層索引較大
        # 建議：從背景到前景疊加，後面的圖層會覆蓋前面的
        
        for i, (layer_img, left, top, width, height) in enumerate(zip(
            layer_images, left_list, top_list, width_list, height_list
        )):
            # 歸一化公式：確保第一個圖層（i=0）有非零值，避免與背景（0）融合
            # 背景保持為 0（初始化值）
            # 第一個圖層（i=0）：(0+1)/num_layers = 1/num_layers（非零）
            # 最後一個圖層（i=num_layers-1）：num_layers/num_layers = 1
            # 這樣可以確保每個圖層都有不同的索引值
            if num_layers > 0:
                normalized_index = (i + 1) / num_layers  # 範圍：[1/num_layers, 1]
            else:
                normalized_index = 0.0
            
            # 轉換為整數座標
            x0 = int(left)
            y0 = int(top)
            x1 = int(left + width)
            y1 = int(top + height)
            
            # 確保在畫布範圍內
            x0 = max(0, min(x0, canvas_width))
            y0 = max(0, min(y0, canvas_height))
            x1 = max(0, min(x1, canvas_width))
            y1 = max(0, min(y1, canvas_height))
            
            if x1 <= x0 or y1 <= y0:
                continue
            
            # TODO: 理解圖層遮罩的生成
            # 1. 如何判斷圖層的哪些像素是有效的（非透明）？
            # 2. 如果圖層是 RGBA，alpha 通道的作用是什麼？
            # 提示：可以使用 alpha 通道作為遮罩
            
            # 轉換圖層為 numpy array
            if isinstance(layer_img, Image.Image):
                layer_array = np.array(layer_img)
            else:
                layer_array = layer_img
            
            # 處理 RGBA 圖像
            if layer_array.shape[2] == 4:
                # 使用 alpha 通道作為遮罩
                alpha = layer_array[:, :, 3] / 255.0  # 歸一化到 [0, 1]
                mask = alpha > 0.1  # 閾值可調整
            else:
                # RGB 圖像，假設所有像素都有效
                mask = np.ones((layer_array.shape[0], layer_array.shape[1]), dtype=bool)
            
            # 調整遮罩尺寸（如果需要）
            if mask.shape[0] != (y1 - y0) or mask.shape[1] != (x1 - x0):
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((x1 - x0, y1 - y0), PILImage.NEAREST)
                mask = np.array(mask_pil) > 128
            
            # TODO: 理解圖層疊加邏輯
            # 1. 為什麼要用 mask 來選擇更新的區域？
            # 2. 如果多個圖層重疊，最後的圖層會覆蓋前面的，這對嗎？
            # 提示：這符合「後面的圖層在前面的圖層之上」的邏輯
            layer_index_map[y0:y1, x0:x1][mask] = normalized_index
        
        return layer_index_map
    
    def apply_augmentation(
        self,
        image: Image.Image,
        layer_index_map: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        對圖像和 ground truth 同時應用資料增強
        
        Args:
        ----
            image: PIL Image
            layer_index_map: numpy array [H, W]
        
        Returns:
        -------
            augmented_image: PIL Image
            augmented_layer_index_map: numpy array [H, W]
        """
        if not self.use_augmentation:
            return image, layer_index_map
        
        aug_image = image.copy()
        aug_map = layer_index_map.copy()
        
        # 水平翻轉
        if random.random() < self.augmentation_config["horizontal_flip_prob"]:
            aug_image = aug_image.transpose(Image.FLIP_LEFT_RIGHT)
            aug_map = np.fliplr(aug_map)
        
        # 垂直翻轉
        if random.random() < self.augmentation_config["vertical_flip_prob"]:
            aug_image = aug_image.transpose(Image.FLIP_TOP_BOTTOM)
            aug_map = np.flipud(aug_map)
        
        # 旋轉（90, 180, 270 度）
        if random.random() < self.augmentation_config["rotation_prob"]:
            rotation_degrees = self.augmentation_config["rotation_degrees"]
            if rotation_degrees == 90:
                # 如果設定為 90，隨機選擇 90, 180, 270
                angle = random.choice([90, 180, 270])
            else:
                angle = rotation_degrees
            
            # PIL Image 旋轉：負角度表示逆時針
            aug_image = aug_image.rotate(-angle, expand=False, fillcolor=(0, 0, 0))
            # 對 Ground Truth 也使用 PIL Image 的 rotate，確保旋轉方向一致
            # 先將 numpy array 轉換為 PIL Image，旋轉後再轉回 numpy array
            aug_map_pil = Image.fromarray((aug_map * 255).astype(np.uint8), mode='L')
            aug_map_pil = aug_map_pil.rotate(-angle, expand=False, fillcolor=0)
            aug_map = np.array(aug_map_pil).astype(np.float32) / 255.0
        
        # 顏色抖動（只應用到圖像，不影響 ground truth）
        if random.random() < self.augmentation_config["color_jitter_prob"]:
            color_jitter = transforms.ColorJitter(
                brightness=self.augmentation_config["color_jitter_brightness"],
                contrast=self.augmentation_config["color_jitter_contrast"],
                saturation=self.augmentation_config["color_jitter_saturation"],
            )
            aug_image = color_jitter(aug_image)
        
        return aug_image, aug_map
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        取得一個樣本
        
        Returns:
        -------
            {
                "image": Tensor [3, H, W],  # RGB 圖像
                "layer_index_map": Tensor [H, W],  # 圖層索引圖，值在 [0, 1]
                "num_layers": int,  # 圖層總數
                "original_size": (H, W),  # 原始尺寸
            }
        """
        if hasattr(self, 'use_coco_format') and self.use_coco_format:
            return self._getitem_coco(idx)
        elif self.use_parsed_format:
            return self._getitem_parsed(idx)
        else:
            return self._getitem_parquet(idx)

    def _load_coco_dataset(self):
        """載入 COCO 格式的資料集 (processed_dlcv_dataset)"""
        import json
        
        # Determine split
        # We try to load 'train.json' or 'val.json' based on self.split
        # Note: 'test' split might strictly be val.json or a specific test.json
        # User script logic usually maps test->val or similar if test doesn't exist.
        target_split = self.split
        if target_split == "test":
             target_split = "val" # Fallback or assume test.json exists if user generated it
             
        json_path = os.path.join(self.data_dir, "annotations", f"{target_split}.json")
        if not os.path.exists(json_path):
             # Fallback: if test missing, try val
             if self.split == "test":
                 json_path = os.path.join(self.data_dir, "annotations", "val.json")
        
        if not os.path.exists(json_path):
            raise ValueError(f"Annotation file not found: {json_path}")
            
        print(f"[Dataset] Loading annotations from {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        self.samples = data['images'] # List of {id, file_name, width, height}
        print(f"[Dataset] {self.split} split: {len(self.samples)} samples loaded")

    def _getitem_coco(self, idx: int) -> Dict[str, torch.Tensor]:
        """從 COCO 格式讀取樣本 (RGB + Generaetd Depth)"""
        sample = self.samples[idx]
        file_name = sample['file_name']
        img_id = sample['id'] # Integer ID
        
        # Paths
        # Structure: images/<split>/file_name
        # But 'file_name' in our generated JSON is "00001.png" OR relative?
        # preprocess_dlcv_dataset.py saved: f"{img_id:08d}.png" as file_name.
        # Images saved in: output_dir / "images" / split / img_filename
        # So we need to join path.
        
        # Determine split subdir name
        # We loaded json from `annotations/{split}.json`. 
        # The images are in `images/{split}/`.
        # However, self.split might be 'test' but we loaded 'val'.
        # We should infer subdir from where we loaded JSON? 
        # Simpler: just try `images/train` or `images/val` or `images/test`.
        
        # Robust logic: We know `sample` came from a specific JSON.
        # If we loaded `val.json`, images are likely in `images/val`.
        # Let's derive image_dir from the json filename we loaded?
        # Ideally _load_coco_dataset stores `image_dir`.
        
        # Re-derive split name used
        target_split = self.split
        if target_split == "test" and not os.path.exists(os.path.join(self.data_dir, "annotations", "test.json")):
             target_split = "val"
        
        image_path = os.path.join(self.data_dir, "images", target_split, file_name)
        depth_path = os.path.join(self.data_dir, "depths", target_split, file_name)
        
        # Load Images
        composite_image = Image.open(image_path).convert("RGB")
        original_size = composite_image.size
        
        if os.path.exists(depth_path):
            # Load Pre-generated Depth
            depth_img = Image.open(depth_path).convert("L")
            if depth_img.size != original_size:
                depth_img = depth_img.resize(original_size, Image.NEAREST)
            layer_index_map = np.array(depth_img).astype(np.float32) / 255.0
        else:
            # Fallback (Should not happen if processed correctly)
            layer_index_map = np.zeros((original_size[1], original_size[0]), dtype=np.float32)
            
        # Apply Augmentation
        composite_image, layer_index_map = self.apply_augmentation(
            composite_image, layer_index_map
        )
        
        # Transform to Tensor
        composite_tensor = self.base_transform(composite_image)
        
        # Resize Depth to Model Input Size
        layer_index_pil = Image.fromarray((layer_index_map * 255).astype(np.uint8), mode='L')
        layer_index_resized = layer_index_pil.resize(
            (self.image_size, self.image_size),
            Image.NEAREST
        )
        layer_index_tensor = torch.from_numpy(
            np.array(layer_index_resized).astype(np.float32) / 255.0
        )
        
        # Num Layers?
        # We don't have layer count in 'sample' (image dict). 
        # We could add it in preprocess script or just default to 0. 
        # It's only used for some specific loss weights or debug?
        # Default to 0.
        num_layers = 0
        
        return {
            "image": composite_tensor,
            "layer_index_map": layer_index_tensor,
            "num_layers": num_layers,
            "original_size": original_size,
        }
    
    def _getitem_parsed(self, idx: int) -> Dict[str, torch.Tensor]:
        """從已解析的格式讀取樣本"""
        import json
        from pathlib import Path
        
        sample_info = self.samples[idx]
        sample_dir = sample_info['sample_dir']
        
        # 讀取 metadata
        metadata_path = sample_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 讀取完整圖片
        whole_image_path = sample_dir / "00_whole_image.png"
        composite_image = Image.open(whole_image_path).convert("RGB")
        original_size = composite_image.size  # (W, H)
        canvas_width, canvas_height = original_size
        
        # 讀取各層圖片和資訊
        layer_images = []
        left_list = []
        top_list = []
        width_list = []
        height_list = []
        
        layers_info = metadata.get('layers', [])
        num_layers = len(layers_info)
        
        for layer_info in layers_info:
            layer_img_path = sample_dir / layer_info['image_path']
            if layer_img_path.exists():
                layer_img = Image.open(layer_img_path)
                layer_images.append(layer_img)
                
                # 從 box 提取位置資訊
                box = layer_info.get('box', [0, 0, canvas_width, canvas_height])
                x0, y0, x1, y1 = box
                left_list.append(x0)
                top_list.append(y0)
                width_list.append(x1 - x0)
                height_list.append(y1 - y0)
        
        # 創建圖層索引圖
        if len(layer_images) == 0:
            layer_index_map = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        else:
            layer_index_map = self.create_layer_index_map(
                layer_images,
                left_list,
                top_list,
                width_list,
                height_list,
                canvas_width,
                canvas_height,
            )
        
        # 應用資料增強（如果啟用）
        composite_image, layer_index_map = self.apply_augmentation(
            composite_image, layer_index_map
        )
        
        # 轉換為 tensor
        composite_tensor = self.base_transform(composite_image)
        
        # 將圖層索引圖也 resize 到模型輸入尺寸
        layer_index_pil = Image.fromarray((layer_index_map * 255).astype(np.uint8), mode='L')
        layer_index_resized = layer_index_pil.resize(
            (self.image_size, self.image_size),
            Image.NEAREST
        )
        layer_index_tensor = torch.from_numpy(
            np.array(layer_index_resized).astype(np.float32) / 255.0
        )
        
        return {
            "image": composite_tensor,
            "layer_index_map": layer_index_tensor,
            "num_layers": num_layers,
            "original_size": original_size,
        }
    
    def _getitem_parquet(self, idx: int) -> Dict[str, torch.Tensor]:
        """從 Parquet 格式讀取樣本"""
        item = self.dataset[idx]
        
        # TODO: 理解資料格式
        # 1. item["preview"] 是什麼？
        # 2. item["image"] 是什麼類型的資料？
        # 3. item["length"] 代表什麼？
        
        # 取得完整圖片（preview）
        preview = item["preview"]
        if isinstance(preview, Image.Image):
            composite_image = preview.convert("RGB")
        else:
            composite_image = Image.fromarray(preview).convert("RGB")
        
        original_size = composite_image.size  # (W, H)
        canvas_width, canvas_height = original_size
        
        # TODO: 理解圖層資料的提取
        # 1. item["image"] 是什麼格式？
        # 2. 如何取得每個圖層的圖片和位置資訊？
        layer_images = item.get("image", [])
        left_list = item.get("left", [])
        top_list = item.get("top", [])
        width_list = item.get("width", [])
        height_list = item.get("height", [])
        num_layers = item.get("length", len(layer_images))
        
        # TODO: 處理邊界情況
        # 1. 如果沒有圖層資料怎麼辦？
        # 2. 如果圖層數量不一致怎麼辦？
        if len(layer_images) == 0:
            # 沒有圖層，創建全零的索引圖
            layer_index_map = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        else:
            # 確保所有列表長度一致
            min_len = min(
                len(layer_images),
                len(left_list),
                len(top_list),
                len(width_list),
                len(height_list),
                num_layers
            )
            
            # 創建圖層索引圖
            layer_index_map = self.create_layer_index_map(
                layer_images[:min_len],
                left_list[:min_len],
                top_list[:min_len],
                width_list[:min_len],
                height_list[:min_len],
                canvas_width,
                canvas_height,
            )
        
        # 應用資料增強（如果啟用）
        composite_image, layer_index_map = self.apply_augmentation(
            composite_image, layer_index_map
        )
        
        # 轉換為 tensor
        composite_tensor = self.base_transform(composite_image)
        
        # 將圖層索引圖也 resize 到模型輸入尺寸
        layer_index_pil = Image.fromarray((layer_index_map * 255).astype(np.uint8), mode='L')
        layer_index_resized = layer_index_pil.resize(
            (self.image_size, self.image_size),
            Image.NEAREST  # 使用最近鄰，保持離散值
        )
        layer_index_tensor = torch.from_numpy(
            np.array(layer_index_resized).astype(np.float32) / 255.0
        )  # [H, W]
        
        return {
            "image": composite_tensor,  # [3, image_size, image_size]
            "layer_index_map": layer_index_tensor,  # [image_size, image_size]
            "num_layers": num_layers,
            "original_size": original_size,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    批次整理函數
    
    TODO: 理解 collate_fn 的作用
    1. 為什麼需要這個函數？
    2. 如果批次中圖像尺寸不同怎麼辦？
    3. 在這個任務中，所有圖像都是相同尺寸（1536x1536），所以可以直接 stack
    """
    images = torch.stack([item["image"] for item in batch])
    layer_maps = torch.stack([item["layer_index_map"] for item in batch])
    num_layers = [item["num_layers"] for item in batch]
    original_sizes = [item["original_size"] for item in batch]
    
    return {
        "image": images,  # [B, 3, H, W]
        "layer_index_map": layer_maps,  # [B, H, W]
        "num_layers": num_layers,
        "original_sizes": original_sizes,
    }



