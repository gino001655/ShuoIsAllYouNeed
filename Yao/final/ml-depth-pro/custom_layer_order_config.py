"""
CUSTOM FILE: Configuration for Layer Order Prediction

配置文件，包含所有超參數和設置
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """訓練配置"""
    
    # 資料相關
    data_dir: str = "../dataset"  # 資料集目錄
    checkpoint_path: str = "./checkpoints/depth_pro.pt"  # 預訓練模型路徑
    
    # 訓練超參數
    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    
    # 損失相關
    use_edge_loss: bool = True
    edge_loss_weight: float = 0.1
    
    # 模型相關
    freeze_encoder: bool = False  # TODO: 思考是否應該凍結編碼器
    
    # 其他
    save_dir: str = "./checkpoints/layer_order"
    val_freq: int = 1  # 每 N 個 epoch 驗證一次
    use_amp: bool = False  # 混合精度訓練
    
    # TODO: 添加其他需要的配置項
    # 例如：資料增強、學習率調度器參數等


@dataclass
class ModelConfig:
    """模型配置"""
    
    patch_encoder_preset: str = "dinov2l16_384"
    image_encoder_preset: str = "dinov2l16_384"
    decoder_features: int = 256
    image_size: int = 1536  # Depth-Pro 的固定輸入尺寸
    
    # TODO: 理解這些參數
    # 1. patch_encoder 和 image_encoder 的區別？
    # 2. decoder_features 控制什麼？
    # 3. 為什麼 image_size 是 1536？


@dataclass
class DatasetConfig:
    """資料集配置"""
    
    image_size: int = 1536
    train_split_ratio: float = 0.9
    val_split_ratio: float = 0.05
    test_split_ratio: float = 0.05
    
    # TODO: 理解資料分割
    # 1. 為什麼要用 90/5/5 分割？
    # 2. 這個比例適合你的資料集大小嗎？


# 預設配置
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_DATASET_CONFIG = DatasetConfig()






