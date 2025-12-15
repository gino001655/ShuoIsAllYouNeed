"""
CUSTOM FILE: Layer Order Prediction Model
基於 Depth-Pro 修改，用於預測圖層順序而非物理深度

主要修改：
1. 移除 FOV (Field of View) 預測模組
2. 輸出改為單通道圖層索引圖（歸一化到 [0, 1]）
3. 簡化 infer 函數，不需要焦距資訊
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch
from torch import nn
from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Lambda,
    Normalize,
    ToTensor,
)

from src.depth_pro.network.decoder import MultiresConvDecoder
from src.depth_pro.network.encoder import DepthProEncoder
from src.depth_pro.network.vit_factory import VIT_CONFIG_DICT, ViTPreset, create_vit


@dataclass
class LayerOrderConfig:
    """Configuration for Layer Order Prediction Model."""
    
    patch_encoder_preset: ViTPreset
    image_encoder_preset: ViTPreset
    decoder_features: int
    # Enable gradient checkpointing inside the ViT blocks (saves VRAM, slower).
    use_grad_checkpointing: bool = False
    
    checkpoint_uri: Optional[str] = None


DEFAULT_LAYER_ORDER_CONFIG = LayerOrderConfig(
    patch_encoder_preset="dinov2l16_384",
    image_encoder_preset="dinov2l16_384",
    checkpoint_uri="./checkpoints/depth_pro.pt",
    decoder_features=256,
    use_grad_checkpointing=False,
)


def create_backbone_model(
    preset: ViTPreset,
    use_grad_checkpointing: bool = False,
) -> Tuple[nn.Module, ViTPreset]:
    """Create and load a backbone model given a config.
    
    TODO: 理解這個函數的作用
    1. 它創建了什麼類型的模型？
    2. use_pretrained=False 意味著什麼？
    3. 為什麼需要返回 config？
    """
    if preset in VIT_CONFIG_DICT:
        config = VIT_CONFIG_DICT[preset]
        model = create_vit(
            preset=preset,
            use_pretrained=False,
            use_grad_checkpointing=use_grad_checkpointing,
        )
    else:
        raise KeyError(f"Preset {preset} not found.")
    
    return model, config


def create_layer_order_model_and_transforms(
    config: LayerOrderConfig = DEFAULT_LAYER_ORDER_CONFIG,
    device: torch.device = torch.device("cpu"),
    precision: torch.dtype = torch.float32,
) -> Tuple['LayerOrderModel', Compose]:
    """Create a Layer Order Prediction model based on Depth-Pro.
    
    Args:
    ----
        config: The configuration for the model architecture.
        device: The device to load the model onto.
        precision: The precision used for the model.
    
    Returns:
    -------
        The LayerOrderModel and associated Transform.
    """
    patch_encoder, patch_encoder_config = create_backbone_model(
        preset=config.patch_encoder_preset,
        use_grad_checkpointing=config.use_grad_checkpointing,
    )
    image_encoder, _ = create_backbone_model(
        preset=config.image_encoder_preset,
        use_grad_checkpointing=config.use_grad_checkpointing,
    )
    

    
    dims_encoder = patch_encoder_config.encoder_feature_dims
    hook_block_ids = patch_encoder_config.encoder_feature_layer_ids
    
    # TODO: 理解 DepthProEncoder 的作用
    # 2. hook_block_ids 的作用是什麼？
    # 3. decoder_features 控制什麼？
    encoder = DepthProEncoder(
        dims_encoder=dims_encoder,
        patch_encoder=patch_encoder,
        image_encoder=image_encoder,
        hook_block_ids=hook_block_ids,
        decoder_features=config.decoder_features,
    )
    
    # TODO: 理解解碼器的結構
    # 1. dims_encoder 列表的順序有什麼意義？
    # 2. dim_decoder 控制什麼？
    decoder = MultiresConvDecoder(
        dims_encoder=[config.decoder_features] + list(encoder.dims_encoder),
        dim_decoder=config.decoder_features,
    )
    
    # TODO: 理解模型初始化
    # 1. last_dims=(32, 1) 的意義是什麼？
    # 2. 為什麼 use_fov_head=False？
    model = LayerOrderModel(
        encoder=encoder,
        decoder=decoder,
        last_dims=(32, 1),  # 中間層 32 維，最後輸出 1 通道
        use_fov_head=False,  # 關鍵：移除 FOV
    ).to(device)
    
    if precision == torch.half:
        model.half()
    
    transform = Compose(
        [
            ToTensor(),
            Lambda(lambda x: x.to(device)),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ConvertImageDtype(precision),
        ]
    )
    
    # TODO: 理解權重載入
    # 1. 為什麼要載入預訓練的 depth_pro 權重？
    # 2. strict=True 意味著什麼？
    # 3. 如果某些層不匹配會發生什麼？
    if config.checkpoint_uri is not None:
        state_dict = torch.load(config.checkpoint_uri, map_location="cpu")
        
        # TODO: 思考：由於我們移除了 FOV head，某些權重可能不匹配
        # 應該如何處理？使用 strict=False 還是手動過濾？
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict=state_dict, strict=False  # 改為 False，因為我們移除了 FOV
        )
        
        if len(unexpected_keys) > 0:
            print(f"[WARNING] Unexpected keys (這些是 FOV 相關的，可以忽略): {unexpected_keys[:5]}...")
        
        # TODO: 理解 missing_keys 的處理
        # 1. 為什麼要過濾掉 "fc_norm"？
        # 2. 還有哪些 key 應該被過濾？
        missing_keys = [key for key in missing_keys if "fc_norm" not in key]
        if len(missing_keys) > 0:
            print(f"[WARNING] Missing keys: {missing_keys[:5]}...")
    
    return model, transform


class LayerOrderModel(nn.Module):
    """Layer Order Prediction Model based on Depth-Pro.
    
    主要修改：
    - 移除 FOV head
    - 輸出改為圖層索引（歸一化到 [0, 1]）
    """
    
    def __init__(
        self,
        encoder: DepthProEncoder,
        decoder: MultiresConvDecoder,
        last_dims: tuple[int, int],
        use_fov_head: bool = False,  # 關鍵：預設為 False
    ):
        """Initialize LayerOrderModel.
        
        Args:
        ----
            encoder: The DepthProEncoder backbone.
            decoder: The MultiresConvDecoder decoder.
            last_dims: The dimension for the last convolution layers (intermediate, output).
            use_fov_head: Whether to use the field-of-view head (should be False).
        """
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
        dim_decoder = decoder.dim_decoder
        
        # TODO: 理解 head 的結構
        # 1. 為什麼第一層是 dim_decoder -> dim_decoder // 2？
        # 2. ConvTranspose2d 的作用是什麼？為什麼 kernel_size=2, stride=2？
        # 3. 最後一層為什麼是 1 通道？
        self.head = nn.Sequential(
            nn.Conv2d(
                dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=1
            ),
            nn.ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.Conv2d(
                dim_decoder // 2,
                last_dims[0],  # 32
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(last_dims[0], last_dims[1], kernel_size=1, stride=1, padding=0),
            # TODO: 思考：應該用什麼激活函數？
            # 選項 1: Sigmoid - 確保輸出在 [0, 1]
            # 選項 2: ReLU - 輸出在 [0, inf)，需要後處理
            # 選項 3: Tanh - 輸出在 [-1, 1]，需要轉換
            # 建議：使用 Sigmoid，因為我們需要 [0, 1] 的圖層索引
            nn.Sigmoid(),  # 關鍵修改：確保輸出在 [0, 1]
        )
        
        # TODO: 理解 bias 初始化
        # 為什麼要將最後一層的 bias 設為 0？
        # 提示：這是一個常見的深度估計技巧
        self.head[4].bias.data.fill_(0)
        
        # TODO: 確認：我們不應該初始化 FOV head
        # 如果 use_fov_head=True，會發生什麼？
        if use_fov_head:
            raise ValueError("LayerOrderModel should not use FOV head!")
    
    @property
    def img_size(self) -> int:
        """Return the internal image size of the network."""
        return self.encoder.img_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for layer order prediction.
        
        Args:
        ----
            x (torch.Tensor): Input image [B, 3, H, W]
        
        Returns:
        -------
            layer_index_map (torch.Tensor): Predicted layer index map [B, 1, H, W], values in [0, 1]
        """
        _, _, H, W = x.shape
        
        # TODO: 理解這個斷言
        # 1. 為什麼輸入尺寸必須是固定的？
        # 2. img_size 是多少？（查看 encoder 的實現）
        assert H == self.img_size and W == self.img_size, \
            f"Input size {H}x{W} must match model size {self.img_size}x{self.img_size}"
        
        # TODO: 理解前向傳播的流程
        # 1. encoder 做了什麼？
        # 2. decoder 做了什麼？
        # 3. head 做了什麼？
        encodings = self.encoder(x)  # 多尺度編碼
        features, _ = self.decoder(encodings)  # 解碼特徵
        layer_index_map = self.head(features)  # 預測圖層索引
        
        # TODO: 確認輸出形狀
        # 應該是什麼形狀？[B, 1, H, W] 還是 [B, H, W]？
        return layer_index_map
    
    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        interpolation_mode: str = "bilinear",
    ) -> Mapping[str, torch.Tensor]:
        """Infer layer order for a given image.
        
        Args:
        ----
            x (torch.Tensor): Input image [3, H, W] or [B, 3, H, W]
            interpolation_mode (str): Interpolation mode for resizing
        
        Returns:
        -------
            Dictionary with "layer_index" tensor [H, W] with values in [0, 1]
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        _, _, H, W = x.shape
        original_size = (H, W)
        resize = H != self.img_size or W != self.img_size
        
        # TODO: 理解 resize 邏輯
        # 1. 為什麼需要 resize 到固定尺寸？
        # 2. 為什麼之後又要 resize 回來？
        if resize:
            x = nn.functional.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode=interpolation_mode,
                align_corners=False,
            )
        
        # TODO: 理解 infer 和 forward 的區別
        # 1. 為什麼 infer 要用 @torch.no_grad()？
        # 2. 什麼時候用 infer，什麼時候用 forward？
        layer_index_map = self.forward(x)  # [B, 1, model_size, model_size]
        
        # TODO: 理解 resize back 的邏輯
        # 1. 為什麼要 resize 回原始尺寸？
        # 2. 這會影響預測的準確性嗎？
        if resize:
            layer_index_map = nn.functional.interpolate(
                layer_index_map,
                size=original_size,
                mode=interpolation_mode,
                align_corners=False,
            )
        
        return {
            "layer_index": layer_index_map.squeeze(0).squeeze(0)  # [H, W]
        }



