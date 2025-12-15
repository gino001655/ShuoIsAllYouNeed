"""
CUSTOM FILE: Layer Order Prediction Loss Functions

實現 Scale-and-Shift Invariant Loss（尺度與位移不變損失）
參考 MiDaS 和 Depth-Pro 論文中的方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def scale_shift_invariant_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Scale-and-Shift Invariant Loss
    
    這個損失函數的核心思想：
    1. 對每張圖的預測值和真實值分別進行標準化
    2. 標準化使用中位數（median）和平均絕對偏差（MAD）
    3. 計算標準化後的 MAE
    
    這樣可以處理：
    - 不同圖片的圖層索引範圍不同（例如 [1,2,3] vs [10,11,12]）
    - 圖層索引的絕對值不重要，重要的是相對順序
    
    Args:
    ----
        pred: 預測的圖層索引圖 [B, H, W] 或 [B, 1, H, W]，值應該在 [0, 1]
        target: 真實的圖層索引圖 [B, H, W] 或 [B, 1, H, W]，值應該在 [0, 1]
        eps: 數值穩定性常數
    
    Returns:
    -------
        loss: scalar tensor
    """
    # TODO: 理解輸入形狀處理
    # 1. 為什麼要統一為 [B, H, W]？
    # 2. 如果輸入是 [B, 1, H, W]，squeeze(1) 做了什麼？
    if pred.dim() == 4:
        pred = pred.squeeze(1)  # [B, H, W]
    if target.dim() == 4:
        target = target.squeeze(1)  # [B, H, W]
    
    B, H, W = pred.shape
    
    # TODO: 理解 flatten 的作用
    # 1. 為什麼要 flatten？
    # 2. view(B, -1) 做了什麼？
    pred_flat = pred.view(B, -1)  # [B, N] where N = H*W
    target_flat = target.view(B, -1)  # [B, N]
    
    losses = []
    
    # TODO: 理解逐圖處理的邏輯
    # 1. 為什麼要對每張圖分別計算標準化？
    # 2. 為什麼不能對整個 batch 一起標準化？
    # 提示：每張圖的圖層索引範圍可能不同
    for b in range(B):
        pred_b = pred_flat[b]  # [N]
        target_b = target_flat[b]  # [N]
        
        # TODO: 理解中位數的作用
        # 1. 為什麼用中位數而不是平均值？
        # 2. 中位數對異常值更魯棒嗎？
        # 提示：中位數不受極端值影響，更適合處理離散的圖層索引
        m_pred = torch.median(pred_b)
        m_target = torch.median(target_b)
        
        # TODO: 理解平均絕對偏差（MAD）
        # 1. MAD 和標準差有什麼不同？
        # 2. 為什麼用 MAD 而不是標準差？
        # 提示：MAD 對異常值更魯棒
        # NOTE:
        # - 在本任務中，target 大多數像素是背景 0，導致 scale（平均絕對偏差）可能非常小。
        # - 若此段在 AMP autocast(FP16) 下執行，(x / tiny_scale) 很容易溢位成 Inf，進而造成 NaN/Inf 梯度。
        # - 這裡做兩件事：加 eps + 對 scale 做下限 clamp，避免 tiny_scale 把數值放大到不合理的程度。
        min_scale = 1e-3
        s_pred = (torch.mean(torch.abs(pred_b - m_pred)) + eps).clamp_min(min_scale)
        s_target = (torch.mean(torch.abs(target_b - m_target)) + eps).clamp_min(min_scale)
        
        # TODO: 理解標準化公式
        # 1. (pred_b - m_pred) / s_pred 做了什麼？
        # 2. 標準化後的值的範圍是什麼？
        # 3. 為什麼這樣可以實現 scale-invariant？
        # 提示：如果所有值都乘以 k，分子和分母都會乘以 k，結果不變
        pred_norm = (pred_b - m_pred) / s_pred
        target_norm = (target_b - m_target) / s_target
        
        # TODO: 理解 MAE Loss
        # 1. 為什麼用 L1（MAE）而不是 L2（MSE）？
        # 2. L1 對邊緣更友好嗎？
        # 提示：L1 鼓勵產生銳利的邊界，適合階梯狀的圖層索引
        loss = torch.mean(torch.abs(pred_norm - target_norm))
        losses.append(loss)
    
    # TODO: 理解最終損失
    # 1. 為什麼要對所有圖的損失求平均？
    # 2. 可以用加權平均嗎？
    return torch.stack(losses).mean()


def edge_preserving_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Edge Preserving Loss
    
    通過比較預測和真實值的梯度來保持銳利的邊緣
    
    TODO: 理解邊緣保持損失
    1. 為什麼圖層索引圖應該有銳利的邊緣？
    2. 梯度損失如何幫助保持邊緣？
    3. 這個損失是必要的嗎？
    """
    # TODO: 理解梯度計算
    # 1. 為什麼要計算水平和垂直梯度？
    # 2. torch.abs 的作用是什麼？
    # 3. 為什麼用 [:, :, :, 1:] - [:, :, :, :-1]？
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    # 水平梯度
    pred_grad_x = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])
    target_grad_x = torch.abs(target[:, :, 1:] - target[:, :, :-1])
    
    # 垂直梯度
    pred_grad_y = torch.abs(pred[:, 1:, :] - pred[:, :-1, :])
    target_grad_y = torch.abs(target[:, 1:, :] - target[:, :-1, :])
    
    # TODO: 理解梯度損失
    # 1. 為什麼用 L1 而不是 L2？
    # 2. 這個損失會鼓勵什麼行為？
    edge_loss = F.l1_loss(pred_grad_x, target_grad_x) + \
                F.l1_loss(pred_grad_y, target_grad_y)
    
    return edge_loss


def ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Explicit Ranking Loss (可選)
    
    確保預測值保持相對順序
    
    TODO: 理解排序損失
    1. 為什麼需要顯式的排序損失？
    2. Scale-Shift Invariant Loss 已經隱式實現了排序，為什麼還需要這個？
    3. 這個損失如何實現？
    
    提示：可以通過對比學習的方式，確保如果 target[i] < target[j]，
    則 pred[i] < pred[j]
    """
    # TODO: 實現排序損失
    # 1. 如何選擇像素對？
    # 2. 如何計算排序損失？
    # 3. margin 的作用是什麼？
    
    # 這是一個可選的實現，主要靠 Scale-Shift Invariant Loss 已經足夠
    # 如果需要更強的排序約束，可以實現這個
    pass
    
    return torch.tensor(0.0, device=pred.device)


class LayerOrderLoss(nn.Module):
    """
    組合損失函數
    
    主要使用 Scale-Shift Invariant Loss
    可選加入 Edge Preserving Loss
    """
    
    def __init__(
        self,
        use_edge_loss: bool = True,
        edge_loss_weight: float = 0.1,
    ):
        """
        Args:
        ----
            use_edge_loss: 是否使用邊緣保持損失
            edge_loss_weight: 邊緣損失的權重
        """
        super().__init__()
        self.use_edge_loss = use_edge_loss
        self.edge_loss_weight = edge_loss_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        計算總損失
        
        Args:
        ----
            pred: 預測的圖層索引圖 [B, 1, H, W] 或 [B, H, W]
            target: 真實的圖層索引圖 [B, 1, H, W] 或 [B, H, W]
        
        Returns:
        -------
            dict with keys:
                - "total_loss": 總損失
                - "scale_shift_loss": Scale-Shift Invariant Loss
                - "edge_loss": Edge Preserving Loss (if used)
        """
        # TODO: 理解損失組合
        # 1. 為什麼主要損失是 scale_shift_loss？
        # 2. edge_loss 的權重應該設多少？
        # 3. 如何平衡不同損失？
        
        # IMPORTANT (Numerical Stability):
        # 在 train loop 中 pred 可能是在 AMP autocast(FP16) 下產生的。
        # 但此任務的 scale/shift 正規化可能把數值放大到 FP16 溢位，造成 Inf/NaN。
        # 因此這裡強制用 FP32 計算 loss（禁用 autocast），避免 loss 計算本身引入溢位。
        with torch.cuda.amp.autocast(enabled=False):
            pred_f = pred.float()
            target_f = target.float()
            scale_shift_loss = scale_shift_invariant_loss(pred_f, target_f)
        
        losses = {
            "scale_shift_loss": scale_shift_loss,
            "total_loss": scale_shift_loss,
        }
        
        if self.use_edge_loss:
            with torch.cuda.amp.autocast(enabled=False):
                edge_loss = edge_preserving_loss(pred_f, target_f)
            losses["edge_loss"] = edge_loss
            losses["total_loss"] = scale_shift_loss + self.edge_loss_weight * edge_loss
        
        return losses





