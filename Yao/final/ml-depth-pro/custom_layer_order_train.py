"""
CUSTOM FILE: Training Script for Layer Order Prediction

訓練腳本，用於 finetune Depth-Pro 模型來預測圖層順序
"""

import os
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from custom_layer_order_model import (
    create_layer_order_model_and_transforms,
    LayerOrderConfig,
    DEFAULT_LAYER_ORDER_CONFIG,
)
from custom_layer_order_dataset import LayerOrderDataset, collate_fn
from custom_layer_order_loss import LayerOrderLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_model(
    config: LayerOrderConfig,
    device: torch.device,
    precision: torch.dtype = torch.float32,
    freeze_encoder: bool = False,
) -> tuple:
    """
    設置模型
    
    TODO: 理解模型設置
    1. 為什麼要載入預訓練的 Depth-Pro 權重？
    2. freeze_encoder 的作用是什麼？
    3. 什麼時候應該凍結編碼器？
    """
    logger.info("[Model] 創建模型...")
    model, transform = create_layer_order_model_and_transforms(
        config=config,
        device=device,
        precision=precision,
    )
    
    # TODO: 理解凍結策略
    # 1. 為什麼要凍結編碼器？
    # 2. 只訓練解碼器和 head 有什麼好處？
    # 3. 什麼時候應該解凍編碼器？
    if freeze_encoder:
        logger.info("[Model] 凍結編碼器參數...")
        for param in model.encoder.parameters():
            param.requires_grad = False
        # 只訓練解碼器和 head
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"[Model] 可訓練參數: {trainable_params:,} / {total_params:,}")
    
    model.train()
    return model, transform


def setup_optimizer(
    model: nn.Module,
    encoder_lr: float = 5e-6,  # Encoder 使用較低學習率（保留預訓練特徵）
    decoder_lr: float = 5e-5,  # Decoder 使用較高學習率（快速適應新任務，10倍關係）
    weight_decay: float = 1e-4,
) -> tuple:
    """
    設置優化器和學習率調度器（使用差異化學習率）
    
    關鍵設計：差異化學習率 (Differential Learning Rates)
    - Encoder (DINO-v2): 較低學習率 (1e-5)，保留預訓練特徵
    - Decoder + Head: 較高學習率 (1e-4)，快速適應新任務
    
    TODO: 理解差異化學習率
    1. 為什麼 Encoder 要用較低學習率？
       提示：Encoder 已經在大量資料上預訓練，學到了通用的視覺特徵
       我們希望保留這些特徵，只做微調
    
    2. 為什麼 Decoder 要用較高學習率？
       提示：Decoder 負責將特徵轉換為最終輸出
       從深度預測改為圖層順序預測，需要較大的調整
    
    3. 如何分離 Encoder 和 Decoder 的參數？
       提示：通過參數名稱前綴（如 "encoder." 和 "decoder."）
    """
    # TODO: 理解參數分組
    # 1. 如何識別 Encoder 參數？
    # 2. 如何識別 Decoder 和 Head 參數？
    # 提示：通過 param_name 的前綴判斷
    
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 分離 Encoder 和 Decoder 參數
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            # Decoder, head, 或其他層
            decoder_params.append(param)
    
    # TODO: 理解參數組設置
    # 1. 為什麼要用 param_groups？
    # 2. 每個組可以有不同的學習率嗎？
    # 提示：PyTorch 的 optimizer 支援多個參數組，每個組可以有不同的超參數
    
    param_groups = [
        {'params': encoder_params, 'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': decoder_params, 'lr': decoder_lr, 'weight_decay': weight_decay},
    ]
    
    optimizer = AdamW(
        param_groups,
        weight_decay=weight_decay,
    )
    
    logger.info(f"[Optimizer] Encoder 參數: {len(encoder_params)}, 學習率: {encoder_lr}")
    logger.info(f"[Optimizer] Decoder 參數: {len(decoder_params)}, 學習率: {decoder_lr}")
    
    # TODO: 理解學習率調度器
    # 1. CosineAnnealingLR 的作用是什麼？
    # 2. T_max 應該設多少？
    # 3. 還有其他調度器可以選擇嗎？
    # 提示：CosineAnnealingLR 會讓學習率從初始值逐漸降到 0，形成餘弦曲線
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=1000,  # TODO: 根據實際 epoch 數調整
    )
    
    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: LayerOrderLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
    debug_nonfinite: bool = False,
    debug_nonfinite_save_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    訓練一個 epoch
    
    TODO: 理解訓練循環
    1. 為什麼要用 model.train()？
    2. optimizer.zero_grad() 的作用是什麼？
    3. loss.backward() 做了什麼？
    4. optimizer.step() 做了什麼？
    """
    model.train()
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")
    
    total_loss = 0.0
    total_scale_shift_loss = 0.0
    total_edge_loss = 0.0
    num_batches = 0
    
    # TODO: 理解混合精度訓練
    # 1. 什麼是混合精度訓練？
    # 2. 為什麼要用混合精度？
    # 3. 什麼時候應該使用？
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad(set_to_none=True)

    def _tensor_stats(x: torch.Tensor) -> str:
        # Keep this cheap (no .item() on huge tensors besides min/max which are scalar ops).
        if x.numel() == 0:
            return "numel=0"
        x_detached = x.detach()
        finite = torch.isfinite(x_detached)
        finite_ratio = float(finite.float().mean().cpu())
        stats = {
            "shape": tuple(x_detached.shape),
            "dtype": str(x_detached.dtype).replace("torch.", ""),
            "min": float(x_detached.nan_to_num(posinf=0.0, neginf=0.0).min().cpu()),
            "max": float(x_detached.nan_to_num(posinf=0.0, neginf=0.0).max().cpu()),
            "finite_ratio": finite_ratio,
        }
        return f"{stats}"

    def _iter_tensors(obj):
        """Yield all tensors inside a nested structure (tensor / list / tuple / dict)."""
        if torch.is_tensor(obj):
            yield obj
        elif isinstance(obj, (list, tuple)):
            for it in obj:
                yield from _iter_tensors(it)
        elif isinstance(obj, dict):
            for it in obj.values():
                yield from _iter_tensors(it)

    def _param_stats(p: torch.Tensor) -> dict:
        p_det = p.detach()
        finite = torch.isfinite(p_det)
        return {
            "shape": tuple(p_det.shape),
            "dtype": str(p_det.dtype).replace("torch.", ""),
            "min": float(p_det.nan_to_num(posinf=0.0, neginf=0.0).min().cpu()),
            "max": float(p_det.nan_to_num(posinf=0.0, neginf=0.0).max().cpu()),
            "finite_ratio": float(finite.float().mean().cpu()),
        }

    def _has_nonfinite_grads(m: nn.Module) -> bool:
        for p in m.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                return True
        return False

    def _find_first_nonfinite_module(
        m: nn.Module,
        images_: torch.Tensor,
    ) -> Optional[dict]:
        """
        Register lightweight forward hooks on all submodules, run a forward pass,
        and report the first module (in execution order) that outputs a non-finite tensor.
        """
        found = {"info": None}
        hooks = []

        def hook_fn(mod, _inp, out):
            if found["info"] is not None:
                return
            for t in _iter_tensors(out):
                if t.numel() == 0:
                    continue
                if not torch.isfinite(t).all():
                    found["info"] = {
                        "module": mod.__class__.__name__,
                        "output_stats": _tensor_stats(t),
                    }
                    return

        # NOTE: named_modules() includes self at first; we hook all so we can pinpoint.
        for _name, mod in m.named_modules():
            hooks.append(mod.register_forward_hook(hook_fn))

        try:
            with torch.no_grad():
                if use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        _ = m.forward(images_)
                else:
                    _ = m.forward(images_)
        finally:
            for h in hooks:
                h.remove()

        return found["info"]

    def _debug_nonfinite(
        reason: str,
        images_: torch.Tensor,
        targets_: torch.Tensor,
        pred_: Optional[torch.Tensor] = None,
    ):
        """
        Best-effort diagnostics when NaN/Inf is detected.
        This is intentionally verbose but only runs when debug_nonfinite=True.
        """
        if not debug_nonfinite:
            return

        # 1) Runtime scalars / optimizer state
        try:
            lrs = [float(pg.get("lr", 0.0)) for pg in optimizer.param_groups]
        except Exception:
            lrs = []
        scale = None
        if use_amp and scaler is not None:
            try:
                scale = float(scaler.get_scale())
            except Exception:
                scale = None

        logger.warning(
            "[DebugNonFinite] reason=%s epoch=%s batch=%s use_amp=%s scaler_scale=%s lrs=%s",
            reason,
            epoch,
            batch_idx,
            use_amp,
            scale,
            lrs,
        )
        logger.warning("[DebugNonFinite] images_stats=%s", _tensor_stats(images_))
        logger.warning("[DebugNonFinite] target_stats=%s", _tensor_stats(targets_))
        if pred_ is not None:
            logger.warning("[DebugNonFinite] pred_stats=%s", _tensor_stats(pred_))

        # 2) Check parameters / gradients (first few offenders)
        bad_params = []
        for name, p in model.named_parameters():
            if p is None:
                continue
            if not torch.isfinite(p.detach()).all():
                bad_params.append(("param", name, _param_stats(p)))
                if len(bad_params) >= 5:
                    break
            if p.grad is not None and (not torch.isfinite(p.grad.detach()).all()):
                # Gradient stats can be huge; keep it short.
                bad_params.append(("grad", name, _param_stats(p.grad)))
                if len(bad_params) >= 5:
                    break
        if bad_params:
            for kind, name, st in bad_params:
                logger.warning("[DebugNonFinite] nonfinite_%s name=%s stats=%s", kind, name, st)
        else:
            logger.warning("[DebugNonFinite] all params/grads look finite (best-effort check).")

        # 3) Try to locate the first module producing non-finite outputs
        first_bad = _find_first_nonfinite_module(model, images_)
        if first_bad is not None:
            logger.warning("[DebugNonFinite] first_nonfinite_module=%s", first_bad)
        else:
            logger.warning("[DebugNonFinite] no non-finite module output detected in hook pass.")

        # 4) Optional: dump tensors for offline inspection
        if debug_nonfinite_save_dir is not None:
            try:
                dump_dir = Path(debug_nonfinite_save_dir) / "nonfinite_dumps"
                dump_dir.mkdir(parents=True, exist_ok=True)
                dump_path = dump_dir / f"epoch_{epoch:04d}_batch_{batch_idx:06d}.pt"
                payload = {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "use_amp": use_amp,
                    "scaler_scale": scale,
                    "lrs": lrs,
                    "images": images_.detach().cpu(),
                    "targets": targets_.detach().cpu(),
                }
                if pred_ is not None:
                    payload["pred"] = pred_.detach().cpu()
                torch.save(payload, dump_path)
                logger.warning("[DebugNonFinite] dump_saved=%s", str(dump_path))
            except Exception as e:
                logger.warning("[DebugNonFinite] failed_to_save_dump: %s", str(e))
    
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)  # [B, 3, H, W]
        layer_maps = batch["layer_index_map"].to(device)  # [B, H, W]
        
        # TODO: 理解前向傳播
        # 1. model.forward() 返回什麼？
        # 2. 輸出形狀是什麼？
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                pred_layer_map = model.forward(images)  # [B, 1, H, W]
                loss_dict = criterion(pred_layer_map, layer_maps)
        else:
            pred_layer_map = model.forward(images)  # [B, 1, H, W]
            loss_dict = criterion(pred_layer_map, layer_maps)
        
        loss = loss_dict["total_loss"]

        # --- Numerical safety: detect NaN/Inf early with actionable logs.
        # In practice, NaN in scale-shift loss almost always means pred or target already contains NaN/Inf.
        if not torch.isfinite(loss):
            logger.warning(
                "[NonFinite] loss became non-finite. "
                f"epoch={epoch} batch={batch_idx} "
                f"loss={loss.detach().cpu()} "
                f"pred_stats={_tensor_stats(pred_layer_map)} "
                f"target_stats={_tensor_stats(layer_maps)} "
                f"use_amp={use_amp}"
            )
            _debug_nonfinite("loss_nonfinite", images, layer_maps, pred_layer_map)
            optimizer.zero_grad(set_to_none=True)
            continue
        if not torch.isfinite(pred_layer_map).all():
            logger.warning(
                "[NonFinite] pred contains NaN/Inf. "
                f"epoch={epoch} batch={batch_idx} "
                f"pred_stats={_tensor_stats(pred_layer_map)} "
                f"target_stats={_tensor_stats(layer_maps)} "
                f"use_amp={use_amp}"
            )
            _debug_nonfinite("pred_nonfinite", images, layer_maps, pred_layer_map)
            optimizer.zero_grad(set_to_none=True)
            continue
        if not torch.isfinite(layer_maps).all():
            logger.warning(
                "[NonFinite] target contains NaN/Inf (unexpected). "
                f"epoch={epoch} batch={batch_idx} "
                f"target_stats={_tensor_stats(layer_maps)}"
            )
            _debug_nonfinite("target_nonfinite", images, layer_maps, pred_layer_map)
            optimizer.zero_grad(set_to_none=True)
            continue

        loss_for_backward = loss / grad_accum_steps
        
        # TODO: 理解反向傳播 + 梯度累積 (Gradient Accumulation)
        # 1. 為什麼 loss 要除以 grad_accum_steps？
        # 2. 什麼時候才 step()？
        # 3. 梯度裁剪應該發生在 step() 前（且要先 unscale）？
        if use_amp and scaler is not None:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()
        
        do_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(dataloader))
        if do_step:
            # TODO: 理解梯度裁剪
            # 1. 為什麼要裁剪梯度？
            # 2. max_norm 應該設多少？
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                # If grads overflowed to inf, clipping can turn inf -> NaN (0 * inf),
                # poisoning weights and making all subsequent losses NaN.
                if _has_nonfinite_grads(model):
                    logger.warning(
                        "[NonFinite] gradients contain NaN/Inf after unscale; "
                        "skipping grad clipping for this step (GradScaler will skip step + lower scale). "
                        f"epoch={epoch} batch={batch_idx}"
                    )
                    _debug_nonfinite("grads_nonfinite_after_unscale", images, layer_maps, pred_layer_map)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isfinite(total_norm):
                    optimizer.step()
                else:
                    logger.warning(
                        "[NonFinite] gradient norm is non-finite; skipping optimizer step. "
                        f"epoch={epoch} batch={batch_idx} total_norm={total_norm}"
                    )
                    _debug_nonfinite("grad_norm_nonfinite", images, layer_maps, pred_layer_map)
            optimizer.zero_grad(set_to_none=True)
        
        # 記錄損失
        total_loss += loss.item()
        total_scale_shift_loss += loss_dict["scale_shift_loss"].item()
        if "edge_loss" in loss_dict:
            total_edge_loss += loss_dict["edge_loss"].item()
        num_batches += 1
        
        # 更新進度條
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ss_loss": f"{loss_dict['scale_shift_loss'].item():.4f}",
            "accum": f"{grad_accum_steps}",
        })
    
    return {
        "loss": total_loss / num_batches,
        "scale_shift_loss": total_scale_shift_loss / num_batches,
        "edge_loss": total_edge_loss / num_batches if total_edge_loss > 0 else 0.0,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: LayerOrderLoss,
    device: torch.device,
) -> Dict[str, float]:
    """
    驗證模型
    
    TODO: 理解驗證流程
    1. 為什麼驗證時要用 model.eval()？
    2. torch.no_grad() 的作用是什麼？
    3. 驗證和訓練有什麼不同？
    """
    model.eval()
    
    total_loss = 0.0
    total_scale_shift_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch["image"].to(device)
            layer_maps = batch["layer_index_map"].to(device)
            
            pred_layer_map = model.forward(images)
            loss_dict = criterion(pred_layer_map, layer_maps)
            
            total_loss += loss_dict["total_loss"].item()
            total_scale_shift_loss += loss_dict["scale_shift_loss"].item()
            num_batches += 1
    
    return {
        "loss": total_loss / num_batches,
        "scale_shift_loss": total_scale_shift_loss / num_batches,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    save_dir: Path,
    save_every_epoch: bool = False,
):
    """
    保存 checkpoint
    
    Args:
    ----
        save_every_epoch: 是否每個 epoch 都保存（False 則每 10 個 epoch 保存一次）
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
    }
    
    # 保存最新 checkpoint
    torch.save(checkpoint, save_dir / "checkpoint_latest.pt")
    
    # 保存最佳 checkpoint
    best_path = save_dir / "checkpoint_best.pt"
    if not best_path.exists() or loss < torch.load(best_path)["loss"]:
        torch.save(checkpoint, best_path)
        logger.info(f"[Checkpoint] 保存最佳模型 (loss: {loss:.4f})")
    
    # 定期保存（每個 epoch 或每 10 個 epoch）
    if save_every_epoch or epoch % 10 == 0:
        torch.save(checkpoint, save_dir / f"checkpoint_epoch_{epoch}.pt")


def save_training_config(args, save_dir: Path, start_time: datetime):
    """
    保存訓練配置和元資訊
    
    Args:
    ----
        args: 命令行參數
        save_dir: 保存目錄
        start_time: 訓練開始時間
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 將 args 轉換為字典
    config_dict = vars(args) if hasattr(args, '__dict__') else dict(args)
    
    # 添加訓練元資訊
    training_info = {
        "start_time": start_time.isoformat(),
        "start_timestamp": start_time.timestamp(),
        "config": config_dict,
    }
    
    # 保存為 JSON
    config_path = save_dir / "training_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[Config] 訓練配置已保存到: {config_path}")


def create_experiment_dir(base_dir: str, experiment_name: Optional[str] = None) -> Path:
    """
    創建實驗目錄（帶時間戳）
    
    Args:
    ----
        base_dir: 基礎目錄
        experiment_name: 實驗名稱（可選）
    
    Returns:
    -------
        experiment_dir: 實驗目錄路徑
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # 生成時間戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        exp_dir = base_path / f"{experiment_name}_{timestamp}"
    else:
        exp_dir = base_path / f"experiment_{timestamp}"
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def save_prediction_examples(
    model: nn.Module,
    val_dataset,
    device: torch.device,
    epoch: int,
    save_dir: Path,
    num_examples: int = 4,
):
    """
    保存預測範例（圖片、GT、預測結果）
    
    Args:
    ----
        model: 訓練中的模型
        val_dataset: 驗證資料集（不使用資料增強）
        device: 設備
        epoch: 當前 epoch
        save_dir: 保存目錄
        num_examples: 生成幾個範例
    """
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = save_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    # 隨機選擇幾個驗證樣本
    indices = np.random.choice(len(val_dataset), min(num_examples, len(val_dataset)), replace=False)
    
    with torch.no_grad():
        for example_idx, idx in enumerate(indices):
            # 取得樣本（驗證集不使用資料增強）
            sample = val_dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)  # [1, 3, H, W]
            layer_map_gt = sample["layer_index_map"]  # [H, W]
            
            # 預測
            pred_layer_map = model.forward(image)  # [1, 1, H, W] 或 [1, H, W]
            
            # 處理預測結果的形狀
            if pred_layer_map.dim() == 4:
                pred_layer_map = pred_layer_map.squeeze(0).squeeze(0)  # [H, W]
            elif pred_layer_map.dim() == 3:
                pred_layer_map = pred_layer_map.squeeze(0)  # [H, W]
            pred_layer_map = pred_layer_map.cpu()
            
            # 轉換圖像為 numpy（反標準化）
            image_np = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
            image_np = (image_np + 1) / 2  # 反標準化 [-1, 1] -> [0, 1]
            image_np = np.clip(image_np, 0, 1)
            
            # 轉換 GT 和預測為 numpy
            layer_map_gt_np = layer_map_gt.numpy()
            pred_layer_map_np = pred_layer_map.numpy()
            
            # 確保 GT 和預測的尺寸一致（如果需要 resize）
            if layer_map_gt_np.shape != pred_layer_map_np.shape:
                # 如果尺寸不一致，將預測 resize 到 GT 的尺寸
                pred_layer_map_pil = Image.fromarray((pred_layer_map_np * 255).astype(np.uint8), mode='L')
                pred_layer_map_pil = pred_layer_map_pil.resize(
                    (layer_map_gt_np.shape[1], layer_map_gt_np.shape[0]),
                    Image.BILINEAR
                )
                pred_layer_map_np = np.array(pred_layer_map_pil).astype(np.float32) / 255.0
            
            # 創建視覺化
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 原始圖像
            axes[0].imshow(image_np)
            axes[0].set_title("Input Image", fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # Ground Truth
            im1 = axes[1].imshow(layer_map_gt_np, cmap='viridis', vmin=0, vmax=1)
            axes[1].set_title("Ground Truth", fontsize=14, fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # 預測結果
            im2 = axes[2].imshow(pred_layer_map_np, cmap='viridis', vmin=0, vmax=1)
            axes[2].set_title("Prediction", fontsize=14, fontweight='bold')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.suptitle(f"Epoch {epoch} - Example {example_idx + 1}", 
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            # 保存圖片
            output_file = examples_dir / f"epoch_{epoch:04d}_example_{example_idx + 1}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
    
    logger.info(f"[Examples] Epoch {epoch} 的範例已保存到: {examples_dir}")


def main(args):
    """
    主訓練函數
    
    TODO: 理解訓練流程
    1. 訓練的整體流程是什麼？
    2. 每個步驟的作用是什麼？
    3. 如何監控訓練進度？
    """
    # 記錄訓練開始時間
    start_time = datetime.now()
    
    # 創建實驗目錄（帶時間戳）
    if args.experiment_name:
        experiment_dir = create_experiment_dir(args.save_dir, args.experiment_name)
    else:
        experiment_dir = create_experiment_dir(args.save_dir)
    
    logger.info(f"[Experiment] 實驗目錄: {experiment_dir}")
    
    # 保存訓練配置
    save_training_config(args, experiment_dir, start_time)
    
    # 設置日誌文件
    log_file = experiment_dir / "training.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("開始訓練")
    logger.info(f"開始時間: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"實驗目錄: {experiment_dir}")
    logger.info("=" * 80)
    
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Device] 使用設備: {device}")
    
    # 設置模型配置
    config = LayerOrderConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri=args.checkpoint_path,
        decoder_features=256,
        use_grad_checkpointing=args.use_grad_checkpointing,
    )
    
    # 設置模型
    model, transform = setup_model(
        config=config,
        device=device,
        precision=torch.float32,
        freeze_encoder=args.freeze_encoder,
    )
    
    # 設置資料集
    logger.info("[Dataset] 載入訓練集...")
    # 資料格式選擇：預設使用 parsed (PNG+JSON)；若指定 --use-parquet-format 則改用原始 Parquet
    if getattr(args, "use_parsed_format", False) and getattr(args, "use_parquet_format", False):
        raise ValueError("請擇一使用 --use-parsed-format 或 --use-parquet-format（不可同時指定）")
    use_parsed_format = True
    if getattr(args, "use_parquet_format", False):
        use_parsed_format = False
    elif getattr(args, "use_parsed_format", False):
        use_parsed_format = True

    # 資料增強配置
    augmentation_config = None
    if args.use_augmentation:
        augmentation_config = {
            "horizontal_flip_prob": args.horizontal_flip_prob,
            "vertical_flip_prob": args.vertical_flip_prob,
            "rotation_prob": args.rotation_prob,
            "rotation_degrees": args.rotation_degrees,
            "color_jitter_prob": args.color_jitter_prob,
            "color_jitter_brightness": args.color_jitter_brightness,
            "color_jitter_contrast": args.color_jitter_contrast,
            "color_jitter_saturation": args.color_jitter_saturation,
        }
        logger.info(f"[Dataset] 啟用資料增強: {augmentation_config}")
    
    train_dataset = LayerOrderDataset(
        data_dir=args.data_dir,
        split="train",
        image_size=1536,
        use_parsed_format=use_parsed_format,
        use_augmentation=args.use_augmentation,
        augmentation_config=augmentation_config,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    logger.info("[Dataset] 載入驗證集...")
    val_dataset = LayerOrderDataset(
        data_dir=args.data_dir,
        split="val",
        image_size=1536,
        use_parsed_format=use_parsed_format,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # 設置損失函數
    criterion = LayerOrderLoss(
        use_edge_loss=args.use_edge_loss,
        edge_loss_weight=args.edge_loss_weight,
    )
    
    # 設置優化器（使用差異化學習率）
    optimizer, scheduler = setup_optimizer(
        model=model,
        encoder_lr=args.encoder_lr if hasattr(args, 'encoder_lr') else args.learning_rate * 0.1,  # Encoder 用較低學習率
        decoder_lr=args.learning_rate,  # Decoder 用較高學習率
        weight_decay=args.weight_decay,
    )
    
    # 恢復訓練（如果指定）
    start_epoch = 0
    if args.resume:
        logger.info(f"[Checkpoint] 從 {args.resume} 恢復訓練...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    
    # 訓練循環
    logger.info("[Training] 開始訓練...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        # 訓練
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            use_amp=args.use_amp,
            grad_accum_steps=args.grad_accum_steps,
            debug_nonfinite=args.debug_nonfinite,
            debug_nonfinite_save_dir=experiment_dir if args.debug_nonfinite_save else None,
        )
        
        logger.info(
            f"[Epoch {epoch}] Train Loss: {train_metrics['loss']:.4f}, "
            f"SS Loss: {train_metrics['scale_shift_loss']:.4f}"
        )
        
        # 每個 epoch 結束後生成範例結果
        save_prediction_examples(
            model=model,
            val_dataset=val_dataset,
            device=device,
            epoch=epoch,
            save_dir=experiment_dir,
            num_examples=args.num_examples,
        )
        
        # 驗證
        if (epoch + 1) % args.val_freq == 0:
            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
            )
            logger.info(
                f"[Epoch {epoch}] Val Loss: {val_metrics['loss']:.4f}, "
                f"SS Loss: {val_metrics['scale_shift_loss']:.4f}"
            )
            
            # 保存 checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_metrics['loss'],
                save_dir=experiment_dir,
                save_every_epoch=args.save_every_epoch,
            )
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                logger.info(f"[Best] 更新最佳驗證損失: {best_val_loss:.4f}")
        
        # 更新學習率
        scheduler.step()
    
    # 記錄訓練結束時間
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("=" * 80)
    logger.info("訓練完成")
    logger.info(f"結束時間: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"總訓練時間: {duration}")
    logger.info(f"實驗目錄: {experiment_dir}")
    logger.info("=" * 80)
    
    # 保存最終訓練資訊
    final_info = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "duration_str": str(duration),
        "best_val_loss": best_val_loss,
    }
    final_info_path = experiment_dir / "training_summary.json"
    with open(final_info_path, 'w', encoding='utf-8') as f:
        json.dump(final_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[Summary] 訓練摘要已保存到: {final_info_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練圖層順序預測模型")
    
    # 資料相關
    parser.add_argument("--data-dir", type=str, required=True,
                       help="資料集目錄路徑")
    parser.add_argument("--checkpoint-path", type=str,
                       default="./checkpoints/depth_pro.pt",
                       help="預訓練模型路徑")
    
    # 訓練相關
    parser.add_argument("--batch-size", type=int, default=4,
                       help="批次大小")
    parser.add_argument("--num-epochs", type=int, default=100,
                       help="訓練輪數")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Decoder 學習率（較高，預設 5e-5）")
    parser.add_argument("--encoder-lr", type=float, default=5e-6,
                       help="Encoder 學習率（較低，預設 5e-6，保留預訓練特徵）")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="權重衰減")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="資料載入進程數")
    
    # 損失相關
    parser.add_argument("--use-edge-loss", action="store_true",
                       help="使用邊緣保持損失")
    parser.add_argument("--edge-loss-weight", type=float, default=0.1,
                       help="邊緣損失權重")
    
    # 模型相關
    parser.add_argument("--freeze-encoder", action="store_true",
                       help="凍結編碼器參數")
    
    # 其他
    parser.add_argument("--save-dir", type=str, default="./checkpoints/layer_order",
                       help="保存目錄（會在此目錄下創建帶時間戳的子目錄）")
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="實驗名稱（可選，會添加到目錄名稱前）")
    parser.add_argument("--resume", type=str, default=None,
                       help="恢復訓練的 checkpoint 路徑")
    parser.add_argument("--val-freq", type=int, default=1,
                       help="驗證頻率（每 N 個 epoch）")
    parser.add_argument("--use-amp", action="store_true",
                       help="使用混合精度訓練")
    parser.add_argument("--use-grad-checkpointing", action="store_true",
                       help="對 ViT encoder 啟用 gradient checkpointing（省 VRAM、會變慢）")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                       help="梯度累積步數（有效 batch = batch_size * grad_accum_steps）")
    parser.add_argument("--save-every-epoch", action="store_true",
                       help="每個 epoch 都保存 checkpoint（否則每 10 個 epoch 保存一次）")
    parser.add_argument("--use-parsed-format", action="store_true", default=False,
                       help="使用已解析的資料格式（PNG+JSON）。若兩者都不指定，預設仍為 parsed。")
    parser.add_argument("--use-parquet-format", action="store_true", default=False,
                       help="使用原始 Parquet 格式（data_dir 需指向包含 snapshots/ 的資料夾，如 ../dataset）")
    
    # 資料增強相關
    parser.add_argument("--use-augmentation", action="store_true",
                       help="啟用資料增強（僅在訓練時）")
    parser.add_argument("--horizontal-flip-prob", type=float, default=0.5,
                       help="水平翻轉機率")
    parser.add_argument("--vertical-flip-prob", type=float, default=0.5,
                       help="垂直翻轉機率")
    parser.add_argument("--rotation-prob", type=float, default=0.5,
                       help="旋轉機率")
    parser.add_argument("--rotation-degrees", type=int, default=90,
                       choices=[90, 180, 270],
                       help="旋轉角度（90/180/270，90 會隨機選擇）")
    parser.add_argument("--color-jitter-prob", type=float, default=0.5,
                       help="顏色抖動機率")
    parser.add_argument("--color-jitter-brightness", type=float, default=0.2,
                       help="顏色抖動亮度範圍")
    parser.add_argument("--color-jitter-contrast", type=float, default=0.2,
                       help="顏色抖動對比度範圍")
    parser.add_argument("--color-jitter-saturation", type=float, default=0.2,
                       help="顏色抖動飽和度範圍")
    parser.add_argument("--num-examples", type=int, default=4,
                       help="每個 epoch 生成幾個預測範例")

    # Debugging: NaN/Inf diagnostics
    parser.add_argument("--debug-nonfinite", action="store_true",
                       help="偵測到 NaN/Inf 時進行更完整的檢查（參數/梯度/第一個出問題的 module）")
    parser.add_argument("--debug-nonfinite-save", action="store_true",
                       help="若啟用 --debug-nonfinite，另外把出問題 batch 的 input/GT/pred dump 到實驗資料夾")
    
    args = parser.parse_args()
    main(args)


