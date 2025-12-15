"""
CUSTOM FILE: Inference script for Layer Order Prediction (parsed_dataset or single image)

Two modes:
1. Dataset mode: Run inference over the whole parsed_dataset folder (PNG + JSON structure).
   Output: out_dir/<split>/<category>/<sample_id>/pred_layer_index.png

2. Single-image mode: Run inference on a single image file.
   Usage: --single-image <path> --out-path <output.png> (or --out-dir for auto-naming)

Notes:
- This script loads YOUR trained checkpoint (from custom_layer_order_train.py), i.e.
  a dict with key "model_state_dict". If you pass a raw state_dict, it also works.
- We keep inference simple and robust: iterate dataset indices so we can preserve
  category/sample_id without modifying Dataset/collate_fn.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from custom_layer_order_dataset import LayerOrderDataset
from custom_layer_order_model import LayerOrderConfig, create_layer_order_model_and_transforms


def _parse_splits(s: str) -> List[str]:
    splits = [x.strip() for x in s.split(",") if x.strip()]
    allowed = {"train", "val", "test"}
    bad = [x for x in splits if x not in allowed]
    if bad:
        raise ValueError(f"Invalid splits {bad}. Allowed: {sorted(allowed)}")
    return splits


def _load_checkpoint_state(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    # Fall back: assume the loaded object itself is a state_dict
    return ckpt


def _save_pred_png(pred_01: np.ndarray, out_path: Path) -> None:
    """
    Save prediction in [0, 1] float32 as a grayscale PNG (0..255).
    """
    pred_u8 = (np.clip(pred_01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred_u8, mode="L").save(out_path)


def infer_dataset(
    *,
    model: torch.nn.Module,
    dataset: LayerOrderDataset,
    device: torch.device,
    out_dir: Path,
    split: str,
    use_amp: bool,
    skip_existing: bool,
    save_npy: bool,
) -> Tuple[int, int]:
    """
    Returns:
        (num_done, num_skipped)
    """
    if not hasattr(dataset, "samples"):
        raise RuntimeError(
            "This script expects parsed_dataset format (dataset.samples exists). "
            "If you want parquet inference, we can add a separate script."
        )

    num_done = 0
    num_skipped = 0

    pbar = tqdm(range(len(dataset)), desc=f"infer-{split}", dynamic_ncols=True)
    for idx in pbar:
        sample_info = dataset.samples[idx]
        category = sample_info.get("category", "unknown")
        sample_id = sample_info.get("sample_id", f"{idx:08d}")

        pred_png_path = out_dir / split / category / sample_id / "pred_layer_index.png"
        pred_npy_path = out_dir / split / category / sample_id / "pred_layer_index.npy"

        if skip_existing and pred_png_path.exists() and (not save_npy or pred_npy_path.exists()):
            num_skipped += 1
            continue

        batch = dataset[idx]
        x = batch["image"].unsqueeze(0).to(device, non_blocking=True)  # [1, 3, H, W]

        with torch.no_grad():
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    pred = model.forward(x)
            else:
                pred = model.forward(x)

            # [1, 1, H, W] -> [H, W]
            pred = pred.squeeze(0).squeeze(0).clamp(0, 1)
            pred_np = pred.detach().cpu().numpy().astype(np.float32)

        _save_pred_png(pred_np, pred_png_path)
        if save_npy:
            pred_npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(pred_npy_path, pred_np)

        num_done += 1

        if (idx + 1) % 50 == 0:
            pbar.set_postfix({"done": num_done, "skipped": num_skipped})

    return num_done, num_skipped


def infer_single_image(
    *,
    model: torch.nn.Module,
    transform,
    image_path: Path,
    out_path: Path,
    device: torch.device,
    use_amp: bool,
    save_npy: bool,
) -> None:
    """
    Run inference on a single image file.
    """
    img = Image.open(image_path).convert("RGB")
    x = transform(img)  # [3, H, W] normalized, on device

    with torch.no_grad():
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                pred = model.infer(x)["layer_index"]  # [H, W]
        else:
            pred = model.infer(x)["layer_index"]  # [H, W]

        pred_np = pred.clamp(0, 1).detach().cpu().numpy().astype(np.float32)

    _save_pred_png(pred_np, out_path)
    if save_npy:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        npy_path = out_path.with_suffix(".npy")
        np.save(npy_path, pred_np)
        print(f"[SingleInfer] Saved .npy: {npy_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference over parsed_dataset for Layer Order Prediction (or single image)"
    )
    parser.add_argument(
        "--single-image",
        type=str,
        default=None,
        help="Path to a single image file to run inference on (if set, ignores --data-dir and --splits).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to parsed_dataset (category/sample_id/...). Required if --single-image is not set.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to trained checkpoint (e.g., .../checkpoint_best.pt).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for predictions (dataset mode) or output file path (single-image mode).",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Output file path for single-image mode (overrides --out-dir if both are set).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to run: train,val,test (default: all). Ignored if --single-image is set.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1536,
        help="Model/Dataset input size (default: 1536).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device string, e.g. "cuda", "cuda:0", "cpu". Default: auto.',
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use AMP autocast during inference (CUDA only).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose output files already exist (dataset mode only).",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save float32 .npy prediction alongside PNG.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Reserved for future DataLoader-based inference. Currently unused.",
    )
    args = parser.parse_args()

    # Single-image mode
    if args.single_image is not None:
        image_path = Path(args.single_image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        ckpt_path = Path(args.ckpt_path)
        if args.out_path is not None:
            out_path = Path(args.out_path)
        elif args.out_dir is not None:
            out_path = Path(args.out_dir)
        else:
            # Default: save next to input image with _pred suffix
            out_path = image_path.parent / f"{image_path.stem}_pred_layer_index.png"

        if args.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)

        print(f"[SingleInfer] device={device}")
        print(f"[SingleInfer] image={image_path}")
        print(f"[SingleInfer] ckpt={ckpt_path}")
        print(f"[SingleInfer] out={out_path}")
        print(f"[SingleInfer] use_amp={bool(args.use_amp)} save_npy={bool(args.save_npy)}")

        # Build model
        config = LayerOrderConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            decoder_features=256,
            use_grad_checkpointing=False,
            checkpoint_uri=None,
        )
        model, transform = create_layer_order_model_and_transforms(
            config=config,
            device=device,
            precision=torch.float32,
        )

        state_dict = _load_checkpoint_state(ckpt_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[SingleInfer][WARN] missing_keys (show up to 20): {missing[:20]}")
        if unexpected:
            print(f"[SingleInfer][WARN] unexpected_keys (show up to 20): {unexpected[:20]}")
        model.eval()

        infer_single_image(
            model=model,
            transform=transform,
            image_path=image_path,
            out_path=out_path,
            device=device,
            use_amp=bool(args.use_amp),
            save_npy=bool(args.save_npy),
        )
        print(f"[SingleInfer] Done. Output saved to: {out_path}")
        return

    # Dataset mode (original behavior)
    if args.data_dir is None:
        raise ValueError("--data-dir is required when --single-image is not set.")

    splits = _parse_splits(args.splits)
    data_dir = Path(args.data_dir)
    ckpt_path = Path(args.ckpt_path)
    if args.out_dir is None:
        raise ValueError("--out-dir is required in dataset mode.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[Infer] device={device}")
    print(f"[Infer] data_dir={data_dir}")
    print(f"[Infer] ckpt_path={ckpt_path}")
    print(f"[Infer] out_dir={out_dir}")
    print(f"[Infer] splits={splits}")
    print(f"[Infer] use_amp={bool(args.use_amp)} skip_existing={bool(args.skip_existing)} save_npy={bool(args.save_npy)}")

    # Build model (we do NOT load depth_pro.pt here; we load your trained ckpt below).
    config = LayerOrderConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        decoder_features=256,
        use_grad_checkpointing=False,
        checkpoint_uri=None,
    )
    model, _transform = create_layer_order_model_and_transforms(
        config=config,
        device=device,
        precision=torch.float32,
    )

    state_dict = _load_checkpoint_state(ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Infer][WARN] missing_keys (show up to 20): {missing[:20]}")
    if unexpected:
        print(f"[Infer][WARN] unexpected_keys (show up to 20): {unexpected[:20]}")
    model.eval()

    total_done = 0
    total_skipped = 0
    for split in splits:
        ds = LayerOrderDataset(
            data_dir=str(data_dir),
            split=split,
            image_size=int(args.image_size),
            use_parsed_format=True,
            use_augmentation=False,
        )
        done, skipped = infer_dataset(
            model=model,
            dataset=ds,
            device=device,
            out_dir=out_dir,
            split=split,
            use_amp=bool(args.use_amp),
            skip_existing=bool(args.skip_existing),
            save_npy=bool(args.save_npy),
        )
        total_done += done
        total_skipped += skipped
        print(f"[Infer] split={split} done={done} skipped={skipped}")

    print(f"[Infer] ALL DONE. total_done={total_done} total_skipped={total_skipped}")


if __name__ == "__main__":
    # Allow environment-variable overrides (useful in shell scripts)
    # If user exports CKPT_PATH/DATA_DIR/OUT_DIR, they can still pass CLI args explicitly.
    # (We keep this minimal; argparse is the source of truth.)
    _ = os.environ  # keep import for potential debugging
    main()


