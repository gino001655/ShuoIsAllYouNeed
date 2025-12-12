#!/usr/bin/env python
"""
Turn LayerD's layer decomposition output (layer-by-layer PNG) into data format CLD needs. 

Input:
  1) LayerD output root directory: each sample has a subdirectory containing 0000.png (background) and subsequent foreground layers.
  2) Source root directory: find the original whole image by matching the filename (without extension) to the original image, as whole_image.

Output:
  - In the target directory, create a subdirectory for each sample, containing:
      whole_image.png (copied from the original data)
      base_image.png (using LayerD's background 0000.png)
      layer_xx.png (single mask layer after connected component segmentation for each foreground layer, corresponding to bbox)
  - metadata.jsonl: each line is a sample, fields match CLD's Dataset requirements:
      whole_image, whole_caption, base_image, layer_count,
      layer_00, layer_00_box, layer_01, layer_01_box, ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from glob import glob
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm


def find_source_image(stem: str, source_root: Path, exts: Iterable[str]) -> Path | None:
    """Find the original whole image by matching the filename (without extension) to the original image, as whole_image."""
    for ext in exts:
        direct = source_root / f"{stem}{ext}"
        if direct.exists():
            return direct
    # Recursive search (to avoid different path levels)
    for ext in exts:
        matches = list(source_root.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]
    return None


def split_connected_components(layer_rgba: Image.Image, min_area: int = 1) -> List[Tuple[Image.Image, Tuple[int, int, int, int], int]]:
    """
    Split the RGBA image into connected components where alpha > 0.
    Return: (single component RGBA image, bbox(x0,y0,x1,y1 half-open), area) list, sorted by area from largest to smallest.
    """
    arr = np.array(layer_rgba)
    alpha = arr[..., 3]
    mask = alpha > 0
    if mask.sum() == 0:
        return []

    structure = np.ones((3, 3), dtype=np.int8)  # 8-connected
    labels, num = ndimage.label(mask, structure=structure)

    comps: List[Tuple[Image.Image, Tuple[int, int, int, int], int]] = []
    for label_idx in range(1, num + 1):
        comp_mask = labels == label_idx
        area = int(comp_mask.sum())
        if area < min_area:
            continue
        ys, xs = np.where(comp_mask)
        x_min, x_max = int(xs.min()), int(xs.max()) + 1  # half-open
        y_min, y_max = int(ys.min()), int(ys.max()) + 1

        comp_arr = np.zeros_like(arr)
        comp_arr[comp_mask] = arr[comp_mask]
        comp_img = Image.fromarray(comp_arr, mode="RGBA")
        comps.append((comp_img, (x_min, y_min, x_max, y_max), area))

    # Sort by area from largest to smallest, then by y, x to maintain reproducibility
    comps.sort(key=lambda x: (-x[2], x[1][1], x[1][0]))
    return comps


def process_sample(case_dir: Path, source_root: Path, output_root: Path, caption: str, exts: Iterable[str]) -> dict | None:
    """
    Convert a single LayerD output directory to the structure CLD needs, return metadata dictionary.
    """
    pngs = sorted(case_dir.glob("*.png"))
    if len(pngs) < 1:
        print(f"[WARN] No PNG found in {case_dir}, skip.")
        return None

    stem = case_dir.name
    sample_dir = output_root / stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Get the original whole image
    src_img_path = find_source_image(stem, source_root, exts)
    if src_img_path is None:
        print(f"[WARN] Cannot find source image for {stem}, skip this sample.")
        return None

    whole_image_out = sample_dir / "whole_image.png"
    if not whole_image_out.exists():
        shutil.copy2(src_img_path, whole_image_out)

    # Background layer (0000.png)
    bg_path = pngs[0]
    base_image_out = sample_dir / "base_image.png"
    if not base_image_out.exists():
        Image.open(bg_path).convert("RGBA").save(base_image_out)

    layer_entries = []
    layer_counter = 0

    for fg_path in pngs[1:]:
        fg_img = Image.open(fg_path).convert("RGBA")
        comps = split_connected_components(fg_img)
        for comp_img, bbox, _area in comps:
            out_path = sample_dir / f"layer_{layer_counter:02d}.png"
            comp_img.save(out_path)
            layer_entries.append((out_path, bbox))
            layer_counter += 1

    meta = {
        "whole_image": str(whole_image_out.resolve()),
        "whole_caption": caption,
        "base_image": str(base_image_out.resolve()),
        "layer_count": layer_counter,
    }

    for idx, (p, bbox) in enumerate(layer_entries):
        meta[f"layer_{idx:02d}"] = str(p.resolve())
        meta[f"layer_{idx:02d}_box"] = list(map(int, bbox))  # [x0, y0, x1, y1] half-open

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LayerD outputs to CLD dataset format.")
    parser.add_argument("--layerd-output-root", required=True, type=Path, help="LayerD output root directory, each sample has a subdirectory containing 0000.png (background) and subsequent foreground layers.")
    parser.add_argument("--source-root", required=True, type=Path, help="Source root directory, find the original whole image by matching the filename (without extension) to the original image, as whole_image.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output directory, containing sample subdirectories and metadata.jsonl.")
    parser.add_argument("--caption", default="A neutral scene description.", help="Default neutral prompt.")
    parser.add_argument("--exts", nargs="+", default=[".png", ".jpg", ".jpeg", ".webp"], help="Search for original whole image extensions.")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    case_dirs = [p for p in args.layerd_output_root.iterdir() if p.is_dir()]
    metas = []
    for case_dir in tqdm(sorted(case_dirs), desc="Converting LayerD outputs"):
        meta = process_sample(case_dir, args.source_root, args.output_root, args.caption, args.exts)
        if meta:
            metas.append(meta)

    metadata_path = args.output_root / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[INFO] Done. {len(metas)} samples written to {metadata_path}")


if __name__ == "__main__":
    main()

