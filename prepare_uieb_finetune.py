#!/usr/bin/env python3
"""Prepare paired UIEB images, Sobel gradients, depth maps, and split files."""

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--depth_anything_root", type=Path, required=True)
    parser.add_argument("--depth_checkpoint", type=Path, required=True)
    parser.add_argument("--train_size", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def compute_grad(img_bgr):
    """Return a normalized uint8 Sobel gradient magnitude."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    max_value = magnitude.max()
    if max_value > 0:
        magnitude = magnitude / max_value * 255.0
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def load_depth_model(root, checkpoint, device):
    """Load the Depth Anything V2 Large model from a local checkout."""
    sys.path.insert(0, str(root.resolve()))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
    )
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"[depth] loaded Depth Anything V2 Large on {device}")
    return model


def infer_depth(model, img_bgr):
    """Infer a relative depth map and normalize it to uint8."""
    height, width = img_bgr.shape[:2]
    depth = model.infer_image(img_bgr)
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max - depth_min > 1e-6:
        depth = (depth - depth_min) / (depth_max - depth_min) * 255.0
    else:
        depth = np.zeros_like(depth)
    depth = depth.astype(np.uint8)
    if depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    return depth


def collect_pairs(raw_dir, reference_dir):
    reference_names = {
        path.name
        for path in reference_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    return sorted(
        path.name
        for path in raw_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.name in reference_names
    )


def main():
    args = parse_args()
    for path, label in (
        (args.raw_dir, "raw image directory"),
        (args.reference_dir, "reference image directory"),
        (args.depth_anything_root, "Depth Anything V2 checkout"),
        (args.depth_checkpoint, "Depth Anything V2 checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    files = collect_pairs(args.raw_dir, args.reference_dir)
    if not files:
        raise RuntimeError("No paired images with matching filenames were found")
    if not 0 < args.train_size < len(files):
        raise ValueError(
            f"--train_size must be between 1 and {len(files) - 1}; got {args.train_size}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(files)
    train_files = files[: args.train_size]
    test_files = files[args.train_size :]

    for name in ("input", "target", "depth", "grad"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_list.txt").write_text(
        "\n".join(train_files) + "\n", encoding="utf-8"
    )
    (args.output_dir / "test_list.txt").write_text(
        "\n".join(test_files) + "\n", encoding="utf-8"
    )

    print(f"[data] {len(files)} pairs: {len(train_files)} train, {len(test_files)} test")
    depth_model = load_depth_model(
        args.depth_anything_root, args.depth_checkpoint, args.device
    )

    for filename in tqdm(files, desc="Preparing UIEB"):
        source_input = args.raw_dir / filename
        source_target = args.reference_dir / filename
        output_input = args.output_dir / "input" / filename
        output_target = args.output_dir / "target" / filename
        output_depth = args.output_dir / "depth" / filename
        output_grad = args.output_dir / "grad" / filename

        if not output_input.exists():
            shutil.copy2(source_input, output_input)
        if not output_target.exists():
            shutil.copy2(source_target, output_target)

        image = None
        if not output_grad.exists() or not output_depth.exists():
            image = cv2.imread(str(source_input), cv2.IMREAD_COLOR)
            if image is None:
                print(f"[skip] failed to read {source_input}")
                continue
        if not output_grad.exists():
            cv2.imwrite(str(output_grad), compute_grad(image))
        if not output_depth.exists():
            with torch.no_grad():
                depth = infer_depth(depth_model, image)
            cv2.imwrite(str(output_depth), depth)

    print(f"[done] prepared dataset at {args.output_dir}")


if __name__ == "__main__":
    main()
