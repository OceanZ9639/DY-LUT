#!/usr/bin/env python3
"""Run DY-LUT inference from a checkpoint or exported LUT."""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch.nn as nn

from models.lut_model import Multi4DLUT
from models.context_gen import ContextGenerator, DualBranchContextEncoder
from models.local_refine import LocalRefine, ResRefine
from utils.interpolation import (
    FastLUTInference, lut_dim_for_mode, context_channels_for_mode, ALL_LUT_MODES,
)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


class _FixedDegWrapper(nn.Module):
    """Reproduce fixed-coordinate or uniform-weight ablations at inference."""
    def __init__(self, base_encoder, fix_deg=None, use_adaptive_weights=True):
        super().__init__()
        self.base         = base_encoder
        self.fix_deg      = fix_deg              # None / 'A' / 'B' / 'AB'
        self.use_adap_w   = use_adaptive_weights
        # Proxy attributes used by FastLUTInference.
        self.is_dual_branch = True
        self.in_channels    = base_encoder.in_channels
        self.num_luts       = base_encoder.num_luts

    def forward(self, x, global_stats=None):
        deg_map, lut_w = self.base(x, global_stats)
        cb = x[:, 1:2]
        cr = x[:, 2:3]
        if self.fix_deg in ('A', 'AB'):
            deg_map = torch.cat([cb, deg_map[:, 1:2]], dim=1)
        if self.fix_deg in ('B', 'AB'):
            deg_map = torch.cat([deg_map[:, 0:1], cr], dim=1)
        if not self.use_adap_w:
            B, K = lut_w.shape
            lut_w = torch.ones(B, K, device=lut_w.device) / K
        return deg_map, lut_w


def load_model(checkpoint_path, device='cuda', lut_mode=None):
    """Load a model and reconstruct its architecture from checkpoint metadata."""
    print(f"Loading model: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    saved_mode = checkpoint.get('lut_mode', 'full')
    mode = lut_mode if lut_mode is not None else saved_mode
    lut_dim = lut_dim_for_mode(mode)

    lut_state = checkpoint.get('lut_model_state', {})
    lut_tensor = lut_state.get('luts.0.lut', None)
    # Recover per-axis bin counts from the LUT tensor shape.
    if lut_tensor is not None:
        bins = list(lut_tensor.shape[:-1])
        if len(set(bins)) == 1:
            bins = bins[0]
    else:
        bins = 17
    num_luts = sum(1 for k in lut_state if k.endswith('.lut')) or 1

    lut_model = Multi4DLUT(
        num_luts=num_luts, bins=bins, output_channels=3, lut_dim=lut_dim,
    ).to(device)
    lut_model.load_state_dict(checkpoint['lut_model_state'])
    lut_model.eval()

    # Reconstruct the saved encoder type.
    ctx_type  = checkpoint.get('context_gen_type', 'single')
    ctx_hid   = checkpoint.get('context_gen_hidden', 16)
    ctx_layers = checkpoint.get('context_gen_layers', 3)
    ctx_nluts = checkpoint.get('context_gen_num_luts', num_luts)

    # Optional features default to disabled for older checkpoints.
    predict_gate = bool(checkpoint.get('predict_depth_gate', False))
    index_remap  = checkpoint.get('index_remap', None)
    ctx_downsample = checkpoint.get('ctx_downsample', None)
    pixel_fusion = bool(checkpoint.get('pixel_fusion', False))
    use_ppm = bool(checkpoint.get('use_ppm', False))

    context_gen = None
    if checkpoint.get('context_gen_state') is not None:
        if ctx_type == 'dual_branch' and mode == 'full':
            ctx_in_ch = checkpoint.get('context_gen_in_channels', 5)
            context_gen = DualBranchContextEncoder(
                in_channels=ctx_in_ch,
                hidden_dim=ctx_hid,
                num_layers=ctx_layers,
                num_luts=ctx_nluts,
                predict_gate=predict_gate,
                pixel_fusion=pixel_fusion,
                use_ppm=use_ppm,
            ).to(device)
            print(f"  [DualBranchContextEncoder] in={ctx_in_ch}ch, hidden={ctx_hid}, "
                  f"num_luts={ctx_nluts}, gate={predict_gate}, "
                  f"pixel_fusion={pixel_fusion}, ppm={use_ppm}")

            # Reproduce fixed-index and fixed-weight ablations when recorded.
            fix_deg       = checkpoint.get('fix_deg_channel', None)       # None/'A'/'B'/'AB'
            use_adap_w    = checkpoint.get('use_adaptive_weights', True)
            if fix_deg or not use_adap_w:
                context_gen = _FixedDegWrapper(context_gen, fix_deg, use_adap_w)
                print(f"  [ablation] fix_deg={fix_deg} use_adaptive_weights={use_adap_w}")
        else:
            ctx_ch = context_channels_for_mode(mode)
            if ctx_ch is not None:
                context_gen = ContextGenerator(
                    in_channels=ctx_ch, hidden_dim=ctx_hid, num_layers=ctx_layers,
                ).to(device)
        if context_gen is not None:
            base = context_gen.base if isinstance(context_gen, _FixedDegWrapper) else context_gen
            base.load_state_dict(checkpoint['context_gen_state'])
            context_gen.eval()

    # Load the optional local refinement module.
    refine = None
    if checkpoint.get('use_local_refine', False) and checkpoint.get('refine_state') is not None:
        refine_dim = checkpoint.get('refine_hidden_dim', 16)
        refine_layers = checkpoint.get('refine_num_layers', 3)
        refine_type = checkpoint.get('refine_type', 'local')
        refine_cls = ResRefine if refine_type == 'res' else LocalRefine
        refine = refine_cls(hidden_dim=refine_dim, num_layers=refine_layers).to(device)
        refine.load_state_dict(checkpoint['refine_state'])
        refine.eval()
        print(f"  [{refine_cls.__name__}] hidden={refine_dim}, layers={refine_layers}")

    extras = []
    if index_remap is not None:
        extras.append(f"index_remap={list(index_remap.keys())}")
    if ctx_downsample is not None:
        extras.append(f"ctx_downsample={ctx_downsample}")
    print(f"  lut_mode={mode}  bins={bins}  num_luts={num_luts}  encoder={ctx_type}  "
          f"refine={refine is not None}  {' '.join(extras)}")
    return lut_model, context_gen, bins, mode, refine, index_remap, ctx_downsample


def load_lut_from_npy(lut_path, context_path, device='cuda', lut_mode='full'):
    print(f"Loading LUT: {lut_path}")
    lut = torch.from_numpy(np.load(lut_path)).to(device)

    ctx_ch = context_channels_for_mode(lut_mode)
    context_gen = None
    if ctx_ch is not None and context_path:
        print(f"Loading context encoder: {context_path}")
        context_gen = ContextGenerator(
            in_channels=ctx_ch, hidden_dim=16, num_layers=3,
        ).to(device)
        context_gen.load_state_dict(torch.load(context_path, map_location=device))
        context_gen.eval()

    return lut, context_gen


def process_folder(input_dir, depth_dir, output_dir, fast_infer, need_depth=True):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {
            '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'
        }
    )
    print(f"Found {len(image_files)} images")

    depth_dir = Path(depth_dir) if depth_dir else None

    # Warm up only when the first input and its required depth map are readable.
    if len(image_files) > 0:
        img0 = cv2.imread(str(image_files[0]))
        d0 = None
        if need_depth and depth_dir:
            dp = depth_dir / image_files[0].name
            if not dp.exists():
                dp = depth_dir / (image_files[0].stem + '.png')
            if dp.exists():
                d0 = cv2.imread(str(dp), cv2.IMREAD_GRAYSCALE)
        if img0 is not None and (not need_depth or d0 is not None):
            for _ in range(3):
                fast_infer.enhance(img0, d0)

    total_time = 0
    total_gpu_time = 0
    processed = 0

    for img_path in image_files:
        depth = None
        if need_depth and depth_dir:
            depth_path = depth_dir / img_path.name
            if not depth_path.exists():
                depth_path = depth_dir / (img_path.stem + '.png')
            if not depth_path.exists():
                print(f"[skip] no matching depth map for {img_path.name}")
                continue
            depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)

        rgb = cv2.imread(str(img_path))
        if rgb is None or (need_depth and depth is None):
            print(f"[skip] failed to read inputs for {img_path.name}")
            continue

        synchronize(fast_infer.device)
        start_time = time.perf_counter()
        enhanced = fast_infer.enhance(rgb, depth)
        synchronize(fast_infer.device)
        elapsed = time.perf_counter() - start_time
        total_time += elapsed
        processed += 1

        net_ms = fast_infer.last_gpu_ms if fast_infer.last_gpu_ms is not None else 0.0
        total_gpu_time += net_ms

        output_path = output_dir / (img_path.stem + '.png')
        cv2.imwrite(str(output_path), enhanced)

        scale_str = (f" scale={fast_infer.last_infer_scale:.2f}"
                     if fast_infer.last_infer_scale < 1.0 else "")
        print(f"Processed {img_path.name}: total={elapsed*1000:.1f}ms, net={net_ms:.1f}ms{scale_str}")

    if processed > 0:
        avg_total_ms = total_time / processed * 1000
        avg_gpu_ms = total_gpu_time / processed
        print(f"\n{'='*50}")
        print(f"Processed images: {processed}")
        print(f"Average end-to-end time: {avg_total_ms:.2f} ms/image")
        print(f"Average network time: {avg_gpu_ms:.2f} ms/image")
        print(f"{'='*50}")
        return avg_total_ms, avg_gpu_ms, processed
    return None, None, 0


def main():
    parser = argparse.ArgumentParser(description='DY-LUT underwater image enhancement')
    parser.add_argument('--checkpoint', type=str, help='checkpoint path')
    parser.add_argument('--lut', type=str, help='exported LUT .npy path')
    parser.add_argument('--context', type=str, help='context encoder state path')
    parser.add_argument('--input', type=str, required=True, help='input image or directory')
    parser.add_argument('--depth', type=str, default=None, help='depth image or directory')
    parser.add_argument('--output', type=str, required=True, help='output image or directory')
    parser.add_argument('--device', type=str, default='cuda', help='inference device')
    parser.add_argument(
        '--lut_mode', type=str, default=None, choices=list(ALL_LUT_MODES),
        help='LUT mode; defaults to checkpoint metadata',
    )
    parser.add_argument(
        '--max_side', type=int, default=None,
        help='maximum inference side length; overrides automatic scaling',
    )
    parser.add_argument(
        '--infer_scale', type=float, default=None,
        help='fixed inference scale in (0,1]; overrides all other scaling options',
    )
    parser.add_argument(
        '--no_auto_scale', action='store_true',
        help='disable adaptive downsampling for native-resolution evaluation',
    )
    parser.add_argument(
        '--ctx_downsample', type=int, default=None,
        help='maximum context-encoder side; 0 disables checkpoint downsampling',
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    index_remap = None
    ctx_downsample = None
    if args.checkpoint:
        lut_model, context_gen, bins, mode, refine, index_remap, ctx_downsample = load_model(
            args.checkpoint, device, lut_mode=args.lut_mode,
        )
        luts, static_weights = lut_model()      # luts: list of all LUT tensors
    elif args.lut:
        mode = args.lut_mode or 'full'
        lut, context_gen = load_lut_from_npy(
            args.lut, args.context, device, lut_mode=mode,
        )
        luts = [lut]
        refine = None
    else:
        parser.error("one of --checkpoint or --lut is required")

    # Command-line value overrides checkpoint metadata; zero disables downsampling.
    if args.ctx_downsample is not None:
        ctx_downsample = args.ctx_downsample if args.ctx_downsample > 0 else None

    need_depth = mode in ('full', 'rgb_depth', 'ycbcr_depth')

    if need_depth and args.depth is None:
        parser.error(f"--depth is required for lut_mode={mode}")

    auto_scale = not args.no_auto_scale
    if args.infer_scale is not None:
        print(f"[scaling] fixed scale: {args.infer_scale}")
    elif args.max_side is not None:
        print(f"[scaling] maximum side: {args.max_side} px")
    elif auto_scale:
        print("[scaling] automatic: >=2880px -> 0.25x, >=1920px -> 0.5x")
    else:
        print("[scaling] disabled; using native resolution")

    fast_infer = FastLUTInference(
        luts, context_gen, device=str(device), mode=mode, refine=refine,
        auto_scale=auto_scale, max_side=args.max_side, infer_scale=args.infer_scale,
        index_remap=index_remap, ctx_downsample=ctx_downsample,
    )
    # For non-DualBranch models with static fusion weights, pass the correct weights
    if not getattr(context_gen, 'is_dual_branch', False) and len(luts) > 1:
        fast_infer._weights = static_weights.to(device)

    input_path = Path(args.input)

    if input_path.is_dir():
        avg_total_ms, avg_gpu_ms, n_images = process_folder(
            args.input, args.depth, args.output, fast_infer, need_depth=need_depth
        )
        if n_images > 0 and avg_gpu_ms is not None:
            n_params = 0
            if args.checkpoint and lut_model is not None:
                n_params = sum(p.numel() for p in lut_model.parameters())
                if context_gen is not None:
                    n_params += sum(p.numel() for p in context_gen.parameters())
                if refine is not None:
                    n_params += sum(p.numel() for p in refine.parameters())
            timing_path = Path(args.output) / "timing.txt"
            with open(timing_path, "w") as f:
                f.write(f"params\t{n_params}\n")
                f.write(f"avg_net_time_ms\t{avg_gpu_ms:.4f}\n")
                f.write(f"avg_e2e_time_ms\t{avg_total_ms:.4f}\n")
                f.write(f"n_images\t{n_images}\n")
            print(f"Timing written to {timing_path}")
    else:
        rgb = cv2.imread(str(input_path))
        if rgb is None:
            raise FileNotFoundError(f"Failed to read input image: {input_path}")
        depth = None
        if need_depth and args.depth:
            depth = cv2.imread(args.depth, cv2.IMREAD_GRAYSCALE)
            if depth is None:
                raise FileNotFoundError(f"Failed to read depth image: {args.depth}")

        for _ in range(3):
            fast_infer.enhance(rgb, depth)

        synchronize(device)
        start_time = time.perf_counter()
        enhanced = fast_infer.enhance(rgb, depth)
        synchronize(device)
        elapsed = time.perf_counter() - start_time

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.output, enhanced)
        net_ms = fast_infer.last_gpu_ms if fast_infer.last_gpu_ms is not None else 0.0
        print(f"Wrote {args.output}: total={elapsed*1000:.1f}ms, net={net_ms:.1f}ms")


if __name__ == "__main__":
    main()
