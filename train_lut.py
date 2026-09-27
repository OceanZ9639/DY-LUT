#!/usr/bin/env python3
"""
Train DY-LUT models.

This script is intentionally aligned with inference.py checkpoint loading:
best_model.pth contains lut_model_state, context_gen_state, refine_state and
all metadata needed by inference.py.
"""

import argparse
import importlib
import importlib.util
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import create_mixed_dataloader
from models.context_gen import ContextGenerator, DualBranchContextEncoder
from models.local_refine import LocalRefine, ResRefine
from models.losses import CombinedLoss
from models.lut_model import Multi4DLUT
from utils.interpolation import (
    ALL_LUT_MODES,
    apply_multi_4d_lut,
    context_channels_for_mode,
    load_index_remap,
    lut_dim_for_mode,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DY-LUT")
    parser.add_argument(
        "--config",
        type=str,
        default="uieb_finetune",
        help="Config name under configs/ without _config, or a .py path.",
    )
    parser.add_argument("--data_root", type=str, default=None, help="Override config.data_root")
    parser.add_argument("--out_dir", type=str, default=None, help="Override checkpoint output dir")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu; default uses config.device")
    parser.add_argument("--epochs", type=int, default=None, help="Override config.num_epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config.batch_size")
    parser.add_argument("--num_workers", type=int, default=None, help="Override config.num_workers")
    parser.add_argument("--lut_mode", type=str, default=None, choices=list(ALL_LUT_MODES))
    parser.add_argument("--resume", type=str, default=None, help="Resume full training state")
    parser.add_argument("--finetune", type=str, default=None, help="Load model weights only")
    parser.add_argument("--no_vgg", action="store_true", help="Disable VGG perceptual loss")
    parser.add_argument("--debug", action="store_true", help="Use config.debug_samples")
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fix_deg_channel", type=str, default=None, choices=["A", "B", "AB"])
    parser.add_argument("--no_adaptive_weights", action="store_true")
    return parser.parse_args()


def load_config(config_arg):
    if config_arg.endswith(".py") or os.path.sep in config_arg or "/" in config_arg:
        path = Path(config_arg).resolve()
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        name = config_arg[:-7] if config_arg.endswith("_config") else config_arg
        module = importlib.import_module(f"configs.{name}_config")
    return module.Config()


def apply_overrides(config, args):
    if args.data_root:
        config.data_root = args.data_root
        for key in ("dataset_1600", "dataset_890"):
            if hasattr(config, key):
                ds = getattr(config, key)
                ds["input_rgb"] = str(Path(args.data_root) / "input")
                ds["input_depth"] = str(Path(args.data_root) / "depth")
                ds["input_grad"] = str(Path(args.data_root) / "grad")
                ds["target"] = str(Path(args.data_root) / "target")
        config.train_list_file = str(Path(args.data_root) / "train_list.txt")
        config.test_list_file = str(Path(args.data_root) / "test_list.txt")
    if args.out_dir:
        config.checkpoint_dir = args.out_dir
    if args.device:
        config.device = args.device
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.lut_mode is not None:
        config.lut_mode = args.lut_mode
    if args.save_interval is not None:
        config.save_interval = args.save_interval
    if args.seed is not None:
        config.seed = args.seed
    if args.debug:
        config.debug = True
    if args.no_vgg:
        config.loss_vgg_weight = 0.0
    if args.fix_deg_channel is not None:
        config.fix_deg_channel = args.fix_deg_channel
    if args.no_adaptive_weights:
        config.use_adaptive_weights = False
    patch_split_files(config)
    return config


def patch_split_files(config):
    if not getattr(config, "use_file_split", False):
        return

    data_root = Path(getattr(config, "data_root", "."))
    fallback_pairs = (
        ("train_list_file", ("train_list.txt", "smdr_train_list.txt")),
        ("test_list_file", ("test_list.txt", "smdr_test_list.txt")),
    )
    for attr, names in fallback_pairs:
        current = Path(getattr(config, attr, ""))
        if current.exists():
            continue
        for name in names:
            candidate = data_root / name
            if candidate.exists():
                setattr(config, attr, str(candidate))
                print(f"[config] {attr} -> {candidate}")
                break


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_models(config, device):
    mode = getattr(config, "lut_mode", "full")
    lut_dim = lut_dim_for_mode(mode)
    num_luts = getattr(config, "num_luts", 1)

    lut_model = Multi4DLUT(
        num_luts=num_luts,
        bins=getattr(config, "lut_bins", 17),
        output_channels=getattr(config, "lut_output_channels", 3),
        lut_dim=lut_dim,
    ).to(device)

    context_gen = None
    if mode == "full" and getattr(config, "use_dual_encoder", False):
        context_gen = DualBranchContextEncoder(
            in_channels=context_channels_for_mode(mode, dual_branch=True),
            hidden_dim=getattr(config, "context_hidden_dim", 32),
            num_layers=getattr(config, "context_num_layers", 3),
            num_luts=num_luts,
            predict_gate=getattr(config, "predict_depth_gate", False),
            pixel_fusion=getattr(config, "pixel_fusion", False),
            use_ppm=getattr(config, "use_ppm", False),
        ).to(device)
    else:
        ctx_ch = context_channels_for_mode(mode)
        if ctx_ch is not None:
            context_gen = ContextGenerator(
                in_channels=ctx_ch,
                hidden_dim=getattr(config, "context_hidden_dim", 16),
                num_layers=getattr(config, "context_num_layers", 3),
            ).to(device)

    refine = None
    if getattr(config, "use_local_refine", False):
        refine_type = getattr(config, "refine_type", "local")
        refine_cls = ResRefine if refine_type == "res" else LocalRefine
        refine = refine_cls(
            hidden_dim=getattr(config, "refine_hidden_dim", 16),
            num_layers=getattr(config, "refine_num_layers", 3),
        ).to(device)

    return lut_model, context_gen, refine


def build_optimizer(config, lut_model, context_gen, refine):
    param_groups = [
        {
            "params": list(lut_model.parameters()),
            "lr": getattr(config, "lr_lut", 2e-4),
            "name": "lut",
        }
    ]
    if context_gen is not None:
        param_groups.append(
            {
                "params": list(context_gen.parameters()),
                "lr": getattr(config, "lr_context", 5e-4),
                "name": "context",
            }
        )
    if refine is not None:
        param_groups.append(
            {
                "params": list(refine.parameters()),
                "lr": getattr(config, "lr_context", 5e-4),
                "name": "refine",
            }
        )
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=getattr(config, "weight_decay", 1e-5),
    )
    for group in optimizer.param_groups:
        group["base_lr"] = group["lr"]
    return optimizer


def build_scheduler(config, optimizer):
    warmup_epochs = int(getattr(config, "warmup_epochs", 0))
    milestones = list(getattr(config, "lr_milestones", []))
    gamma = getattr(config, "lr_gamma", 1.0)

    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    shifted_milestones = [
        max(0, int(m) - warmup_epochs) for m in milestones if int(m) > warmup_epochs
    ]
    main = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=shifted_milestones, gamma=gamma
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, main],
        milestones=[warmup_epochs],
    )


def make_criterion(config, device):
    return CombinedLoss(
        w_l1=getattr(config, "loss_l1_weight", 1.0),
        w_l2=getattr(config, "loss_l2_weight", 0.0),
        w_ssim=getattr(config, "loss_ssim_weight", 1.0),
        w_chroma=getattr(config, "loss_chroma_weight", 0.0),
        w_tv=getattr(config, "loss_tv_weight", 5e-5),
        w_mn=getattr(config, "loss_mn_weight", 2.0),
        w_contrast=getattr(config, "loss_contrast_weight", 0.0),
        w_vgg=getattr(config, "loss_vgg_weight", 0.0),
        w_fft=getattr(config, "loss_fft_weight", 0.0),
        w_grad=getattr(config, "loss_grad_weight", 0.0),
        w_cb=getattr(config, "loss_cb_weight", 1.5),
        w_cr=getattr(config, "loss_cr_weight", 1.5),
        w_cdi_range=getattr(config, "loss_cdi_range_weight", 0.0),
        w_cdi_decor=getattr(config, "loss_cdi_decor_weight", 0.0),
        mn_relax=getattr(config, "loss_mn_relax", False),
        bins=getattr(config, "lut_bins", 17),
        output_channels=getattr(config, "lut_output_channels", 3),
    ).to(device)


def move_batch(batch, device):
    rgb, grad, depth, target, global_stats = batch
    return (
        rgb.to(device, non_blocking=True),
        grad.to(device, non_blocking=True),
        depth.to(device, non_blocking=True),
        target.to(device, non_blocking=True),
        global_stats.to(device, non_blocking=True),
    )


def forward_model(rgb, depth, global_stats, lut_model, context_gen, refine, mode,
                  index_remap=None):
    luts, weights = lut_model()
    pred_rgb, aux = apply_multi_4d_lut(
        rgb,
        depth,
        luts,
        weights,
        context_gen,
        mode=mode,
        global_stats=global_stats,
        index_remap=index_remap,
        return_aux=True,
    )
    pred_ycbcr = None
    if refine is not None:
        pred_ycbcr = refine(rgb_to_ycbcr(pred_rgb), rgb_to_ycbcr(rgb))
        pred_rgb = ycbcr_to_rgb(pred_ycbcr)
    return pred_rgb, pred_ycbcr, luts, aux


def apply_training_ablation(config, rgb, pred_rgb, luts, context_gen, mode, depth, global_stats):
    """Re-run only for ablation flags that are also recorded by inference.py checkpoints."""
    fix_deg = getattr(config, "fix_deg_channel", None)
    use_adaptive = getattr(config, "use_adaptive_weights", True)
    if mode != "full" or not getattr(context_gen, "is_dual_branch", False):
        return pred_rgb
    if fix_deg is None and use_adaptive:
        return pred_rgb

    from utils.interpolation import compute_gradient, quadrilinear_interpolation

    ycbcr = rgb_to_ycbcr(rgb)
    y = ycbcr[:, 0:1]
    cb = ycbcr[:, 1:2]
    cr = ycbcr[:, 2:3]
    grad = compute_gradient(y)
    ctx_input = torch.cat([y, cb, cr, depth, grad], dim=1)
    deg_map, adaptive_w = context_gen(ctx_input, global_stats=global_stats)

    if fix_deg in ("A", "AB"):
        deg_map = torch.cat([cb, deg_map[:, 1:2]], dim=1)
    if fix_deg in ("B", "AB"):
        deg_map = torch.cat([deg_map[:, 0:1], cr], dim=1)
    if not use_adaptive:
        adaptive_w = torch.ones_like(adaptive_w) / adaptive_w.size(1)

    shape = luts[0].shape
    clamp_idx = lambda x, b: torch.clamp(x * (b - 1), 0, b - 1 - 1e-5)
    d0 = clamp_idx(y.squeeze(1), shape[0])
    d1 = clamp_idx(depth.squeeze(1), shape[1])
    d2 = clamp_idx(deg_map[:, 0], shape[2])
    d3 = clamp_idx(deg_map[:, 1], shape[3])
    delta = None
    for i, lut in enumerate(luts):
        val = quadrilinear_interpolation(d0, d1, d2, d3, lut)
        if adaptive_w.dim() == 4:
            w = adaptive_w[:, i:i + 1]
        else:
            w = adaptive_w[:, i].view(-1, 1, 1, 1)
        delta = val * w if delta is None else delta + val * w
    return ycbcr_to_rgb(ycbcr + delta)


def train_one_epoch(loader, models, criterion, optimizer, scaler, config, device,
                    index_remap=None):
    lut_model, context_gen, refine = models
    lut_model.train()
    if context_gen is not None:
        context_gen.train()
    if refine is not None:
        refine.train()

    use_amp = bool(getattr(config, "mixed_precision", False)) and device.type == "cuda"
    mode = getattr(config, "lut_mode", "full")
    totals = {}
    n = 0

    for batch in loader:
        rgb, _grad, depth, target, global_stats = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            pred_rgb, pred_ycbcr, luts, aux = forward_model(
                rgb, depth, global_stats, lut_model, context_gen, refine, mode,
                index_remap=index_remap,
            )
            pred_rgb = apply_training_ablation(
                config, rgb, pred_rgb, luts, context_gen, mode, depth, global_stats
            )
            if refine is not None and (getattr(config, "fix_deg_channel", None) is not None or not getattr(config, "use_adaptive_weights", True)):
                pred_ycbcr = refine(rgb_to_ycbcr(pred_rgb), rgb_to_ycbcr(rgb))
                pred_rgb = ycbcr_to_rgb(pred_ycbcr)
            loss, loss_dict = criterion(
                pred_rgb, target, luts, pred_ycbcr=pred_ycbcr,
                deg_map=aux.get("deg_map"),
            )

        if use_amp:
            scaler.scale(loss).backward()
            if getattr(config, "grad_clip", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters(models), getattr(config, "grad_clip", 1.0)
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if getattr(config, "grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters(models), getattr(config, "grad_clip", 1.0)
                )
            optimizer.step()

        batch_size = rgb.size(0)
        n += batch_size
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size

    return {key: value / max(n, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(loader, models, criterion, config, device, index_remap=None):
    lut_model, context_gen, refine = models
    lut_model.eval()
    if context_gen is not None:
        context_gen.eval()
    if refine is not None:
        refine.eval()

    mode = getattr(config, "lut_mode", "full")
    totals = {}
    n = 0
    mse_sum = 0.0

    for batch in loader:
        rgb, _grad, depth, target, global_stats = move_batch(batch, device)
        pred_rgb, pred_ycbcr, luts, aux = forward_model(
            rgb, depth, global_stats, lut_model, context_gen, refine, mode,
            index_remap=index_remap,
        )
        pred_rgb = apply_training_ablation(
            config, rgb, pred_rgb, luts, context_gen, mode, depth, global_stats
        )
        if refine is not None and (getattr(config, "fix_deg_channel", None) is not None or not getattr(config, "use_adaptive_weights", True)):
            pred_ycbcr = refine(rgb_to_ycbcr(pred_rgb), rgb_to_ycbcr(rgb))
            pred_rgb = ycbcr_to_rgb(pred_ycbcr)
        loss, loss_dict = criterion(
            pred_rgb, target, luts, pred_ycbcr=pred_ycbcr,
            deg_map=aux.get("deg_map"),
        )

        batch_size = rgb.size(0)
        n += batch_size
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        mse = torch.mean((pred_rgb - target) ** 2, dim=(1, 2, 3))
        mse_sum += float(mse.sum().item())

    result = {key: value / max(n, 1) for key, value in totals.items()}
    avg_mse = mse_sum / max(n, 1)
    result["psnr"] = -10.0 * np.log10(max(avg_mse, 1e-12))
    return result


@torch.no_grad()
def auth_validate(models, config, device, index_remap=None):
    """Validate native-resolution outputs with the release evaluation protocol."""
    import cv2

    lut_model, context_gen, refine = models
    lut_model.eval()
    if context_gen is not None:
        context_gen.eval()
    if refine is not None:
        refine.eval()

    mode = getattr(config, "lut_mode", "full")
    ds = config.dataset_1600
    with open(config.test_list_file) as f:
        names = [line.strip() for line in f if line.strip()]

    luts, weights = lut_model()
    psnrs = []
    for name in names:
        bgr = cv2.imread(os.path.join(ds["input_rgb"], name), cv2.IMREAD_COLOR)
        dep = cv2.imread(os.path.join(ds["input_depth"], name), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(ds["target"], name), cv2.IMREAD_COLOR)
        if bgr is None or dep is None or gt is None:
            continue
        rgb = torch.from_numpy(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        ).float().div_(255.0).unsqueeze(0).to(device)
        depth = torch.from_numpy(dep).float().div_(255.0).unsqueeze(0).unsqueeze(0).to(device)

        pred = apply_multi_4d_lut(
            rgb, depth, luts, weights, context_gen, mode=mode, index_remap=index_remap
        )
        if refine is not None:
            pred = ycbcr_to_rgb(refine(rgb_to_ycbcr(pred), rgb_to_ycbcr(rgb)))

        pred_u8 = pred.squeeze(0).mul(255).clamp_(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        pred_bgr = pred_u8[:, :, ::-1]
        pred_256 = cv2.resize(pred_bgr, (256, 256)).astype(np.float32) / 255.0
        gt_256 = cv2.resize(gt, (256, 256)).astype(np.float32) / 255.0
        rmse = float(np.sqrt(np.mean((pred_256 - gt_256) ** 2)))
        psnrs.append(20.0 * np.log10(1.0 / max(rmse, 1e-12)))

    return float(np.mean(psnrs)) if psnrs else 0.0


def trainable_parameters(models):
    for model in models:
        if model is None:
            continue
        yield from model.parameters()


def save_config_txt(config, out_dir):
    with open(out_dir / "config.txt", "w", encoding="utf-8") as f:
        for key in sorted(k for k in dir(config) if not k.startswith("_")):
            value = getattr(config, key)
            if callable(value):
                continue
            f.write(f"{key}: {value}\n")


def checkpoint_dict(config, epoch, best_val, best_psnr, models, optimizer=None, scheduler=None):
    lut_model, context_gen, refine = models
    context_state = context_gen.state_dict() if context_gen is not None else None
    ctx_type = "none"
    ctx_in = None
    ctx_num_luts = getattr(config, "num_luts", 1)
    if context_gen is not None:
        ctx_type = "dual_branch" if getattr(context_gen, "is_dual_branch", False) else "single"
        ctx_in = getattr(context_gen, "in_channels", None)
        ctx_num_luts = getattr(context_gen, "num_luts", ctx_num_luts)

    state = {
        "epoch": epoch,
        "best_val_loss": best_val,
        "best_val_psnr": best_psnr,
        "lut_mode": getattr(config, "lut_mode", "full"),
        "lut_model_state": lut_model.state_dict(),
        "context_gen_state": context_state,
        "context_gen_type": ctx_type,
        "context_gen_hidden": getattr(config, "context_hidden_dim", 16),
        "context_gen_layers": getattr(config, "context_num_layers", 3),
        "context_gen_num_luts": ctx_num_luts,
        "context_gen_in_channels": ctx_in,
        "use_local_refine": refine is not None,
        "refine_state": refine.state_dict() if refine is not None else None,
        "refine_hidden_dim": getattr(config, "refine_hidden_dim", 16),
        "refine_num_layers": getattr(config, "refine_num_layers", 3),
        "refine_type": getattr(config, "refine_type", "local"),
        "fix_deg_channel": getattr(config, "fix_deg_channel", None),
        "use_adaptive_weights": getattr(config, "use_adaptive_weights", True),
        # Optional model features are stored for automatic reconstruction at inference.
        "lut_bins_per_axis": getattr(lut_model, "bins_per_axis", None),
        "predict_depth_gate": getattr(config, "predict_depth_gate", False),
        "index_remap": getattr(config, "_index_remap_cpu", None),
        "ctx_downsample": getattr(config, "ctx_downsample", None),
        "pixel_fusion": getattr(config, "pixel_fusion", False),
        "use_ppm": getattr(config, "use_ppm", False),
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    return state


def export_best_parts(out_dir, models):
    lut_model, context_gen, refine = models
    luts, _weights = lut_model()
    for i, lut in enumerate(luts):
        np.save(out_dir / f"best_lut_{i}.npy", lut.detach().cpu().numpy())
    if context_gen is not None:
        torch.save(context_gen.state_dict(), out_dir / "best_context_gen.pth")
    if refine is not None:
        torch.save(refine.state_dict(), out_dir / "best_refine.pth")


def save_checkpoint(path, config, epoch, best_val, best_psnr, models, optimizer=None, scheduler=None, export_parts=False):
    state = checkpoint_dict(config, epoch, best_val, best_psnr, models, optimizer, scheduler)
    torch.save(state, path)
    if export_parts:
        export_best_parts(path.parent, models)


def load_model_weights(path, models, optimizer=None, scheduler=None, resume=False, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=False)
    lut_model, context_gen, refine = models

    if "lut_model_state" in state:
        lut_model.load_state_dict(state["lut_model_state"])
    else:
        lut_model.load_state_dict(state)

    if context_gen is not None and state.get("context_gen_state") is not None:
        context_gen.load_state_dict(state["context_gen_state"])
    if refine is not None and state.get("refine_state") is not None:
        refine.load_state_dict(state["refine_state"])

    if resume and optimizer is not None and state.get("optimizer_state") is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    if resume and scheduler is not None and state.get("scheduler_state") is not None:
        scheduler.load_state_dict(state["scheduler_state"])

    start_epoch = int(state.get("epoch", -1)) + 1 if resume else 0
    best_val = float(state.get("best_val_loss", float("inf"))) if resume else float("inf")
    best_psnr = float(state.get("best_val_psnr", 0.0)) if resume else 0.0
    return start_epoch, best_val, best_psnr


def fmt_metrics(metrics):
    keys = ["total", "l1_y", "l1_cb", "l1_cr", "ssim", "tv", "mn", "vgg", "l2", "grad", "cdi_rng", "cdi_dec", "psnr"]
    parts = []
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return " ".join(parts)


def main():
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    set_seed(getattr(config, "seed", 42))

    requested_device = getattr(config, "device", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; using CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)

    out_dir = Path(getattr(config, "checkpoint_dir", "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config_txt(config, out_dir)

    print(f"[device] {device}")
    print(f"[output] {out_dir}")
    print(f"[mode] lut_mode={config.lut_mode} num_luts={config.num_luts} bins={config.lut_bins}")

    train_loader = create_mixed_dataloader(config, split="train")
    val_loader = create_mixed_dataloader(config, split="val")
    if len(train_loader) == 0:
        raise RuntimeError("Train dataloader is empty. Check dataset paths, split files, and batch_size.")

    models = build_models(config, device)
    optimizer = build_optimizer(config, *models)
    scheduler = build_scheduler(config, optimizer)
    scaler = GradScaler(enabled=bool(getattr(config, "mixed_precision", False)) and device.type == "cuda")
    criterion = make_criterion(config, device)

    # Optional fixed index-remapping tables.
    index_remap = None
    remap_path = getattr(config, "index_remap_path", None)
    if remap_path:
        index_remap = load_index_remap(remap_path, device)
        config._index_remap_cpu = {k: v.cpu() for k, v in index_remap.items()}
        print(f"[index_remap] loaded from {remap_path}: axes={list(index_remap.keys())}")

    start_epoch = 0
    best_val = float("inf")
    best_psnr = 0.0
    if args.resume:
        start_epoch, best_val, best_psnr = load_model_weights(
            args.resume, models, optimizer=optimizer, scheduler=scheduler, resume=True, device=device
        )
        print(f"[resume] {args.resume} from epoch index {start_epoch}")
    elif args.finetune:
        load_model_weights(args.finetune, models, resume=False, device=device)
        print(f"[finetune] loaded weights from {args.finetune}")

    patience = getattr(config, "early_stopping_patience", 0)
    min_delta = getattr(config, "min_delta", 0.0)
    bad_epochs = 0
    num_epochs = getattr(config, "num_epochs", 1)
    save_interval = getattr(config, "save_interval", 10)
    auth_interval = int(getattr(config, "auth_val_interval", 0))
    best_auth = 0.0

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, num_epochs):
        last_epoch = epoch
        epoch_start = time.time()
        lr_text = ", ".join(f"{g.get('name', 'group')}={g['lr']:.2e}" for g in optimizer.param_groups)

        train_metrics = train_one_epoch(train_loader, models, criterion, optimizer, scaler, config, device, index_remap=index_remap)
        val_metrics = validate(val_loader, models, criterion, config, device, index_remap=index_remap)
        val_loss = val_metrics.get("total", float("inf"))
        elapsed = time.time() - epoch_start

        print(
            f"[epoch {epoch + 1:04d}/{num_epochs}] {elapsed:.1f}s lr({lr_text}) "
            f"train: {fmt_metrics(train_metrics)} | val: {fmt_metrics(val_metrics)}"
        )

        val_psnr = val_metrics.get("psnr", 0.0)
        select_psnr = getattr(config, "select_metric", "loss") == "psnr"
        if select_psnr:
            improved = val_psnr > best_psnr + min_delta
        else:
            improved = val_loss < best_val - min_delta
        if improved:
            best_val = min(best_val, val_loss)
            best_psnr = max(best_psnr, val_psnr)
            bad_epochs = 0
        else:
            bad_epochs += 1

        scheduler.step()

        if improved:
            best_path = out_dir / "best_model.pth"
            save_checkpoint(best_path, config, epoch, best_val, best_psnr, models, optimizer, scheduler, export_parts=True)
            print(f"[save] best -> {best_path} val_total={best_val:.6f}")

        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            latest_path = out_dir / "latest_model.pth"
            save_checkpoint(latest_path, config, epoch, best_val, best_psnr, models, optimizer, scheduler)
            interval_path = out_dir / f"epoch_{epoch + 1:04d}.pth"
            shutil.copy2(latest_path, interval_path)
            print(f"[save] latest -> {latest_path}")

        # Select a separate checkpoint under the native-resolution evaluation path.
        if auth_interval > 0 and ((epoch + 1) % auth_interval == 0 or epoch == num_epochs - 1):
            t_auth = time.time()
            auth_psnr = auth_validate(models, config, device, index_remap=index_remap)
            print(f"[auth_val] epoch {epoch + 1}: PSNR={auth_psnr:.4f} ({time.time() - t_auth:.0f}s)")
            if auth_psnr > best_auth:
                best_auth = auth_psnr
                save_checkpoint(out_dir / "best_auth_model.pth", config, epoch, best_val, auth_psnr, models)
                print(f"[save] best_auth -> {out_dir / 'best_auth_model.pth'} auth_psnr={auth_psnr:.4f}")

        if patience > 0 and bad_epochs >= patience:
            print(f"[early_stop] no improvement for {bad_epochs} epochs")
            break

    latest_path = out_dir / "latest_model.pth"
    save_checkpoint(latest_path, config, last_epoch, best_val, best_psnr, models, optimizer, scheduler)
    print(f"[done] best_val_total={best_val:.6f}")


if __name__ == "__main__":
    main()
