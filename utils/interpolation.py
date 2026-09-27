#!/usr/bin/env python3
"""LUT interpolation and inference utilities.

Supported modes include RGB and YCbCr lookup tables with optional depth or context
coordinates. The implementation supports independent bin counts per axis, fixed
monotonic index-remapping tables, an optional depth-reliability gate, and optional
context-encoder downsampling.
"""

import torch
import torch.nn.functional as F

_SOBEL_CACHE = {}

LUT_MODE_FULL         = 'full'
LUT_MODE_RGB3D        = 'rgb3d'
LUT_MODE_RGB_DEPTH    = 'rgb_depth'
LUT_MODE_RGB_CONTEXT  = 'rgb_context'
LUT_MODE_YCBCR_DEPTH  = 'ycbcr_depth'
LUT_MODE_YCBCR_CONTEXT = 'ycbcr_context'
LUT_MODE_YCBCR_2D     = 'ycbcr2d'

ALL_LUT_MODES = (
    LUT_MODE_FULL, LUT_MODE_RGB3D, LUT_MODE_RGB_DEPTH, LUT_MODE_RGB_CONTEXT,
    LUT_MODE_YCBCR_DEPTH, LUT_MODE_YCBCR_CONTEXT, LUT_MODE_YCBCR_2D,
)


def lut_dim_for_mode(mode):
    if mode in (LUT_MODE_RGB3D, LUT_MODE_YCBCR_DEPTH, LUT_MODE_YCBCR_CONTEXT):
        return 3
    if mode == LUT_MODE_YCBCR_2D:
        return 2
    return 4


def context_channels_for_mode(mode, dual_branch=False):
    """Return the context-encoder input width, or None when no encoder is needed."""
    if mode == LUT_MODE_FULL:
        return 5 if dual_branch else 6
    if mode == LUT_MODE_RGB_CONTEXT:
        return 4          # RGB(3) + Grad(1)
    if mode == LUT_MODE_YCBCR_CONTEXT:
        return 5          # RGB(3) + Grad(1) + Cr(1)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 色彩空间
# ─────────────────────────────────────────────────────────────────────────────

def rgb_to_ycbcr(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y  =  0.299   * r + 0.587   * g + 0.114   * b
    cb = 128./256. - 0.168736 * r - 0.331264 * g + 0.5     * b
    cr = 128./256. + 0.5      * r - 0.418688 * g - 0.081312 * b
    return torch.stack([y, cb, cr], dim=1)


def ycbcr_to_rgb(ycbcr):
    y, cb, cr = ycbcr[:, 0], ycbcr[:, 1], ycbcr[:, 2]
    r = y + (cr - 0.5) * 1.402
    g = y - (cb - 0.5) * 0.344136 - (cr - 0.5) * 0.714136
    b = y + (cb - 0.5) * 1.772
    return torch.clamp(torch.stack([r, g, b], dim=1), 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 梯度
# ─────────────────────────────────────────────────────────────────────────────

def _get_sobel_kernels(device):
    key = device
    if key not in _SOBEL_CACHE:
        sx = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3).to(device)
        sy = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3).to(device)
        _SOBEL_CACHE[key] = (sx, sy)
    return _SOBEL_CACHE[key]


def compute_gradient(y_channel):
    """y_channel: [B,1,H,W] → gradient: [B,1,H,W] ∈ [0,1]"""
    sobel_x, sobel_y = _get_sobel_kernels(str(y_channel.device))
    grad_x = F.conv2d(y_channel, sobel_x, padding=1)
    grad_y = F.conv2d(y_channel, sobel_y, padding=1)
    gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    min_val = gradient.flatten(2).min(dim=2, keepdim=True)[0].unsqueeze(-1)
    max_val = gradient.flatten(2).max(dim=2, keepdim=True)[0].unsqueeze(-1)
    return (gradient - min_val) / (max_val - min_val + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Fixed index remapping.
# ─────────────────────────────────────────────────────────────────────────────

def apply_remap_1d(x, table):
    """用固定单调表对 [0,1] 值做重映射（线性插值）。

    Args:
        x:     任意形状 tensor ∈ [0,1]
        table: [T] 单调递增 tensor，table[0]=0, table[-1]=1
    Returns:
        重映射后的 tensor，形状同 x
    """
    if table is None:
        return x
    T = table.shape[0]
    pos = torch.clamp(x, 0, 1) * (T - 1)
    lo = torch.clamp(pos.floor().long(), 0, T - 1)
    hi = torch.clamp(lo + 1, 0, T - 1)
    frac = pos - lo.float()
    return table[lo] * (1 - frac) + table[hi] * frac


def load_index_remap(remap, device):
    """remap: None / dict{'y':1D array,'d':1D array} / npz 路径。返回 dict of tensors 或 None。"""
    if remap is None:
        return None
    if isinstance(remap, str):
        import numpy as np
        data = np.load(remap)
        remap = {k: data[k] for k in data.files}
    out = {}
    for key, val in remap.items():
        t = val if isinstance(val, torch.Tensor) else torch.as_tensor(val, dtype=torch.float32)
        out[key] = t.to(device).float()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 插值核心（逐轴 bins：一律从 lut.shape 读取）
# ─────────────────────────────────────────────────────────────────────────────

def _floor_ceil_alpha(idx, bins):
    fl = torch.clamp(torch.floor(idx).long(), 0, bins - 1)
    ce = torch.clamp(fl + 1, 0, bins - 1)
    alpha = (idx - fl.float()).unsqueeze(-1)
    return fl, ce, alpha


def _lerp(v1, v2, a):
    return v1 * (1 - a) + v2 * a


def bilinear_interpolation(d0, d1, lut):
    f0, c0, a0 = _floor_ceil_alpha(d0, lut.shape[0])
    f1, c1, a1 = _floor_ceil_alpha(d1, lut.shape[1])
    result = _lerp(
        _lerp(lut[f0, f1], lut[f0, c1], a1),
        _lerp(lut[c0, f1], lut[c0, c1], a1),
        a0)
    return result.permute(0, 3, 1, 2)


def trilinear_interpolation(d0, d1, d2, lut):
    f0, c0, a0 = _floor_ceil_alpha(d0, lut.shape[0])
    f1, c1, a1 = _floor_ceil_alpha(d1, lut.shape[1])
    f2, c2, a2 = _floor_ceil_alpha(d2, lut.shape[2])
    result = _lerp(
        _lerp(
            _lerp(lut[f0, f1, f2], lut[f0, f1, c2], a2),
            _lerp(lut[f0, c1, f2], lut[f0, c1, c2], a2),
            a1),
        _lerp(
            _lerp(lut[c0, f1, f2], lut[c0, f1, c2], a2),
            _lerp(lut[c0, c1, f2], lut[c0, c1, c2], a2),
            a1),
        a0)
    return result.permute(0, 3, 1, 2)


def quadrilinear_interpolation(d0, d1, d2, d3, lut):
    f0, c0, a0 = _floor_ceil_alpha(d0, lut.shape[0])
    f1, c1, a1 = _floor_ceil_alpha(d1, lut.shape[1])
    f2, c2, a2 = _floor_ceil_alpha(d2, lut.shape[2])
    f3, c3, a3 = _floor_ceil_alpha(d3, lut.shape[3])
    result = _lerp(
        _lerp(
            _lerp(
                _lerp(lut[f0, f1, f2, f3], lut[f0, f1, f2, c3], a3),
                _lerp(lut[f0, f1, c2, f3], lut[f0, f1, c2, c3], a3),
                a2),
            _lerp(
                _lerp(lut[f0, c1, f2, f3], lut[f0, c1, f2, c3], a3),
                _lerp(lut[f0, c1, c2, f3], lut[f0, c1, c2, c3], a3),
                a2),
            a1),
        _lerp(
            _lerp(
                _lerp(lut[c0, f1, f2, f3], lut[c0, f1, f2, c3], a3),
                _lerp(lut[c0, f1, c2, f3], lut[c0, f1, c2, c3], a3),
                a2),
            _lerp(
                _lerp(lut[c0, c1, f2, f3], lut[c0, c1, f2, c3], a3),
                _lerp(lut[c0, c1, c2, f3], lut[c0, c1, c2, c3], a3),
                a2),
            a1),
        a0)
    return result.permute(0, 3, 1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 加权求和（静态权重 [K] 和自适应权重 [B,K] 两个版本）
# ─────────────────────────────────────────────────────────────────────────────

def _clamp_idx(x, bins):
    return torch.clamp(x * (bins - 1), 0, bins - 1 - 1e-5)


def _weighted_interp(luts, weights, interp_fn, *indices):
    """静态权重: weights [K]，全局共享"""
    if len(luts) == 1:
        return interp_fn(*indices, luts[0])
    out = None
    for i, lut in enumerate(luts):
        val = interp_fn(*indices, lut)
        out = val * weights[i] if out is None else out + val * weights[i]
    return out


def _weighted_interp_adaptive(luts, weights_bk, interp_fn, *indices):
    """
    自适应权重:
      weights_bk [B, K]       — 图像级融合（每图一个系数，原版行为）
      weights_bk [B, K, H, W] — S1 逐像素融合（每像素独立的凸组合）
    每个 LUT 结果 [B,C,H,W] 乘以对应权重后累加。
    """
    if len(luts) == 1:
        return interp_fn(*indices, luts[0])
    per_pixel = weights_bk.dim() == 4
    out = None
    for i, lut in enumerate(luts):
        val = interp_fn(*indices, lut)                   # [B, C, H, W]
        if per_pixel:
            w = weights_bk[:, i:i + 1]                   # [B, 1, H, W]
        else:
            w = weights_bk[:, i].view(-1, 1, 1, 1)       # [B, 1, 1, 1]
        out = val * w if out is None else out + val * w
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Unified LUT application entry point.
# ─────────────────────────────────────────────────────────────────────────────

def apply_multi_4d_lut(rgb, depth, luts, weights, context_gen, mode='full',
                       global_stats=None, use_grad=True,
                       index_remap=None, ctx_downsample=None, return_aux=False):
    """Apply one or more LUTs and optionally return encoder diagnostics.

    DualBranchContextEncoder uses (Y, depth, learned-A, learned-B) coordinates and
    adaptive weights. ContextGenerator uses (Y, Cb, depth, context) coordinates and
    the supplied static weights.
    """
    use_dual = getattr(context_gen, 'is_dual_branch', False)

    # Adaptive indices and weights.
    if mode == LUT_MODE_FULL and use_dual:
        ycbcr = rgb_to_ycbcr(rgb)
        y  = ycbcr[:, 0:1]
        cb = ycbcr[:, 1:2]
        cr = ycbcr[:, 2:3]

        grad = compute_gradient(y)

        if use_grad:
            ctx_input = torch.cat([y, cb, cr, depth, grad], dim=1)
        else:
            ctx_input = torch.cat([y, cb, cr, depth], dim=1)

        # Optionally run the context encoder at reduced resolution.
        H, W = ctx_input.shape[2], ctx_input.shape[3]
        if ctx_downsample is not None and max(H, W) > ctx_downsample:
            scale = ctx_downsample / max(H, W)
            small = F.interpolate(ctx_input, scale_factor=scale, mode='bilinear',
                                  align_corners=False, recompute_scale_factor=False)
            deg_map, adaptive_w = context_gen(small, global_stats=global_stats)
            deg_map = F.interpolate(deg_map, size=(H, W), mode='bilinear',
                                    align_corners=False)
            # Upsample pixel-level weights with the degradation coordinates.
            if adaptive_w.dim() == 4:
                adaptive_w = F.interpolate(adaptive_w, size=(H, W), mode='bilinear',
                                           align_corners=False)
        else:
            deg_map, adaptive_w = context_gen(ctx_input, global_stats=global_stats)
        # deg_map: [B,2,H,W] 或 [B,3,H,W]（第 3 通道为深度可靠性门 α）

        # Fixed index remapping.
        y_idx = y.squeeze(1)
        d_idx = depth.squeeze(1)
        if index_remap is not None:
            y_idx = apply_remap_1d(y_idx, index_remap.get('y'))
            d_idx = apply_remap_1d(d_idx, index_remap.get('d'))

        # LUT 索引: (Y, Depth, LearnedA, LearnedB)，逐轴 bins
        shape = luts[0].shape
        d0 = _clamp_idx(y_idx,        shape[0])
        d1 = _clamp_idx(d_idx,        shape[1])
        d2 = _clamp_idx(deg_map[:, 0], shape[2])
        d3 = _clamp_idx(deg_map[:, 1], shape[3])

        delta_ycbcr = _weighted_interp_adaptive(
            luts, adaptive_w, quadrilinear_interpolation, d0, d1, d2, d3
        )

        # Optional depth-reliability gate.
        gate = None
        if deg_map.shape[1] >= 3:
            gate = deg_map[:, 2:3]                          # [B,1,H,W] ∈ [0,1]
            d_mean = depth.mean(dim=(2, 3), keepdim=True).expand_as(depth)
            dm_idx = d_mean.squeeze(1)
            if index_remap is not None:
                dm_idx = apply_remap_1d(dm_idx, index_remap.get('d'))
            d1_mean = _clamp_idx(dm_idx, shape[1])
            delta_mean = _weighted_interp_adaptive(
                luts, adaptive_w, quadrilinear_interpolation, d0, d1_mean, d2, d3
            )
            delta_ycbcr = gate * delta_ycbcr + (1 - gate) * delta_mean

        enhanced = ycbcr_to_rgb(ycbcr + delta_ycbcr)
        if return_aux:
            aux = {'deg_map': deg_map[:, 0:2], 'lut_weights': adaptive_w, 'gate': gate}
            return enhanced, aux
        return enhanced

    # Compatibility path with fixed indices and static weights.
    if mode == LUT_MODE_FULL:
        ycbcr = rgb_to_ycbcr(rgb)
        y  = ycbcr[:, 0:1]
        cb = ycbcr[:, 1:2]
        cr = ycbcr[:, 2:3]

        grad = compute_gradient(y)
        context = context_gen(torch.cat([rgb, grad, depth, cr], dim=1))

        shape = luts[0].shape
        d0 = _clamp_idx(y.squeeze(1),       shape[0])
        d1 = _clamp_idx(cb.squeeze(1),      shape[1])
        d2 = _clamp_idx(depth.squeeze(1),   shape[2])
        d3 = _clamp_idx(context.squeeze(1), shape[3])

        delta_ycbcr = _weighted_interp(
            luts, weights, quadrilinear_interpolation, d0, d1, d2, d3
        )
        out = ycbcr_to_rgb(ycbcr + delta_ycbcr)
        return (out, {}) if return_aux else out

    # ── rgb3d ─────────────────────────────────────────────────────────────────
    if mode == LUT_MODE_RGB3D:
        shape = luts[0].shape
        d0 = _clamp_idx(rgb[:, 0], shape[0])
        d1 = _clamp_idx(rgb[:, 1], shape[1])
        d2 = _clamp_idx(rgb[:, 2], shape[2])
        delta_rgb = _weighted_interp(luts, weights, trilinear_interpolation, d0, d1, d2)
        out = torch.clamp(rgb + delta_rgb, 0, 1)
        return (out, {}) if return_aux else out

    # ── rgb_depth ─────────────────────────────────────────────────────────────
    if mode == LUT_MODE_RGB_DEPTH:
        shape = luts[0].shape
        d0 = _clamp_idx(rgb[:, 0], shape[0])
        d1 = _clamp_idx(rgb[:, 1], shape[1])
        d2 = _clamp_idx(rgb[:, 2], shape[2])
        d3 = _clamp_idx(depth.squeeze(1), shape[3])
        delta_rgb = _weighted_interp(luts, weights, quadrilinear_interpolation, d0, d1, d2, d3)
        out = torch.clamp(rgb + delta_rgb, 0, 1)
        return (out, {}) if return_aux else out

    # ── rgb_context ───────────────────────────────────────────────────────────
    if mode == LUT_MODE_RGB_CONTEXT:
        y = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        grad = compute_gradient(y)
        context = context_gen(torch.cat([rgb, grad], dim=1))
        shape = luts[0].shape
        d0 = _clamp_idx(rgb[:, 0], shape[0])
        d1 = _clamp_idx(rgb[:, 1], shape[1])
        d2 = _clamp_idx(rgb[:, 2], shape[2])
        d3 = _clamp_idx(context.squeeze(1), shape[3])
        delta_rgb = _weighted_interp(luts, weights, quadrilinear_interpolation, d0, d1, d2, d3)
        out = torch.clamp(rgb + delta_rgb, 0, 1)
        return (out, {}) if return_aux else out

    # ── ycbcr_depth ───────────────────────────────────────────────────────────
    if mode == LUT_MODE_YCBCR_DEPTH:
        ycbcr = rgb_to_ycbcr(rgb)
        shape = luts[0].shape
        d0 = _clamp_idx(ycbcr[:, 0], shape[0])
        d1 = _clamp_idx(ycbcr[:, 1], shape[1])
        d2 = _clamp_idx(depth.squeeze(1), shape[2])
        delta_ycbcr = _weighted_interp(luts, weights, trilinear_interpolation, d0, d1, d2)
        out = ycbcr_to_rgb(ycbcr + delta_ycbcr)
        return (out, {}) if return_aux else out

    # ── ycbcr_context ─────────────────────────────────────────────────────────
    if mode == LUT_MODE_YCBCR_CONTEXT:
        ycbcr = rgb_to_ycbcr(rgb)
        y  = ycbcr[:, 0:1]
        cr = ycbcr[:, 2:3]
        grad = compute_gradient(y)
        context = context_gen(torch.cat([rgb, grad, cr], dim=1))
        shape = luts[0].shape
        d0 = _clamp_idx(ycbcr[:, 0], shape[0])
        d1 = _clamp_idx(ycbcr[:, 1], shape[1])
        d2 = _clamp_idx(context.squeeze(1), shape[2])
        delta_ycbcr = _weighted_interp(luts, weights, trilinear_interpolation, d0, d1, d2)
        out = ycbcr_to_rgb(ycbcr + delta_ycbcr)
        return (out, {}) if return_aux else out

    # ── ycbcr2d ───────────────────────────────────────────────────────────────
    if mode == LUT_MODE_YCBCR_2D:
        ycbcr = rgb_to_ycbcr(rgb)
        shape = luts[0].shape
        d0 = _clamp_idx(ycbcr[:, 0], shape[0])
        d1 = _clamp_idx(ycbcr[:, 1], shape[1])
        delta_ycbcr = _weighted_interp(luts, weights, bilinear_interpolation, d0, d1)
        out = ycbcr_to_rgb(ycbcr + delta_ycbcr)
        return (out, {}) if return_aux else out

    raise ValueError(f"未知的 lut_mode: {mode}")


def apply_4d_lut_inference(rgb, depth, lut, context_gen, mode='full'):
    """推理用简化接口（单 LUT）"""
    return apply_multi_4d_lut(
        rgb, depth, [lut], torch.ones(1).to(rgb.device), context_gen, mode=mode
    )


# ─────────────────────────────────────────────────────────────────────────────
# 高速推理封装
# ─────────────────────────────────────────────────────────────────────────────

# 自动档阈值：(最长边下限, 对应缩放比)，从大到小检测
_AUTO_SCALE_THRESHOLDS = [
    (2880, 0.25),   # ≥ 2880px (≈3K/4K)  → 1/4
    (1920, 0.50),   # ≥ 1920px (≈1080p)  → 1/2
]


class FastLUTInference:
    """Inference wrapper with pinned transfers and optional resolution scaling.

    `infer_scale` has highest priority, followed by `max_side`, then the automatic
    thresholds. Enhanced output is resized back to the original input dimensions.
    `index_remap` supplies optional Y/depth remapping tables, and `ctx_downsample`
    limits only the context encoder resolution.
    """

    def __init__(self, luts, context_gen, device='cuda', max_h=256, max_w=256,
                 mode='full', refine=None,
                 auto_scale=True, max_side=None, infer_scale=None,
                 index_remap=None, ctx_downsample=None):
        self.device = torch.device(device)
        # 兼容单 LUT（tensor）和多 LUT（list）
        if isinstance(luts, (list, tuple)):
            self.luts = luts
        else:
            self.luts = [luts]
        self.context_gen = context_gen
        self.refine = refine
        self.mode = mode
        self.auto_scale = auto_scale
        self.max_side = max_side
        self.infer_scale = infer_scale
        self.index_remap = load_index_remap(index_remap, self.device)
        self.ctx_downsample = ctx_downsample

        if self.device.type == 'cuda':
            self._rgb_pin = torch.empty(1, 3, max_h, max_w, dtype=torch.uint8).pin_memory()
            self._dep_pin = torch.empty(1, 1, max_h, max_w, dtype=torch.uint8).pin_memory()
            self._stream = torch.cuda.Stream(device=self.device)
        else:
            self._rgb_pin = torch.empty(1, 3, max_h, max_w, dtype=torch.uint8)
            self._dep_pin = torch.empty(1, 1, max_h, max_w, dtype=torch.uint8)
            self._stream = None

        self._weights = torch.ones(1, device=self.device)
        self.last_gpu_ms = None
        self.last_infer_scale = 1.0  # 记录每次实际使用的缩放比，方便外部查询

    def _compute_scale(self, h, w):
        """优先级：infer_scale > max_side > auto_scale > 不缩放。"""
        if self.infer_scale is not None:
            return float(self.infer_scale)
        longest = max(h, w)
        if self.max_side is not None:
            return (self.max_side / longest) if longest > self.max_side else 1.0
        if self.auto_scale:
            for threshold, scale in _AUTO_SCALE_THRESHOLDS:
                if longest >= threshold:
                    return scale
        return 1.0

    def _resize_pinned(self, h, w):
        if self._rgb_pin.shape[2] < h or self._rgb_pin.shape[3] < w:
            if self.device.type == 'cuda':
                self._rgb_pin = torch.empty(1, 3, h, w, dtype=torch.uint8).pin_memory()
                self._dep_pin = torch.empty(1, 1, h, w, dtype=torch.uint8).pin_memory()
            else:
                self._rgb_pin = torch.empty(1, 3, h, w, dtype=torch.uint8)
                self._dep_pin = torch.empty(1, 1, h, w, dtype=torch.uint8)

    def _prepare_tensors(self, rgb, depth):
        import numpy as np

        rgb_is_tensor = isinstance(rgb, torch.Tensor)
        dep_is_tensor = isinstance(depth, torch.Tensor) if depth is not None else False

        if rgb_is_tensor:
            rgb_t = rgb if rgb.is_cuda else rgb.to(self.device, non_blocking=True)
            dep_t = (depth if depth.is_cuda else depth.to(self.device, non_blocking=True)) \
                    if dep_is_tensor else None
        else:
            if depth is not None and len(depth.shape) == 3:
                import cv2
                depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

            h, w = rgb.shape[0], rgb.shape[1]
            self._resize_pinned(h, w)

            rgb_chw = np.ascontiguousarray(rgb[:, :, ::-1].transpose(2, 0, 1))
            self._rgb_pin[0, :, :h, :w].copy_(torch.from_numpy(rgb_chw))

            if self.device.type == 'cuda':
                with torch.cuda.stream(self._stream):
                    rgb_gpu = self._rgb_pin[:, :, :h, :w].to(self.device, non_blocking=True)
                    rgb_t = rgb_gpu.float().div_(255.0)

                    if depth is not None:
                        self._dep_pin[0, 0, :h, :w].copy_(torch.from_numpy(depth))
                        dep_gpu = self._dep_pin[:, :, :h, :w].to(self.device, non_blocking=True)
                        dep_t = dep_gpu.float().div_(255.0)
                    else:
                        dep_t = None
                self._stream.synchronize()
            else:
                rgb_t = self._rgb_pin[:, :, :h, :w].float().div_(255.0).to(self.device)
                if depth is not None:
                    self._dep_pin[0, 0, :h, :w].copy_(torch.from_numpy(depth))
                    dep_t = self._dep_pin[:, :, :h, :w].float().div_(255.0).to(self.device)
                else:
                    dep_t = None

        return rgb_t, dep_t

    def _forward(self, rgb_in, dep_in):
        enhanced = apply_multi_4d_lut(
            rgb_in, dep_in, self.luts, self._weights,
            self.context_gen, mode=self.mode,
            index_remap=self.index_remap, ctx_downsample=self.ctx_downsample,
        )
        if self.refine is not None:
            enhanced_ycbcr = rgb_to_ycbcr(enhanced)
            input_ycbcr   = rgb_to_ycbcr(rgb_in)
            refined_ycbcr  = self.refine(enhanced_ycbcr, input_ycbcr)
            enhanced = ycbcr_to_rgb(refined_ycbcr)
        return enhanced

    def enhance(self, rgb, depth=None):
        """
        Args:
            rgb:   np.ndarray(H,W,3) BGR uint8 OR torch.Tensor [1,3,H,W] float GPU
            depth: np.ndarray(H,W) uint8 OR torch.Tensor [1,1,H,W] float GPU
        Returns:
            np.ndarray(H,W,3) BGR uint8（分辨率与输入一致，自适应下采后会上采回来）
        """
        rgb_t, dep_t = self._prepare_tensors(rgb, depth)

        # ── 自适应下采样 ────────────────────────────────────────────
        orig_h, orig_w = rgb_t.shape[2], rgb_t.shape[3]
        scale = self._compute_scale(orig_h, orig_w)
        self.last_infer_scale = scale

        if scale < 1.0:
            infer_h = max(1, int(round(orig_h * scale)))
            infer_w = max(1, int(round(orig_w * scale)))
            rgb_in = torch.nn.functional.interpolate(
                rgb_t, (infer_h, infer_w), mode='bilinear', align_corners=False)
            dep_in = (torch.nn.functional.interpolate(
                dep_t, (infer_h, infer_w), mode='bilinear', align_corners=False)
                      if dep_t is not None else None)
        else:
            rgb_in, dep_in = rgb_t, dep_t
        # ────────────────────────────────────────────────────────────

        if rgb_t.is_cuda:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            with torch.no_grad():
                enhanced = self._forward(rgb_in, dep_in)
            end_evt.record()
            torch.cuda.synchronize(self.device)
            self.last_gpu_ms = float(start_evt.elapsed_time(end_evt))
        else:
            with torch.no_grad():
                enhanced = self._forward(rgb_in, dep_in)
            self.last_gpu_ms = None

        # ── 上采回原始分辨率 ─────────────────────────────────────────
        if scale < 1.0:
            enhanced = torch.nn.functional.interpolate(
                enhanced, (orig_h, orig_w), mode='bilinear', align_corners=False)
        # ────────────────────────────────────────────────────────────

        out = enhanced.squeeze(0).mul_(255).clamp_(0, 255).byte()
        out_np = out.permute(1, 2, 0).cpu().numpy()
        return out_np[:, :, ::-1].copy()
