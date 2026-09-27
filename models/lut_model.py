#!/usr/bin/env python3
"""Optimizable 2D, 3D, and 4D lookup tables.

`bins` may be an integer shared by all axes or a sequence containing one resolution
per LUT axis.
"""

import torch
import torch.nn as nn


def normalize_bins(bins, lut_dim):
    """Normalize bin counts to one integer per LUT axis."""
    if isinstance(bins, (list, tuple)):
        if len(bins) != lut_dim:
            raise ValueError(f"bins length {len(bins)} != lut_dim {lut_dim}")
        return [int(b) for b in bins]
    return [int(bins)] * lut_dim


class OptimizableLUT(nn.Module):
    """可优化的 LUT，残差输出。lut_dim=2, 3 或 4
    init_mode:
      'noise'  — 小随机噪声（默认，兼容旧行为）
      'zero'   — 全零（无修正）
      'bias'   — 带通道偏置（差异化初始化用）
    """
    def __init__(self, bins=17, output_channels=3, lut_dim=4,
                 init_noise=0.0, init_mode='noise', init_bias=None):
        super(OptimizableLUT, self).__init__()
        self.bins_per_axis = normalize_bins(bins, lut_dim)
        self.bins = self.bins_per_axis[0]   # 向后兼容：对称时等于每轴 bins
        self.output_channels = output_channels
        self.lut_dim = lut_dim
        shape = list(self.bins_per_axis) + [output_channels]

        if init_mode == 'bias' and init_bias is not None:
            data = torch.randn(*shape) * (init_noise if init_noise > 0 else 0.01)
            for ch in range(output_channels):
                data[..., ch] += init_bias[ch]
            self.lut = nn.Parameter(data)
        elif init_noise > 0:
            self.lut = nn.Parameter(torch.randn(*shape) * init_noise)
        else:
            self.lut = nn.Parameter(torch.zeros(*shape))

    def forward(self):
        return self.lut


class Multi4DLUT(nn.Module):
    """
    多个 LUT 组合（兼容 3D / 4D，逐轴 bins）
    init_noise > 0: 每个 LUT 用小随机值初始化（多LUT+DualEncoder 必须）
    """
    def __init__(self, num_luts=1, bins=17, output_channels=3, lut_dim=4, init_noise=0.0):
        super(Multi4DLUT, self).__init__()
        self.num_luts = num_luts
        self.bins_per_axis = normalize_bins(bins, lut_dim)
        self.bins = self.bins_per_axis[0]
        self.output_channels = output_channels
        self.lut_dim = lut_dim

        # 差异化初始化：每个 LUT 有不同的初始偏置
        # LUT0: 零偏置（保守修正）
        # LUT1: 提亮+补红（ΔY>0, ΔCr>0）
        # LUT2: 压暗+去蓝（ΔY<0, ΔCb<0）
        lut_biases = [
            [0.0, 0.0, 0.0],        # LUT0: 中性
            [+0.05, 0.0, +0.05],     # LUT1: 提亮+补红
            [-0.05, -0.05, 0.0],     # LUT2: 压暗+去蓝
        ]
        self.luts = nn.ModuleList()
        for i in range(num_luts):
            bias = lut_biases[i] if i < len(lut_biases) else [0.0]*output_channels
            self.luts.append(OptimizableLUT(
                self.bins_per_axis, output_channels, lut_dim,
                init_noise=init_noise, init_mode='bias', init_bias=bias,
            ))

        if num_luts > 1:
            self.fusion_weights = nn.Parameter(torch.ones(num_luts) / num_luts)
        else:
            self.register_buffer('fusion_weights', torch.ones(1))

    def forward(self):
        luts = [lut() for lut in self.luts]
        if self.num_luts > 1:
            weights = torch.softmax(self.fusion_weights, dim=0)
        else:
            weights = self.fusion_weights
        return luts, weights


if __name__ == "__main__":
    for dim in (2, 3, 4):
        model = Multi4DLUT(num_luts=1, bins=17, output_channels=3, lut_dim=dim)
        total_params = sum(p.numel() for p in model.parameters())
        luts, weights = model()
        print(f"[{dim}D] 参数量: {total_params:,}  LUT形状: {luts[0].shape}")

    # 非对称 bins
    model = Multi4DLUT(num_luts=3, bins=[33, 13, 25, 25], output_channels=3, lut_dim=4)
    total_params = sum(p.numel() for p in model.parameters())
    luts, weights = model()
    print(f"[asym 33/13/25/25 K=3] 参数量: {total_params:,}  LUT形状: {luts[0].shape}")
