#!/usr/bin/env python3
"""
LocalRefine: 轻量级 3×3 邻域细化模块

在 LUT 查表输出后做局部空间修正，弥补 LUT 逐像素独立查表的局限。
设计要点：
  - 仅用 3×3 depthwise separable conv，极小参数量（< 2K）
  - 输入：LUT 增强后 YCbCr(3ch) + 原始输入 YCbCr(3ch) = 6ch
  - 输出：残差 delta YCbCr(3ch)，加回 LUT 输出
  - 全链路 YCbCr，和 LUT 在同一颜色空间工作
"""

import torch
import torch.nn as nn


class LocalRefine(nn.Module):
    def __init__(self, hidden_dim=16, num_layers=3):
        super().__init__()
        in_ch = 6  # LUT output YCbCr(3) + input YCbCr(3)
        layers = []
        layers.append(nn.Conv2d(in_ch, hidden_dim, 3, padding=1, groups=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim))
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        out_conv = nn.Conv2d(hidden_dim, 3, 3, padding=1)
        nn.init.normal_(out_conv.weight, 0, 1e-4)
        nn.init.zeros_(out_conv.bias)
        layers.append(out_conv)

        self.net = nn.Sequential(*layers)

    def forward(self, lut_ycbcr, input_ycbcr):
        """
        Args:
            lut_ycbcr:   [B, 3, H, W] LUT 增强后的 YCbCr
            input_ycbcr: [B, 3, H, W] 原始输入的 YCbCr
        Returns:
            refined:     [B, 3, H, W] 细化后的 YCbCr
        """
        x = torch.cat([lut_ycbcr, input_ycbcr], dim=1)  # [B, 6, H, W]
        delta = self.net(x)                                # [B, 3, H, W]
        return lut_ycbcr + delta


class _ResBlock(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class ResRefine(nn.Module):
    """Stronger spatial refine: residual CNN with dilations for a larger receptive
    field, correcting spatial errors the per-pixel LUT cannot. Signature-compatible
    with LocalRefine (hidden_dim, num_layers); num_layers = number of residual blocks.
    Input: LUT-enhanced YCbCr(3) + original YCbCr(3) = 6ch; output: refined YCbCr.
    """
    def __init__(self, hidden_dim=64, num_layers=6):
        super().__init__()
        in_ch = 6
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        dilations = [1, 2, 4, 2, 1]
        self.blocks = nn.Sequential(*[
            _ResBlock(hidden_dim, dilation=dilations[i % len(dilations)])
            for i in range(num_layers)
        ])
        out_conv = nn.Conv2d(hidden_dim, 3, 3, padding=1)
        nn.init.normal_(out_conv.weight, 0, 1e-4)
        nn.init.zeros_(out_conv.bias)
        self.tail = out_conv

    def forward(self, lut_ycbcr, input_ycbcr):
        x = torch.cat([lut_ycbcr, input_ycbcr], dim=1)
        h = self.head(x)
        h = self.blocks(h)
        delta = self.tail(h)
        return lut_ycbcr + delta


if __name__ == "__main__":
    m = LocalRefine(hidden_dim=16, num_layers=3)
    p = sum(x.numel() for x in m.parameters())
    print(f"LocalRefine params: {p:,} ({p/1e3:.1f}K)")
    lut_out = torch.rand(2, 3, 64, 64)
    inp = torch.rand(2, 3, 64, 64)
    out = m(lut_out, inp)
    diff = (out - lut_out).abs().mean()
    print(f"shape: {out.shape}, init delta: {diff:.6f} (should be ~0)")
