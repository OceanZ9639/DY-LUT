#!/usr/bin/env python3
"""Context encoders used by DY-LUT checkpoints.

ContextGenerator accepts six channels and produces one context index. It is retained
for compatibility with earlier checkpoints. DualBranchContextEncoder accepts Y, Cb,
Cr, depth, and gradient channels; it produces two learned LUT indices and adaptive
LUT weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Encoder retained for earlier checkpoints.
# ─────────────────────────────────────────────────────────────────────────────
class ContextGenerator(nn.Module):
    """Produce a single context coordinate for a 4D LUT."""
    def __init__(self, in_channels=6, hidden_dim=16, num_layers=3):
        super(ContextGenerator, self).__init__()
        self.is_dual_branch = False
        self.in_channels = in_channels

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(hidden_dim, affine=True)
        )

        mid_layers = []
        for _ in range(num_layers):
            mid_layers.append(self._make_conv_block(hidden_dim, hidden_dim))
        self.mid_layers = nn.Sequential(*mid_layers)

        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def _make_conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(out_c, affine=True)
        )

    def forward(self, x):
        feat = self.input_layer(x)
        identity = feat
        out = self.mid_layers(feat)
        out = out + identity
        return self.output_layer(out)


# ─────────────────────────────────────────────────────────────────────────────
# Dual-branch context encoder used by the released checkpoints.
# ─────────────────────────────────────────────────────────────────────────────
class DualBranchContextEncoder(nn.Module):
    """Predict pixel-level degradation coordinates and adaptive LUT weights."""

    def __init__(self, in_channels=5, hidden_dim=32, num_layers=3, num_luts=3,
                 predict_gate=False, pixel_fusion=False, use_ppm=False):
        super(DualBranchContextEncoder, self).__init__()
        self.is_dual_branch = True
        self.in_channels = in_channels
        self.num_luts = num_luts
        # Optionally append a pixel-level depth-reliability gate.
        self.predict_gate = predict_gate
        self.deg_out_channels = 3 if predict_gate else 2
        # Optionally predict pixel-level LUT weights.
        self.pixel_fusion = pixel_fusion
        # Optionally add pyramid-pooled context.
        self.use_ppm = use_ppm

        # ── 共享输入层 ────────────────────────────────────────────────────────
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(hidden_dim, affine=True)
        )

        # ── 局部分支：捕获像素级退化模式 ──────────────────────────────────────
        local_layers = []
        for _ in range(max(1, num_layers - 1)):
            local_layers.append(self._make_conv_block(hidden_dim, hidden_dim))
        self.local_branch = nn.Sequential(*local_layers)

        # ── 全局分支：捕获图像级场景上下文 ───────────────────────────────────
        # 直接从原始输入做 GAP（绕过 InstanceNorm，保留图像级统计差异）
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ── 融合局部+全局 ──────────────────────────────────────────────────────
        self.fusion = self._make_conv_block(hidden_dim * 2, hidden_dim)

        # ── 输出头1：像素级退化描述符（2 通道，作为 LUT 第3、4 维；
        #    predict_gate=True 时第 3 通道为深度可靠性门 α） ─────────────────
        # 使用默认 Kaiming 初始化，保证梯度能穿过此层回传到 encoder
        self.deg_head = nn.Sequential(
            nn.Conv2d(hidden_dim, self.deg_out_channels, 1),
            nn.Sigmoid()
        )
        if predict_gate:
            # α 初始化偏向 1（信任深度），让训练从"与无门版本等价"处起步
            nn.init.zeros_(self.deg_head[0].weight[2:3])
            nn.init.constant_(self.deg_head[0].bias[2], 3.0)   # sigmoid(3)≈0.95

        # ── 输出头2：图像级 LUT 选择权重（从纯全局特征预测）──────────────────
        self.weight_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_luts),
        )
        nn.init.zeros_(self.weight_fc[-1].bias)

        # Identity-initialized pyramid context keeps the initial mapping unchanged.
        if use_ppm:
            self.ppm_pools = (1, 4, 8)
            ppm_ch = max(hidden_dim // 4, 8)
            self.ppm_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(hidden_dim, ppm_ch, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                for _ in self.ppm_pools
            ])
            self.ppm_merge = nn.Conv2d(hidden_dim + ppm_ch * len(self.ppm_pools), hidden_dim, 1)
            nn.init.zeros_(self.ppm_merge.weight)
            nn.init.zeros_(self.ppm_merge.bias)
            with torch.no_grad():
                for c in range(hidden_dim):
                    self.ppm_merge.weight[c, c, 0, 0] = 1.0

        # A zero-initialized residual extends image-level logits to pixel-level logits.
        if pixel_fusion:
            self.pix_w_head = nn.Conv2d(hidden_dim, num_luts, 1)
            nn.init.zeros_(self.pix_w_head.weight)
            nn.init.zeros_(self.pix_w_head.bias)

    def _make_conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(out_c, affine=True)
        )

    def forward(self, x, global_stats=None):
        """
        Args:
            x: [B, 5, H, W]  Y(1)+Cb(1)+Cr(1)+Depth(1)+Grad(1)
            global_stats: [B, 5] 完整图像的全局统计量（训练时由 dataloader 提供）
                          推理时为 None，自动从 x 计算 GAP
        Returns:
            degradation_map: [B, 2, H, W] ∈ [0,1]
                             （predict_gate=True 时为 [B, 3, H, W]，
                               第 3 通道为深度可靠性门 α）
            lut_weights:     [B, num_luts]  (Softmax 归一化)
        """
        feat = self.input_layer(x)              # [B, hidden, H, W]

        # 局部特征
        local_feat = self.local_branch(feat)    # [B, hidden, H, W]

        # 全局特征：训练时用完整图统计量，推理时用当前输入的 GAP
        if global_stats is not None:
            g = global_stats                                       # [B, in_channels]
        else:
            g = self.global_pool(x).squeeze(-1).squeeze(-1)        # [B, in_channels]
        g = self.global_fc(g)                                      # [B, hidden]
        global_feat = g.unsqueeze(-1).unsqueeze(-1).expand_as(local_feat)  # [B, hidden, H, W]

        # 融合
        fused = self.fusion(torch.cat([local_feat, global_feat], dim=1))  # [B, hidden, H, W]

        # Optional pyramid-pooled context.
        if self.use_ppm:
            h, w = fused.shape[2], fused.shape[3]
            pyramid = [fused]
            for pool_size, conv in zip(self.ppm_pools, self.ppm_convs):
                p = conv(F.adaptive_avg_pool2d(fused, pool_size))
                pyramid.append(F.interpolate(p, size=(h, w), mode='bilinear',
                                             align_corners=False))
            fused = self.ppm_merge(torch.cat(pyramid, dim=1))

        # 头1：退化描述符（直接输出，由辅助损失引导学习色度信息）
        deg_map = self.deg_head(fused)          # [B, 2, H, W] ∈ [0,1]

        # 头2：LUT 权重（图像级 logit 从纯全局特征 g 预测）
        w_logit = self.weight_fc(g)                            # [B, num_luts]
        if self.pixel_fusion:
            # Pixel-level logit residual.
            pix_logit = self.pix_w_head(fused)                 # [B, K, H, W]
            lut_w = torch.softmax(
                w_logit.unsqueeze(-1).unsqueeze(-1) + pix_logit, dim=1
            )                                                  # [B, K, H, W]
        else:
            lut_w = torch.softmax(w_logit, dim=1)              # [B, num_luts]

        return deg_map, lut_w


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== ContextGenerator ===")
    old = ContextGenerator(in_channels=6, hidden_dim=16, num_layers=3)
    p = sum(x.numel() for x in old.parameters())
    print(f"  parameters: {p:,} ({p/1e3:.1f}K)")
    x6 = torch.randn(2, 6, 256, 256)
    ctx = old(x6)
    print(f"  input: {x6.shape} output: {ctx.shape} range: [{ctx.min():.3f}, {ctx.max():.3f}]")

    print("\n=== DualBranchContextEncoder (num_luts=3) ===")
    enc = DualBranchContextEncoder(in_channels=5, hidden_dim=32, num_layers=3, num_luts=3)
    p = sum(x.numel() for x in enc.parameters())
    print(f"  parameters: {p:,} ({p/1e3:.1f}K)")
    x5 = torch.randn(2, 5, 256, 256)
    deg, lw = enc(x5)
    print(f"  input: {x5.shape}")
    print(f"  deg_map: {deg.shape} range: [{deg.min():.3f}, {deg.max():.3f}]")
    print(f"  lut_weights: {lw.shape}  sum: {lw.sum(dim=1)}")
