#!/usr/bin/env python3
"""
损失函数  L1 + SSIM + Chroma + TV + MN
TV / MN 自动适配 3D / 4D LUT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size).contiguous()


class SSIM_Loss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_Loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel != self.channel or self.window.dtype != img1.dtype:
            self.window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.channel = channel
        window = self.window.to(img1.device)
        return 1 - self._ssim(img1, img2, window, self.window_size, channel)

    def _ssim(self, img1, img2, window, window_size, channel):
        pad = window_size // 2
        mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
        mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


class TVLUTLoss(nn.Module):
    """Total Variation 平滑性约束，自动适配任意维度的 LUT（3D / 4D）"""
    def forward(self, luts):
        total_tv = 0
        for lut in luts:
            ndim = lut.ndim - 1                       # 排除最后的输出通道维
            tv = 0
            for d in range(ndim):
                diff = lut.narrow(d, 0, lut.shape[d] - 1) - lut.narrow(d, 1, lut.shape[d] - 1)
                weight = torch.ones_like(diff)
                idx_first = [slice(None)] * diff.ndim
                idx_last  = [slice(None)] * diff.ndim
                idx_first[d] = slice(0, 1)
                idx_last[d]  = slice(-1, None)
                weight[tuple(idx_first)] = 2.0
                weight[tuple(idx_last)]  = 2.0
                tv += torch.mean(diff ** 2 * weight)
            total_tv += tv
        return total_tv / len(luts)


class MNLUTLoss(nn.Module):
    """单调性约束：第 0 维（Y 轴）对第 0 输出通道（ΔY）的单调正则。

    relax=False（默认，原版行为）：要求残差 ΔY 沿 Y 单调不减，
        即 relu(T[i] - T[i+1])。这比"总映射单调"严格——过约束。
    relax=True（S5）：只要求总映射 Y + ΔY 沿 Y 单调不减。
        离散条件: T[i+1] - T[i] >= -(Y[i+1]-Y[i]) = -1/(N-1)
        惩罚项:   relu(T[i] - T[i+1] - 1/(N-1))
        保留防 banding/反序语义，释放 LUT 表达局部对比度调整的自由度。
    """
    def __init__(self, relax=False):
        super(MNLUTLoss, self).__init__()
        self.relu = nn.ReLU()
        self.relax = relax

    def forward(self, luts):
        total_mn = 0
        for lut in luts:
            margin = (1.0 / (lut.shape[0] - 1)) if self.relax else 0.0
            dif = lut[:-1, ..., 0] - lut[1:, ..., 0] - margin
            total_mn += torch.mean(self.relu(dif))
        return total_mn / len(luts)


# 向后兼容
TV_4D_Loss = TVLUTLoss
MN_4D_Loss = MNLUTLoss


class ChromaLoss(nn.Module):
    """YCbCr 空间色度损失"""
    def __init__(self):
        super(ChromaLoss, self).__init__()

    def forward(self, pred_rgb, target_rgb):
        pred_ycbcr = self._rgb_to_ycbcr(pred_rgb)
        target_ycbcr = self._rgb_to_ycbcr(target_rgb)
        cb_loss = F.l1_loss(pred_ycbcr[:, 1], target_ycbcr[:, 1])
        cr_loss = F.l1_loss(pred_ycbcr[:, 2], target_ycbcr[:, 2])
        return cb_loss + cr_loss

    @staticmethod
    def _rgb_to_ycbcr(rgb):
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128. / 256. - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128. / 256. + 0.5 * r - 0.418688 * g - 0.081312 * b
        return torch.stack([y, cb, cr], dim=1)


class ContrastLoss(nn.Module):
    """LAB 空间 L 通道对比度损失 — 防止输出亮度动态范围收缩。
    
    只惩罚「预测对比度 < 目标对比度」的情形，允许预测对比度等于或超过目标。
    对比度代理量：L 通道在全图上的标准差。
    """

    def forward(self, pred_rgb, target_rgb):
        pred_L   = self._luma(pred_rgb)    # [B, H, W]
        target_L = self._luma(target_rgb)
        pred_std   = pred_L.flatten(1).std(dim=1)    # [B]
        target_std = target_L.flatten(1).std(dim=1)
        # relu: 只有 pred 对比度低于 target 时才产生梯度
        return F.relu(target_std - pred_std).mean()

    @staticmethod
    def _luma(rgb):
        """sRGB → 近似感知亮度 L（0~1 范围，可微）"""
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        # ITU-R BT.709 luma
        return 0.2126 * r + 0.7152 * g + 0.0722 * b


class VGGPerceptualLoss(nn.Module):
    """VGG-16 Perceptual Loss (features from relu1_2, relu2_2, relu3_3, relu4_3)"""
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=True).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        # relu1_2=3, relu2_2=8, relu3_3=15, relu4_3=22
        self.slices = nn.ModuleList([
            vgg[:4], vgg[4:9], vgg[9:16], vgg[16:23]
        ])
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred_n   = (pred   - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        loss = 0.0
        x, y = pred_n, target_n
        for s in self.slices:
            x = s(x)
            with torch.no_grad():
                y = s(y)
            loss += F.l1_loss(x, y)
        return loss


class FFTLoss(nn.Module):
    """频域 L1 损失：惩罚频谱差异，保留高频细节"""
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


class GradientLoss(nn.Module):
    """边缘/梯度 L1 损失：保持结构清晰"""
    def forward(self, pred, target):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


class CDIRegLoss(nn.Module):
    """Range-coverage and decorrelation regularization for learned LUT indices."""
    def forward(self, deg_map):
        # deg_map: [B, 2, H, W] ∈ [0,1]
        flat = deg_map.flatten(2)                       # [B, 2, N]
        # range: 用逐图 min/max（soft，通过采样值本身可微）
        mins = flat.min(dim=2).values                   # [B, 2]
        maxs = flat.max(dim=2).values                   # [B, 2]
        range_loss = (mins.abs() + (1 - maxs).abs()).mean()
        # decorrelation
        centered = flat - flat.mean(dim=2, keepdim=True)
        std = centered.std(dim=2) + 1e-6                # [B, 2]
        cov = (centered[:, 0] * centered[:, 1]).mean(dim=1)   # [B]
        corr = cov / (std[:, 0] * std[:, 1])
        decor_loss = corr.abs().mean()
        return range_loss, decor_loss


class CombinedLoss(nn.Module):
    """YCbCr reconstruction, LUT, perceptual, gradient, and index losses."""
    def __init__(self, w_l1=1.0, w_ssim=1.0, w_chroma=1.0, w_tv=0.0001, w_mn=10.0,
                 w_contrast=0.0, w_l2=0.0, w_vgg=0.0, w_fft=0.0, w_grad=0.0,
                 w_cb=1.5, w_cr=1.5,
                 w_cdi_range=0.0, w_cdi_decor=0.0,
                 mn_relax=False,
                 bins=17, output_channels=3):
        super(CombinedLoss, self).__init__()
        self.w_l1 = w_l1
        self.w_l2 = w_l2
        self.w_ssim = w_ssim
        self.w_chroma = w_chroma
        self.w_tv = w_tv
        self.w_mn = w_mn
        self.w_contrast = w_contrast
        self.w_vgg = w_vgg
        self.w_fft = w_fft
        self.w_grad = w_grad
        self.w_cb = w_cb
        self.w_cr = w_cr
        self.w_cdi_range = w_cdi_range
        self.w_cdi_decor = w_cdi_decor

        self.ssim_loss = SSIM_Loss()
        self.tv_loss = TVLUTLoss()
        self.mn_loss = MNLUTLoss(relax=mn_relax)
        self.vgg_loss = VGGPerceptualLoss() if w_vgg > 0 else None
        self.grad_loss = GradientLoss() if w_grad > 0 else None
        self.cdi_reg = CDIRegLoss() if (w_cdi_range > 0 or w_cdi_decor > 0) else None

    @staticmethod
    def _rgb_to_ycbcr(rgb):
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128. / 256. - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128. / 256. + 0.5 * r - 0.418688 * g - 0.081312 * b
        return torch.stack([y, cb, cr], dim=1)

    def forward(self, pred_rgb, target_rgb, luts, pred_ycbcr=None, deg_map=None):
        # If pred_ycbcr provided, use it; else convert from RGB
        if pred_ycbcr is None:
            pred_ycbcr = self._rgb_to_ycbcr(pred_rgb)
        target_ycbcr = self._rgb_to_ycbcr(target_rgb)

        # ── L_rec: YCbCr 分通道加权 L1 + Y-only SSIM ──
        l1_y  = F.l1_loss(pred_ycbcr[:, 0], target_ycbcr[:, 0])
        l1_cb = F.l1_loss(pred_ycbcr[:, 1], target_ycbcr[:, 1])
        l1_cr = F.l1_loss(pred_ycbcr[:, 2], target_ycbcr[:, 2])
        l1_ycbcr = self.w_l1 * l1_y + self.w_cb * l1_cb + self.w_cr * l1_cr

        ssim = self.ssim_loss(pred_ycbcr[:, 0:1], target_ycbcr[:, 0:1])  # Y-only SSIM
        rec = l1_ycbcr + self.w_ssim * ssim

        # ── L_lut: TV + MN ──
        tv = self.tv_loss(luts)
        mn = self.mn_loss(luts)
        lut_reg = self.w_tv * tv + self.w_mn * mn

        # ── L_detail: VGG(RGB) ──
        vgg = self.vgg_loss(pred_rgb, target_rgb) if self.vgg_loss is not None else torch.tensor(0.0, device=pred_rgb.device)
        detail = self.w_vgg * vgg

        # ── RGB-L2 (directly tied to PSNR) ──
        if self.w_l2 > 0:
            l2 = F.mse_loss(pred_rgb, target_rgb)
        else:
            l2 = torch.tensor(0.0, device=pred_rgb.device)

        # ── Gradient (Y-only edge) ──
        if self.grad_loss is not None:
            grad = self.grad_loss(pred_ycbcr[:, 0:1], target_ycbcr[:, 0:1])
        else:
            grad = torch.tensor(0.0, device=pred_rgb.device)

        # Learned-index range and decorrelation regularization.
        cdi_range = torch.tensor(0.0, device=pred_rgb.device)
        cdi_decor = torch.tensor(0.0, device=pred_rgb.device)
        if self.cdi_reg is not None and deg_map is not None:
            cdi_range, cdi_decor = self.cdi_reg(deg_map[:, 0:2])

        total = (rec + lut_reg + detail + self.w_l2 * l2 + self.w_grad * grad
                 + self.w_cdi_range * cdi_range + self.w_cdi_decor * cdi_decor)

        loss_dict = {
            'total': total.item(),
            'l1_y': l1_y.item(),
            'l1_cb': l1_cb.item(),
            'l1_cr': l1_cr.item(),
            'ssim': ssim.item(),
            'tv': tv.item(),
            'mn': mn.item(),
            'vgg': vgg.item(),
            'l2': l2.item(),
            'grad': grad.item(),
            'cdi_rng': cdi_range.item(),
            'cdi_dec': cdi_decor.item(),
        }
        return total, loss_dict


if __name__ == "__main__":
    B, C, H, W = 4, 3, 128, 128
    pred = torch.rand(B, C, H, W)
    target = torch.rand(B, C, H, W)

    for dim in (3, 4):
        shape = [17] * dim + [3]
        luts = [torch.randn(*shape)]
        criterion = CombinedLoss()
        loss, loss_dict = criterion(pred, target, luts)
        print(f"[{dim}D LUT] 损失: ", {k: f"{v:.4f}" for k, v in loss_dict.items()})
