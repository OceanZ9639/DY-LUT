#!/usr/bin/env python3
"""Evaluate PSNR and SSIM with the SMDR-IS protocol:

  - resize both enhanced and gt to 256x256
  - PSNR = 20*log10(1/rmse) on [0,1]
  - SSIM = pytorch_msssim.ssim(data_range=1.0)

Images are matched by filename stem.
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from pytorch_msssim import ssim as _ssim


def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff ** 2).mean().sqrt()
    return 20 * torch.log10(1 / rmse)


def torchSSIM(tar_img, prd_img):
    return _ssim(tar_img, prd_img, data_range=1.0, size_average=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    names = sorted(n for n in os.listdir(args.pred)
                   if n.lower().endswith(('.png', '.jpg', '.jpeg')))
    psnrs, ssims = [], []
    for n in names:
        stem = os.path.splitext(n)[0]
        gtp = None
        for ext in ('.png', '.jpg', '.jpeg'):
            cand = os.path.join(args.gt, stem + ext)
            if os.path.exists(cand):
                gtp = cand
                break
        enhanced = cv2.imread(os.path.join(args.pred, n))
        if gtp is None or enhanced is None:
            print(f"[skip] {n}")
            continue
        gt = cv2.imread(gtp)
        enhanced = cv2.resize(enhanced, [256, 256])
        gt = cv2.resize(gt, [256, 256])
        gt_t = TF.to_tensor(gt)
        en_t = TF.to_tensor(enhanced)
        p = torchPSNR(gt_t, en_t).item()
        s = torchSSIM(gt_t.unsqueeze(0), en_t.unsqueeze(0)).item()
        if not args.quiet:
            print(f"{n} PSNR={p:.4f} SSIM={s:.4f}")
        psnrs.append(p)
        ssims.append(s)

    if not psnrs:
        raise RuntimeError("No matching prediction/ground-truth image pairs were found")

    print(f"N={len(psnrs)}")
    print(f"PSNR >> Mean: {np.mean(psnrs):.4f} std: {np.std(psnrs):.4f}")
    print(f"SSIM >> Mean: {np.mean(ssims):.4f} std: {np.std(ssims):.4f}")


if __name__ == '__main__':
    main()
