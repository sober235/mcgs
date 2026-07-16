#!/usr/bin/env python
"""Print the MC-GS winner reg_aligned unified-gauge PSNR + NMSE (the numbers to beat).
Source of truth = results_arc/reg_aligned_volumes.npz (recon/GT magnitudes, both contrasts)."""
import numpy as np
ROOT = "/data0/congcong/code/GS/Paired_ARC_GS_general_framework"
d = np.load(f"{ROOT}/results_arc/reg_aligned_volumes.npz")

def psnr_uni(rec, gt, p=99.9):
    dr = float(np.percentile(gt, p)); mse = float(np.mean((rec - gt) ** 2))
    return 10.0 * np.log10(dr ** 2 / mse)

def nmse(rec, gt):
    return float(((rec - gt) ** 2).sum() / (gt ** 2).sum())

for name, rk, gk in [("T1", "recT1", "gtT1"), ("FLAIR", "recFL", "gtFL")]:
    rec = d[rk].astype("float64"); gt = d[gk].astype("float64")
    print(f"MC-GS winner {name}: PSNR_uni(P99.9)={psnr_uni(rec, gt):.3f} dB  "
          f"NMSE={nmse(rec, gt):.4f}  (@max PSNR={10*np.log10(gt.max()**2/np.mean((rec-gt)**2)):.3f})")
