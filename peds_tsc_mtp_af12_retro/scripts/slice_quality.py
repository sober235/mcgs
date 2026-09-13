#!/usr/bin/env python3
"""slice_quality.py -- head-masked PSNR and SSIM of the nine row-figure methods on given slices, with MC-GS's rank.

  python scripts/slice_quality.py --axis 0 --slices 142 174 178
PSNR: the fig_row_final.py gauge (head mask |GT| > 0.05 * P99.9 of the volume, peak = max|GT| of the volume).
SSIM: 2D SSIM map of |recon| against |GT| on the slice (Gaussian window, sigma 1.5, data range = max|GT| of the
volume), averaged over the head voxels of the slice. --ssim uniform: 7x7 uniform window instead, the skimage default
that paired_gs/metrics/compute_ssim uses.
"""
import argparse, json, re
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

ap = argparse.ArgumentParser()
ap.add_argument("--axis", type=int, default=0, choices=[0, 1, 2])
ap.add_argument("--slices", type=int, nargs="+", required=True)
ap.add_argument("--ssim", choices=["gaussian", "uniform"], default="gaussian")
args = ap.parse_args()

R = Path(__file__).resolve().parents[1]
summary = json.load(open(R / "logs/summary.json"))


def arms(c):
    rows = {r["method"]: r for r in summary if r["contrast"] == c}

    def lam_file(row_key, prefix):
        lam = rows.get(row_key, {}).get("lam")
        for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
            m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
            if lam is not None and m and abs(float(m.group(1)) - float(lam)) < 1e-12:
                return p
        raise FileNotFoundError(f"{prefix} {c} lam={lam}")

    return [("CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"),
            ("CS-SENSE", lam_file("cssense_o200", "cssense_o200")),
            ("Joint SENSE", lam_file("jcssense", "jcssense_o200")),
            ("L1-ESPIRiT", lam_file("l1espirit", "l1espirit")),
            ("ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
            ("3D-DIP", R / f"recon/dip_{c}_complex.npy"),
            ("3D-INR", R / f"recon/inr_{c}_complex.npy"),
            ("GS-single", R / f"recon/gs_single_ar_{c}_mag.npy"),
            ("MC-GS", R / f"recon/gs_joint_ar_{c}_mag.npy")]


def take(vol, s):
    sl = [slice(None)] * 3; sl[args.axis] = s
    return vol[tuple(sl)]


res = {}
for c in ("T1w", "T2s"):
    gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
    head = gt > 0.05 * float(np.quantile(gt, 0.999))
    gmax = float(gt.max())
    vols = {lab: np.abs(np.load(p)).astype(np.float32) for lab, p in arms(c)}
    for s in args.slices:
        g, m = take(gt, s).astype(np.float64), take(head, s)
        for lab, v in vols.items():
            r = take(v, s).astype(np.float64)
            e = r[m] - g[m]
            psnr = 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
            kw = (dict(gaussian_weights=True, sigma=1.5, use_sample_covariance=False) if args.ssim == "gaussian"
                  else dict(win_size=7))
            _, smap = structural_similarity(g, r, data_range=gmax, full=True, **kw)
            res[(c, s, lab)] = (psnr, float(smap[m].mean()))

plane = {0: "transverse", 1: "coronal", 2: "sagittal"}[args.axis]
print(f"{plane} slices, SSIM window {args.ssim}; for each contrast: MC-GS rank of 9 and its lead over the best other method [who]")
for s in args.slices:
    cells = []
    for c in ("T1w", "T2s"):
        for k, name in ((0, "PSNR"), (1, "SSIM")):
            vals = {lab: res[(c, s, lab)][k] for lab in [a for a, _ in arms(c)]}
            order = sorted(vals, key=lambda a: -vals[a])
            other = next(a for a in order if a != "MC-GS")
            fmt = "+.3f" if k == 0 else "+.4f"
            cells.append(f"{c} {name} #{order.index('MC-GS') + 1} {vals['MC-GS'] - vals[other]:{fmt}} [{other}]")
    print(f"  slice {s:3d}: " + " | ".join(cells))
