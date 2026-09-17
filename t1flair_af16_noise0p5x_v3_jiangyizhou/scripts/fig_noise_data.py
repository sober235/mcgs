#!/usr/bin/env python3
"""fig_noise_data.py -- what 0.5 sigma0 of added measurement noise looks like (noise study, 2026-09-16).

Rows per contrast: the clean fully sampled image (the ground truth, recon/gt_<c>_complex.npy), the fully sampled image
from the noisy measurements (recon/fullnoisy_<c>_complex.npy, written by make_noisy_kspace.py) and
|noisy - clean| / max(GT) on jet, fixed [0, 0.1]. Columns: transverse 180 / coronal 124 / sagittal 76 (the
figure-of-record slices), displayed exactly as fig_three_planes_compact.py does: origin lower, axis 0 transposed and
turned eyes up, axes 1 and 2 as stored (head up). Axes are placed in inches: every panel is 2.6 in tall and as wide
as its aspect, so the columns carry no slack.
Writes figures/t1flair_v3_af16n05_noise_data.png and the same file under /home/congcongliu/figs/t1flair_v3_af16_noise0p5x/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parents[1]
FIGS = Path("/home/congcongliu/figs/t1flair_v3_af16_noise0p5x")
SL = (180, 124, 76)
NAMES = ("Transverse", "Coronal", "Sagittal")
rep = json.load(open(R / "logs/make_noisy_kspace.json"))


def plane(v, axis, idx):
    s = [slice(None)] * 3
    s[axis] = idx
    x = v[tuple(s)]
    return np.rot90(x.T, 1) if axis == 0 else x          # the compact figure's D() with --tra_rot 1


gt0 = np.load(R / "recon/gt_T1_complex.npy")
shapes = [plane(np.abs(gt0), a, SL[a]).shape for a in range(3)]
PH, GAP, LEFT, TOP, BOT, CB_PAD, CB_W, RIGHT = 2.6, 0.06, 0.75, 0.95, 0.08, 0.08, 0.14, 0.55   # inches
PW = [PH * w / h for h, w in shapes]
fig_w = LEFT + sum(PW) + 2 * GAP + CB_PAD + CB_W + RIGHT
fig_h = TOP + 6 * PH + 5 * GAP + BOT
fig = plt.figure(figsize=(fig_w, fig_h))


def box(x, y, w, h):                                      # inches from the top-left corner -> figure fraction
    return [x / fig_w, 1 - (y + h) / fig_h, w / fig_w, h / fig_h]


for ci, c in enumerate(("T1", "FLAIR")):
    info = rep["contrasts"][c]
    gt = gt0 if c == "T1" else np.load(R / f"recon/gt_{c}_complex.npy")
    noisy = np.load(R / f"recon/fullnoisy_{c}_complex.npy")
    g, n = np.abs(gt), np.abs(noisy)
    vmax = float(np.quantile(g, 0.999))
    err = np.abs(noisy - gt) / float(g.max())
    rows = ((g, f"{c}: clean full data (GT)\nair noise {info['sigma0_img']:.4f}", "gray", vmax),
            (n, f"{c}: noisy full data\nair noise {info['air_noise_noisy']:.4f}", "gray", vmax),
            (err, f"{c}: |noisy − clean| / max(GT)", "jet", 0.1))
    for ri, (v, title, cmap, hi) in enumerate(rows):
        r = ci * 3 + ri
        y = TOP + r * (PH + GAP)
        for ai in range(3):
            ax = fig.add_axes(box(LEFT + sum(PW[:ai]) + GAP * ai, y, PW[ai], PH))
            im = ax.imshow(plane(v, ai, SL[ai]), cmap=cmap, vmin=0, vmax=hi, origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{NAMES[ai]} {SL[ai]}", fontsize=12)
            if ai == 0:
                ax.set_ylabel(title, fontsize=10)
        fig.colorbar(im, cax=fig.add_axes(box(LEFT + sum(PW) + 2 * GAP + CB_PAD, y + 0.08 * PH, CB_W, 0.84 * PH)))
fig.suptitle("jiangyizhou, noise study: complex Gaussian noise added to the measured k-space\n"
             f"per-sample std = 0.5 x the data's own noise (seed {rep['seed']})", fontsize=12, y=1 - 0.08 / fig_h, va="top")
FIGS.mkdir(parents=True, exist_ok=True)
for d in (R / "figures", FIGS):
    fig.savefig(d / "t1flair_v3_af16n05_noise_data.png", dpi=110, bbox_inches="tight", pad_inches=0.06)
    print("wrote", d / "t1flair_v3_af16n05_noise_data.png")
