#!/usr/bin/env python3
"""fig_slice.py -- one named slice, all eight methods, magnitudes + fixed-range error maps.

Unlike make_figures.py (which always takes the mid-slice of each axis) this takes an explicit
axis and index, so a slice picked by scripts/find_best_slice.py can be shown directly. Titles
carry that slice's OWN head-masked PSNR, not the volume number.
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", required=True, choices=["T1", "FLAIR"])
ap.add_argument("--axis", type=int, required=True, choices=[0, 1, 2])
ap.add_argument("--index", type=int, required=True)
ap.add_argument("--methods", default="zf,cgsense,cssense,zsssdu,dip,inr,gs_single,gs_joint_v2")   # the superseded joint arm is left out: v2 is the proposed method
ap.add_argument("--tag", default="best")
args = ap.parse_args()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_retro")
F = Path("/home/congcongliu/figs/t1flair_af12"); F.mkdir(parents=True, exist_ok=True)
ANAT = {0: "transverse (axial)", 1: "coronal", 2: "sagittal"}   # adjudicated 2026-09-09, see README
LABEL = {"zf": "Zero-filled", "cgsense": "CG-SENSE", "cssense": "CS-SENSE (TV)",
         "zsssdu": "ZS-SSDU", "dip": "3D-DIP", "inr": "3D-INR",
         "gs_single": "MC-GS single", "gs_joint": "MC-GS joint",
         "gs_single_v2": "MC-GS single v2", "gs_joint_v2": "MC-GS joint (ours)"}


def load(m, c):
    for suf in ("_complex.npy", "_mag.npy"):
        p = R / f"recon/{m}_{c}{suf}"
        if p.exists():
            return np.abs(np.load(p)).astype(np.float32)
    return None


c, ax, i = args.contrast, args.axis, args.index
gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
gmax = float(gt.max())
head = gt > 0.05 * float(np.quantile(gt, 0.999))
sl = [slice(None)] * 3; sl[ax] = i; sl = tuple(sl)
gp, hp = gt[sl], head[sl]
g = gp[hp].astype(np.float64)

methods = [m for m in args.methods.split(",") if load(m, c) is not None]
psnr = {}
for m in methods:
    e = load(m, c)[sl][hp].astype(np.float64) - g
    psnr[m] = 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
rank = {m: r + 1 for r, m in enumerate(sorted(psnr, key=lambda k: -psnr[k]))}

ncol = len(methods) + 1
# Layout: every panel the same size (the displayed array is gp.T, so height/width = shape[0]/shape[1]),
# near-zero gaps, no per-panel colorbars -- one shared error-map colorbar on the far right, and the
# sampling pattern the whole row is conditioned on in the bottom-left cell.
import scipy.io
m2 = scipy.io.loadmat(R / "data/mask_vdpois_256x176_af12.mat")["mask"].astype(float)
W = 2.6                                          # inches per column
hw = gp.shape[1] / gp.shape[0]                   # displayed panel height / width
fig = plt.figure(figsize=(W * ncol + 0.45, 2 * W * hw + 1.15))
left, right, bottom, top = 0.004, 1 - 0.45 / (W * ncol + 0.45), 0.055, 0.86
gs = fig.add_gridspec(2, ncol, left=left, right=right, bottom=bottom, top=top, wspace=0.015, hspace=0.015)
axes = np.array([[fig.add_subplot(gs[r, k]) for k in range(ncol)] for r in range(2)])
vmax = float(np.quantile(gp[hp], 0.95)) if hp.any() else float(np.quantile(gp, 0.999))   # in-head P95 window, same as make_figures.py
axes[0, 0].imshow(gp.T, cmap="gray", vmin=0, vmax=vmax, origin="lower", aspect="auto")
axes[0, 0].set_title(f"Ground truth ({c})", fontsize=9, fontweight="bold")
axes[1, 0].imshow(m2.T, cmap="gray", origin="lower", interpolation="nearest", aspect="auto")
axes[1, 0].set_xlabel(f"AF12 vd-Poisson mask (ky,kz), {100 * m2.mean():.2f} % of the PE plane", fontsize=7)
im = None
for j, m in enumerate(methods, start=1):
    v = load(m, c)[sl]
    axes[0, j].imshow(v.T, cmap="gray", vmin=0, vmax=vmax, origin="lower", aspect="auto")
    mark = "  (best)" if rank[m] == 1 else ""
    axes[0, j].set_title(f"{LABEL[m]}{mark}\n{psnr[m]:.2f} dB  rank {rank[m]}",
                         fontsize=8.5, fontweight="bold" if rank[m] == 1 else "normal", pad=2)
    im = axes[1, j].imshow((np.abs(v - gp) / gmax).T, cmap="jet", vmin=0.0, vmax=0.1, origin="lower", aspect="auto")
for a in axes.ravel():
    a.set_xticks([]); a.set_yticks([])
pos = axes[1, -1].get_position()
cax = fig.add_axes([pos.x1 + 0.004, pos.y0, 0.010, pos.height])
cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=7)
cb.set_label("|recon - GT| / max(GT)", fontsize=7)
fig.suptitle(f"{c}  retrospective AF 12  --  axis{ax} slice {i} ({ANAT[ax]})   "
             f"(head-masked PSNR of this slice; error maps jet, fixed [0, 0.1])",
             fontsize=11, fontweight="bold", y=0.985)
out = F / f"t1flair_af12_{c}_{args.tag}_axis{ax}_{i}.png"
fig.savefig(out, dpi=110, bbox_inches="tight"); fig.savefig(R / f"figures/{out.name}", dpi=110, bbox_inches="tight")
print("wrote", out)
print(json.dumps({m: round(psnr[m], 3) for m in sorted(psnr, key=lambda k: -psnr[k])}, indent=None))
