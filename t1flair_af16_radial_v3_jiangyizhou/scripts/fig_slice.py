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
ap.add_argument("--methods", default="zf,cgsense,cssense,zsssdu,dip,inr,gs_single,gs_joint")
ap.add_argument("--tag", default="best")
args = ap.parse_args()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_radial_v3_jiangyizhou")
F = Path("/home/congcongliu/figs/t1flair_v3_af24"); F.mkdir(parents=True, exist_ok=True)
ANAT = {0: "transverse (axial)", 1: "coronal", 2: "sagittal"}   # adjudicated 2026-09-09, see README
LABEL = {"zf": "Zero-filled", "cgsense": "CG-SENSE", "cssense": "CS-SENSE (TV)",
         "zsssdu": "ZS-SSDU", "dip": "3D-DIP", "inr": "3D-INR",
         "gs_single": "MC-GS single", "gs_joint": "MC-GS joint"}


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
fig, axes = plt.subplots(2, ncol, figsize=(3.4 * ncol, 7.6))
vmax = float(np.quantile(gp, 0.999))
axes[0, 0].imshow(gp.T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
axes[0, 0].set_title(f"Ground truth ({c})", fontsize=12, fontweight="bold")
axes[1, 0].imshow(hp.T, cmap="gray", origin="lower")
axes[1, 0].set_xlabel(f"head mask on this slice\n{int(hp.sum())} voxels", fontsize=9)
for j, m in enumerate(methods, start=1):
    v = load(m, c)[sl]
    axes[0, j].imshow(v.T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
    mark = "  <-- best" if rank[m] == 1 else ""
    axes[0, j].set_title(f"{LABEL[m]}{mark}\nslice head PSNR {psnr[m]:.2f} dB  (rank {rank[m]})",
                         fontsize=10, fontweight="bold" if rank[m] == 1 else "normal")
    im = axes[1, j].imshow((np.abs(v - gp) / gmax).T, cmap="jet", vmin=0.0, vmax=0.1, origin="lower")
    axes[1, j].set_title("|recon - GT| / max(GT)", fontsize=9)
    cb = fig.colorbar(im, ax=axes[1, j], fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=6)
for a in axes.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"{c}  retrospective AF 24  --  axis{ax} slice {i} ({ANAT[ax]})   "
             f"(head-masked PSNR of THIS slice; error maps jet, fixed [0, 0.1])",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = F / f"t1flair_v3_af16r_{c}_{args.tag}_axis{ax}_{i}.png"
fig.savefig(out, dpi=110, bbox_inches="tight"); fig.savefig(R / f"figures/{out.name}", dpi=110, bbox_inches="tight")
print("wrote", out)
print(json.dumps({m: round(psnr[m], 3) for m in sorted(psnr, key=lambda k: -psnr[k])}, indent=None))
