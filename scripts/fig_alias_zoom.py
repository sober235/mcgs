#!/usr/bin/env python
"""fig_alias_zoom.py — definitive residual-aliasing check. For each contrast/view: GT (full, ROI box) |
GT zoom | recon zoom | recon zoom LOW-window | |recon-GT| zoom (jet). Matched windows GT<->recon so any
coherent structure the recon has in a HOMOGENEOUS region that GT lacks = residual aliasing (vs random
speckle = grain / vs blur = detail-loss).
"""
import argparse, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--label", default="recon")
ap.add_argument("--out", required=True)
ap.add_argument("--err_scale", type=float, default=0.08)
ap.add_argument("--lowwin", type=float, default=0.15)
args = ap.parse_args()

d = np.load(args.npz)
# (contrast, recon, gt, view-name, slicer, ROI on the rot90'd slice [r0,r1,c0,c1])
PANELS = [
    ("T1 axial",    d["recT1"], d["gtT1"], lambda v: v[:, :, 128], (55, 150, 80, 185)),
    ("T1 coronal",  d["recT1"], d["gtT1"], lambda v: v[128, :, :], (30, 120, 70, 175)),
    ("FLAIR axial", d["recFL"], d["gtFL"], lambda v: v[:, :, 128], (55, 150, 80, 185)),
]
cols = ["GT (full, ROI)", "GT  zoom", f"{args.label} zoom", f"{args.label} zoom low[0,{args.lowwin:g}]",
        f"|{args.label} − GT| zoom"]
nrow = len(PANELS)
fig, ax = plt.subplots(nrow, 5, figsize=(5 * 3.4, nrow * 3.4))
for r, (name, rec, gt, slc, roi) in enumerate(PANELS):
    G = np.rot90(np.abs(slc(gt))); R = np.rot90(np.abs(slc(rec)))
    gmax = float(G.max()) + 1e-9
    r0, r1, c0, c1 = roi
    Gz, Rz = G[r0:r1, c0:c1], R[r0:r1, c0:c1]
    ax[r, 0].imshow(G, cmap="gray", vmin=0, vmax=gmax)
    ax[r, 0].add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, ec="red", fc="none", lw=1.5))
    ax[r, 1].imshow(Gz, cmap="gray", vmin=0, vmax=0.45 * gmax)
    ax[r, 2].imshow(Rz, cmap="gray", vmin=0, vmax=0.45 * gmax)
    ax[r, 3].imshow(Rz, cmap="gray", vmin=0, vmax=args.lowwin * gmax)
    ax[r, 4].imshow(np.abs(Rz - Gz) / gmax, cmap="jet", vmin=0, vmax=args.err_scale)
    for c in range(5):
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        if r == 0: ax[r, c].set_title(cols[c], fontsize=11)
    ax[r, 0].set_ylabel(name, fontsize=11)
fig.suptitle(f"{args.label} — residual-aliasing zoom check (coherent structure in flat tissue = aliasing)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(args.out, dpi=160, bbox_inches="tight")
print(f"saved -> {args.out}")
