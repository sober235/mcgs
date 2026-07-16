#!/usr/bin/env python
"""fig_compare_zoom.py — zoom-compare two recons vs GT (blur vs speckle tradeoff).
Columns: GT | reconA | reconB | |reconA-GT| | |reconB-GT|  (recons at matched window, err jet).
"""
import argparse, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--npz_a", required=True); ap.add_argument("--label_a", default="A")
ap.add_argument("--npz_b", required=True); ap.add_argument("--label_b", default="B")
ap.add_argument("--out", required=True); ap.add_argument("--err_scale", type=float, default=0.08)
args = ap.parse_args()

A = np.load(args.npz_a); B = np.load(args.npz_b)
# (name, recA, recB, gt, slicer, ROI on rot90'd slice)
P = [
    ("T1 axial",    A["recT1"], B["recT1"], A["gtT1"], lambda v: v[:, :, 128], (55, 150, 80, 185)),
    ("FLAIR axial", A["recFL"], B["recFL"], A["gtFL"], lambda v: v[:, :, 128], (55, 150, 80, 185)),
]
cols = ["GT zoom", f"{args.label_a} zoom", f"{args.label_b} zoom", f"|{args.label_a}−GT|", f"|{args.label_b}−GT|"]
fig, ax = plt.subplots(len(P), 5, figsize=(5 * 3.5, len(P) * 3.5))
for r, (name, ra, rb, gt, slc, roi) in enumerate(P):
    G = np.rot90(np.abs(slc(gt))); RA = np.rot90(np.abs(slc(ra))); RB = np.rot90(np.abs(slc(rb)))
    gmax = float(G.max()) + 1e-9; r0, r1, c0, c1 = roi
    Gz, Az, Bz = G[r0:r1, c0:c1], RA[r0:r1, c0:c1], RB[r0:r1, c0:c1]
    ax[r, 0].imshow(Gz, cmap="gray", vmin=0, vmax=0.45 * gmax)
    ax[r, 1].imshow(Az, cmap="gray", vmin=0, vmax=0.45 * gmax)
    ax[r, 2].imshow(Bz, cmap="gray", vmin=0, vmax=0.45 * gmax)
    ax[r, 3].imshow(np.abs(Az - Gz) / gmax, cmap="jet", vmin=0, vmax=args.err_scale)
    ax[r, 4].imshow(np.abs(Bz - Gz) / gmax, cmap="jet", vmin=0, vmax=args.err_scale)
    for c in range(5):
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        if r == 0: ax[r, c].set_title(cols[c], fontsize=11)
    ax[r, 0].set_ylabel(name, fontsize=11)
fig.suptitle("AF=12 blur↔speckle tradeoff — zoom (flat-tissue speckle = residual aliasing)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.99]); fig.savefig(args.out, dpi=160, bbox_inches="tight")
print(f"saved -> {args.out}")
