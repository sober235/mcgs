#!/usr/bin/env python
"""fig_recon_view.py — large, high-res recon view for visual aliasing inspection.
Columns: GT | recon | recon @ LOW window (reveals low-amplitude aliasing haze in flat/background regions).
Rows: T1 / FLAIR x axial / coronal / sagittal. Big panels so aliasing & fine detail are actually visible.
"""
import argparse, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--label", default="recon")
ap.add_argument("--out", required=True)
ap.add_argument("--lowwin", type=float, default=0.12, help="low window max as fraction of GT max")
args = ap.parse_args()

d = np.load(args.npz)
VIEWS = [("axial", lambda v: v[:, :, 128]),
         ("coronal", lambda v: v[128, :, :]),
         ("sagittal", lambda v: v[:, 110, :])]
rows = [("T1", d["recT1"], d["gtT1"]), ("FLAIR", d["recFL"], d["gtFL"])]

nrow = len(rows) * len(VIEWS)
cols = ["GT", args.label, f"{args.label}  (window [0,{args.lowwin:g}]×max)"]
fig, ax = plt.subplots(nrow, 3, figsize=(3 * 3.6, nrow * 3.6))
for ri, (con, rec, gt) in enumerate(rows):
    gmax = float(np.abs(gt).max()) + 1e-9
    for vi, (vname, slc) in enumerate(VIEWS):
        r = ri * len(VIEWS) + vi
        g = np.rot90(np.abs(slc(gt))); rc = np.rot90(np.abs(slc(rec)))
        ax[r, 0].imshow(g, cmap="gray", vmin=0, vmax=gmax)
        ax[r, 1].imshow(rc, cmap="gray", vmin=0, vmax=gmax)
        ax[r, 2].imshow(rc, cmap="gray", vmin=0, vmax=args.lowwin * gmax)   # low window -> haze visible
        for c in range(3):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0: ax[r, c].set_title(cols[c], fontsize=12)
        ax[r, 0].set_ylabel(f"{con}\n{vname}", fontsize=11)
fig.suptitle(f"{args.label} — AF=12 recon (large view for aliasing inspection)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(args.out, dpi=155, bbox_inches="tight")
print(f"saved -> {args.out}")
