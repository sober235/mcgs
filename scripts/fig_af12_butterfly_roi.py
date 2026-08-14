#!/usr/bin/env python3
"""AF12 transverse butterfly-region ROI figure."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


NPZ = Path("results_arc/af12_ar_r1_e060_volumes.npz")
OUT = Path("results_arc/AF12_autoresearch_e060_butterfly_transverse.png")
Z = 160
# Coordinates after np.rot90(volume[:, :, Z]): rows, columns.
R0, R1, C0, C1 = 55, 150, 80, 185


def axial(v):
    return np.rot90(np.abs(v[:, :, Z]))


d = np.load(NPZ)
fig, ax = plt.subplots(2, 5, figsize=(15.5, 6.3), facecolor="white")
for row, (name, k) in enumerate((("T1W", "T1"), ("FLAIR", "FL"))):
    gt = axial(d[f"gt{k}"])
    rec = axial(d[f"rec{k}"])
    peak = float(np.max(np.abs(d[f"gt{k}"])))
    vmax = float(np.quantile(gt[gt > 0.01 * peak], 0.995))
    crop_gt = gt[R0:R1, C0:C1]
    crop_rec = rec[R0:R1, C0:C1]
    err = np.abs(crop_rec - crop_gt) / peak
    for col, image, cmap, lo, hi in (
        (0, gt, "gray", 0, vmax),
        (1, rec, "gray", 0, vmax),
        (2, crop_gt, "gray", 0, vmax),
        (3, crop_rec, "gray", 0, vmax),
        (4, err, "magma", 0, 0.10),
    ):
        ax[row, col].imshow(image, cmap=cmap, vmin=lo, vmax=hi)
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
        for spine in ax[row, col].spines.values():
            spine.set_visible(True)
            spine.set_color("#C9C2B8")
            spine.set_linewidth(0.7)
    # Rectangle uses x=column, y=row in the full transverse image.
    for col in (0, 1):
        ax[row, col].add_patch(Rectangle((C0, R0), C1 - C0, R1 - R0,
                                         fill=False, edgecolor="#C2413B", linewidth=1.4))
    ax[row, 0].set_ylabel(name, fontsize=12, fontweight="semibold")

titles = ("GT · full transverse", "eps=0.06 · full transverse",
          "GT · butterfly ROI", "eps=0.06 · butterfly ROI", "|recon−GT| · ROI")
for col, title in enumerate(titles):
    ax[0, col].set_title(title, fontsize=10.5, fontweight="semibold" if col == 3 else "normal")
fig.suptitle("AF≈12 tuned reconstruction · transverse z=160 · butterfly-region localization",
             fontsize=14, fontweight="semibold", y=0.98)
fig.text(0.5, 0.015, "Red box marks the ventricular butterfly ROI; error uses a shared [0, 0.10] GT-peak scale.",
         ha="center", fontsize=9.5, color="#5B6470")
fig.tight_layout(rect=[0.02, 0.045, 0.995, 0.94], w_pad=0.8)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=220, bbox_inches="tight")
print(OUT)
