#!/usr/bin/env python3
"""fig_mask_support.py -- the one AF24 mask against each contrast's acquired k-space region.

The acquired region is read from each contrast's own fully sampled k-space (coil 0; the vendor shutter zeros whole
readout lines, identically in every coil). Panels: the mask with the FLAIR shutter edge; T1 and FLAIR with the mask
points that land on acquired data (white) and on the unacquired corners (red). AF = PE-plane points / sampled points
that carry that contrast's data (user decision 2026-09-14). Writes
figures/t1flair_v3_af24_mask_T1_vs_FLAIR.png and the same file in the served figs folder.
"""
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

R = Path(__file__).resolve().parents[1]
m = sio.loadmat(R / "data/mask_vdpois_256x176_af24.mat")["mask"].astype(bool)
sup = {}
for c in ("T1", "FLAIR"):
    with h5py.File(R / f"data/ksp_{c}.h5", "r") as f:
        sup[c] = np.any(np.abs(np.array(f["kspace"][0])) > 0, axis=0)
fig, axes = plt.subplots(1, 3, figsize=(14, 6.4))
axes[0].imshow(m, cmap="gray", interpolation="nearest")
axes[0].contour(sup["FLAIR"], levels=[0.5], colors="r", linewidths=1.2)
axes[0].set_title(f"One AF24 mask, shared by T1 and FLAIR\n{int(m.sum())} of {m.size} points sampled\n"
                  f"red line: edge of the FLAIR acquired region", fontsize=10.5)
for a, c in zip(axes[1:], ("T1", "FLAIR")):
    rgb = np.zeros(m.shape + (3,))
    rgb[sup[c]] = 0.18
    rgb[m & sup[c]] = 1.0
    rgb[m & ~sup[c]] = (1.0, 0.1, 0.1)
    a.imshow(rgb, interpolation="nearest")
    n_in, n_out = int((m & sup[c]).sum()), int((m & ~sup[c]).sum())
    a.set_title(f"{c}: acquired region {100 * sup[c].mean():.1f}% of the plane (grey)\n"
                f"{n_in} points on data (white), {n_out} on corners (red)\n"
                f"AF = {m.size} / {int(m.sum())} sampled points = {m.size / m.sum():.2f} (one mask, both contrasts)\n"
                f"{n_in} of them carry {c} data", fontsize=10.5)
for a in axes:
    a.set_xlabel("kz (176)"); a.set_ylabel("ky (256)"); a.set_xticks([]); a.set_yticks([])
fig.suptitle("jiangyizhou retrospective AF 24: one mask, two contrasts   (AF = whole-plane points / sampled points, user 2026-09-16)", fontsize=12.5, fontweight="bold")
fig.tight_layout(w_pad=2.5)
for d in (Path("/home/congcongliu/figs/t1flair_v3_af24"), R / "figures"):
    fig.savefig(d / "t1flair_v3_af24_mask_T1_vs_FLAIR.png", dpi=600)
    fig.savefig(d / "t1flair_v3_af24_mask_T1_vs_FLAIR.pdf")
print("wrote", R / "figures/t1flair_v3_af24_mask_T1_vs_FLAIR.png")
