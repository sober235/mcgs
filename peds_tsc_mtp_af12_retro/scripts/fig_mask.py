#!/usr/bin/env python3
"""fig_mask.py -- the sampling mask, the known zeros, the acquired support and their union."""
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

R = Path("/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro")
F = Path("/home/congcongliu/figs/peds_tsc_af12"); F.mkdir(parents=True, exist_ok=True)
s = json.load(open(R / "logs/prep_mask.json"))
omega = scipy.io.loadmat(R / "data/mask_af12_2d.mat")["mask"].astype(bool)
d = np.load(R / "data/masks_T1w.npz")
sup, zero = d["envelope"].astype(bool), d["m_zero"].astype(bool)

panels = [(omega, f"sampling mask Omega\n{int(omega.sum())} points, AF {s['AF_vs_full_grid']:.2f} of the "
                  f"{sup.size}-point matrix\n({s['AF_vs_support']:.2f}x of the {int(sup.sum())} acquired positions)"),
          (sup, f"acquired support S\n{int(sup.sum())} of {sup.size} positions\n(elliptical envelope, PE 0..226)"),
          (zero, f"known zeros Z\n{int(zero.sum())} positions\n(partial Fourier PE 227..303 and the corners)"),
          (omega | zero, f"forward-operator support Omega | Z\n{int((omega | zero).sum())} positions "
                         f"({100 * (omega | zero).mean():.1f} % of the plane)\ngiven to every method alike")]
fig, axes = plt.subplots(1, 4, figsize=(17, 6.4))
for a, (m, t) in zip(axes, panels):
    rgb = np.zeros((*sup.shape, 3)); rgb[sup] = 0.18; rgb[m] = 1.0
    a.imshow(rgb, interpolation="nearest", aspect="auto")
    a.set_title(t, fontsize=10)
    a.set_xlabel("SPE (104)"); a.set_ylabel("PE (304)"); a.set_xticks([]); a.set_yticks([])
fig.suptitle("16-year-old MTP, retrospective AF 12: one mask for both contrasts (grey = the acquired support)",
             fontsize=13, fontweight="bold")
fig.tight_layout(w_pad=2.0)
for p in (F / "mask_af12.png", R / "figures/mask_af12.png"):
    fig.savefig(p, dpi=140)
print("wrote", F / "mask_af12.png")
