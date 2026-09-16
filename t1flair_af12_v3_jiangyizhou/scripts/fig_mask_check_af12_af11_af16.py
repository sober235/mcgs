#!/usr/bin/env python
"""fig_mask_check_af12_af11_af16.py -- the three vd-Poisson masks of this subject side by side:
AF 12 (this topic, the T1 panels), AF 11 (this topic, the design-B FLAIR mask) and AF 16 (the other
topic). Top row is the full (ky, kz) phase-encode plane, bottom row the fully sampled centre.

Rewritten 2026-09-16 from the mask files: the 2026-09-13 figure of the same name was drawn ad hoc and
left no script, so there was no way to re-export it as a vector PDF. Every number on the figure is
recomputed here from the masks themselves, not copied off the old PNG.

Writes figures/mask_check_af12_af11_af16.{png,pdf} and mirrors both to the served folder.
"""
import shutil
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42          # TrueType: submission systems reject Type 3
import matplotlib.pyplot as plt

R = Path("/data0/congcong/code/GS/Results_20260908")
A12 = R / "T1FLAIR_AF12_v3_jiangyizhou"
SERVED = Path("/home/congcongliu/figs/t1flair_v3_af12")
CENTRE = 56                                        # side of the centre crop in the bottom row

MASKS = [("AF 12", "this topic, T1 panels",      A12 / "data/mask_vdpois_256x176_af12.mat"),
         ("AF 11", "this topic, design-B FLAIR", A12 / "data/mask_vdpois_256x176_af11.mat"),
         ("AF 16", "the other topic",            R / "T1FLAIR_AF16_v3_jiangyizhou/data/mask_vdpois_256x176_af16.mat")]

m = [sio.loadmat(p)["mask"].astype(bool) for _, _, p in MASKS]
NPL = m[0].size
assert all(x.shape == m[0].shape for x in m), [x.shape for x in m]

shared = int((m[0] & m[2]).sum())
fig, ax = plt.subplots(2, 3, figsize=(11.2, 8.4))
fig.suptitle(
    f"The three vd-Poisson masks of this subject, full (ky, kz) plane {m[0].shape[0]} x {m[0].shape[1]} (top) "
    f"and the fully sampled centre (bottom).\n"
    "Same generator and the same 24x24 calibration block, so they look alike at figure scale; "
    "the point counts and the patterns differ.\n"
    f"Overlap: AF12 vs AF16 {shared} shared points = {100 * shared / m[0].sum():.0f} % of AF12 / "
    f"{100 * shared / m[2].sum():.0f} % of AF16 (independent realisations).", fontsize=11)

h = CENTRE // 2
for j, ((name, where, _), mm) in enumerate(zip(MASKS, m)):
    n = int(mm.sum())
    ax[0, j].imshow(mm.T, cmap="gray", interpolation="nearest", aspect="equal")
    ax[0, j].set_title(f"{name}  ({where})\n{n} / {NPL} PE points = {100 * n / NPL:.2f} %   "
                       f"(AF {NPL / n:.2f})", fontsize=10.5)
    c = mm[mm.shape[0] // 2 - h: mm.shape[0] // 2 + h, mm.shape[1] // 2 - h: mm.shape[1] // 2 + h]
    ax[1, j].imshow(c.T, cmap="gray", interpolation="nearest", aspect="equal")
    ax[1, j].set_title(f"centre {CENTRE}x{CENTRE}: {100 * c.mean():.1f} % sampled", fontsize=10.5)
    for a in (ax[0, j], ax[1, j]):
        a.set_xticks([]); a.set_yticks([])

fig.tight_layout(rect=(0, 0, 1, 0.93))
out = A12 / "figures/mask_check_af12_af11_af16.png"
fig.savefig(out, dpi=600)
fig.savefig(out.with_suffix(".pdf"))
for s in (out, out.with_suffix(".pdf")):
    shutil.copy2(s, SERVED / s.name)
print("wrote", out, "and", out.with_suffix(".pdf").name)
print(f"  points: " + ", ".join(f"{n[0]} {int(x.sum())}" for n, x in zip(MASKS, m)) +
      f";  AF12&AF16 shared {shared}")
