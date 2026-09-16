#!/usr/bin/env python3
"""fig_mask_series_af24.py -- the jiangyizhou house masks side by side: the AF12, AF16 and AF20 topics' masks and the AF24 one.

Each panel shows the sampled points (white) and the edge of FLAIR's acquired region (red). Titles carry the generator's
accelx, the point count, the 24x24 centre, and the acceleration under the user's definition (2026-09-14): PE-plane
points / sampled points that carry that contrast's data (T1 = the whole plane; FLAIR = only where its shutter acquired
data). Read-only on every topic's data/. Writes figures/t1flair_v3_af24_mask_series_af12_16_20_24.png and the served
figs folder.
"""
import json
import shutil
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

RES = Path("/data0/congcong/code/GS/Results_20260908")
R = RES / "T1FLAIR_AF24_v3_jiangyizhou"
PICK = json.load(open(R / "logs/prep_03_pick_mask_af24.json"))
MASKS = [("AF12 topic", "3", RES / "T1FLAIR_AF12_v3_jiangyizhou/data/mask_vdpois_256x176_af12.mat"),
         ("AF16 topic", "4", RES / "T1FLAIR_AF16_v3_jiangyizhou/data/mask_vdpois_256x176_af16.mat"),
         ("AF20 topic", "5", RES / "T1FLAIR_AF20_v3_jiangyizhou/data/mask_vdpois_256x176_af20.mat"),
         ("AF24 topic (new)", f"{PICK['chosen']['accelx']:.4f}", R / "data/mask_vdpois_256x176_af24.mat")]
OUT = R / "figures/t1flair_v3_af24_mask_series_af12_16_20_24.png"
SERVED = Path("/home/congcongliu/figs/t1flair_v3_af24")

with h5py.File(R / "data/ksp_FLAIR.h5", "r") as f:              # symlink into the AF12 topic: READ ONLY
    sup = np.any(np.abs(np.array(f["kspace"][0])) > 0, axis=0)
assert int(sup.sum()) == 34139

fig, axes = plt.subplots(1, 4, figsize=(17, 7.4))
for a, (name, accelx, p) in zip(axes, MASKS):
    m = sio.loadmat(p)["mask"].astype(bool)
    n, n_fl = int(m.sum()), int((m & sup).sum())
    cal = int(m[128 - 12:128 + 12, 88 - 12:88 + 12].sum())
    a.imshow(m, cmap="gray", interpolation="nearest")
    a.contour(sup, levels=[0.5], colors="r", linewidths=1.0)
    a.set_title(f"{name}, accelx {accelx}\n{n} of {m.size} points, 24x24 centre {cal}/576\n"
                f"AF {m.size / n:.2f}x on both contrasts;  {n_fl} points carry FLAIR data ({n - n_fl} outside its scan)", fontsize=11)
    a.set_xlabel("kz (176)"); a.set_ylabel("ky (256)"); a.set_xticks([]); a.set_yticks([])
    print(f"{name}: {p.name} accelx {accelx} n {n} centre {cal} T1 {m.size / n:.3f} FLAIR {m.size / n_fl:.3f} outside {n - n_fl}")
fig.suptitle("jiangyizhou house vd-Poisson masks, vdPoisMex(256,176,256,176, accelx, 4, 24, 0, 1); red line = edge of FLAIR's acquired region\n"
             "AF = whole-plane points / sampled points (user 2026-09-16): one mask, one number for both contrasts", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.0)
fig.savefig(OUT, dpi=600)
fig.savefig(OUT.with_suffix(".pdf"))
shutil.copy2(OUT.with_suffix(".pdf"), SERVED / OUT.with_suffix(".pdf").name)
shutil.copy2(OUT, SERVED / OUT.name)
print("wrote", OUT)
