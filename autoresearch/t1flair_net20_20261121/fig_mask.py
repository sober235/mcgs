#!/usr/bin/env python3
"""fig_mask.py -- the net-20x mask next to the existing masks of the series (display only, CPU)."""
import os
from pathlib import Path
import h5py, numpy as np, scipy.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = Path("/data0/congcong/code/GS/Results_20260908")
M = [("nominal AF16", "T1FLAIR_AF16_v3_jiangyizhou", "af16"), ("nominal AF20", "T1FLAIR_AF20_v3_jiangyizhou", "af20"),
     ("NEW: net 20x", "T1FLAIR_AF20net_v3_jiangyizhou", "af20n"), ("AF24", "T1FLAIR_AF24_v3_jiangyizhou", "af24")]
with h5py.File(T / "T1FLAIR_AF12_v3_jiangyizhou/data/ksp_FLAIR.h5", "r") as f:
    sup = np.any(np.abs(np.array(f["kspace"][0])) > 0, axis=0)
fig, ax = plt.subplots(2, 4, figsize=(17, 12.4))
for j, (lab, topic, tag) in enumerate(M):
    m = scipy.io.loadmat(T / topic / f"data/mask_vdpois_256x176_{tag}.mat")["mask"].astype(bool)
    n, cal = int(m.sum()), int(m[116:140, 76:100].sum())
    ax[0, j].imshow(m, cmap="gray", interpolation="nearest")
    ax[0, j].set_title(f"{lab}\n{n} of {m.size} points = net {m.size / n:.2f}x\n24x24 centre: {cal}/576",
                       fontsize=13, fontweight="bold" if tag == "af20n" else "normal", color="#C00000" if tag == "af20n" else "k")
    rgb = np.zeros(m.shape + (3,)); rgb[sup] = 0.22
    rgb[m & sup] = 1.0; rgb[m & ~sup] = (1.0, 0.25, 0.25)
    ax[1, j].imshow(rgb, interpolation="nearest")
    ax[1, j].set_title(f"on FLAIR's acquired region (grey)\n{int((m & sup).sum())} points carry FLAIR data, {int((m & ~sup).sum())} outside (red)", fontsize=11)
    for a in ax[:, j]:
        a.set_xlabel("kz (176)"); a.set_ylabel("ky (256)")
fig.suptitle("jiangyizhou T1 + FLAIR: vd-Poisson masks on the phase-encode plane (one mask for both contrasts); "
             "net acceleration = plane points / sampled points, calibration region included", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = Path(os.environ.get("MCGS_ARTIFACT_DIR", ".")) / "mask_net20.png"
fig.savefig(out, dpi=110); print("wrote", out)
