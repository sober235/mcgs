#!/usr/bin/env python3
"""fig_kspace_full_jyz.py -- the fully sampled k-space of the jiangyizhou T1 + FLAIR pair, as stored in every jyz topic's
data/ksp_{T1,FLAIR}.h5 (coil-first (26, x, ky, kz), the AF12 topic's export of the CFL source), for inspection
(user 2026-09-15). Per contrast: log10 |k| root-sum-of-squares over the 26 coils on the (ky, kz) plane at the readout
index of peak energy; the same on the (kx, ky) plane at kz = 88; and the acquired support -- the (ky, kz) lines with any
non-zero sample along the readout in coil 0, the rule behind the red shutter line of the mask figures. Read-only.
Writes figures/t1flair_v3_jyz_full_kspace.png to both diff-mask topics and the served figs folder.
"""
import shutil
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path("/data0/congcong/code/GS/Results_20260908")
TOPICS = [RES / "T1FLAIR_AF16_gauss_v3_jiangyizhou", RES / "T1FLAIR_AF16_radial_v3_jiangyizhou"]
SERVED = Path("/home/congcongliu/figs/t1flair_v3_af16_diffmask")
NAME = "t1flair_v3_jyz_full_kspace.png"
CY, CZ = 128, 88

fig, axes = plt.subplots(2, 3, figsize=(17, 11.5), gridspec_kw=dict(width_ratios=[1, 1.35, 1]))
for row, c in enumerate(("T1", "FLAIR")):
    with h5py.File(TOPICS[0] / f"data/ksp_{c}.h5", "r") as f:              # symlink into the AF12 topic: READ ONLY
        d = f["kspace"]
        k0 = np.array(d[0])                                                  # coil 0, (x, ky, kz)
        ex = (np.abs(k0) ** 2).sum(axis=(1, 2))
        xc = int(np.argmax(ex))
        sup = np.any(np.abs(k0) > 0, axis=0)                                 # (ky, kz) lines with data
        pe = np.sqrt((np.abs(np.array(d[:, xc, :, :])) ** 2).sum(axis=0))    # (ky, kz) at the peak readout index, RSS
        ro = np.sqrt((np.abs(np.array(d[:, :, :, CZ])) ** 2).sum(axis=0))    # (x, ky) at kz = 88, RSS
    nl = int(sup.sum())
    top = pe.max()
    a = axes[row, 0]
    im = a.imshow(np.log10(pe / top + 1e-7), cmap="magma", vmin=-6, vmax=0, interpolation="nearest")
    a.contour(sup.astype(float), levels=[0.5], colors="r", linewidths=1.0)
    a.set_title(f"{c}: log10 |k| / max, RSS over 26 coils\n(ky, kz) plane at readout index {xc} (peak energy)\n"
                f"red: edge of the acquired region", fontsize=10)
    a.set_xlabel("kz (176)"); a.set_ylabel("ky (256)")
    plt.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    a = axes[row, 1]
    im = a.imshow(np.log10(ro.T / ro.max() + 1e-7), cmap="magma", vmin=-6, vmax=0, interpolation="nearest", aspect="auto")
    a.set_title(f"{c}: log10 |k| / max, RSS over 26 coils\n(kx, ky) plane at kz = {CZ}", fontsize=10)
    a.set_xlabel("readout kx (256)"); a.set_ylabel("ky (256)")
    plt.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    a = axes[row, 2]
    a.imshow(sup, cmap="gray", interpolation="nearest")
    a.set_title(f"{c}: acquired support (any non-zero sample along the readout, coil 0)\n"
                f"{nl} of {sup.size} (ky, kz) lines = {100 * nl / sup.size:.1f} % of the plane", fontsize=10)
    a.set_xlabel("kz (176)"); a.set_ylabel("ky (256)")
    print(f"[{c}] peak readout index {xc} (energy share of that index {ex[xc] / ex.sum():.3f}); "
          f"support {nl}/{sup.size} = {nl / sup.size:.4f}; |k| max at (ky, kz) = {np.unravel_index(pe.argmax(), pe.shape)}; "
          f"ky rows with data: {np.flatnonzero(sup.any(1)).min()}-{np.flatnonzero(sup.any(1)).max()}, "
          f"kz cols with data: {np.flatnonzero(sup.any(0)).min()}-{np.flatnonzero(sup.any(0)).max()}", flush=True)
fig.suptitle("jiangyizhou: the fully sampled k-space behind every jyz topic (data/ksp_T1.h5, data/ksp_FLAIR.h5)", fontsize=12)
fig.tight_layout()
SERVED.mkdir(parents=True, exist_ok=True)
first = TOPICS[0] / "figures" / NAME
fig.savefig(first, dpi=130)
for T in TOPICS[1:]:
    shutil.copy2(first, T / "figures" / NAME)
shutil.copy2(first, SERVED / NAME)
print(f"http://localhost:8765/{SERVED.name}/{NAME}\n{SERVED / NAME}\n" + "\n".join(str(T / 'figures' / NAME) for T in TOPICS))
