#!/usr/bin/env python3
"""fig_raw_kspace_support.py -- which phase-encode lines the raw jiangyizhou T1 and FLAIR scans actually acquired (2026-09-15).

Reads the raw BART CFL k-space (every readout sample, every coil) and sums |k|^2 per (ky, kz) line. A line counts as
not acquired when its energy is exactly zero. T1: 0 of 45056 lines empty. FLAIR: 10917 empty -- the k-space corners
outside the inscribed ellipse (vendor elliptical scanning) and the first 20 ky rows -- 34139 acquired. Right panels:
the AF 24 mask on each contrast (white = sampled on data, red = sampled where that contrast has no data). Writes
figures/t1flair_v3_af24_raw_kspace_support_T1_vs_FLAIR.png and the served figs folder.
"""
import shutil
import numpy as np, scipy.io as sio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SRC = "/data1/data1_congcong/data/MC_Jia/Train_SpiritDiffusion_CFL/vol3_jiangyizhou"
R = "/data0/codec" if False else "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou"
OUT = f"{R}/figures/t1flair_v3_af24_raw_kspace_support_T1_vs_FLAIR.png"
SERVED = "/home/congcongliu/figs/t1flair_v3_af24"


def line_energy(base):
    dims = [int(x) for x in open(base + ".hdr").read().split("\n")[1].split()]
    k = np.memmap(base + ".cfl", dtype=np.complex64, mode="r", shape=(int(np.prod(dims)),)).reshape(dims, order="F")
    e = np.zeros(dims[1:3])
    for x in range(dims[0]):
        e += (np.abs(np.asarray(k[x, :, :, :, 0])) ** 2).sum(axis=-1)
    return e


E = {"T1": line_energy(f"{SRC}/t1_fsp_train_jiangyizhou_cc"), "FLAIR": line_energy(f"{SRC}/flair_train_jiangyizhou_cc")}
m = sio.loadmat(f"{R}/data/mask_vdpois_256x176_af24.mat")["mask"].astype(bool)
lo = np.log10(min(v[v > 0].min() for v in E.values())); hi = np.log10(max(v.max() for v in E.values()))
fig, ax = plt.subplots(1, 4, figsize=(17, 6.6))
for i, c in enumerate(("T1", "FLAIR")):
    e = E[c]; acq = e > 0
    img = np.full(e.shape, np.nan); img[acq] = np.log10(e[acq])
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("red")
    ax[i].imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    ax[i].set_title(f"{c} raw k-space, log10 energy per PE line\n(sum over 256 readout samples x 26 coils)\n"
                    f"red = exactly zero, not acquired: {int((~acq).sum())} of {e.size} lines", fontsize=10.5)
    rgb = np.zeros(e.shape + (3,)); rgb[acq] = 0.18; rgb[m & acq] = 1.0; rgb[m & ~acq] = (1.0, 0.1, 0.1)
    ax[2 + i].imshow(rgb, interpolation="nearest")
    n_on = int((m & acq).sum())
    ax[2 + i].set_title(f"AF 24 mask on {c}: {int(m.sum())} sampled points\n{n_on} on acquired data (white), "
                        f"{int((m & ~acq).sum())} on empty lines (red)\n"
                        f"AF = {e.size} / {int(m.sum())} = {e.size / m.sum():.2f} (label, both contrasts); "
                        f"{e.size / n_on:.2f} against the points carrying data", fontsize=10.5)
for a in ax:
    a.set_xlabel("kz (176)"); a.set_ylabel("ky (256)"); a.set_xticks([]); a.set_yticks([])
fig.suptitle("jiangyizhou raw scans: T1 acquired the whole phase-encode plane; FLAIR skipped the corners (elliptical scanning) "
             "and ky rows 0-19", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=130); shutil.copy2(OUT, SERVED)
print("wrote", OUT)
