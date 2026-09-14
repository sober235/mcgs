#!/usr/bin/env python3
"""fig_mask_reinspect.py -- the one AF 24 mask as each channel loads it (cloned from the AF20 re-inspection of 2026-09-14, read-only on data/).

Top row: the T1 channel's and the FLAIR channel's mask_path from configs/t1flair_v3_af24_jyz.yaml, and their XOR (0 lines).
Bottom row: log10 per-line k-space energy of each contrast on the mask, and FLAIR's full k-space with its acquired region.
AF = PE-plane lines / sampled lines that carry that contrast's data (user decision 2026-09-14): T1 45056/1877 = 24.00,
FLAIR 45056/1766 = 25.51 (111 lines fall in FLAIR's unacquired corners). Writes figures/ and the served figs folder."""
import hashlib, re, shutil
import h5py, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou"
CFG = "/data0/congcong/code/GS/mcgs_af16_wt/configs/t1flair_v3_af24_jyz.yaml"
OUT = f"{R}/figures/t1flair_v3_af24_mask_T1_FLAIR_reinspect.png"
SERVED = "/home/congcongliu/figs/t1flair_v3_af24"
torch.set_num_threads(16)

paths = re.findall(r"mask_path:\s*(\S+)", open(CFG).read())          # [plain=T1, contrast=FLAIR]
assert len(paths) == 2, paths
def load_mask(p):
    with h5py.File(p, "r") as f:
        m3 = f["mask"][:] > 0
    assert (m3 == m3[0:1]).all(), "mask not constant along the readout"
    return m3[0]
mT1, mFL = load_mask(paths[0]), load_mask(paths[1])
md5 = {p: hashlib.md5(open(p, "rb").read()).hexdigest()[:8] for p in set(paths)}
N = mT1.size

def line_energy(contrast):                                              # sum over coils and readout of |k|^2, per PE line
    with h5py.File(f"{R}/data/ksp_{contrast}.h5", "r") as f:
        y = torch.from_numpy(f["kspace"][:])
    return (y.real ** 2 + y.imag ** 2).sum(dim=(0, 1)).numpy()
eT1, eFL = line_energy("T1"), line_energy("FLAIR")
supFL = eFL > 0

nT1, nFL, ndiff = int(mT1.sum()), int(mFL.sum()), int((mT1 ^ mFL).sum())
fl_sig, fl_empty = int((mFL & supFL).sum()), int((mFL & ~supFL).sum())
t1_sig = int((mT1 & (eT1 > 0)).sum())
print(f"T1 mask {paths[0]} md5 {md5[paths[0]]}: {nT1} lines")
print(f"FLAIR mask {paths[1]} md5 {md5[paths[1]]}: {nFL} lines; lines differing from the T1 mask {ndiff}")
print(f"T1 sampled lines with signal {t1_sig} -> AF {N / t1_sig:.2f}; FLAIR sampled lines with signal {fl_sig} -> AF {N / fl_sig:.2f}, without {fl_empty}; FLAIR acquired region {int(supFL.sum())}")

def logimg(e, keep):
    v = np.full(e.shape, np.nan); ok = keep & (e > 0); v[ok] = np.log10(e[ok]); return v
lo = np.nanpercentile(np.log10(np.r_[eT1[eT1 > 0], eFL[eFL > 0]]), 1); hi = np.log10(max(eT1.max(), eFL.max()))
cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("black")

fig, ax = plt.subplots(2, 3, figsize=(14, 17.5))
kw = dict(interpolation="nearest", aspect="equal")
ax[0, 0].imshow(mT1, cmap="gray", **kw)
ax[0, 0].set_title(f"T1 mask (the T1 channel's mask_path)\n{nT1} of {N} PE lines", fontsize=11)
ax[0, 1].imshow(mFL, cmap="gray", **kw)
ax[0, 1].set_title(f"FLAIR mask (the FLAIR channel's mask_path)\n{nFL} of {N} PE lines", fontsize=11)
ax[0, 2].imshow(mT1 ^ mFL, cmap="gray", vmin=0, vmax=1, **kw)
ax[0, 2].set_title(f"T1 mask XOR FLAIR mask\n{ndiff} lines differ\n(both read mask_af24_3d.h5, md5 {md5[paths[0]]})", fontsize=11)
ax[1, 0].imshow(logimg(eT1, mT1), cmap=cmap, vmin=lo, vmax=hi, **kw)
ax[1, 0].set_title(f"T1 k-space on the mask (log10 line energy)\n{t1_sig} of {nT1} sampled lines carry signal\nAF = {N} / {t1_sig} = {N / t1_sig:.2f}", fontsize=11)
ax[1, 1].imshow(logimg(eFL, mFL), cmap=cmap, vmin=lo, vmax=hi, **kw)
yy, zz = np.nonzero(mFL & ~supFL)
ax[1, 1].scatter(zz, yy, s=9, c="red", marker="s", linewidths=0)
ax[1, 1].contour(supFL, levels=[0.5], colors="r", linewidths=0.9)
ax[1, 1].set_title(f"FLAIR k-space on the mask (log10 line energy)\n{fl_sig} carry signal, {fl_empty} (red) fall where FLAIR has no data\nAF = {N} / {fl_sig} = {N / fl_sig:.2f}   (red line = acquired region)", fontsize=11)
ax[1, 2].imshow(logimg(eFL, np.ones_like(supFL)), cmap=cmap, vmin=lo, vmax=hi, **kw)
ax[1, 2].contour(supFL, levels=[0.5], colors="r", linewidths=0.9)
ax[1, 2].set_title(f"FLAIR full k-space (log10 line energy)\nacquired region {int(supFL.sum())} of {N} lines\n(vendor shutter; rows ky < 20 not acquired)", fontsize=11)
for a in ax.ravel():
    a.set_xlabel("kz (176)"); a.set_ylabel("ky (256)"); a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"jiangyizhou AF24 topic: one mask for T1 and FLAIR   (AF = whole-plane lines / sampled lines that carry data:  T1 {N / t1_sig:.2f},  FLAIR {N / fl_sig:.2f})",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT, dpi=130); shutil.copy2(OUT, SERVED)
print("wrote", OUT)
