"""Transverse z=160 ("butterfly" ventricle slice): AF12 vs AF10, GT | ZF | recon | error(jet [0,0.1])
plus the ventricular ROI zoom (rows 55:150, cols 80:185 after rot90, same box as scripts/fig_af12_butterfly_roi.py).
usage: fig_butterfly_z160.py <af12.npz> <af10.npz> <out_prefix>
"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
Z = 160; R0, R1, C0, C1 = 55, 150, 80, 185
runs = [("AF12 (net 11.83)", np.load(sys.argv[1])), ("AF10 (net 9.87)", np.load(sys.argv[2]))]
out_prefix = sys.argv[3]
ax_ = lambda v: np.rot90(np.abs(v[:, :, Z]))
def psnr(a, g, peak): return 20 * np.log10(peak / np.sqrt(np.mean((a - g) ** 2)))
for con, k in (("T1", "T1"), ("FLAIR", "FL")):
    fig, axes = plt.subplots(2, 7, figsize=(25, 7.8))
    for r, (name, d) in enumerate(runs):
        gt3 = d[f"gt{k}"].astype(np.float32); peak = float(gt3.max())
        gt, zf, rec = ax_(gt3), ax_(d[f"zf{k}"]), ax_(d[f"rec{k}"])
        vmax = float(np.quantile(gt[gt > 0.01 * peak], 0.995))
        err = np.abs(rec - gt) / peak
        roi = (slice(R0, R1), slice(C0, C1))
        head = gt > 0.05 * np.quantile(gt3, 0.999)
        p_slice = psnr(rec[head], gt[head], peak); p_zf = psnr(zf[head], gt[head], peak)
        p_roi = psnr(rec[roi], gt[roi], peak); nmse_roi = ((rec[roi] - gt[roi]) ** 2).sum() / (gt[roi] ** 2).sum()
        panels = [("GT (red box = butterfly ROI)", gt, "gray", 0, vmax), (f"zero-filled  head-PSNR {p_zf:.2f} dB", zf, "gray", 0, vmax),
                  (f"recon  head-PSNR {p_slice:.2f} dB", rec, "gray", 0, vmax), ("|recon-GT|/max(GT)", err, "jet", 0, 0.1),
                  ("GT ROI", gt[roi], "gray", 0, vmax), (f"recon ROI  PSNR {p_roi:.2f} dB, NMSE {nmse_roi*100:.2f}%", rec[roi], "gray", 0, vmax),
                  ("error ROI", err[roi], "jet", 0, 0.1)]
        for c, (title, img, cmap, lo, hi) in enumerate(panels):
            ax = axes[r, c]; im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
            if cmap == "jet":
                cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.04); fig.colorbar(im, cax=cax)
            if c in (0, 1, 2, 3):
                ax.add_patch(Rectangle((C0, R0), C1 - C0, R1 - R0, fill=False, edgecolor="red", linewidth=1.2))
        axes[r, 0].set_ylabel(name, fontsize=12)
    fig.suptitle(f"{con}  transverse z={Z} (ventricular butterfly slice)  AF12 tvs4k_e040 vs AF10 rerun 2026-09-04  |  error: jet, fixed [0, 0.1] of GT peak", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = f"{out_prefix}_{con}.png"; fig.savefig(out, dpi=110); plt.close(fig); print("saved", out)
