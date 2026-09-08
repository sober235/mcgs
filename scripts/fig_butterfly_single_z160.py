"""Transverse z=160 (ventricular butterfly slice) of one run: GT | recon | error | ROI GT | ROI recon | ROI error, T1 and FLAIR.
usage: fig_butterfly_single_z160.py <volumes.npz> <out.png> "<label>"
"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
Z = 160; R0, R1, C0, C1 = 55, 150, 80, 185
npz, out, label = sys.argv[1:4]
z = np.load(npz)
sl = lambda v: np.rot90(np.abs(v[:, :, Z]))
fig, axes = plt.subplots(2, 6, figsize=(24, 8.4))
for r, (con, rk, gk) in enumerate([("T1", "recT1", "gtT1"), ("FLAIR", "recFL", "gtFL")]):
    gt = sl(z[gk]); rec = sl(z[rk]); gmax = float(np.abs(z[gk]).max())
    head = gt > 0.05 * np.percentile(gt, 99.9)
    err = np.abs(rec - gt) / gmax
    psnr_head = 20 * np.log10(gmax / np.sqrt(np.mean((rec[head] - gt[head]) ** 2)))
    gt_roi, rec_roi, err_roi = gt[R0:R1, C0:C1], rec[R0:R1, C0:C1], err[R0:R1, C0:C1]
    psnr_roi = 20 * np.log10(gmax / np.sqrt(np.mean((rec_roi - gt_roi) ** 2)))
    vmax = float(np.percentile(gt[head], 99.5))
    panels = [(gt, "GT (red box = butterfly ROI)", "gray"), (rec, f"{label} recon  slice head-PSNR {psnr_head:.2f} dB", "gray"),
              (err, "|recon-GT|/max(GT)", "jet"), (gt_roi, "GT ROI", "gray"),
              (rec_roi, f"recon ROI  PSNR {psnr_roi:.2f} dB", "gray"), (err_roi, "error ROI", "jet")]
    for c, (img, title, cmap) in enumerate(panels):
        ax = axes[r, c]
        if cmap == "gray":
            ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
        else:
            im = ax.imshow(img, cmap="jet", vmin=0, vmax=0.1)
            cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.04); fig.colorbar(im, cax=cax)
        if c < 3:
            ax.add_patch(Rectangle((C0, R0), C1 - C0, R1 - R0, fill=False, edgecolor="red", lw=1.2))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=11)
        if c == 0: ax.set_ylabel(con, fontsize=14)
fig.suptitle(f"{label} — transverse z=160 ventricular butterfly slice — error jet fixed [0, 0.1] of GT peak", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=120); print("saved", out)
