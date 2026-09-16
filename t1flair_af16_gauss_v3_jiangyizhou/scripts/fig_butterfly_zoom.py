#!/usr/bin/env python3
"""fig_butterfly_zoom.py -- the lateral-ventricle ("butterfly") region of one transverse slice, zoomed.

Same slice as fig_slice.py (axis0 = 168 for this subject), cropped to a window around the head
centroid so the ventricles fill the panel. Two blocks of five: GT | ZF | CG-SENSE | CS-SENSE | ZS-SSDU
and 3D-DIP | 3D-INR | MC-GS single | MC-GS joint | MC-GS joint @AF12 (same subject, same slice, same
GT; reference only). Each block: magnitude row + error row (jet, FIXED [0, 0.1] of the GT peak).
Titles carry the head-masked PSNR of the CROP only.
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", required=True, choices=["T1", "FLAIR"])
ap.add_argument("--axis", type=int, default=0)
ap.add_argument("--index", type=int, default=168)
ap.add_argument("--half", type=int, nargs=2, default=[52, 38], help="crop half-widths along the two in-plane axes")
ap.add_argument("--shift", type=int, nargs=2, default=[0, 0], help="crop centre offset from the head centroid")
args = ap.parse_args()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_gauss_v3_jiangyizhou")
R12 = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou")
F = Path("/home/congcongliu/figs/t1flair_v3_af16_diffmask_gauss"); F.mkdir(parents=True, exist_ok=True)
LABEL = {"zf": "Zero-filled", "cgsense": "CG-SENSE", "cssense": "CS-SENSE (TV)", "zsssdu": "ZS-SSDU",
         "dip": "3D-DIP", "inr": "3D-INR", "gs_single": "MC-GS single", "gs_joint": "MC-GS joint",
         "gs_joint@AF12": "MC-GS joint @AF12 (ref.)"}


def load(root, m, c):
    for suf in ("_complex.npy", "_mag.npy"):
        p = root / f"recon/{m}_{c}{suf}"
        if p.exists():
            return np.abs(np.load(p)).astype(np.float32)
    return None


c, ax, i = args.contrast, args.axis, args.index
gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
gmax = float(gt.max())
head = gt > 0.05 * float(np.quantile(gt, 0.999))
sl = [slice(None)] * 3; sl[ax] = i; sl = tuple(sl)
gp, hp = gt[sl], head[sl]
yy, xx = np.nonzero(hp)
cy, cx = int(yy.mean()) + args.shift[0], int(xx.mean()) + args.shift[1]
h0, h1 = args.half
win = (slice(max(cy - h0, 0), cy + h0), slice(max(cx - h1, 0), cx + h1))
gcrop, hcrop = gp[win], hp[win]
g = gcrop[hcrop].astype(np.float64)

vols = {m: load(R, m, c) for m in ("zf", "cgsense", "cssense", "zsssdu", "dip", "inr", "gs_single", "gs_joint")}
vols["gs_joint@AF12"] = load(R12, "gs_joint", c)
vols = {m: v for m, v in vols.items() if v is not None}
psnr = {}
for m, v in vols.items():
    e = v[sl][win][hcrop].astype(np.float64) - g
    psnr[m] = 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
rank = {m: r + 1 for r, m in enumerate(sorted((k for k in psnr if "@" not in k), key=lambda k: -psnr[k]))}

blocks = [["GT", "zf", "cgsense", "cssense", "zsssdu"], ["dip", "inr", "gs_single", "gs_joint", "gs_joint@AF12"]]
vmax = float(np.quantile(gcrop, 0.999))
fig, axes = plt.subplots(4, 5, figsize=(4.0 * 5, 4.0 * 4 * (2 * h1) / (2 * h0) * 1.22))
for b, names in enumerate(blocks):
    for j, m in enumerate(names):
        a_img, a_err = axes[2 * b, j], axes[2 * b + 1, j]
        if m == "GT":
            a_img.imshow(gcrop.T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
            a_img.set_title(f"Ground truth ({c})\ncrop {2*h0}x{2*h1} around the head centroid", fontsize=11, fontweight="bold")
            # locator: the whole slice with the crop outlined
            a_err.imshow(gp.T, cmap="gray", vmin=0, vmax=float(np.quantile(gp, 0.999)), origin="lower")
            import matplotlib.patches as patches
            a_err.add_patch(patches.Rectangle((win[0].start, win[1].start), win[0].stop - win[0].start,
                                              win[1].stop - win[1].start, fill=False, edgecolor="red", linewidth=1.5))
            a_err.set_title(f"crop location on slice {i}; head voxels in crop: {int(hcrop.sum())}", fontsize=9)
            continue
        v = vols[m][sl][win]
        a_img.imshow(v.T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
        extra = f"  (rank {rank[m]}/8)" if m in rank else ""
        best = m in rank and rank[m] == 1
        a_img.set_title(f"{LABEL[m]}{'  <-- best' if best else ''}\ncrop head PSNR {psnr[m]:.2f} dB{extra}",
                        fontsize=10, fontweight="bold" if best else "normal")
        im = a_err.imshow((np.abs(v - gcrop) / gmax).T, cmap="jet", vmin=0.0, vmax=0.1, origin="lower")
        a_err.set_title("|recon - GT| / max(GT)", fontsize=9)
        cb = fig.colorbar(im, ax=a_err, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=6)
for a in axes.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"{c}  jiangyizhou  retrospective AF 24 -- butterfly region, axis{ax} slice {i} (transverse)   "
             f"error maps jet, fixed [0, 0.1]", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = F / f"t1flair_v3_af16g_{c}_butterflyzoom_axis{ax}_{i}.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); fig.savefig(R / f"figures/{out.name}", dpi=130, bbox_inches="tight")
print("wrote", out)
print("crop centre", (cy, cx), "window", win)
print(json.dumps({m: round(psnr[m], 3) for m in sorted(psnr, key=lambda k: -psnr[k])}, indent=None))
