#!/usr/bin/env python3
"""fig_planes_v2.py -- the three anatomical planes in one figure, each at its best-quality slice.

For each axis the slice is the one where MC-GS joint v2 has the highest head-masked PSNR among
slices with >= 20000 head voxels (logs/best_slice_v2.json, written by find_best_slice_v2.py), unless
--slices a,b,c is given. Layout: per plane two rows (magnitudes, then |recon-GT|/max(GT) in jet on a
fixed [0, 0.1] range), all panels the same width, minimal gaps, the AF12 sampling pattern in the
bottom-left cell of the first plane, one shared error colorbar on the right. Titles carry each
slice's own head-masked PSNR and rank. Grey window = in-head P95 of that GT slice.

usage: fig_planes_v2.py --topic <dir> --contrast T1 --prefix t1t2_af12 [--slices 150,113,77] [--ours gs_joint_v2]
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--topic", required=True)
ap.add_argument("--contrast", required=True)
ap.add_argument("--prefix", required=True, help="figure name prefix and figs sub-directory, e.g. t1t2_af12")
ap.add_argument("--slices", default=None, help="axis0,axis1,axis2 indices; default = per-plane best from logs/best_slice_v2.json")
ap.add_argument("--ours", default="gs_joint_v2")
ap.add_argument("--tag", default="best3planes")
args = ap.parse_args()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

R = Path(args.topic)
F = Path("/home/congcongliu/figs") / args.prefix; F.mkdir(parents=True, exist_ok=True)
ANAT = {0: "transverse", 1: "coronal", 2: "sagittal"}
ORDER = ["zf", "cgsense", "cssense", "zsssdu", "dip", "inr", "gs_single", "gs_joint", "gs_single_v2", "gs_joint_v2"]
LABEL = {"zf": "Zero-filled", "cgsense": "CG-SENSE", "cssense": "CS-SENSE (TV)", "zsssdu": "ZS-SSDU",
         "dip": "3D-DIP", "inr": "3D-INR", "gs_single": "MC-GS single", "gs_joint": "MC-GS joint",
         "gs_single_v2": "MC-GS single v2", "gs_joint_v2": "MC-GS joint v2"}


def load(m, c):
    for suf in ("_complex.npy", "_mag.npy"):
        p = R / f"recon/{m}_{c}{suf}"
        if p.exists():
            return np.abs(np.load(p)).astype(np.float32)
    return None


c = args.contrast
gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
gmax = float(gt.max())
head = gt > 0.05 * float(np.quantile(gt, 0.999))
vols = {m: load(m, c) for m in ORDER}
methods = [m for m in ORDER if vols[m] is not None]
if args.slices:
    idx = [int(x) for x in args.slices.split(",")]
else:
    best = json.load(open(R / "logs/best_slice_v2.json"))[c]
    idx = [max((r for r in best if r["axis"] == ax), key=lambda r: r["ours"])["index"] for ax in range(3)]
m2 = scipy.io.loadmat(R / "data/mask_vdpois_256x176_af12.mat")["mask"].astype(float)

ncol = len(methods) + 1
W = 2.6
planes = []
for ax in range(3):
    sl = [slice(None)] * 3; sl[ax] = idx[ax]; sl = tuple(sl)
    gp, hp = gt[sl], head[sl]
    g = gp[hp].astype(np.float64)
    ps = {}
    for m in methods:
        e = vols[m][sl][hp].astype(np.float64) - g
        ps[m] = 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
    rank = {m: r + 1 for r, m in enumerate(sorted(ps, key=lambda k: -ps[k]))}
    planes.append(dict(ax=ax, sl=sl, gp=gp, hp=hp, ps=ps, rank=rank, hw=gp.shape[1] / gp.shape[0]))
heights = []
for p in planes:
    heights += [p["hw"], p["hw"]]
fig_h = W * sum(heights) + 1.6
fig_w = W * ncol + 0.45
fig = plt.figure(figsize=(fig_w, fig_h))
left, right, bottom, top = 0.012, 1 - 0.45 / fig_w, 0.02, 1 - 0.9 / fig_h
# outer grid = one row per plane (room for the next plane's titles), inner grid = image row + error row
outer = fig.add_gridspec(3, 1, height_ratios=[p["hw"] for p in planes], left=left, right=right,
                         bottom=bottom, top=top, hspace=0.16)
inner = [outer[k].subgridspec(2, ncol, wspace=0.015, hspace=0.015) for k in range(3)]
im = None
for k, p in enumerate(planes):
    gs = inner[k]; r0, r1 = 0, 1
    gp, hp = p["gp"], p["hp"]
    vmax = float(np.quantile(gp[hp], 0.95)) if hp.any() else float(np.quantile(gp, 0.999))
    a = fig.add_subplot(gs[r0, 0]); a.imshow(gp.T, cmap="gray", vmin=0, vmax=vmax, origin="lower", aspect="auto")
    a.set_title(f"Ground truth ({c})", fontsize=9, fontweight="bold", pad=2)
    a.set_ylabel(f"axis{p['ax']} slice {idx[p['ax']]}\n{ANAT[p['ax']]}", fontsize=9, fontweight="bold")
    a.set_xticks([]); a.set_yticks([])
    b = fig.add_subplot(gs[r1, 0]); b.set_xticks([]); b.set_yticks([])
    if k == 0:
        b.imshow(m2.T, cmap="gray", origin="lower", interpolation="nearest", aspect="auto")
        b.set_xlabel(f"AF12 vd-Poisson mask (ky,kz), {100 * m2.mean():.2f} % of the PE plane", fontsize=7)
    else:
        b.axis("off")
    for j, m in enumerate(methods, start=1):
        v = vols[m][p["sl"]]
        a = fig.add_subplot(gs[r0, j]); a.imshow(v.T, cmap="gray", vmin=0, vmax=vmax, origin="lower", aspect="auto")
        mark = "  (best)" if p["rank"][m] == 1 else ""
        a.set_title(f"{LABEL[m]}{mark}\n{p['ps'][m]:.2f} dB  rank {p['rank'][m]}", fontsize=8.5,
                    fontweight="bold" if p["rank"][m] == 1 else "normal", pad=2)
        a.set_xticks([]); a.set_yticks([])
        b = fig.add_subplot(gs[r1, j])
        im = b.imshow((np.abs(v - gp) / gmax).T, cmap="jet", vmin=0.0, vmax=0.1, origin="lower", aspect="auto")
        b.set_xticks([]); b.set_yticks([])
        if k == 2 and j == len(methods):
            pos = b.get_position()
            cax = fig.add_axes([pos.x1 + 0.004, pos.y0, 0.010, pos.height])
            cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=7); cb.set_label("|recon - GT| / max(GT)", fontsize=7)
fig.suptitle(f"{c}  retrospective AF 12  --  three planes, each at the slice where {LABEL[args.ours]} is best   "
             f"(head-masked PSNR of each slice; error maps jet, fixed [0, 0.1])", fontsize=11, fontweight="bold", y=0.995)
out = F / f"{args.prefix}_{c}_{args.tag}.png"
fig.savefig(out, dpi=110, bbox_inches="tight"); fig.savefig(R / f"figures/{out.name}", dpi=110, bbox_inches="tight")
print("wrote", out, "slices", idx)
for p in planes:
    print(f"  axis{p['ax']} idx {idx[p['ax']]}: " + ", ".join(f"{m} {p['ps'][m]:.2f} (rank {p['rank'][m]})" for m in methods if p['rank'][m] <= 3))
