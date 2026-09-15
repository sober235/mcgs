#!/usr/bin/env python3
"""fig_recon_only.py -- the reconstruction itself, large, without the comparison furniture.

One row per plane, reference on the left and the reconstruction on the right, each row drawn at that
plane's true physical proportions (the slab is 0.7 x 0.7 x 1.35 mm, so the coronal and sagittal rows
are genuinely shorter than the transverse one; stretching them to a common height would be a lie).
All planes vertex-up, grey window per plane from the reference's P99.9.

  ./scripts/py scripts/fig_recon_only.py --contrast T1w --key gs_joint_p3f
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parents[1]
FIGS = Path("/home/congcongliu/figs/peds_tsc_af12")
VOXEL = (0.7, 0.7, 1.35)                      # mm along (RO, PE, SPE)

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", required=True, choices=["T1w", "T2s"])
ap.add_argument("--key", default="gs_joint_p3f")
ap.add_argument("--label", default=None)
ap.add_argument("--slices", type=int, nargs=3, default=[53, 160, 152], metavar=("TRA", "COR", "SAG"))
ap.add_argument("--pw", type=float, default=5.6, help="panel width, inches")
ap.add_argument("--dpi", type=int, default=200)
args = ap.parse_args()
c, (tra, cor, sag) = args.contrast, args.slices
label = args.label or args.key

gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
p = R / f"recon/{args.key}_{c}_mag.npy"
rec = np.abs(np.load(p if p.exists() else R / f"recon/{args.key}_{c}_complex.npy")).astype(np.float32)
assert rec.shape == gt.shape, (rec.shape, gt.shape)

# axis 2 = transverse, axis 0 = coronal, axis 1 = sagittal (adjudicated against the vendor DICOM)
CUTS = [(2, tra, "transverse", (0, 1)), (0, cor, "coronal", (2, 1)), (1, sag, "sagittal", (2, 0))]

def plane(axis, idx):
    sl = [slice(None)] * 3; sl[axis] = idx
    return gt[tuple(sl)].T if axis == 2 else gt[tuple(sl)].T, rec[tuple(sl)].T

pw, gap, top_in, lab_in = args.pw, 0.05, 0.42, 0.62
rows = []
for axis, idx, name, (ra, ca) in CUTS:
    g, x = plane(axis, idx)
    h_mm, w_mm = g.shape[0] * VOXEL[ra], g.shape[1] * VOXEL[ca]
    rows.append(dict(g=g, x=x, name=name, idx=idx, h=pw * h_mm / w_mm,
                     vmax=float(np.quantile(g, 0.999))))
fw = lab_in + 2 * pw + gap
fh = top_in + sum(r["h"] for r in rows) + 2 * gap
fig = plt.figure(figsize=(fw, fh))

y = top_in
for r in rows:
    for j, (img, who) in enumerate(((r["g"], "reference"), (r["x"], label))):
        a = fig.add_axes([(lab_in + j * (pw + gap)) / fw, 1 - (y + r["h"]) / fh, pw / fw, r["h"] / fh])
        a.imshow(img, cmap="gray", vmin=0, vmax=r["vmax"], origin="upper", aspect="auto")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        if y == top_in:
            a.set_title(who, fontsize=12, fontweight="bold" if j else "normal", pad=4)
    fig.text(lab_in / fw * 0.42, 1 - (y + r["h"] / 2) / fh, f"{r['name']} {r['idx']}",
             rotation=90, ha="center", va="center", fontsize=11)
    y += r["h"] + gap
fig.suptitle(f"{c} — retrospective AF 12.00 — {label}", fontsize=13, fontweight="bold", y=1 - 0.09 / fh, va="top")
FIGS.mkdir(parents=True, exist_ok=True)
out = f"peds_af12_{c}_recon_{args.key}_{tra}_{cor}_{sag}.png"
for d in (FIGS, R / "figures"):
    fig.savefig(d / out, dpi=args.dpi)
print("wrote", FIGS / out)
