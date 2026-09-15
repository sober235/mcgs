#!/usr/bin/env python3
"""fig_recon_only.py -- the reconstruction itself, large, without the comparison furniture.

Top row: the reference (GT). Bottom row: one arm's reconstruction. Three planes, same grey window
per column, panels drawn at the true 0.7 x 0.7 x 1.35 mm proportions, all displayed vertex-up.

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

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", required=True, choices=["T1w", "T2s"])
ap.add_argument("--key", default="gs_joint_p3f", help="registered arm key, e.g. gs_joint_p3f")
ap.add_argument("--label", default=None)
ap.add_argument("--slices", type=int, nargs=3, default=[53, 160, 152], metavar=("TRA", "COR", "SAG"))
ap.add_argument("--dpi", type=int, default=200)
args = ap.parse_args()
c, tra, cor, sag = args.contrast, *args.slices
label = args.label or args.key

gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
p = R / f"recon/{args.key}_{c}_mag.npy"
if not p.exists():
    p = R / f"recon/{args.key}_{c}_complex.npy"
rec = np.abs(np.load(p)).astype(np.float32)
assert rec.shape == gt.shape, (rec.shape, gt.shape)

# axis 2 = transverse, axis 0 = coronal, axis 1 = sagittal (adjudicated against the vendor DICOM)
def planes(v):
    return [v[:, :, tra].T, v[cor, :, :].T, v[:, sag, :].T]

# physical aspect: in-plane 0.7 mm, through-plane 1.35 mm
ASPECT = [1.0, 0.7 / 1.35, 0.7 / 1.35]
NAMES = [f"transverse {tra}", f"coronal {cor}", f"sagittal {sag}"]
G, X = planes(gt), planes(rec)
win = [float(np.quantile(g, 0.999)) for g in G]

fig, ax = plt.subplots(2, 3, figsize=(13.5, 8.2))
for j in range(3):
    for i, (img, row) in enumerate(((G[j], "reference"), (X[j], label))):
        a = ax[i, j]
        a.imshow(img, cmap="gray", vmin=0, vmax=win[j], origin="lower", aspect=ASPECT[j])
        a.set_xticks([]); a.set_yticks([])
        if i == 0:
            a.set_title(NAMES[j], fontsize=11)
        if j == 0:
            a.set_ylabel(row, fontsize=11)
fig.suptitle(f"{c} — retrospective AF 12.00, {label}", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
FIGS.mkdir(parents=True, exist_ok=True)
out = f"peds_af12_{c}_recon_{args.key}_{tra}_{cor}_{sag}.png"
for d in (FIGS, R / "figures"):
    fig.savefig(d / out, dpi=args.dpi, bbox_inches="tight")
print("wrote", FIGS / out)
