#!/usr/bin/env python3
"""butterfly_probe.py -- where is the "butterfly" (the lateral ventricles) on the transverse slices of jiangyizhou?

The house butterfly slice of this subject is transverse 168 (fig_butterfly_final.py / fig_row_final.py, crop half-widths
52 x 38 around the head centroid). The three-pattern figure takes a SQUARE zoom box given by its centre (J, K) in volume
axes 1 and 2, so this draws the ground truth of a few transverse slices with (J, K) grid lines to read the box from,
and reports, per slice, the centroid of the head and of the dark (CSF) pixels in the middle of the head.

CPU only, read-only on the topic. Uses the GT for anatomy only; nothing here enters a reconstruction or a score.

  gpu.py run --cpu -- @python autoresearch/diffmask_af16_masks_20261111/butterfly_probe.py [--slices 160 164 168 172 176]
                                                                                          [--box W J K]
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.ndimage import binary_fill_holes, label  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--slices", type=int, nargs="+", default=[160, 164, 168, 172, 176])
ap.add_argument("--box", type=int, nargs=3, action="append", default=[], metavar=("W", "J", "K"), help="draw this box on every slice")
ap.add_argument("--tag", default="")
a = ap.parse_args()

T = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_uniform_v3_jiangyizhou")
ART = Path(os.environ.get("MCGS_ARTIFACT_DIR", "."))
ART.mkdir(parents=True, exist_ok=True)
gt = {c: np.abs(np.load(T / f"recon/gt_{c}_complex.npy")).astype(np.float32) for c in ("T1", "FLAIR")}
head = gt["T1"] > 0.05 * float(np.quantile(gt["T1"], 0.999))
rep = {}
fig, axes = plt.subplots(2, len(a.slices), figsize=(4.6 * len(a.slices), 2 * 4.6 * 256 / 176 * 0.72), squeeze=False)
for j, z in enumerate(a.slices):
    hp = binary_fill_holes(head[z])
    yy, xx = np.nonzero(hp)
    cj, ck = float(yy.mean()), float(xx.mean())
    # CSF of the ventricles: dark on T1 inside the filled head, in the middle third of the head's extent
    g = gt["T1"][z]
    jj, kk = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    mid = hp & (np.abs(jj - cj) < 0.22 * (yy.max() - yy.min())) & (np.abs(kk - ck) < 0.22 * (xx.max() - xx.min()))
    dark = mid & (g < 0.30 * float(np.quantile(g[hp], 0.5)))
    lab, n = label(dark)
    comps = sorted(((int((lab == i).sum()), i) for i in range(1, n + 1)), reverse=True)[:4]
    boxes = []
    for size, i in comps:
        if size < 25:
            continue
        y, x = np.nonzero(lab == i)
        boxes.append(dict(size=size, J=[int(y.min()), int(y.max())], K=[int(x.min()), int(x.max())],
                          centre=[round(float(y.mean()), 1), round(float(x.mean()), 1)]))
    allv = np.isin(lab, [i for s, i in comps if s >= 25])
    union = None
    if allv.any():
        y, x = np.nonzero(allv)
        union = dict(J=[int(y.min()), int(y.max())], K=[int(x.min()), int(x.max())],
                     centre=[round((y.min() + y.max()) / 2, 1), round((x.min() + x.max()) / 2, 1)], voxels=int(allv.sum()))
    rep[z] = dict(head_centroid=[round(cj, 1), round(ck, 1)], head_J=[int(yy.min()), int(yy.max())],
                  head_K=[int(xx.min()), int(xx.max())], dark_components=boxes, dark_union=union)
    for r, c in enumerate(("T1", "FLAIR")):
        ax = axes[r, j]
        s = gt[c][z]
        ax.imshow(s, cmap="gray", vmin=0, vmax=float(np.quantile(s, 0.999)), origin="upper", interpolation="nearest")
        ax.set_xticks(range(0, 176, 16))
        ax.set_yticks(range(0, 256, 16))
        ax.tick_params(labelsize=6)
        ax.grid(color="#FFC23D", lw=0.3, alpha=0.55)
        ax.set_title(f"{c}  transverse {z}   (rows = J, axis 1; columns = K, axis 2)", fontsize=8)
        ax.plot([ck], [cj], "+", color="cyan", ms=8)
        if r == 0 and union:
            ax.add_patch(Rectangle((union["K"][0] - 0.5, union["J"][0] - 0.5), union["K"][1] - union["K"][0] + 1,
                                   union["J"][1] - union["J"][0] + 1, fill=False, ec="red", lw=0.8, ls="--"))
        for w, bj, bk in a.box:
            ax.add_patch(Rectangle((bk - w // 2 - 0.5, bj - w // 2 - 0.5), w, w, fill=False, ec="lime", lw=1.0))
fig.tight_layout()
out = ART / f"butterfly_probe{a.tag}.png"
fig.savefig(out, dpi=110)
json.dump(rep, open(ART / f"butterfly_probe{a.tag}.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
print("wrote", out)
