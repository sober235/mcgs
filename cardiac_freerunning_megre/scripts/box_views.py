"""Helper for Task 12 (spec 5.4 criterion b, 2026-09-11): place one subject's liver-dome and heart boxes.
R0 echo 3 (the echo s07_polarity.py uses) on a 10-voxel grid in transverse, coronal and sagittal slices, with
the subject's boxes from polarity_boxes.json, or V07's boxes dashed while the subject has none yet.
Usage: box_views.py <Vxx>  ->  <Vxx>_box_views.png"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from common import TOPIC, recon_dir, save_fig  # noqa: E402

s = sys.argv[1]
src = "ref_e3.npy" if (recon_dir(s) / "ref_e3.npy").exists() else "r0_e3.npy"
x = np.abs(np.load(recon_dir(s) / src))
ori = json.loads((TOPIC / "orientation.json").read_text())[s]
SI, AP, LR = int(ori["si_axis"]), int(ori["ap_axis"]), int(ori["lr_axis"])
vmax = float(np.quantile(x[16:96, 16:96, 16:96], 0.995))
table = json.loads((TOPIC / "polarity_boxes.json").read_text())
own = s in table
boxes = {k: table[s if own else "V07"][k] for k in ("dome", "heart")}
COLOR = {"dome": "c", "heart": "y"}
NAME = {LR: f"axis{LR} (L-R)", AP: f"axis{AP} (P->A)", SI: f"axis{SI} ({'feet->head' if ori['superior_is_high_index'] else 'head->feet'})"}
head_up = not ori["superior_is_high_index"]        # origin='upper' puts the low index at the top
# fixed axis, its positions, horizontal axis, vertical axis, flip vertical so the head is up
PLANES = ((SI, (30, 45, 60, 75, 90), LR, AP, False), (AP, (45, 60, 75), LR, SI, head_up), (LR, (30, 45, 60, 75), AP, SI, head_up))
fig, ax = plt.subplots(3, 5, figsize=(21, 13))
for r, (fa, pos, h, v, up) in enumerate(PLANES):
    for c in range(5):
        a = ax[r, c]
        if c >= len(pos):
            a.axis("off")
            continue
        img = np.take(x, pos[c], axis=fa)            # rows = axis h, columns = axis v
        a.imshow(img.T, cmap="gray", vmin=0, vmax=vmax, origin="upper" if up else "lower")
        for name, b in boxes.items():
            inside = b[fa][0] <= pos[c] < b[fa][1]
            a.add_patch(Rectangle((b[h][0], b[v][0]), b[h][1] - b[h][0], b[v][1] - b[v][0], fill=False,
                                  ec=COLOR[name], lw=1.2, ls="-" if own else "--", alpha=1.0 if inside else 0.35))
        a.set_title(f"{NAME[fa].split()[0]} = {pos[c]}", fontsize=9)
        a.set_xlabel(NAME[h], fontsize=8)
        a.set_ylabel(NAME[v], fontsize=8)
        a.set_xticks(range(0, 112, 10))
        a.set_yticks(range(0, 112, 10))
        a.tick_params(labelsize=6)
        a.grid(color="r", lw=0.3, alpha=0.5)
fig.suptitle(f"{s} {src.split('_')[0].upper()} echo 3: dome (cyan) and heart (yellow) boxes"
             + ("" if own else " - V07's boxes, dashed: not placed for this subject yet"))
fig.tight_layout()
save_fig(fig, f"{s}_box_views.png")
