"""preview_planes.py -- contact sheet of the tangbing references, to choose the three planes of the ablation figure.

CPU, read-only. Reads the reference volumes saved by tn_t1t2tb_fin_iso (gtT1 / gtFL) and writes one JPEG with
candidate transverse / coronal / sagittal slices of T1 and T2 to MCGS_ARTIFACT_DIR. No number is computed.
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_framework_ablation_20261110/trainer/results_arc")
z = np.load(RES / "tn_t1t2tb_fin_iso_volumes.npz")
print("keys", sorted(z.files), flush=True)
vols = {"T1": z["gtT1"], "T2": z["gtFL"]}
shape = vols["T1"].shape
head = vols["T1"] > 0.05 * np.quantile(vols["T1"], 0.999)
idx = np.argwhere(head)
lo, hi = idx.min(0), idx.max(0)
com = np.round(idx.mean(0)).astype(int)
print("shape", shape, "head box", lo.tolist(), hi.tolist(), "centre of mass", com.tolist(), flush=True)
cand = {a: [int(round(lo[a] + f * (hi[a] - lo[a]))) for f in (0.35, 0.42, 0.5, 0.58, 0.65)] for a in range(3)}
cand[0] += [150]; cand[1] += [113]; cand[2] += [77]
fig, axes = plt.subplots(6, 6, figsize=(24, 24))
for r, (name, axis) in enumerate((n, a) for a in range(3) for n in ("T1", "T2")):
    v = vols[name]
    for c, i in enumerate(cand[axis]):
        s = np.take(v, i, axis=axis)
        s = np.rot90(s.T, 1) if axis == 0 else s
        ax = axes[r, c]
        ax.imshow(s, cmap="gray", vmin=0, vmax=np.quantile(s, 0.999) / 1.5, origin="lower")
        ax.set_title(f"{name} axis {axis} index {i}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
art = Path(os.environ["MCGS_ARTIFACT_DIR"])
fig.savefig(art / "tangbing_planes_preview.jpg", dpi=70)
print("wrote", art / "tangbing_planes_preview.jpg", flush=True)
