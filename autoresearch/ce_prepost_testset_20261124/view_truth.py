#!/usr/bin/env python3
"""view_truth.py -- three central planes of a subject's truth (CS-SENSE, all acquired lines), plain scan and enhanced,
and their difference, to check by eye (CPU).   gpu.py run --cpu -- @python .../view_truth.py --subject S --csm_crop 0.8
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_prepost_testset_20261124")
ART = Path(os.environ.get("MCGS_ARTIFACT_DIR", "."))
ap = argparse.ArgumentParser()
ap.add_argument("--subject", required=True)
ap.add_argument("--csm_crop", type=float, default=None)
a = ap.parse_args()
T = OUT / "subjects" / a.subject / ("truth" if a.csm_crop is None else f"truth_crop{a.csm_crop:g}")
sel = json.load(open(T / "truth_selection.json"))
v = {s: np.abs(np.load(sel["scans"][s]["truth"])) for s in ("pre", "post")}
top = max(np.percentile(v[s], 99.9) for s in v)
rows = [("plain scan", v["pre"], "gray", 0, top), ("enhanced", v["post"], "gray", 0, top),
        ("enhanced - plain", v["post"] - v["pre"], "RdBu_r", -0.4 * top, 0.4 * top)]
nx, ny, nz = v["pre"].shape
fig, ax = plt.subplots(3, 3, figsize=(11, 11))
for i, (name, x, cm, lo, hi) in enumerate(rows):
    for j, (sl, t) in enumerate(((x[:, :, nz // 2], f"kz {nz // 2}"), (x[nx // 2], f"x {nx // 2}"), (x[:, ny // 2, :], f"ky {ny // 2}"))):
        ax[i, j].imshow(sl, cmap=cm, vmin=lo, vmax=hi)
        ax[i, j].set_title(f"{a.subject} {name}, lambda {sel['scans']['pre' if i == 0 else 'post']['lam']:g}: {t}" if i < 2 else f"{name}: {t}", fontsize=8)
        ax[i, j].axis("off")
fig.tight_layout()
fig.savefig(ART / f"truth_{a.subject}.png", dpi=110)
print("VIEW OK")
