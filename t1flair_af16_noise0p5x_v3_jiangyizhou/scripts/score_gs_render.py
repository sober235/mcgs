#!/usr/bin/env python3
"""score_gs_render.py -- the render-only diagnostic of spec section 6 (noise study, 2026-09-16).

The GS arms are scored, like every topic of the series, on x_dc: the model k-space with the MEASURED samples put back.
With noisy measurements that step returns the noise on the acquired points to the image. The runs were saved with
--save_render, so each npz also holds the render before data consistency. This scores both volumes of each GS run
through this topic's Pair (clean ground truth), so the two numbers differ only by the data-consistency step.
Writes logs/gs_render_<c>.json (rows gs_joint_p3_dc / _render, gs_single_p3_dc / _render); these rows are a
diagnostic and never enter logs/summary.json.
usage: score_gs_render.py --gpu N --joint EXP --single_T1 EXP --single_FLAIR EXP
"""
import argparse
import json
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--joint", required=True)
ap.add_argument("--single_T1", required=True)
ap.add_argument("--single_FLAIR", required=True)
a = ap.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1t2_common import Pair, R  # noqa: E402

ARC = Path("/data0/congcong/code/GS/mcgs_noise_wt/results_arc")
# (key, contrast) -> (experiment, npz channel): cycle.py names the channels recT1 (plain) and recFL (contrast);
# a single-contrast run carries its contrast in both, and its plain channel is used
RUNS = {("gs_joint_p3", "T1"): (a.joint, "recT1"), ("gs_joint_p3", "FLAIR"): (a.joint, "recFL"),
        ("gs_single_p3", "T1"): (a.single_T1, "recT1"), ("gs_single_p3", "FLAIR"): (a.single_FLAIR, "recT1")}
dev = torch.device("cuda:0")
for c in ("T1", "FLAIR"):
    P = Pair(c, dev)
    rows = []
    for (key, cc), (exp, ch) in RUNS.items():
        if cc != c:
            continue
        with np.load(ARC / f"{exp}_volumes.npz") as z:
            vols = {"dc": np.asarray(z[f"{ch}_complex"]), "render": np.asarray(z[f"{ch}_render_complex"])}
        for kind, v in vols.items():
            x = torch.from_numpy(v).to(torch.complex64).to(dev)
            assert tuple(x.shape) == tuple(P.gt.shape), (exp, kind, tuple(x.shape))
            row = P.score(x, f"{key}_{kind}", exp=exp)
            rows.append(row)
            print(f"[{c}] {key:13s} {kind:6s} head PSNR {row['head_psnr']:.3f}  SSIM {row['head_ssim']:.4f}  "
                  f"HFEN {row['head_hfen']:.4f}  sharp {row['sharp']:.3f}", flush=True)
    json.dump({"contrast": c, "scale_p999": P.scale, "rows": rows,
               "note": "diagnostic: each GS volume with (dc) and without (render) hard data consistency, both "
                       "scored through this topic's Pair; not an arm"},
              open(R / f"logs/gs_render_{c}.json", "w"), indent=1)
    del P
    torch.cuda.empty_cache()
