#!/usr/bin/env python3
"""ce_score_heldout.py -- held-out residual of the 95%-line runs of CG-SENSE, ZS-SSDU, GS-single and MC-GS (2026-09-15).

Same scorer as every other held-out number in this topic (ce_common.kres_metrics: alpha refit on each mask).
GS volumes come in (256,176,232) and are mapped with transpose(0,2,1). Writes heldout4_scores.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ce_common as C  # noqa: E402

torch.set_num_threads(16)
T = C.TOPIC
VOLS = {  # label -> {scan: (path, is_gs_layout)}
    "CG-SENSE": {s: (T / f"classical_fit/cgsense_{s}_fit_complex.npy", False) for s in C.SCANS},
    "ZS-SSDU": {s: (T / f"classical_fit/zsssdu_{s}_fit_complex.npy", False) for s in C.SCANS},
    "GS-single": {"pre": (T / "gs_fit/single_pre/gs_single_pre_raw_complex.npy", True),
                  "post": (T / "gs_fit/single_post/gs_single_post_raw_complex.npy", True)},
    "MC-GS": {s: (T / f"gs_fit/joint/gs_joint_{s}_raw_complex.npy", True) for s in C.SCANS},
}
fit, ho = C.load_split()
out = {}
for s in C.SCANS:
    y, S, m2 = C.load_scan(s)
    for label, per in VOLS.items():
        p, gs = per[s]
        x = torch.from_numpy(np.load(p))
        if gs:
            x = x.permute(0, 2, 1).contiguous()
        r = C.kres_metrics(x, y, S, {"full": m2, "fit": fit, "ho": ho})
        out.setdefault(label, {})[s] = dict(path=str(p), kres_full=r["full"]["kres"], kres_fit=r["fit"]["kres"],
                                            kres_heldout=r["ho"]["kres"])
        print(f"{label:<10s} {s:<4s} full {r['full']['kres']:.5f} fit {r['fit']['kres']:.5f} held-out {r['ho']['kres']:.5f}", flush=True)
    del y, S
C.dump_json(out, T / "heldout4_scores.json")
print("DONE", flush=True)
