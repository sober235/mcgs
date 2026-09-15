#!/usr/bin/env python3
"""ce_split.py -- freeze ce_dip_inr.py's 95/5 held-out line split to a file and check the scorer.

The split is regenerated with the same code and seed and saved as heldout_split/split_95_5_seed0.npz, so the
arms added on 2026-09-14 (which run under two torch versions) all use one bit-identical split. Checks, on CPU:
  * fit + held-out partition the acquired plane, 7663 lines -> 7280 / 383 (the counts ce_dip_inr.py logged);
  * ce_common.kres_metrics reproduces kres_summary.json for CG-SENSE (full mask);
  * the held-out residual of the archived 3D-DIP / 3D-INR volumes reproduces their kres_heldout, which can only
    happen if the split is the one those runs used.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ce_common as C  # noqa: E402

torch.set_num_threads(16)
t0 = time.time()
summary = json.load(open(C.TOPIC / "kres_summary.json"))
report = {"torch": torch.__version__, "checks": []}
for scan in C.SCANS:
    y, S, m2 = C.load_scan(scan)
    fit, ho, nl, nho = C.make_split(m2)
    assert not bool((fit & ho).any()) and bool(((fit | ho) == m2).all())
    print(f"[{scan}] lines {nl} fit {int(fit.sum())} held-out {int(ho.sum())}  (load {time.time() - t0:.0f}s)", flush=True)
    if scan == "pre":
        C.SPLIT.parent.mkdir(parents=True, exist_ok=True)
        np.savez(C.SPLIT, fit=fit.numpy(), ho=ho.numpy(), lines=nl, heldout_lines=nho, seed=0, holdout=0.05,
                 torch=torch.__version__)
    else:
        prev = np.load(C.SPLIT)
        assert bool((torch.from_numpy(prev["fit"]) == fit).all()), "pre and post masks give different splits"
    for label, rel, ref_key, field in [("CG-SENSE", f"classical/cgsense_{scan}_complex.npy", f"{scan}|CG-SENSE", "full"),
                                       ("3D-DIP", f"dipinr/dip_{scan}_complex.npy", None, "ho"),
                                       ("3D-INR", f"dipinr/inr_{scan}_complex.npy", None, "ho")]:
        x = torch.from_numpy(np.load(C.TOPIC / rel))
        r = C.kres_metrics(x, y, S, {"full": m2, "fit": fit, "ho": ho})
        if ref_key:
            ref = summary[ref_key]
        else:
            ref = json.load(open(C.TOPIC / rel.replace("_complex.npy", "_meta.json")))["kres_heldout"]
        row = dict(scan=scan, method=label, field=field, recomputed=r[field]["kres"], archived=ref,
                   diff=r[field]["kres"] - ref, kres_full=r["full"]["kres"], kres_fit=r["fit"]["kres"], kres_ho=r["ho"]["kres"])
        report["checks"].append(row)
        print(f"  {label:9s} {field:4s} recomputed {row['recomputed']:.6f} archived {ref:.6f} diff {row['diff']:+.1e}"
              f"   (full {row['kres_full']:.5f} fit {row['kres_fit']:.5f} ho {row['kres_ho']:.5f})  {time.time() - t0:.0f}s", flush=True)
    del y, S
worst = max(abs(c["diff"]) for c in report["checks"])
report["max_abs_diff"] = worst
C.dump_json(report, C.SPLIT.parent / "split_check.json")
print(f"max |diff| {worst:.2e}  -> {'PASS' if worst < 5e-4 else 'FAIL'}", flush=True)
