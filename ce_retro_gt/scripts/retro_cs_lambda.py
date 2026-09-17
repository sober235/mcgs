#!/usr/bin/env python3
"""retro_cs_lambda.py -- pick the CS-SENSE lambda for each simulated case on held-out lines (2026-09-17).

The first pass reused the prospective lambda (2e-3) and it was far too strong on this data: CS-SENSE washed out a
quarter to a half of the true uptake and scored below zero-filled. Truth is available here, but tuning one arm on
truth while the others get nothing would tilt the comparison, so lambda is chosen exactly as in the prospective
topic -- on 5% held-out phase-encode lines -- and only then refit on all of them.

  python scripts/retro_cs_lambda.py --case UCSF-PDGM-0004 --gpu 7 [--lams 2e-4,5e-4,1e-3,2e-3]
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

T = Path(__file__).resolve().parents[1]
CE = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/scripts")
PY = "/home/congcongliu/anaconda3/envs/GSTMI/bin/python"
ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--lams", default="5e-5,1e-4,2e-4,5e-4,1e-3,2e-3")
ap.add_argument("--outer", type=int, default=200)
a = ap.parse_args()
S = T / "sim" / a.case
R = T / "recon" / a.case / "tv"
env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID",
           CE_WORK_PRE=str(S / "work_pre"), CE_WORK_POST=str(S / "work_post"),
           CE_SPLIT=str(S / "split_95_5_seed0.npz"))
lams = [float(v) for v in a.lams.split(",")]


def run(scan, mask, ls):
    subprocess.run([PY, "-u", str(CE / "ce_tv_sense.py"), "--mode", "cs", "--scan", scan, "--mask", mask,
                    "--lams", ",".join(f"{l:g}" for l in ls), "--outer", str(a.outer), "--gpu", str(a.gpu),
                    "--out", str(R)], env=env, cwd=str(T), check=True)


sel = {}
for scan in ("pre", "post"):
    todo = [l for l in lams if not (R / f"cssense_o{a.outer}_lam{l:g}_fit_{scan}_meta.json").exists()]
    if todo:
        run(scan, "fit", todo)
    rows = {}
    for l in lams:
        m = json.load(open(R / f"cssense_o{a.outer}_lam{l:g}_fit_{scan}_meta.json"))
        rows[f"{l:g}"] = m["kres_heldout"]
    best = min(rows, key=rows.get)
    sel[scan] = dict(lam=float(best), heldout=rows[best], grid=rows,
                     at_edge="low" if float(best) == min(lams) else ("high" if float(best) == max(lams) else None))
    print(f"{a.case} {scan}: held-out " + "  ".join(f"{k}:{v:.5f}" for k, v in rows.items()) + f"  -> lambda {best}")
    if not (R / f"cssense_o{a.outer}_lam{float(best):g}_full_{scan}_meta.json").exists():
        run(scan, "full", [float(best)])
json.dump(sel, open(T / "recon" / a.case / "cs_lambda.json", "w"), indent=1)
print(f"wrote {T / 'recon' / a.case / 'cs_lambda.json'}")
