#!/usr/bin/env python3
"""tune_knee.py -- one round of the knee MC-GS tuning loop (autoresearch, one card, candidates run one after another).

A round is a JSON file rounds/<round>.json: {"keeper": [trainer args appended to the brain recipe],
"candidates": {"<name>": [the one change of this candidate, as trainer args]}}. Each candidate trains the JOINT
model once through run_knee.py (seed 42, no volumes) with keeper + its change, experiment tag <round>_<name>.
Nothing is overwritten (run_knee.py claims); results go to MCGS_ARTIFACT_DIR/<round>.jsonl. No clock values.

  gpu.py run --gpu N -- @python autoresearch/uih_knee_tune_20261126/tune_knee.py --subject 6 --round r1
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "uih_knee_t1t2_20261125/run_knee.py"
OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_uih_knee_t1t2_20261125")
ap = argparse.ArgumentParser()
ap.add_argument("--subject", type=int, required=True)
ap.add_argument("--round", required=True)
ap.add_argument("--row", default="full", choices=["full", "single_T1", "single_T2"])
a = ap.parse_args()
R = json.load(open(HERE / "rounds" / f"{a.round}.json"))
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", "."))
rows = []
for name, change in R["candidates"].items():
    tag = f"{a.round}_{name}"
    extra = [*R["keeper"], *change]
    cmd = [sys.executable, "-u", str(RUN), "--subject", str(a.subject), "--rows", a.row, "--tag", tag, "--no_volumes",
           "--extra", *extra]
    rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    p = OUT / f"trainer/results_arc/knee_s{a.subject:02d}_{tag}_{a.row}_score.json"
    row = dict(round=a.round, name=name, row=a.row, change=change, keeper=R["keeper"], rc=rc)
    if p.exists():
        s = json.load(open(p))
        for k, blk in (("T1", "plain"), ("T2", "contrast")):
            if s.get(blk):
                key = k if a.row == "full" else a.row.split("_")[1]
                row[key] = {m: s[blk][m] for m in ("head_psnr", "head_ssim", "head_hfen", "nmse")}
        row.update(npts=s.get("npts"), vram_gb=s.get("vram_gb"))
    rows.append(row)
    print(json.dumps(row), flush=True)
    (art / f"{a.round}_{a.row}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print("ROUND_DONE", a.round, flush=True)
