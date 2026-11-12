#!/usr/bin/env python3
"""stage2_smoke.py -- the stage-two smoke of program.md section 10: four 30-step trainings in one job.

    gpu.py run --gpu 2 -- @python autoresearch/ablation_coupling_tune_20261111/stage2_smoke.py <tag>

In order: independent T1 and independent T2 with the symmetric phase TV, the full model with the same prior, both
cross-contrast guides and those two runs as comparators, and the no-prior row (where the symmetric prior must
stay at zero). Each is one tune.py --smoke call in the smoke tree; the first failure stops the job. <tag> makes
the experiment names unique (tn_t1t2_s2smoke<tag>_*).
"""
import re
import subprocess
import sys
from pathlib import Path

TUNE = str(Path(__file__).with_name("tune.py"))
tag = sys.argv[1] if len(sys.argv) > 1 else ""
assert re.fullmatch(r"[a-z0-9]*", tag), "tag: lower-case letters and digits"
SYM = "--sym=--lambda_ptv 0.03"
OV = "--ov=--set loss.edge_tv_guide_alpha=0.5 --set loss.ptv_guide_alpha=0.25"
n = f"s2smoke{tag}"
RUNS = [
    ["--name", f"{n}_t1", "--kind", "nojoint_p", SYM, "--hyp=stage-2 smoke: independent T1 with phase TV"],
    ["--name", f"{n}_t2", "--kind", "nojoint_c", SYM, "--hyp=stage-2 smoke: independent T2 with phase TV"],
    ["--name", f"{n}_full", SYM, OV, "--ref_p", f"tn_t1t2_{n}_t1", "--ref_c", f"tn_t1t2_{n}_t2",
     "--hyp=stage-2 smoke: full with phase TV, edge-TV guide 0.5 and phase-TV guide 0.25"],
    ["--name", f"{n}_noprior", "--kind", "noprior", SYM, OV,
     "--hyp=stage-2 smoke: no-prior row, the symmetric prior must stay at zero"],
]
for r in RUNS:
    rc = subprocess.run([sys.executable, "-u", TUNE, "--task", "t1t2", "--smoke", *r]).returncode
    if rc:
        sys.exit(rc)
