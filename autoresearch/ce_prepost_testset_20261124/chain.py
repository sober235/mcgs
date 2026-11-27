#!/usr/bin/env python3
"""chain.py -- one card, one job: for each subject, subject_topic.py (AF12 mask + simulation) then run_subject.py --all.

  gpu.py run --gpu ID -- @python autoresearch/ce_prepost_testset_20261124/chain.py --subjects A,B,C [--csm_crop 0.8]
A failing topic step stops the chain (nothing downstream is valid without it); a failing method stage is reported by
run_subject.py and the chain goes on to the next subject. No clock value is printed.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--subjects", required=True)
ap.add_argument("--csm_crop", default="0.8")
ap.add_argument("--topic_only", action="store_true")
a = ap.parse_args()
rcs = {}
for s in a.subjects.split(","):
    rc = subprocess.run([sys.executable, "-u", str(HERE / "subject_topic.py"), "--subject", s, "--csm_crop", a.csm_crop]).returncode
    print(f"CHAIN topic {s} rc={rc}", flush=True)
    if rc:
        sys.exit(rc)
    if not a.topic_only:
        rcs[s] = subprocess.run([sys.executable, "-u", str(HERE / "run_subject.py"), "--subject", s, "--all"]).returncode
        print(f"CHAIN methods {s} rc={rcs[s]}", flush=True)
print(f"CHAIN DONE {rcs}", flush=True)
sys.exit(1 if any(rcs.values()) else 0)
