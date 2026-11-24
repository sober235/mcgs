#!/usr/bin/env python3
"""check_all.py -- CPU checks of the different-mask test-set study, in order: frozen settings, topic clones, the two
runner self-tests. Each step's verdict is printed; a failing step does not stop the later ones.

    gpu.py run --cpu -- @python autoresearch/diffmask_testset_20261122/check_all.py
"""
import subprocess
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
bad = []
for step in (["ts_common.py", "--selftest"], ["make_topics.py"], ["run_gs.py", "--selftest"], ["run_arms.py", "--selftest"]):
    r = subprocess.run([sys.executable, "-u", str(H / step[0]), *step[1:]], capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-12:]
    print(f"===== {' '.join(step)}: rc {r.returncode}", flush=True)
    for line in tail:
        print("   ", line[:400], flush=True)
    if r.returncode:
        bad.append(step[0])
print("CHECKS", "FAILED: " + ", ".join(bad) if bad else "OK", flush=True)
sys.exit(1 if bad else 0)
