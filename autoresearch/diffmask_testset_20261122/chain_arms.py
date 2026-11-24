#!/usr/bin/env python3
"""chain_arms.py -- server job: run the comparator rows of this study, gated by the jiangyizhou control check.

Two copies may run at once (2026-11-23: card 3 ceded by session plain-enhan under the 4-card cap, plus this study's
share of card 5 once its Gaussian worker has ended):
  1. unless --no_wait, wait (no GPU use) while a run_gs.py process of this study is alive;
  2. if OUT/control_check.json exists, use its verdict; otherwise run the two jiangyizhou control rows
     (jyz:uniform:classical, jyz:uniform:cssense) and check every result against the registered value of the
     uniform-density topic: |head PSNR difference| <= --tol_db, then write the verdict to OUT/control_check.json.
     If the control rows were claimed by the other copy, wait for its verdict instead of rerunning them;
  3. only if the controls match, run the 96 formal comparator rows (claims skip rows taken or finished by the
     other copy). Exit 2 = control check failed or no verdict.
No server clock is printed or recorded.

    gpu.py run --gpu 3 -- @python autoresearch/diffmask_testset_20261122/chain_arms.py --no_wait
    gpu.py run --gpu 5 -- @python autoresearch/diffmask_testset_20261122/chain_arms.py
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_arms import CONTROL_ROWS, FORMAL_ROWS   # noqa: E402
from ts_common import OUT                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tol_db", type=float, default=0.005)
ap.add_argument("--poll", type=int, default=120)
ap.add_argument("--no_wait", action="store_true", help="do not wait for the Gaussian worker")
ap.add_argument("--verdict_timeout_h", type=float, default=48.0)
args = ap.parse_args()
log = lambda *a: print(*a, flush=True)
MARK = OUT / "control_check.json"
CONTROLS = ["jyz:uniform:classical", "jyz:uniform:cssense"]
assert all(c in CONTROL_ROWS for c in CONTROLS)


def gs_alive():
    out = subprocess.run(["pgrep", "-f", "python.*diffmask_testset_20261122/run_gs.py"],
                         capture_output=True, text=True).stdout
    return [p for p in out.split() if int(p) != os.getpid()]


def read_mark():
    return json.loads(MARK.read_text()) if MARK.exists() else None


def run_rows(rows):
    return subprocess.run([sys.executable, "-u", str(HERE / "run_arms.py"), "--rows", *rows]).returncode


if not args.no_wait:
    n = 0
    while gs_alive():
        if n % 30 == 0:
            log("waiting: the Gaussian worker of this study is still running")
        n += 1
        time.sleep(args.poll)
    log("Gaussian worker has ended")
log("starting the comparator rows")

verdict = read_mark()
if verdict is None:
    rc = run_rows(CONTROLS)
    art = Path(os.environ["MCGS_ARTIFACT_DIR"])
    recf = art / f"record_arms_gpu{os.environ['CUDA_VISIBLE_DEVICES']}_{CONTROLS[0].replace(':', '_')}.json"
    rec = json.loads(recf.read_text()) if recf.exists() else {"runs": []}
    diffs = [(r["contrast"], r["method"], r["diff"]["head_psnr"])
             for run in rec["runs"] for r in run["results"] if "diff" in r]
    for d in diffs:
        log(f"control {d[0]:5s} {d[1]:24s} head PSNR diff {d[2]:+.6f} dB")
    ran = {run["row"] for run in rec["runs"]}
    if rc == 0 and ran != set(CONTROLS):
        log(f"control rows {sorted(set(CONTROLS) - ran)} were claimed by the other copy; waiting for its verdict")
        t0 = time.time()
        while read_mark() is None and time.time() - t0 < args.verdict_timeout_h * 3600:
            time.sleep(args.poll)
        verdict = read_mark()
        if verdict is None:
            log("CONTROL CHECK FAILED (no verdict from the other copy); formal rows NOT started")
            sys.exit(2)
    else:
        ok = rc == 0 and len(diffs) >= 6 and all(abs(d[2]) <= args.tol_db for d in diffs)
        verdict = dict(ok=ok, rc=rc, tol_db=args.tol_db, gpu=os.environ["CUDA_VISIBLE_DEVICES"],
                       diffs=[dict(contrast=c, method=m, head_psnr_diff=v) for c, m, v in diffs])
        tmp = MARK.with_suffix(".tmp")
        tmp.write_text(json.dumps(verdict, indent=2))
        tmp.replace(MARK)
else:
    log(f"using the verdict in {MARK.name}")

if not verdict["ok"]:
    log(f"CONTROL CHECK FAILED (rc {verdict['rc']}, {len(verdict['diffs'])} compared results); formal rows NOT started")
    sys.exit(2)
log(f"CONTROL CHECK OK ({len(verdict['diffs'])} results within {verdict['tol_db']} dB); formal comparator rows follow")
rc = run_rows(FORMAL_ROWS)
log(f"CHAIN DONE rc {rc}")
sys.exit(rc)
