#!/usr/bin/env python3
"""move_gs_worker.py -- LOCAL helper: move this study's Gaussian worker to another card at a row boundary.

User decision 2026-11-22 (relayed by session mcgs-32, agreed with mcgs-0f): the jobs of cards 5 and 6 are combined
on card 5, so that card 6 becomes free. The worker has no stop switch, so this helper
  1. waits until the server answers and until one more row has a score (= a row has just ended);
  2. stops the worker and its trainer (only processes of this study: run_gs.py of this folder, experiments dm_*);
  3. removes what the just-started next row left (its log, result folder, volumes file) and its claim -- only for
     experiments dm_* under this study's OUT that have NO score;
  4. starts the worker again with the formal row list on the target card (finished rows are skipped by their claims).
Only local clock times are printed.

    /usr/bin/python3 -u autoresearch/diffmask_testset_20261122/move_gs_worker.py --to 5
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = "/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_diffmask_testset_20261122"
SSH = ["ssh", "-p", "2222", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=20", "alias"]
SCANS = ["vol1_huangyuwei", "vol2_jiangdian", "vol4_tangbing", "vol5_sunxinyi", "vol6_jiangdian_790", "vol7_tangbin_790"]
GS_ROWS = [f"{s}:{p}:{k}" for s in SCANS for p in ("uniform", "radial") for k in ("joint", "singleT1", "singleFLAIR")]
now = lambda: time.strftime("%Y-%m-%d %H:%M:%S %Z")
say = lambda *a: print(now(), *a, flush=True)


def remote(cmd, timeout=120):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode == 255:
        raise ConnectionError(r.stderr.strip()[:120])
    return r.stdout


def state():
    """(number of scored rows, worker alive)"""
    out = remote(f"ls {OUT}/trainer/results_arc | grep -c '_score.json$'; "
                 "pgrep -f 'diffmask_testset_20261122/run_gs.py' | wc -l").split()
    return int(out[0]), int(out[1]) > 0


ap = argparse.ArgumentParser()
ap.add_argument("--to", type=int, required=True)
args = ap.parse_args()

n0 = None
while True:
    try:
        n, alive = state()
    except Exception as e:
        time.sleep(60)
        continue
    if n0 is None:
        n0 = n
        say(f"server reachable; {n} scored rows, worker alive: {alive}")
    if not alive or n > n0:
        break
    time.sleep(20)
say(f"row boundary ({n} scored rows, worker alive: {alive}); stopping the worker")

STOP = f"""
pkill -f 'diffmask_testset_20261122/run_gs.py'; sleep 1; pkill -f 'arc/cycle.py --exp dm_'; sleep 8
pkill -9 -f 'arc/cycle.py --exp dm_'; sleep 2
echo left=$(pgrep -f 'diffmask_testset_20261122/run_gs.py|arc/cycle.py --exp dm_' | wc -l)
cd {OUT} || exit 1
for c in claims/dm_*; do
  e=$(basename "$c")
  if [ ! -e "trainer/results_arc/${{e}}_score.json" ]; then
    rm -f "trainer/results_arc/${{e}}.log" "trainer/results_arc/${{e}}_volumes.npz"
    rm -rf "trainer/results_arc/${{e}}"
    rmdir "$c" && echo "cleared $e"
  fi
done
"""
out = remote(STOP, timeout=300)
say("stop + clean-up:", " | ".join(out.split("\n")).strip(" |"))
assert "left=0" in out, "a process of this study is still alive; not relaunching"

r = subprocess.run(["/usr/bin/python3", "../gpu-tools/gpu.py", "run", "--gpu", str(args.to), "--", "@python",
                    "autoresearch/diffmask_testset_20261122/run_gs.py", "--rows", *GS_ROWS],
                   cwd=REPO, capture_output=True, text=True, timeout=900)
try:
    jid = json.loads(r.stdout[r.stdout.index("{"):])["id"].rsplit("-", 1)[-1]
    say(f"RELAUNCHED on card {args.to}, job ...{jid}")
except Exception:
    say(f"RELAUNCH FAILED (rc {r.returncode}): {r.stdout[-300:]} {r.stderr[-300:]}")
    sys.exit(1)
time.sleep(90)
try:
    say("cards (index, util, MiB):", remote("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader").replace("\n", " | "))
except Exception:
    pass
say("MOVE DONE")
