#!/usr/bin/env python3
"""watch_launch.py -- LOCAL helper: wait for empty amax cards and start the study's workers on them.

A card counts as empty when it has no compute process of anyone and under 300 MiB in use on two checks 75 s apart
(workspace GPU rule). Each empty card gets one worker through gpu.py run --gpu ID, alternating between the Gaussian
runner and the comparator runner, every worker with the full row list (claims prevent duplicates); the first
worker of each kind starts with that runner's jiangyizhou control row. Stops after --max workers. Prints only local
clock times; nothing of the server clock is read.

    /usr/bin/python3 autoresearch/diffmask_testset_20261122/watch_launch.py --max 4
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]
SSH = ["ssh", "-p", "2222", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "alias"]
SCANS = ["vol1_huangyuwei", "vol2_jiangdian", "vol4_tangbing", "vol5_sunxinyi", "vol6_jiangdian_790", "vol7_tangbin_790"]
PATTERNS = ["uniform", "radial"]
GS_ROWS = [f"{s}:{p}:{k}" for s in SCANS for p in PATTERNS for k in ("joint", "singleT1", "singleFLAIR")]
ARM_ROWS = [f"{s}:{p}:{a}" for s in SCANS for p in PATTERNS
            for a in ("classical", "cssense", "jcssense", "spirit", "zsssdu", "dip", "inr", "mcinr")]
now = lambda: time.strftime("%Y-%m-%d %H:%M %Z")


def empty_cards():
    q = ("nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits; echo ==; "
         "nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader")
    out = subprocess.run(SSH + [q], capture_output=True, text=True, timeout=120).stdout
    top, _, apps = out.partition("==")
    busy = {u.strip() for u in apps.split() if u.strip()}
    cards = set()
    for line in top.strip().splitlines():
        i, u, m = [x.strip() for x in line.split(",")]
        if u not in busy and int(m) < 300:
            cards.add(int(i))
    return cards


ap = argparse.ArgumentParser()
ap.add_argument("--max", type=int, default=4)
ap.add_argument("--poll", type=int, default=240)
ap.add_argument("--skip", type=int, nargs="*", default=[], help="cards never taken")
args = ap.parse_args()
started, mine = [], set()
while len(started) < args.max:
    try:
        a = empty_cards() - mine - set(args.skip)
        if a:
            time.sleep(75)
            a &= empty_cards()
        if not a:
            time.sleep(args.poll)
            continue
        card = min(a)
        kind = "gs" if len(started) % 2 == 0 else "arms"
        first = not any(k == kind for k, _ in started)
        if kind == "gs":
            rows = (["jyz:uniform:joint"] if first else []) + GS_ROWS
            script = "autoresearch/diffmask_testset_20261122/run_gs.py"
        else:
            rows = (["jyz:uniform:classical", "jyz:uniform:cssense"] if first else []) + ARM_ROWS
            script = "autoresearch/diffmask_testset_20261122/run_arms.py"
        r = subprocess.run(["/usr/bin/python3", "../gpu-tools/gpu.py", "run", "--gpu", str(card), "--", "@python", script,
                            "--rows", *rows], cwd=REPO, capture_output=True, text=True, timeout=900)
        try:
            jid = json.loads(r.stdout[r.stdout.index("{"):])["id"].rsplit("-", 1)[-1]
        except Exception:
            print(f"{now()} launch on card {card} gave no job id (rc {r.returncode}); retrying later", flush=True)
            time.sleep(args.poll)
            continue
        started.append((kind, card))
        mine.add(card)
        print(f"{now()} LAUNCH {kind} worker on card {card}, job ...{jid}, {len(rows)} rows", flush=True)
    except Exception as e:                                   # network hiccup: keep waiting
        print(f"{now()} check failed ({type(e).__name__}); retrying", flush=True)
        time.sleep(args.poll)
print(f"{now()} WATCH DONE: {started}", flush=True)
