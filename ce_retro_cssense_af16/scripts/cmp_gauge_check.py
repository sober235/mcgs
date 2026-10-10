#!/usr/bin/env python3
"""cmp_gauge_check.py -- re-score every volume that make_cmp_topic.py installed in a confirmation topic (each json row
carrying 'improved_from') through score_volume.py on the CPU in double precision, against the row that was installed
for it. One line per volume, 'GAUGE OK' or the mismatch. 2026-10-10 comparison-arm audit.
  RETRO_TOPIC=<cmp topic> python scripts/cmp_gauge_check.py > logs/gauge_check_cmp.txt"""
import json, os, subprocess, sys
from pathlib import Path

R = Path(os.environ["RETRO_TOPIC"]); S = Path(__file__).resolve().parent
bad = 0
for f in sorted(R.glob("logs/*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not isinstance(d, dict) or "improved" not in d:
        continue
    for r in d.get("rows", []):
        if "improved_from" not in r:
            continue
        m, scan = r["method"], r["contrast"]
        cmd = [sys.executable, str(S / "score_volume.py"), "--scan", scan, "--file", f"recon/{m}_{scan}_complex.npy",
               "--method", m, "--check", f"logs/{f.name}", "--gpu", "-1"]
        p = subprocess.run(cmd, cwd=R, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        last = (p.stdout.strip().splitlines() or ["(no output)"])[-1] if p.returncode == 0 else f"FAILED rc={p.returncode}: {p.stderr.strip()[-300:]}"
        bad += "GAUGE OK" not in last
        print(f"{scan:4s} {m:24s} <- {r['improved_from']:34s} {last}", flush=True)
print(f"{'ALL GAUGE OK' if bad == 0 else f'{bad} PROBLEM(S)'}")
