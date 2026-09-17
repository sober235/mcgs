#!/usr/bin/env python3
"""check_scale_levels.py -- every per-contrast job file of every noise topic records the zero-noise topic's scale_p999,
and each summary.json has 24 rows (12 arms x 2 contrasts). Read-only; run after both finalizes (2026-09-17).
"""
import json, sys
from pathlib import Path
H = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou")
ref = {c: json.load(open(H / f"logs/classical_{c}.json"))["scale_p999"] for c in ("T1", "FLAIR")}
for N in sorted(Path("/data0/congcong/code/GS/Results_20260908").glob("T1FLAIR_AF16_noise*_v3_jiangyizhou")):
    L = N.name.split("_")[2]
    if not (N / "logs/summary.json").exists():
        print(f"{L}: no summary.json yet"); continue
    bad, n, worst = [], 0, 0.0
    for p in sorted(N.glob("logs/*.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict) or "scale_p999" not in d:
            continue
        c = d.get("contrast")
        if c in ref:
            n += 1
            rel = abs(d["scale_p999"] - ref[c]) / ref[c]
            worst = max(worst, rel)
            if rel > 1e-6:
                bad.append((p.name, d["scale_p999"]))
    s = json.load(open(N / "logs/summary.json"))
    arms = sorted({r["method"] for r in s})
    print(f"{L}: {n} per-contrast job files checked; worst rel {worst:.2e}; scale mismatches: {bad}")
    print(f"  summary rows: {len(s)}; arms ({len(arms)}): {arms}")
    print(f"  ref scale T1 {ref['T1']} FLAIR {ref['FLAIR']}")
