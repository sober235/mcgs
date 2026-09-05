#!/usr/bin/env python3
"""Evaluate one arc/cycle.py score JSON against gates.json (spec 2026-09-05 §2), per contrast.

usage: eval_gates.py --gates gates.json --score <exp>_score.json [--out verdict.json]
                     [--append decisions.jsonl --phase 1 --note "..."]
The visual gate is not computed here; record it in --note.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

METRICS = ("head_psnr", "head_ssim", "head_hfen")


def evaluate(gates: dict, score: dict) -> dict:
    v = {"run": score["exp"], "gates_reference_run": gates["reference_run"], "contrasts": {}}
    for key, g in gates["contrasts"].items():
        s = score[key]
        r = {"name": g["name"]}
        for m in METRICS:
            val, ref, tol = float(s[m]), g[m]["ref"], g[m]["tol"]
            if g[m]["direction"] == "min":
                ok, margin = val <= ref + tol, (ref + tol) - val
            else:
                ok, margin = val >= ref - tol, val - (ref - tol)
            r[m] = {"value": val, "ref": ref, "tol": tol, "pass": bool(ok), "margin": float(margin)}
        e, eb = float(s["e_unobs"]), g["e_unobs_baseline"]
        r["e_unobs"] = {"value": e, "baseline": eb, "pass": bool(e <= eb), "margin": float(eb - e)}
        n, nmax = float(s["grain_ratio"]), gates["grain_ratio_max"]
        r["grain_ratio"] = {"value": n, "max": nmax, "pass": bool(n <= nmax), "margin": float(nmax - n)}
        r["n_pass"] = sum(int(r[k]["pass"]) for k in (*METRICS, "e_unobs", "grain_ratio"))
        r["all_pass"] = r["n_pass"] == 5
        v["contrasts"][key] = r
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", required=True)
    ap.add_argument("--score", required=True)
    ap.add_argument("--out")
    ap.add_argument("--append")
    ap.add_argument("--phase", default="")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    gates = json.loads(Path(a.gates).read_text())
    score = json.loads(Path(a.score).read_text())
    v = evaluate(gates, score)
    for key, r in v["contrasts"].items():
        cells = " ".join(f"{m}={'PASS' if r[m]['pass'] else 'FAIL'}({r[m]['value']:.4f},margin {r[m]['margin']:+.4f})"
                         for m in (*METRICS, "e_unobs", "grain_ratio"))
        print(f"{v['run']} {r['name']:6s} {r['n_pass']}/5 {cells}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(v, indent=2))
    if a.append:
        rec = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "run": v["run"],
               "phase": a.phase, "note": a.note,
               "verdict": {k: {"n_pass": r["n_pass"], "all_pass": r["all_pass"]} for k, r in v["contrasts"].items()},
               "score": str(Path(a.score))}
        with open(a.append, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
