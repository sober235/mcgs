#!/usr/bin/env python3
"""Build the pre-registered gates.json (spec 2026-09-05 §2) from three head-only score JSONs
(scripts/score_volumes_headonly.py) and the baseline's e_unobs replay JSON
(scripts/autoresearch_kspace_score.py).

    tol = max(2 * |ref - repeat|, floor)     floors: 0.005 dB / 0.0005 SSIM / 0.001 HFEN

usage: make_gates.py --ref REF.json --repeat REP.json --baseline BASE.json --baseline-kspace BASE_kspace.json
                     --out gates.json [--names T1 FLAIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FLOORS = {"head_psnr": 0.005, "head_ssim": 0.0005, "head_hfen": 0.001}
DIRECTION = {"head_psnr": "max", "head_ssim": "max", "head_hfen": "min"}
SPEC = "docs/superpowers/specs/2026-09-05-af12-to-af10-quality-design.md §2"


def build_gates(ref: dict, rep: dict, base: dict, base_kspace: dict,
                names=("T1", "FLAIR"), floors: dict = FLOORS) -> dict:
    gates = {
        "spec": SPEC,
        "reference_run": ref["run"], "repeat_run": rep["run"], "baseline_run": base["run"],
        "tolerance_rule": "tol = max(2*|ref-repeat|, floor); pass iff value >= ref-tol (max) or value <= ref+tol (min)",
        "tolerance_floor": dict(floors),
        "e_unobs_rule": "pass iff e_unobs <= baseline e_unobs (checkpoint replay of the AF12 baseline)",
        "grain_ratio_max": 1.0,
        "visual_gate": "butterfly z=160 + three views, judged by the user, recorded in decisions.jsonl",
        "contrasts": {},
    }
    for key, name in zip(("plain", "contrast"), names):
        c = {"name": name}
        for m in ("head_psnr", "head_ssim", "head_hfen"):
            delta = abs(float(ref[key][m]) - float(rep[key][m]))
            c[m] = {"ref": float(ref[key][m]), "repeat": float(rep[key][m]),
                    "tol": max(2.0 * delta, float(floors[m])), "direction": DIRECTION[m]}
        c["e_unobs_baseline"] = float(base_kspace["modalities"][key]["e_unobs"])
        c["baseline_head"] = {m: float(base[key][m]) for m in ("head_psnr", "head_ssim", "head_hfen", "grain_ratio")}
        gates["contrasts"][key] = c
    return gates


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--repeat", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--baseline-kspace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", nargs=2, default=("T1", "FLAIR"))
    a = ap.parse_args(argv)
    load = lambda p: json.loads(Path(p).read_text())
    g = build_gates(load(a.ref), load(a.repeat), load(a.baseline), load(a.baseline_kspace), tuple(a.names))
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f"{out} exists; gates are pre-registered and must not be rewritten in place")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(g, indent=2))
    print(json.dumps(g, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
