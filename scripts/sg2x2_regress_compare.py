#!/usr/bin/env python3
"""Compare two arc/cycle.py logs of the same 50-iteration joint smoke (before / after the sg2x2 edits).
Usage: sg2x2_regress_compare.py results_arc/sg2x2_regress_before.log results_arc/sg2x2_regress_after.log"""
from __future__ import annotations

import re
import sys

ITER = re.compile(r"^\[iter\s+(\d+)/(\d+)\]\s+(.*)$")
KV = re.compile(r"(\w+)=([0-9.eE+-]+)")


def parse(path: str) -> dict:
    out = {}
    for line in open(path):
        m = ITER.match(line.strip())
        if m:
            out[f"iter{int(m.group(1))}"] = {k: float(v) for k, v in KV.findall(m.group(3))}
        elif re.match(r"^(T1|FLAIR)\s+psnr=", line):
            out[line.split()[0]] = {k: float(v) for k, v in KV.findall(line)}
    return out


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def main() -> None:
    before, after = parse(sys.argv[1]), parse(sys.argv[2])
    ok = True
    for it, tol in (("iter1", 1e-4), ("iter50", 1e-2)):
        for k in ("dc_p", "dc_c", "R_geo", "R_wav"):
            a, b = before[it][k], after[it][k]
            good = rel(a, b) <= tol or (a == 0 and b == 0)
            ok &= good
            print(f"{it:6s} {k:6s} before={a:.6e} after={b:.6e} rel={rel(a, b):.2e} tol={tol:g} {'ok' if good else 'FAIL'}")
    for ch in ("T1", "FLAIR"):
        for k in ("head_psnr", "psnr"):
            a, b = before[ch][k], after[ch][k]
            good = abs(a - b) <= 0.02
            ok &= good
            print(f"{ch:6s} {k:9s} before={a:.3f} after={b:.3f} diff={b - a:+.4f} tol=0.02 {'ok' if good else 'FAIL'}")
    print("REGRESSION PASS" if ok else "REGRESSION FAIL")
    sys.exit(0 if ok else 4)


if __name__ == "__main__":
    main()
