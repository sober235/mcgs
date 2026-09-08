#!/usr/bin/env python3
"""Head-only metrics (spec 2026-09-05 §2.1) for a saved `*_volumes.npz` (recT1/gtT1/zfT1/recFL/gtFL/zfFL,
magnitudes). Used to compute the AF10 reference values and to re-score historical runs. The npz has no
phase, so e_unobs cannot be computed here; that comes from arc/cycle.py or the checkpoint replay.

usage: score_volumes_headonly.py <volumes.npz> --run <name> --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diag_artefact.headonly import score_headonly  # noqa: E402

KEYS = (("plain", "recT1", "gtT1", "zfT1"), ("contrast", "recFL", "gtFL", "zfFL"))


def score_npz(path: Path, run: str) -> dict:
    z = np.load(path)
    out = {"run": run, "npz": str(Path(path).resolve())}
    for key, rk, gk, zk in KEYS:
        rec, gt, zf = (z[k].astype(np.float32) for k in (rk, gk, zk))
        d = score_headonly(rec, gt)
        d["zf"] = score_headonly(zf, gt)
        out[key] = d
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = score_npz(Path(a.npz), a.run)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d, indent=2))
    for key, *_ in KEYS:
        s = d[key]
        print(f"{a.run} {key:8s} head_psnr={s['head_psnr']:.3f} head_ssim={s['head_ssim']:.4f} "
              f"head_hfen={s['head_hfen']:.4f} N={s['grain_ratio']:.3f} (zf head_psnr {s['zf']['head_psnr']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
