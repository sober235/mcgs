#!/usr/bin/env python3
"""hp_summarize.py -- one table for the hyper-parameter sensitivity runs
(spec docs/superpowers/specs/2026-09-16-hparam-sensitivity-design.md).

Input : results_arc/<exp>_score.json (arc/cycle.py) for the six runs; <exp>_volumes.npz when present (head NMSE on the
        scorer's head mask |GT| > 0.05 P99.9|GT|, diag_artefact.headonly.head_mask -- the json has no head_nmse).
Output: results_arc/hp_summary.md and hp_summary.csv; the anchor check of hp_n36 against the registered gs_joint_p3.
usage: hp_summarize.py [--arc results_arc] [--anchor hp_n36] [--anchor_ref 35.706 34.928] [--tol 0.02]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diag_artefact.headonly import head_mask  # noqa: E402

# exp, shared count, private count (spec section 4)
RUNS = [
    ("hp_n09", 900_000, 0),
    ("hp_n18", 1_800_000, 0),
    ("hp_n36", 3_600_000, 0),
    ("hp_n72", 7_200_000, 0),
    ("hp_p05", 3_600_000, 180_000),
    ("hp_p10", 3_600_000, 360_000),
]
CONTRASTS = (("T1", "plain", "recT1", "gtT1"), ("FLAIR", "contrast", "recFL", "gtFL"))
JSON_KEYS = ("head_psnr", "head_ssim", "head_hfen", "psnr", "ssim", "nmse", "head_psnr_cplx")


def head_nmse(rec: np.ndarray, gt: np.ndarray) -> float:
    rec = np.asarray(rec, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    head = head_mask(gt.astype(np.float32))
    den = float((gt[head] ** 2).sum())
    if den == 0.0:
        return float("inf")
    return float(((rec - gt)[head] ** 2).sum() / den)


def load_run(arc: Path, exp: str) -> dict | None:
    p = arc / f"{exp}_score.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    out = {"npts": int(d.get("npts", 0)), "elapsed_h": float(d.get("elapsed_s", float("nan"))) / 3600.0,
           "vram_gb": float(d.get("vram_gb", float("nan")))}
    vol = arc / f"{exp}_volumes.npz"
    z = np.load(vol) if vol.exists() else None
    for name, key, rk, gk in CONTRASTS:
        c = d.get(key, {})
        for k in JSON_KEYS:
            out[f"{name}_{k}"] = float(c[k]) if k in c else float("nan")
        out[f"{name}_head_nmse"] = head_nmse(z[rk], z[gk]) if z is not None and rk in z.files else float("nan")
    return out


def build_rows(arc: Path, runs=RUNS) -> list[dict]:
    rows = []
    for exp, n_shared, n_private in runs:
        row = {"exp": exp, "n_shared": n_shared, "n_private": n_private, "ratio": n_private / n_shared}
        got = load_run(Path(arc), exp)
        row["status"] = "done" if got else "missing"
        if got is None:
            got = {"npts": 0, "elapsed_h": float("nan"), "vram_gb": float("nan")}
            for name, *_ in CONTRASTS:
                for k in JSON_KEYS + ("head_nmse",):
                    got[f"{name}_{k}"] = float("nan")
        row.update(got)
        rows.append(row)
    return rows


def anchor_check(rows: list[dict], anchor: str, ref: tuple[float, float], tol: float) -> dict:
    row = next((r for r in rows if r["exp"] == anchor), None)
    if row is None or row["status"] != "done":
        return {"anchor": anchor, "ref": ref, "tol": tol, "delta": (float("nan"), float("nan")), "passed": None}
    delta = (row["T1_head_psnr"] - ref[0], row["FLAIR_head_psnr"] - ref[1])
    return {"anchor": anchor, "ref": ref, "tol": tol, "delta": delta,
            "passed": bool(all(abs(x) <= tol for x in delta))}


def _f(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render_md(rows: list[dict], check: dict) -> str:
    lines = ["| exp | N_s | N_p | N_p/N_s | T1 head PSNR / SSIM / HFEN / NMSE | FLAIR head PSNR / SSIM / HFEN / NMSE | "
             "T1 whole PSNR / NMSE | FLAIR whole PSNR / NMSE | cplx T1 / FLAIR | h | GB |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] != "done":
            lines.append(f"| {r['exp']} | {r['n_shared']/1e6:.1f}M | {r['n_private']/1e3:.0f}k | {r['ratio']:.0%} | (missing) | | | | | | |")
            continue

        def cell(c):
            return (f"{_f(r[c + '_head_psnr'])} / {_f(r[c + '_head_ssim'], 4)} / {_f(r[c + '_head_hfen'], 4)} / "
                    f"{_f(r[c + '_head_nmse'], 5)}")

        def whole(c):
            return f"{_f(r[c + '_psnr'])} / {_f(r[c + '_nmse'], 5)}"

        lines.append(f"| {r['exp']} | {r['n_shared']/1e6:.1f}M | {r['n_private']/1e3:.0f}k | {r['ratio']:.0%} | {cell('T1')} | "
                     f"{cell('FLAIR')} | {whole('T1')} | {whole('FLAIR')} | {_f(r['T1_head_psnr_cplx'])} / "
                     f"{_f(r['FLAIR_head_psnr_cplx'])} | {_f(r['elapsed_h'], 1)} | {_f(r['vram_gb'], 1)} |")
    base = next((r for r in rows if r["exp"] == check["anchor"] and r["status"] == "done"), None)
    if base is not None:
        lines.append("")
        lines.append(f"Deltas against {check['anchor']} (head PSNR T1 / FLAIR, dB):")
        for r in rows:
            if r["status"] == "done" and r["exp"] != check["anchor"]:
                lines.append(f"- {r['exp']}: {r['T1_head_psnr'] - base['T1_head_psnr']:+.3f} / "
                             f"{r['FLAIR_head_psnr'] - base['FLAIR_head_psnr']:+.3f}")
    verdict = {True: "PASS", False: "FAIL", None: "not run yet"}[check["passed"]]
    lines.append("")
    lines.append(f"Anchor {check['anchor']} vs registered gs_joint_p3 {check['ref'][0]:.3f} / {check['ref'][1]:.3f}: "
                 f"delta {_f(check['delta'][0])} / {_f(check['delta'][1])} dB, tolerance {check['tol']:.2f} -> {verdict}")
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict], path: Path) -> None:
    cols = ["exp", "n_shared", "n_private", "ratio", "status", "npts", "elapsed_h", "vram_gb"]
    for name, *_ in CONTRASTS:
        cols += [f"{name}_{k}" for k in JSON_KEYS + ("head_nmse",)]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc", default=str(ROOT / "results_arc"))
    ap.add_argument("--anchor", default="hp_n36")
    ap.add_argument("--anchor_ref", nargs=2, type=float, default=[35.706, 34.928])
    ap.add_argument("--tol", type=float, default=0.02)
    a = ap.parse_args(argv)
    arc = Path(a.arc)
    rows = build_rows(arc)
    check = anchor_check(rows, a.anchor, tuple(a.anchor_ref), a.tol)
    md = render_md(rows, check)
    (arc / "hp_summary.md").write_text(md)
    write_csv(rows, arc / "hp_summary.csv")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
