#!/usr/bin/env python3
"""ca_summarize.py -- Stage 1 summariser for the coupling-attribution study
(spec docs/superpowers/specs/2026-09-20-coupling-attribution-design.md).

Splits the published MC-GS minus GS-single head-PSNR margin into a shared-geometry part (M0, joint
minus single at lambda_edge 0) and a common-edge-prior part (delta = M1 - M0, M1 the margin at the
published lambda_edge 0.001).

Input : results_arc/<exp>_score.json (arc/cycle.py) for the six Stage-1 experiments;
        <exp>_volumes.npz for head NMSE (the json has no head NMSE), on the scorer's head mask
        |GT| > 0.05 * P99.9|GT| (diag_artefact.headonly.head_mask) -- same method as scripts/hp_summarize.py.
Output: results_arc/ca_summary.md and ca_summary.csv; exit 1 if either single-arm anchor
        (ca_s_t1_p3, ca_s_fl_p3) is more than 0.02 dB from its registered GS-single value.
usage: ca_summarize.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diag_artefact.headonly import head_mask  # noqa: E402

# Each cell of the 2x2 design: (arm, lambda_edge, contrast) -> (exp, channel).
# A joint run carries T1 in its 'plain' channel and FLAIR in its 'contrast' channel; a single run
# reports its one contrast under 'plain' (its 'contrast' channel is a second, unreported
# reconstruction -- see task-3-brief.md).
CELLS = [
    ("joint", 0.001, "T1", "hp_n36", "plain"),
    ("joint", 0.001, "FLAIR", "hp_n36", "contrast"),
    ("single", 0.001, "T1", "ca_s_t1_p3", "plain"),
    ("single", 0.001, "FLAIR", "ca_s_fl_p3", "plain"),
    ("joint", 0.0, "T1", "ca_j_edge0", "plain"),
    ("joint", 0.0, "FLAIR", "ca_j_edge0", "contrast"),
    ("single", 0.0, "T1", "ca_s_t1_edge0", "plain"),
    ("single", 0.0, "FLAIR", "ca_s_fl_edge0", "plain"),
]

# cycle.py's fixed npz keys, by channel (add_gs_arm.py) -- independent of which contrast that
# channel actually holds.
NPZ_KEYS = {"plain": ("recT1", "gtT1"), "contrast": ("recFL", "gtFL")}
JSON_KEYS = ("head_psnr", "head_ssim", "head_hfen", "psnr", "ssim", "nmse", "head_psnr_cplx")

ANCHOR_REF = {"T1": 35.558, "FLAIR": 34.886}
ANCHOR_EXP = {"T1": "ca_s_t1_p3", "FLAIR": "ca_s_fl_p3"}
TOL = 0.02


def margins(rows):
    """rows[(arm, lambda_edge)][contrast] -> head PSNR.  Returns per-contrast M1 / M0 / delta.

    M1 = joint - single at the published lambda_edge 0.001 (the margin the paper quotes)
    M0 = joint - single at lambda_edge 0 (what shared geometry alone buys)
    delta = M1 - M0 (what the common-edge prior buys)
    """
    out = {}
    for c in ("T1", "FLAIR"):
        m1 = rows[("joint", 0.001)][c] - rows[("single", 0.001)][c]
        m0 = rows[("joint", 0.0)][c] - rows[("single", 0.0)][c]
        out[c] = {"M1": m1, "M0": m0, "delta": m1 - m0}
    return out


def head_nmse(rec: np.ndarray, gt: np.ndarray) -> float:
    rec = np.asarray(rec, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    head = head_mask(gt.astype(np.float32))
    den = float((gt[head] ** 2).sum())
    if den == 0.0:
        return float("inf")
    return float(((rec - gt)[head] ** 2).sum() / den)


def load_cell(arc: Path, exp: str, channel: str, contrast: str) -> dict:
    d = json.load(open(arc / f"{exp}_score.json"))
    name = d.get(f"{channel}_name")
    assert name == contrast, f"{exp}: {channel}_name is {name!r}, expected {contrast!r}"
    c = d[channel]
    out = {"exp": exp, "channel": channel}
    for k in JSON_KEYS:
        out[k] = float(c[k]) if k in c else float("nan")
    rk, gk = NPZ_KEYS[channel]
    with np.load(arc / f"{exp}_volumes.npz") as z:
        out["head_nmse"] = head_nmse(z[rk], z[gk])
    return out


def build_cells(arc: Path) -> dict:
    """(arm, lambda_edge) -> {"T1": {...metrics...}, "FLAIR": {...metrics...}}"""
    out = {}
    for arm, lam, contrast, exp, channel in CELLS:
        out.setdefault((arm, lam), {})[contrast] = load_cell(Path(arc), exp, channel, contrast)
    return out


def anchor_check(cells: dict) -> dict:
    out = {}
    for c in ("T1", "FLAIR"):
        value = cells[("single", 0.001)][c]["head_psnr"]
        delta = value - ANCHOR_REF[c]
        out[c] = {"exp": ANCHOR_EXP[c], "value": value, "ref": ANCHOR_REF[c], "delta": delta,
                   "passed": abs(delta) <= TOL}
    return out


def _f(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render_md(cells: dict, m: dict, gate: dict) -> str:
    lines = ["| arm | lambda_edge | contrast | exp | channel | head PSNR / SSIM / HFEN / NMSE | "
             "whole PSNR / NMSE | cplx head PSNR |",
             "|---|---|---|---|---|---|---|---|"]
    for arm, lam, contrast, exp, channel in CELLS:
        v = cells[(arm, lam)][contrast]
        lines.append(f"| {arm} | {lam} | {contrast} | {exp} | {channel} | "
                     f"{_f(v['head_psnr'])} / {_f(v['head_ssim'], 4)} / {_f(v['head_hfen'], 4)} / "
                     f"{_f(v['head_nmse'], 5)} | {_f(v['psnr'])} / {_f(v['nmse'], 5)} | "
                     f"{_f(v['head_psnr_cplx'])} |")
    lines.append("")
    lines.append("| contrast | M1 (joint - single @ 0.001) | M0 (joint - single @ 0) | delta (M1 - M0) |")
    lines.append("|---|---|---|---|")
    for c in ("T1", "FLAIR"):
        lines.append(f"| {c} | {m[c]['M1']:+.3f} | {m[c]['M0']:+.3f} | {m[c]['delta']:+.3f} |")
    lines.append("")
    verdict = "PASS" if gate["T1"]["passed"] and gate["FLAIR"]["passed"] else "FAIL"
    lines.append(f"Anchor gate (single arm at lambda_edge 0.001 vs registered GS-single, tolerance {TOL:.2f} dB): "
                 f"{gate['T1']['exp']} T1 {_f(gate['T1']['value'])} vs {gate['T1']['ref']:.3f} "
                 f"(delta {gate['T1']['delta']:+.3f}); {gate['FLAIR']['exp']} FLAIR {_f(gate['FLAIR']['value'])} "
                 f"vs {gate['FLAIR']['ref']:.3f} (delta {gate['FLAIR']['delta']:+.3f}) -> {verdict}")
    return "\n".join(lines) + "\n"


def write_csv(cells: dict, path: Path) -> None:
    cols = ["arm", "lambda_edge", "contrast", "exp", "channel"] + list(JSON_KEYS) + ["head_nmse"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for arm, lam, contrast, exp, channel in CELLS:
            v = cells[(arm, lam)][contrast]
            row = {"arm": arm, "lambda_edge": lam, "contrast": contrast, "exp": exp, "channel": channel}
            row.update({k: v[k] for k in JSON_KEYS})
            row["head_nmse"] = v["head_nmse"]
            w.writerow(row)


def main(argv=None) -> int:
    arc = ROOT / "results_arc"
    cells = build_cells(arc)
    margin_rows = {key: {c: v["head_psnr"] for c, v in per_contrast.items()} for key, per_contrast in cells.items()}
    m = margins(margin_rows)
    gate = anchor_check(cells)
    md = render_md(cells, m, gate)
    (arc / "ca_summary.md").write_text(md)
    write_csv(cells, arc / "ca_summary.csv")
    print(md)
    if not (gate["T1"]["passed"] and gate["FLAIR"]["passed"]):
        print("ANCHOR GATE FAILED -- the single-arm anchors do not reproduce; stop and investigate.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
