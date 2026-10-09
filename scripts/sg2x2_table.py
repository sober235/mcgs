#!/usr/bin/env python3
"""The 2x2 table of the shared-geometry ablation (spec section 5.3 / 6) from arc/cycle.py score jsons.

Runs are named <prefix>_s<seed>_<CELL>; a joint run scores T1 on its plain slot and FLAIR on its contrast slot, a
single run scores its one contrast on the plain slot. Head NMSE is recomputed from the saved volumes on the scorer's
head mask (the json has none), as scripts/hp_summarize.py does; NaN when no volumes were saved.
    env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/sg2x2_table.py --prefix sg2x2_pilot_af16 --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diag_artefact.headonly import head_mask  # noqa: E402

CELLS = ("A", "B", "C", "D")
CONTRASTS = ("T1", "FLAIR")
METRICS = (("head_psnr", "PSNR", "{:.3f}"), ("head_ssim", "SSIM", "{:.4f}"), ("head_nmse", "NMSE", "{:.5f}"),
           ("head_hfen", "HFEN", "{:.4f}"))


def head_nmse(rec: np.ndarray, gt: np.ndarray) -> float:
    rec = np.asarray(rec, dtype=np.float64); gt = np.asarray(gt, dtype=np.float64)
    head = head_mask(gt.astype(np.float32))
    den = float((gt[head] ** 2).sum())
    return float(((rec - gt)[head] ** 2).sum() / den) if den > 0 else float("nan")


def _metrics(score: dict, vol, rk: str, gk: str) -> dict:
    out = {k: float(score.get(k, float("nan"))) for k, _, _ in METRICS if k != "head_nmse"}
    out["head_nmse"] = head_nmse(vol[rk], vol[gk]) if vol is not None and rk in vol.files else float("nan")
    return out


def collect(arc: Path, prefix: str, seeds: list[int]) -> dict:
    got = {}
    for s in seeds:
        for cell in CELLS:
            if cell in ("A", "C"):
                exp = f"{prefix}_s{s}_{cell}"
                p = arc / f"{exp}_score.json"
                if not p.exists():
                    continue
                row = json.load(open(p))
                vp = arc / f"{exp}_volumes.npz"
                vol = np.load(vp) if vp.exists() else None
                got[(cell, "T1", s)] = _metrics(row["plain"], vol, "recT1", "gtT1")
                got[(cell, "FLAIR", s)] = _metrics(row["contrast"], vol, "recFL", "gtFL")
            else:
                for c in CONTRASTS:
                    exp = f"{prefix}_s{s}_{cell}_{c}"
                    p = arc / f"{exp}_score.json"
                    if not p.exists():
                        continue
                    row = json.load(open(p))
                    vp = arc / f"{exp}_volumes.npz"
                    vol = np.load(vp) if vp.exists() else None
                    got[(cell, c, s)] = _metrics(row["plain"], vol, "rec", "gt")
    return got


def _mean_sd(vals: list[float]) -> tuple[float, float, int]:
    """Mean, SD and the count k of non-NaN values; SD is NaN when k < 2 (undefined, not zero)."""
    v = [x for x in vals if not math.isnan(x)]
    k = len(v)
    if k == 0:
        return float("nan"), float("nan"), 0
    m = sum(v) / k
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (k - 1)) if k > 1 else float("nan")
    return m, sd, k


def _fmt(m: float, sd: float, fmt: str, signed: bool = False, k: int | None = None, n: int | None = None) -> str:
    if k == 0 or math.isnan(m):
        return "—"
    s = fmt.format(abs(m)) if signed else fmt.format(m)
    if signed:
        s = ("+" if m >= 0 else "−") + s
    sd_txt = "n/a" if k == 1 else fmt.format(sd)
    cell = f"{s} ± {sd_txt}"
    if k is not None and n is not None and k < n:
        cell += f" (n={k}/{n})"
    return cell


def table(got: dict, seeds: list[int]) -> str:
    lines = ["| cell | contrast | " + " | ".join(n for _, n, _ in METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for cell in CELLS:
        for c in CONTRASTS:
            vals = [got.get((cell, c, s)) for s in seeds]
            cells = []
            for k, _, fmt in METRICS:
                m, sd, kk = _mean_sd([v[k] if v else float("nan") for v in vals])
                cells.append(_fmt(m, sd, fmt, k=kk, n=len(seeds)))
            lines.append(f"| {cell} | {c} | " + " | ".join(cells) + " |")
    diffs = (("A−B", "A", "B"), ("C−D", "C", "D"))
    for name, x, y in diffs:
        for c in CONTRASTS:
            cells = []
            for k, _, fmt in METRICS:
                d = [got[(x, c, s)][k] - got[(y, c, s)][k] if (x, c, s) in got and (y, c, s) in got else float("nan")
                     for s in seeds]
                m, sd, kk = _mean_sd(d)
                cells.append(_fmt(m, sd, fmt, signed=True, k=kk, n=len(seeds)))
            lines.append(f"| {name} | {c} | " + " | ".join(cells) + " |")
    for c in CONTRASTS:
        cells = []
        for k, _, fmt in METRICS:
            d = []
            for s in seeds:
                keys = [(q, c, s) for q in CELLS]
                if all(kk in got for kk in keys):
                    d.append((got[keys[0]][k] - got[keys[1]][k]) - (got[keys[2]][k] - got[keys[3]][k]))
                else:
                    d.append(float("nan"))
            m, sd, kk = _mean_sd(d)
            cells.append(_fmt(m, sd, fmt, signed=True, k=kk, n=len(seeds)))
        lines.append(f"| (A−B)−(C−D) | {c} | " + " | ".join(cells) + " |")
    n = len(seeds)
    return (f"Head-only metrics (|GT| > 0.05 P99.9), mean ± SD over {n} seed(s) {seeds}; differences are per-seed "
            f"paired differences. PILOT numbers on one subject are engineering checks, not evidence.\n\n" + "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--arc", default="results_arc"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arc = ROOT / a.arc
    md = table(collect(arc, a.prefix, a.seeds), a.seeds)
    out = Path(a.out) if a.out else arc / f"{a.prefix}_table.md"
    out.write_text(md)
    print(md); print(f"wrote {out}")


if __name__ == "__main__":
    main()
