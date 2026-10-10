#!/usr/bin/env python3
"""Subject-level 2x2 table of the sg2x2 core test (spec section 5.3).

Runs are named <prefix>_<subject>_s<seed>_<CELL> (the cells of scripts/sg2x2_table.collect). Per subject each
difference is paired seed by seed over the seeds present in every cell it needs, then averaged (spec §4, §5.3);
the cell means average each cell's own seeds; then, over the TEST subjects only
(scripts/sg2x2_core_subjects.py), the paired
differences A−B (shared geometry with the per-channel priors), C−D (shared geometry without explicit priors) and
(A−B)−(C−D) are summarised as N, mean, a 95 % subject-level bootstrap interval (10,000 resamples, generator seed 0),
SD and k/N: the number of subjects in which the difference points the "better" way for that metric (higher PSNR /
SSIM, lower NMSE / HFEN). A subject enters a row only when every cell the row needs exists for it. The development
subject is listed separately and never enters the statistics.

    env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/sg2x2_core_table.py --prefix sg2x2_core_af16 --seeds 42 [43 44]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sg2x2_core_subjects import SUBJECTS, core_test_subjects  # noqa: E402
from sg2x2_table import CELLS, CONTRASTS, METRICS, collect     # noqa: E402

HIGHER_BETTER = {"head_psnr": True, "head_ssim": True, "head_nmse": False, "head_hfen": False}
ROWS = ("A−B", "C−D", "(A−B)−(C−D)")
N_BOOT = 10_000


def _mean(vals: list[float]) -> float:
    v = [x for x in vals if not math.isnan(x)]
    return float(sum(v) / len(v)) if v else float("nan")


def _means(got: dict, seeds: list[int]) -> dict:
    """(cell, contrast) -> {metric: mean over the seeds present, "n_seeds": k}"""
    out = {}
    for cell in CELLS:
        for c in CONTRASTS:
            rows = [got[(cell, c, s)] for s in seeds if (cell, c, s) in got]
            if rows:
                out[(cell, c)] = {k: _mean([r[k] for r in rows]) for k, _, _ in METRICS}
                out[(cell, c)]["n_seeds"] = len(rows)
    return out


def subject_means(arc: Path, prefix: str, subject: str, seeds: list[int]) -> dict:
    """(cell, contrast) -> {metric: mean over the seeds present, "n_seeds": k}"""
    return _means(collect(Path(arc), f"{prefix}_{subject}", seeds), seeds)


def differences(means: dict, contrast: str, metric: str) -> dict:
    v = {cell: means.get((cell, contrast), {}).get(metric, float("nan")) for cell in CELLS}
    ab, cd = v["A"] - v["B"], v["C"] - v["D"]
    return {"A−B": ab, "C−D": cd, "(A−B)−(C−D)": ab - cd,
            "A-B": ab, "C-D": cd, "(A-B)-(C-D)": ab - cd}


ROW_CELLS = {"A−B": ("A", "B"), "C−D": ("C", "D"), "(A−B)−(C−D)": ("A", "B", "C", "D")}


def paired_differences(got: dict, contrast: str, metric: str, seeds: list[int]) -> dict:
    """row -> mean, over the seeds present in every cell the row needs, of that seed's difference (NaN when no seed
    qualifies): seeds are paired across the cells first (spec §4), then averaged within the subject (spec §5.3)."""
    out = {}
    for row, cells in ROW_CELLS.items():
        vals = []
        for s in seeds:
            if all((cell, contrast, s) in got for cell in cells):
                one = {(cell, contrast): got[(cell, contrast, s)] for cell in cells}
                d = differences(one, contrast, metric)[row]
                if not math.isnan(d):
                    vals.append(d)
        out[row] = _mean(vals)
    return out


def summary(values: list[float], higher_better: bool) -> dict:
    x = np.array([v for v in values if not math.isnan(v)], dtype=float)
    n = int(x.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "ci": (float("nan"), float("nan")), "k_better": 0}
    if n >= 2:
        boots = np.random.default_rng(0).choice(x, size=(N_BOOT, n), replace=True).mean(axis=1)
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        sd = float(x.std(ddof=1))
    else:
        ci, sd = (float("nan"), float("nan")), float("nan")
    k = int((x > 0).sum()) if higher_better else int((x < 0).sum())
    return {"n": n, "mean": float(x.mean()), "sd": sd, "ci": ci, "k_better": k}


def _f(v: float, fmt: str, signed: bool = True) -> str:
    if math.isnan(v):
        return "n/a"
    return ("+" if signed and v >= 0 else "") + fmt.format(v)


def table(arc: Path, prefix: str, seeds: list[int]) -> str:
    tests = core_test_subjects()
    dev = [s for s, (_, r) in SUBJECTS.items() if r == "dev"]
    gots = {s: collect(Path(arc), f"{prefix}_{s}", seeds) for s in tests + dev}
    means = {s: _means(gots[s], seeds) for s in tests + dev}
    have = [s for s in tests if means[s]]
    lines = [f"CORE TEST: {len(have)} of {len(tests)} test subjects with results, seeds {seeds}. Head-only metrics "
             f"(|GT| > 0.05 P99.9). Per subject each difference is paired seed by seed, then averaged; intervals are "
             f"95 % subject-level bootstrap "
             f"({N_BOOT} resamples, seed 0); k/N = subjects in which the shared-geometry cell is better (interaction: the "
             f"difference points the better way). The development subject ({', '.join(dev)}) is listed separately and "
             f"never enters the statistics.", "",
             "| row | contrast | metric | N | mean | 95 % CI | SD | k/N |", "|---|---|---|---|---|---|---|---|"]
    for row in ROWS:
        for c in CONTRASTS:
            for key, name, fmt in METRICS:
                vals = [paired_differences(gots[s], c, key, seeds)[row] for s in tests]
                sm = summary(vals, HIGHER_BETTER[key])
                lines.append(f"| {row} | {c} | {name} | {sm['n']} | {_f(sm['mean'], fmt)} | "
                             f"[{_f(sm['ci'][0], fmt)}, {_f(sm['ci'][1], fmt)}] | {_f(sm['sd'], fmt, False)} | "
                             f"{sm['k_better']}/{sm['n']} |")
    lines += ["", "Cell means over the test subjects (mean ± SD, N):", "",
              "| cell | contrast | " + " | ".join(n for _, n, _ in METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for cell in CELLS:
        for c in CONTRASTS:
            cells = []
            for key, _, fmt in METRICS:
                v = np.array([means[s][(cell, c)][key] for s in tests if (cell, c) in means[s]
                              and not math.isnan(means[s][(cell, c)][key])])
                cells.append("n/a" if v.size == 0 else
                             f"{fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1)) if v.size > 1 else 'n/a'} ({v.size})")
            lines.append(f"| {cell} | {c} | " + " | ".join(cells) + " |")
    lines += ["", "Per subject, head PSNR (T1 / FLAIR) of A, B, C, D and A−B, C−D:", "",
              "| subject | role | A | B | C | D | A−B | C−D |", "|---|---|---|---|---|---|---|---|"]
    for s in tests + dev:
        if not means[s]:
            lines.append(f"| {s} | {SUBJECTS[s][1]} | no runs |  |  |  |  |  |")
            continue
        def pair(cell):
            return " / ".join(_f(means[s].get((cell, c), {}).get("head_psnr", float("nan")), "{:.3f}", False)
                              for c in CONTRASTS)
        dd = {c: paired_differences(gots[s], c, "head_psnr", seeds) for c in CONTRASTS}
        lines.append(f"| {s} | {SUBJECTS[s][1]} | {pair('A')} | {pair('B')} | {pair('C')} | {pair('D')} | "
                     f"{' / '.join(_f(dd[c]['A−B'], '{:.3f}') for c in CONTRASTS)} | "
                     f"{' / '.join(_f(dd[c]['C−D'], '{:.3f}') for c in CONTRASTS)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--arc", default="results_arc")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    md = table(ROOT / a.arc, a.prefix, a.seeds)
    out = Path(a.out) if a.out else ROOT / a.arc / f"{a.prefix}_core_table.md"
    with open(out, "x") as f:
        f.write(md)
    print(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
