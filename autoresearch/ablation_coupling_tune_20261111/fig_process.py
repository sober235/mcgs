#!/usr/bin/env python3
"""fig_process.py -- how the stage-three tuning went on tangbing T1+T2 (program.md section 11.8).

Two panels (T1, T2), one y-axis each: head PSNR of the full MC-GS keeper after each round, and of the best
independent single-contrast model run so far (the comparator of the decision metric). The gap between the two
lines is the margin printed under each round. Numbers are read from experiments.jsonl beside this file (rows of
task t1t2tb, seed 42); the two registered starting points (V32 full, V32 independents) are the values of
RESULTS.md section 7. CPU only, reads no server file.

    gpu.py run --cpu -- @python autoresearch/ablation_coupling_tune_20261111/fig_process.py [KEEP_DIR]
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = {r["name"]: r for r in map(json.loads, open(HERE / "experiments.jsonl")) if r.get("task") == "t1t2tb"}
REG_FULL = {"T1": 33.791, "T2": 34.656}                  # fw_t1t2tbv32_full
REG_IND = {"T1": 33.795, "T2": 34.760}                   # fw_t1t2tb_nojoint_T1 / _T2
# round label, change kept in that round, the keeper's full run, independents that joined the pool in that round
STAGES = [("Start", "V32 recipe", None, []),
          ("R1", "T2 wavelet\n0.05 → 0.08", "r1_wav008_full", ["r1_khi3_t2", "r1_wav008_t2"]),
          ("R2", "common-edge\nprior *\n0 → 0.001", "r2_edge001_full", ["r2_wav012_t2", "r2_geo005_t2", "r2_geo005_t1"]),
          ("R3", "rotation LR\n0.0003\n→ 0.008", "r3_rot008_full", ["r3_rot008_t1", "r3_rot008_t2"]),
          ("R4", "scale LR\n0.002\n→ 0.004", "r4_scl004_full", ["r4_scl004_t1", "r4_scl004_t2"]),
          ("R5", "edge-gate\nsharpness *\n10 → 100", "r5_scl004_ekappa100_full", [])]
full, best = {"T1": [], "T2": []}, {"T1": [], "T2": []}
pool = {c: [REG_IND[c]] for c in ("T1", "T2")}
for _, _, exp, ind in STAGES:
    for n in ind:
        r = rows[n]
        c = "T1" if r["kind"] == "nojoint_p" else "T2"
        pool[c].append(r[c]["head_psnr"])
    for c in ("T1", "T2"):
        full[c].append(REG_FULL[c] if exp is None else rows[exp][c]["head_psnr"])
        best[c].append(max(pool[c]))

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_FULL, C_IND = "#2a78d6", "#eb6834"                     # categorical slots 1 and 2 of the reference palette
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig = plt.figure(figsize=(14.5, 7.4), facecolor=SURF)
gs = fig.add_gridspec(2, 2, height_ratios=(6, 1), hspace=0.06, wspace=0.2, left=0.07, right=0.965, top=0.87, bottom=0.25)
x = list(range(len(STAGES)))
for j, c in enumerate(("T1", "T2")):
    ax, am = fig.add_subplot(gs[0, j]), fig.add_subplot(gs[1, j])
    ax.set_facecolor(SURF)
    ax.plot(x, best[c], color=C_IND, lw=2, marker="s", ms=8, mec=SURF, mew=2, label="best independent model so far", zorder=3)
    ax.plot(x, full[c], color=C_FULL, lw=2, marker="o", ms=9, mec=SURF, mew=2, label="full MC-GS (kept configuration)", zorder=4)
    lo, hi = min(full[c] + best[c]), max(full[c] + best[c])
    span = hi - lo
    ax.set_ylim(lo - 0.14 * span, hi + 0.14 * span)
    ax.annotate(f"{full[c][-1]:.3f}", (x[-1], full[c][-1]), xytext=(10, 3), textcoords="offset points", color=INK, fontsize=11)
    ax.annotate(f"{best[c][-1]:.3f}", (x[-1], best[c][-1]), xytext=(10, -10), textcoords="offset points", color=INK, fontsize=11)
    ax.annotate(f"{full[c][0]:.3f}", (x[0], full[c][0]), xytext=(0, -17), textcoords="offset points", color=INK, fontsize=11, ha="center")
    ax.annotate(f"{best[c][0]:.3f}", (x[0], best[c][0]), xytext=(0, 10), textcoords="offset points", color=INK, fontsize=11, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.set_xlim(-0.55, len(x) - 0.3)
    ax.set_ylabel("head PSNR (dB)")
    ax.set_title(c, fontsize=15, fontweight="bold", loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    if j == 0:
        ax.legend(loc="upper left", frameon=False, fontsize=11)
    # margin row: full minus best independent, one number per round
    am.set_facecolor(SURF)
    am.set_xlim(ax.get_xlim())
    am.set_ylim(0, 1)
    for i in x:
        m = full[c][i] - best[c][i]
        am.text(i, 0.5, f"{m:+.3f}", ha="center", va="center", fontsize=12.5,
                fontweight="bold" if m > 0 else "normal", color=INK if m > 0 else INK2)
    am.set_yticks([])
    am.set_ylabel("full − indep.\n(dB)", rotation=0, ha="right", va="center", fontsize=9.5, labelpad=6)
    am.set_xticks(x)
    am.set_xticklabels([f"{r}\n{t}" for r, t, _, _ in STAGES], fontsize=9.5, linespacing=1.25)
    am.tick_params(length=0, pad=8)
    for sp in ("top", "right", "left"):
        am.spines[sp].set_visible(False)
fig.suptitle("Tuning of the full MC-GS against the independent single-contrast models  —  T1+T2, AF12 (tangbing), seed 42",
             fontsize=14, fontweight="bold", x=0.02, ha="left", color=INK)
fig.text(0.02, 0.02, "Each round adds one change to the kept configuration.  * = setting of the joint model only; the other changes are also given to the "
         "independent models, which are rerun.\nThe comparator is the best independent model of all runs so far, so its line can only rise.  "
         "One subject, tuned on its own ground-truth score; seed 7 repeats the last point: T1 +0.124, T2 +0.031 dB.",
         fontsize=9.5, color=INK2, ha="left", va="bottom")
out = Path(os.environ.get("MCGS_ARTIFACT_DIR", HERE))
fig.savefig(out / "t1t2tb_tuning_process.png", dpi=200, facecolor=SURF)
if len(sys.argv) > 1:                                    # optional: a directory that also gets the PNG and a vector PDF
    keep = Path(sys.argv[1])
    keep.mkdir(parents=True, exist_ok=True)
    fig.savefig(keep / "t1t2tb_tuning_process.png", dpi=200, facecolor=SURF)
    fig.savefig(keep / "t1t2tb_tuning_process.pdf", facecolor=SURF)
    print("kept", keep)
print("T1 full", [round(v, 3) for v in full["T1"]], "best", [round(v, 3) for v in best["T1"]])
print("T2 full", [round(v, 3) for v in full["T2"]], "best", [round(v, 3) for v in best["T2"]])
print("wrote", out / "t1t2tb_tuning_process.png")
