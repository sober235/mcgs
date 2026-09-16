#!/usr/bin/env python3
"""hp_fig_curves.py -- the two sensitivity panels of the hyper-parameter study (spec section 1), as small multiples:
column 1 = shared Gaussian count N_s (log-2 axis, 0.9M .. 7.2M), column 2 = private ratio N_p / N_s at N_s = 3.6M;
row 1 = head PSNR, row 2 = head SSIM. Two series per panel, T1 and FLAIR (both MC-GS joint), hue by contrast in the
trend topic's palette (validated with the dataviz checker: blue #2a78d6, orange #eb6834, all six checks pass), circle vs
square markers with a surface ring, one legend for the figure, the two lines direct-labelled at their right ends, the
published recipe (3.6M, N_p = 0) marked by a hairline and a label, no value on any point (the table hp_summary.md carries
the numbers). Reads results_arc/hp_summary.csv (scripts/hp_summarize.py); draws whatever runs are done and says
"(partial)" in the title while any of the six is missing.
Output: <topic>/figures/hp_sensitivity_psnr_ssim.{png,pdf} (200 dpi PNG, TrueType PDF) and a copy of the PNG in
/home/congcongliu/figs/t1flair_v3_af16_hparam/ (served at http://localhost:8765/t1flair_v3_af16_hparam/).
usage: hp_fig_curves.py [--csv results_arc/hp_summary.csv] [--topic <folder>]
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOPIC = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_hparam_v3_jiangyizhou")
SERVED = Path("/home/congcongliu/figs/t1flair_v3_af16_hparam")
BLUE, ORANGE = "#2a78d6", "#eb6834"                      # T1, FLAIR (categorical, fixed order)
SURF, INK, INK2, GRID, BASE = "#ffffff", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SERIES = (("T1", BLUE, "o"), ("FLAIR", ORANGE, "s"))
RECIPE_N, RECIPE_RATIO = 3.6, 0.0

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "text.color": INK,
})


def load(csv_path: Path):
    rows = [r for r in csv.DictReader(open(csv_path)) if r["status"] == "done"]
    n_all = sum(1 for _ in csv.DictReader(open(csv_path)))
    cap = sorted((r for r in rows if int(r["n_private"]) == 0), key=lambda r: int(r["n_shared"]))
    priv = sorted((r for r in rows if int(r["n_shared"]) == 3_600_000), key=lambda r: float(r["ratio"]))
    return cap, priv, len(rows), n_all


def panel(ax, series, xkey, metric, ylabel, xlabel, recipe_x, recipe_text, top_row=False):
    xs = [int(r[xkey]) / 1e6 if xkey == "n_shared" else float(r[xkey]) for r in series]
    ax.axvline(recipe_x, color=BASE, lw=1.0, zorder=1)
    ax.text(recipe_x, 1.0, recipe_text, transform=ax.get_xaxis_transform(),
            ha="center" if xkey == "n_shared" else "left", va="bottom", fontsize=7, color=INK2)
    for name, color, marker in SERIES:
        ys = [float(r[f"{name}_{metric}"]) for r in series]
        ax.plot(xs, ys, "-", color=color, lw=1.6, zorder=3, solid_joinstyle="round", solid_capstyle="round")
        ax.plot(xs, ys, linestyle="none", marker=marker, ms=6, mfc=color, mec=SURF, mew=1.2, zorder=4)
        if xs:
            ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=7.5, color=INK)
    if xkey == "n_shared":
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.9, 1.8, 3.6, 7.2])
        ax.set_xticklabels(["0.9M", "1.8M", "3.6M", "7.2M"])
        ax.set_xlim(0.9 / 1.25, 7.2 * 1.45)
        ax.minorticks_off()
    else:
        ax.set_xticks([0.0, 0.05, 0.10])
        ax.set_xticklabels(["0 %", "5 %", "10 %"])
        ax.set_xlim(-0.012, 0.128)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="y", color=GRID, lw=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, width=0.8)
    if top_row:
        ax.tick_params(labelbottom=False)
    if metric == "head_ssim":
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
    else:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "results_arc" / "hp_summary.csv"))
    ap.add_argument("--topic", default=str(TOPIC))
    a = ap.parse_args(argv)
    cap, priv, n_done, n_all = load(Path(a.csv))
    partial = n_done < n_all
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), dpi=200)
    panel(axes[0, 0], cap, "n_shared", "head_psnr", "head PSNR (dB)", "", RECIPE_N, "recipe", top_row=True)
    panel(axes[1, 0], cap, "n_shared", "head_ssim", "head SSIM", "shared Gaussians $N_s$", RECIPE_N, "")
    panel(axes[0, 1], priv, "ratio", "head_psnr", "", "", RECIPE_RATIO, " published recipe ($N_p$ = 0)", top_row=True)
    panel(axes[1, 1], priv, "ratio", "head_ssim", "", "private ratio $N_p / N_s$ at $N_s$ = 3.6M", RECIPE_RATIO, "")
    handles = [Line2D([], [], color=c, marker=m, ms=6, mfc=c, mec=SURF, mew=1.2, lw=1.6, label=n) for n, c, m in SERIES]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.985), frameon=False, ncol=2,
               handlelength=3.0, numpoints=1, fontsize=8)
    title = ("MC-GS sensitivity to the Gaussian budget" + (" (partial: %d of %d runs)" % (n_done, n_all) if partial else ""))
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=10, color=INK, weight="medium")
    fig.text(0.01, 0.005,
             "jiangyizhou T1 + FLAIR, nominal AF 16 house vd-Poisson mask (T1 15.19x / FLAIR 16.49x), frozen phase-3 recipe, "
             "4000 iterations, seed 42.\nHead-masked metrics (|GT| > 0.05 P99.9). Left: shared count only. Right: FLAIR-private "
             "Gaussians added on top of the 3.6M recipe. Numbers: hp_summary.md.",
             ha="left", va="bottom", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.955))
    out_dir = Path(a.topic) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    SERVED.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = out_dir / f"hp_sensitivity_psnr_ssim.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor=SURF)
        if ext == "png":
            shutil.copy(out, SERVED / out.name)
    print(f"wrote {out_dir}/hp_sensitivity_psnr_ssim.{{png,pdf}} ({n_done}/{n_all} runs) and {SERVED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
