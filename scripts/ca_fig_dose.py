#!/usr/bin/env python3
"""ca_fig_dose.py -- Stage 2 dose-response figure for the coupling-attribution study
(spec docs/superpowers/specs/2026-09-20-coupling-attribution-design.md, section 3, Stage 2).

Small multiples, 2 rows x 2 columns, columns T1 / FLAIR sharing the x-axis top-to-bottom. Row 1 =
head PSNR (dB): the joint arm (5 points: 0, 0.0003, 0.001, 0.003, 0.01) and the single arm (3 points:
0, 0.001, 0.003 -- drawn only where a run exists, not interpolated across the two lambda_edge values
the single arm never ran). Row 2 = the margin joint - single head PSNR, at the 3 lambda_edge values
where both arms exist -- its own axis, not a twin of row 1, so a reader can read the margin curve's
shape (monotonic vs. peak-and-fall) without it sharing pixels with two absolute-PSNR curves ~150x
larger. x = lambda_edge, log-scaled in every panel, with 0 placed at a fixed offset left of the
smallest run value (0.0003) since a log axis cannot represent zero -- a dotted separator marks that
gap as an axis break, not a to-scale distance, and the tick is still labelled "0" so the point is
never silently dropped. A vertical grey line marks the published lambda_edge 0.001 in every panel.

Revision note (previous version used a twin axis for the margin, matched to the primary axis's
dB-per-pixel scale to keep slopes comparable -- see git history). That fixed the slope-comparison
hazard but left a second one: positional confusability, where two quantities on independently
re-centred axes could land at the same pixel height by coincidence, and the margin curve's own shape
in isolation (the thing the spec's decision rule turns on) was harder to read sharing a panel with two
other curves. Small multiples put the margin on its own axis and its own row instead, which removes
both problems without dropping the "same x, aligned across quantities" property -- that is exactly
what sharing the x-axis column-wise gives.

Colour: two categorical slots for the two identity series (joint, single) validated with the dataviz
skill's checker -- node scripts/validate_palette.js "#2a78d6,#eb6834" --mode light -> ALL CHECKS PASS
(worst adjacent CVD dE 24.7, normal-vision dE 33.6, both >> the 8 / 15 targets) -- and a third,
deliberately non-categorical neutral grey (#6b6a63, WCAG contrast 5.43:1 on the #ffffff surface) for
the margin, since it is a derived quantity rather than a third identity (dataviz color-formula.md's
documented scope: the six-check validator covers categorical identity palettes, not a lone
status/derived colour, which is instead checked for WCAG contrast). Palette and marks otherwise
follow scripts/hp_fig_curves.py (house style for this topic).

Input : results_arc/ca_dose.csv (scripts/ca_summarize.py).
Output: figures/ca_dose_lambda_edge.{png,pdf} (200 dpi PNG, TrueType PDF).
usage: ca_fig_dose.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results_arc" / "ca_dose.csv"
OUT_DIR = ROOT / "figures"

JOINT, SINGLE, MARGIN = "#2a78d6", "#eb6834", "#6b6a63"          # categorical slots 1/2, neutral grey
SURF, INK, INK2, GRID, BASE = "#ffffff", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
PUBLISHED = 0.001
GAP = 0.8   # decades of visual gap between the "0" tick and the smallest nonzero lambda_edge tick

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "text.color": INK,
})


def load(csv_path: Path):
    """-> {contrast: {"joint": [(lambda, psnr), ...], "single": [...], "margin": [...]}}"""
    out = {"T1": {"joint": [], "single": [], "margin": []},
           "FLAIR": {"joint": [], "single": [], "margin": []}}
    for r in csv.DictReader(open(csv_path)):
        out[r["contrast"]][r["arm"]].append((float(r["lambda_edge"]), float(r["head_psnr"])))
    for c in out:
        for arm in out[c]:
            out[c][arm].sort(key=lambda p: p[0])
    return out


def xpos(lam: float, zero_x: float) -> float:
    return zero_x if lam == 0.0 else math.log10(lam)


def axis_cosmetics(ax, zero_x, lam_ticks, xlim, show_xlabel, ylabel, yfmt):
    """Shared x-axis treatment (published marker, zero-offset tick, axis-break separator) and grid /
    spine / label chrome, common to every panel regardless of which quantity it plots."""
    ax.axvline(xpos(PUBLISHED, zero_x), color=BASE, lw=1.0, zorder=1)
    ax.text(xpos(PUBLISHED, zero_x), 1.0, " published", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=7, color=INK2)
    break_x = (zero_x + xpos(min(l for l in lam_ticks if l > 0), zero_x)) / 2
    ax.axvline(break_x, color=GRID, lw=1.0, linestyle=":", zorder=1)

    ax.set_xlim(*xlim)
    ax.set_xticks([xpos(l, zero_x) for l in lam_ticks])
    ax.minorticks_off()
    if show_xlabel:
        ax.set_xticklabels([("0" if l == 0.0 else f"{l:g}") for l in lam_ticks])
        ax.set_xlabel(r"$\lambda_{edge}$ (log scale; 0 offset left of 0.0003, axis break)")
    else:
        ax.tick_params(labelbottom=False)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color=GRID, lw=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(yfmt))


def panel_psnr(ax, data, contrast, zero_x, lam_ticks, xlim, show_xlabel, ylabel, title=None):
    joint = data[contrast]["joint"]
    single = data[contrast]["single"]

    jx = [xpos(l, zero_x) for l, _ in joint]
    jy = [v for _, v in joint]
    ax.plot(jx, jy, "-", color=JOINT, lw=1.6, zorder=3, solid_joinstyle="round", solid_capstyle="round")
    ax.plot(jx, jy, linestyle="none", marker="o", ms=7, mfc=JOINT, mec=SURF, mew=1.2, zorder=4)
    if jx:
        ax.annotate("joint", (jx[-1], jy[-1]), xytext=(6, 4), textcoords="offset points",
                     va="center", fontsize=7.5, color=INK)

    sx = [xpos(l, zero_x) for l, _ in single]
    sy = [v for _, v in single]
    ax.plot(sx, sy, "--", color=SINGLE, lw=1.6, zorder=3, dash_capstyle="round")
    ax.plot(sx, sy, linestyle="none", marker="s", ms=7, mfc=SINGLE, mec=SURF, mew=1.2, zorder=4)
    if sx:
        ax.annotate("single", (sx[-1], sy[-1]), xytext=(6, -8), textcoords="offset points",
                     va="center", fontsize=7.5, color=INK)

    if title:
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    axis_cosmetics(ax, zero_x, lam_ticks, xlim, show_xlabel, ylabel, "%.2f")


def panel_margin(ax, data, contrast, zero_x, lam_ticks, xlim, show_xlabel, ylabel):
    margin = data[contrast]["margin"]

    mx = [xpos(l, zero_x) for l, _ in margin]
    my = [v for _, v in margin]
    ax.plot(mx, my, ":", color=MARGIN, lw=1.6, zorder=3)
    ax.plot(mx, my, linestyle="none", marker="D", ms=6, mfc=MARGIN, mec=SURF, mew=1.2, zorder=4)
    if mx:
        ax.annotate("margin", (mx[-1], my[-1]), xytext=(6, 0), textcoords="offset points",
                     va="center", fontsize=7.5, color=INK)

    axis_cosmetics(ax, zero_x, lam_ticks, xlim, show_xlabel, ylabel, "%.3f")


def main() -> int:
    data = load(CSV_PATH)
    lam_ticks = [0.0, 0.0003, 0.001, 0.003, 0.01]
    zero_x = min(math.log10(l) for l in lam_ticks if l > 0) - GAP
    xlim = (zero_x - 0.35, max(math.log10(l) for l in lam_ticks if l > 0) + 0.35)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2), dpi=200, sharex="col")
    for col, contrast in enumerate(("T1", "FLAIR")):
        panel_psnr(axes[0, col], data, contrast, zero_x, lam_ticks, xlim, show_xlabel=False,
                   ylabel=("head PSNR (dB)" if col == 0 else ""), title=contrast)
        panel_margin(axes[1, col], data, contrast, zero_x, lam_ticks, xlim, show_xlabel=True,
                     ylabel=("margin, joint − single (dB)" if col == 0 else ""))

    handles = [
        Line2D([], [], color=JOINT, marker="o", ms=7, mfc=JOINT, mec=SURF, mew=1.2, lw=1.6, linestyle="-", label="joint"),
        Line2D([], [], color=SINGLE, marker="s", ms=7, mfc=SINGLE, mec=SURF, mew=1.2, lw=1.6, linestyle="--", label="single"),
        Line2D([], [], color=MARGIN, marker="D", ms=6, mfc=MARGIN, mec=SURF, mew=1.2, lw=1.6, linestyle=":", label="margin (joint − single)"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=False, ncol=3,
               handlelength=2.6, numpoints=1, fontsize=8)
    fig.text(0.01, 0.99, "MC-GS dose-response on the common-edge prior weight " + r"$\lambda_{edge}$",
              ha="left", va="top", fontsize=10, color=INK, weight="medium")
    fig.text(0.01, 0.005,
              "jiangyizhou T1 + FLAIR, nominal AF 16 house vd-Poisson mask, frozen phase-3 recipe, 4000 iterations, "
              "seed 42. Head-masked PSNR (|GT| > 0.05 P99.9). Top row: joint (5 lambda_edge values) vs single (3: "
              "0, 0.001, 0.003 -- drawn only where run, not interpolated). Bottom row: the margin at the 3 "
              "lambda_edge values where both arms exist, sharing the x-axis with the row above. "
              "Numbers: results_arc/ca_dose.md.",
              ha="left", va="bottom", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.subplots_adjust(hspace=0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"ca_dose_lambda_edge.{ext}", bbox_inches="tight", facecolor=SURF)
    print(f"wrote {OUT_DIR}/ca_dose_lambda_edge.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
