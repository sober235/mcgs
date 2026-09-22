#!/usr/bin/env python3
"""ca_fig_dose.py -- Stage 2 dose-response figure for the coupling-attribution study
(spec docs/superpowers/specs/2026-09-20-coupling-attribution-design.md, section 3, Stage 2).

Two panels, T1 and FLAIR. x = lambda_edge, log-scaled, with 0 placed at a fixed offset left of the
smallest run value (0.0003) since a log axis cannot represent zero -- a dotted separator marks that
gap as an axis break, not a to-scale distance, and the tick is still labelled "0" so the point is
never silently dropped. y (left) = head PSNR (dB) for the joint arm (5 points: 0, 0.0003, 0.001,
0.003, 0.01) and the single arm (3 points: 0, 0.001, 0.003 -- drawn only where a run exists, not
interpolated across the two lambda_edge values the single arm never ran). y (right, twin axis) =
the margin joint - single head PSNR, at the 3 lambda_edge values where both arms exist. A vertical
grey line marks the published lambda_edge 0.001.

The margin's twin axis is a deliberate departure from this skill's default "no dual-axis" rule: the
margin is not an independent measure but a difference of the two curves already on the primary axis,
and the dose-response question this figure exists to answer (spec section 3) is exactly whether that
difference moves with lambda_edge -- reading it off a third, separately-scaled panel would hide the
lambda_edge alignment the reader needs to check. Colour is otherwise by the book: two categorical
slots for the two identity series (joint, single) validated with the dataviz skill's checker --
node scripts/validate_palette.js "#2a78d6,#eb6834" --mode light -> ALL CHECKS PASS (worst adjacent
CVD dE 24.7, normal-vision dE 33.6, both >> the 8 / 15 targets) -- and a third, deliberately
non-categorical neutral grey (#6b6a63, WCAG contrast 5.43:1 on the #ffffff surface) for the margin,
since it is a derived quantity rather than a third identity (dataviz color-formula.md's documented
scope: the six-check validator covers categorical identity palettes, not a lone status/derived
colour, which is instead checked for WCAG contrast). Palette and marks otherwise follow
scripts/hp_fig_curves.py (house style for this topic).

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


def shared_span(primary_vals, margin_vals, pad=1.4, floor=0.02):
    """Equal dB-per-pixel span for the primary and twin axes (both are dB: absolute head PSNR and a
    difference of two head-PSNR curves), so a reader can compare slopes across the two axes without
    the arbitrary rescaling a plain twinx() would introduce -- see the module docstring."""
    span = max(max(primary_vals) - min(primary_vals), max(margin_vals) - min(margin_vals), floor) * pad
    p_c = (max(primary_vals) + min(primary_vals)) / 2
    m_c = (max(margin_vals) + min(margin_vals)) / 2
    return span, p_c, m_c


def panel(ax, ax2, data, contrast, zero_x, lam_ticks, xlim):
    joint = data[contrast]["joint"]
    single = data[contrast]["single"]
    margin = data[contrast]["margin"]

    ax.axvline(xpos(PUBLISHED, zero_x), color=BASE, lw=1.0, zorder=1)
    ax.text(xpos(PUBLISHED, zero_x), 1.0, " published", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=7, color=INK2)
    break_x = (zero_x + xpos(min(l for l in lam_ticks if l > 0), zero_x)) / 2
    ax.axvline(break_x, color=GRID, lw=1.0, linestyle=":", zorder=1)

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

    mx = [xpos(l, zero_x) for l, _ in margin]
    my = [v for _, v in margin]
    ax2.plot(mx, my, ":", color=MARGIN, lw=1.6, zorder=2)
    ax2.plot(mx, my, linestyle="none", marker="D", ms=6, mfc=MARGIN, mec=SURF, mew=1.2, zorder=4)
    if mx:
        ax2.annotate("margin", (mx[-1], my[-1]), xytext=(6, 0), textcoords="offset points",
                      va="center", fontsize=7.5, color=INK)

    span, p_c, m_c = shared_span(jy + sy, my)
    ax.set_ylim(p_c - span / 2, p_c + span / 2)
    ax2.set_ylim(m_c - span / 2, m_c + span / 2)

    ax.set_xlim(*xlim)
    ax.set_xticks([xpos(l, zero_x) for l in lam_ticks])
    ax.set_xticklabels([("0" if l == 0.0 else f"{l:g}") for l in lam_ticks])
    ax.minorticks_off()
    ax.set_title(contrast, fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel(r"$\lambda_{edge}$ (log scale; 0 offset left of 0.0003, axis break)")
    ax.set_ylabel("head PSNR (dB)")
    ax2.set_ylabel("margin, joint − single (dB)", color=INK2)
    ax.grid(True, axis="y", color=GRID, lw=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)   # ax2 (twinx) owns the right edge instead
    ax2.spines["top"].set_visible(False)
    ax.tick_params(length=2.5, width=0.8)
    ax2.tick_params(length=2.5, width=0.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))


def main() -> int:
    data = load(CSV_PATH)
    lam_ticks = [0.0, 0.0003, 0.001, 0.003, 0.01]
    zero_x = min(math.log10(l) for l in lam_ticks if l > 0) - GAP
    xlim = (zero_x - 0.35, max(math.log10(l) for l in lam_ticks if l > 0) + 0.35)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.3), dpi=200)
    twins = [ax.twinx() for ax in axes]
    for ax, ax2, contrast in zip(axes, twins, ("T1", "FLAIR")):
        panel(ax, ax2, data, contrast, zero_x, lam_ticks, xlim)

    handles = [
        Line2D([], [], color=JOINT, marker="o", ms=7, mfc=JOINT, mec=SURF, mew=1.2, lw=1.6, linestyle="-", label="joint"),
        Line2D([], [], color=SINGLE, marker="s", ms=7, mfc=SINGLE, mec=SURF, mew=1.2, lw=1.6, linestyle="--", label="single"),
        Line2D([], [], color=MARGIN, marker="D", ms=6, mfc=MARGIN, mec=SURF, mew=1.2, lw=1.6, linestyle=":", label="margin (joint − single, right axis)"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.985), frameon=False, ncol=3,
               handlelength=2.6, numpoints=1, fontsize=8)
    fig.text(0.01, 0.985, "MC-GS dose-response on the common-edge prior weight " + r"$\lambda_{edge}$",
              ha="left", va="top", fontsize=10, color=INK, weight="medium")
    fig.text(0.01, 0.005,
              "jiangyizhou T1 + FLAIR, nominal AF 16 house vd-Poisson mask, frozen phase-3 recipe, 4000 iterations, "
              "seed 42. Head-masked PSNR (|GT| > 0.05 P99.9). Joint: 5 lambda_edge values; single: 3 (0, 0.001, "
              "0.003) -- drawn only where run, not interpolated. Margin: the 3 lambda_edge values where both arms "
              "exist. Numbers: results_arc/ca_dose.md.",
              ha="left", va="bottom", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"ca_dose_lambda_edge.{ext}", bbox_inches="tight", facecolor=SURF)
    print(f"wrote {OUT_DIR}/ca_dose_lambda_edge.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
