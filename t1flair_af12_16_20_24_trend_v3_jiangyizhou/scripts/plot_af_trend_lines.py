#!/usr/bin/env python3
"""plot_af_trend_lines.py -- head-only metric vs acceleration factor, one line per method; T1 left, FLAIR right.

Input : logs/rescore_af<AF>_<C>.json (scripts/rescore_topic.py: one scorer, bit-identical GT, four topics).
Output: figures/jyz_t1flair_af12_16_20_24_head_<psnr|ssim|hfen>_lines.{png,pdf}, mirrored to the served figs folder.
Methods shown = the eleven columns of the 2026-09-16 hi-res figures (ZF, CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU,
3D-DIP, 3D-INR, MC-INR, GS-single, MC-GS). CS-SENSE is the converged-TV run; "CS-SENSE (35 it)" and L1-ESPIRiT stay in the
json as data and are not drawn. AF axis = PE-plane lines / mask sampled lines (user 2026-09-16), one number per topic, both contrasts.
Encoding: hue = method family (seven categorical slots in vertical order, validated with the dataviz palette checker on white),
solid line + filled marker = multi-contrast joint, dashed line + open marker = single-contrast; ZF in gray as the floor.
Layout (user's choice 2026-09-16): one-line title on top, a 6 x 2 legend grouped by family under the panels, three-line footnote.
A direct-label variant (names at the line ends, no per-method legend) was tried the same day and not kept.
"""
import argparse, json, os, shutil
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

T = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou"
SERVED = "/home/congcongliu/figs/t1flair_v3_af_trend"
AFS = ("12", "16", "20", "24")
AFX = {"12": 11.71, "16": 15.19, "20": 18.56, "24": 24.00}          # PE-plane lines / mask sampled lines, one number per topic
FLAIR_DATA_LINES = {"12": 3504, "16": 2733, "20": 2257, "24": 1766}  # of the sampled lines, those inside the FLAIR scan support
CONTRASTS = ("T1", "FLAIR")

ap = argparse.ArgumentParser()
ap.add_argument("--dark", action="store_true", help="dark theme (user 2026-09-16 evening): the same seven hues at their dark-surface "
                "steps of the dataviz palette (validated on #1a1a19), light ink, heavier lines; files get the suffix _dark")
args = ap.parse_args()
if args.dark:
    BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"
    GRAY, SURF, INK, INK2, MUTED, GRID, BASE = "#a8a69e", "#1a1a19", "#ffffff", "#e6e5df", "#a8a69e", "#33332f", "#4a4a46"
    LW_J, LW_S, LW_Z, MS = 2.2, 1.7, 1.3, 6.0
else:
    BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"
    GRAY, SURF, INK, INK2, MUTED, GRID, BASE = "#898781", "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    LW_J, LW_S, LW_Z, MS = 1.6, 1.3, 1.1, 5.5
SFX = "_dark" if args.dark else ""
# json label, display name, family hue, marker, multi-contrast joint?   (list order = legend order, families side by side)
METHODS = [
    ("MC-GS",                   "MC-GS",       BLUE,    "o", True),
    ("GS-single",               "GS-single",   BLUE,    "o", False),
    ("Joint SENSE",             "Joint SENSE", ORANGE,  "s", True),
    ("CS-SENSE (converged TV)", "CS-SENSE",    ORANGE,  "s", False),
    ("MC-INR",                  "MC-INR",      GREEN,   "^", True),
    ("3D-INR",                  "3D-INR",      GREEN,   "^", False),
    ("ZS-SSDU",                 "ZS-SSDU",     AQUA,    "D", False),
    ("L1-SPIRiT",               "L1-SPIRiT",   YELLOW,  "v", False),
    ("CG-SENSE",                "CG-SENSE",    MAGENTA, "p", False),
    ("3D-DIP",                  "3D-DIP",      VIOLET,  "h", False),
    ("ZF",                      "ZF",          GRAY,    "x", False),
]
METRICS = [("head_psnr", "Head-only PSNR (dB)", "PSNR"),
           ("head_ssim", "Head-only SSIM", "SSIM"),
           ("head_nmse", "Head-only NMSE (lower is better)", "NMSE"),     # scripts/head_nmse.py, same head mask as the gauge
           ("head_hfen", "Head-only HFEN (lower is better)", "HFEN")]

V = {}
for af in AFS:
    for c in CONTRASTS:
        for r in json.load(open(f"{T}/logs/rescore_af{af}_{c}.json")):
            if "head_psnr" in r:
                V[(af, c, r["label"])] = r
missing = [(af, c, lab) for af in AFS for c in CONTRASTS for lab, *_ in METHODS if (af, c, lab) not in V]
assert not missing, missing
for af in AFS:                                                # head-only NMSE (user 2026-09-16 evening), see scripts/head_nmse.py
    for c in CONTRASTS:
        for r in json.load(open(f"{T}/logs/head_nmse_af{af}_{c}.json"))["rows"]:
            if (af, c, r["label"]) in V:
                V[(af, c, r["label"])]["head_nmse"] = r["head_nmse"]
assert all("head_nmse" in V[(af, c, lab)] for af in AFS for c in CONTRASTS for lab, *_ in METHODS)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.edgecolor": BASE, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "pdf.fonttype": 42, "ps.fonttype": 42})

x = np.array([AFX[af] for af in AFS])
os.makedirs(f"{T}/figures", exist_ok=True); os.makedirs(SERVED, exist_ok=True)
for m, ylabel, short in METRICS:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.75), sharey=True)
    for ax, c in zip(axes, CONTRASTS):
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for lab, name, col, mk, joint in METHODS:
            y = np.array([V[(af, c, lab)][m] for af in AFS])
            if joint:                                     # multi-contrast: solid line, filled marker, drawn under the dashed singles
                ax.plot(x, y, "-", color=col, lw=LW_J, solid_capstyle="round", zorder=2)
                ax.plot(x, y, mk, ms=MS, mfc=col, mec=SURF, mew=0.8, zorder=5)
            elif lab == "ZF":
                ax.plot(x, y, "--", color=col, lw=LW_Z, dashes=(4, 2.5), zorder=3)
                ax.plot(x, y, mk, ms=MS, mec=col, mew=1.2, zorder=4)
            else:                                         # single-contrast: dashed line on top (shows where it coincides), open marker
                ax.plot(x, y, "--", color=col, lw=LW_S, dashes=(4, 2.5), dash_capstyle="round", zorder=3)
                ax.plot(x, y, mk, ms=MS, mfc=SURF, mec=col, mew=1.2, zorder=4)
            if lab == "MC-GS":                            # the one direct label: the series the story is about
                ax.annotate("MC-GS", (x[-1], y[-1]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK2)
        ax.set_title(c, loc="left", fontsize=10, color=INK, fontweight="bold", pad=6)
        ax.set_xticks(x); ax.set_xticklabels([f"AF {af}\n{AFX[af]:.2f}×" for af in AFS])
        ax.set_xlim(10.6, 26.3); ax.set_xlabel("Acceleration factor", fontsize=8)
        ax.tick_params(length=2.5, width=0.6)
    axes[0].set_ylabel(ylabel, fontsize=8.5)
    vals = np.array([V[(af, c, lab)][m] for af in AFS for c in CONTRASTS for lab, *_ in METHODS])
    lo, hi = vals.min(), vals.max(); pad = 0.06 * (hi - lo)
    axes[0].set_ylim(lo - pad, hi + pad)

    handles = [Line2D([], [], color=col, lw=LW_J if joint else LW_S, ls="-" if joint else (0, (4, 2.5)), marker=mk, ms=MS,
                      mfc=(col if joint else SURF) if mk != "x" else "none", mec=SURF if joint else col, mew=0.8 if joint else 1.2)
               for lab, name, col, mk, joint in METHODS]
    fig.legend(handles, [name for _, name, *_ in METHODS], loc="lower center", ncol=6, frameon=False, fontsize=7.5,
               handlelength=3.0, columnspacing=1.4, handletextpad=0.6, bbox_to_anchor=(0.5, 0.09))
    fig.suptitle(f"jiangyizhou 3D T1 + FLAIR, retrospective undersampling: head-only {short} vs acceleration (one scorer, same GT)",
                 x=0.02, ha="left", fontsize=8.5, color=INK, y=0.985)
    fig.text(0.02, 0.012,
             "Solid line + filled marker = multi-contrast joint; dashed line + open marker = single-contrast; same hue = same method family. "
             "CS-SENSE = converged TV.\n"
             "One vd-Poisson mask per AF, shared by both contrasts. AF = PE-plane lines (45056) ÷ sampled lines (3847 / 2966 / 2427 / 1877).\n"
             "The FLAIR scan covers 34139 of the 45056 lines (elliptical shutter), so "
             + " / ".join(str(FLAIR_DATA_LINES[af]) for af in AFS) + " of the sampled lines carry FLAIR data.",
             fontsize=6.2, color=MUTED, va="bottom", linespacing=1.45)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.895, bottom=0.335, wspace=0.08)
    out = f"{T}/figures/jyz_t1flair_af12_16_20_24_head_{short.lower()}_lines{SFX}"
    fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".pdf")
    for ext in (".png", ".pdf"):
        shutil.copy2(out + ext, SERVED); shutil.copy2(out + ext, f"{SERVED}/lines_{short.lower()}{SFX}{ext}")
    plt.close(fig); print("wrote", out + ".png / .pdf")
