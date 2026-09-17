#!/usr/bin/env python3
"""plot_af12_methods.py -- head-only metric of every arm at retrospective AF 12 as a line chart over the methods, T1w left,
T2* right, in the chrome of the adult T1 + FLAIR trend figure (T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou/scripts/
plot_af_trend_lines.py; user 2026-09-16: same format, and a line chart). This line has one acceleration factor, so the x-axis
carries the eleven arms in the column order of the figures of record (ZF ... MC-GS) and one neutral line per panel joins their
values; the markers keep the trend figure's encoding (hue = method family, filled = multi-contrast joint, open = single-contrast,
ZF gray) and the 6 x 2 legend grouped by family, the title line and the footnote are kept.
Input : logs/rescore.json (primary region, head inside partitions 12..92) and logs/rescore_zf_primary.json (ZF, same gauge,
scripts/score_zf_primary.py). MC-GS = gs_joint_p3f (stage-3 final, 9.6M points + union guide); GS-single = the best registered
single arm per contrast (T1w gs_single_p3f, T2s gs_single_ar), as in the figures of record. L1-ESPIRiT (not counted since
2026-09-15) and the other GS rows stay in the json and are not drawn.
Output: figures/peds_af12_T1w_T2s_head_<psnr|ssim|hfen>_methods.{png,pdf}, mirrored to /home/congcongliu/figs/peds_tsc_af12/
(also as methods_<psnr|ssim|hfen>.*). Earlier the same day: a dot chart (methods on y) and a one-column strip at x = AF 12, both replaced.
"""
import json, os, shutil
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

T = "/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro"
SERVED = "/home/congcongliu/figs/peds_tsc_af12"
CONTRASTS = ("T1w", "T2s")
PANEL = {"T1w": "T1w", "T2s": "T2*"}
AF = 12.00                                            # PE-plane points / sampled points (prep_mask.json AF_vs_full_grid)
XORDER = ["ZF", "CG-SENSE", "CS-SENSE", "Joint SENSE", "L1-SPIRiT", "ZS-SSDU", "3D-DIP", "3D-INR", "MC-INR", "GS-single", "MC-GS"]

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"
GRAY, SURF, INK, INK2, MUTED, GRID, BASE = "#898781", "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# json key per contrast, display name, family hue, marker, multi-contrast joint?   (list order = legend order = dodge order)
METHODS = [
    ({"T1w": "gs_joint_p3f", "T2s": "gs_joint_p3f"},   "MC-GS",       BLUE,    "o", True),
    ({"T1w": "gs_single_p3f", "T2s": "gs_single_ar"},  "GS-single",   BLUE,    "o", False),
    ({"T1w": "jcssense", "T2s": "jcssense"},           "Joint SENSE", ORANGE,  "s", True),
    ({"T1w": "cssense_o200", "T2s": "cssense_o200"},   "CS-SENSE",    ORANGE,  "s", False),
    ({"T1w": "mcinr", "T2s": "mcinr"},                 "MC-INR",      GREEN,   "^", True),
    ({"T1w": "inr", "T2s": "inr"},                     "3D-INR",      GREEN,   "^", False),
    ({"T1w": "zsssdu", "T2s": "zsssdu"},               "ZS-SSDU",     AQUA,    "D", False),
    ({"T1w": "spirit", "T2s": "spirit"},               "L1-SPIRiT",   YELLOW,  "v", False),
    ({"T1w": "cgsense", "T2s": "cgsense"},             "CG-SENSE",    MAGENTA, "p", False),
    ({"T1w": "dip", "T2s": "dip"},                     "3D-DIP",      VIOLET,  "h", False),
    ({"T1w": "zf", "T2s": "zf"},                       "ZF",          GRAY,    "x", False),
]
METRICS = [("head_psnr", "Head-only PSNR (dB)", "PSNR"),
           ("head_ssim", "Head-only SSIM", "SSIM"),
           ("head_hfen", "Head-only HFEN (lower is better)", "HFEN")]

prim = json.load(open(f"{T}/logs/rescore.json")); zf = json.load(open(f"{T}/logs/rescore_zf_primary.json"))
assert prim["region"] == zf["region"] == "head inside partitions 12..92", (prim["region"], zf["region"])
rows = dict(prim["rows"]); rows.update(zf["rows"])
V = {(c, name): rows[f"{key[c]}|{c}"] for key, name, *_ in METHODS for c in CONTRASTS}
SPEC = {name: (col, mk, joint) for _, name, col, mk, joint in METHODS}
assert sorted(XORDER) == sorted(SPEC), set(XORDER) ^ set(SPEC)
mask = json.load(open(f"{T}/logs/prep_mask.json"))
assert abs(mask["AF_vs_full_grid"] - AF) < 0.01, mask["AF_vs_full_grid"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.edgecolor": BASE, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "pdf.fonttype": 42, "ps.fonttype": 42})

x = np.arange(len(XORDER))
os.makedirs(f"{T}/figures", exist_ok=True); os.makedirs(SERVED, exist_ok=True)
for m, ylabel, short in METRICS:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.85), sharey=True)
    for ax, c in zip(axes, CONTRASTS):
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        y = np.array([V[(c, name)][m] for name in XORDER])
        ax.plot(x, y, "-", color=MUTED, lw=1.1, solid_capstyle="round", solid_joinstyle="round", zorder=2)   # one neutral line per panel
        for xi, yi, name in zip(x, y, XORDER):
            col, mk, joint = SPEC[name]
            if joint:
                ax.plot(xi, yi, mk, ms=5.5, mfc=col, mec=SURF, mew=0.8, zorder=5)
            elif name == "ZF":
                ax.plot(xi, yi, mk, ms=5.5, mec=col, mew=1.2, zorder=4)
            else:
                ax.plot(xi, yi, mk, ms=5.5, mfc=SURF, mec=col, mew=1.2, zorder=4)
        ax.set_title(PANEL[c], loc="left", fontsize=10, color=INK, fontweight="bold", pad=6)
        ax.set_xticks(x); ax.set_xticklabels(XORDER, rotation=40, ha="right", rotation_mode="anchor", fontsize=7)
        ax.set_xlim(-0.6, len(XORDER) - 0.4); ax.set_xlabel(f"Method (retrospective AF {AF:.0f}, {AF:.2f}×)", fontsize=8)
        ax.tick_params(length=2.5, width=0.6)
    axes[0].set_ylabel(ylabel, fontsize=8.5)
    vals = np.array([V[(c, name)][m] for c in CONTRASTS for name in XORDER])
    lo, hi = vals.min(), vals.max(); pad = 0.06 * (hi - lo)
    axes[0].set_ylim(lo - pad, hi + pad)
    if m == "head_psnr":
        from matplotlib.ticker import MultipleLocator
        axes[0].yaxis.set_major_locator(MultipleLocator(2))

    handles = [Line2D([], [], ls="none", marker=mk, ms=5.5, mfc=(col if joint else SURF) if mk != "x" else "none",
                      mec=SURF if joint else col, mew=0.8 if joint else 1.2)
               for _, name, col, mk, joint in METHODS]
    fig.legend(handles, [name for _, name, *_ in METHODS], loc="lower center", ncol=6, frameon=False, fontsize=7.5,
               handlelength=1.6, columnspacing=1.6, handletextpad=0.5, bbox_to_anchor=(0.5, 0.15))
    fig.suptitle(f"16-year-old TSC exam, 3D MTP T1w + T2*, retrospective AF 12: head-only {short} of every arm (one scorer)",
                 x=0.02, ha="left", fontsize=8.5, color=INK, y=0.985)
    fig.text(0.02, 0.012,
             "Filled marker = multi-contrast joint; open marker = single-contrast; same hue = same method family. "
             "T1w = FA 16°, TE 3.25 ms; T2* = FA 4°, TE 27.41 ms.\n"
             "MC-GS = stage-3 final (9.6M points, union guide); GS-single = the best registered single arm per contrast; "
             "CS-SENSE = converged TV; λ of CS-SENSE, Joint SENSE, L1-SPIRiT swept here.\n"
             f"One vd-Poisson mask for both contrasts. AF = PE-plane points ({mask['plane_points']}) ÷ sampled points "
             f"({mask['budget']}) = {mask['AF_vs_full_grid']:.2f}, i.e. {mask['AF_vs_support']:.2f}× of the {mask['n_support']} acquired positions.\n"
             f"The partial-Fourier block and the elliptical corners ({mask['n_zero']} points) are measured zeros for every arm; "
             "one scorer, head inside the vendor's central 80 partitions.\n"
             "Reference = the GRAPPA-completed prospective 2× k-space, not a fully sampled truth.",
             fontsize=6.0, color=MUTED, va="bottom", linespacing=1.45)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.915, bottom=0.40, wspace=0.08)
    out = f"{T}/figures/peds_af12_T1w_T2s_head_{short.lower()}_methods"
    fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".pdf")
    for ext in (".png", ".pdf"):
        shutil.copy2(out + ext, SERVED); shutil.copy2(out + ext, f"{SERVED}/methods_{short.lower()}{ext}")
    plt.close(fig); print("wrote", out + ".png / .pdf")
