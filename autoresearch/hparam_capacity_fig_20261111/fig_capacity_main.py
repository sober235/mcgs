#!/usr/bin/env python
"""fig_capacity_main.py -- main-text version of the Gaussian-capacity sensitivity figure (2026-11-11).

Two panels cut from the five-panel Extended Data figure
(Results_20260908/T1FLAIR_AF16_hparam_v3_jiangyizhou/scripts/hp_fig_ablation_nature.py, 2026-09-17):

  a  head PSNR against the number of shared Gaussians (0.9M / 1.8M / 3.6M / 7.2M), relative to 3.6M
  b  training wall-clock time and peak GPU memory, indexed to 0.9M, with the Gaussian count as reference

Claim carried by the figure: too little capacity lowers both contrasts; doubling beyond 3.6M trades T1
for FLAIR without a consistent gain, at extra time and memory. HFEN, SSIM and the private-Gaussian
ablation stay in the Extended Data figure (user decision 2026-11-11: HFEN not considered for now).

Same data file, fonts, palette and mark sizes as the source figure. One run per setting (seed 42), so no
variability is drawn. Reads logs/hp_summary.csv only; writes PNG / PDF / SVG and the plotted values to
MCGS_ARTIFACT_DIR. Nothing under Results_20260908 is written. CPU only.
"""
import csv
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.transforms import ScaledTranslation

T = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_hparam_v3_jiangyizhou")
SRC = T / "logs/hp_summary.csv"
OUT = Path(os.environ["MCGS_ARTIFACT_DIR"])
STEM = OUT / "hp_capacity_main"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 6.5,
    "axes.titlesize": 7,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "axes.edgecolor": "#4D4D4D",
    "xtick.color": "#4D4D4D",
    "ytick.color": "#4D4D4D",
    "text.color": "#272727",
    "axes.labelcolor": "#272727",
})

C = {"T1": "#3775BA", "FLAIR": "#E28E2C"}
MARK = {"T1": "o", "FLAIR": "s"}
INK, MID, LIGHT = "#272727", "#767676", "#BDBDBD"
LW, MS, RING = 1.2, 4.2, 0.8

# ---------------------------------------------------------------- data
raw = SRC.read_bytes()
print("source", SRC, "sha256", hashlib.sha256(raw).hexdigest())
rows = {r["exp"]: r for r in csv.DictReader(raw.decode().splitlines())}
val = lambda e, k: float(rows[e][k])
CAP = ["hp_n09", "hp_n18", "hp_n36", "hp_n72"]
REF = "hp_n36"
assert all(rows[e]["status"] == "done" for e in CAP)
assert [int(val(e, "n_shared")) for e in CAP] == [900000, 1800000, 3600000, 7200000]
millions = [val(e, "n_shared") / 1e6 for e in CAP]
x = np.arange(len(CAP))                                    # each step doubles the count


def d_psnr(c):
    return np.array([val(e, f"{c}_head_psnr") - val(REF, f"{c}_head_psnr") for e in CAP])


minutes = np.array([val(e, "elapsed_h") * 60 for e in CAP])
vram = np.array([val(e, "vram_gb") for e in CAP])
count = np.array([val(e, "n_shared") for e in CAP])
for arr in (minutes, vram, count):
    assert np.all(np.isfinite(arr)) and not np.any(arr <= 0), "non-positive value on the log axis"

# the wording drawn in a is written only because the data say so
dT, dF = d_psnr("T1"), d_psnr("FLAIR")
assert dT[0] < -0.02 and dF[0] < -0.02, "0.9M is not below 3.6M on both contrasts"
assert dT[3] < 0 < dF[3], "7.2M is not a T1-down / FLAIR-up trade-off"


# ---------------------------------------------------------------- helpers
def panel_label(ax, s):
    off = ScaledTranslation(-26 / 72, 3 / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, s, transform=ax.transAxes + off, fontsize=8, fontweight="bold",
            ha="left", va="bottom", color="black")


def series(ax, xs, ys, c):
    ax.plot(xs, ys, color=C[c], lw=LW, solid_capstyle="round", zorder=3)
    ax.plot(xs, ys, ls="none", marker=MARK[c], ms=MS, mfc=C[c], mec="white", mew=RING, zorder=4)


def capacity_ticks(ax):
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m:g}" for m in millions])
    ax.get_xticklabels()[2].set_fontweight("bold")         # the setting used in this work
    ax.spines["bottom"].set_bounds(0, 3)


# ---------------------------------------------------------------- layout
fig = plt.figure(figsize=(7.2, 2.55))                     # 183 mm double column
gs = GridSpec(1, 2, figure=fig, left=0.075, right=0.985, top=0.80, bottom=0.155,
              wspace=0.24, width_ratios=[1.12, 1.0])
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

handles = [
    Line2D([], [], color=C["T1"], lw=LW, marker="o", ms=MS, mfc=C["T1"], mec="white", mew=RING, label="T1"),
    Line2D([], [], color=C["FLAIR"], lw=LW, marker="s", ms=MS, mfc=C["FLAIR"], mec="white", mew=RING,
           label="FLAIR"),
    Line2D([], [], ls="none", marker="o", ms=8.6, mfc="none", mec=MID, mew=0.9,
           label="Best setting for that contrast"),
    Line2D([], [], ls="none", marker="o", ms=MS, mfc="white", mec=INK, mew=0.9,
           label="Setting used in this work (3.6M)"),
]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.068, 0.995), ncol=4, handlelength=1.8,
           columnspacing=1.8, handletextpad=0.5, borderaxespad=0.2)

# a -- head PSNR against the shared count
ax = ax_a
ax.axhline(0, color=LIGHT, lw=0.7, zorder=1)
for c, y in (("T1", dT), ("FLAIR", dF)):
    series(ax, x, y, c)
    b = int(np.argmax(y))
    ax.plot(x[b], y[b], ls="none", marker="o", ms=8.6, mfc="none", mec=C[c], mew=0.9, zorder=6)
ax.plot(2, 0, ls="none", marker="o", ms=MS, mfc="white", mec=INK, mew=0.9, zorder=5)
capacity_ticks(ax)
ax.set_xlim(-0.3, 4.25)
ax.set_ylim(-0.88, 0.14)
ax.set_yticks([-0.8, -0.6, -0.4, -0.2, 0.0])
ax.set_ylabel("Δ PSNR (dB)")
ax.set_xlabel("Shared Gaussians (millions)")
ax.set_title("Head PSNR", loc="left", pad=3)
ax.text(0.16, dF[0], "Under-capacity\n" + f"FLAIR {dF[0]:+.2f} dB, T1 {dT[0]:+.2f} dB".replace("-", "−"),
        fontsize=6, color=MID, ha="left", va="center", linespacing=1.3)
ax.text(3.22, dF[3] + 0.012, f"FLAIR {dF[3]:+.2f} dB".replace("-", "−"), fontsize=6, ha="left", va="center")
ax.text(3.22, dT[3] - 0.012, f"T1 {dT[3]:+.2f} dB".replace("-", "−"), fontsize=6, ha="left", va="center")
panel_label(ax, "a")

# b -- compute cost, every series indexed to 0.9M on one log axis
ax = ax_b
ax.set_yscale("log", base=2)
ax.plot(x, count / count[0], color=LIGHT, lw=LW, zorder=2)
ax.plot(x, vram / vram[0], color=MID, lw=LW, zorder=3)
ax.plot(x, vram / vram[0], ls="none", marker="D", ms=MS - 0.6, mfc=MID, mec="white", mew=RING, zorder=4)
ax.plot(x, minutes / minutes[0], color=INK, lw=LW, zorder=3)
ax.plot(x, minutes / minutes[0], ls="none", marker="^", ms=MS + 0.4, mfc=INK, mec="white", mew=RING, zorder=4)
ax.text(3.2, count[-1] / count[0], f"×{count[-1] / count[0]:.0f} Gaussians", fontsize=6, ha="left",
        va="center", color=MID)
ax.text(3.2, minutes[-1] / minutes[0], f"×{minutes[-1] / minutes[0]:.2f} time\n"
        f"({minutes[0]:.0f} → {minutes[-1]:.0f} min)", fontsize=6, ha="left", va="bottom", linespacing=1.25)
ax.text(3.2, vram[-1] / vram[0], f"×{vram[-1] / vram[0]:.2f} peak GPU memory\n"
        f"({vram[0]:.1f} → {vram[-1]:.1f} GB)", fontsize=6, ha="left", va="top", linespacing=1.25, color=MID)
ax.set_ylim(0.58, 11)
ax.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"×{v:g}"))
ax.yaxis.set_minor_locator(FixedLocator([]))
capacity_ticks(ax)
ax.set_xlim(-0.3, 5.9)
ax.set_ylabel("Relative to 0.9M")
ax.set_xlabel("Shared Gaussians (millions)")
ax.set_title("Compute cost", loc="left", pad=3)
panel_label(ax, "b")

# ---------------------------------------------------------------- QA gate (same audit as the source figure)
sys.path.insert(0, str(T / "scripts"))
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None
    print("alignment audit: SKIPPED (audit_panel_alignment.py not importable)")
if require_matplotlib_panel_alignment is not None:
    require_matplotlib_panel_alignment(
        fig,
        json_out=f"{STEM}.alignment.json",
        overlay_svg=f"{STEM}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    print("alignment audit: PASS (strict, 1.5 pt)")

# ---------------------------------------------------------------- export
fig.savefig(f"{STEM}.svg", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{STEM}.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{STEM}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

with open(f"{STEM}_source_data.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["panel", "experiment", "contrast", "x", "x_unit", "value", "value_unit"])
    for c, y in (("T1", dT), ("FLAIR", dF)):
        for e, m, a in zip(CAP, millions, y):
            w.writerow(["a", e, c, m, "million shared Gaussians", f"{a:.4f}", "dB vs 3.6M"])
    for e, m, t, v, n in zip(CAP, millions, minutes, vram, count):
        w.writerow(["b", e, "joint", m, "million shared Gaussians", f"{t:.2f}", "wall-clock minutes"])
        w.writerow(["b", e, "joint", m, "million shared Gaussians", f"{v:.2f}", "peak GPU memory GB"])
        w.writerow(["b", e, "joint", m, "million shared Gaussians", f"{n:.0f}", "shared Gaussians"])

print("fonts", sorted({t.get_fontname() for t in fig.findobj(matplotlib.text.Text)}))
print("wrote", STEM, "(png 600 dpi, pdf, svg, source data)")
