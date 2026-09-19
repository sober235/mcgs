#!/usr/bin/env python3
"""fig_lesion_metrics.py -- lesion contrast and CNR fidelity across the eleven arms, as line plots.

Two panels, the topic's column order on the x axis (ZF ... MC-GS), one line per scan:

  left    lesion contrast (mean_wm - mean_core) / mean_wm, as a percentage error against the truth's contrast
  right   lesion CNR (mean_wm - mean_core) / std_wm, likewise

Zero is the truth, drawn as a solid line: a point below zero is an arm that has washed the lesion out (or, on the
CNR panel, buried it in its own noise). Both quantities are computed in white-matter units over the whole 3-D
lesion, in fixed regions drawn on the truth, with no segmentation model anywhere -- see lesion_preservation.py for
why that matters and for what the core region is (a hand-guided threshold region, not a radiologist's outline).

Reads logs/lesion_preservation.json. Writes figures_seg/ce_retro_af16_lesion_metrics.png (+ .pdf with --pdf),
copied to ~/figs/ce_retro_seg/.

  env PYTHONNOUSERSITE=1 PYTHONPATH= <GSTMI python> scripts/fig_lesion_metrics.py
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from arms import ORDER, LABEL

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
FIGDIR = R / "figures_seg"
FIGDIR_SERVED = Path("/home/congcongliu/figs/ce_retro_seg")

SCANS = ("pre", "post")
SCAN_STYLE = {"pre": dict(color="#1f77b4", marker="o", ls="-", label="pre-contrast"),
              "post": dict(color="#d62728", marker="s", ls="--", label="post-contrast")}
PANELS = [("contrast_err", "lesion contrast", "(mean$_{WM}$ - mean$_{core}$) / mean$_{WM}$"),
          ("cnr_err", "lesion CNR", "(mean$_{WM}$ - mean$_{core}$) / std$_{WM}$")]
MCGS = "gs_joint"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    doc = json.load(open(R / "logs/lesion_preservation.json"))
    rows = {(r["scan"], r["arm"]): r for r in doc["rows"]}
    xs = np.arange(len(ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    for ax, (field, title, formula) in zip(axes, PANELS):
        ax.axhline(0, color="0.35", lw=1.4, zorder=1)
        ax.text(-0.45, 0, "truth ", color="0.35", fontsize=9, va="bottom", ha="left")
        j = ORDER.index(MCGS)
        series = {s: [100 * rows[(s, k)][field] for k in ORDER] for s in SCANS}
        # the two scans' stars sit almost on top of each other: the higher one is labelled above, the other below
        upper = max(SCANS, key=lambda s: series[s][j])
        for scan in SCANS:
            ys = series[scan]
            ax.plot(xs, ys, lw=1.6, ms=5.5, zorder=3, **SCAN_STYLE[scan])
            ax.plot([xs[j]], [ys[j]], marker="*", ms=16, color=SCAN_STYLE[scan]["color"], zorder=4,
                    mec="black", mew=0.6)
            best = int(np.argmax([-abs(v) for v in ys]))
            off = (9, 6) if scan == upper else (9, -14)
            ax.annotate(f"{ys[j]:+.2f}%", (xs[j], ys[j]), textcoords="offset points", xytext=off,
                        fontsize=8.5, color=SCAN_STYLE[scan]["color"], ha="left",
                        fontweight="bold" if best == j else "normal")
        ax.set_xlim(-0.6, len(ORDER) - 0.1)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABEL[k] for k in ORDER], rotation=40, ha="right", fontsize=9.5)
        ax.set_ylabel("error against the truth (%)", fontsize=10)
        ax.set_title(f"{title}   {formula}", fontsize=11)
        ax.grid(axis="y", color="0.9", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=9.5, frameon=False, loc="lower left")
    fig.suptitle("CE retro AF 16.00 -- lesion conspicuity kept by each arm (3-D, fixed regions on the truth, "
                 "no segmentation model);  star = MC-GS", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    FIGDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR_SERVED.mkdir(parents=True, exist_ok=True)
    png = FIGDIR / "ce_retro_af16_lesion_metrics.png"
    fig.savefig(png, dpi=a.dpi)
    if a.pdf:
        fig.savefig(FIGDIR / "ce_retro_af16_lesion_metrics.pdf")
    plt.close(fig)
    shutil.copy2(png, FIGDIR_SERVED / png.name)
    print(f"wrote {png}\n      http://localhost:8765/ce_retro_seg/{png.name}")
    for field, title, _ in PANELS:
        for scan in SCANS:
            vals = {k: rows[(scan, k)][field] for k in ORDER}
            best = min(vals, key=lambda k: abs(vals[k]))
            print(f"[{scan}] {title}: best {LABEL[best]} {100 * vals[best]:+.2f}%, "
                  f"MC-GS {100 * vals[MCGS]:+.2f}%, GS-single {100 * vals['gs_single']:+.2f}%")


if __name__ == "__main__":
    main()
