"""Interim figure: MC-GS minus GS-single (head PSNR, head SSIM) for every (scan, AF, contrast) of the test set.

Reads table.json (written by collect_stats.py, one scoring code for every volume) from this folder and writes
interim_gs_pair.png / .pdf into MCGS_ARTIFACT_DIR (or ./figures). No server data are read; no clock is printed.
Used while the comparator rows are still running; the final figure is collect_stats.py's testset_summary.
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ALIGNED = ("vol1_huangyuwei", "vol2_jiangdian", "vol4_tangbing")
SCANS = ALIGNED + ("vol5_sunxinyi", "vol6_jiangdian_790", "vol7_tangbin_790")
AFS = (12, 16, 20, 24)
BLUE, RED = "#3C5488", "#E64B35"


def main():
    cells = json.load(open(HERE / "table.json"))["cells"].values()
    diff = {}
    for c in cells:
        m = c["methods"]
        g, s = m["MC-GS"], m["GS-single"]
        diff[(c["scan"], c["af"], c["contrast"])] = (g["head_psnr"] - s["head_psnr"],
                                                       1e3 * (g["head_ssim"] - s["head_ssim"]))
    assert len(diff) == 48, len(diff)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 110 / 25.4), sharex=True)
    ylab = ("MC-GS minus GS-single\nhead PSNR (dB)", "MC-GS minus GS-single\nhead SSIM (×10$^{-3}$)")
    for i in range(2):
        for j, con in enumerate(("T1", "FLAIR")):
            ax = axes[i, j]
            ax.axhline(0, color="k", lw=0.6)
            for k, sc in enumerate(SCANS):
                al = sc in ALIGNED
                x = [a + (k - 2.5) * 0.35 for a in AFS]
                y = [diff[(sc, a, con)][i] for a in AFS]
                ax.plot(x, y, ls="-", lw=0.4, color=BLUE if al else RED, alpha=0.5)
                ax.plot(x, y, ls="none", marker="o" if al else "^", ms=3.2,
                        mfc=BLUE if al else "white", mec=BLUE if al else RED, mew=0.6)
                ax.text(x[-1] + 0.25, y[-1], sc.split("_")[0], fontsize=5, va="center",
                        color=BLUE if al else RED)
            ax.set_xticks(AFS, [f"AF{a}" for a in AFS])
            ax.set_xlim(10, 26.5)
            if j == 0:
                ax.set_ylabel(ylab[i])
            ax.text(0.015, 0.97, con, transform=ax.transAxes, ha="left", va="top", fontsize=6.5)
            ax.text(-0.17 if j == 0 else -0.08, 1.04, "abcd"[2 * i + j], transform=ax.transAxes, fontsize=8,
                    fontweight="bold", ha="left", va="bottom")
    h = [plt.Line2D([], [], ls="none", marker="o", ms=3.2, mfc=BLUE, mec=BLUE, label="aligned scans (vol1, vol2, vol4)"),
         plt.Line2D([], [], ls="none", marker="^", ms=3.2, mfc="white", mec=RED, label="offset scans (vol5, vol6, vol7)")]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.54, 0.995), fontsize=6, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(os.environ.get("MCGS_ARTIFACT_DIR", HERE / "figures"))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "interim_gs_pair.png", dpi=600)
    fig.savefig(out / "interim_gs_pair.pdf", metadata={"CreationDate": None})
    print("INTERIM_OK", len(diff))


if __name__ == "__main__":
    main()
