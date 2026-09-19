#!/usr/bin/env python3
"""fig_lesion_profile.py -- the lesion boundary seen as a profile, not as a table.

Three panels:
  left, middle   mean intensity in white-matter units against the signed distance to the lesion boundary
                 (negative inside the lesion, positive in the surrounding tissue), pre and post. The truth is the
                 thick black line. One curve reads out all three things the table lists separately: the height of
                 the step is the lesion contrast, how fast it climbs is the boundary width, and how far a curve
                 sits from the truth is that arm's error. The shaded corridor is the truth's own +/- 1 sigma within
                 each distance bin -- tissue heterogeneity, not noise, so it is context rather than a tolerance.
  right          contrast error against CNR error, one marker per arm and scan. The origin is the truth: the
                 closer to it, the better the lesion is preserved. This is the same pair of numbers the line plot
                 in fig_lesion_metrics.py shows, rearranged so "who is nearest the truth" is a distance on the page.

Everything is computed in the fixed regions of lesion_preservation.py, over the whole 3-D lesion, with no
segmentation model. Reads logs/lesion_preservation.json for the scatter and the volumes for the profiles.

  env PYTHONNOUSERSITE=1 PYTHONPATH= <GSTMI python> scripts/fig_lesion_profile.py [--arms ...]

Writes figures_seg/ce_retro_af16_lesion_profile.png, copied to ~/figs/ce_retro_seg/.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
PROS = Path(os.environ.get("CE_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast"))
ROI_NPZ = PROS / "lesion_candidates/enhance_rois_cssense.npz"
FIGDIR = R / "figures_seg"
FIGDIR_SERVED = Path("/home/congcongliu/figs/ce_retro_seg")

SCANS = ("pre", "post")
MM = 6                      # how far either side of the boundary the profile runs
MCGS = "gs_joint"
PALETTE = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#ff7f0e", "#d62728"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="cssense_o200,jcssense,zsssdu,inr,mcinr,gs_single,gs_joint",
                    help="arm keys to draw, in order (default: the seven of the axial figure)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    from arms import LABEL, arm_files
    keys = [k.strip() for k in a.arms.split(",") if k.strip()]

    z = np.load(ROI_NPZ, allow_pickle=True)
    core, wm = z["core"], z["roi2"]
    sd = ndi.distance_transform_edt(~core) - ndi.distance_transform_edt(core)
    summary = json.load(open(R / "logs/summary.json"))
    doc = json.load(open(R / "logs/lesion_preservation.json"))
    rows = {(r["scan"], r["arm"]): r for r in doc["rows"]}

    # no bin at 0: the signed distance is <= -1 strictly inside the core and >= +1 strictly outside, so a zero bin
    # would always be empty and would break every curve in the middle
    xs = np.array([x for x in range(-MM, MM + 1) if x != 0])
    prof = {}
    for scan in SCANS:
        gt = np.abs(np.load(R / f"recon/gt_{scan}_complex.npy")).astype(np.float64)
        head = gt > 0.05 * float(np.quantile(gt, 0.999))
        sel = (np.abs(sd) <= MM) & head
        d = np.rint(sd[sel]).astype(int)
        bins = {x: (d == x) for x in xs}
        g_wm = gt / float(np.median(gt[wm]))
        gv = g_wm[sel]
        prof[(scan, "truth")] = np.array([gv[bins[x]].mean() if bins[x].any() else np.nan for x in xs])
        for key, _, path in arm_files(scan, summary):
            if key not in keys:
                continue
            v = np.abs(np.load(path)).astype(np.float64)
            vv = (v / float(np.median(v[wm])))[sel]
            prof[(scan, key)] = np.array([vv[bins[x]].mean() if bins[x].any() else np.nan for x in xs])

    fig = plt.figure(figsize=(15.5, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05], height_ratios=[2.1, 1.0], hspace=0.08, wspace=0.26)
    for j, scan in enumerate(SCANS):
        ax = fig.add_subplot(gs[0, j])
        axr = fig.add_subplot(gs[1, j], sharex=ax)
        t = prof[(scan, "truth")]
        for aa in (ax, axr):
            aa.axvspan(-MM, 0, color="0.94", zorder=0)
            aa.grid(color="0.92", lw=0.8)
            aa.set_axisbelow(True)
            for sp in ("top", "right"):
                aa.spines[sp].set_visible(False)
        for i, k in enumerate(keys):
            col = PALETTE[i % len(PALETTE)]
            lw = 2.4 if k == MCGS else 1.3
            ax.plot(xs, prof[(scan, k)], color=col, lw=lw, zorder=4 if k == MCGS else 2, label=LABEL[k])
            axr.plot(xs, prof[(scan, k)] - t, color=col, lw=lw, zorder=4 if k == MCGS else 2)
        ax.plot(xs, t, color="black", lw=2.8, zorder=5, label="truth")
        axr.axhline(0, color="black", lw=1.6, zorder=5)
        ax.set_xlim(-MM, MM)
        ax.tick_params(labelbottom=False)
        ax.set_ylabel("intensity (WM units)", fontsize=10)
        ax.set_title(f"{scan}-contrast", fontsize=11)
        ax.text(-MM + 0.3, ax.get_ylim()[1], " lesion", va="top", ha="left", fontsize=9, color="0.45")
        axr.set_xlabel("signed distance to the lesion boundary (mm)", fontsize=10)
        axr.set_ylabel("arm $-$ truth", fontsize=9.5)
        if j == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)

    ax = fig.add_subplot(gs[:, 2])
    ax.axhline(0, color="0.5", lw=1.0, zorder=1)
    ax.axvline(0, color="0.5", lw=1.0, zorder=1)
    for i, k in enumerate(keys):
        for scan, mk in zip(SCANS, ("o", "s")):
            r = rows[(scan, k)]
            xy = (100 * r["contrast_err"], 100 * r["cnr_err"])
            ax.scatter(*xy, marker="*" if k == MCGS else mk, s=300 if k == MCGS else 60,
                       color=PALETTE[i % len(PALETTE)], zorder=4 if k == MCGS else 3,
                       edgecolor="black", linewidth=0.5)
    ax.scatter([0], [0], marker="P", s=130, color="black", zorder=5)
    ax.annotate("truth", (0, 0), textcoords="offset points", xytext=(9, 4), fontsize=9)
    ax.set_xlabel("lesion contrast error (%)", fontsize=10)
    ax.set_ylabel("lesion CNR error (%)", fontsize=10)
    ax.set_title("distance from the truth  (circle = pre, square = post)", fontsize=11)
    ax.grid(color="0.92", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # the useful arms crowd the corner next to the truth, so the corner gets its own inset
    axi = ax.inset_axes([0.50, 0.07, 0.47, 0.47])   # the lower-right of this panel is empty
    for i, k in enumerate(keys):
        for scan, mk in zip(SCANS, ("o", "s")):
            r = rows[(scan, k)]
            axi.scatter(100 * r["contrast_err"], 100 * r["cnr_err"], marker="*" if k == MCGS else mk,
                        s=260 if k == MCGS else 55, color=PALETTE[i % len(PALETTE)],
                        zorder=4 if k == MCGS else 3, edgecolor="black", linewidth=0.5)
    axi.scatter([0], [0], marker="P", s=110, color="black", zorder=5)
    axi.set_xlim(-3.6, 0.4); axi.set_ylim(-6.0, 0.7)
    axi.axhline(0, color="0.5", lw=0.8); axi.axvline(0, color="0.5", lw=0.8)
    axi.tick_params(labelsize=8)
    axi.set_title("the corner, enlarged", fontsize=8.5, pad=3)
    ax.indicate_inset_zoom(axi, edgecolor="0.4")

    fig.suptitle("CE retro AF 16.00 -- the lesion boundary as a profile: step height = contrast, slope = boundary "
                 "width, distance from the truth = error", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    FIGDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR_SERVED.mkdir(parents=True, exist_ok=True)
    png = FIGDIR / "ce_retro_af16_lesion_profile.png"
    fig.savefig(png, dpi=a.dpi)
    if a.pdf:
        fig.savefig(FIGDIR / "ce_retro_af16_lesion_profile.pdf")
    plt.close(fig)
    shutil.copy2(png, FIGDIR_SERVED / png.name)
    print(f"wrote {png}\n      http://localhost:8765/ce_retro_seg/{png.name}")
    for scan in SCANS:
        t = prof[(scan, "truth")]
        dev = {k: float(np.nanmean(np.abs(prof[(scan, k)] - t))) for k in keys}
        best = min(dev, key=dev.get)
        print(f"[{scan}] mean |profile - truth| over the band: best {LABEL[best]} {dev[best]:.4f}, "
              f"MC-GS {dev[MCGS]:.4f}, GS-single {dev['gs_single']:.4f}")


if __name__ == "__main__":
    main()
