#!/usr/bin/env python3
"""fig_lesion_profile_nature.py -- the lesion boundary profile figure, drawn to a flagship journal contract.

FIGURE CONTRACT
  Core conclusion   At AF 16.00, MC-GS keeps the lesion's boundary contrast and conspicuity closest to the truth,
                    while the two implicit-neural arms systematically fill the lesion in.
  Results question  Does joint Gaussian-splatting reconstruction preserve the lesion as an imaging finding, and not
                    only the global fidelity the PSNR table reports?
  Evidence chain    a  pre-contrast boundary profile with its residual strip -- the phenomenon itself: step height
                       is the lesion contrast, slope is the boundary width, residual is each arm's deviation
                    b  the same on the post-contrast acquisition -- replication on a second, independently
                       undersampled scan rather than a second metric on the same one
                    c  contrast error against CNR error -- the claim turned into a distance from the truth
  Archetype         quantitative grid
  Export            183 mm double column x 95 mm, 7 pt text, 6 pt ticks, 8 pt bold lower-case panel labels, Arial,
                    editable TrueType text (PDF fonttype 42, SVG fonttype none), 600 dpi TIFF, source-data CSV,
                    blocking render-time panel-alignment gate, PDF collision audit after every layout change.

DATA INTEGRITY
  Seven of the topic's eleven arms are drawn, the subset the user fixed for this figure series (CS-SENSE,
  Joint SENSE, ZS-SSDU, 3D-INR, MC-INR, GS-single, MC-GS); ZF, CG-SENSE, L1-SPIRiT and 3D-DIP are omitted by that
  request, not for convenience, and all eleven remain in logs/lesion_preservation.json. Every voxel of the lesion
  and of the 6 mm collar around it enters the profile; nothing is subsampled. Regions are the fixed ones of
  lesion_preservation.py, drawn once on the truth, identical for every arm, with no segmentation model anywhere.

  env PYTHONNOUSERSITE=1 PYTHONPATH= FIGURE_QA_SCRIPTS=<publication QA helpers> <python> scripts/fig_lesion_profile_nature.py

Writes figures_seg/nature/ce_retro_af16_lesion_profile.{pdf,svg,tiff,png} plus the source-data CSV and the
alignment report; the PNG is a screen preview of the same render, not a separate figure.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
PROS = Path(os.environ.get("CE_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast"))
ROI_NPZ = PROS / "lesion_candidates/enhance_rois_cssense.npz"
OUTDIR = R / "figures_seg/nature"
FIGDIR_SERVED = Path("/home/congcongliu/figs/ce_retro_seg")

SCANS = ("pre", "post")
MM = 6
MIN_N = 50            # a distance bin thinner than this is not averaged; see the note in profiles()
MCGS, GSS = "gs_joint", "gs_single"
MM_PER_IN = 25.4
FIG_W_MM, FIG_H_MM = 183.0, 100.0

# One restrained palette: a neutral family for the model-based baselines, one signal family for the Gaussian
# splatting arms, one amber accent for the implicit-neural arms that lose the lesion.
STYLE = {
    "cssense_o200": dict(color="#8C8C8C", lw=0.7, zorder=2),
    "jcssense":     dict(color="#B5B5B5", lw=0.7, zorder=2),
    "zsssdu":       dict(color="#5A5A5A", lw=0.7, zorder=2),
    "inr":          dict(color="#E08214", lw=0.9, zorder=3),
    "mcinr":        dict(color="#B35806", lw=0.9, zorder=3),
    GSS:            dict(color="#7FB3E0", lw=0.9, zorder=4),
    MCGS:           dict(color="#0B4F9C", lw=1.5, zorder=6),
}
MARKER = {MCGS: "*", GSS: "o", "cssense_o200": "s", "jcssense": "D", "zsssdu": "^", "inr": "v", "mcinr": "P"}
TRUTH_KW = dict(color="black", lw=1.2, zorder=5)


def profiles(keys, arm_files, summary):
    """mean intensity in white-matter units against the signed distance to the lesion boundary.

    No bin at zero: the signed distance is <= -1 strictly inside the core and >= +1 strictly outside it, so a zero
    bin is always empty and would break every curve in the middle."""
    z = np.load(ROI_NPZ, allow_pickle=True)
    core, wm = z["core"], z["roi2"]
    sd = ndi.distance_transform_edt(~core) - ndi.distance_transform_edt(core)
    # The deepest interior bin can be empty: the lesion is not 2*MM mm thick everywhere, so a -MM shell exists
    # only if some voxel is that far inside. Bins empty on either scan are dropped and reported rather than
    # silently averaged over nothing.
    # No head mask here. The lesion and its 6 mm collar sit deep in the brain, so the mask removes no background;
    # what it does do is select differently on the two scans, because its threshold is 0.05 * P99.9 of that scan
    # and the post-contrast P99.9 is raised by the enhancing vessels. That cut the dark lesion interior harder on
    # post (12 voxels at -5 mm against 20 on pre) and made the two panels sample different voxels. Dropping it
    # gives both scans the identical voxel set. Bins thinner than MIN_N voxels are dropped instead of averaged.
    cand = [x for x in range(-MM, MM + 1) if x != 0]
    prof, counts = {}, {}
    keep = None
    for scan in SCANS:
        sel = np.abs(sd) <= MM
        d = np.rint(sd[sel]).astype(int)
        occupied = [x for x in cand if (d == x).sum() >= MIN_N]
        keep = occupied if keep is None else [x for x in keep if x in occupied]
    xs = np.array(keep)
    for scan in SCANS:
        gt = np.abs(np.load(R / f"recon/gt_{scan}_complex.npy")).astype(np.float64)
        sel = np.abs(sd) <= MM
        d = np.rint(sd[sel]).astype(int)
        bins = {x: (d == x) for x in xs}
        counts[scan] = np.array([int(bins[x].sum()) for x in xs])
        gv = (gt / float(np.median(gt[wm])))[sel]
        prof[(scan, "truth")] = np.array([gv[bins[x]].mean() for x in xs])
        for key, _, path in arm_files(scan, summary):
            if key not in keys:
                continue
            v = np.abs(np.load(path)).astype(np.float64)
            vv = (v / float(np.median(v[wm])))[sel]
            prof[(scan, key)] = np.array([vv[bins[x]].mean() for x in xs])
    dropped = [x for x in cand if x not in keep]
    if dropped:
        print(f"distance bins dropped for holding fewer than {MIN_N} voxels: {dropped} mm")
    return xs, prof, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="cssense_o200,jcssense,zsssdu,inr,mcinr,gs_single,gs_joint")
    ap.add_argument("--tiff_dpi", type=int, default=600)
    a = ap.parse_args()
    keys = [k.strip() for k in a.arms.split(",") if k.strip()]

    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
    })
    import matplotlib.pyplot as plt

    from arms import LABEL, arm_files
    summary = json.load(open(R / "logs/summary.json"))
    rows = {(r["scan"], r["arm"]): r for r in json.load(open(R / "logs/lesion_preservation.json"))["rows"]}
    xs, prof, counts = profiles(keys, arm_files, summary)

    fig = plt.figure(figsize=(FIG_W_MM / MM_PER_IN, FIG_H_MM / MM_PER_IN))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.05, 1.0], width_ratios=[1, 1, 1],
                          left=0.062, right=0.995, top=0.905, bottom=0.185, wspace=0.30, hspace=0.12)
    axes_top, axes_res = [], []
    for j, scan in enumerate(SCANS):
        ax = fig.add_subplot(gs[0, j])
        axr = fig.add_subplot(gs[1, j], sharex=ax)
        axes_top.append(ax); axes_res.append(axr)
        t = prof[(scan, "truth")]
        for aa in (ax, axr):
            aa.axvspan(-MM, 0, color="#F2F2F2", lw=0, zorder=0)
        for k in keys:
            ax.plot(xs, prof[(scan, k)], **STYLE[k])
            axr.plot(xs, prof[(scan, k)] - t, **STYLE[k])
        ax.plot(xs, t, **TRUTH_KW)
        axr.axhline(0.0, **{**TRUTH_KW, "lw": 0.9})
        ax.set_xlim(-MM, MM)
        ax.set_ylim(0.20, 0.66)
        axr.set_ylim(-0.055, 0.125)
        ax.tick_params(labelbottom=False)
        axr.set_xlabel("Signed distance to lesion boundary (mm)")
        ax.set_ylabel("Intensity (white-matter units)" if j == 0 else "")
        axr.set_ylabel("Arm − truth" if j == 0 else "")
        ax.set_title(f"{'Pre' if scan == 'pre' else 'Post'}-contrast", pad=2.5)
        ax.text(-MM + 0.35, 0.645, "lesion", fontsize=6, color="#6E6E6E", va="top")
        ax.text(MM - 0.35, 0.645, "tissue", fontsize=6, color="#6E6E6E", va="top", ha="right")
    # one shared legend in the empty strip below the panels: inside either profile panel it would sit on the
    # curves or on the shaded lesion half, which the rendered collision audit rejects
    handles = [plt.Line2D([], [], **TRUTH_KW, label="Truth")]
    handles += [plt.Line2D([], [], **{kk: vv for kk, vv in STYLE[k].items() if kk != "zorder"}, label=LABEL[k])
                for k in keys]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.36, 0.005), ncol=8, handlelength=1.6,
               columnspacing=1.1, handletextpad=0.45, borderaxespad=0.0)

    axc = fig.add_subplot(gs[:, 2])
    axc.axhline(0, color="#BFBFBF", lw=0.6, zorder=1)
    axc.axvline(0, color="#BFBFBF", lw=0.6, zorder=1)
    for k in keys:
        for scan, fill in zip(SCANS, (True, False)):
            r = rows[(scan, k)]
            axc.scatter(100 * r["contrast_err"], 100 * r["cnr_err"], marker=MARKER[k],
                        s=42 if k == MCGS else 13, zorder=6 if k == MCGS else 3,
                        facecolor=STYLE[k]["color"] if fill else "white",
                        edgecolor=STYLE[k]["color"], linewidth=0.6)
    axc.scatter([0], [0], marker="+", s=34, color="black", linewidth=0.9, zorder=7)
    axc.annotate("Truth", (0, 0), xytext=(3.5, 2.5), textcoords="offset points", fontsize=6)
    axc.annotate("MC-GS", (100 * rows[("pre", MCGS)]["contrast_err"], 100 * rows[("pre", MCGS)]["cnr_err"]),
                 xytext=(-30, -10), textcoords="offset points", fontsize=6, color=STYLE[MCGS]["color"],
                 fontweight="bold")
    axc.set_xlabel("Lesion contrast error (%)")
    axc.set_ylabel("Lesion CNR error (%)")
    axc.set_title("Distance from the truth", pad=2.5)
    axc.set_xlim(-11.5, 1.8)
    axc.set_ylim(-27.5, 3.5)
    axc.scatter([], [], marker="o", s=13, facecolor="#5A5A5A", edgecolor="#5A5A5A", linewidth=0.6, label="Pre")
    axc.scatter([], [], marker="o", s=13, facecolor="white", edgecolor="#5A5A5A", linewidth=0.6, label="Post")
    axc.legend(loc="lower right", handletextpad=0.35, labelspacing=0.3, borderaxespad=0.25)

    # one shared baseline for every panel label: panel c spans two rows, so the same axes-fraction offset would
    # put its label higher than a and b and the alignment gate would (correctly) reject the layout
    fig.canvas.draw()
    label_y = axes_top[0].get_position().y1 + 0.022
    for ax, lab in ((axes_top[0], "a"), (axes_top[1], "b"), (axc, "c")):
        fig.text(ax.get_position().x0 - 0.048, label_y, lab, fontsize=8, fontweight="bold",
                 va="bottom", ha="left")

    # source data, so the plotted numbers travel with the figure
    OUTDIR.mkdir(parents=True, exist_ok=True)
    stem = OUTDIR / "ce_retro_af16_lesion_profile"
    with open(f"{stem}.source-data.csv", "w") as f:
        f.write("panel,scan,series,distance_mm,n_voxels,intensity_wm_units,residual_vs_truth\n")
        for j, scan in enumerate(SCANS):
            for name in ["truth"] + keys:
                p = prof[(scan, name)]
                t = prof[(scan, "truth")]
                for i, x in enumerate(xs):
                    f.write(f"{'ab'[j]},{scan},{LABEL.get(name, name)},{x},{counts[scan][i]},"
                            f"{p[i]:.6f},{p[i] - t[i]:.6f}\n")
        f.write("panel,scan,series,contrast_error_pct,cnr_error_pct\n")
        for scan in SCANS:
            for k in keys:
                r = rows[(scan, k)]
                f.write(f"c,{scan},{LABEL[k]},{100 * r['contrast_err']:.4f},{100 * r['cnr_err']:.4f}\n")

    sys.path.insert(0, os.environ.get("FIGURE_QA_SCRIPTS", ""))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    report = require_matplotlib_panel_alignment(
        fig, json_out=f"{stem}.alignment.json", overlay_svg=f"{stem}.alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    print(f"alignment: {report.get('status', '?')}")

    for ext, kw in (("pdf", {}), ("svg", {}), ("tiff", dict(dpi=a.tiff_dpi)), ("png", dict(dpi=400))):
        fig.savefig(f"{stem}.{ext}", **kw)      # no bbox_inches: the canvas must stay exactly 183 mm wide
    plt.close(fig)
    FIGDIR_SERVED.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f"{stem}.png", FIGDIR_SERVED / "ce_retro_af16_lesion_profile_nature.png")
    print(f"wrote {stem}.{{pdf,svg,tiff,png}} at {FIG_W_MM} x {FIG_H_MM} mm")
    print("      http://localhost:8765/ce_retro_seg/ce_retro_af16_lesion_profile_nature.png")


if __name__ == "__main__":
    main()
