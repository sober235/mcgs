#!/usr/bin/env python3
"""fig_af_overview.py -- one overview figure of the T1 + FLAIR acceleration study (jiangyizhou, AF 12 / 16 / 20 / 24),
no line graphs (user, 2026-11-20: "the ordinary line graphs feel a bit unprofessional"; metrics PSNR, SSIM, NMSE only).

(a) T1 and (b) FLAIR heatmaps: rows = the 11 compared methods, columns = PSNR | SSIM | NMSE x AF 12 / 16 / 20 / 24
    (head-masked, whole volume). Colour = the value on one single-hue scale per metric block (darker = better), so a row
    read left to right shows how that method degrades with acceleration; the best cell of each column is outlined.
    ZF is drawn with the scale clipped (its values are far below the rest) and says so.
(c) per-slice paired comparison: on every slice whose head covers >= 5 % of the slice (all three planes pooled), the
    improvement of MC-GS over GS-single and over the strongest non-GS method (by volume PSNR at that AF and contrast),
    as violins per AF; PSNR in dB, SSIM in absolute units, NMSE as relative reduction in %. Above each violin: share of
    slices where MC-GS is better. Wilcoxon signed-rank p values go into the record JSON.

Sources (read only, nothing recomputed from volumes except the slice-selection mask):
  volume PSNR / SSIM  Results_20260908/T1FLAIR_AF<N>_v3_jiangyizhou/logs/summary.json (head_psnr, head_ssim)
  volume NMSE         T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou/logs/head_nmse_af<N>_<c>.json
  per-slice metrics   T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou/logs/slice_trend_<c>.json (allslices)
  slice selection     head = |GT| > 0.05 P99.9 of the volume; slices with head fraction >= 0.05 (the trend study's
                      rule; the counts are checked against its 'keep' record)
Writes PNG (200 dpi), PDF (vector text, heatmap cells vector, violins vector) and record JSON to MCGS_ARTIFACT_DIR,
optionally copied to --save_dir.

  gpu.py run --cpu -- @python autoresearch/t1flair_af_record_fig_20261120/fig_af_overview.py --save_dir <server folder>
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--save_dir", default="")
ap.add_argument("--nfig", action="store_true",
                help="edition drawn with the nature-figure skill (user, 2026-11-22; implies --nature): its palette, 5 pt "
                     "glyph floor, values printed only in the two rows the claim rests on (GS-single, MC-GS) with a "
                     "colour scale per block, no number above each violin, render-time panel-alignment gate, "
                     "SVG / PDF / TIFF / PNG export and source-data CSV files; every number unchanged")
ap.add_argument("--nature", action="store_true",
                help="Nature house style (user, 2026-11-22; implies --article): 183 mm wide, sans-serif 5-7 pt, bold "
                     "lower-case panel letters, white background, no in-figure title or caption (the legend goes to "
                     "<stem>_legend.txt), no grid lines, 0.5 pt axes, NPG colours; every number unchanged")
ap.add_argument("--article", action="store_true",
                help="edition for the article (user, 2026-11-22): acceleration factor only (no point-count sentence), "
                     "no subject name in the title; every number unchanged")
ap.add_argument("--png_dpi", type=int, default=200)
ap.add_argument("--pdf_dpi", type=int, default=600)
args = ap.parse_args()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

RES = Path("/data0/congcong/code/GS/Results_20260908")
TR = RES / "T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou"
AFS = (12, 16, 20, 24)
CONTRASTS = ("T1", "FLAIR")
GSKEY = {12: "p14", 16: "p3", 20: "p3", 24: "p3"}
# column order of the image figures; label in the figure -> label in the trend records
METHODS = [("ZF", "ZF"), ("CG-SENSE", "CG-SENSE"), ("CS-SENSE", "CS-SENSE (converged TV)"), ("Joint SENSE", "Joint SENSE"),
           ("L1-SPIRiT", "L1-SPIRiT"), ("ZS-SSDU", "ZS-SSDU"), ("3D-DIP", "3D-DIP"), ("3D-INR", "3D-INR"),
           ("MC-INR", "MC-INR"), ("GS-single", "GS-single"), ("MC-GS", "MC-GS")]
KEY = lambda af: {"ZF": "zf", "CG-SENSE": "cgsense", "CS-SENSE": "cssense_o200", "Joint SENSE": "jcssense",
                  "L1-SPIRiT": "spirit", "ZS-SSDU": "zsssdu", "3D-DIP": "dip", "3D-INR": "inr", "MC-INR": "mcinr",
                  "GS-single": f"gs_single_{GSKEY[af]}", "MC-GS": f"gs_joint_{GSKEY[af]}"}
LABELS = [m for m, _ in METHODS]
NONGS = [m for m in LABELS if m not in ("GS-single", "MC-GS")]
log = lambda *a: print(*a, flush=True)
SK = args.nfig
NAT = args.nature or SK
args.article = args.article or NAT
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", "af_overview_out")).resolve()
art.mkdir(parents=True, exist_ok=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------- volume metrics ----------------
V = {}          # (c, af, label) -> dict(psnr, ssim, nmse)
src = {}
for af in AFS:
    T = RES / f"T1FLAIR_AF{af}_v3_jiangyizhou"
    summ = json.load(open(T / "logs/summary.json"))
    src[f"summary_af{af}"] = sha(T / "logs/summary.json")
    for c in CONTRASTS:
        rows = {r["method"]: r for r in summ if r["contrast"] == c}
        nm_path = TR / f"logs/head_nmse_af{af}_{c}.json"
        src[nm_path.name] = sha(nm_path)
        nm = {r["key"]: r for r in json.load(open(nm_path))["rows"]}
        for lab in LABELS:
            k = KEY(af)[lab]
            r, n = rows[k], nm[k]
            assert abs(n["head_psnr_rescored"] - r["head_psnr"]) < 1e-3, (af, c, lab)   # the two records agree
            V[(c, af, lab)] = dict(psnr=r["head_psnr"], ssim=r["head_ssim"], nmse=n["head_nmse"])

# ---------------- per-slice metrics and the slice selection ----------------
S, KEEP = {}, {}
for c in CONTRASTS:
    p = TR / f"logs/slice_trend_{c}.json"
    src[p.name] = sha(p)
    d = json.load(open(p))
    gt = np.abs(np.load(RES / f"T1FLAIR_AF16_v3_jiangyizhou/recon/gt_{c}_complex.npy")).astype(np.float32)
    for af in (12, 20, 24):         # one truth for all four topics
        g2 = np.abs(np.load(RES / f"T1FLAIR_AF{af}_v3_jiangyizhou/recon/gt_{c}_complex.npy")).astype(np.float32)
        assert np.array_equal(g2, gt), f"GT of AF{af} differs from AF16 ({c})"
    head = gt > 0.05 * float(np.quantile(gt, 0.999))
    for ax in range(3):
        frac = head.mean(axis=tuple(i for i in range(3) if i != ax))
        KEEP[(c, ax)] = frac >= 0.05
        assert int(KEEP[(c, ax)].sum()) == int(d["keep"][str(ax)]), (c, ax, int(KEEP[(c, ax)].sum()), d["keep"])
    for lab, tlab in METHODS:
        for af in AFS:
            for m in ("psnr", "ssim", "nmse"):
                S[(c, af, lab, m)] = np.concatenate(
                    [np.asarray(d["allslices"][f"{tlab}|{ax}"][str(af)][m], float)[KEEP[(c, ax)]] for ax in range(3)])
NSL = len(S[("T1", 12, "MC-GS", "psnr")])
log("slices per contrast and AF (three planes pooled):", NSL)

# ---------------- improvement of MC-GS on every slice ----------------
def improve(m, mc, ot):
    if m == "psnr":
        return mc - ot
    if m == "ssim":
        return mc - ot
    return 100.0 * (ot - mc) / ot          # NMSE: relative reduction, %


REC = dict(sources_sha256=src, n_slices=NSL, volume=dict(), slices=dict(), mcgs_first_every_column={})
for c in CONTRASTS:
    for af in AFS:
        strongest = max(NONGS, key=lambda lab: V[(c, af, lab)]["psnr"])
        for comp in ("GS-single", strongest):
            for m in ("psnr", "ssim", "nmse"):
                dlt = improve(m, S[(c, af, "MC-GS", m)], S[(c, af, comp, m)])
                p = float(wilcoxon(dlt, alternative="greater").pvalue)
                REC["slices"][f"{c}|AF{af}|{m}|vs {comp}"] = dict(
                    comparator=comp, median=float(np.median(dlt)), mean=float(np.mean(dlt)),
                    win_share=float(np.mean(dlt > 0)), wilcoxon_p_greater=p)
        REC["slices"][f"{c}|AF{af}|strongest_non_gs"] = strongest
    for m in ("psnr", "ssim", "nmse"):
        for af in AFS:
            vals = {lab: V[(c, af, lab)][m] for lab in LABELS}
            best = (min if m == "nmse" else max)(vals, key=vals.get)
            REC["mcgs_first_every_column"][f"{c}|{m}|AF{af}"] = best
            REC["volume"][f"{c}|{m}|AF{af}"] = vals

# ---------------- figure ----------------
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # sequential, light -> dark
CMAP = LinearSegmentedColormap.from_list("seqblue", BLUE_RAMP)
C_GS, C_NG = "#eb6834", "#2a78d6"      # categorical slots 2 / 1: vs GS-single, vs strongest non-GS method
plt.rcParams.update({"font.size": 10, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
Z = dict(cell=8.6, met=11, afh=9, ylab=10, ttl=12.5, vttl=11, vy=9.5, pct=8.2, leg=9.5, best_lw=1.6, iqr=2.2, dot=26,
         dot_lw=1.2, vlw=0.8, alpha=0.55, pad=26)
if NAT:
    SURF, INK, INK2, BASE = "#ffffff", "#000000", "#000000", "#000000"
    BLUE_RAMP = ["#ffffff", "#dfe5f0", "#b3c0d9", "#8497bd", "#5a72a3", "#3C5488", "#273a63"]
    CMAP = LinearSegmentedColormap.from_list("npgblue", BLUE_RAMP)
    C_GS, C_NG = "#E64B35", "#3C5488"                      # NPG palette
    Z = dict(cell=5.0, met=7, afh=5.5, ylab=6.5, ttl=7, vttl=7, vy=6.5, pct=5, leg=6, best_lw=0.8, iqr=1.0, dot=7,
             dot_lw=0.5, vlw=0.4, alpha=0.75, pad=15)
    plt.rcParams.update({"font.size": 6, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "savefig.facecolor": SURF, "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2, "ytick.major.size": 2, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
if SK:
    BLUE_RAMP = ["#EEF2F8", "#C9D8EC", "#9DB9DD", "#6F97CB", "#3775BA", "#0F4D92", "#0A3466"]
    CMAP = LinearSegmentedColormap.from_list("skillblue", BLUE_RAMP)
    C_GS, C_NG = "#B64342", "#0F4D92"                      # the skill's red_strong / blue_main
    INK = INK2 = BASE = "#272727"
    Z.update(cell=5.5, met=7, afh=6, ylab=6.5, ttl=7, vttl=7, vy=6.5, pct=5, leg=6, alpha=0.8)
    plt.rcParams.update({"font.size": 6.5, "svg.fonttype": "none", "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "axes.edgecolor": BASE, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "text.color": INK, "xtick.labelsize": 6, "ytick.labelsize": 6})
    sys.path.insert(0, str(Path(__file__).resolve().parent / "nature_qa"))
    from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
CBARS = []
SHOW = (lambda m, v: 100.0 * v if (SK and m == "nmse") else v)      # skill edition prints NMSE in per cent
MET = [("psnr", "PSNR (dB)", "{:.2f}"), ("ssim", "SSIM", "{:.3f}"),
       ("nmse", "NMSE (%)", "{:.2f}") if SK else ("nmse", "NMSE", "{:.4f}")]
if NAT:
    fig = plt.figure(figsize=(7.2, 6.3 if SK else 6.15))                  # 183 mm, the double-column width; compact (user, 2026-11-22)
    gs_top = fig.add_gridspec(1, 2, left=0.082, right=0.999, top=0.998, bottom=0.588, wspace=0.185)
    gs_bot = fig.add_gridspec(2, 3, left=0.062, right=0.997, top=0.528, bottom=0.058, hspace=0.36, wspace=0.26)
else:
    fig = plt.figure(figsize=(17.0, 15.2))
    gs_top = fig.add_gridspec(1, 2, left=0.075, right=0.985, top=0.905, bottom=0.535, wspace=0.16)
    gs_bot = fig.add_gridspec(2, 3, left=0.06, right=0.985, top=0.445, bottom=0.05, hspace=0.42, wspace=0.2)


def heat(ax, c, tag):
    ncol = 4 * len(MET)
    gap = 0.35                       # gap between metric blocks, in cell units
    xs = [b * (4 + gap) + i for b in range(len(MET)) for i in range(4)]
    for b, (m, mname, fmt) in enumerate(MET):
        vals = np.array([[V[(c, af, lab)][m] for af in AFS] for lab in LABELS])
        better = -vals if m == "nmse" else vals                    # larger = better for the colour
        lo = np.min(better[[i for i, lab in enumerate(LABELS) if lab != "ZF"]])
        hi = np.max(better)
        norm = Normalize(lo, hi, clip=True)
        for i, lab in enumerate(LABELS):
            for j, af in enumerate(AFS):
                x = xs[b * 4 + j]
                v = norm(better[i, j])
                col = CMAP(0.0 if lab == "ZF" else 0.08 + 0.92 * v)
                ax.add_patch(Rectangle((x, i), 0.96, 0.92, fc=col, ec=SURF, lw=0))
                dark = (lab != "ZF") and (0.08 + 0.92 * v) > 0.55
                best = REC["mcgs_first_every_column"][f"{c}|{m}|AF{af}"] == lab
                if not SK or lab in ("GS-single", "MC-GS"):
                    ax.text(x + 0.48, i + 0.47, fmt.format(SHOW(m, vals[i, j])), ha="center", va="center", fontsize=Z["cell"],
                            color="#ffffff" if dark else INK, fontweight="bold" if best else "normal")
                if best and not SK:
                    ax.add_patch(Rectangle((x + 0.03, i + 0.03), 0.90, 0.86, fill=False, ec=INK, lw=Z["best_lw"]))
        if SK:                      # the colour scale of this block: worst method other than ZF -> best method
            cax = ax.inset_axes([xs[b * 4], len(LABELS) + 0.3, 3.96, 0.26], transform=ax.transData)
            cax.imshow(np.linspace(0.08, 1.0, 256)[None, :], cmap=CMAP, vmin=0, vmax=1, aspect="auto", extent=(0, 1, 0, 1))
            cax.set_yticks([]); cax.set_xticks([0, 1])
            cax.set_xticklabels([fmt.format(SHOW(m, -lo if m == "nmse" else lo)), fmt.format(SHOW(m, -hi if m == "nmse" else hi))], fontsize=5.5)
            for t_, ha_ in zip(cax.get_xticklabels(), ("left", "right")):
                t_.set_ha(ha_)
            # one outline around the run of columns in which a method is the best (no stroke between its numbers)
            for i, lab in enumerate(LABELS):
                run = [j for j, af in enumerate(AFS) if REC["mcgs_first_every_column"][f"{c}|{m}|AF{af}"] == lab]
                for j0 in [j for j in run if j - 1 not in run]:
                    j1 = j0
                    while j1 + 1 in run:
                        j1 += 1
                    ax.add_patch(Rectangle((xs[b * 4 + j0] + 0.02, i + 0.02), (j1 - j0) + 0.92, 0.88, fill=False, ec=INK, lw=Z["best_lw"]))
            cax.tick_params(length=1.5, width=0.5, pad=1)
            for sp in cax.spines.values():
                sp.set_linewidth(0.4)
            CBARS.append(cax)
        xc = xs[b * 4] + 2 - 0.02
        ax.text(xc, -1.05, mname, ha="center", va="center", fontsize=Z["met"], color=INK, fontweight="bold")
        for j, af in enumerate(AFS):
            ax.text(xs[b * 4 + j] + 0.48, -0.35, f"{af}" if NAT else f"AF{af}", ha="center", va="center", fontsize=Z["afh"], color=INK2)
    ax.set_xlim(-0.1, xs[-1] + 1.05)
    ax.set_ylim(len(LABELS) + (1.25 if SK else 0.02), -1.32 if NAT else -1.5)
    ax.set_yticks(np.arange(len(LABELS)) + 0.46)
    ax.set_yticklabels(LABELS, fontsize=Z["ylab"])
    for t, lab in zip(ax.get_yticklabels(), LABELS):
        if lab == "MC-GS":
            t.set_fontweight("bold"); t.set_color(INK)
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    if NAT:
        ax.text(xs[0] - 0.15, -1.05, c, ha="right", va="center", fontsize=Z["ttl"], color=INK)
        ax.text(-0.16, 0.995, tag.strip("()"), transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="left")
        ax.text(xs[0] - 0.15, -0.35, "AF", ha="right", va="center", fontsize=Z["afh"], color=INK2)
    else:
        ax.set_title(f"{tag}  {c}: head-masked whole-volume metrics", loc="left", fontsize=12.5, color=INK, pad=26)


heat(fig.add_subplot(gs_top[0, 0]), "T1", "(a)")
heat(fig.add_subplot(gs_top[0, 1]), "FLAIR", "(b)")
CAP_AB = ("Colour: value on one scale per metric block (darker = better; NMSE scale inverted); ZF below the "
          "scale, shown lightest.  Outlined + bold: best of the 11 methods in that column.  "
          "One vd-Poisson mask per acceleration factor for both contrasts.")
if not NAT:
  fig.text(0.075, 0.515, "Colour: value on one scale per metric block (darker = better; NMSE scale inverted); ZF below the "
         "scale, shown lightest.  Outlined + bold: best of the 11 methods in that column.  "
         + ("One vd-Poisson mask per acceleration factor for both contrasts." if args.article else
            "AF = whole phase-encode plane / sampled points (11.71 / 15.19 / 18.56 / 24.00x), one vd-Poisson mask per AF "
            "for both contrasts."), fontsize=9.2, color=INK2, ha="left", va="top", wrap=True)


def violins(ax, c, m, ylab, title):
    pos_gs = np.arange(len(AFS)) * 1.0 - 0.19
    pos_ng = np.arange(len(AFS)) * 1.0 + 0.19
    if NAT:
        ax.axhline(0, color="#7f7f7f", lw=0.5, ls=(0, (3, 2)), zorder=1)
    else:
        ax.axhline(0, color=BASE, lw=1.1, zorder=1)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    tops = []
    for k, af in enumerate(AFS):
        strongest = REC["slices"][f"{c}|AF{af}|strongest_non_gs"]
        for comp, pos, col in (("GS-single", pos_gs[k], C_GS), (strongest, pos_ng[k], C_NG)):
            dlt = improve(m, S[(c, af, "MC-GS", m)], S[(c, af, comp, m)])
            lo, hi = np.percentile(dlt, [0.5, 99.5])
            body = ax.violinplot(np.clip(dlt, lo, hi), positions=[pos], widths=0.34, showextrema=False)
            for pc in body["bodies"]:
                pc.set_facecolor(col); pc.set_edgecolor(col); pc.set_alpha(Z["alpha"]); pc.set_linewidth(Z["vlw"])
            q1, med, q3 = np.percentile(dlt, [25, 50, 75])
            ax.plot([pos, pos], [q1, q3], color=INK, lw=Z["iqr"], solid_capstyle="round", zorder=3)
            ax.scatter([pos], [med], s=Z["dot"], color=SURF, edgecolor=INK, lw=Z["dot_lw"], zorder=4)
            tops.append((pos, hi, float(np.mean(dlt > 0))))
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for pos, hi, w in ([] if SK else tops):
        ax.text(pos, ymax + 0.02 * span, f"{100 * w:.0f}%", ha="center", va="bottom", fontsize=Z["pct"], color=INK2)
    ax.set_ylim(min(ymin, -0.04 * span), ymax + (0.02 if SK else 0.12) * span)
    ax.set_xticks(np.arange(len(AFS)))
    ax.set_xticklabels([f"{af}" if NAT else f"AF{af}" for af in AFS])
    if NAT:
        ax.set_xlabel("Acceleration factor", fontsize=Z["vy"], labelpad=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_ylabel(ylab, fontsize=Z["vy"])
    ax.set_title(title, loc="left", fontsize=Z["vttl"], color=INK, pad=2 if NAT else 6)


MLAB = {"psnr": ("ΔPSNR (dB)", "PSNR gain"), "ssim": ("ΔSSIM", "SSIM gain"), "nmse": ("NMSE reduction (%)", "NMSE reduction")}
for i, c in enumerate(CONTRASTS):
    for j, m in enumerate(("psnr", "ssim", "nmse")):
        ylab, ttl = MLAB[m]
        axv = fig.add_subplot(gs_bot[i, j])
        violins(axv, c, m, ylab, c if NAT else f"{'(c)  ' if (i, j) == (0, 0) else ''}{c} · {ttl} of MC-GS per slice")
        if NAT and (i, j) == (0, 0):
            axv.text(-0.2, 1.04, "c", transform=axv.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
from matplotlib.patches import Patch  # noqa: E402
fig.legend(handles=[Patch(fc=C_GS, ec=C_GS, alpha=Z["alpha"], label="MC-GS vs GS-single"),
                    Patch(fc=C_NG, ec=C_NG, alpha=Z["alpha"], label="MC-GS vs strongest non-GS method "
                                                                   "(CS-SENSE on T1, Joint SENSE on FLAIR)")],
           loc="upper left", bbox_to_anchor=(0.1, 0.592) if NAT else (0.06, 0.478), ncol=2, frameon=False, fontsize=Z["leg"])
CAP_C = (f"{NSL} slices per contrast and acceleration factor (transverse, coronal and sagittal slices whose head covers "
         ">= 5% of the slice). Above zero = MC-GS better. Bar = interquartile range, dot = median, violin trimmed to "
         "the 0.5-99.5 percentiles. Number above each violin = share of slices where MC-GS is better; "
         "Wilcoxon signed-rank p < 1e-60 in every pair.")
if not NAT:
  fig.text(0.06, 0.03, f"(c) {NSL} slices per contrast and AF (transverse, coronal and sagittal slices whose head covers "
         ">= 5% of the slice). Above zero = MC-GS better.\nBar = interquartile range, dot = median, violin trimmed to "
         "the 0.5-99.5 percentiles. Number above each violin = share of slices where MC-GS is better; "
         "Wilcoxon signed-rank p < 1e-60 in every pair.",
         fontsize=9.2, color=INK2, ha="left", va="top")
if not NAT:
  fig.suptitle("MC-GS across acceleration factors: " + ("" if args.article else "jiangyizhou ") + "T1 + FLAIR, retrospective vd-Poisson undersampling",
             x=0.01, ha="left", y=0.985, fontsize=14.5, fontweight="bold", color=INK)
if not NAT:
  fig.text(0.01, 0.958, "11 methods, one scorer, one ground truth; single subject, seed 42; GS-single = the registered arm "
         "(one scan fed to both channels of the joint model); ZS-SSDU with 5 residual blocks.",
         ha="left", fontsize=10, color=INK2)

stem = "t1flair_af12_16_20_24_overview_psnr_ssim_nmse" + ("_nfig" if SK else "_nature" if NAT else "_article" if args.article else "")
if SK:
    CAP_AB = ("Colour: value on the scale drawn under each block, from the weakest method other than zero filling (ZF, "
              "off the scale, lightest) to the best method; NMSE in per cent, its scale inverted so that darker is better in every block. "
              "Values are printed for MC-GS and the strongest comparison, GS-single; bold and outlined = best of the 11 "
              "methods in that column. All values are in the source data. One vd-Poisson mask per acceleration factor "
              "for both contrasts.")
    ws = [v["win_share"] for v in REC["slices"].values() if isinstance(v, dict)]
    CAP_C = (f"n = {NSL} slices per contrast and acceleration factor (transverse, coronal and sagittal slices whose head "
             "covers >= 5% of the slice; slices of one volume, not independent subjects). Above the dashed line = MC-GS "
             "better. Dot, median; bar, interquartile range; violin trimmed to the 0.5-99.5 percentiles. MC-GS is better "
             f"in {100 * min(ws):.0f}-{100 * max(ws):.0f}% of the slices, depending on the pair (source data); one-sided "
             "Wilcoxon signed-rank test, P < 1e-60 in every pair, not corrected for multiple comparisons.")
if NAT:
    args.png_dpi = max(args.png_dpi, 600)
    (art / f"{stem}_legend.txt").write_text(
        "MC-GS across acceleration factors: T1 + FLAIR, retrospective vd-Poisson undersampling.\n"
        "a, b, Head-masked whole-volume PSNR, SSIM and NMSE of the 11 methods at acceleration factors 12, 16, 20 and 24 "
        "for T1 (a) and FLAIR (b). " + CAP_AB + "\n"
        "c, Per-slice gain of MC-GS over GS-single and over the strongest non-GS method (CS-SENSE on T1, Joint SENSE on "
        "FLAIR): PSNR gain, SSIM gain and relative NMSE reduction. " + CAP_C + "\n"
        "One scorer and one ground truth for all methods; single subject, seed 42; GS-single = the registered arm (one "
        "scan fed to both channels of the joint model); ZS-SSDU with 5 residual blocks.\n")
EXTRA = []
if SK:
    fig.canvas.draw()
    rep_al = require_matplotlib_panel_alignment(fig, json_out=str(art / f"{stem}.alignment.json"),
                                                overlay_svg=str(art / f"{stem}.alignment.svg"), tolerance_pt=1.5,
                                                gutter_tolerance_pt=1.5, strict=True, exclude_axes=CBARS)
    log("panel alignment gate:", rep_al.get("status", rep_al.get("verdict", "passed")))
    fig.savefig(art / f"{stem}.svg")
    fig.savefig(art / f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    with open(art / f"{stem}_source_ab.csv", "w") as f:
        f.write("panel,contrast,metric,acceleration_factor,method,value\n")
        for c in CONTRASTS:
            for m, _, _ in MET:
                for af in AFS:
                    for lab in LABELS:
                        f.write(f"{'a' if c == 'T1' else 'b'},{c},{m},{af},{lab},{V[(c, af, lab)][m]!r}\n")
    with open(art / f"{stem}_source_c_summary.csv", "w") as f:
        f.write("contrast,acceleration_factor,metric,comparator,n_slices,median,q1,q3,share_mcgs_better,wilcoxon_p_one_sided\n")
        for c in CONTRASTS:
            for af in AFS:
                for comp in ("GS-single", REC["slices"][f"{c}|AF{af}|strongest_non_gs"]):
                    for m in ("psnr", "ssim", "nmse"):
                        d_ = improve(m, S[(c, af, "MC-GS", m)], S[(c, af, comp, m)])
                        q1, md, q3 = np.percentile(d_, [25, 50, 75])
                        r_ = REC["slices"][f"{c}|AF{af}|{m}|vs {comp}"]
                        f.write(f"{c},{af},{m},{comp},{len(d_)},{md!r},{q1!r},{q3!r},{r_['win_share']!r},{r_['wilcoxon_p_greater']!r}\n")
    with open(art / f"{stem}_source_c_slices.csv", "w") as f:
        f.write("contrast,acceleration_factor,metric,comparator,slice_index,mcgs_improvement\n")
        for c in CONTRASTS:
            for af in AFS:
                for comp in ("GS-single", REC["slices"][f"{c}|AF{af}|strongest_non_gs"]):
                    for m in ("psnr", "ssim", "nmse"):
                        for i_, v_ in enumerate(improve(m, S[(c, af, "MC-GS", m)], S[(c, af, comp, m)])):
                            f.write(f"{c},{af},{m},{comp},{i_},{v_!r}\n")
    EXTRA = [f"{stem}.svg", f"{stem}.tiff", f"{stem}.alignment.json", f"{stem}_source_ab.csv",
             f"{stem}_source_c_summary.csv", f"{stem}_source_c_slices.csv"]
fig.savefig(art / f"{stem}.png", dpi=args.png_dpi)
fig.savefig(art / f"{stem}.pdf", dpi=args.pdf_dpi)
plt.close(fig)

firsts = REC["mcgs_first_every_column"]
n1 = sum(v == "MC-GS" for v in firsts.values())
log(f"MC-GS best in {n1} / {len(firsts)} heatmap columns; others:", {k: v for k, v in firsts.items() if v != "MC-GS"})
for k, v in REC["slices"].items():
    if isinstance(v, dict):
        log(f"  {k:<44} median {v['median']:+.4f}  win {100 * v['win_share']:5.1f}%  p {v['wilcoxon_p_greater']:.1e}")
(art / f"record_{stem}.json").write_text(json.dumps(REC, indent=1))
outs = [f"{stem}.png", f"{stem}.pdf", f"record_{stem}.json"] + ([f"{stem}_legend.txt"] if NAT else []) + EXTRA
for n in outs:
    log(f"  sha256 {sha(art / n)}  {n}")
if args.save_dir:
    dst = Path(args.save_dir); dst.mkdir(parents=True, exist_ok=True)
    for s_ in [art / n for n in outs] + [Path(__file__).resolve()]:
        t = dst / s_.name
        if t.exists() and sha(t) != sha(s_):
            raise SystemExit(f"{t} exists with different content; not overwritten")
        if not t.exists():
            shutil.copyfile(s_, t); log("  saved", t)
log("DONE")
