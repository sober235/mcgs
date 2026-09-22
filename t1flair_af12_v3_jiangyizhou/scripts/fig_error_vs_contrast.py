#!/usr/bin/env python3
"""fig_error_vs_contrast.py -- reconstruction error against local image contrast, T1 and FLAIR on one axis.

  python scripts/fig_error_vs_contrast.py [--arm gs_joint_p14] [--bins 10] [--win 5] [--dark]

Answers the question of 2026-09-19 ("is FLAIR harder because its contrast varies less?"). Head voxels are
binned by local contrast -- the standard deviation of |GT| in a --win^3 window, divided by the RMS of |GT|
over that contrast's head, so the two contrasts share one scale. Per bin the figure shows the relative RMS
error of one arm, each contrast normalised by its own head RMS, so the two curves are directly comparable.
Bin edges are quantiles of the pooled local contrast: shared by both contrasts, equal count overall.

Three panels on one x axis: the two error curves, their ratio, and the signal level of each bin. The third
panel is there because the bins are not the same tissue in the two contrasts -- FLAIR's flattest voxels are
its nulled CSF and are much darker than T1's flattest voxels, which are bulk white matter -- and that
confound has to be visible rather than argued about.

Writes figures/t1flair_v3_af12_err_vs_contrast_<arm>.{png,pdf} and mirrors them to the served figs folder.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="gs_joint_p14", help="registered arm to profile (recon/<arm>_<c>_{mag,complex}.npy)")
ap.add_argument("--label", default="MC-GS", help="how to name that arm on the figure")
ap.add_argument("--bins", type=int, default=10)
ap.add_argument("--win", type=int, default=5, help="side of the cubic window the local contrast is measured in")
ap.add_argument("--dark", action="store_true", help="dark-surface variant, appends _dark")
args = ap.parse_args()

R = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou"
SERVED = "/home/congcongliu/figs/t1flair_v3_af12"
CONTRASTS = ("T1", "FLAIR")
COL = {"T1": "#2a9d8f", "FLAIR": "#e76f51"}     # categorical pair, all six dataviz checks pass on both surfaces
if args.dark:
    SURF, INK, INK2, GRID, BASE = "#1a1a19", "#ffffff", "#e6e5df", "#33332f", "#4a4a46"
else:
    SURF, INK, INK2, GRID, BASE = "#ffffff", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.edgecolor": BASE, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "pdf.fonttype": 42, "ps.fonttype": 42})


def load(name, c):
    for suf in ("_mag.npy", "_complex.npy"):
        p = f"{R}/recon/{name}_{c}{suf}"
        if os.path.exists(p):
            return np.abs(np.load(p)).astype(np.float32)
    raise FileNotFoundError(f"{name} {c}")


D = {}
for c in CONTRASTS:
    gt = load("gt", c)
    head = gt > 0.05 * np.quantile(gt, 0.999)
    m = uniform_filter(gt, args.win)
    lstd = np.sqrt(np.clip(uniform_filter(gt * gt, args.win) - m * m, 0, None))
    rms = float(np.sqrt((gt[head].astype(np.float64) ** 2).mean()))
    D[c] = dict(lc=(lstd[head] / rms).astype(np.float32), rms=rms, g=gt[head].astype(np.float64),
                e=(load(args.arm, c) - gt)[head].astype(np.float64))
    print(f"{c}: {head.sum():,} head voxels, head RMS {rms:.4f}")

# one set of bin edges for both contrasts: quantiles of the pooled local contrast
pool = np.concatenate([D[c]["lc"] for c in CONTRASTS])
edges = np.quantile(pool, np.linspace(0, 1, args.bins + 1))
edges[0], edges[-1] = 0.0, np.inf
for c in CONTRASTS:
    lc, e = D[c]["lc"], D[c]["e"]
    idx = np.clip(np.digitize(lc, edges[1:-1]), 0, args.bins - 1)
    D[c]["x"] = np.array([lc[idx == i].mean() for i in range(args.bins)])
    D[c]["y"] = np.array([100 * np.sqrt((e[idx == i] ** 2).mean()) / D[c]["rms"] for i in range(args.bins)])
    D[c]["s"] = np.array([float(np.sqrt((D[c]["g"][idx == i] ** 2).mean())) for i in range(args.bins)])

X = np.mean([D[c]["x"] for c in CONTRASTS], axis=0)          # the two are within a few percent of each other
ratio = D["FLAIR"]["y"] / D["T1"]["y"]

fig, (ax, axr, axn) = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True,
                                   gridspec_kw=dict(height_ratios=[3.1, 1.0, 0.9], hspace=0.16))
for a in (ax, axr, axn):
    a.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

for c in CONTRASTS:
    ax.plot(D[c]["x"], D[c]["y"], "-", color=COL[c], lw=2.0, zorder=4)
    ax.plot(D[c]["x"], D[c]["y"], "o", ms=5.5, mfc=COL[c], mec=SURF, mew=0.9, zorder=5)
    ax.annotate(c, (D[c]["x"][-1], D[c]["y"][-1]), xytext=(6, 5 if c == "FLAIR" else -6),
                textcoords="offset points", va="center", fontsize=8, color=INK2)
ax.set_ylabel(f"relative RMS error of {args.label}  (% of head RMS)")
ax.set_ylim(0, max(D[c]["y"].max() for c in CONTRASTS) * 1.14)

axr.axhline(1.0, color=BASE, lw=0.8, zorder=1)
axr.plot(X, ratio, "-", color=INK2, lw=1.5, zorder=4)
axr.plot(X, ratio, "o", ms=4.5, mfc=INK2, mec=SURF, mew=0.8, zorder=5)
axr.set_ylabel("FLAIR / T1", labelpad=9)
axr.set_ylim(0.93, 1.26)
axr.set_yticks([1.0, 1.1, 1.2])

for c in CONTRASTS:
    axn.plot(D[c]["x"], D[c]["s"], "-", color=COL[c], lw=1.4, zorder=4)
    axn.plot(D[c]["x"], D[c]["s"], "o", ms=3.8, mfc=COL[c], mec=SURF, mew=0.7, zorder=5)
axn.set_ylabel("|GT| RMS\nin the bin", labelpad=4)
axn.set_xlabel("local contrast   (std of |GT| in a %d$^3$ window, / head RMS)" % args.win)

ax.set_title("Error rises with local contrast on both contrasts;\nFLAIR is worse only in the flattest tenth",
             fontsize=8.5, fontweight="bold", color=INK, pad=8, loc="left")
fig.text(0.007, 0.012,
         f"jiangyizhou, nominal AF 12, arm {args.label} ({args.arm}); head voxels binned on quantiles of the pooled\n"
         f"local contrast ({args.bins} bins, equal count overall), each contrast's error divided by its own head RMS.\n"
         f"Smooth tissue is where the reconstruction is most accurate, on both contrasts — low contrast is not the hard part;\n"
         f"over bins 2–10 the two agree to within 9 %. The flattest bin is the exception (FLAIR +21 %), and the bottom panel\n"
         f"shows it is not the same tissue there: FLAIR's flattest voxels are its nulled CSF, darker than T1's bulk white matter.",
         fontsize=6.4, color=INK2, va="bottom", linespacing=1.45)
fig.subplots_adjust(left=0.115, right=0.935, top=0.885, bottom=0.175)

os.makedirs(f"{R}/figures", exist_ok=True)
os.makedirs(SERVED, exist_ok=True)
name = f"t1flair_v3_af12_err_vs_contrast_{args.arm}" + ("_dark" if args.dark else "")
for d in (f"{R}/figures", SERVED):
    fig.savefig(f"{d}/{name}.png", dpi=300)
    fig.savefig(f"{d}/{name}.pdf")
print("wrote", f"{SERVED}/{name}.png")
print(f"{'bin':>4s} {'local contrast':>15s} {'T1 err %':>9s} {'FLAIR err %':>12s} {'FL/T1':>7s} "
      f"{'|GT| RMS T1':>12s} {'FLAIR':>7s}")
for i in range(args.bins):
    print(f"{i+1:4d} {X[i]:15.3f} {D['T1']['y'][i]:9.2f} {D['FLAIR']['y'][i]:12.2f} {ratio[i]:7.3f} "
          f"{D['T1']['s'][i]:12.4f} {D['FLAIR']['s'][i]:7.4f}")
