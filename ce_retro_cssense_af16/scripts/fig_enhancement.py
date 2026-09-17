#!/usr/bin/env python3
"""fig_enhancement.py -- the post-contrast enhancement figure: pre / post / post-pre per plane, every arm plus GT.

Ported from ce_figure_paper.py's --diff path (CE_pre_post_contrast topic). That topic has no ground truth and puts
its 22 volumes on one common intensity scale by alpha-fitting each against its own measured k-space (a GPU/torch
step, cached to disk); this topic's recon/*.npy are already in the shared CS-SENSE-truth gauge (recon_classical.py,
recon_spirit.py, recon_zsssdu.py, recon_dip_inr.py, run_mcinr.py, extract_gs_volumes.py all divide by the same
truth scale) and adds a genuine ground truth (recon/gt_<scan>_complex.npy / sim/gt.npz), so the alpha-fit / k-space
/ GPU loading block is replaced by a direct load through arms.arm_files(), and the truth is a twelfth column.

  python scripts/fig_enhancement.py

9 rows (3 planes x {pre, post, post-pre}) x 12 columns (GT + the 11 arms of arms.ORDER).
"""
import argparse
import os
import sys
from pathlib import Path

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(R / "figures"))
ap.add_argument("--dpi", type=int, default=600)
ap.add_argument("--vmax", type=float, default=1.0)
ap.add_argument("--margin", type=int, default=4)
ap.add_argument("--zlo", type=int, default=105, help="inferior cut of the brain slab (axis0)")
ap.add_argument("--slices", default="201,131,48",
                help="axial z, coronal y, sagittal x (axis 0, 1, 2); default = the lesion core from "
                     "CE_pre_post_contrast/scripts/ce_lesion_candidates.py slices (2026-09-15)")
ap.add_argument("--probe", action="store_true", help="dump an orientation check only")
ap.add_argument("--suffix", default="", help="appended to the output file name")
ap.add_argument("--dvmax", type=float, default=0.20, help="top of the post - pre scale; 0.20 = 0.6 white-matter units")
ap.add_argument("--dsmooth", type=float, default=1.0,
                help="Gaussian sigma on the displayed post - pre (the analysis in ce_lesion_candidates.py uses 1.0); "
                     "0 shows it unsmoothed, which is dominated by the noise of two independent reconstructions")
args = ap.parse_args()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, gridspec
from scipy.ndimage import gaussian_filter
for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    try: font_manager.fontManager.addfont(_f)
    except Exception: pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms import arm_files                                                   # noqa: E402
ROI_NPZ = "/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz"
mag = {}
with np.load(R / "sim/gt.npz") as g:
    for scan in ("pre", "post"):
        mag[(scan, "GT")] = (np.abs(g[scan]) / float(g[f"scale_{scan}"])).astype(np.float32)
for scan in ("pre", "post"):
    for key, label, p in arm_files(scan):
        v = np.load(p)
        mag[(scan, label)] = (np.abs(v) if np.iscomplexobj(v) else v).astype(np.float32)
COLS = [("GT", None)] + [(label, None) for _, label, _ in arm_files("pre")]

OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)

# ---- head bounding box (shared by every panel), from the truth ------------------------
head = mag[("pre", "GT")] > 0.08
# axis0 is S-I with the neck/shoulders at low indices; keep the brain only.
z0 = args.zlo
zi_pr = np.where(head.any(axis=(1, 2)))[0]
z1 = min(int(zi_pr[-1]) + 1 + args.margin, head.shape[0])
def span(ax_):                                    # in-plane bbox, brain slab only
    h = head[z0:z1]
    pr = h.sum(axis=tuple(a for a in range(3) if a != ax_)).astype(float)
    i = np.where(pr > 0.08 * pr.max())[0]          # ignore stray coil-noise voxels at the edges
    return max(int(i[0]) - args.margin, 0), min(int(i[-1]) + 1 + args.margin, h.shape[ax_])
y0, y1 = span(1); x0, x1 = span(2)
log(f"brain bbox: axis0 {z0}:{z1}  axis1 {y0}:{y1}  axis2 {x0}:{x1}")

# ---- slice indices: all three planes through the lesion core (ce_lesion_candidates.py slices) ----
zi, yi, xi = (int(v) for v in args.slices.split(","))
log(f"lesion slices: axial z={zi}  coronal y={yi}  sagittal x={xi}")

# volume layout is (S-I, A-P, L-R); flip S-I so superior is up in coronal/sagittal
def planes(v):
    ax  = v[zi, y0:y1, x0:x1]                      # axial   : (A-P, L-R), anterior (eyes) up (2026-09-15, was rot90)
    cor = v[z0:z1, yi, x0:x1][::-1]                # coronal : (S-I, L-R)
    sag = v[z0:z1, y0:y1, xi][::-1, ::-1]          # sagittal: (S-I, A-P); A-P index 0 is anterior, so anterior is right
    return ax, cor, sag

if args.probe:
    fig, axs = plt.subplots(1, 3, figsize=(9, 4))
    for k, (im, t) in enumerate(zip(planes(mag[("post", "CG-SENSE")]), ("axial", "coronal", "sagittal"))):
        axs[k].imshow(im, cmap="gray", vmin=0, vmax=args.vmax); axs[k].set_title(f"{t} {im.shape}")
        axs[k].set_xticks([]); axs[k].set_yticks([])
    fig.tight_layout(); fig.savefig(OUT / "orient_probe.png", dpi=110)
    log("probe saved"); sys.exit(0)

# ---- layout -----------------------------------------------------------------------------
PLANE = ("Axial", "Coronal", "Sagittal")
ROI, CORE = None, None
rp = Path(ROI_NPZ)
if rp.exists():
    with np.load(rp) as r:
        ROI = {n: r[f"roi{i}"] for i, n in enumerate(r["names"])}
        CORE = r["core"]
    log(f"reference regions from {rp.name}: " + ", ".join(f"{k} {int(v.sum())}" for k, v in ROI.items()))
else:
    log(f"no {rp}; run ce_lesion_candidates.py enhance to get the numbers")


def enh_units(label):
    """post - pre with each scan divided by its own white-matter median, so that 1 = white matter."""
    a, b = mag[("pre", label)], mag[("post", label)]
    return b / np.median(b[ROI["正常白质"]]) - a / np.median(a[ROI["正常白质"]])


ROWS = [(s, pi, f"{t}\n{PLANE[pi]}") for pi in range(3)
        for s, t in (("pre", "Pre-contrast"), ("post", "Post-contrast"), ("diff", "Post − Pre"))]
for label, _ in COLS:                          # the enhancement itself, on the shared truth gauge
    e = mag[("post", label)] - mag[("pre", label)]
    mag[("diff", label)] = gaussian_filter(e, args.dsmooth) if args.dsmooth > 0 else e

shapes = planes(mag[("pre", "GT")])
ncol, nrow = len(COLS), len(ROWS)
COLW = 1.55                                        # inches per column
hr = [shapes[r[1]].shape[0] / shapes[r[1]].shape[1] for r in ROWS]
pad_b = 0.78 if ROI else 0.04                      # room for the per-column numbers and the caption
pad_t = 0.30 + 0.22                                # column-title row (as ce_figure_paper.py) + one line for fig.suptitle
figw = COLW * ncol + 0.62 + 0.42                   # + row labels + colorbar
figh = COLW * sum(hr) + pad_t + pad_b
fig = plt.figure(figsize=(figw, figh))
gs = gridspec.GridSpec(nrow, ncol, figure=fig, height_ratios=hr, wspace=0.003, hspace=0.003,
                       left=0.62/figw, right=1 - 0.42/figw, top=1 - pad_t/figh, bottom=pad_b/figh)
im = hdiff = None
last = {}
for ri, (scan, pi, rlab) in enumerate(ROWS):
    for ci, (label, _) in enumerate(COLS):
        ax = fig.add_subplot(gs[ri, ci])
        d = scan == "diff"
        h = ax.imshow(planes(mag[(scan, label)])[pi], cmap="magma" if d else "gray", vmin=0,
                      vmax=args.dvmax if d else args.vmax, interpolation="nearest", aspect="auto")
        if d:
            hdiff = h
            if CORE is not None:                   # where the lesion is, for reference
                ax.contour(planes(CORE)[pi], levels=[0.5], colors="#FFC23D", linewidths=0.35)
        else:
            im = h
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if ri == 0: ax.set_title(label, fontsize=9, pad=3)
        if ci == 0: ax.set_ylabel(rlab, fontsize=8, labelpad=2)
        if ri == nrow - 1: last[label] = ax
if ROI:                                            # enhancement of the reference regions, per method
    wmm = ROI["正常白质"]
    for label, _ in COLS:
        e = enh_units(label)                                        # each scan in its own white-matter units
        last[label].set_xlabel(f"血管 {e[ROI['血管/静脉窦']].mean():+.2f}   脉络丛 {e[ROI['脉络丛']].mean():+.2f}\n"
                               f"病灶 {e[ROI['病灶核心']].mean():+.3f}   白质 {e[wmm].mean():+.3f}", fontsize=6.5, labelpad=3)
    _col_labels = [l for l, _ in COLS]
    # anchor by label, not position: COLS[0] is "GT" here (ce_figure_paper.py had no truth column, so its
    # COLS[1] was "CG-SENSE"; a positional lookup here would silently resolve to "ZF", the worst possible
    # white-matter reference at AF16). Prefer CS-SENSE; fall back to the first non-ZF, non-GT arm.
    _wm_ref = "CS-SENSE" if "CS-SENSE" in _col_labels else next((l for l in _col_labels if l not in ("ZF", "GT")), None)
    if _wm_ref is None:
        raise SystemExit("fig_enhancement.py: no arm besides ZF/GT available to anchor the white-matter unit on")
    wmu = float(np.median(mag[("pre", _wm_ref)][wmm]))               # one white-matter unit on the common scale
    fig.text(0.5, 0.010, "每列下方 = 该方法的「增强 − 平扫」在参考区的均值，两次扫描各按自身白质中位数归一（白质 = 1）。"
             "血管/静脉窦与脉络丛是血池和无血脑屏障的结构，本该强化；病灶核心与正常白质接近 0 表示没有漏出。"
             f"橙线 = 病灶核心。差值行与图像行共用同一强度尺度，色标 0–{args.dvmax:g} 相当于 {args.dvmax / wmu:.1f} 个白质单位，"
             f"显示前做了 σ={args.dsmooth:g} 的高斯平滑（数值未平滑）。",
             ha="center", fontsize=8)
fig.suptitle("UIH pre/post pair, CS-SENSE truth, Poisson-disc AF 16.00, 11 arms + truth",
             fontsize=11, fontweight="bold", y=1 - 0.06 / figh)
if ROI:
    cax = fig.add_axes([1 - 0.34/figw, 0.45, 0.10/figw, 0.45])
    cb2 = fig.colorbar(hdiff, cax=fig.add_axes([1 - 0.34/figw, pad_b/figh + 0.02, 0.10/figw, 0.33]))
    cb2.set_label("post − pre (same scale)", fontsize=8); cb2.ax.tick_params(labelsize=7)
else:
    cax = fig.add_axes([1 - 0.34/figw, 0.10, 0.10/figw, 0.78])
cb = fig.colorbar(im, cax=cax)
cb.set_label("normalized signal intensity (a.u.)", fontsize=8)
cb.ax.tick_params(labelsize=7)
for ext in ("png", "pdf"):
    q = OUT / f"ce_retro_af16_enhancement{args.suffix}.{ext}"
    fig.savefig(q, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
    log(f"saved {q}")
plt.close(fig)
