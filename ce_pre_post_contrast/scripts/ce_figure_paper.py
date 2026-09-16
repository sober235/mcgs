#!/usr/bin/env python3
"""ce_figure_paper.py -- publication figure for the UIH prospective CE pair.

Reconstructed weighted images only (no error maps): 3 orthogonal planes x {pre, post}
x 11 methods.  Volumes are cropped to the head bounding box and packed with near-zero
inter-panel spacing.  All 22 volumes are put on ONE common intensity scale by the
best-fit complex scale against the measured k-space,

    alpha = argmin_a || M .( a.A(x) - y ) ||_2 ,

so a single colorbar is meaningful across every panel and pre/post enhancement is a
real intensity difference rather than a per-panel normalisation artefact.
"""
import os, sys, json, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0, help="-1 = CPU")
ap.add_argument("--out", default="/home/congcongliu/figs/ce_pair")
ap.add_argument("--dpi", type=int, default=600)
ap.add_argument("--vmax", type=float, default=1.0)
ap.add_argument("--margin", type=int, default=4)
ap.add_argument("--zlo", type=int, default=105, help="inferior cut of the brain slab (axis0)")
ap.add_argument("--slices", default="201,131,48",
                help="axial z, coronal y, sagittal x (axis 0, 1, 2); default = the lesion core from ce_lesion_candidates.py "
                     "slices (2026-09-15). Earlier figures used 170,162,86")
ap.add_argument("--probe", action="store_true", help="dump an orientation check only")
ap.add_argument("--cache", default="/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/figures/cache/paper_mag.npz",
                help="alpha-scaled magnitudes (float16); reused while every input volume is unchanged")
ap.add_argument("--recompute", action="store_true")
ap.add_argument("--drop", nargs="*", default=["L1-ESPIRiT"],
                help="column labels to leave out; default drops L1-ESPIRiT (user, 2026-09-15: L1-SPIRiT is the L1 arm). '--drop' with no labels renders every column")
ap.add_argument("--suffix", default="", help="appended to the output file name")
ap.add_argument("--diff", action="store_true",
                help="add a post - pre row per plane and, per column, the enhancement of the reference regions "
                     "(2026-09-16). Output suffix defaults to _enhancement")
ap.add_argument("--dvmax", type=float, default=0.20, help="top of the post - pre scale; 0.20 = 0.6 white-matter units")
ap.add_argument("--dsmooth", type=float, default=1.0,
                help="Gaussian sigma on the displayed post - pre (the analysis in ce_lesion_candidates.py uses 1.0); "
                     "0 shows it unsmoothed, which is dominated by the noise of two independent reconstructions")
ap.add_argument("--rois", default="lesion_candidates/enhance_rois_cssense.npz",
                help="masks written by ce_lesion_candidates.py enhance; without them --diff draws no numbers")
ap.add_argument("--zoom", action="store_true",
                help="instead of the panel figure: zoomed crops of the axial slice, three rows -- the lesion and a "
                     "vessel region in post - pre, and the background in post (2026-09-16)")
ap.add_argument("--zoom_size", type=int, default=72, help="side of the zoom boxes, mm")
ap.add_argument("--bgvmax", type=float, default=0.12, help="top of the grey scale in the background row")
args = ap.parse_args()
if args.diff and not args.suffix:
    args.suffix = "_enhancement"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.gpu >= 0 else ""
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np, torch, torch.fft as tfft
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, gridspec
for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    try: font_manager.fontManager.addfont(_f)
    except Exception: pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

dev = torch.device("cuda:0" if args.gpu >= 0 else "cpu")
if args.gpu < 0:
    torch.set_num_threads(16)
W = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast")
RR = Path("/data0/congcong/code/NV-Generate-CTMR/recon_real")
WORK = {"pre": RR / "work_ce_t1_16887049370010", "post": RR / "work_ce_t1_16887049370060"}
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)

def _c(fn, x, d): return tfft.fftshift(fn(tfft.ifftshift(x, dim=d), dim=d, norm="ortho"), dim=d)
def fft3c(x):  return _c(tfft.fftn,  x, (-3, -2, -1))
def ifft3c(x): return _c(tfft.ifftn, x, (-3, -2, -1))

# lambdas of the 2026-09-14 arms: held-out selection (scripts/ce_select_lambda.py, written by ce_orchestrate.py)
SEL = {fam: json.load(open(W / "logs/ext_20260914" / f"selection_{fam}.json"))["best"]
       for fam in ("cs_pre", "cs_post", "joint", "l1_pre", "l1_post")}
COLS = [
    ("ZF",                        lambda s: W / f"classical/zf_{s}_complex.npy"),
    ("CG-SENSE",                  lambda s: W / f"classical/cgsense_{s}_complex.npy"),
    ("CS-SENSE",                  lambda s: W / f"tv_sense/cssense_o200_lam{SEL['cs_' + s]:g}_full_{s}_complex.npy"),
    ("Joint SENSE",               lambda s: W / f"tv_sense/jsense_o200_lam{SEL['joint']:g}_full_{s}_complex.npy"),
    ("L1-ESPIRiT",                lambda s: W / f"l1espirit/l1espirit_i100_lam{SEL['l1_' + s]:g}_full_{s}_complex.npy"),
    ("ZS-SSDU",                   lambda s: W / f"classical/zsssdu_{s}_complex.npy"),
    ("3D-DIP",                    lambda s: W / f"dipinr/dip_{s}_complex.npy"),
    ("3D-INR",                    lambda s: W / f"dipinr/inr_{s}_complex.npy"),
    ("MC-INR",                    lambda s: W / f"mcinr/mcinr_full_{s}_complex.npy"),
    ("GS-single",                 lambda s: W / f"gs_single_{s}/gs_single_{s}_raw_complex.npy"),
    ("MC-GS",                     lambda s: W / f"gs_joint/gs_joint_{s}_raw_complex.npy"),
]
# L1-SPIRiT (2026-09-15): inserted after L1-ESPIRiT once its held-out selection exists (ce_orchestrate.py plan spirit)
SP = {sc: json.load(open(W / "logs/ext_20260914" / f"selection_spirit_{sc}.json")) for sc in ("pre", "post")
      if (W / "logs/ext_20260914" / f"selection_spirit_{sc}.json").exists()}
if len(SP) == 2:
    COLS.insert([l for l, _ in COLS].index("L1-ESPIRiT") + 1,
                ("L1-SPIRiT", lambda s: W / f"spirit/spirit_ks{SP[s]['ksize']}_i150_lam{SP[s]['lam']:g}_full_{s}_complex.npy"))
COLS = [c for c in COLS if c[0] not in set(args.drop)]
CHUNK = 6
cache = Path(args.cache)
colsig = lambda sc, pth: json.dumps([str(pth(sc)), os.path.getmtime(pth(sc))])
stored = {}
if cache.exists() and not args.recompute:
    with np.load(cache) as c:
        stored = {k: c[k] for k in c.files if k != "sig"}
mag, zfmag = {}, {}
for sc in ("pre", "post"):
    if f"zf|{sc}" in stored:
        zfmag[sc] = stored[f"zf|{sc}"].astype(np.float32)
    for label, pth in COLS:
        k = f"{sc}|{label}"
        if k in stored and "sig|" + k in stored and str(stored["sig|" + k]) == colsig(sc, pth):
            mag[(sc, label)] = stored[k].astype(np.float32)
todo = {sc: [(l, pth) for l, pth in COLS if (sc, l) not in mag] for sc in ("pre", "post")}
log(f"cache {cache.name}: {2 * len(COLS) - sum(map(len, todo.values()))} of {2 * len(COLS)} volumes reused")
for scan in [sc for sc in ("pre", "post") if todo[sc] or sc not in zfmag]:
    log(f"=== {scan} ===")
    y  = torch.from_numpy(np.load(WORK[scan] / "ksp_full.npy")).to(dev)
    S  = torch.from_numpy(np.load(WORK[scan] / "csm.npy")).to(dev)
    m3 = torch.from_numpy(np.load(WORK[scan] / "mask2d.npy")).to(dev) > 0.5
    y = y * m3; NC = y.shape[0]; G = tuple(m3.shape)
    acc = torch.zeros(G, dtype=torch.complex64, device=dev)
    for i in range(0, NC, CHUNK):
        acc += (S[i:i+CHUNK].conj() * ifft3c(y[i:i+CHUNK])).sum(0)
    zfmag[scan] = acc.abs().cpu().numpy()
    for label, pth in todo[scan]:
        p = pth(scan)
        x = torch.from_numpy(np.load(p)).to(dev)
        if tuple(x.shape) != G:                              # GS renders in (X, Z, Y)
            x = x.permute(0, 2, 1).contiguous()
        num = torch.zeros((), dtype=torch.complex64, device=dev); den = torch.zeros((), device=dev)
        for i in range(0, NC, CHUNK):
            km = fft3c(S[i:i+CHUNK] * x) * m3
            num = num + (km.conj() * y[i:i+CHUNK]).sum(); den = den + km.abs().square().sum()
        alpha = num / den.clamp_min(1e-30)
        mag[(scan, label)] = (alpha * x).abs().cpu().numpy()
        log(f"  {label:<10s} |alpha|={abs(alpha.item()):.4f}")
        del x
    del y, S, m3, acc; torch.cuda.empty_cache()
if any(todo.values()) or any(f"zf|{sc}" not in stored for sc in ("pre", "post")):
    cache.parent.mkdir(parents=True, exist_ok=True)
    allsig = dict(COLS)
    for sc in zfmag:
        stored[f"zf|{sc}"] = zfmag[sc].astype(np.float16)
    for (sc, l), m in mag.items():
        stored[f"{sc}|{l}"], stored[f"sig|{sc}|{l}"] = m.astype(np.float16), np.array(colsig(sc, allsig[l]))
    np.savez(cache, **stored)
    log(f"cached magnitudes to {cache}")

# ---- one common intensity scale for all 22 volumes ------------------------------------
REF = float(np.percentile(zfmag["pre"][::3], 99.9))
log(f"common scale (P99.9 of |ZF_pre|) = {REF:.4e}")
for k in mag: mag[k] = mag[k] / REF

# ---- head bounding box (shared by every panel) ----------------------------------------
head = (zfmag["pre"] / REF) > 0.08
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

# ---- layout ---------------------------------------------------------------------------
PLANE = ("Axial", "Coronal", "Sagittal")
ROI, CORE = None, None
if args.diff or args.zoom:
    from scipy.ndimage import gaussian_filter
    rp = W / args.rois
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


# ---- zoomed comparison: what the proposed arm keeps and what it does not invent --------
if args.zoom:
    if ROI is None:
        sys.exit("--zoom needs the reference regions: run ce_lesion_candidates.py enhance first")
    from scipy.ndimage import binary_dilation, uniform_filter
    from matplotlib.patches import Rectangle
    S, H, Wd = args.zoom_size, mag[("pre", "ZF")].shape[1], mag[("pre", "ZF")].shape[2]
    D = {l: gaussian_filter(mag[("post", l)] - mag[("pre", l)], args.dsmooth) for l, _ in COLS}
    wmu = float(np.median(mag[("pre", "CS-SENSE")][ROI["正常白质"]]))     # one white-matter unit, common scale
    win = lambda m: uniform_filter(m.astype(np.float32), S, mode="constant") * S * S   # window count / mean
    clip = lambda yy, xx: (int(np.clip(yy, 0, H - S)), int(np.clip(xx, 0, Wd - S)))
    pts = np.argwhere(CORE[zi])
    ly, lx = clip(*(int(v) - S // 2 for v in pts.mean(0)))                # lesion box
    cnt = win(D["CS-SENSE"][zi] > 0.3 * wmu)                              # vessel box: most enhancing voxels,
    cnt[max(ly - S, 0):ly + 2 * S, max(lx - S, 0):lx + 2 * S] = 0         # clear of the lesion
    vy, vx = clip(*(int(v) - S // 2 for v in np.unravel_index(int(cnt.argmax()), cnt.shape)))
    avg = np.mean([mag[("post", l)][zi] for l, _ in COLS], axis=0)        # background box: outside the head,
    bgw = np.where(win(binary_dilation(head[zi], iterations=4)) > 0, -1.0,            # where the arms put most signal
                   uniform_filter(avg, S, mode="constant"))
    by, bx = clip(*(int(v) - S // 2 for v in np.unravel_index(int(bgw.argmax()), bgw.shape)))
    BOX = [("病灶（增强 − 平扫）", "#FFC23D", (ly, lx)), ("血管（增强 − 平扫）", "#3BE8FF", (vy, vx)),
           ("颅外背景（增强）", "#FFFFFF", (by, bx))]
    for t, _, (yy, xx) in BOX:
        log(f"{t}: y {yy}:{yy + S}  x {xx}:{xx + S}")
    outside = ~binary_dilation(head, iterations=6)
    num = {}
    for label, _ in COLS:
        e, b = enh_units(label), mag[("post", label)]
        num[label] = (float(e[ROI["病灶核心"]].mean()), float(e[ROI["血管/静脉窦"]].mean()),
                      float(b[outside].mean() / np.median(b[ROI["正常白质"]])))
    fig, axs = plt.subplots(3, len(COLS) + 1, figsize=(1.45 * (len(COLS) + 1.2), 6.1), layout="compressed",
                            gridspec_kw=dict(width_ratios=[(x1 - x0) / (y1 - y0)] + [1.0] * len(COLS)))
    for r, (title, col, (yy, xx)) in enumerate(BOX):
        ax = axs[r, 0]
        ax.imshow(mag[("post", "CS-SENSE")][zi, y0:y1, x0:x1], cmap="gray", vmin=0, vmax=args.vmax)
        ax.add_patch(Rectangle((xx - x0, yy - y0), S, S, fill=False, ec=col, lw=0.9))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_ylabel(title, fontsize=8.5)
        if r == 0: ax.set_title(f"定位 z={zi}", fontsize=8.5)
        for c, (label, _) in enumerate(COLS, start=1):
            a2 = axs[r, c]
            if r < 2:
                a2.imshow(D[label][zi, yy:yy + S, xx:xx + S], cmap="magma", vmin=0, vmax=args.dvmax)
                a2.set_xlabel(f"病灶 {num[label][0]:+.3f}" if r == 0 else f"血管 {num[label][1]:+.2f}", fontsize=7)
            else:
                a2.imshow(mag[("post", label)][zi, yy:yy + S, xx:xx + S], cmap="gray", vmin=0, vmax=args.bgvmax)
                a2.set_xlabel(f"颅外 {num[label][2]:.3f}", fontsize=7)
            a2.set_xticks([]); a2.set_yticks([])
            if r == 0: a2.set_title(label, fontsize=8.5)
    fig.suptitle(f"放大对比（横断位 z={zi}，{S}×{S} mm 框）：上两行是「增强 − 平扫」，下行是增强图的颅外背景\n"
                 "每格下方的数值是全脑统计，不是该框内的：病灶核心与血管/静脉窦为对应 ROI 的「增强 − 平扫」均值，"
                 "颅外为头掩膜膨胀 6 mm 之外全部体素的增强图均值，三者同以白质 = 1\n"
                 "病灶核心越接近 0 越好（不造假强化），血管/静脉窦越大说明强化保得越住，颅外越小背景越干净", fontsize=10)
    q = OUT / f"ce_pair_zoom_mcgs{args.suffix}.png"
    fig.savefig(q, dpi=min(args.dpi, 300), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log(f"saved {q}")
    sys.exit(0)

if args.diff:
    ROWS = [(s, pi, f"{t}\n{PLANE[pi]}") for pi in range(3)
            for s, t in (("pre", "Pre-contrast"), ("post", "Post-contrast"), ("diff", "Post − Pre"))]
    for label, _ in COLS:                          # the enhancement itself, on the same common scale
        e = mag[("post", label)] - mag[("pre", label)]
        mag[("diff", label)] = gaussian_filter(e, args.dsmooth) if args.dsmooth > 0 else e
else:
    ROWS = [("pre", 0, "Pre-contrast\nAxial"), ("pre", 1, "Pre-contrast\nCoronal"), ("pre", 2, "Pre-contrast\nSagittal"),
            ("post", 0, "Post-contrast\nAxial"), ("post", 1, "Post-contrast\nCoronal"), ("post", 2, "Post-contrast\nSagittal")]
shapes = planes(mag[("pre", "ZF")])
ncol, nrow = len(COLS), len(ROWS)
COLW = 1.55                                        # inches per column
hr = [shapes[r[1]].shape[0] / shapes[r[1]].shape[1] for r in ROWS]
pad_b = 0.78 if ROI else 0.04                      # room for the per-column numbers and the caption
figw = COLW * ncol + 0.62 + 0.42                   # + row labels + colorbar
figh = COLW * sum(hr) + 0.30 + pad_b
fig = plt.figure(figsize=(figw, figh))
gs = gridspec.GridSpec(nrow, ncol, figure=fig, height_ratios=hr, wspace=0.003, hspace=0.003,
                       left=0.62/figw, right=1 - 0.42/figw, top=1 - 0.30/figh, bottom=pad_b/figh)
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
    wmu = float(np.median(mag[("pre", COLS[1][0])][wmm]))           # one white-matter unit on the common scale
    fig.text(0.5, 0.010, "每列下方 = 该方法的「增强 − 平扫」在参考区的均值，两次扫描各按自身白质中位数归一（白质 = 1）。"
             "血管/静脉窦与脉络丛是血池和无血脑屏障的结构，本该强化；病灶核心与正常白质接近 0 表示没有漏出。"
             f"橙线 = 病灶核心。差值行与图像行共用同一强度尺度，色标 0–{args.dvmax:g} 相当于 {args.dvmax / wmu:.1f} 个白质单位，"
             f"显示前做了 σ={args.dsmooth:g} 的高斯平滑（数值未平滑）。",
             ha="center", fontsize=8)
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
    q = OUT / f"ce_pair_recon_3planes{args.suffix}.{ext}"
    fig.savefig(q, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
    log(f"saved {q}")
plt.close(fig)
