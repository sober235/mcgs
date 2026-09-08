#!/usr/bin/env python3
"""ce_figure_paper.py -- publication figure for the UIH prospective CE pair.

Reconstructed weighted images only (no error maps): 3 orthogonal planes x {pre, post}
x 8 methods.  Volumes are cropped to the head bounding box and packed with near-zero
inter-panel spacing.  All 16 volumes are put on ONE common intensity scale by the
best-fit complex scale against the measured k-space,

    alpha = argmin_a || M .( a.A(x) - y ) ||_2 ,

so a single colorbar is meaningful across every panel and pre/post enhancement is a
real intensity difference rather than a per-panel normalisation artefact.
"""
import os, sys, json, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--out", default="/home/congcongliu/figs/ce_pair")
ap.add_argument("--dpi", type=int, default=600)
ap.add_argument("--vmax", type=float, default=1.0)
ap.add_argument("--margin", type=int, default=4)
ap.add_argument("--zlo", type=int, default=105, help="inferior cut of the brain slab (axis0)")
ap.add_argument("--zax", type=int, default=170, help="axial display level (axis0)")
ap.add_argument("--probe", action="store_true", help="dump an orientation check only")
args = ap.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
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

dev = torch.device("cuda:0")
W = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast")
RR = Path("/data0/congcong/code/NV-Generate-CTMR/recon_real")
WORK = {"pre": RR / "work_ce_t1_16887049370010", "post": RR / "work_ce_t1_16887049370060"}
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)

def _c(fn, x, d): return tfft.fftshift(fn(tfft.ifftshift(x, dim=d), dim=d, norm="ortho"), dim=d)
def fft3c(x):  return _c(tfft.fftn,  x, (-3, -2, -1))
def ifft3c(x): return _c(tfft.ifftn, x, (-3, -2, -1))

COLS = [
    ("ZF",                        lambda s: W / f"classical/zf_{s}_complex.npy"),
    ("CG-SENSE",                  lambda s: W / f"classical/cgsense_{s}_complex.npy"),
    ("CS-SENSE",                  lambda s: W / f"classical/cssense_{s}_complex.npy"),
    ("ZS-SSDU",                   lambda s: W / f"classical/zsssdu_{s}_complex.npy"),
    ("3D-DIP",                    lambda s: W / f"dipinr/dip_{s}_complex.npy"),
    ("3D-INR",                    lambda s: W / f"dipinr/inr_{s}_complex.npy"),
    ("GS single",                 lambda s: W / f"gs_single_{s}/gs_single_{s}_raw_complex.npy"),
    ("GS joint",                  lambda s: W / f"gs_joint/gs_joint_{s}_raw_complex.npy"),
]
CHUNK = 6
mag, zfmag = {}, {}
for scan in ("pre", "post"):
    log(f"=== {scan} ===")
    y  = torch.from_numpy(np.load(WORK[scan] / "ksp_full.npy")).to(dev)
    S  = torch.from_numpy(np.load(WORK[scan] / "csm.npy")).to(dev)
    m3 = torch.from_numpy(np.load(WORK[scan] / "mask2d.npy")).to(dev) > 0.5
    y = y * m3; NC = y.shape[0]; G = tuple(m3.shape)
    acc = torch.zeros(G, dtype=torch.complex64, device=dev)
    for i in range(0, NC, CHUNK):
        acc += (S[i:i+CHUNK].conj() * ifft3c(y[i:i+CHUNK])).sum(0)
    zfmag[scan] = acc.abs().cpu().numpy()
    for label, pth in COLS:
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

# ---- one common intensity scale for all 16 volumes ------------------------------------
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

# ---- slice indices: put all three planes through the enhancing lesion -----------------
a, b = mag[("pre", "CG-SENSE")], mag[("post", "CG-SENSE")]
d = np.clip(b - a, 0, None) * head
from scipy.ndimage import uniform_filter
core = np.zeros_like(d, dtype=bool); core[args.zax-12:args.zax+12, 40:-40, 30:-30] = True
ds = uniform_filter(d * core, size=5)
zi, yi, xi = (int(v) for v in np.unravel_index(int(np.argmax(ds)), d.shape))
zi = args.zax                                     # fix the axial level; coronal/sagittal follow the lesion
log(f"lesion-centred slices: axial z={zi}  coronal y={yi}  sagittal x={xi}")

# volume layout is (S-I, A-P, L-R); flip S-I so superior is up in coronal/sagittal
def planes(v):
    ax  = np.rot90(v[zi, y0:y1, x0:x1])            # axial   : (L-R, A-P)
    cor = v[z0:z1, yi, x0:x1][::-1]                # coronal : (S-I, L-R)
    sag = v[z0:z1, y0:y1, xi][::-1, ::-1]          # sagittal: (S-I, A-P), anterior left
    return ax, cor, sag

if args.probe:
    fig, axs = plt.subplots(1, 3, figsize=(9, 4))
    for k, (im, t) in enumerate(zip(planes(mag[("post", "CG-SENSE")]), ("axial", "coronal", "sagittal"))):
        axs[k].imshow(im, cmap="gray", vmin=0, vmax=args.vmax); axs[k].set_title(f"{t} {im.shape}")
        axs[k].set_xticks([]); axs[k].set_yticks([])
    fig.tight_layout(); fig.savefig(OUT / "orient_probe.png", dpi=110)
    log("probe saved"); sys.exit(0)

# ---- layout ---------------------------------------------------------------------------
ROWS = [("pre", 0, "Pre-contrast\nAxial"), ("pre", 1, "Pre-contrast\nCoronal"), ("pre", 2, "Pre-contrast\nSagittal"),
        ("post", 0, "Post-contrast\nAxial"), ("post", 1, "Post-contrast\nCoronal"), ("post", 2, "Post-contrast\nSagittal")]
shapes = planes(mag[("pre", "ZF")])
ncol, nrow = len(COLS), len(ROWS)
COLW = 1.55                                        # inches per column
hr = [shapes[r[1]].shape[0] / shapes[r[1]].shape[1] for r in ROWS]
figw = COLW * ncol + 0.62 + 0.42                   # + row labels + colorbar
figh = COLW * sum(hr) + 0.34
fig = plt.figure(figsize=(figw, figh))
gs = gridspec.GridSpec(nrow, ncol, figure=fig, height_ratios=hr, wspace=0.003, hspace=0.003,
                       left=0.62/figw, right=1 - 0.42/figw, top=1 - 0.30/figh, bottom=0.04/figh)
for ri, (scan, pi, rlab) in enumerate(ROWS):
    for ci, (label, _) in enumerate(COLS):
        ax = fig.add_subplot(gs[ri, ci])
        im = ax.imshow(planes(mag[(scan, label)])[pi], cmap="gray", vmin=0, vmax=args.vmax,
                       interpolation="nearest", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if ri == 0: ax.set_title(label, fontsize=9, pad=3)
        if ci == 0: ax.set_ylabel(rlab, fontsize=8, labelpad=2)
cax = fig.add_axes([1 - 0.34/figw, 0.10, 0.10/figw, 0.78])
cb = fig.colorbar(im, cax=cax)
cb.set_label("normalized signal intensity (a.u.)", fontsize=8)
cb.ax.tick_params(labelsize=7)
for ext in ("png", "pdf"):
    q = OUT / f"ce_pair_recon_3planes.{ext}"
    fig.savefig(q, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
    log(f"saved {q}")
plt.close(fig)
