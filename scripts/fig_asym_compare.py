"""Image comparison for the asymmetric-acceleration probe: FLAIR at a harsh mask, GS-single against MC-GS with a T1
partner at AF 15.19 and with a fully sampled T1 partner (oracle). Same conventions as the jyz figures of record
(fig_three_planes_compact.py): planes 180 / 124 / 76, transverse eyes-up, per-plane grey window at the GT's p99.9 (the
sagittal one matched by the head median), in-panel head-only slice PSNR / SSIM (7x7 uniform) / NMSE in orange-yellow with
the best of each plane in bold, error |recon - GT| / max(GT) in jet [0, 0.1] inside the head only, a 48-voxel box
magnified in a corner, and the sampling mask of that plane left of each error row.

usage: fig_asym_compare.py LEVEL      LEVEL = 32 (accelx 8, actual 26.68x) or 64 (accelx 16, actual 43.32x)
"""
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_opening, uniform_filter
from skimage.metrics import structural_similarity

matplotlib.rcParams["pdf.fonttype"] = 42
LEVEL = int(sys.argv[1])
ROOT = Path(__file__).resolve().parents[1]
RA = ROOT / "results_arc"
TOPIC = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou")
MASK = ROOT / f"data/mask_af{LEVEL}_probe_3d.h5"
FL = {32: "fl32", 64: "fl64"}[LEVEL]
ARMS = [("GS-SINGLE", f"wd_va_s_{FL}", "recT1"),                      # single arm: FLAIR sits in the plain channel
        ("MC-GS, T1 AF 15.19", f"wd_va_j_t16_{FL}", "recFL"),
        ("MC-GS, T1 FULL (ORACLE)", f"wd_va_j_tfull_{FL}", "recFL")]
SLICES, INSET, CORNER = (180, 124, 76), 48, ("tr", "bl", "bl")
METRIC_COLOR, ZC, ZLW, ERR_VMAX = "#FFC23D", "#FFC23D", 1.2, 0.1

gt = np.abs(np.load(TOPIC / "recon/gt_FLAIR_complex.npy")).astype(np.float32)
gmax = float(gt.max())
head = gt > 0.05 * float(np.quantile(gt, 0.999))
vols = {}
for lab, exp, key in ARMS:
    with np.load(RA / f"{exp}_volumes.npz") as z:
        vols[lab] = np.abs(np.asarray(z[key])).astype(np.float32)
    assert vols[lab].shape == gt.shape, (lab, vols[lab].shape)
with h5py.File(MASK) as f:
    mask3 = np.array(f["mask"]) > 0
npts = int(mask3[0].sum())
af_actual = mask3[0].size / npts
dens = uniform_filter(mask3[0].astype(float), size=9, mode="constant")
kyc, kzc = (int(v) for v in np.unravel_index(np.argmax(dens), dens.shape))
kcen = (mask3.shape[0] // 2, kyc, kzc)
DISK = np.hypot(*np.mgrid[-6:7, -6:7]) <= 6


def head_region(m):
    return binary_dilation(binary_fill_holes(binary_opening(binary_fill_holes(m), structure=DISK)), iterations=2)


def D(x, axis):
    return np.rot90(x.T, 1) if axis == 0 else x


BOX = np.zeros(gt.shape, bool)
_h = INSET // 2
BOX[SLICES[0] - _h:SLICES[0] + _h, SLICES[1] - _h:SLICES[1] + _h, SLICES[2] - _h:SLICES[2] + _h] = True
planes = []
for axis, idx in enumerate(SLICES):
    sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
    g2, m2 = gt[sl], head[sl]
    g64 = g2.astype(np.float64); g = g64[m2]
    ps, ss, nm = {}, {}, {}
    for lab, _, _ in ARMS:
        r = vols[lab][sl].astype(np.float64)
        e = r[m2] - g
        ps[lab] = 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
        _, smap = structural_similarity(g64, r, data_range=gmax, full=True, win_size=7)
        ss[lab] = float(smap[m2].mean())
        nm[lab] = float((e * e).sum() / (g * g).sum())
    bm = D(BOX[sl], axis)
    rr, cc = np.nonzero(bm)
    planes.append(dict(axis=axis, idx=idx, sl=sl, g2=g2, emask=head_region(m2), ps=ps, ss=ss, nm=nm,
                       zoom=dict(x0=int(cc.min()), y0=int(rr.min()), w=INSET),
                       best=dict(psnr=max(ps, key=ps.get), ssim=max(ss, key=ss.get), nmse=min(nm, key=nm.get)),
                       vmax=float(np.quantile(g2, 0.999)), med=float(np.median(g2[m2])),
                       kmask=mask3.take(kcen[axis], axis=axis)))
ratio = float(np.mean([p["med"] / p["vmax"] for p in planes if p["axis"] != 2]))
for p in planes:
    if p["axis"] == 2:
        p["vmax"] = p["med"] / ratio

pw, gap, top_in = 2.6, 0.025, 1.25
ncol = len(ARMS) + 1
row_h = []
for p in planes:
    Hh, Ww = D(p["g2"], p["axis"]).shape
    row_h += [pw * Hh / Ww] * 2
fw = ncol * pw + (ncol - 1) * gap + 0.9
fh = top_in + sum(row_h) + gap * (len(row_h) - 1) + 0.15
fig = plt.figure(figsize=(fw, fh), facecolor="black")


def axes_at(x0, y0, w, h):
    a = fig.add_axes([x0 / fw, 1 - (y0 + h) / fh, w / fw, h / fh])
    a.set_xticks([]); a.set_yticks([]); a.set_facecolor("black")
    for sp in a.spines.values():
        sp.set_visible(False)
    return a


def put_inset(parent, img, z, x0_in, y0_in, h_in, corner, **imkw):
    parent.add_patch(Rectangle((z["x0"], z["y0"]), z["w"], z["w"], fill=False, ec=ZC, lw=ZLW))
    side, pad = 0.38 * pw, 0.012 * pw
    left = x0_in + (pad if corner in ("bl", "tl") else pw - pad - side)
    top = y0_in + (pad if corner in ("tl", "tr") else h_in - pad - side)
    a = fig.add_axes([left / fw, 1 - (top + side) / fh, side / fw, side / fh])
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_edgecolor(ZC); sp.set_linewidth(ZLW + 0.6)
    a.imshow(img[z["y0"]:z["y0"] + z["w"], z["x0"]:z["x0"] + z["w"]], origin="lower", aspect="auto", **imkw)


y = top_in
for k, p in enumerate(planes):
    ax_ = p["axis"]
    h = row_h[2 * k]
    for j in range(ncol):
        x = j * (pw + gap)
        a = axes_at(x, y, pw, h)
        b = axes_at(x, y + h + gap, pw, h)
        if j == 0:
            a.imshow(D(p["g2"], ax_), cmap="gray", vmin=0, vmax=p["vmax"], origin="lower", aspect="auto")
            put_inset(a, D(p["g2"], ax_), p["zoom"], x, y, h, CORNER[ax_], cmap="gray", vmin=0, vmax=p["vmax"])
            b.imshow(D(p["kmask"], ax_), cmap="gray", vmin=0, vmax=1, origin="lower", aspect="auto",
                     interpolation="nearest")
            a.text(0.03, 0.97, ("Transverse", "Coronal", "Sagittal")[ax_], transform=a.transAxes, color="white",
                   fontsize=11, fontweight="bold", va="top")
            if k == 0:
                a.set_title("GT (FLAIR)", fontsize=15, color="white", pad=4)
            continue
        lab = ARMS[j - 1][0]
        v = vols[lab][p["sl"]]
        a.imshow(D(v, ax_), cmap="gray", vmin=0, vmax=p["vmax"], origin="lower", aspect="auto")
        put_inset(a, D(v, ax_), p["zoom"], x, y, h, CORNER[ax_], cmap="gray", vmin=0, vmax=p["vmax"])
        for i, (txt, met) in enumerate(((f"{p['ps'][lab]:.2f} dB", "psnr"), (f"SSIM {p['ss'][lab]:.4f}", "ssim"),
                                         (f"NMSE {p['nm'][lab]:.5f}", "nmse"))):
            a.text(0.03, 1 - (0.06 + 0.2 * i) / h, txt, transform=a.transAxes, color=METRIC_COLOR, fontsize=10.5, va="top",
                   fontweight="bold" if lab == p["best"][met] else "normal")
        emap = D(np.abs(v - p["g2"]) / gmax * p["emask"], ax_)
        im = b.imshow(emap, cmap="jet", vmin=0.0, vmax=ERR_VMAX, origin="lower", aspect="auto")
        put_inset(b, emap, p["zoom"], x, y + h + gap, h, CORNER[ax_], cmap="jet", vmin=0.0, vmax=ERR_VMAX)
        if k == 0:
            a.set_title(lab, fontsize=15, color="white", pad=4)
        if k == 0 and j == ncol - 1:
            cax = fig.add_axes([(x + pw + 0.12) / fw, 1 - (y + 2 * h + gap) / fh, 0.13 / fw, h / fh])
            cb = fig.colorbar(im, cax=cax)
            cb.set_ticks([0, 0.05, 0.1]); cb.ax.tick_params(colors="white", labelsize=9)
    y += 2 * h + 2 * gap
nominal = {32: 32, 64: 64}[LEVEL]
fig.text(0.5, 1 - 0.12 / fh, f"FLAIR  jiangyizhou  nominal AF {nominal}, actual {af_actual:.2f}x  ({npts} of "
         f"{mask3[0].size} PE points)", ha="center", va="top", color="white", fontsize=13, fontweight="bold")
fig.text(0.5, 1 - 0.36 / fh, "T1 partner at AF 15.19 (mask of the AF 16 topic) or fully sampled (oracle); GS-single sees "
         "FLAIR alone", ha="center", va="top", color="white", fontsize=10)
fig.text(0.5, 1 - 0.56 / fh, "head-only slice PSNR / SSIM (7x7) / NMSE, best of each plane in bold;  error |recon - GT| "
         "/ max(GT), jet [0, 0.1], inside the head", ha="center", va="top", color="white", fontsize=8.5)
fig.text(0.5, 1 - 0.72 / fh, "48-voxel box magnified in a corner;  left of each error row: the sampling mask in that "
         "plane (readout fully sampled)", ha="center", va="top", color="white", fontsize=8.5)
out = ROOT / f"figures/widen/asym_flair_actual{af_actual:.2f}x_compare.png"
fig.savefig(out, dpi=130, facecolor="black")
print("wrote", out)
for p in planes:
    print(f"  {('transverse', 'coronal', 'sagittal')[p['axis']]} {p['idx']}: " +
          "  ".join(f"{lab} {p['ps'][lab]:.2f}/{p['ss'][lab]:.4f}" for lab, _, _ in ARMS))
