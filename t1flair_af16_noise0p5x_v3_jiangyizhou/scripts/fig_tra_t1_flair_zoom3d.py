#!/usr/bin/env python3
"""fig_tra_t1_flair_zoom3d.py -- one transverse slice, T1 on top and FLAIR below, each with its error row, a magnified
box in every panel, the sampling mask under the T1 ground truth and the added Gaussian noise as a 3-D surface in the
bottom-left corner (user 2026-09-17, after fig_tra_t1_flair_noise.py).

  python scripts/fig_tra_t1_flair_zoom3d.py --slice 170 [--box J K] [--pdf]

Columns, panel metrics and error maps as fig_tra_t1_flair_noise.py (GT, ZF, CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT,
ZS-SSDU, 3D-DIP, 3D-INR, MC-INR, GS-single, MC-GS; head-masked slice PSNR / SSIM / NMSE, best of each row in bold;
|recon - GT| / max(GT) on jet [0, 0.1] inside the head).
Box: W x W (default 48) on the slice. --box J K gives its centre (axis-1, axis-2 voxel); otherwise it is chosen on this
slice as the window, at least --fill head, with the largest smaller-of-the-two-contrasts lead of MC-GS over the best
other column (GS-single included), lead = 10 log10(SE_other / SE_MC-GS) over the head pixels of the window. The box is
outlined on every panel and magnified into the corner that does not cover it (top-left is left to the metrics).
Left column: GT; under T1 the sampling mask (the k-space plane through the centre; one mask for both contrasts); under
FLAIR the complex Gaussian noise added to the FLAIR measurements, |noisy - clean| / max(GT) of the fully sampled
SENSE-combined images, drawn as a surface whose height is that intensity (jet [0, --noise_vmax]).
Writes t1flair_v3_af16<prefix>_T1FLAIR_tra<slice>_zoom<W>_noise3d_headerr_p3_mcinr_spirit_zf_caps.png (+ .pdf).
"""
import argparse
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--slice", type=int, default=170, help="transverse slice (axis 0)")
ap.add_argument("--inset", type=int, default=48, metavar="W")
ap.add_argument("--box", type=int, nargs=2, default=None, metavar=("J", "K"), help="box centre; default: chosen")
ap.add_argument("--fill", type=float, default=0.85)
ap.add_argument("--stride", type=int, default=4)
ap.add_argument("--noise3d", choices=["slice", "box"], default="box",
                help="surface over the zoom box (default) or over the whole slice (every 2nd pixel)")
ap.add_argument("--noise_vmax", type=float, default=0.01)
ap.add_argument("--mcgs_key", default="gs_joint_p3")
ap.add_argument("--single_key", default="gs_single_p3")
ap.add_argument("--err_vmax", type=float, default=0.1)
ap.add_argument("--png_dpi", type=int, default=130)
ap.add_argument("--pdf", action="store_true")
ap.add_argument("--pdf_dpi", type=int, default=600)
args = ap.parse_args()

import h5py
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.image as mimage
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_opening
from skimage.metrics import structural_similarity

R = Path(__file__).resolve().parents[1]
LEVEL = R.name.split("_")[2]
PREFIX = {"noise2x": "af16n", "noise1x": "af16n1", "noise0p5x": "af16n05", "noise0p25x": "af16n025",
          "noise0p75x": "af16n075"}[LEVEL]
F = Path(f"/home/congcongliu/figs/t1flair_v3_af16_{LEVEL}"); F.mkdir(parents=True, exist_ok=True)
MULT = float(json.load(open(R / "logs/make_noisy_kspace.json"))["multiplier"])
_ex = json.load(open(R / "logs/prep_04_export.json"))
S, W = args.slice, args.inset
METRIC_COLOR = "#FFC23D"
ZC, ZLW = METRIC_COLOR, 2.2
DISK = np.hypot(*np.mgrid[-6:7, -6:7]) <= 6
summary = json.load(open(R / "logs/summary.json"))


def D(x):
    return np.rot90(x.T, 1)                                    # transverse, eyes up (compact figures, --tra_rot 1)


def head_region(m):
    return binary_dilation(binary_fill_holes(binary_opening(binary_fill_holes(m), structure=DISK)), iterations=2)


def columns(c):
    rows = {r["method"]: r for r in summary if r["contrast"] == c}

    def lam_file(key, prefix):
        lam = rows[key]["lam"]
        for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
            m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
            if m and abs(float(m.group(1)) - float(lam)) < 1e-12:
                return p
        raise FileNotFoundError(f"{prefix} {c} lam={lam}")

    sp = rows["spirit"]
    return [("zf", "ZF", R / f"recon/zf_{c}_complex.npy"),
            ("cgsense", "CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"),
            ("cssense_o200", "CS-SENSE", lam_file("cssense_o200", "cssense_o200")),
            ("jcssense", "Joint SENSE", lam_file("jcssense", "jcssense_o200")),
            ("spirit", "L1-SPIRiT", R / f"recon/spirit_ks{sp['ksize']}_lam{sp['lam']:g}_{c}_complex.npy"),
            ("zsssdu", "ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
            ("dip", "3D-DIP", R / f"recon/dip_{c}_complex.npy"),
            ("inr", "3D-INR", R / f"recon/inr_{c}_complex.npy"),
            ("mcinr", "MC-INR", R / f"recon/mcinr_{c}_complex.npy"),
            (args.single_key, "GS-single", R / f"recon/{args.single_key}_{c}_mag.npy"),
            (args.mcgs_key, "MC-GS", R / f"recon/{args.mcgs_key}_{c}_mag.npy")]


def window_sums(a):
    """sums of a over every W x W window, indexed by the window's top-left (axis-1, axis-2) corner"""
    c = np.pad(np.cumsum(np.cumsum(a, axis=0), axis=1), ((1, 0), (1, 0)))
    return c[W:, W:] - c[:-W, W:] - c[W:, :-W] + c[:-W, :-W]


blocks, box_lead = [], {}
for c in ("T1", "FLAIR"):
    gt_c = np.load(R / f"recon/gt_{c}_complex.npy")
    gt = np.abs(gt_c).astype(np.float32)
    gmax = float(gt.max())
    head = gt > 0.05 * float(np.quantile(gt, 0.999))
    g2, m2 = gt[S], head[S]
    g64 = g2.astype(np.float64)
    g = g64[m2]
    cols, ps, ss, nm, se = [], {}, {}, {}, {}
    for key, lab, path in columns(c):
        v = np.abs(np.load(path)[S]).astype(np.float32)
        r = v.astype(np.float64)
        e = r[m2] - g
        ps[key] = 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
        _, smap = structural_similarity(g64, r, data_range=gmax, full=True, win_size=7)
        ss[key] = float(smap[m2].mean())
        nm[key] = float((e * e).sum() / (g * g).sum())
        se[key] = window_sums((r - g64) ** 2 * m2)
        cols.append((key, lab, v))
    fillw = window_sums(m2.astype(np.float64)) / (W * W)
    others = np.stack([se[k] for k, _, _ in columns(c) if k != args.mcgs_key])
    with np.errstate(divide="ignore", invalid="ignore"):
        box_lead[c] = np.where(fillw >= args.fill, 10 * np.log10(others.min(axis=0) / se[args.mcgs_key]), -np.inf)
    blocks.append(dict(c=c, g2=g2, gmax=gmax, emask=head_region(m2), cols=cols, ps=ps, ss=ss, nm=nm, se=se,
                       noise=np.abs(np.load(R / f"recon/fullnoisy_{c}_complex.npy")[S] - gt_c[S]) / gmax,
                       vmax=float(np.quantile(g2, 0.999)),
                       best=dict(psnr=max(ps, key=ps.get), ssim=max(ss, key=ss.get), nmse=min(nm, key=nm.get))))

score = np.minimum(box_lead["T1"], box_lead["FLAIR"])
if args.box is None:
    grid = np.full(score.shape, -np.inf)
    grid[::args.stride, ::args.stride] = score[::args.stride, ::args.stride]
    j0, k0 = (int(t) for t in np.unravel_index(np.argmax(grid), grid.shape))
else:
    j0, k0 = args.box[0] - W // 2, args.box[1] - W // 2
print(f"box {W}x{W}: centre (j, k) = ({j0 + W // 2}, {k0 + W // 2}); lead of MC-GS over the best other column "
      f"T1 {box_lead['T1'][j0, k0]:+.3f} dB, FLAIR {box_lead['FLAIR'][j0, k0]:+.3f} dB")
BOX = np.zeros(blocks[0]["g2"].shape, bool)
BOX[j0:j0 + W, k0:k0 + W] = True
rows_, cols_ = np.nonzero(D(BOX))
ZOOM = dict(x0=int(cols_.min()), y0=int(rows_.min()), w=W)
Hd, Wd = D(BOX).shape

ncol = 12
pw, gap = 2.6, 0.025
h = pw * Hd / Wd
side, pad = 0.38 * pw, 0.012 * pw                             # inset side and margin (inches), as the compact figures


def corner_box(corner):
    """the inset's extent in display pixels (x from the left, y from the bottom: origin lower)"""
    s_px, p_px = side / pw * Wd, pad / pw * Wd
    x0 = p_px if corner[1] == "l" else Wd - p_px - s_px
    y0 = Hd - p_px - s_px if corner[0] == "t" else p_px
    return x0, y0, x0 + s_px, y0 + s_px


def overlaps(corner):
    x0, y0, x1, y1 = corner_box(corner)
    return not (x1 < ZOOM["x0"] or x0 > ZOOM["x0"] + W or y1 < ZOOM["y0"] or y0 > ZOOM["y0"] + W)


CORNER = next((cn for cn in ("tr", "br", "bl") if not overlaps(cn)), "tr")
print(f"inset corner {CORNER}")

top_in, cbpad, cbw, cbroom, CBTRIM = 1.15, 0.06, 0.13, 0.60, 0.10
LM = 0.75                                                     # left margin for the 3-D panel's z labels (inches)
fw = LM + ncol * pw + (ncol - 1) * gap + cbpad + cbw + cbroom
fh = top_in + 4 * h + 3 * gap + 0.16
fig = plt.figure(figsize=(fw, fh))


def axes_at(x0, y0, w, hh):
    a = fig.add_axes([(LM + x0) / fw, 1 - (y0 + hh) / fh, w / fw, hh / fh])
    a.set_xticks([]); a.set_yticks([])
    for s_ in a.spines.values():
        s_.set_visible(False)
    return a


def label(a, text, bold=False, line=0, fs=12, color="white"):
    dy = line * fs * 1.28 / 72 / h if line else 0.0
    t = a.text(0.025, 0.975 - dy, text, transform=a.transAxes, ha="left", va="top", color=color, fontsize=fs,
               fontweight="bold" if bold else "normal")
    t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])


def put_inset(parent, img, x_in, y_in, **imkw):
    z = ZOOM
    parent.add_patch(Rectangle((z["x0"] - 0.5, z["y0"] - 0.5), z["w"], z["w"], fill=False, ec=ZC, lw=ZLW))
    left = LM + x_in + (pad if CORNER[1] == "l" else pw - pad - side)
    top = y_in + (pad if CORNER[0] == "t" else h - pad - side)
    a = fig.add_axes([left / fw, 1 - (top + side) / fh, side / fw, side / fh])
    a.set_xticks([]); a.set_yticks([])
    for sp_ in a.spines.values():
        sp_.set_visible(True); sp_.set_edgecolor(ZC); sp_.set_linewidth(ZLW + 0.6)
    a.imshow(img[z["y0"]:z["y0"] + z["w"], z["x0"]:z["x0"] + z["w"]], origin="lower", aspect="auto", **imkw)


with h5py.File(R / "data/mask_af16_3d.h5") as f_:
    mask3 = np.array(f_["mask"]) > 0
kmask = mask3[mask3.shape[0] // 2]                             # readout fully sampled: every kx plane is the same

y = top_in
for bi, b in enumerate(blocks):
    for j in range(ncol):
        x = j * (pw + gap)
        a = axes_at(x, y, pw, h)
        if j == 0:
            gim = a.imshow(D(b["g2"]), cmap="gray", vmin=0, vmax=b["vmax"], origin="lower", aspect="auto")
            put_inset(a, D(b["g2"]), x, y, cmap="gray", vmin=0, vmax=b["vmax"])
            label(a, b["c"], bold=True, fs=14)
            if bi == 0:
                a.set_title("GT", fontsize=14, pad=4)
                e = axes_at(x, y + h + gap, pw, h)
                e.imshow(D(kmask), cmap="gray", vmin=0, vmax=1, origin="lower", aspect="auto", interpolation="nearest")
                label(e, "sampling mask", bold=True, fs=12)
                label(e, "(both contrasts)", line=1, fs=10.5)
            else:                                              # bottom-left corner: the added noise as a surface
                # the z labels sit left of the projected box: keep the axes clear of the figure edge, title on top
                ax3 = fig.add_axes([(LM + x - 0.04 * pw) / fw, 1 - (y + h + gap + h) / fh, 0.92 * pw / fw, 0.82 * h / fh],
                                   projection="3d")
                Z = np.clip(D(b["noise"]), 0, args.noise_vmax)
                if args.noise3d == "box":
                    Z = Z[ZOOM["y0"]:ZOOM["y0"] + W, ZOOM["x0"]:ZOOM["x0"] + W]
                else:
                    Z = Z[::2, ::2]
                Yg, Xg = np.mgrid[0:Z.shape[0], 0:Z.shape[1]]
                surf = ax3.plot_surface(Xg, Yg, Z, cmap="jet", vmin=0, vmax=args.noise_vmax, rstride=1, cstride=1,
                                        linewidth=0, antialiased=False, rasterized=True)
                ax3.set_zlim(0, args.noise_vmax)
                ax3.set_xticks([]); ax3.set_yticks([])
                ax3.set_zticks([0, args.noise_vmax / 2, args.noise_vmax])
                ax3.set_zticklabels(["0", f"{args.noise_vmax / 2:g}", f"{args.noise_vmax:g}"])
                ax3.tick_params(axis="z", labelsize=8.5, pad=0)
                ax3.set_zlabel("intensity", fontsize=10, labelpad=2)
                ax3.view_init(elev=35, azim=30)                 # z axis on the left, clear of the next panel
                ax3.set_box_aspect((Z.shape[1], Z.shape[0], 0.7 * max(Z.shape)), zoom=1.1)
                where = f"in the {W}x{W} box" if args.noise3d == "box" else "whole slice"
                fig.text((LM + x + pw / 2) / fw, 1 - (y + h + gap + 0.06) / fh,
                         f"added noise, {b['c']} ({MULT:g}σ0)\n|noisy - clean| / max(GT)\n{where}",
                         ha="center", va="top", fontsize=10.5)
            continue
        e = axes_at(x, y + h + gap, pw, h)
        key, lab, v = b["cols"][j - 1]
        a.imshow(D(v), cmap="gray", vmin=0, vmax=b["vmax"], origin="lower", aspect="auto")
        put_inset(a, D(v), x, y, cmap="gray", vmin=0, vmax=b["vmax"])
        for i, (txt, met) in enumerate(((f"{b['ps'][key]:.2f} dB", "psnr"), (f"SSIM {b['ss'][key]:.4f}", "ssim"),
                                         (f"NMSE {b['nm'][key]:.5f}", "nmse"))):
            label(a, txt, bold=key == b["best"][met], line=i, fs=10.5, color=METRIC_COLOR)
        emap = D(np.abs(v - b["g2"]) / b["gmax"] * b["emask"])
        im = e.imshow(emap, cmap="jet", vmin=0.0, vmax=args.err_vmax, origin="lower", aspect="auto")
        put_inset(e, emap, x, y + h + gap, cmap="jet", vmin=0.0, vmax=args.err_vmax)
        if bi == 0:
            a.set_title(lab if "SPIRiT" in lab else lab.upper(), fontsize=14, pad=4)
    x_cb = (LM + ncol * pw + (ncol - 1) * gap + cbpad) / fw
    for handle, tp, tk in ((gim, y, MaxNLocator(5)), (im, y + h + gap + CBTRIM, np.linspace(0, args.err_vmax, 5))):
        cax = fig.add_axes([x_cb, 1 - (tp + h - CBTRIM) / fh, cbw / fw, (h - CBTRIM) / fh])
        cb = fig.colorbar(handle, cax=cax, ticks=tk)
        cb.ax.tick_params(labelsize=9.5)
    y += 2 * h + 2 * gap

NPL = int(_ex["mask2d_shape"][0]) * int(_ex["mask2d_shape"][1])
fl = _ex["contrasts"]["FLAIR"]
_CN = {"bl": "bottom-left", "br": "bottom-right", "tr": "top-right"}
fig.suptitle(f"jiangyizhou  nominal AF 16, actual {_ex['mask2d_AF_rect']:.2f}x, measurements + {MULT:g}σ0 noise  --  "
             f"Transverse {S}  (T1 top, FLAIR bottom)", fontsize=13, fontweight="bold", y=1 - 0.10 / fh, va="top")
lines = ["head-masked slice PSNR / SSIM (7x7 window) / NMSE in each panel, the best of each row in bold;   "
         f"error |recon - GT| / max(GT), jet [0, {args.err_vmax:g}]   (inside the head only);   "
         f"the {W}x{W} box outlined on the slice is magnified into the {_CN[CORNER]} corner",
         "left column: under T1 the sampling mask (k-space, one mask for both contrasts); bottom-left: the complex "
         f"Gaussian noise added to the FLAIR measurements, |noisy - clean| / max(GT) of the fully sampled images, "
         f"as a surface (height = intensity, jet [0, {args.noise_vmax:g}]"
         + (", whole slice, every 2nd pixel)" if args.noise3d == "slice" else f", the {W}x{W} box)"),
         f"AF = {NPL} plane points / {_ex['mask2d_samples']} sampled points, the same on both contrasts;   "
         f"the FLAIR scan acquired {round(fl['support_frac'] * NPL)} of the {NPL} lines, so "
         f"{round(fl['sampled_frac_of_plane'] * NPL)} sampled points carry FLAIR data;   grey bars: each contrast's window"]
fig.text(0.5, 1 - 0.30 / fh, "\n".join(lines), ha="center", va="top", fontsize=10)
name = (f"t1flair_v3_{PREFIX}_T1FLAIR_tra{S}_zoom{W}_noise3d" + ("" if args.noise3d == "box" else "slice")
        + "_headerr_p3_mcinr_spirit_zf_caps.png")
for d in (F, R / "figures"):
    fig.savefig(d / name, dpi=args.png_dpi)
    if args.pdf:
        for im_ in fig.findobj(mimage.AxesImage):
            im_.set_rasterized(True)
        fig.savefig(d / name.replace(".png", ".pdf"), dpi=args.pdf_dpi)
print("wrote", F / name)
for b in blocks:
    se = {k: float(v[j0, k0]) for k, v in b["se"].items()}
    best_other = min((k for k in se if k != args.mcgs_key), key=se.get)
    print(f"  {b['c']:5s}: slice PSNR first {b['best']['psnr']}, SSIM first {b['best']['ssim']}, NMSE first "
          f"{b['best']['nmse']}; in the box MC-GS SE {se[args.mcgs_key]:.4g}, best other {best_other} {se[best_other]:.4g}")
