#!/usr/bin/env python3
"""fig_tra_t1_flair_noise.py -- one transverse slice, T1 on top and FLAIR below, each with its error row; the left cell
of each error row shows the complex Gaussian noise that was added to that contrast's measurements (user 2026-09-17).

  python scripts/fig_tra_t1_flair_noise.py --slice 170 [--pdf]

Columns as fig_three_planes_compact.py with --with_zf --l1_arm spirit --caps (GT, ZF, CG-SENSE, CS-SENSE at its selected
lambda, Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR, GS-single, MC-GS) and the same panel gauge: head mask
|GT| > 0.05 * P99.9 of the volume; PSNR at the volume's max|GT| with the MSE over the head pixels of the slice; SSIM =
7x7 uniform-window map averaged over the head pixels; NMSE over the head pixels; the best of each row in bold. Error
maps |recon - GT| / max(GT), jet [0, 0.1], inside the head. The noise cell: |noisy - clean| / max(GT) of the fully
sampled SENSE-combined images (recon/fullnoisy_<c>_complex.npy - recon/gt_<c>_complex.npy, both in gauge units), the
whole field of view, same jet scale; it replaces the sampling-mask cell of the compact figures. Display as the compact
figures: axis 0 transposed and turned eyes up, origin lower.
Writes t1flair_v3_af16<prefix>_T1FLAIR_tra<slice>_noisecell_headerr_p3_mcinr_spirit_zf_caps.png to the level's served
figs folder and figures/ (and a vector PDF with --pdf).
"""
import argparse
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--slice", type=int, default=170, help="transverse slice (axis 0)")
ap.add_argument("--mcgs_key", default="gs_joint_p3")
ap.add_argument("--single_key", default="gs_single_p3")
ap.add_argument("--err_vmax", type=float, default=0.1)
ap.add_argument("--noise_vmax", type=float, default=None,
                help="top of the noise cell's jet scale (default: the error scale). 0.01 = ten times amplified, with its own\n"
                     "small colorbar inside the cell and _noisescale<v> in the file name; the method error maps keep [0, err_vmax]")
ap.add_argument("--png_dpi", type=int, default=130)
ap.add_argument("--pdf", action="store_true")
ap.add_argument("--pdf_dpi", type=int, default=600)
args = ap.parse_args()
NV = args.noise_vmax if args.noise_vmax is not None else args.err_vmax

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.image as mimage
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_opening
from skimage.metrics import structural_similarity

R = Path(__file__).resolve().parents[1]
LEVEL = R.name.split("_")[2]                                   # noise0p5x, noise1x, ...
PREFIX = {"noise2x": "af16n", "noise1x": "af16n1", "noise0p5x": "af16n05", "noise0p25x": "af16n025",
          "noise0p75x": "af16n075"}[LEVEL]
F = Path(f"/home/congcongliu/figs/t1flair_v3_af16_{LEVEL}"); F.mkdir(parents=True, exist_ok=True)
MULT = float(json.load(open(R / "logs/make_noisy_kspace.json"))["multiplier"])
_ex = json.load(open(R / "logs/prep_04_export.json"))
S = args.slice
METRIC_COLOR = "#FFC23D"
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


blocks = []
for c in ("T1", "FLAIR"):
    gt_c = np.load(R / f"recon/gt_{c}_complex.npy")
    gt = np.abs(gt_c).astype(np.float32)
    gmax = float(gt.max())
    head = gt > 0.05 * float(np.quantile(gt, 0.999))
    g2, m2 = gt[S], head[S]
    g64 = g2.astype(np.float64)
    g = g64[m2]
    noise2 = np.abs(np.load(R / f"recon/fullnoisy_{c}_complex.npy")[S] - gt_c[S]) / gmax
    cols, ps, ss, nm = [], {}, {}, {}
    for key, lab, path in columns(c):
        v = np.abs(np.load(path)[S]).astype(np.float32)
        r = v.astype(np.float64)
        e = r[m2] - g
        ps[key] = 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
        _, smap = structural_similarity(g64, r, data_range=gmax, full=True, win_size=7)
        ss[key] = float(smap[m2].mean())
        nm[key] = float((e * e).sum() / (g * g).sum())
        cols.append((key, lab, v))
    blocks.append(dict(c=c, g2=g2, gmax=gmax, emask=head_region(m2), noise=noise2,
                       noise_rms=float(np.sqrt(np.mean(noise2 ** 2))),   # rms of |noisy - clean| / max(GT) over the slice
                       cols=cols, ps=ps, ss=ss, nm=nm, vmax=float(np.quantile(g2, 0.999)),
                       best=dict(psnr=max(ps, key=ps.get), ssim=max(ss, key=ss.get), nmse=min(nm, key=nm.get))))

ncol = 12
pw, gap = 2.6, 0.025
H, W = D(blocks[0]["g2"]).shape
h = pw * H / W
top_in, cbpad, cbw, cbroom, CBTRIM = 1.15, 0.06, 0.13, 0.60, 0.10
fw = ncol * pw + (ncol - 1) * gap + cbpad + cbw + cbroom
fh = top_in + 4 * h + 3 * gap + 0.16
fig = plt.figure(figsize=(fw, fh))


def axes_at(x0, y0, w, hh):
    a = fig.add_axes([x0 / fw, 1 - (y0 + hh) / fh, w / fw, hh / fh])
    a.set_xticks([]); a.set_yticks([])
    for s_ in a.spines.values():
        s_.set_visible(False)
    return a


def label(a, text, bold=False, line=0, fs=12, color="white", va_top=True):
    dy = line * fs * 1.28 / 72 / h if line else 0.0
    y = (0.975 - dy) if va_top else (0.025 + dy)
    t = a.text(0.025, y, text, transform=a.transAxes, ha="left", va="top" if va_top else "bottom", color=color,
               fontsize=fs, fontweight="bold" if bold else "normal")
    t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])


y = top_in
for bi, b in enumerate(blocks):
    for j in range(ncol):
        x = j * (pw + gap)
        a = axes_at(x, y, pw, h)
        e = axes_at(x, y + h + gap, pw, h)
        if j == 0:
            gim = a.imshow(D(b["g2"]), cmap="gray", vmin=0, vmax=b["vmax"], origin="lower", aspect="auto")
            label(a, b["c"], bold=True, fs=14)
            nim = e.imshow(D(b["noise"]), cmap="jet", vmin=0.0, vmax=NV, origin="lower", aspect="auto")
            label(e, "added noise", bold=True, fs=12)
            label(e, f"{MULT:g}σ0 per sample", line=1, fs=10.5, color=METRIC_COLOR)
            label(e, f"rms {b['noise_rms']:.4f}", line=2, fs=10.5, color=METRIC_COLOR)
            if NV != args.err_vmax:
                label(e, f"colour scale [0, {NV:g}]", line=3, fs=10.5, color="white")
                ncax = fig.add_axes([(x + 0.10 * pw) / fw, 1 - (y + h + gap + 0.93 * h) / fh, 0.80 * pw / fw, 0.035 * h / fh])
                ncb = fig.colorbar(nim, cax=ncax, orientation="horizontal", ticks=[0, NV / 2, NV])
                ncb.ax.tick_params(labelsize=9, colors="white")
                for t_ in ncb.ax.get_xticklabels():
                    t_.set_path_effects([pe.withStroke(linewidth=2.0, foreground="black")])
            if bi == 0:
                a.set_title("GT", fontsize=14, pad=4)
            continue
        key, lab, v = b["cols"][j - 1]
        a.imshow(D(v), cmap="gray", vmin=0, vmax=b["vmax"], origin="lower", aspect="auto")
        for i, (txt, met) in enumerate(((f"{b['ps'][key]:.2f} dB", "psnr"), (f"SSIM {b['ss'][key]:.4f}", "ssim"),
                                         (f"NMSE {b['nm'][key]:.5f}", "nmse"))):
            label(a, txt, bold=key == b["best"][met], line=i, fs=10.5, color=METRIC_COLOR)
        im = e.imshow(D(np.abs(v - b["g2"]) / b["gmax"] * b["emask"]), cmap="jet", vmin=0.0, vmax=args.err_vmax,
                      origin="lower", aspect="auto")
        if bi == 0:
            a.set_title(lab if "SPIRiT" in lab else lab.upper(), fontsize=14, pad=4)
    x_cb = (ncol * pw + (ncol - 1) * gap + cbpad) / fw
    for handle, tp, tk in ((gim, y, MaxNLocator(5)), (im, y + h + gap + CBTRIM, np.linspace(0, args.err_vmax, 5))):
        cax = fig.add_axes([x_cb, 1 - (tp + h - CBTRIM) / fh, cbw / fw, (h - CBTRIM) / fh])
        cb = fig.colorbar(handle, cax=cax, ticks=tk)
        cb.ax.tick_params(labelsize=9.5)
    y += 2 * h + 2 * gap

NPL = int(_ex["mask2d_shape"][0]) * int(_ex["mask2d_shape"][1])
fl = _ex["contrasts"]["FLAIR"]
fig.suptitle(f"jiangyizhou  nominal AF 16, actual {_ex['mask2d_AF_rect']:.2f}x, measurements + {MULT:g}σ0 noise  --  "
             f"Transverse {S}  (T1 top, FLAIR bottom)", fontsize=13, fontweight="bold", y=1 - 0.10 / fh, va="top")
lines = ["head-masked slice PSNR / SSIM (7x7 window) / NMSE in each panel, the best of each row in bold;   "
         f"error |recon - GT| / max(GT), jet [0, {args.err_vmax:g}]   (inside the head only)",
         "left of each error row: the complex Gaussian noise added to that contrast's measurements, shown as "
         "|noisy - clean| / max(GT) of the fully sampled SENSE-combined images, whole field of view, "
         + ("same scale" if NV == args.err_vmax else f"on jet [0, {NV:g}] ({args.err_vmax / NV:g} times the error scale)"),
         f"AF = {NPL} plane points / {_ex['mask2d_samples']} sampled points, the same on both contrasts (one mask);   "
         f"the FLAIR scan acquired {round(fl['support_frac'] * NPL)} of the {NPL} lines, so "
         f"{round(fl['sampled_frac_of_plane'] * NPL)} sampled points carry FLAIR data;   "
         "grey bars: each contrast's window"]
fig.text(0.5, 1 - 0.30 / fh, "\n".join(lines), ha="center", va="top", fontsize=10)
name = (f"t1flair_v3_{PREFIX}_T1FLAIR_tra{S}_noisecell" + ("" if NV == args.err_vmax else f"_noisescale{NV:g}")
        + "_headerr_p3_mcinr_spirit_zf_caps.png")
for d in (F, R / "figures"):
    fig.savefig(d / name, dpi=args.png_dpi)
    if args.pdf:
        for im_ in fig.findobj(mimage.AxesImage):
            im_.set_rasterized(True)
        fig.savefig(d / name.replace(".png", ".pdf"), dpi=args.pdf_dpi)
print("wrote", F / name)
for b in blocks:
    order = sorted(b["ps"], key=lambda k: -b["ps"][k])
    print(f"  {b['c']:5s} tra {S}: PSNR first {order[0]} ({b['ps'][order[0]]:.3f}), second {order[1]} "
          f"({b['ps'][order[1]]:.3f}); SSIM first {b['best']['ssim']}; NMSE first {b['best']['nmse']}; "
          f"noise rms {b['noise_rms']:.4f} x max(GT)")
