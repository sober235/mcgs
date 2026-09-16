#!/usr/bin/env python3
"""fig_masks_transverse_stack.py -- the three sampling patterns of the diff-mask study stacked in ONE figure.

  python scripts/fig_masks_transverse_stack.py --contrast T1 [--tra 180] [--pdf]

Rows, top to bottom: vd-Poisson (the AF16 topic, the comparison column of the study), Gaussian, pseudo-radial. Each
pattern contributes one transverse slice (default 180, the figure-of-record slice, eyes up) as an image row
(GT + ZF + the ten methods of the 2026-09-16 figure set) over an error row (|recon - GT| / max|GT|, jet, fixed
[0, --err_vmax], inside the head only). User request 2026-09-16: every image row carries a grey colorbar and every
error row a jet colorbar at its right end. The left cell of each error row is that pattern's (ky, kz) sampling mask.
Method names as the 2026-09-16 set (ZF right after GT, upper case except SPIRiT); each panel carries its head-masked
slice PSNR / SSIM / NMSE (the best of each row in bold). Metrics as fig_three_planes_compact.py: head mask
|GT| > 0.05 * P99.9 of the volume, PSNR at the volume's max|GT|, SSIM = 2-D map (7x7 uniform window, data range
max|GT|) averaged over the head pixels, NMSE = sum (recon - GT)^2 / sum GT^2 over the head pixels. Grey window = the
slice's P99.9 of the GT (the GT is the same volume in all three topics, so the three rows share one window).
Reads the three topics only; writes to this (Gaussian) topic's figures/, the home of the cross-pattern comparison,
and to /home/congcongliu/figs/t1flair_v3_af16_diffmask/ (served at http://localhost:8765/t1flair_v3_af16_diffmask/).
"""
import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mimage  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_opening  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", required=True, choices=["T1", "FLAIR"])
ap.add_argument("--tra", type=int, default=180, help="transverse slice (axis 0); 180 = the figure-of-record slice")
ap.add_argument("--tra_rot", type=int, default=1, choices=[0, 1, 2, 3], help="quarter turns after the transpose; 1 = eyes up")
ap.add_argument("--err_vmax", type=float, default=0.1, help="top of the fixed jet scale, as a fraction of max|GT|")
ap.add_argument("--gap", type=float, default=0.025, help="gap between neighbouring panels, inches")
ap.add_argument("--ssim", choices=["uniform", "gaussian"], default="uniform")
ap.add_argument("--mcgs_key", default="gs_joint_p3", help="registered joint arm in the MC-GS column")
ap.add_argument("--single_key", default="gs_single_p3", help="registered single arm in the GS-single column")
ap.add_argument("--no_caps", action="store_true", help="method names as registered instead of upper case")
ap.add_argument("--no_metrics", action="store_true")
ap.add_argument("--png_dpi", type=int, default=200)
ap.add_argument("--pdf", action="store_true", help="also write a vector PDF (text vector, panels rasterised at --pdf_dpi)")
ap.add_argument("--pdf_dpi", type=int, default=300)
ap.add_argument("--tag", default="_p3_zf_caps", help="suffix of the output file name")
args = ap.parse_args()
c = args.contrast
CAPS = not args.no_caps

RES = Path("/data0/congcong/code/GS/Results_20260908")
TOPICS = [("vd-Poisson", RES / "T1FLAIR_AF16_v3_jiangyizhou"),
          ("Gaussian", RES / "T1FLAIR_AF16_gauss_v3_jiangyizhou"),
          ("Pseudo-radial", RES / "T1FLAIR_AF16_radial_v3_jiangyizhou")]
OUT = RES / "T1FLAIR_AF16_gauss_v3_jiangyizhou" / "figures"
F = Path("/home/congcongliu/figs/t1flair_v3_af16_diffmask")
F.mkdir(parents=True, exist_ok=True)
METRIC_COLOR = "#FFC23D"
SSIM_KW = (dict(win_size=7) if args.ssim == "uniform"
           else dict(gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
DISK = np.hypot(*np.mgrid[-6:7, -6:7]) <= 6


def arms_for(R: Path):
    """the twelve columns of one topic: (key, label, saved volume), swept arms at the lambda summarize.py kept"""
    rows = {r["method"]: r for r in json.load(open(R / "logs/summary.json")) if r["contrast"] == c}

    def lam_file(row_key, prefix):
        lam = rows.get(row_key, {}).get("lam")
        if lam is None:
            return None
        for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
            m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
            if m and abs(float(m.group(1)) - float(lam)) < 1e-12:
                return p
        return None

    def spirit_file():
        r = rows.get("spirit", {})
        if r.get("lam") is None or not r.get("ksize"):
            return None
        p = R / f"recon/spirit_ks{r['ksize']}_lam{r['lam']:g}_{c}_complex.npy"
        return p if p.exists() else None

    arms = [("zf", "ZF", R / f"recon/zf_{c}_complex.npy"),
            ("cgsense", "CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"),
            ("cssense_o200", "CS-SENSE", lam_file("cssense_o200", "cssense_o200")),
            ("jcssense", "Joint SENSE", lam_file("jcssense", "jcssense_o200")),
            ("spirit", "L1-SPIRiT", spirit_file()),
            ("zsssdu", "ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
            ("dip", "3D-DIP", R / f"recon/dip_{c}_complex.npy"),
            ("inr", "3D-INR", R / f"recon/inr_{c}_complex.npy"),
            ("mcinr", "MC-INR", R / f"recon/mcinr_{c}_complex.npy"),
            (args.single_key, "GS-single", R / f"recon/{args.single_key}_{c}_mag.npy"),
            (args.mcgs_key, "MC-GS", R / f"recon/{args.mcgs_key}_{c}_mag.npy")]
    missing = [lab for _, lab, p in arms if p is None or not Path(p).exists()]
    assert not missing, f"{R.name}: missing volumes for {c}: {missing}"
    return arms


def D(x):
    """display transform of a transverse slice: transpose, then --tra_rot quarter turns (1 = eyes up)"""
    return np.rot90(x.T, args.tra_rot)


def head_region(m):
    """error display region: the filled head mask without thin protrusions, opened with a radius-6 disk, plus a 2-px rim"""
    return binary_dilation(binary_fill_holes(binary_opening(binary_fill_holes(m), structure=DISK)), iterations=2)


def slice_mag(path):
    v = np.load(path, mmap_mode="r")
    return np.abs(np.asarray(v[args.tra])).astype(np.float32)


ROWS = []
gmax_ref = None
for name, R in TOPICS:
    arms = arms_for(R)
    gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
    gmax = float(gt.max())
    head = gt > 0.05 * float(np.quantile(gt, 0.999))
    if gmax_ref is None:
        gmax_ref = gmax
    assert abs(gmax - gmax_ref) <= 1e-6 * gmax_ref, f"{R.name}: GT differs from the first topic's GT (max {gmax} vs {gmax_ref})"
    g2, m2 = gt[args.tra], head[args.tra]
    g64 = g2.astype(np.float64)
    g = g64[m2]
    vols, ps, ss, nm = {}, {}, {}, {}
    for key, _, path in arms:
        v = slice_mag(path)
        vols[key] = v
        r = v.astype(np.float64)
        e = r[m2] - g
        ps[key] = 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
        _, smap = structural_similarity(g64, r, data_range=gmax, full=True, **SSIM_KW)
        ss[key] = float(smap[m2].mean())
        nm[key] = float((e * e).sum() / (g * g).sum())
    with h5py.File(R / "data/mask_af16_3d.h5") as f:
        mask3 = np.array(f["mask"]) > 0
    assert mask3.shape == gt.shape and np.all(mask3 == mask3[:1]), f"{R.name}: expected a readout-separable 3-D mask"
    af = json.load(open(R / "logs/prep_04_export.json"))["contrasts"][c]["AF_vs_full_rect"]
    ROWS.append(dict(name=name, af=af, arms=arms, g2=g2, gmax=gmax, vols=vols, ps=ps, ss=ss, nm=nm,
                     emask=head_region(m2), vmax=float(np.quantile(g2, 0.999)), kmask=mask3[mask3.shape[0] // 2],
                     best=dict(psnr=max(ps, key=ps.get), ssim=max(ss, key=ss.get), nmse=min(nm, key=nm.get))))

ARM_LABELS = [lab for _, lab, _ in ROWS[0]["arms"]]
ncol = len(ARM_LABELS) + 1
pw, gap = 2.6, args.gap
CBTRIM = 0.10
top_in, cbpad, cbw, cbroom = 1.15, 0.06, 0.13, 0.60
H, W = D(ROWS[0]["g2"]).shape
ph = pw * H / W                                  # every panel is a transverse slice: one height
fw = ncol * pw + (ncol - 1) * gap + cbpad + cbw + cbroom
fh = top_in + 6 * ph + 5 * gap + 0.16
fig = plt.figure(figsize=(fw, fh))


def axes_at(x0, y0, w, h):
    a = fig.add_axes([x0 / fw, 1 - (y0 + h) / fh, w / fw, h / fh])
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)
    return a


def label(a, text, bold=False, line=0, fs=12, h=None, color="white"):
    dy = line * fs * 1.28 / 72 / h if line else 0.0
    t = a.text(0.025, 0.975 - dy, text, transform=a.transAxes, ha="left", va="top", color=color, fontsize=fs,
               fontweight="bold" if bold else "normal")
    t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])


def _err_ticks(vmax):
    for step in (0.01, 0.02, 0.025, 0.03, 0.04, 0.05):
        n = vmax / step
        if abs(n - round(n)) < 1e-9 and 3 <= round(n) <= 5:
            return np.arange(0, round(n) + 1) * step
    return np.linspace(0.0, vmax, 4)


def title_text(lab):
    return (lab if "SPIRiT" in lab else lab.upper()) if CAPS else lab


y = top_in
x_cb = (ncol * pw + (ncol - 1) * gap + cbpad) / fw
for k, p in enumerate(ROWS):
    gim = im = None
    for j in range(ncol):
        x = j * (pw + gap)
        a = axes_at(x, y, pw, ph)
        b = axes_at(x, y + ph + gap, pw, ph)
        if j == 0:
            gim = a.imshow(D(p["g2"]), cmap="gray", vmin=0, vmax=p["vmax"], origin="lower", aspect="auto")
            label(a, f"{p['name']}  {p['af']:.2f}x", bold=True)
            b.imshow(D(p["kmask"]), cmap="gray", vmin=0, vmax=1, origin="lower", aspect="auto", interpolation="nearest")
            if k == 0:
                a.set_title(f"GT ({c})", fontsize=14, pad=4)
            continue
        key, lab, _ = p["arms"][j - 1]
        v = p["vols"][key]
        a.imshow(D(v), cmap="gray", vmin=0, vmax=p["vmax"], origin="lower", aspect="auto")
        if not args.no_metrics:
            for i, (txt, met) in enumerate(((f"{p['ps'][key]:.2f} dB", "psnr"), (f"SSIM {p['ss'][key]:.4f}", "ssim"),
                                             (f"NMSE {p['nm'][key]:.5f}", "nmse"))):
                label(a, txt, bold=key == p["best"][met], line=i, fs=10.5, h=ph, color=METRIC_COLOR)
        emap = D(np.abs(v - p["g2"]) / p["gmax"] * p["emask"])
        im = b.imshow(emap, cmap="jet", vmin=0.0, vmax=args.err_vmax, origin="lower", aspect="auto")
        if k == 0:
            a.set_title(title_text(lab), fontsize=14, pad=4)
    # colorbars: the outer end of each bar flush with its row, the facing ends pulled in by CBTRIM so the grey bar's 0 and
    # the error bar's ceiling do not sit on top of each other (as fig_three_planes_compact.py --gray_cbar)
    for handle, top, tk in ((gim, y, MaxNLocator(5)), (im, y + ph + gap + CBTRIM, _err_ticks(args.err_vmax))):
        cax = fig.add_axes([x_cb, 1 - (top + ph - CBTRIM) / fh, cbw / fw, (ph - CBTRIM) / fh])
        cb = fig.colorbar(handle, cax=cax, ticks=tk)
        cb.ax.tick_params(labelsize=9.5)
    y += 2 * ph + 2 * gap

fig.suptitle(f"{c}  jiangyizhou  nominal AF 16  --  {'Transverse' if CAPS else 'transverse'} {args.tra}  --  "
             f"three sampling patterns, one frozen MC-GS recipe, classical lambdas re-selected per pattern",
             fontsize=13, fontweight="bold", y=1 - 0.10 / fh, va="top")
_cap = []
if not args.no_metrics:
    _win = "7x7 window" if args.ssim == "uniform" else "Gaussian window, sigma 1.5"
    _cap.append(f"head-masked slice PSNR / SSIM ({_win}) / NMSE in each panel, the best of each row in bold")
_cap.append(f"error |recon - GT| / max(GT), jet [0, {args.err_vmax:g}]   (inside the head only)")
_cap.append("left of each error row: that pattern's (ky, kz) sampling mask")
_cap.append("grey bar right of each image row: the slice window (P99.9 of the GT slice, the same for the three rows); "
            "AF label = PE-plane points / sampled points carrying this contrast's data")
fig.text(0.5, 1 - 0.40 / fh, ";   ".join(_cap[:3]) + "\n" + _cap[3], ha="center", va="top", fontsize=11)
name = f"t1flair_v3_af16_masks_stack_{c}_tra{args.tra}_headerr{args.tag}.png"
for d in (F, OUT):
    fig.savefig(d / name, dpi=args.png_dpi)
    if args.pdf:
        for im_ in fig.findobj(mimage.AxesImage):
            im_.set_rasterized(True)
        fig.savefig(d / name.replace(".png", ".pdf"), dpi=args.pdf_dpi)
print("wrote", F / name, "and", OUT / name)
for p in ROWS:
    msg = []
    for met, d, hi in (("PSNR", p["ps"], True), ("SSIM", p["ss"], True), ("NMSE", p["nm"], False)):
        order = sorted(d, key=(lambda k: -d[k]) if hi else (lambda k: d[k]))
        msg.append(f"{met} MC-GS #{order.index(args.mcgs_key) + 1} (first: {order[0]})")
    print(f"  {p['name']:13s} {p['af']:.2f}x transverse {args.tra} (grey window {p['vmax']:.3f}): " + "; ".join(msg))
