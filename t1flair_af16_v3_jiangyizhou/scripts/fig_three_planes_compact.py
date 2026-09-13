#!/usr/bin/env python3
"""fig_three_planes_compact.py -- the three-plane comparison (transverse / coronal / sagittal) in one compact figure.

  python scripts/fig_three_planes_compact.py --contrast T1 --slices 174 110 70

Layout (user request 2026-09-11): per plane an image row (GT + the nine methods of fig_row_final.py) and an error
row (|recon - GT| / max(GT), jet, fixed [0, 0.1]). The left cell of each error row is the sampling mask in that
plane: the plane of the 3-D mask through the k-space centre, in the same orientation as the images (the readout,
axis 0, is fully sampled, so the coronal and sagittal cuts are lines). Method names only above the top row; each
panel carries its head-masked slice PSNR, SSIM and NMSE inside the image (each metric's best in the plane in bold);
minimal gaps; one error
colorbar, right of the last row. Grey window: transverse and coronal as fig_row_final.py (the slice's P99.9); the
sagittal window is set so that its in-head median has the same grey level as the mean of the other two planes
(a lateral sagittal slice carries bright scalp fat that pushes its P99.9 up and darkens the brain).
Slice metrics, head mask |GT| > 0.05 * P99.9 of the volume: PSNR at the volume's max|GT|; SSIM = the 2-D SSIM map
(7x7 uniform window as paired_gs/metrics/compute_ssim; --ssim gaussian: sigma 1.5; data range = max|GT| of the
volume) averaged over the head pixels, as slice_quality.py; NMSE = compute_nmse restricted to the head pixels,
sum (recon - GT)^2 / sum GT^2.
Title: nominal AF 16 and the actual acceleration against the contrast's own acquired support (logs/prep_04_export.json).
Writes t1flair_v3_af16_<contrast>_row3planes_compact_<tra>_<cor>_<sag>.png to the served figs folder and figures/.
"""
import argparse, json, re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", required=True, choices=["T1", "FLAIR"])
ap.add_argument("--slices", type=int, nargs=3, metavar=("TRA", "COR", "SAG"))
ap.add_argument("--cube", type=int, nargs=4, metavar=("I", "J", "K", "W"),
                help="one 3-D box: the three orthogonal centre planes through (I, J, K), each cropped to the W x W face "
                     "(scripts/best_cube_mcgs.py). Sets --slices and crops every panel and every metric to the face.")
ap.add_argument("--gap", type=float, default=0.025, help="gap between neighbouring panels, inches")
ap.add_argument("--ssim", choices=["uniform", "gaussian"], default="uniform", help="SSIM window: 7x7 uniform or Gaussian sigma 1.5")
ap.add_argument("--mask_error", action="store_true", help="error maps only inside the filled head mask (the region the metrics use); file suffix _headerr")
ap.add_argument("--mcgs_key", default="gs_joint_ar", help="which registered joint arm fills the MC-GS column")
ap.add_argument("--single_key", default="gs_single_ar", help="which registered single arm fills the GS-single column")
ap.add_argument("--tag", default="", help="extra suffix for the output file name, e.g. _p3")
ap.add_argument("--inset", type=int, default=0, metavar="W",
                help="full slices plus a WxW zoom inset in the bottom-left of every method panel, the box centred on "
                     "the point where the three slices meet (the face of the WxWxW cube of best_cube_mcgs.py); the box "
                     "is outlined on the slice and its own PSNR is printed in the inset")
args = ap.parse_args()
if (args.slices is None) == (args.cube is None):
    ap.error("give either --slices TRA COR SAG or --cube I J K W")
CROP = None
if args.cube:
    _i, _j, _k, _w = args.cube; _h = _w // 2
    args.slices = [_i, _j, _k]
    CROP = {0: (slice(_j - _h, _j + _h), slice(_k - _h, _k + _h)),      # in-plane axes (1, 2)
            1: (slice(_i - _h, _i + _h), slice(_k - _h, _k + _h)),      # in-plane axes (0, 2)
            2: (slice(_i - _h, _i + _h), slice(_j - _h, _j + _h))}      # in-plane axes (0, 1)

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_opening, uniform_filter
from skimage.metrics import structural_similarity

R = Path(__file__).resolve().parents[1]
F = Path("/home/congcongliu/figs/t1flair_v3_af16"); F.mkdir(parents=True, exist_ok=True)
c = args.contrast
METRIC_COLOR = "#FFC23D"          # orange-yellow for the in-panel metrics (user request 2026-09-11); best of each plane in bold
SSIM_KW = (dict(win_size=7) if args.ssim == "uniform"
           else dict(gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
_af = json.load(open(R / "logs/prep_04_export.json"))["contrasts"][c]    # nominal AF 16 label (user decision 2026-09-11)
AF_LABEL = f"nominal AF 16, actual {_af['AF_vs_own_support']:.2f}x" + (" of its acquired k-space" if _af["support_frac"] < 1 else "")
rows = {r["method"]: r for r in json.load(open(R / "logs/summary.json")) if r["contrast"] == c}


def lam_file(row_key, prefix):
    """saved volume of a swept family (recon/<prefix>_lam<L>_<c>_complex.npy) at the lambda kept for its row"""
    lam = rows.get(row_key, {}).get("lam")
    if lam is None:
        return None
    for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
        m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
        if m and abs(float(m.group(1)) - float(lam)) < 1e-12:
            return p
    return None


ARMS = [("cgsense", "CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"),
        ("cssense_o200", "CS-SENSE", lam_file("cssense_o200", "cssense_o200")),
        ("jcssense", "Joint SENSE", lam_file("jcssense", "jcssense_o200")),
        ("l1espirit", "L1-ESPIRiT", lam_file("l1espirit", "l1espirit")),
        ("zsssdu", "ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
        ("dip", "3D-DIP", R / f"recon/dip_{c}_complex.npy"),
        ("inr", "3D-INR", R / f"recon/inr_{c}_complex.npy"),
        (args.single_key, "GS-single", R / f"recon/{args.single_key}_{c}_mag.npy"),
        (args.mcgs_key, "MC-GS", R / f"recon/{args.mcgs_key}_{c}_mag.npy")]
missing = [lab for _, lab, p in ARMS if p is None or not Path(p).exists()]
assert not missing, f"missing volumes for {c}: {missing}"

gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
gmax = float(gt.max())
head = gt > 0.05 * float(np.quantile(gt, 0.999))
vols = {key: np.abs(np.load(path)).astype(np.float32) for key, _, path in ARMS}

with h5py.File(R / "data/mask_af16_3d.h5") as f:
    mask3 = np.array(f["mask"]) > 0
assert mask3.shape == gt.shape and np.all(mask3 == mask3[:1]), "expected a 3-D mask with the readout (axis 0) fully sampled"
dens = uniform_filter(mask3[0].astype(float), size=9, mode="constant")     # the (ky, kz) pattern; densest = k-space centre
kyc, kzc = (int(v) for v in np.unravel_index(np.argmax(dens), dens.shape))
kcen = (mask3.shape[0] // 2, kyc, kzc)

DISK = np.hypot(*np.mgrid[-6:7, -6:7]) <= 6


def head_region(m):
    """--mask_error display region: the filled head mask without thin protrusions (ghost lines in the air that pass the
    intensity threshold), opened with a radius-6 disk, plus a 2-px rim"""
    return binary_dilation(binary_fill_holes(binary_opening(binary_fill_holes(m), structure=DISK)), iterations=2)


def D(x, axis):
    return x.T if axis == 0 else x          # as fig_row_final.py: axis 0 transposed; axes 1, 2 with axis 0 upright


ANAT = {0: "transverse", 1: "coronal", 2: "sagittal"}
planes = []
for axis, idx in enumerate(args.slices):
    sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
    cr = CROP[axis] if CROP else (slice(None), slice(None))
    g2, m2 = gt[sl][cr], head[sl][cr]
    g64 = g2.astype(np.float64)
    g = g64[m2]
    ps, ss, nm = {}, {}, {}
    for key, _, _ in ARMS:
        r = vols[key][sl][cr].astype(np.float64)
        e = r[m2] - g
        ps[key] = 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
        _, smap = structural_similarity(g64, r, data_range=gmax, full=True, **SSIM_KW)
        ss[key] = float(smap[m2].mean())
        nm[key] = float((e * e).sum() / (g * g).sum())
        zoom = None
    if args.inset:
        hw = args.inset // 2
        i0, j0, k0 = args.slices
        # D() transposes axis 0, so give the box directly in display (col, row) order
        x0, y0 = {0: (j0 - hw, k0 - hw), 1: (k0 - hw, i0 - hw), 2: (j0 - hw, i0 - hw)}[axis]
        zoom = dict(x0=int(x0), y0=int(y0), w=int(args.inset))
    planes.append(dict(axis=axis, idx=idx, sl=sl, cr=cr, g2=g2, zoom=zoom, ps=ps, ss=ss, nm=nm, emask=head_region(m2),
                       best=dict(psnr=max(ps, key=ps.get), ssim=max(ss, key=ss.get), nmse=min(nm, key=nm.get)),
                       vmax=float(np.quantile(g2, 0.999)), med=float(np.median(g2[m2])),
                       kmask=mask3.take(kcen[axis], axis=axis)))
if args.inset:
    for p in planes:
        z, ax_ = p["zoom"], p["axis"]
        cut = (slice(z["y0"], z["y0"] + z["w"]), slice(z["x0"], z["x0"] + z["w"]))
        gz = D(gt[p["sl"]], ax_)[cut].astype(np.float64)
        mz = D(head[p["sl"]], ax_)[cut]
        zp = {}
        for key, _, _ in ARMS:
            e = D(vols[key][p["sl"]], ax_)[cut].astype(np.float64)[mz] - gz[mz]
            zp[key] = 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
        p["zoom"]["psnr"] = zp
        p["zoom"]["best"] = max(zp, key=zp.get)
        p["zoom"]["head_frac"] = float(mz.mean())
ratio = float(np.mean([p["med"] / p["vmax"] for p in planes if p["axis"] != 2]))
for p in planes:
    p["vmax_p999"] = p["vmax"]
    if p["axis"] == 2:
        p["vmax"] = p["med"] / ratio


ncol = len(ARMS) + 1
pw, gap = 2.6, args.gap                     # panel width and gap (in)
top_in, cbpad, cbw, cbroom = 1.15, 0.06, 0.13, 0.42
row_h = []
for p in planes:
    H, W = D(p["g2"], p["axis"]).shape
    row_h += [pw * H / W] * 2
fw = ncol * pw + (ncol - 1) * gap + cbpad + cbw + cbroom
fh = top_in + sum(row_h) + (len(row_h) - 1) * gap + 0.16            # room below for the last colorbar label
fig = plt.figure(figsize=(fw, fh))


def axes_at(x0, y0, w, h):
    """axes from a top-left corner and size in inches"""
    a = fig.add_axes([x0 / fw, 1 - (y0 + h) / fh, w / fw, h / fh])
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)
    return a


def label(a, text, bold=False, line=0, fs=12, h=None, color="white"):
    dy = line * fs * 1.28 / 72 / h if line else 0.0     # line spacing in axes units of a panel h inches tall
    t = a.text(0.025, 0.975 - dy, text, transform=a.transAxes, ha="left", va="top", color=color, fontsize=fs,
               fontweight="bold" if bold else "normal")
    t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])



ZC = METRIC_COLOR                                # zoom-box outline / inset border (user 2026-09-13: orange-yellow)


def put_inset(parent, img, z, x0_in, y0_in, pw_in, h_in, **imkw):
    """draw the zoom box on `parent` and paste the magnified box in the panel's bottom-left corner"""
    parent.add_patch(Rectangle((z["x0"], z["y0"]), z["w"], z["w"], fill=False, ec=ZC, lw=1.1))
    side = 0.38 * pw_in                                        # inset side, inches (square)
    a = fig.add_axes([(x0_in + 0.012 * pw_in) / fw, 1 - (y0_in + h_in - 0.012 * pw_in) / fh, side / fw, side / fh])
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_edgecolor(ZC); sp.set_linewidth(1.4)
    a.imshow(img[z["y0"]:z["y0"] + z["w"], z["x0"]:z["x0"] + z["w"]], origin="lower", aspect="auto", **imkw)
    return a


y, im = top_in, None
for k, p in enumerate(planes):
    ax_, h = p["axis"], row_h[2 * k]
    for j in range(ncol):
        x = j * (pw + gap)
        a = axes_at(x, y, pw, h)
        b = axes_at(x, y + h + gap, pw, h)
        if j == 0:
            a.imshow(D(p["g2"], ax_), cmap="gray", vmin=0, vmax=p["vmax"], origin="lower", aspect="auto")
            if p["zoom"]:
                put_inset(a, D(p["g2"], ax_), p["zoom"], x, y, pw, h, cmap="gray", vmin=0, vmax=p["vmax"])
            label(a, f"{ANAT[ax_]} {p['idx']}", bold=True)
            b.imshow(D(p["kmask"], ax_), cmap="gray", vmin=0, vmax=1, origin="lower", aspect="auto", interpolation="nearest")
            if k == 0:
                a.set_title(f"GT ({c})", fontsize=14, fontweight="bold", pad=4)
            continue
        key, lab, _ = ARMS[j - 1]
        v = vols[key][p["sl"]][p["cr"]]
        a.imshow(D(v, ax_), cmap="gray", vmin=0, vmax=p["vmax"], origin="lower", aspect="auto")
        if p["zoom"]:
            put_inset(a, D(v, ax_), p["zoom"], x, y, pw, h, cmap="gray", vmin=0, vmax=p["vmax"])
        for i, (txt, met) in enumerate(((f"{p['ps'][key]:.2f} dB", "psnr"), (f"SSIM {p['ss'][key]:.4f}", "ssim"),
                                         (f"NMSE {p['nm'][key]:.5f}", "nmse"))):
            label(a, txt, bold=key == p["best"][met], line=i, fs=10.5, h=h, color=METRIC_COLOR)
        emap = D(np.abs(v - p["g2"]) / gmax * (p["emask"] if args.mask_error else 1), ax_)
        im = b.imshow(emap, cmap="jet", vmin=0.0, vmax=0.1, origin="lower", aspect="auto")
        if p["zoom"]:
            put_inset(b, emap, p["zoom"], x, y + h + gap, pw, h, cmap="jet", vmin=0.0, vmax=0.1)
        if k == 0:
            a.set_title(lab, fontsize=14, pad=4)
    y += 2 * h + 2 * gap
last_top = top_in + sum(row_h[:-1]) + (len(row_h) - 1) * gap
cax = fig.add_axes([(ncol * pw + (ncol - 1) * gap + cbpad) / fw, 1 - (last_top + row_h[-1]) / fh, cbw / fw, row_h[-1] / fh])
cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=10)
_cube_txt = f"  --  {args.cube[3]}^3 box centred ({args.cube[0]}, {args.cube[1]}, {args.cube[2]})" if args.cube else ""
fig.suptitle(f"{c}  jiangyizhou  {AF_LABEL}  --  transverse {args.slices[0]} / coronal {args.slices[1]} / sagittal {args.slices[2]}" + _cube_txt,
             fontsize=13, fontweight="bold", y=1 - 0.10 / fh, va="top")
fig.text(0.5, 1 - 0.40 / fh, f"head-masked slice PSNR / SSIM ({'7x7 window' if args.ssim == 'uniform' else 'Gaussian window, sigma 1.5'}) / NMSE "
         f"in each panel, the best of each plane in bold;   error |recon - GT| / max(GT), jet [0, 0.1];   "
         f"left of each error row: the sampling mask in that plane (k-space, never cropped)"
         + (f"\nbottom-left of every panel: the {args.inset}x{args.inset} box outlined on the slice, magnified (no separate metrics; the in-panel numbers are the whole slice's)" if args.inset else "") + ("   (error shown inside the head only)" if args.mask_error else ""), ha="center", va="top", fontsize=11)
name = (f"t1flair_v3_af16_{c}_cube{args.cube[3]}_{args.cube[0]}_{args.cube[1]}_{args.cube[2]}" if args.cube else
        f"t1flair_v3_af16_{c}_row3planes_compact_{'_'.join(map(str, args.slices))}" + (f"_inset{args.inset}" if args.inset else "")) + f"{'_headerr' if args.mask_error else ''}{args.tag}.png"
for d in (F, R / "figures"):
    fig.savefig(d / name, dpi=130)
print("wrote", F / name, f"k-space centre (kx, ky, kz) = {kcen}")
for p in planes:
    msg = []
    for met, d, hi in (("PSNR", p["ps"], True), ("SSIM", p["ss"], True), ("NMSE", p["nm"], False)):
        order = sorted(d, key=(lambda k: -d[k]) if hi else (lambda k: d[k]))
        msg.append(f"{met} MC-GS #{order.index(args.mcgs_key) + 1} (first: {order[0]})")
    line = f"  {ANAT[p['axis']]} {p['idx']} (grey window {p['vmax']:.3f}): " + "; ".join(msg)
    if p["zoom"]:
        zp = p["zoom"]["psnr"]
        comp = {k: v for k, v in zp.items() if k not in (args.mcgs_key, args.single_key)}
        bc = max(comp, key=comp.get)
        line += (f"  |  {args.inset}-box PSNR: MC-GS {zp[args.mcgs_key]:.2f}, best competing {comp[bc]:.2f} "
                 f"[{bc}], lead {zp[args.mcgs_key] - comp[bc]:+.3f} dB (printed only, not on the figure)")
    print(line)
