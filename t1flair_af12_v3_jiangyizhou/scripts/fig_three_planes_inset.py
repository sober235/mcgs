#!/usr/bin/env python3
"""fig_three_planes_inset.py -- the three-plane comparison in the AF16 phase-3 style, on this AF 12 topic.

Ported 2026-09-13 from T1FLAIR_AF16_v3_jiangyizhou/scripts/fig_three_planes_compact.py (the parallel AF16 loop's
phase-3 figure): zoom insets, per-plane grey colorbars, the transverse quarter-turn, the vector PDF. Added here:
--mask_path and --af_note, which this topic needs for design B (MC-GS FLAIR on the AF 11 mask). The older
fig_three_planes_compact.py of this topic is untouched and still reproduces the 174 / 110 / 47 pair.

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
Title: AF 12 (or --af_note) and the actual acceleration against the contrast's own acquired support (logs/prep_04_export.json).
Writes t1flair_v3_af12_<contrast>_row3planes_compact_<tra>_<cor>_<sag>.png to the served figs folder and figures/.
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
ap.add_argument("--no_mcinr", action="store_true",
                help="leave the MC-INR column out: reproduces the nine-method figures of record made before MC-INR was registered (the AF20 script has the opposite default, --with_mcinr)")
ap.add_argument("--l1_arm", default="espirit", choices=["espirit", "spirit", "both"],
                help="which L1 arm fills that column: L1-ESPIRiT, the SENSE-type default; L1-SPIRiT, the "
                     "k-space one; or both side by side. Non-default adds _spirit / _bothl1 to the file name, "
                     "so the figure of record is never overwritten.")
ap.add_argument("--with_zf", action="store_true",
                help="add a zero-filled column right after GT (user 2026-09-16); appends _zf to the file name")
ap.add_argument("--mask_path", default=None,
                help="3-D sampling mask h5 shown in the left cell of each error row (default data/mask_af12_3d.h5); "
                     "design B uses data/mask_af11_3d.h5 for the FLAIR figure")
ap.add_argument("--af_note", default=None, help="override the AF text in the title")
ap.add_argument("--no_metrics", action="store_true",
                help="do not print the PSNR / SSIM / NMSE numbers inside the plane panels (user 2026-09-13); the\n"
                     "ranking still goes to stdout")
ap.add_argument("--pdf", action="store_true", help="also write a vector PDF next to the PNG")
ap.add_argument("--pdf_dpi", type=int, default=600, help="raster resolution of the image panels inside that PDF")
ap.add_argument("--png_dpi", type=int, default=130,
                help="resolution of the PNG (default 130, the value every figure of record was made at)")
ap.add_argument("--err_vmax", type=float, default=0.1,
                help="top of the fixed jet scale of the error maps, as a fraction of max|GT| (default 0.1, the house "
                     "convention). 0.05 uses the full colour range: 99 percent of the in-head error is under 0.054 for "
                     "every arm but CG-SENSE, so at 0.1 the panels are all dark blue. Fixed for every panel either way.")
ap.add_argument("--tra_rot", type=int, default=1, choices=[0, 1, 2, 3],
                help="quarter turns applied to the TRANSVERSE panel on top of the historical transpose. Default 1 puts "
                     "the eyes up (user 2026-09-13); 0 is the orientation the older figures of this topic use.")
ap.add_argument("--inset_corner", nargs=3, default=["bl", "bl", "bl"], choices=["bl", "br", "tl", "tr"],
                metavar=("TRA", "COR", "SAG"),
                help="which corner of the panel each plane's zoom inset sits in (default bottom-left everywhere); "
                     "put it where it cannot touch the outlined box, e.g. tr bl bl")
ap.add_argument("--gray_cbar", action="store_true",
                help="add a grey-scale colorbar right of the image row and an error colorbar right of the error row")
ap.add_argument("--cbar_planes", type=int, nargs="+", default=[0, 1, 2], choices=[0, 1, 2],
                help="which planes carry those colorbars (default all three; e.g. 0 = the transverse row only). Each "
                     "plane has its own grey window, so a plane without a bar has no scale of its own on the figure.")
ap.add_argument("--inset", type=int, default=0, metavar="W",
                help="full slices plus a WxW zoom inset in the bottom-left of every method panel, the box centred on "
                     "the point where the three slices meet (the face of the WxWxW cube of best_cube_mcgs.py); the box "
                     "is outlined on the slice and its own PSNR is printed in the inset")
ap.add_argument("--caps", action="store_true",
                help="typography (user 2026-09-16): plane names capitalised, method names in the first row upper case; appends _caps")
args = ap.parse_args()
args.tag += {"espirit": "", "spirit": "_spirit", "both": "_bothl1"}[args.l1_arm]
if args.with_zf:
    args.tag += "_zf"
if args.caps:
    args.tag += "_caps"
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
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import matplotlib.image as mimage
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_opening, uniform_filter
from skimage.metrics import structural_similarity

R = Path(__file__).resolve().parents[1]
F = Path("/home/congcongliu/figs/t1flair_v3_af12"); F.mkdir(parents=True, exist_ok=True)
c = args.contrast
METRIC_COLOR = "#FFC23D"          # orange-yellow for the in-panel metrics (user request 2026-09-11); best of each plane in bold
SSIM_KW = (dict(win_size=7) if args.ssim == "uniform"
           else dict(gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
_ex = json.load(open(R / "logs/prep_04_export.json"))
_af = _ex["contrasts"][c]    # AF label from the export json
# AF label gauge (user 2026-09-14): whole phase-encode plane / the points this contrast actually
# has data on = AF_vs_full_rect = BART's Acc. AF_vs_own_support (against the contrast's own
# vendor shutter) is the superseded one and read 9.74 / 12.49 / 15.13 on FLAIR at AF 12/16/20.
# AF label gauge (user 2026-09-16, as in the AF 24 topic): whole PE plane / the points the mask samples -- one number for
# both contrasts, since both read the same mask; how many sampled points carry this contrast's data goes under the panels.
# Replaces the per-contrast gauge of 2026-09-14 (AF_vs_full_rect), which in turn replaced AF_vs_own_support on 2026-09-15.
AF_LABEL = args.af_note if args.af_note else f"nominal AF 12, actual {_ex['mask2d_AF_rect']:.2f}x"
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


def spirit_file():
    """saved volume of the L1-SPIRiT row: recon/spirit_ks<K>_lam<L>_<c>_complex.npy at the kernel size
    and lambda summarize.py kept (its sweep spans both, so lam_file's prefix_lam<L> pattern misses it)."""
    r = rows.get("spirit", {})
    if r.get("lam") is None or not r.get("ksize"):
        return None
    q = R / f"recon/spirit_ks{r['ksize']}_lam{r['lam']:g}_{c}_complex.npy"
    return q if q.exists() else None


L1_ARMS = {"espirit": [("l1espirit", "L1-ESPIRiT", lam_file("l1espirit", "l1espirit"))],
           "spirit": [("spirit", "L1-SPIRiT", spirit_file())],
           "both": [("l1espirit", "L1-ESPIRiT", lam_file("l1espirit", "l1espirit")),
                    ("spirit", "L1-SPIRiT", spirit_file())]}[args.l1_arm]


ARMS = [*([("zf", "ZF", R / f"recon/zf_{c}_complex.npy")] if args.with_zf else []),
        ("cgsense", "CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"),
        ("cssense_o200", "CS-SENSE", lam_file("cssense_o200", "cssense_o200")),
        ("jcssense", "Joint SENSE", lam_file("jcssense", "jcssense_o200")),
        *L1_ARMS,
        ("zsssdu", "ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
        ("dip", "3D-DIP", R / f"recon/dip_{c}_complex.npy"),
        ("inr", "3D-INR", R / f"recon/inr_{c}_complex.npy"),
        *([] if args.no_mcinr else [("mcinr", "MC-INR", R / f"recon/mcinr_{c}_complex.npy")]),
        (args.single_key, "GS-single", R / f"recon/{args.single_key}_{c}_mag.npy"),
        (args.mcgs_key, "MC-GS", R / f"recon/{args.mcgs_key}_{c}_mag.npy")]
missing = [lab for _, lab, p in ARMS if p is None or not Path(p).exists()]
assert not missing, f"missing volumes for {c}: {missing}"

gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
gmax = float(gt.max())
head = gt > 0.05 * float(np.quantile(gt, 0.999))
vols = {key: np.abs(np.load(path)).astype(np.float32) for key, _, path in ARMS}

with h5py.File(Path(args.mask_path) if args.mask_path else R / "data/mask_af12_3d.h5") as f:
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
    """display transform: axis 0 transposed as in fig_row_final.py, then --tra_rot quarter turns (1 = eyes up);
    axes 1 and 2 are shown with axis 0 upright."""
    return np.rot90(x.T, args.tra_rot) if axis == 0 else x


ANAT = {0: "transverse", 1: "coronal", 2: "sagittal"}
BOX = None
if args.inset:
    _hw = args.inset // 2
    BOX = np.zeros(gt.shape, bool)
    _i, _j, _k = args.slices
    BOX[_i - _hw:_i + _hw, _j - _hw:_j + _hw, _k - _hw:_k + _hw] = True
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
        # the box is defined in volume coordinates and pushed through the same D(), so it lands correctly
        # whatever orientation the panel is drawn in
        bm = D(BOX[sl], axis)
        rows, cols = np.nonzero(bm)
        zoom = dict(x0=int(cols.min()), y0=int(rows.min()), w=int(args.inset))
        assert cols.max() - cols.min() + 1 == args.inset and rows.max() - rows.min() + 1 == args.inset, \
            f"box is {cols.max()-cols.min()+1} x {rows.max()-rows.min()+1}, expected {args.inset} square"
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
CBTRIM = 0.10                                # inches pulled off the facing ends of the two colorbars
top_in, cbpad, cbw, cbroom = 1.15, 0.06, 0.13, (0.60 if args.gray_cbar else 0.42)   # wider label room when every row has a bar
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
ZLW = 2.2                                        # its line width (user 2026-09-13: thicker); the inset border is ZLW + 0.6


def put_inset(parent, img, z, x0_in, y0_in, pw_in, h_in, corner="bl", **imkw):
    """draw the zoom box on `parent` and paste the magnified box in the given corner of the panel"""
    parent.add_patch(Rectangle((z["x0"], z["y0"]), z["w"], z["w"], fill=False, ec=ZC, lw=ZLW))
    side, pad = 0.38 * pw_in, 0.012 * pw_in                    # inset side and margin, inches (square inset)
    left = x0_in + (pad if corner in ("bl", "tl") else pw_in - pad - side)
    top = y0_in + (pad if corner in ("tl", "tr") else h_in - pad - side)
    a = fig.add_axes([left / fw, 1 - (top + side) / fh, side / fw, side / fh])
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_edgecolor(ZC); sp.set_linewidth(ZLW + 0.6)
    a.imshow(img[z["y0"]:z["y0"] + z["w"], z["x0"]:z["x0"] + z["w"]], origin="lower", aspect="auto", **imkw)
    return a


def _err_ticks(vmax):
    """round tick steps for the error bar that always include vmax (0.1 -> 0, .025, .05, .075, .1; 0.09 -> 0, .03, .06, .09)"""
    for step in (0.01, 0.02, 0.025, 0.03, 0.04, 0.05):
        n = vmax / step
        if abs(n - round(n)) < 1e-9 and 3 <= round(n) <= 5:
            return np.arange(0, round(n) + 1) * step
    return np.linspace(0.0, vmax, 4)


y, im = top_in, None
for k, p in enumerate(planes):
    ax_, h = p["axis"], row_h[2 * k]
    for j in range(ncol):
        x = j * (pw + gap)
        a = axes_at(x, y, pw, h)
        b = axes_at(x, y + h + gap, pw, h)
        if j == 0:
            gim = a.imshow(D(p["g2"], ax_), cmap="gray", vmin=0, vmax=p["vmax"], origin="lower", aspect="auto")
            if p["zoom"]:
                put_inset(a, D(p["g2"], ax_), p["zoom"], x, y, pw, h, corner=args.inset_corner[ax_], cmap="gray", vmin=0, vmax=p["vmax"])
            label(a, ANAT[ax_].capitalize() if args.caps else ANAT[ax_], bold=True)          # plane name only (user 2026-09-13); the slice indices are in the title
            b.imshow(D(p["kmask"], ax_), cmap="gray", vmin=0, vmax=1, origin="lower", aspect="auto", interpolation="nearest")
            if k == 0:
                a.set_title(f"GT ({c})", fontsize=14, pad=4)
            continue
        key, lab, _ = ARMS[j - 1]
        v = vols[key][p["sl"]][p["cr"]]
        a.imshow(D(v, ax_), cmap="gray", vmin=0, vmax=p["vmax"], origin="lower", aspect="auto")
        if p["zoom"]:
            put_inset(a, D(v, ax_), p["zoom"], x, y, pw, h, corner=args.inset_corner[ax_], cmap="gray", vmin=0, vmax=p["vmax"])
        if not args.no_metrics:
            for i, (txt, met) in enumerate(((f"{p['ps'][key]:.2f} dB", "psnr"), (f"SSIM {p['ss'][key]:.4f}", "ssim"),
                                             (f"NMSE {p['nm'][key]:.5f}", "nmse"))):
                label(a, txt, bold=key == p["best"][met], line=i, fs=10.5, h=h, color=METRIC_COLOR)
        emap = D(np.abs(v - p["g2"]) / gmax * (p["emask"] if args.mask_error else 1), ax_)
        im = b.imshow(emap, cmap="jet", vmin=0.0, vmax=args.err_vmax, origin="lower", aspect="auto")
        if p["zoom"]:
            put_inset(b, emap, p["zoom"], x, y + h + gap, pw, h, corner=args.inset_corner[ax_], cmap="jet", vmin=0.0, vmax=args.err_vmax)
        if k == 0:
            a.set_title((lab if "SPIRiT" in lab else lab.upper()) if args.caps else lab,
                        fontsize=14, pad=4)
    if args.gray_cbar and ax_ in args.cbar_planes:
        x_cb = (ncol * pw + (ncol - 1) * gap + cbpad) / fw
        # the outer end of each bar is flush with its row; the two ends that face each other are pulled in by CBTRIM so
        # the grey bar's 0 and the error bar's ceiling do not sit on top of each other (user 2026-09-13)
        for handle, top, tk in ((gim, y, MaxNLocator(5)),
                                (im, y + h + gap + CBTRIM, _err_ticks(args.err_vmax))):
            cax_ = fig.add_axes([x_cb, 1 - (top + h - CBTRIM) / fh, cbw / fw, (h - CBTRIM) / fh])
            cbx = fig.colorbar(handle, cax=cax_, ticks=tk)
            cbx.ax.tick_params(labelsize=9.5)
    y += 2 * h + 2 * gap
last_top = top_in + sum(row_h[:-1]) + (len(row_h) - 1) * gap
if not args.gray_cbar:                       # with --gray_cbar every row already carries its own bar
    cax = fig.add_axes([(ncol * pw + (ncol - 1) * gap + cbpad) / fw, 1 - (last_top + row_h[-1]) / fh, cbw / fw, row_h[-1] / fh])
    cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=10)
_cube_txt = f"  --  {args.cube[3]}^3 box centred ({args.cube[0]}, {args.cube[1]}, {args.cube[2]})" if args.cube else ""
_P = [ANAT[i].capitalize() if args.caps else ANAT[i] for i in (0, 1, 2)]
fig.suptitle(f"{c}  jiangyizhou  {AF_LABEL}  --  {_P[0]} {args.slices[0]} / {_P[1]} {args.slices[1]} / {_P[2]} {args.slices[2]}" + _cube_txt,
             fontsize=13, fontweight="bold", y=1 - 0.10 / fh, va="top")
_cap = []
if not args.no_metrics:
    _win = "7x7 window" if args.ssim == "uniform" else "Gaussian window, sigma 1.5"
    _cap.append(f"head-masked slice PSNR / SSIM ({_win}) / NMSE in each panel, the best of each plane in bold")
_cap.append(f"error |recon - GT| / max(GT), jet [0, {args.err_vmax:g}]" + ("   (inside the head only)" if args.mask_error else ""))
_cap.append("left of each error row: the sampling mask in that plane (k-space, never cropped)")
NPL = int(_ex["mask2d_shape"][0]) * int(_ex["mask2d_shape"][1])
_cap.append(f"AF = {NPL} plane points / {_ex['mask2d_samples']} sampled points, the same on both contrasts (one mask)"
            + ("" if _af["support_frac"] >= 1 else
               f";   this scan acquired {round(_af['support_frac'] * NPL)} of the {NPL} lines, so {round(_af['sampled_frac_of_plane'] * NPL)} "
               f"sampled points carry {c} data"))
if args.gray_cbar:
    _cap.append("grey bar right of the " + " / ".join(ANAT[a] for a in sorted(args.cbar_planes))
            + " image row: that plane's window (each plane has its own)")
_CN = {"bl": "bottom-left", "br": "bottom-right", "tl": "top-left", "tr": "top-right"}
_cap2 = (f"in every panel, the {args.inset}x{args.inset} box outlined on the slice, magnified into the "
         + " / ".join(f"{_CN[args.inset_corner[a]]} ({ANAT[a]})" for a in (0, 1, 2)) if args.inset else "")
fig.text(0.5, 1 - 0.40 / fh, ";   ".join(_cap) + ("\n" + _cap2 if _cap2 else ""),
         ha="center", va="top", fontsize=11)
name = (f"t1flair_v3_af12_{c}_cube{args.cube[3]}_{args.cube[0]}_{args.cube[1]}_{args.cube[2]}" if args.cube else
        f"t1flair_v3_af12_{c}_row3planes_compact_{'_'.join(map(str, args.slices))}" + (f"_inset{args.inset}" if args.inset else "")) + f"{'_headerr' if args.mask_error else ''}{args.tag}.png"
for d in (F, R / "figures"):
    fig.savefig(d / name, dpi=args.png_dpi)
    if args.pdf:
        # vector PDF: text, boxes and colorbar frames stay vector; the image panels are rasterised at --pdf_dpi
        for im_ in fig.findobj(mimage.AxesImage):
            im_.set_rasterized(True)
        fig.savefig(d / name.replace(".png", ".pdf"), dpi=args.pdf_dpi)
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
