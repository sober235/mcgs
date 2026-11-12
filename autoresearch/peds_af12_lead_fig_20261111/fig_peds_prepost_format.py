#!/usr/bin/env python3
"""fig_peds_prepost_format.py -- the pediatric TSC T1w + T2* AF12 three-plane figure in the format of the
plain-scan / enhanced figure of record (ce_af16_lesion_figs_20261107/fig_lesion_plane_prepost.py
--three TRA COR SAG --err_layout zoom --cbar single; results/ce_prepost_record.png).

What is taken from that format
  one figure for both contrasts; rows top to bottom: transverse T1w, transverse T2*, coronal T1w, coronal T2*,
  sagittal T1w, sagittal T2*; under each image row a zoom row: per method the magnified box of the image (left)
  and the same box of the error map (right); GT column of the zoom row: GT box | sampling mask in that plane
  columns GT, ZF, CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR, GS-single, MC-GS
  the box is only outlined on the image panels (no corner inset); head-masked slice PSNR / SSIM / NMSE in every
  panel, the best of each row in bold; error |recon - GT| / max|GT| of the volume, jet [0, 0.1], inside the head
  one grey bar (intensity as a fraction of each row's own window) and one error bar, beside the first row pair

What stays this topic's own (Peds_TSC_MTP_AF12_retro, its scripts/fig_three_planes_inset.py)
  volumes: the registered arms resolved as that script does (logs/summary.json; L1 = L1-SPIRiT; MC-GS =
  gs_joint_p5; GS-single = gs_single_p3f on T1w, gs_single_ar on T2*)
  anatomy: volume (RO 320, PE 304, SPE 104), axis 2 transverse, 0 coronal, 1 sagittal; transverse drawn as the
  vendor exports it, the other two transposed, vertex up; voxels 0.7 x 0.7 x 1.35 mm, panels at true proportions
  grey window: slice P99.9, the sagittal row at the in-head median level of the other two planes
  metric arithmetic: head = |GT| > 0.05 P99.9 of the volume, PSNR at max|GT|, 7x7 SSIM map averaged over the
  head pixels, head NMSE (the same arithmetic as the plain-scan / enhanced figure)

Zoom box: a box of --box voxels along (RO, PE, SPE), default 48 x 48 x 24 = 33.6 x 33.6 x 32.4 mm, so it is square
on the page in all three planes like the 48 x 48 box of the record figure (this topic's 48 x 48 x 16 box is a
landscape rectangle on the coronal and sagittal planes). Centre: where the three slices meet, or --box_center.

Slices: --slices TRA COR SAG, default the slice of each plane on which both contrasts lead every arm by most
(rule of draw_lead_slice.py, from logs/find_slice_all_first_{T1w,T2s}.json).

Check: every drawn PSNR (and SSIM where recorded) is compared with the topic's per-slice record
logs/find_slice_all_first_<c>.json and the script stops on a difference, so the drawn volumes are the scored ones.
Read-only on the topic; writes PNG, preview, optional PDF and a JSON record with every volume's SHA-256 into
MCGS_ARTIFACT_DIR. CPU only. Neither GT nor any ROI enters a reconstruction.

  gpu.py run --cpu -- @python autoresearch/peds_af12_lead_fig_20261111/fig_peds_prepost_format.py
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--slices", type=int, nargs=3, default=None, metavar=("TRA", "COR", "SAG"),
                help="transverse (SPE), coronal (RO), sagittal (PE); default: the best both-contrasts-first slice of each plane")
ap.add_argument("--box", type=int, nargs=3, default=[48, 48, 24], metavar=("RO", "PE", "SPE"), help="zoom box extent in voxels")
ap.add_argument("--box_center", type=int, nargs=3, default=[-1, -1, -1], metavar=("RO", "PE", "SPE"),
                help="centre of the zoom box; -1 = the matching slice index")
ap.add_argument("--err_vmax", type=float, default=0.1)
ap.add_argument("--brightness", type=float, default=1.0, help="display only: window top / brightness (the record figure uses 1.5 for CE)")
ap.add_argument("--title_fs", type=float, default=22.0)
ap.add_argument("--png_dpi", type=int, default=200)
ap.add_argument("--pdf", action="store_true")
ap.add_argument("--pdf_dpi", type=int, default=600)
ap.add_argument("--sag_mask_rot", type=int, default=1, choices=[0, 1, 2, 3],
                help="quarter turns of the sampling-mask cell of the sagittal rows (user 2026-11-11: 1, so its sampled lines "
                     "run vertically as on the transverse rows and on the record figure; 0 = the orientation of the "
                     "sagittal image above it, where the readout runs left-right and the lines are horizontal)")
ap.add_argument("--tag", default="", help="suffix of the output file name")
args = ap.parse_args()

import h5py  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.image as mimage  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_opening, uniform_filter  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402

os.nice(10)
R = Path("/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro")
CONTRASTS = ("T1w", "T2s")
SHOW = {"T1w": "T1w", "T2s": "T2*"}
MCGS_KEY = "gs_joint_p5"
SINGLE_KEY = {"T1w": "gs_single_p3f", "T2s": "gs_single_ar"}
MC = "MC-GS"
PLANES = [(2, "Transverse"), (0, "Coronal"), (1, "Sagittal")]      # display order; this topic's axes
VOXEL = (0.7, 0.7, 1.35)                                           # mm along (RO, PE, SPE)
METRICS = ("psnr", "ssim", "nmse")
METRIC_COLOR = "#FFC23D"
ZC, ZLW = METRIC_COLOR, 2.2
log = lambda *a: print(*a, flush=True)
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", "peds_prepost_format_out")).resolve()
art.mkdir(parents=True, exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def D(x, axis):
    """this topic's display: transverse as the vendor exports it, the other two transposed (partition axis vertical)"""
    return x if axis == 2 else x.T


def phys(axis, shape):
    """(height, width) in mm of a displayed (rows, cols) extent"""
    ra, ca = {0: (2, 1), 1: (2, 0), 2: (0, 1)}[axis]
    return shape[0] * VOXEL[ra], shape[1] * VOXEL[ca]


# ---------------- slices ----------------
J = {c: json.load(open(R / f"logs/find_slice_all_first_{c}.json")) for c in CONTRASTS}
for c in CONTRASTS:
    assert J[c]["key"] == MCGS_KEY, (c, J[c]["key"])
M = {c: {(e["plane"], e["idx"]): e for e in J[c]["slices"]} for c in CONTRASTS}
if args.slices:
    slices = list(args.slices)
else:
    both = [k for k in M["T1w"] if k in M["T2s"] and M["T1w"][k]["wins_all"] and M["T2s"][k]["wins_all"]]
    ranked = sorted(both, key=lambda k: min(M["T1w"][k]["d_psnr"], M["T2s"][k]["d_psnr"]), reverse=True)
    slices = [next(k for k in ranked if k[0] == name.lower())[1] for _, name in PLANES]
log("slices (transverse, coronal, sagittal):", slices)

# ---------------- volumes: the registered arms, as the topic's figure script ----------------
summary = json.load(open(R / "logs/summary.json"))


def arms(c):
    rows = {r["method"]: r for r in summary if r["contrast"] == c}

    def lam_file(row_key, prefix):
        lam = rows[row_key]["lam"]
        hit = [p for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy")
               if (m := re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name))
               and abs(float(m.group(1)) - float(lam)) < 1e-12]
        assert len(hit) == 1, f"{c} {row_key}: expected one volume at lambda {lam}, got {hit}"
        return hit[0]

    return [("zf", "ZF", R / f"recon/zf_{c}_complex.npy"),
            ("cgsense", "CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"),
            ("cssense_o200", "CS-SENSE", lam_file("cssense_o200", "cssense_o200")),
            ("jcssense", "Joint SENSE", lam_file("jcssense", "jcssense_o200")),
            ("spirit", "L1-SPIRiT", R / f"recon/{rows['spirit']['src']}_{c}_complex.npy"),
            ("zsssdu", "ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
            ("dip", "3D-DIP", R / f"recon/dip_{c}_complex.npy"),
            ("inr", "3D-INR", R / f"recon/inr_{c}_complex.npy"),
            ("mcinr", "MC-INR", R / f"recon/mcinr_{c}_complex.npy"),
            (SINGLE_KEY[c], "GS-single", R / f"recon/{SINGLE_KEY[c]}_{c}_mag.npy"),
            (MCGS_KEY, MC, R / f"recon/{MCGS_KEY}_{c}_mag.npy")]


ARMS = {c: arms(c) for c in CONTRASTS}
LABELS = [lab for _, lab, _ in ARMS["T1w"]]
assert LABELS == [lab for _, lab, _ in ARMS["T2s"]] and LABELS[-1] == MC
KEY = {c: {lab: key for key, lab, _ in ARMS[c]} for c in CONTRASTS}
gt, gmax, head, vol = {}, {}, {}, {c: {} for c in CONTRASTS}
record = dict(topic=str(R), format="fig_lesion_plane_prepost.py --three --err_layout zoom --cbar single",
              mcgs_key=MCGS_KEY, single_key=SINGLE_KEY, slices=dict(zip(("transverse", "coronal", "sagittal"), slices)),
              volumes={c: {} for c in CONTRASTS}, gt_sha256={}, rows=[])
for c in CONTRASTS:
    g = R / f"recon/gt_{c}_complex.npy"
    gt[c] = np.abs(np.load(g)).astype(np.float32)
    gmax[c] = float(gt[c].max())
    head[c] = gt[c] > 0.05 * float(np.quantile(gt[c], 0.999))
    record["gt_sha256"][c] = sha(g)
    for key, lab, p in ARMS[c]:
        assert p.exists(), f"missing volume {p}"
        vol[c][lab] = np.abs(np.load(p)).astype(np.float32)
        assert vol[c][lab].shape == gt[c].shape, (c, lab, vol[c][lab].shape)
        record["volumes"][c][lab] = dict(key=key, path=str(p), sha256=sha(p))
SHAPE = gt["T1w"].shape
assert gt["T2s"].shape == SHAPE
log("volumes loaded and hashed:", SHAPE)

with h5py.File(R / "data/mask_af12_3d.h5", "r") as f:               # the sampling pattern Omega
    mask3 = np.array(f["mask"]) > 0
assert mask3.shape == SHAPE and np.all(mask3 == mask3[:1]), "expected a 3-D mask, readout (axis 0) fully sampled"
dens = uniform_filter(mask3[0].astype(float), size=9, mode="constant")
kcen = (mask3.shape[0] // 2, *(int(v) for v in np.unravel_index(np.argmax(dens), dens.shape)))
NPL, NSAMP = mask3.shape[1] * mask3.shape[2], int(mask3[0].sum())
AF_OWN = json.load(open(R / "logs/prep_04_export.json"))["contrasts"]["T1w"]["AF_vs_own_support"]
DISK = np.hypot(*np.mgrid[-6:7, -6:7]) <= 6


def head_region(m):
    """display region of the error maps: filled head without thin protrusions + 2-px rim (house rule)"""
    return binary_dilation(binary_fill_holes(binary_opening(binary_fill_holes(m), structure=DISK)), iterations=2)


def plane_metrics(rec, truth, mask, gm):
    t, v = truth.astype(np.float64), rec.astype(np.float64)
    _, smap = structural_similarity(t, v, data_range=gm, full=True, win_size=7)
    e = v[mask] - t[mask]
    return {"psnr": float(20 * np.log10(gm) - 10 * np.log10(float((e * e).mean()))),
            "ssim": float(smap[mask].mean()),
            "nmse": float((e * e).sum() / (t[mask] ** 2).sum())}


# ---------------- zoom box, in volume coordinates ----------------
_tra, _cor, _sag = slices
cen = [v if v >= 0 else d for v, d in zip(args.box_center, (_cor, _sag, _tra))]
hb = [v // 2 for v in args.box]
for name, c0, h_, idx, n in zip(("RO", "PE", "SPE"), cen, hb, (_cor, _sag, _tra), SHAPE):
    assert 0 <= c0 - h_ and c0 + h_ <= n, f"the zoom box leaves the volume along {name}"
    assert c0 - h_ <= idx < c0 + h_, f"the {name} slice {idx} misses the zoom box {c0 - h_}..{c0 + h_ - 1}"
BOX = np.zeros(SHAPE, bool)
BOX[cen[0] - hb[0]:cen[0] + hb[0], cen[1] - hb[1]:cen[1] + hb[1], cen[2] - hb[2]:cen[2] + hb[2]] = True
BOX_MM = [2 * h_ * v for h_, v in zip(hb, VOXEL)]
record["zoom_box"] = dict(centre_ro_pe_spe=cen, extent_voxels=[2 * h_ for h_ in hb], extent_mm=BOX_MM)

# ---------------- rows ----------------
TOL = {"psnr": 1e-6, "ssim": 1e-7}
rows, checked, maxdiff = [], 0, {"psnr": 0.0, "ssim": 0.0}
for (axis, pname), idx in zip(PLANES, slices):
    sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
    bm = D(BOX[sl], axis)
    rr, cc = np.nonzero(bm)
    zoom = dict(x0=int(cc.min()), y0=int(rr.min()), w=int(cc.max() - cc.min() + 1), h=int(rr.max() - rr.min() + 1))
    for c in CONTRASTS:
        g2, m2 = gt[c][sl], head[c][sl]
        num = {lab: plane_metrics(vol[c][lab][sl], g2, m2, gmax[c]) for lab in LABELS}
        e = M[c].get((pname.lower(), idx))
        if e is None:
            log(f"{pname} {idx} {c}: not in the topic's per-slice record (head fraction below its threshold); numbers not cross-checked")
        else:
            for lab in LABELS[1:]:                       # ZF is not in the record
                for m in ("psnr", "ssim"):
                    ref = (e.get(m) or {}).get(KEY[c][lab])
                    if ref is None:
                        continue
                    d = abs(num[lab][m] - ref)
                    maxdiff[m] = max(maxdiff[m], d); checked += 1
                    assert d <= TOL[m], f"{pname} {idx} {c} {lab} {m}: drawn {num[lab][m]} vs recorded {ref}"
        best = {m: (min if m == "nmse" else max)(LABELS, key=lambda lab: num[lab][m]) for m in METRICS}
        margin = {}
        for m in METRICS:
            opp = [num[lab][m] for lab in LABELS[:-1]]
            margin[m] = (min(opp) - num[MC][m]) if m == "nmse" else (num[MC][m] - max(opp))
        rows.append(dict(axis=axis, pname=pname, idx=idx, c=c, sl=sl, g2=g2, emask=head_region(m2), num=num, best=best,
                         margin=margin, zoom=zoom, vmax=float(np.quantile(g2, 0.999)), med=float(np.median(g2[m2]))))
log(f"{checked} panel numbers match logs/find_slice_all_first_*.json (max |diff| PSNR {maxdiff['psnr']:.2e} dB, SSIM {maxdiff['ssim']:.2e})")
record["check"] = dict(n_compared=checked, max_abs_diff=maxdiff, tolerance=TOL)

for c in CONTRASTS:      # house grey windows: slice P99.9; the sagittal row at the other planes' in-head median level
    other = [r["med"] / r["vmax"] for r in rows if r["c"] == c and r["axis"] != 1]
    for r in rows:
        if r["c"] == c and r["axis"] == 1 and other:
            r["vmax"] = r["med"] / float(np.mean(other))
for r in rows:
    r["vmax"] /= args.brightness

# ---------------- figure (geometry of fig_lesion_plane_prepost.render, zoom rows, single bars) ----------------
ncol = len(LABELS) + 1
pw, gap, zgap = 2.6, 0.025, 0.02
CBTRIM, top_in, cbpad, cbw, cbroom = 0.10, 1.35, 0.06, 0.13, 0.95
half = (pw - zgap) / 2
hs, ehs = [], []
for r in rows:
    H, W = phys(r["axis"], D(r["g2"], r["axis"]).shape)
    hs.append(pw * H / W)
    bh, bw = phys(r["axis"], (r["zoom"]["h"], r["zoom"]["w"]))
    ehs.append(half * bh / bw)                     # the magnified box keeps its true proportions
fw = ncol * pw + (ncol - 1) * gap + cbpad + cbw + cbroom
fh = top_in + sum(hs) + sum(ehs) + (2 * len(rows) - 1) * gap + 0.16
fig = plt.figure(figsize=(fw, fh))


def crop(img, z):
    return img[z["y0"]:z["y0"] + z["h"], z["x0"]:z["x0"] + z["w"]]


def zframe(a):
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_edgecolor(ZC); sp.set_linewidth(ZLW)


def axes_at(x0, y0, w, hh):
    a = fig.add_axes([x0 / fw, 1 - (y0 + hh) / fh, w / fw, hh / fh])
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)
    return a


def label(a, text, bold=False, line=0, fs=12, hh=None, color="white"):
    dy = line * fs * 1.28 / 72 / hh if line else 0.0
    t = a.text(0.025, 0.975 - dy, text, transform=a.transAxes, ha="left", va="top", color=color, fontsize=fs,
               fontweight="bold" if bold else "normal")
    t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])


def outline(a, z):
    a.add_patch(Rectangle((z["x0"] - 0.5, z["y0"] - 0.5), z["w"], z["h"], fill=False, ec=ZC, lw=ZLW))


IMKW = dict(origin="upper", aspect="auto")
y, im = top_in, None
for k, (r, h, eh) in enumerate(zip(rows, hs, ehs)):
    c, axis, z = r["c"], r["axis"], r["zoom"]
    gkw = dict(cmap="gray", vmin=0, vmax=r["vmax"], **IMKW)
    for j in range(ncol):
        x = j * (pw + gap)
        a = axes_at(x, y, pw, h)
        bL = axes_at(x, y + h + gap, half, eh)
        bR = axes_at(x + half + zgap, y + h + gap, half, eh)
        if j == 0:
            a.imshow(D(r["g2"], axis), **gkw)
            outline(a, z)
            label(a, r["pname"], bold=True, fs=13)
            label(a, SHOW[c], bold=True, line=1, fs=13, hh=h)
            bL.imshow(crop(D(r["g2"], axis), z), **gkw)
            zframe(bL)
            bR.set_facecolor("black")
            kplane = D(mask3.take(kcen[axis], axis=axis), axis)
            if axis == 1:
                kplane = np.rot90(kplane, args.sag_mask_rot)
            bR.imshow(kplane, cmap="gray", vmin=0, vmax=1, interpolation="nearest", **IMKW)
            if k == 0:
                a.set_title("GT", fontsize=args.title_fs, pad=4)
            continue
        lab = LABELS[j - 1]
        v = vol[c][lab][r["sl"]]
        a.imshow(D(v, axis), **gkw)
        outline(a, z)
        n = r["num"][lab]
        for i, (txt, met) in enumerate(((f"{n['psnr']:.2f} dB", "psnr"), (f"SSIM {n['ssim']:.4f}", "ssim"),
                                         (f"NMSE {n['nmse']:.5f}", "nmse"))):
            label(a, txt, bold=lab == r["best"][met], line=i, fs=10.5, hh=h, color=METRIC_COLOR)
        emap = D(np.abs(v - r["g2"]) / gmax[c] * r["emask"], axis)
        bL.imshow(crop(D(v, axis), z), **gkw)
        im = bR.imshow(crop(emap, z), cmap="jet", vmin=0.0, vmax=args.err_vmax, **IMKW)
        zframe(bL); zframe(bR)
        if k == 0:
            a.set_title(lab if "SPIRiT" in lab else lab.upper(), fontsize=args.title_fs, pad=4)
    if k == 0:           # one grey bar in units of each row's own window, one error bar (same scale in every row)
        x_cb = (ncol * pw + (ncol - 1) * gap + cbpad) / fw
        e0 = y + h + gap + 0.04
        gsm = ScalarMappable(norm=Normalize(0, 1), cmap="gray")
        for handle, t0, t1, tk, lab_ in ((gsm, y, y + h - CBTRIM, np.linspace(0, 1, 5), "intensity / row window"),
                                         (im, e0, y + h + gap + eh, np.arange(0, 5) * args.err_vmax / 4, "|error| / max GT")):
            cax = fig.add_axes([x_cb, 1 - t1 / fh, cbw / fw, (t1 - t0) / fh])
            cb = fig.colorbar(handle, cax=cax, ticks=tk)
            cb.ax.tick_params(labelsize=9.5)
            cb.set_label(lab_, fontsize=10)
    y += h + eh + 2 * gap

t_, c_, s_ = slices
fig.suptitle(f"16yo TSC MTP, T1w + T2*   AF 12 of the full matrix, {AF_OWN:.2f}x of its acquired k-space   --   "
             f"Transverse {t_} / Coronal {c_} / Sagittal {s_}, T1w and T2*   --   MC-GS = {MCGS_KEY} (phase TV, 8000 it.)",
             fontsize=13, fontweight="bold", y=1 - 0.10 / fh, va="top")
win = "slice P99.9" + (f" / {args.brightness:g}" if args.brightness != 1 else "")
cap = ["head-masked slice PSNR / SSIM (7x7 window) / NMSE in each panel = computed from the topic's registered volumes, "
       "the best of each row in bold",
       f"error |recon - GT| / max(GT), jet [0, {args.err_vmax:g}]   (inside the head only)",
       "row under each image row: left = the magnified box of the image, right = the same box of the error map; "
       "GT column: GT box | sampling mask in that plane (k-space"
       + (f", sagittal turned {90 * args.sag_mask_rot} deg)" if args.sag_mask_rot else ")"),
       f"AF = {NPL} plane points / {NSAMP} sampled points, the same mask for T1w and T2*",
       f"grey bar: intensity as a fraction of each row's own window ({win})"]
cap2 = (f"in every panel, the {BOX_MM[0]:.1f} x {BOX_MM[1]:.1f} x {BOX_MM[2]:.1f} mm box ({2 * hb[0]} x {2 * hb[1]} x {2 * hb[2]} voxels) "
        f"centred at RO {cen[0]} / PE {cen[1]} / SPE {cen[2]} is outlined on the slice and magnified in the row below")
fig.text(0.5, 1 - 0.40 / fh, ";   ".join(cap[:2]) + "\n" + ";   ".join(cap[2:]) + "\n" + cap2, ha="center", va="top", fontsize=11)

stem = f"peds_af12_three_planes_{t_}_{c_}_{s_}_T1w_T2s_prepost_format{args.tag}"
fig.savefig(art / f"{stem}.png", dpi=args.png_dpi)
if args.pdf:
    for im_ in fig.findobj(mimage.AxesImage):
        im_.set_rasterized(True)
    fig.savefig(art / f"{stem}.pdf", dpi=args.pdf_dpi)
plt.close(fig)
from PIL import Image  # noqa: E402
Image.MAX_IMAGE_PIXELS = None
with Image.open(art / f"{stem}.png") as I:
    I.convert("RGB").resize((2000, round(2000 * I.height / I.width)), Image.LANCZOS).save(art / f"{stem}_preview.jpg", quality=90)

log("\nMC-GS on each row (rank among the 11 drawn arms; margin over the best other drawn arm):")
all_first = True
for r in rows:
    rank = {m: 1 + sorted(LABELS, key=lambda lab: r["num"][lab][m], reverse=m != "nmse").index(MC) for m in METRICS}
    all_first &= all(v == 1 for v in rank.values())
    record["rows"].append(dict(plane=r["pname"], index=r["idx"], contrast=r["c"], numbers=r["num"], best=r["best"],
                               mcgs_rank=rank, mcgs_margin=r["margin"], window=r["vmax"]))
    n = r["num"][MC]
    log(f"  {r['pname']:<10} {r['idx']:>3} {SHOW[r['c']]:<3}  PSNR {n['psnr']:.3f} dB #{rank['psnr']} ({r['margin']['psnr']:+.3f})"
        f"  SSIM {n['ssim']:.4f} #{rank['ssim']} ({r['margin']['ssim']:+.4f})  NMSE {n['nmse']:.5f} #{rank['nmse']} ({r['margin']['nmse']:+.5f})")
record["mcgs_first_on_every_row_and_metric"] = bool(all_first)
record["figure"] = f"{stem}.png"
(art / f"record_{stem}.json").write_text(json.dumps(record, indent=2))
log("MC-GS first on PSNR, SSIM and NMSE in all six rows:", all_first)
log("wrote", art / f"{stem}.png")
log("DONE")
