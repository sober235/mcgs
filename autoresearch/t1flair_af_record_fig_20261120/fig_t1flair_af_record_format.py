#!/usr/bin/env python3
"""fig_t1flair_af_record_format.py -- the T1 + FLAIR acceleration study (jiangyizhou, AF 12 / 16 / 20 / 24) redrawn in
the format of the plain-scan / enhanced figure of record, ONE FIGURE PER ACCELERATION FACTOR (user, 2026-11-20:
"redraw this figure, arranging the figures of different acceleration ratios in the same style as the plain scan and
enhancement ones. Each acceleration is one picture.").

Format (fig_lesion_plane_prepost.py --three --err_layout zoom --cbar single, as fig_peds_prepost_format.py and
uniform_af16_fig_20261112/fig_t1flair_record_format.py): T1 and FLAIR take the places of plain scan and enhanced.
Rows: transverse T1, transverse FLAIR, coronal T1, coronal FLAIR, sagittal T1, sagittal FLAIR. Under each image row a
zoom row (left = the magnified box of the image, right = the same box of the error map; GT column: GT box | the
sampling mask in that plane). Head-masked slice PSNR / SSIM / NMSE in every panel, best of each row in bold; one grey
bar and one error bar.

What stays as in the figures of record of these four topics (Results_20260908/T1FLAIR_AF12_16_20_24_figures_hires,
scripts/render_hires_final.sh): slices 180 / 124 / 76, the 48-voxel box where they meet, the twelve columns (GT, ZF,
CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR, GS-single, MC-GS), the registered arms
(AF12: gs_*_p14, AF16 / 20 / 24: gs_*_p3; swept arms at the lambda of logs/summary.json), transverse eyes up,
origin lower, grey window = slice P99.9 with the sagittal row at the in-head median level of the other two planes,
error = |recon - GT| / max GT, jet [0, 0.1], inside the head only; head = |GT| > 0.05 P99.9 of the volume, PSNR at
max|GT|, 7x7 uniform SSIM map averaged over the head pixels, head NMSE. AF label = whole phase-encode plane / sampled
points of the one mask both contrasts share (mask2d_AF_rect of logs/prep_04_export.json, the house gauge).

Checks: every volume's head PSNR is recomputed and compared with the topic's logs/summary.json (stops above 0.005 dB).
Read-only on the topics; writes PNG, a 2000-px JPEG preview and a JSON record (SHA-256 of every volume, all panel
numbers) into MCGS_ARTIFACT_DIR. CPU only. Neither GT nor any ROI enters a reconstruction.

  gpu.py run --cpu -- @python autoresearch/t1flair_af_record_fig_20261120/fig_t1flair_af_record_format.py --af 16
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--af", type=int, required=True, choices=[12, 16, 20, 24])
ap.add_argument("--slices", type=int, nargs=3, default=[180, 124, 76], metavar=("TRA", "COR", "SAG"))
ap.add_argument("--box", type=int, default=48, help="edge of the zoom cube in voxels, centred where the slices meet")
ap.add_argument("--box_centres", type=int, nargs=6, default=None, metavar="V",
                help="one zoom box per plane instead of the cube where the slices meet: the box centre on the two "
                     "in-plane volume axes, for transverse (axes 1 2), coronal (axes 0 2), sagittal (axes 0 1); "
                     "e.g. the positions find_zoom_box.py reports (user, 2026-11-21)")
ap.add_argument("--err_vmax", type=float, default=0.1)
ap.add_argument("--zoom_err_smooth", type=int, default=0,
                help="edge of a local-mean window applied to the error map of the zoom rows (0 = none; 5 with "
                     "--err_vmax 0.03 is the most legible setting of zoom_presentation_test.py, user 2026-11-21)")
ap.add_argument("--err_cmap", default="jet", help="colour map of the error maps (jet = the record format; e.g. hot)")
ap.add_argument("--arrow", action="store_true",
                help="one arrow per row, at the same place in every zoom panel: where the error of the other methods "
                     "(median over the ten reconstructions other than ZF and MC-GS, 7x7 local mean) exceeds MC-GS's "
                     "most (user, 2026-11-21)")
ap.add_argument("--arrow_color", default="#00E5FF")
ap.add_argument("--arrow_mode", choices=["gap", "edge"], default="gap",
                help="gap: the point of the largest error gap anywhere in the box; edge: the same, restricted to tissue "
                     "boundaries (the fifth of the box with the steepest GT gradient), so the arrow marks a boundary "
                     "(user, 2026-11-21)")
ap.add_argument("--role", default="", help="what the figure stands for in the article, put at the head of the title")
ap.add_argument("--save_data", action="store_true",
                help="also write <stem>_data.npz: every slice the figure draws (each method, GT, head mask, error-display "
                     "region, sampling mask plane) in volume orientation, with the box / arrow positions (user, 2026-11-21)")
ap.add_argument("--arrow_on", choices=["zoom", "main"], default="zoom",
                help="draw the arrow in the zoom panels, or on the slice images above them (user, 2026-11-21)")
ap.add_argument("--title_fs", type=float, default=22.0)
ap.add_argument("--png_dpi", type=int, default=200)
ap.add_argument("--pdf", action="store_true",
                help="also write a PDF: text, frames and bars vector, every image panel embedded at --pdf_dpi")
ap.add_argument("--pdf_dpi", type=int, default=600)
ap.add_argument("--save_dir", default="",
                help="after drawing, also copy PNG / PDF / preview / record JSON and this script into this server folder "
                     "(created if missing; an existing file is never overwritten with different content)")
ap.add_argument("--tol_db", type=float, default=0.005, help="largest allowed |volume head PSNR - summary.json|")
ap.add_argument("--single", choices=["registered", "indep"], default="registered",
                help="GS-single column: 'registered' = the topic's gs_single_<key> arm (the joint trainer fed the same "
                     "scan in both channels); 'indep' = the genuinely independent single-contrast model "
                     "(SingleContrastTrainer, autoresearch/t1flair_af_indep_single_20261121; AF16: fw_t1flair_nojoint_* of "
                     "mcgs_framework_ablation_20261110), checked against its own score json (user, 2026-11-21)")
args = ap.parse_args()

import h5py  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_opening, gaussian_filter, uniform_filter  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402

os.nice(10)
R = Path(f"/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF{args.af}_v3_jiangyizhou")
GSKEY = "p14" if args.af == 12 else "p3"          # the registered arms of the figures of record (render_hires_final.sh)
CONTRASTS = ("T1", "FLAIR")
MC = "MC-GS"
PLANES = [(0, "Transverse"), (1, "Coronal"), (2, "Sagittal")]
METRICS = ("psnr", "ssim", "nmse")
METRIC_COLOR = "#FFC23D"
ZC, ZLW = METRIC_COLOR, 2.2
log = lambda *a: print(*a, flush=True)
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", "t1flair_af_record_out")).resolve()
art.mkdir(parents=True, exist_ok=True)
assert R.is_dir(), f"no topic {R}"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def D(x, axis):
    """the topics' display transform (fig_three_planes_compact.py / _inset.py, --tra_rot 1: eyes up)"""
    return np.rot90(x.T, 1) if axis == 0 else x


# ---------------- which volume fills which column (as the topics' own figure scripts) ----------------
def sources(c):
    rows = {r["method"]: r for r in json.load(open(R / "logs/summary.json")) if r["contrast"] == c}

    def lam_file(key, prefix):
        lam = rows[key]["lam"]
        hit = [p for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy")
               if (m := re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name))
               and abs(float(m.group(1)) - float(lam)) < 1e-12]
        assert len(hit) == 1, (c, key, lam, hit)
        return hit[0], f"lam {float(lam):g}"

    sp = rows["spirit"]
    cols = [("ZF", "zf", (R / f"recon/zf_{c}_complex.npy", "")),
            ("CG-SENSE", "cgsense", (R / f"recon/cgsense_{c}_complex.npy", "")),
            ("CS-SENSE", "cssense_o200", lam_file("cssense_o200", "cssense_o200")),
            ("Joint SENSE", "jcssense", lam_file("jcssense", "jcssense_o200")),
            ("L1-SPIRiT", "spirit", (R / f"recon/spirit_ks{sp['ksize']}_lam{sp['lam']:g}_{c}_complex.npy",
                                     f"ks {sp['ksize']} lam {sp['lam']:g}")),
            ("ZS-SSDU", "zsssdu", (R / f"recon/zsssdu_{c}_complex.npy", "")),
            ("3D-DIP", "dip", (R / f"recon/dip_{c}_complex.npy", "")),
            ("3D-INR", "inr", (R / f"recon/inr_{c}_complex.npy", "")),
            ("MC-INR", "mcinr", (R / f"recon/mcinr_{c}_complex.npy", "")),
            ("GS-single", f"gs_single_{GSKEY}", (R / f"recon/gs_single_{GSKEY}_{c}_mag.npy", GSKEY)),
            (MC, f"gs_joint_{GSKEY}", (R / f"recon/gs_joint_{GSKEY}_{c}_mag.npy", GSKEY))]
    out = {}
    for lab, key, (p, note) in cols:
        assert p.exists(), f"{c} {lab}: no volume {p}"
        out[lab] = dict(path=p, key=key, note=note, logged=rows[key]["head_psnr"])
    if args.single == "indep":       # the independent single-contrast run: plain channel of its own npz, its own score
        stem_ = ARC_FW / f"fw_t1flair_nojoint_{c}" if args.af == 16 else ARC_IS / f"is_af{args.af}_indep_{c}"
        p, sj = Path(f"{stem_}_volumes.npz"), json.load(open(f"{stem_}_score.json"))
        assert p.exists() and sj["plain_name"] == c and sj["contrast"] is None, (c, p)
        assert "single" in sj["argv"][sj["argv"].index("--trainer") + 1]
        out["GS-single"] = dict(path=p, key=stem_.name, note="independent (SingleContrastTrainer)",
                                logged=sj["plain"]["head_psnr"], npz_key="rec")
    return out


ARC_IS = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_t1flair_af_indep_single_20261121/trainer/results_arc")
ARC_FW = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_framework_ablation_20261110/trainer/results_arc")


def load_vol(s):
    if "npz_key" in s:
        with np.load(s["path"]) as z:
            return np.abs(np.asarray(z[s["npz_key"]])).astype(np.float32)
    return np.abs(np.load(s["path"])).astype(np.float32)


SRC = {c: sources(c) for c in CONTRASTS}
LABELS = list(SRC["T1"])
EX = json.load(open(R / "logs/prep_04_export.json"))

# ---------------- volumes, checked against the topic's summary ----------------
gt, gmax, head, vol = {}, {}, {}, {c: {} for c in CONTRASTS}
record = dict(topic=str(R), af_nominal=args.af, gs_key=GSKEY,
              format="fig_lesion_plane_prepost.py --three --err_layout zoom --cbar single (T1 / FLAIR in place of pre / post)",
              slices=dict(zip(("transverse", "coronal", "sagittal"), args.slices)),
              volumes={c: {} for c in CONTRASTS}, gt_sha256={}, rows=[])
worst = 0.0
log("volume head PSNR (recomputed) | logs/summary.json head_psnr | source")
for c in CONTRASTS:
    g = R / f"recon/gt_{c}_complex.npy"
    gt[c] = np.abs(np.load(g)).astype(np.float32)
    gmax[c] = float(gt[c].max())
    head[c] = gt[c] > 0.05 * float(np.quantile(gt[c], 0.999))
    record["gt_sha256"][c] = sha(g)
    gh = gt[c][head[c]].astype(np.float64)
    for lab in LABELS:
        s = SRC[c][lab]
        v = load_vol(s)
        assert v.shape == gt[c].shape, (c, lab, v.shape)
        vol[c][lab] = v
        e = v[head[c]].astype(np.float64) - gh
        mine = float(20 * np.log10(gmax[c]) - 10 * np.log10(float((e * e).mean())))
        d = mine - s["logged"]
        worst = max(worst, abs(d))
        record["volumes"][c][lab] = dict(path=str(s["path"]), summary_key=s["key"], note=s["note"], sha256=sha(s["path"]),
                                         volume_head_psnr=mine, summary_head_psnr=s["logged"])
        log(f"  {c:<5} {lab:<12} {mine:8.3f} | {s['logged']:8.3f} ({d:+.4f}) | {s['path'].name}  {s['note']}")
record["max_abs_diff_vs_summary_head_psnr"] = worst
log(f"largest |difference| to logs/summary.json: {worst:.5f} dB")
assert worst <= args.tol_db, f"a volume does not reproduce its registered head PSNR ({worst:.4f} dB)"
SHAPE = gt["T1"].shape
assert gt["FLAIR"].shape == SHAPE

with h5py.File(R / f"data/mask_af{args.af}_3d.h5", "r") as f:
    mask3 = np.array(f["mask"]) > 0
assert mask3.shape == SHAPE and np.all(mask3 == mask3[:1]), "expected a 3-D mask, readout (axis 0) fully sampled"
NPL, NSAMP = mask3.shape[1] * mask3.shape[2], int(mask3[0].sum())
assert NSAMP == EX["mask2d_samples"], (NSAMP, EX["mask2d_samples"])
dens = uniform_filter(mask3[0].astype(float), size=9, mode="constant")
kcen = (mask3.shape[0] // 2, *(int(v) for v in np.unravel_index(np.argmax(dens), dens.shape)))
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


# ---------------- zoom box: the cube where the three slices meet ----------------
hw = args.box // 2
for idx, n in zip(args.slices, SHAPE):
    assert 0 <= idx - hw and idx + hw <= n, f"the {args.box}-voxel cube around {args.slices} leaves the volume {SHAPE}"
BOX = np.zeros(SHAPE, bool)
BOX[tuple(slice(i - hw, i + hw) for i in args.slices)] = True
record["zoom_box"] = dict(centre=list(args.slices), edge_voxels=args.box)
if args.box_centres:
    record["zoom_box"] = dict(edge_voxels=args.box, per_plane_centres=dict(
        transverse=args.box_centres[0:2], coronal=args.box_centres[2:4], sagittal=args.box_centres[4:6]),
        note="chosen with find_zoom_box.py against GT, for display only")

# ---------------- rows ----------------
rows = []
for (axis, pname), idx in zip(PLANES, args.slices):
    sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
    box2 = BOX[sl]
    if args.box_centres:
        ca, cb = args.box_centres[2 * axis:2 * axis + 2]
        box2 = np.zeros_like(box2)
        assert hw <= ca <= box2.shape[0] - hw and hw <= cb <= box2.shape[1] - hw, (pname, ca, cb, box2.shape)
        box2[ca - hw:ca + hw, cb - hw:cb + hw] = True
    rr, cc = np.nonzero(D(box2, axis))
    zoom = dict(x0=int(cc.min()), y0=int(rr.min()), w=int(cc.max() - cc.min() + 1), h=int(rr.max() - rr.min() + 1))
    for c in CONTRASTS:
        g2, m2 = gt[c][sl], head[c][sl]
        num = {lab: plane_metrics(vol[c][lab][sl], g2, m2, gmax[c]) for lab in LABELS}
        best = {m: (min if m == "nmse" else max)(LABELS, key=lambda lab: num[lab][m]) for m in METRICS}
        margin = {}
        for m in METRICS:
            opp = [num[lab][m] for lab in LABELS if lab != MC]
            margin[m] = (min(opp) - num[MC][m]) if m == "nmse" else (num[MC][m] - max(opp))
        rows.append(dict(axis=axis, pname=pname, idx=idx, c=c, sl=sl, g2=g2, emask=head_region(m2), num=num, best=best,
                         margin=margin, zoom=zoom, vmax=float(np.quantile(g2, 0.999)), med=float(np.median(g2[m2]))))
for c in CONTRASTS:      # house grey windows: slice P99.9; the sagittal row at the other planes' in-head median level
    other = [r["med"] / r["vmax"] for r in rows if r["c"] == c and r["axis"] != 2]
    for r in rows:
        if r["c"] == c and r["axis"] == 2:
            r["vmax"] = r["med"] / float(np.mean(other))

# ---------------- figure (geometry of fig_lesion_plane_prepost.render, zoom rows, single bars) ----------------
ncol = len(LABELS) + 1
pw, gap, zgap = 2.6, 0.025, 0.02
CBTRIM, cbpad, cbw, cbroom = 0.10, 0.06, 0.13, 0.95
half = (pw - zgap) / 2
hs, ehs = [], []
for r in rows:
    H, W = D(r["g2"], r["axis"]).shape
    hs.append(pw * H / W)
    ehs.append(half * r["zoom"]["h"] / r["zoom"]["w"])
fw = ncol * pw + (ncol - 1) * gap + cbpad + cbw + cbroom

t_, c_, s_ = args.slices
AFR = float(EX["mask2d_AF_rect"])
TITLE = (f"jiangyizhou T1 + FLAIR, vd-Poisson mask, nominal AF {args.af}, actual {AFR:.2f}x"
         f"   --   Transverse {t_} / Coronal {c_} / Sagittal {s_}")
if args.role:
    TITLE = (f"{args.role}: T1 + FLAIR, net acceleration {AFR:.2f}x (nominal AF {args.af}), vd-Poisson mask"
             f"   --   Transverse {t_} / Coronal {c_} / Sagittal {s_}")
fl = EX["contrasts"]["FLAIR"]
CAP = ["head-masked slice PSNR / SSIM (7x7 window) / NMSE in each panel, the best of each row in bold",
       f"error |recon - GT| / max(GT), {args.err_cmap} [0, {args.err_vmax:g}], inside the head only"
       + (f", shown as its {args.zoom_err_smooth}x{args.zoom_err_smooth} local mean" if args.zoom_err_smooth else ""),
       "row under each image row: left = the magnified box of the image, right = the same box of the error map; "
       "GT column: GT box | sampling mask in that plane (k-space)",
       f"AF = {NPL} plane points / {NSAMP} sampled points, one mask for both contrasts; the FLAIR scan acquired "
       f"{round(fl['support_frac'] * NPL)} of the {NPL} lines, so {round(fl['sampled_frac_of_plane'] * NPL)} sampled points carry FLAIR data",
       (f"MC-GS / GS-single = the registered {GSKEY} arms" if args.single == "registered" else
        f"MC-GS = the registered {GSKEY} arm; GS-single = an independent single-contrast model per contrast (own geometry, "
        f"no cross-contrast term, the {GSKEY} recipe with that contrast's own wavelet weight, 3.6M Gaussians)")
       + "; CS-SENSE (converged TV), Joint SENSE and L1-SPIRiT at the lambda of the topic summary",
       (f"the {args.box}-voxel box where the three slices meet is outlined on every slice" if not args.box_centres else
        f"one {args.box}-voxel box per plane (the same for T1 and FLAIR), placed where MC-GS's error is furthest below the "
        f"best other method (chosen against GT, for display only)")
       + "; grey bar: fraction of each row's own window (slice P99.9)"]
if args.arrow:
    CAP.append(("arrow (same place in every panel of a row):" if args.arrow_on == "zoom" else "arrow on each slice (same place along a row):") + " where the error of the other methods (median of the nine "
               "reconstructions other than ZF and MC-GS, 7x7 local mean) exceeds MC-GS's most"
               + (", among the tissue boundaries of the box (steepest fifth of the GT gradient)" if args.arrow_mode == "edge" else "")
               + "; chosen against GT, for display only")


def wrap(parts, width_in, fs):
    """join caption parts with ';   ' and break lines so that none is wider than the figure"""
    meas = plt.figure(figsize=(1, 1)); rend = meas.canvas.get_renderer()
    lines, cur = [], ""
    for p in parts:
        trial = (cur + ";   " + p) if cur else p
        if cur and meas.text(0, 0, trial, fontsize=fs).get_window_extent(rend).width > 0.97 * width_in * 100.0:
            lines.append(cur); cur = p
        else:
            cur = trial
    plt.close(meas)
    return lines + [cur]


cap_lines = wrap(CAP, fw, 11)
cap_y = 0.10 + 13 * 1.35 / 72 + 0.06
top_in = cap_y + len(cap_lines) * 11 * 1.32 / 72 + 0.10 + (args.title_fs * 1.2 + 4) / 72
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


# no rasterized=True: the PDF backend then embeds each image at --pdf_dpi over its own panel only, instead of a
# full-figure raster buffer per panel (the 600-dpi house renders of these topics needed ~100 GB of RAM that way)
IMKW = dict(origin="lower", aspect="auto")
ARROW_C = args.arrow_color


def arrow_at(a, pt, z, main=False, shape=None):
    """an arrow whose head sits at pt (crop pixel x, y); zoom panel: tail towards the centre of the box;
    slice image (main): head at the same anatomical point, tail towards the centre of the image"""
    x, y = pt
    if main:
        x, y = x + z["x0"], y + z["y0"]
        v, L = np.array([shape[1] / 2 - x, shape[0] / 2 - y], float), 0.11 * shape[1]
    else:
        v, L = np.array([z["w"] / 2 - x, z["h"] / 2 - y], float), 13.0
    n = float(np.hypot(*v))
    v = v / n * L if n > 4 else np.array([L, L]) / 1.4
    a.annotate("", xy=(x, y), xytext=(x + v[0], y + v[1]),
               arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=2.2 if main else 1.8, mutation_scale=15 if main else 13,
                               shrinkA=0, shrinkB=1))


if args.arrow:
    for r in rows:
        c, axis, z = r["c"], r["axis"], r["zoom"]
        sm = {lab: uniform_filter(np.abs(vol[c][lab][r["sl"]] - r["g2"]) / gmax[c] * r["emask"], size=7, mode="nearest")
              for lab in LABELS if lab != "ZF"}
        sc = np.median([sm[lab] for lab in sm if lab != MC], axis=0) - sm[MC]
        sc = D(sc, axis)[z["y0"]:z["y0"] + z["h"], z["x0"]:z["x0"] + z["w"]].copy()
        m = 7
        sc[:m] = sc[-m:] = -np.inf; sc[:, :m] = sc[:, -m:] = -np.inf
        if args.arrow_mode == "edge":
            gy, gx = np.gradient(gaussian_filter(r["g2"] / gmax[c], 1.0))
            gm = D(np.hypot(gx, gy), axis)[z["y0"]:z["y0"] + z["h"], z["x0"]:z["x0"] + z["w"]]
            sc[gm < np.percentile(gm, 80)] = -np.inf
            r["arrow_edge_grad"] = float(gm[np.unravel_index(np.argmax(sc), sc.shape)])
        yy, xx = np.unravel_index(np.argmax(sc), sc.shape)
        r["arrow"] = (int(xx), int(yy))
        r["arrow_gap"] = float(sc[yy, xx])
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
            label(a, c, bold=True, line=1, fs=13, hh=h)
            bL.imshow(crop(D(r["g2"], axis), z), **gkw)
            if args.arrow and args.arrow_on == "zoom":
                arrow_at(bL, r["arrow"], z)
            elif args.arrow:
                arrow_at(a, r["arrow"], z, True, D(r["g2"], axis).shape)
            zframe(bL)
            bR.set_facecolor("black")
            bR.imshow(D(mask3.take(kcen[axis], axis=axis), axis), cmap="gray", vmin=0, vmax=1, interpolation="nearest", **IMKW)
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
        emap = np.abs(v - r["g2"]) / gmax[c] * r["emask"]
        if args.zoom_err_smooth:
            emap = uniform_filter(emap, size=args.zoom_err_smooth, mode="nearest")
        emap = D(emap, axis)
        bL.imshow(crop(D(v, axis), z), **gkw)
        im = bR.imshow(crop(emap, z), cmap=args.err_cmap, vmin=0.0, vmax=args.err_vmax, **IMKW)
        if args.arrow and args.arrow_on == "zoom":
            arrow_at(bL, r["arrow"], z); arrow_at(bR, r["arrow"], z)
        elif args.arrow:
            arrow_at(a, r["arrow"], z, True, D(v, axis).shape)
        zframe(bL); zframe(bR)
        if k == 0:
            a.set_title(lab if "SPIRiT" in lab else lab.upper(), fontsize=args.title_fs, pad=4)
    if k == 0:           # one grey bar in units of each row's own window, one error bar (same scale in every row)
        x_cb = (ncol * pw + (ncol - 1) * gap + cbpad) / fw
        e0 = y + h + gap + 0.04
        gsm = ScalarMappable(norm=Normalize(0, 1), cmap="gray")
        for handle, t0, t1, tk, lab_ in ((gsm, y, y + h - CBTRIM, np.linspace(0, 1, 5), "intensity / row window"),
                                         (im, e0, y + h + gap + eh, np.arange(0, 5) * args.err_vmax / 4, "|error| / max GT" + (f"\n({args.zoom_err_smooth}x{args.zoom_err_smooth} mean)" if args.zoom_err_smooth else ""))):
            cax = fig.add_axes([x_cb, 1 - t1 / fh, cbw / fw, (t1 - t0) / fh])
            cb = fig.colorbar(handle, cax=cax, ticks=tk)
            cb.ax.tick_params(labelsize=9.5)
            cb.set_label(lab_, fontsize=10)
    y += h + eh + 2 * gap

fig.suptitle(TITLE, fontsize=13, fontweight="bold", y=1 - 0.10 / fh, va="top")
fig.text(0.5, 1 - cap_y / fh, "\n".join(cap_lines), ha="center", va="top", fontsize=11, linespacing=1.32)

stem = f"t1flair_af{args.af}_three_planes_{t_}_{c_}_{s_}_record_format" + ("_indep_single" if args.single == "indep" else "") \
    + ("_zoomsel" if args.box_centres else "") + (f"_err{args.err_vmax:g}" if args.err_vmax != 0.1 else "") \
    + (f"_mean{args.zoom_err_smooth}" if args.zoom_err_smooth else "") \
    + (f"_{args.err_cmap}" if args.err_cmap != "jet" else "") + ("_arrow" if args.arrow else "") + ("main" if args.arrow and args.arrow_on == "main" else "") \
    + ("_edge" if args.arrow and args.arrow_mode == "edge" else "")
if args.arrow:
    record["arrows"] = [dict(plane=r["pname"], contrast=r["c"], crop_xy=r["arrow"], gap=r["arrow_gap"]) for r in rows]
    record["arrow_mode"] = args.arrow_mode
record["error_display"] = dict(vmax=args.err_vmax, zoom_local_mean=args.zoom_err_smooth)
record["gs_single"] = args.single
fig.savefig(art / f"{stem}.png", dpi=args.png_dpi)
if args.pdf:
    fig.savefig(art / f"{stem}.pdf", dpi=args.pdf_dpi)
plt.close(fig)
from PIL import Image  # noqa: E402
Image.MAX_IMAGE_PIXELS = None
with Image.open(art / f"{stem}.png") as I:
    log("figure size (px):", I.size)
    I.convert("RGB").resize((2000, round(2000 * I.height / I.width)), Image.LANCZOS).save(art / f"{stem}_preview.jpg", quality=90)

log(f"\nMC-GS on each row (rank among the {len(LABELS)} arms; margin over the best other arm):")
all_first = True
for r in rows:
    rank = {m: 1 + sorted(LABELS, key=lambda lab: r["num"][lab][m], reverse=m != "nmse").index(MC) for m in METRICS}
    ru = {m: (min if m == "nmse" else max)((lab for lab in LABELS if lab != MC), key=lambda lab: r["num"][lab][m]) for m in METRICS}
    all_first &= all(v == 1 for v in rank.values())
    record["rows"].append(dict(plane=r["pname"], index=r["idx"], contrast=r["c"], numbers=r["num"], best=r["best"],
                               mcgs_rank=rank, mcgs_margin=r["margin"], best_other=ru, window=r["vmax"]))
    n = r["num"][MC]
    log(f"  {r['pname']:<10} {r['idx']:>3} {r['c']:<5}  PSNR {n['psnr']:.3f} dB #{rank['psnr']} ({r['margin']['psnr']:+.3f} vs {ru['psnr']})"
        f"  SSIM {n['ssim']:.4f} #{rank['ssim']} ({r['margin']['ssim']:+.4f})  NMSE {n['nmse']:.5f} #{rank['nmse']} ({r['margin']['nmse']:+.5f})")
record["mcgs_first_on_every_row_and_metric"] = bool(all_first)
record["figure"] = f"{stem}.png"
record["figure_pdf"] = f"{stem}.pdf" if args.pdf else None
record["pdf_image_dpi"] = args.pdf_dpi if args.pdf else None
record["png_dpi"] = args.png_dpi
(art / f"record_{stem}.json").write_text(json.dumps(record, indent=2))
log("MC-GS first on PSNR, SSIM and NMSE in all six rows:", all_first)
outs = [f"{stem}.png", f"{stem}_preview.jpg", f"record_{stem}.json"] + ([f"{stem}.pdf"] if args.pdf else [])
if args.save_data:
    dat = {}
    for r in rows:
        k = f"{r['pname'].lower()}{r['idx']}_{r['c']}"
        dat[f"{k}__GT"] = r["g2"]
        dat[f"{k}__head_mask"] = head[r["c"]][r["sl"]]
        dat[f"{k}__error_region"] = r["emask"]
        for lab in LABELS:
            dat[f"{k}__{lab.replace(' ', '_')}"] = vol[r["c"]][lab][r["sl"]]
    for (axis, pname), idx in zip(PLANES, args.slices):
        dat[f"{pname.lower()}_sampling_mask_plane"] = mask3.take(kcen[axis], axis=axis)
    dat["meta_json"] = np.array(json.dumps(dict(
        note="magnitude slices in volume orientation (display: transverse = rot90(x.T), others as stored, origin lower); "
             "error = |recon - GT| / gt_max inside error_region; metrics inside head_mask with PSNR peak gt_max",
        af_nominal=args.af, net_acceleration=AFR, slices=record["slices"], gt_max=gmax, zoom_box=record["zoom_box"],
        zoom_display_px={f"{r['pname']}_{r['c']}": r["zoom"] for r in rows}, arrows=record.get("arrows"),
        error_display=record["error_display"], error_cmap=args.err_cmap, volumes=record["volumes"])))
    np.savez_compressed(art / f"{stem}_data.npz", **dat)
    outs.append(f"{stem}_data.npz")
for name in outs:
    log(f"  sha256 {sha(art / name)}  {name}")
if args.save_dir:
    import shutil
    dst = Path(args.save_dir)
    dst.mkdir(parents=True, exist_ok=True)
    me = Path(__file__).resolve()
    for src in [art / n for n in outs] + [me]:
        t = dst / src.name
        if t.exists():
            assert sha(t) == sha(src), f"{t} exists with different content; not overwritten"
            log("  already there, identical:", t.name)
            continue
        shutil.copyfile(src, t)
        assert sha(t) == sha(src)
        log("  saved", t)
log("wrote", art / f"{stem}.png")
log("DONE")
