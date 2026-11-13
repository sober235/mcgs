#!/usr/bin/env python3
"""fig_masks_record_format.py -- the three sampling patterns of the diff-mask study at ONE transverse slice, in the
format of the plain-scan / enhanced figure of record
(ce_af16_lesion_figs_20261107/fig_lesion_plane_prepost.py --three --err_layout zoom --cbar single; the same geometry
as uniform_af16_fig_20261112/fig_t1flair_record_format.py and peds_af12_lead_fig_20261111/fig_peds_prepost_format.py).

User request 2026-11-12: "make the drawing styles of the above three masks consistent with the plain-scan and
enhancement methods" and show the slice with the butterfly area. The record figure has three planes x two contrasts;
here the three planes become the three sampling patterns, so the page keeps its six row pairs:

    vd-Poisson T1, vd-Poisson FLAIR, uniform density T1, uniform density FLAIR, pseudo-radial T1, pseudo-radial FLAIR

Under each image row a zoom row: left = the magnified box of the image, right = the same box of the error map; in the
GT column: GT box | that pattern's (ky, kz) sampling mask. Head-masked slice PSNR / SSIM / NMSE in every panel, the
best of each row in bold. One grey bar and one error bar. Both contrasts in one figure (the older three-pattern
figure, scripts/fig_masks_transverse_stack.py, is one figure per contrast with whole error rows and a bar per row).

Columns and numbers are those of scripts/fig_masks_transverse_stack.py: the volumes each topic registered
(logs/summary.json; swept arms at the lambda summarize.py kept; GS arms gs_single_p3 / gs_joint_p3), head =
|GT| > 0.05 P99.9 of the volume, PSNR at max|GT|, 7x7 SSIM map averaged over the head pixels, head NMSE; the slice is
shown transposed and turned a quarter (eyes up), origin lower. Default slice 168 with a 64 x 64 box at (J, K) =
(128, 88): the lateral ventricles, as the butterfly figure of 2026-11-12.

--scan LO HI adds a table over transverse slices LO..HI: on how many of the 18 conditions (3 patterns x 2 contrasts x
3 metrics) MC-GS is first, its smallest PSNR lead, and how much ventricle (dark T1 pixels) the box holds.

Read-only on the three topics; CPU only. Writes PNG, a reduced preview and a JSON record (every volume's SHA-256)
into MCGS_ARTIFACT_DIR; with --save_topic also the PNG into the uniform-density topic's figures/ (the home of the
cross-pattern figures). The GT is used for scoring and display only.

  gpu.py run --cpu -- @python autoresearch/diffmask_af16_masks_20261111/fig_masks_record_format.py [--tra 168]
                                                    [--box 64 --center 128 88] [--scan 150 185] [--save_topic]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--tra", type=int, default=168, help="transverse slice (axis 0); 168 = the house butterfly slice")
ap.add_argument("--box", type=int, default=64, help="edge of the zoom box in voxels")
ap.add_argument("--center", type=int, nargs=2, default=[128, 88], metavar=("J", "K"), help="box centre, volume axes 1 and 2")
ap.add_argument("--what", default="lateral ventricles, butterfly area", help="words after the slice number in the title")
ap.add_argument("--err_vmax", type=float, default=0.1)
ap.add_argument("--brightness", type=float, default=1.25, help="display only: window top = slice P99.9 / brightness")
ap.add_argument("--title_fs", type=float, default=22.0)
ap.add_argument("--png_dpi", type=int, default=200)
ap.add_argument("--mcgs_key", default="gs_joint_p3")
ap.add_argument("--single_key", default="gs_single_p3")
ap.add_argument("--scan", type=int, nargs=2, metavar=("LO", "HI"), help="also tabulate transverse slices LO..HI")
ap.add_argument("--no_fig", action="store_true", help="only the numbers (and --scan)")
ap.add_argument("--save_topic", action="store_true", help="also copy the PNG into the uniform-density topic's figures/")
ap.add_argument("--out_dir", help="a NEW server folder for a high-resolution copy: PNG at --png_dpi (+ PDF with --pdf), "
                                  "preview, record JSON, MANIFEST.json and this script; large files are written only there")
ap.add_argument("--pdf", action="store_true", help="also write a PDF (vector text, image panels at --pdf_dpi)")
ap.add_argument("--pdf_dpi", type=int, default=600)
ap.add_argument("--note", default="", help="free text for MANIFEST.json (who asked, local date)")
ap.add_argument("--tag", default="", help="suffix of the output file name")
args = ap.parse_args()

import h5py  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_opening  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402

os.nice(10)
RES = Path("/data0/congcong/code/GS/Results_20260908")
TOPICS = [("vd-Poisson", RES / "T1FLAIR_AF16_v3_jiangyizhou"),
          ("Uniform density", RES / "T1FLAIR_AF16_uniform_v3_jiangyizhou"),
          ("Pseudo-radial", RES / "T1FLAIR_AF16_radial_v3_jiangyizhou")]
OUT_TOPIC = RES / "T1FLAIR_AF16_uniform_v3_jiangyizhou" / "figures"
CONTRASTS = ("T1", "FLAIR")
MC = "MC-GS"
METRICS = ("psnr", "ssim", "nmse")
METRIC_COLOR = "#FFC23D"
ZC, ZLW = METRIC_COLOR, 2.2
DISK = np.hypot(*np.mgrid[-6:7, -6:7]) <= 6
log = lambda *a: print(*a, flush=True)
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", "masks_record_format_out")).resolve()
art.mkdir(parents=True, exist_ok=True)
NEED = sorted({args.tra} | (set(range(args.scan[0], args.scan[1] + 1)) if args.scan else set()))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def D(x):
    """display transform of a transverse slice (fig_masks_transverse_stack.py, --tra_rot 1): transpose, a quarter turn"""
    return np.rot90(x.T, 1)


def head_region(m):
    """display region of the error maps: filled head without thin protrusions + 2-px rim (house rule)"""
    return binary_dilation(binary_fill_holes(binary_opening(binary_fill_holes(m), structure=DISK)), iterations=2)


def arms_for(R, c):
    """the eleven reconstruction columns of one topic, as fig_masks_transverse_stack.py: (key, label, saved volume)"""
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
            (args.mcgs_key, MC, R / f"recon/{args.mcgs_key}_{c}_mag.npy")]
    missing = [lab for _, lab, p in arms if p is None or not Path(p).exists()]
    assert not missing, f"{R.name}: missing volumes for {c}: {missing}"
    return arms, rows


# ---------------- volumes: only the needed transverse slices are kept ----------------
GT, GMAX, HEAD = {}, {}, {}
SL = {}                                   # (pattern, contrast) -> label -> {slice index: 2-D magnitude}
KMASK, AF, NSAMP = {}, {}, {}
LABELS = None
record = dict(format="fig_lesion_plane_prepost.py --three --err_layout zoom --cbar single (three planes -> three sampling patterns)",
              slice=args.tra, zoom_box=dict(centre_JK=list(args.center), edge_voxels=args.box),
              topics={n: str(R) for n, R in TOPICS}, gt_sha256={}, volumes={}, rows=[])
worst = 0.0
log("volume head PSNR (this script) | the topic's logged head PSNR | file")
for name, R in TOPICS:
    assert R.is_dir(), f"no topic {R}"
    with h5py.File(R / "data/mask_af16_3d.h5", "r") as f:
        mask3 = np.array(f["mask"]) > 0
    assert np.all(mask3 == mask3[:1]), f"{R.name}: expected a 3-D mask with the readout (axis 0) fully sampled"
    KMASK[name], NSAMP[name] = mask3[mask3.shape[0] // 2], int(mask3[0].sum())
    exp = json.load(open(R / "logs/prep_04_export.json"))["contrasts"]
    record["volumes"][name] = {}
    for c in CONTRASTS:
        AF[(name, c)] = float(exp[c]["AF_vs_full_rect"])
        gp = R / f"recon/gt_{c}_complex.npy"
        g = np.abs(np.load(gp)).astype(np.float32)
        if c not in GT:
            GT[c], GMAX[c] = g, float(g.max())
            HEAD[c] = g > 0.05 * float(np.quantile(g, 0.999))
            record["gt_sha256"][c] = sha(gp)
        else:
            dg = float(np.abs(g - GT[c]).max())
            assert g.shape == GT[c].shape and dg <= 1e-6 * GMAX[c], f"{R.name} {c}: GT differs from the first topic's GT by {dg}"
        assert mask3.shape == GT[c].shape, (R.name, mask3.shape)
        arms, summ = arms_for(R, c)
        labs = [lab for _, lab, _ in arms]
        LABELS = LABELS or labs
        assert labs == LABELS
        SL[(name, c)], record["volumes"][name][c] = {}, {}
        hm = HEAD[c]
        gh = GT[c][hm].astype(np.float64)
        for key, lab, p in arms:
            v = np.abs(np.load(p)).astype(np.float32)
            assert v.shape == GT[c].shape, (name, c, lab, v.shape)
            e = v[hm].astype(np.float64) - gh
            mine = float(20 * np.log10(GMAX[c]) - 10 * np.log10(float((e * e).mean())))
            logged = summ.get(key, {}).get("head_psnr")
            d = None if logged is None else mine - float(logged)
            worst = max(worst, abs(d or 0.0))
            SL[(name, c)][lab] = {z: v[z].copy() for z in NEED}
            record["volumes"][name][c][lab] = dict(key=key, path=str(p), sha256=sha(p), volume_head_psnr=mine,
                                                   logged_head_psnr=logged, lam=summ.get(key, {}).get("lam"),
                                                   ksize=summ.get(key, {}).get("ksize"))
            log(f"  {name:<16} {c:<5} {lab:<12} {mine:8.3f} | " + ("    n/a " if d is None else f"{float(logged):8.3f} ({d:+.4f})")
                + f" | {Path(p).name}")
            del v
record["max_abs_diff_vs_logged_head_psnr"] = worst
log(f"largest |difference| to the logged head PSNR: {worst:.4f} dB"
    + ("" if worst < 0.02 else "   <-- WARNING: some column differs from the number its topic logged"))
SHAPE = GT["T1"].shape

hw, (J, K) = args.box // 2, args.center
BM = np.zeros(SHAPE[1:], bool)
BM[J - hw:J + hw, K - hw:K + hw] = True
assert BM.sum() == args.box ** 2, "the zoom box leaves the slice"
rr, cc = np.nonzero(D(BM))
ZOOM = dict(x0=int(cc.min()), y0=int(rr.min()), w=int(cc.max() - cc.min() + 1), h=int(rr.max() - rr.min() + 1))


def plane_metrics(rec, truth, mask, gm):
    t, v = truth.astype(np.float64), rec.astype(np.float64)
    _, smap = structural_similarity(t, v, data_range=gm, full=True, win_size=7)
    e = v[mask] - t[mask]
    return {"psnr": float(20 * np.log10(gm) - 10 * np.log10(float((e * e).mean()))),
            "ssim": float(smap[mask].mean()),
            "nmse": float((e * e).sum() / (t[mask] ** 2).sum())}


def stats(z):
    """the six rows of slice z: numbers of every column, the best column per metric, MC-GS's lead, the box PSNR"""
    out = []
    for name, _ in TOPICS:
        for c in CONTRASTS:
            g2, m2 = GT[c][z], HEAD[c][z]
            num = {lab: plane_metrics(SL[(name, c)][lab][z], g2, m2, GMAX[c]) for lab in LABELS}
            best = {m: (min if m == "nmse" else max)(LABELS, key=lambda lab: num[lab][m]) for m in METRICS}
            other = {m: (min if m == "nmse" else max)((lab for lab in LABELS if lab != MC), key=lambda lab: num[lab][m]) for m in METRICS}
            lead = {m: (num[other[m]][m] - num[MC][m]) if m == "nmse" else (num[MC][m] - num[other[m]][m]) for m in METRICS}
            bz = BM & m2
            g64 = g2.astype(np.float64)
            bp = {lab: float(20 * np.log10(GMAX[c]) - 10 * np.log10(float(((SL[(name, c)][lab][z][bz].astype(np.float64) - g64[bz]) ** 2).mean())))
                  for lab in LABELS} if bz.any() else {}
            bo = max((lab for lab in LABELS if lab != MC), key=bp.get) if bp else None
            out.append(dict(name=name, c=c, z=z, g2=g2, m2=m2, num=num, best=best, other=other, lead=lead, box_psnr=bp,
                            box_other=bo, box_lead=(bp[MC] - bp[bo]) if bp else None, box_head_frac=float(bz.sum() / BM.sum())))
    return out


rows = stats(args.tra)
for r in rows:
    r["emask"] = head_region(r["m2"])
    r["vmax"] = float(np.quantile(r["g2"], 0.999)) / args.brightness

log(f"\ntransverse {args.tra}: MC-GS on each row (lead over the best other column; rank 1 = first of {len(LABELS)})")
all_first = True
for r in rows:
    rank = {m: 1 + sorted(LABELS, key=lambda lab: r["num"][lab][m], reverse=m != "nmse").index(MC) for m in METRICS}
    all_first &= all(v == 1 for v in rank.values())
    n = r["num"][MC]
    record["rows"].append(dict(pattern=r["name"], contrast=r["c"], af=AF[(r["name"], r["c"])], numbers=r["num"], best=r["best"],
                               mcgs_rank=rank, mcgs_lead=r["lead"], best_other=r["other"], box_psnr=r["box_psnr"],
                               box_lead=r["box_lead"], box_best_other=r["box_other"], box_head_frac=r["box_head_frac"],
                               window=r["vmax"]))
    log(f"  {r['name']:<16} {r['c']:<5} PSNR {n['psnr']:.3f} dB #{rank['psnr']} ({r['lead']['psnr']:+.3f} vs {r['other']['psnr']})"
        f"  SSIM {n['ssim']:.4f} #{rank['ssim']} ({r['lead']['ssim']:+.4f} vs {r['other']['ssim']})"
        f"  NMSE {n['nmse']:.5f} #{rank['nmse']} ({r['lead']['nmse']:+.5f} vs {r['other']['nmse']})"
        f"  |  box PSNR {r['box_psnr'][MC]:.2f} ({r['box_lead']:+.3f} vs {r['box_other']})")
record["mcgs_first_on_every_row_and_metric"] = bool(all_first)
log("MC-GS first on PSNR, SSIM and NMSE in all six rows:", all_first)

# ---------------- optional scan over transverse slices ----------------
if args.scan:
    log(f"\nscan, transverse {args.scan[0]}..{args.scan[1]}: firsts = conditions (of 18) where MC-GS is first; "
        f"box = rows (of 6) where MC-GS has the best box PSNR; dark = dark T1 pixels (ventricle) inside the box")
    log("  slice  firsts  box  min PSNR lead (row, vs)                       min box lead    dark   head px")
    scan = []
    for z in range(args.scan[0], args.scan[1] + 1):
        if not (HEAD["T1"][z].sum() > 2000):
            continue
        st = stats(z)
        firsts = sum(r["lead"][m] > 0 for r in st for m in METRICS)
        boxf = sum((r["box_lead"] or -1) > 0 for r in st)
        w = min(st, key=lambda r: r["lead"]["psnr"])
        wb = min(st, key=lambda r: r["box_lead"] if r["box_lead"] is not None else 9)
        hp = binary_fill_holes(HEAD["T1"][z])
        g = GT["T1"][z]
        dark = int((BM & hp & (g < 0.30 * float(np.median(g[hp])))).sum())
        scan.append(dict(slice=z, firsts=int(firsts), box_firsts=int(boxf), min_psnr_lead=w["lead"]["psnr"],
                         min_psnr_lead_row=f"{w['name']} {w['c']}", min_psnr_lead_vs=w["other"]["psnr"],
                         min_box_lead=wb["box_lead"], dark_in_box=dark, head_px=int(HEAD["T1"][z].sum()),
                         rows=[dict(pattern=r["name"], contrast=r["c"], lead=r["lead"], best_other=r["other"], box_lead=r["box_lead"]) for r in st]))
        log(f"  {z:>5}  {firsts:>2}/18  {boxf}/6  {w['lead']['psnr']:+.3f} dB ({w['name']} {w['c']}, {w['other']['psnr']})".ljust(72)
            + f"{wb['box_lead']:+.3f} dB   {dark:>5}   {int(HEAD['T1'][z].sum())}")
    record["scan"] = scan
    full = [s["slice"] for s in scan if s["firsts"] == 18]
    log("slices with MC-GS first on all 18 conditions:", full or "none")

stem = f"t1flair_v3_af16_masks_record_format_tra{args.tra}_box{args.box}{args.tag}"
if args.no_fig:
    (art / f"record_{stem}.json").write_text(json.dumps(record, indent=2))
    log("DONE (numbers only)")
    raise SystemExit(0)

# ---------------- figure (geometry of fig_lesion_plane_prepost.render, zoom rows, single bars) ----------------
ncol = len(LABELS) + 1
pw, gap, zgap = 2.6, 0.025, 0.02
CBTRIM, cbpad, cbw, cbroom = 0.10, 0.06, 0.13, 0.95
half = (pw - zgap) / 2
H, W = D(rows[0]["g2"]).shape
h = pw * H / W
eh = half * ZOOM["h"] / ZOOM["w"]
fw = ncol * pw + (ncol - 1) * gap + cbpad + cbw + cbroom

TITLE = (f"jiangyizhou T1 + FLAIR, nominal AF 16   --   three sampling patterns: vd-Poisson / uniform density / pseudo-radial"
         f"   --   Transverse {args.tra}" + (f" ({args.what})" if args.what else "")
         + "   --   one frozen MC-GS recipe, classical lambdas re-selected per pattern")
win = "slice P99.9" + (f" / {args.brightness:g}" if args.brightness != 1 else "")
CAP = ["head-masked slice PSNR / SSIM (7x7 window) / NMSE in each panel, the best of each row in bold",
       f"error |recon - GT| / max(GT), jet [0, {args.err_vmax:g}], inside the head only",
       "row under each image row: left = the magnified box of the image, right = the same box of the error map; "
       "GT column: GT box | that pattern's (ky, kz) sampling mask",
       "AF in the GT panel = PE-plane points / sampled points carrying that contrast's data ("
       + ", ".join(f"{n} {NSAMP[n]} points" for n, _ in TOPICS) + ")",
       f"the {args.box}x{args.box} box, the same in every row, is outlined on every slice; "
       f"grey bar: intensity as a fraction of each row's own window ({win})"]


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
fh = top_in + len(rows) * (h + eh) + (2 * len(rows) - 1) * gap + 0.16
fig = plt.figure(figsize=(fw, fh))


def crop(img):
    return img[ZOOM["y0"]:ZOOM["y0"] + ZOOM["h"], ZOOM["x0"]:ZOOM["x0"] + ZOOM["w"]]


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


def outline(a):
    a.add_patch(Rectangle((ZOOM["x0"] - 0.5, ZOOM["y0"] - 0.5), ZOOM["w"], ZOOM["h"], fill=False, ec=ZC, lw=ZLW))


IMKW = dict(origin="lower", aspect="auto")
y, im = top_in, None
for k, r in enumerate(rows):
    name, c = r["name"], r["c"]
    gkw = dict(cmap="gray", vmin=0, vmax=r["vmax"], **IMKW)
    for j in range(ncol):
        x = j * (pw + gap)
        a = axes_at(x, y, pw, h)
        bL = axes_at(x, y + h + gap, half, eh)
        if j == 0:
            a.imshow(D(r["g2"]), **gkw)
            outline(a)
            label(a, name, bold=True, fs=13)
            label(a, c, bold=True, line=1, fs=13, hh=h)
            label(a, f"AF {AF[(name, c)]:.2f}x", bold=True, line=2, fs=13, hh=h)
            bL.imshow(crop(D(r["g2"])), **gkw)
            zframe(bL)
            # the pattern's sampling mask at its true aspect, on black, in the right half of the cell
            axes_at(x + half + zgap, y + h + gap, half, eh).set_facecolor("black")
            mw = eh * W / H
            bM = axes_at(x + half + zgap + (half - mw) / 2, y + h + gap, mw, eh)
            bM.imshow(D(KMASK[name]), cmap="gray", vmin=0, vmax=1, interpolation="nearest", **IMKW)
            if k == 0:
                a.set_title("GT", fontsize=args.title_fs, pad=4)
            continue
        bR = axes_at(x + half + zgap, y + h + gap, half, eh)
        lab = LABELS[j - 1]
        v = SL[(name, c)][lab][args.tra]
        a.imshow(D(v), **gkw)
        outline(a)
        n = r["num"][lab]
        for i, (txt, met) in enumerate(((f"{n['psnr']:.2f} dB", "psnr"), (f"SSIM {n['ssim']:.4f}", "ssim"),
                                         (f"NMSE {n['nmse']:.5f}", "nmse"))):
            label(a, txt, bold=lab == r["best"][met], line=i, fs=10.5, hh=h, color=METRIC_COLOR)
        emap = D(np.abs(v - r["g2"]) / GMAX[c] * r["emask"])
        bL.imshow(crop(D(v)), **gkw)
        im = bR.imshow(crop(emap), cmap="jet", vmin=0.0, vmax=args.err_vmax, **IMKW)
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

fig.suptitle(TITLE, fontsize=13, fontweight="bold", y=1 - 0.10 / fh, va="top")
fig.text(0.5, 1 - cap_y / fh, "\n".join(cap_lines), ha="center", va="top", fontsize=11, linespacing=1.32)
dest = art
if args.out_dir:
    dest = Path(args.out_dir)
    assert not dest.exists() or not any(dest.iterdir()), f"{dest} exists and is not empty: not overwriting"
    dest.mkdir(parents=True, exist_ok=True)
    stem += f"_{args.png_dpi}dpi"
log(f"figure {fw:.2f} x {fh:.2f} in, PNG at {args.png_dpi} dpi -> {dest / (stem + '.png')}")
fig.savefig(dest / f"{stem}.png", dpi=args.png_dpi)
if args.pdf:
    import matplotlib.image as mimage  # noqa: E402
    plt.rcParams["pdf.fonttype"] = 42
    for im_ in fig.findobj(mimage.AxesImage):
        im_.set_rasterized(True)
    pdf_name = f"{stem.rsplit(f'_{args.png_dpi}dpi', 1)[0]}_{args.pdf_dpi}dpi.pdf"
    fig.savefig(dest / pdf_name, dpi=args.pdf_dpi)
    log("wrote", dest / pdf_name)
plt.close(fig)
from PIL import Image  # noqa: E402
Image.MAX_IMAGE_PIXELS = None
with Image.open(dest / f"{stem}.png") as I:
    png_px, png_dpi_meta = I.size, I.info.get("dpi")
    log("figure size (px):", png_px, " dpi in the PNG header:", png_dpi_meta)
    I.convert("RGB").resize((2400, round(2400 * I.height / I.width)), Image.LANCZOS).save(dest / f"{stem}_preview.jpg", quality=88)
record["figure"] = f"{stem}.png"
record["figure_sha256"] = sha(dest / f"{stem}.png")
record["figure_px"], record["figure_dpi"] = list(png_px), args.png_dpi
(dest / f"record_{stem}.json").write_text(json.dumps(record, indent=2))
if args.out_dir:
    shutil.copyfile(Path(__file__).resolve(), dest / Path(__file__).name)
    files = {}
    for f in sorted(dest.iterdir()):
        if f.is_file():
            files[f.name] = dict(bytes=f.stat().st_size, sha256=sha(f))
    files[f"{stem}.png"].update(px=list(png_px), dpi=args.png_dpi)
    if args.pdf:
        files[pdf_name].update(image_dpi=args.pdf_dpi, text="vector (TrueType embedded)")
    rows6 = {f"{r['name']} {r['c']}": dict(mcgs_psnr=round(r["num"][MC]["psnr"], 3), lead_psnr=round(r["lead"]["psnr"], 3),
                                           vs=r["other"]["psnr"], box_lead_psnr=round(r["box_lead"], 3), box_vs=r["box_other"],
                                           mcgs_first={m: r["lead"][m] > 0 for m in METRICS}) for r in rows}
    man = dict(what="diff-mask study (jiangyizhou T1 + FLAIR, nominal AF 16): the three sampling patterns at transverse "
                    f"{args.tra} ({args.what}) in the plain-scan / enhanced record format",
               note=args.note, rows="vd-Poisson / uniform density (new, 2026-11-12) / pseudo-radial (September result kept)",
               columns=["GT"] + LABELS, zoom_box=record["zoom_box"], topics=record["topics"], gt_sha256=record["gt_sha256"],
               per_row=rows6, mcgs_first_on_every_row_and_metric=record["mcgs_first_on_every_row_and_metric"],
               volumes="every column's file and SHA-256: record_" + stem + ".json",
               regenerate=f"gpu.py run --cpu -- @python autoresearch/diffmask_af16_masks_20261111/{Path(__file__).name} "
                          f"--tra {args.tra} --box {args.box} --center {J} {K} --png_dpi {args.png_dpi}"
                          + (f" --pdf --pdf_dpi {args.pdf_dpi}" if args.pdf else "") + " --out_dir <new folder>",
               files=files)
    (dest / "MANIFEST.json").write_text(json.dumps(man, indent=2))
    for f in (f"{stem}_preview.jpg", f"record_{stem}.json", "MANIFEST.json"):      # the small files also to the artifacts
        shutil.copyfile(dest / f, art / f)
    log("folder", dest, ":", ", ".join(f"{k} ({v['bytes'] / 2**20:.1f} MiB)" for k, v in files.items()))
if args.save_topic:
    OUT_TOPIC.mkdir(parents=True, exist_ok=True)
    for f in (f"{stem}.png", f"record_{stem}.json"):
        shutil.copyfile(dest / f, OUT_TOPIC / f)
    log("copied to", OUT_TOPIC)
log("wrote", dest / f"{stem}.png")
log("DONE")
