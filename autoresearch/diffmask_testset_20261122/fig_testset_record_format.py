#!/usr/bin/env python3
"""fig_testset_record_format.py -- one TEST scan of the different-mask study (T1 + FLAIR, acceleration factor 16) at one
transverse slice, in the article format of the tuning-subject figure the user approved on 2026-11-26
(diffmask_af16_masks_20261111/fig_masks_record_format.py --box_select --err_cmap inferno --arrow --arrow_mode own
--arrow_inner 28 --arrow_margin 8 --arrow_on zoom --role ...). This file is a copy of that script with another loader.

Rows: uniform density T1, uniform density FLAIR, pseudo-radial T1, pseudo-radial FLAIR (the vd-Poisson pattern belongs
to the acceleration-factor test-set study and is not drawn here). Columns: GT, ZF, nine comparison methods, MC-GS.

Volumes, ground truth and head mask are those of the study's common rescoring (collect_stats.availability, the
uniform-density clone's t1t2_common.Pair, diag_artefact.headonly.head_mask), so the volume head PSNR printed here must
equal the logged one. The slice is NOT chosen by the results: --tra -1 (default) takes the transverse slice with the
largest head area of the T1 ground truth. The zoom box (--box_select) and the arrows are chosen against GT, for display
only. Panel numbers are slice metrics inside the head; the best of a row is bold whichever method it is.

  gpu.py run --cpu -- @python autoresearch/diffmask_testset_20261122/fig_testset_record_format.py --scan_name vol1_huangyuwei
      --box 48 --box_select --what "" --err_cmap inferno --arrow --arrow_mode own --arrow_inner 28 --arrow_margin 8
      --arrow_on zoom --brightness 1 --role "Sampling-mask generalization, test scan" --tag _article
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--scan_name", required=True, help="test scan, a key of ts_common.SCANS")
ap.add_argument("--tra", type=int, default=-1, help="transverse slice (axis 0); -1 = the slice with the largest head area of the T1 GT")
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
# article look (user, 2026-11-22), the same options as fig_t1flair_af_record_format.py / fig_lesion_plane_prepost.py;
# the defaults keep the record figure of 2026-11-13 unchanged
ap.add_argument("--err_cmap", default="jet", help="colour map of the error maps (jet = the record format; article: inferno)")
ap.add_argument("--arrow", action="store_true",
                help="one arrow per row, at the same place in every panel of the row: where the error of the other methods "
                     "(median over the nine reconstructions other than ZF and MC-GS, 7x7 local mean) exceeds MC-GS's most")
ap.add_argument("--arrow_color", default="#FF1A1A")
ap.add_argument("--arrow_mode", choices=["gap", "edge", "max", "own"], default="edge",
                help="gap: the largest error gap anywhere in the box; edge: the same, restricted to tissue boundaries "
                     "(the fifth of the box with the steepest GT gradient); own (user, 2026-11-26: 'make the arrow in "
                     "zoom in view to point the area with the largest error'): in every panel the arrow points at the "
                     "largest error of THAT panel's own error box (7x7 local mean), GT panel: the largest median error "
                     "of the ten reconstructions; max: one place per row, the largest median error of the nine "
                     "reconstructions other than ZF and MC-GS")
ap.add_argument("--arrow_inner", type=int, default=0,
                help="keep the arrow off the edge of the brain (user, 2026-11-26: 'but not point the edge of the brain'): "
                     "only points at least this many voxels inside the outline of the filled head mask of the slice "
                     "are candidates; the scalp and skull are about 13 voxels thick at transverse 205, so 20 keeps the "
                     "arrow about 7 voxels inside the brain surface (the head mask does not separate brain from scalp)")
ap.add_argument("--arrow_margin", type=int, default=3,
                help="modes max / own: candidates lie at least this many pixels inside the zoom box (3 = the whole 7x7 "
                     "window inside the box; larger keeps the arrow head off the frame)")
ap.add_argument("--arrow_on", choices=["zoom", "main"], default="zoom",
                help="zoom: only in the magnified weighted-image box (not in the error map, not on the slice; user rule "
                     "2026-11-22); main: on the slice images")
ap.add_argument("--role", default="", help="what the figure stands for in the article: head of the title; the subject "
                                           "name and the point-count / net-acceleration wording are then left out")
ap.add_argument("--save_data", action="store_true", help="also write <stem>_data.npz: every slice the figure draws")
# the way the article figure of the acceleration study was drawn (t1flair_af_record_fig_20261120: find_zoom_box.py, then
# fig_t1flair_af_record_format.py --box_centres ... --arrow_on main; user, 2026-11-22: "understand how the figure with
# af = 20 was plotted, then redraw this graph"). There one box per plane; here one box per sampling pattern.
ap.add_argument("--box_select", action="store_true",
                help="one zoom box per sampling pattern (the same for T1 and FLAIR), placed where MC-GS's box-mean error "
                     "is furthest below the best other method's (the criterion of find_zoom_box.py: boxes with >= "
                     "--head_frac of their pixels in the head; score summed over the two contrasts). Chosen against "
                     "GT, for display only")
ap.add_argument("--box_centres", type=int, nargs=6, default=None, metavar="V",
                help="one zoom box per sampling pattern, given: J K of vd-Poisson, uniform density, pseudo-radial")
ap.add_argument("--head_frac", type=float, default=0.95)
ap.add_argument("--box_inner", type=int, default=0,
                help="--box_select: the whole box must lie at least this many voxels inside the outline of the filled head "
                     "mask (keeps the box off the scalp, the skull and the brain surface; user, 2026-11-26: the arrow "
                     "must not point at the edge of the brain)")
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
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes, binary_opening, gaussian_filter, uniform_filter  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402

os.nice(10)
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ts_common as TSC  # noqa: E402
import collect_stats as CS  # noqa: E402
from run_arms import topic_dir  # noqa: E402
assert args.scan_name in TSC.SCANS, args.scan_name
assert not args.save_topic, "--save_topic is not used for the test scans"
TOPICS = [(TSC.LABEL[af], af) for af in TSC.AFS]          # (row name, pattern key)
OUT_TOPIC = None
CONTRASTS = ("T1", "FLAIR")
MC = "MC-GS"
METRICS = ("psnr", "ssim", "nmse")
METRIC_COLOR = "#FFC23D"
ZC, ZLW = METRIC_COLOR, 2.2
DISK = np.hypot(*np.mgrid[-6:7, -6:7]) <= 6
log = lambda *a: print(*a, flush=True)
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", "masks_record_format_out")).resolve()
art.mkdir(parents=True, exist_ok=True)


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
import torch  # noqa: E402
torch.set_num_threads(4)
T_SCORE = topic_dir(args.scan_name, "uniform", TSC.OUT)    # the ONE scoring clone of the scan (as collect_stats.eval_scan)
sys.path.insert(0, str(T_SCORE / "scripts"))
import t1t2_common as TC  # noqa: E402
assert Path(TC.R) == T_SCORE, TC.R
from diag_artefact.headonly import head_mask  # noqa: E402
GT, GMAX, HEAD = {}, {}, {}
SL = {}                                   # (pattern, contrast) -> label -> {slice index: 2-D magnitude}
KMASK, AF, NSAMP = {}, {}, {}
LABELS = list(CS.LABELS)
for c in CONTRASTS:
    P = TC.Pair(c, torch.device("cpu"))
    g = np.abs(P.gt.cpu().numpy()).astype(np.float32)
    GT[c], GMAX[c], HEAD[c] = g, float(g.max()), np.asarray(head_mask(g)).astype(bool)
    del P
if args.tra < 0:
    area = np.array([binary_fill_holes(m).sum() for m in HEAD["T1"]])
    args.tra = int(np.argmax(area))
    log(f"--tra -1: transverse slice with the largest head area of the T1 GT = {args.tra} (of {len(area)}; area {int(area.max())} px)")
NEED = sorted({args.tra} | (set(range(args.scan[0], args.scan[1] + 1)) if args.scan else set()))
record = dict(format="fig_masks_record_format.py article format, test scan: sampling patterns as rows",
              scan=args.scan_name, group=TSC.SCANS[args.scan_name]["group"], person=TSC.SCANS[args.scan_name]["person"],
              slice=args.tra, zoom_box=dict(centre_JK=list(args.center), edge_voxels=args.box),
              topics={n: str(topic_dir(args.scan_name, af, TSC.OUT)) for n, af in TOPICS}, scoring_clone=str(T_SCORE),
              volumes={}, rows=[])
av = CS.availability()
worst = 0.0
log("volume head PSNR (this script) | the logged head PSNR | file")
for name, af in TOPICS:
    with h5py.File(TSC.MASK(af), "r") as f:
        mask3 = np.array(f["mask"]) > 0
    assert np.all(mask3 == mask3[:1]), f"{af}: expected a 3-D mask with the readout (axis 0) fully sampled"
    KMASK[name], NSAMP[name] = mask3[mask3.shape[0] // 2], int(mask3[0].sum())
    record["volumes"][name] = {}
    for c in CONTRASTS:
        AF[(name, c)] = float(mask3[0].size / NSAMP[name])
        assert mask3.shape == GT[c].shape, (af, mask3.shape, GT[c].shape)
        SL[(name, c)], record["volumes"][name][c] = {}, {}
        hm = HEAD[c]
        gh = GT[c][hm].astype(np.float64)
        for lab in LABELS:
            info = av[(args.scan_name, af, c, lab)]
            assert isinstance(info, dict), (args.scan_name, af, c, lab, info)
            p, member = Path(info["file"]), info["member"]
            v = np.abs(np.load(p) if member is None else np.load(p)[member]).astype(np.float32)
            assert v.shape == GT[c].shape, (name, c, lab, v.shape)
            e = v[hm].astype(np.float64) - gh
            mine = float(20 * np.log10(GMAX[c]) - 10 * np.log10(float((e * e).mean())))
            logged = info["logged"]["head_psnr"]
            d = mine - float(logged)
            worst = max(worst, abs(d))
            SL[(name, c)][lab] = {z: v[z].copy() for z in NEED}
            record["volumes"][name][c][lab] = dict(path=str(p), member=member, crc_key=CS.crc_key(p, member),
                                                   volume_head_psnr=mine, logged_head_psnr=logged)
            log(f"  {name:<16} {c:<5} {lab:<12} {mine:8.3f} | {float(logged):8.3f} ({d:+.4f}) | {p.name}")
            del v
record["max_abs_diff_vs_logged_head_psnr"] = worst
log(f"largest |difference| to the logged head PSNR: {worst:.4f} dB"
    + ("" if worst < 0.02 else "   <-- WARNING: some column differs from the number its topic logged"))
SHAPE = GT["T1"].shape

hw, (J, K) = args.box // 2, args.center
assert not (args.box_select and args.box_centres), "--box_select and --box_centres exclude each other"
CENTRES = {name: (J, K) for name, _ in TOPICS}
if args.box_centres:
    CENTRES = {name: tuple(args.box_centres[2 * i:2 * i + 2]) for i, (name, _) in enumerate(TOPICS)}
if args.box_select:
    # find_zoom_box.py on this slice: per method the box mean of the figure's error map; score = (lowest box mean
    # among the ten other columns) - (MC-GS box mean), summed over T1 and FLAIR; only boxes inside the head count
    record["box_select"] = dict(criterion="find_zoom_box.py (t1flair_af_record_fig_20261120): MC-GS box-mean error furthest "
                                          "below the best other column, T1 + FLAIR", head_frac=args.head_frac,
                             box_inner_voxels=args.box_inner, rows=[])
    log(f"\nbox selection on transverse {args.tra}, {args.box}-voxel box, per sampling pattern (T1 + FLAIR):")
    for name, _ in TOPICS:
        score, ok, bmean = 0.0, None, {}
        for c in CONTRASTS:
            m = HEAD[c][args.tra]
            g = GT[c][args.tra].astype(np.float64)
            bmean[c] = {lab: uniform_filter(np.abs(SL[(name, c)][lab][args.tra] - g) / GMAX[c] * m, size=args.box, mode="constant")
                        for lab in LABELS}
            valid = uniform_filter(m.astype(np.float32), size=args.box, mode="constant") >= args.head_frac
            valid[:hw] = valid[-hw:] = False
            valid[:, :hw] = valid[:, -hw:] = False
            if args.box_inner > 0:
                core = binary_erosion(binary_fill_holes(m), iterations=args.box_inner).astype(np.float32)
                valid &= uniform_filter(core, size=args.box, mode="constant") >= 0.999
            ok = valid if ok is None else ok & valid
            score = score + np.minimum.reduce([bmean[c][lab] for lab in LABELS if lab != MC]) - bmean[c][MC]
        assert ok.any(), f"{name}: no {args.box}-voxel box with {args.head_frac:g} of its pixels in the head"
        pos = tuple(int(v) for v in np.unravel_index(np.argmax(np.where(ok, score, -np.inf)), score.shape))
        CENTRES[name] = pos
        for c in CONTRASTS:
            means = {lab: float(bmean[c][lab][pos]) for lab in LABELS}
            best = min((lab for lab in LABELS if lab != MC), key=means.get)
            record["box_select"]["rows"].append(dict(pattern=name, contrast=c, centre_JK=list(pos), box_mean_error=means,
                                                     best_other=best, mcgs_over_best_other=means[MC] / means[best]))
            log(f"  {name:<16} {c:<5} centre (J, K) = {pos}  MC-GS {means[MC]:.5f}  best other {best} {means[best]:.5f} "
                f"({100 * (1 - means[MC] / means[best]):+.1f}% lower)")
PER_PATTERN = len(set(CENTRES.values())) > 1
BMS, ZOOMS = {}, {}
for name, _ in TOPICS:
    j0, k0 = CENTRES[name]
    BMS[name] = np.zeros(SHAPE[1:], bool)
    BMS[name][j0 - hw:j0 + hw, k0 - hw:k0 + hw] = True
    assert BMS[name].sum() == args.box ** 2, f"{name}: the zoom box leaves the slice"
    rr, cc = np.nonzero(D(BMS[name]))
    ZOOMS[name] = dict(x0=int(cc.min()), y0=int(rr.min()), w=int(cc.max() - cc.min() + 1), h=int(rr.max() - rr.min() + 1))
record["zoom_box"] = dict(edge_voxels=args.box, per_pattern_centres_JK={n: list(v) for n, v in CENTRES.items()},
                          note=("chosen with the find_zoom_box.py criterion against GT, for display only" if args.box_select
                                else "given" if args.box_centres else "one box for all rows"))
BM, ZOOM = BMS[TOPICS[0][0]], ZOOMS[TOPICS[0][0]]        # geometry (every box has the same size); --scan uses the first


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
            bz = BMS[name] & m2
            g64 = g2.astype(np.float64)
            bp = {lab: float(20 * np.log10(GMAX[c]) - 10 * np.log10(float(((SL[(name, c)][lab][z][bz].astype(np.float64) - g64[bz]) ** 2).mean())))
                  for lab in LABELS} if bz.any() else {}
            bo = max((lab for lab in LABELS if lab != MC), key=bp.get) if bp else None
            out.append(dict(name=name, c=c, z=z, g2=g2, m2=m2, num=num, best=best, other=other, lead=lead, box_psnr=bp,
                            box_other=bo, box_lead=(bp[MC] - bp[bo]) if bp else None, box_head_frac=float(bz.sum() / BMS[name].sum())))
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
log("MC-GS first on PSNR, SSIM and NMSE in all rows:", all_first)

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

stem = f"dm_testset_{args.scan_name}_af16_masks_tra{args.tra}_box{args.box}{args.tag}"
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

TITLE = (f"{args.role or 'Different sampling masks'}: {args.scan_name} ({TSC.SCANS[args.scan_name]['group']} scan), T1 + FLAIR, "
         f"acceleration factor 16   --   uniform density / pseudo-radial   --   Transverse {args.tra}"
         + (f" ({args.what})" if args.what else ""))
win = "slice P99.9" + (f" / {args.brightness:g}" if args.brightness != 1 else "")
CAP = ["head-masked slice PSNR / SSIM (7x7 window) / NMSE in each panel, the best of each row in bold",
       f"error |recon - GT| / max(GT), {args.err_cmap} [0, {args.err_vmax:g}], inside the head only",
       "row under each image row: left = the magnified box of the image, right = the same box of the error map; "
       "GT column: GT box | that pattern's (ky, kz) sampling mask",
       "AF in the GT panel = PE-plane points / sampled points carrying that contrast's data ("
       + ", ".join(f"{n} {NSAMP[n]} points" for n, _ in TOPICS) + ")",
       (f"one {args.box}-voxel box per sampling mask (the same for T1 and FLAIR), placed where MC-GS's error is furthest "
        f"below the best other method" + (f", the whole box at least {args.box_inner} voxels inside the head outline" if args.box_inner else "")
        + " (chosen against GT, for display only); " if args.box_select else
        f"one {args.box}-voxel box per sampling mask (the same for T1 and FLAIR), outlined on every slice; " if PER_PATTERN else
        f"the {args.box}x{args.box} box, the same in every row, is outlined on every slice; ")
       + f"grey bar: intensity as a fraction of each row's own window ({win})"]
if args.role:        # article figures carry the acceleration factor only (user, 2026-11-22): no point-count line
    CAP = [x for x in CAP if not x.startswith("AF in the GT panel")]
if args.arrow:
    _where = "red arrow in the magnified image box" if args.arrow_on == "zoom" else "arrow on each slice"
    CAP.append((_where + ": the largest error of that panel's own error box (7x7 local mean); GT panel: the largest "
                "median error of the ten reconstructions" if args.arrow_mode == "own" else
                _where + " (same place along a row): the largest error of the other methods (median of the nine "
                "reconstructions other than ZF and MC-GS, 7x7 local mean)" if args.arrow_mode == "max" else
                _where + " (same place along a row): where the error of the other methods (median of the nine reconstructions "
                "other than ZF and MC-GS, 7x7 local mean) exceeds MC-GS's most")
               + (", among the tissue boundaries of the box (steepest fifth of the GT gradient)" if args.arrow_mode == "edge" else "")
               + (f", at least {args.arrow_inner} voxels inside the head outline" if args.arrow_inner > 0 else "")
               + ("; chosen against GT, for display only" if args.box_select else ""))


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


def crop(img, z=None):
    z = z or ZOOM
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


def outline(a, z=None):
    z = z or ZOOM
    a.add_patch(Rectangle((z["x0"] - 0.5, z["y0"] - 0.5), z["w"], z["h"], fill=False, ec=ZC, lw=ZLW))


IMKW = dict(origin="lower", aspect="auto")


def arrow_at(a, pt, main=False, shape=None, z=None):
    """an arrow whose head sits at pt (crop pixel x, y); zoom panel: tail towards the centre of the box;
    slice image (main): head at the same anatomical point, tail towards the centre of the image"""
    z = z or ZOOM
    x, y = pt
    if main:
        x, y = x + z["x0"], y + z["y0"]
        v, L = np.array([shape[1] / 2 - x, shape[0] / 2 - y], float), 0.11 * shape[1]
    else:
        v, L = np.array([z["w"] / 2 - x, z["h"] / 2 - y], float), 13.0 * args.box / 48
    n = float(np.hypot(*v))
    v = v / n * L if n > 4 else np.array([L, L]) / 1.4
    a.annotate("", xy=(x, y), xytext=(x + v[0], y + v[1]),
               arrowprops=dict(arrowstyle="-|>", color=args.arrow_color, lw=2.2 if main else 1.8,
                               mutation_scale=15 if main else 13, shrinkA=0, shrinkB=1))


if args.arrow:
    for r in rows:
        sm_all = {lab: uniform_filter(np.abs(SL[(r["name"], r["c"])][lab][args.tra] - r["g2"]) / GMAX[r["c"]] * r["emask"],
                                      size=7, mode="nearest") for lab in LABELS}
        sm = {lab: v_ for lab, v_ in sm_all.items() if lab != "ZF"}

        inner = np.ones(r["g2"].shape, bool)
        if args.arrow_inner > 0:
            inner = binary_erosion(binary_fill_holes(r["m2"] > 0), iterations=args.arrow_inner)
        inner = crop(D(inner), ZOOMS[r["name"]]).astype(bool)
        assert inner[args.arrow_margin:-args.arrow_margin, args.arrow_margin:-args.arrow_margin].any(), f"--arrow_inner {args.arrow_inner}: no candidate point left in the box of {r['name']}"

        def box_peak(x, m):
            """(x, y) and value of the largest entry of the zoom box of x, m pixels from its edge excluded"""
            s_ = crop(D(x), ZOOMS[r["name"]]).copy()
            s_[:m] = s_[-m:] = -np.inf; s_[:, :m] = s_[:, -m:] = -np.inf
            s_[~inner] = -np.inf
            y_, x_ = np.unravel_index(np.argmax(s_), s_.shape)
            return (int(x_), int(y_)), float(s_[y_, x_])

        if args.arrow_mode in ("max", "own"):
            if args.arrow_mode == "own":
                r["arrow_own"] = {lab: box_peak(sm_all[lab], args.arrow_margin) for lab in LABELS}
                r["arrow"], r["arrow_gap"] = box_peak(np.median(list(sm.values()), axis=0), args.arrow_margin)
                for lab in LABELS:
                    log(f"  arrow {r['name']:<16} {r['c']:<5} {lab:<12} crop (x, y) = {r['arrow_own'][lab][0]}, "
                        f"local-mean error {r['arrow_own'][lab][1]:.4f}")
            else:
                r["arrow"], r["arrow_gap"] = box_peak(np.median([sm[lab] for lab in sm if lab != MC], axis=0), args.arrow_margin)
            log(f"  arrow {r['name']:<16} {r['c']:<5} crop (x, y) = {r['arrow']}, median error {r['arrow_gap']:.4f}")
            continue
        sc = crop(D(np.median([sm[lab] for lab in sm if lab != MC], axis=0) - sm[MC]), ZOOMS[r["name"]]).copy()
        m = 7
        sc[:m] = sc[-m:] = -np.inf; sc[:, :m] = sc[:, -m:] = -np.inf
        sc[~inner] = -np.inf
        if args.arrow_mode == "edge":
            gy, gx = np.gradient(gaussian_filter(r["g2"] / GMAX[r["c"]], 1.0))
            gmag = crop(D(np.hypot(gx, gy)), ZOOMS[r["name"]])
            sc[gmag < np.percentile(gmag, 80)] = -np.inf
        yy, xx = np.unravel_index(np.argmax(sc), sc.shape)
        r["arrow"], r["arrow_gap"] = (int(xx), int(yy)), float(sc[yy, xx])
        log(f"  arrow {r['name']:<16} {r['c']:<5} crop (x, y) = {r['arrow']}, error gap {r['arrow_gap']:.4f}")
    record["arrows"] = [dict(pattern=r["name"], contrast=r["c"], crop_xy=r["arrow"], gap=r["arrow_gap"],
                             **({"per_panel": {lab: dict(crop_xy=p_, local_mean_error=e_) for lab, (p_, e_) in r["arrow_own"].items()}}
                                if "arrow_own" in r else {})) for r in rows]
    record["arrow_mode"], record["arrow_on"] = args.arrow_mode, args.arrow_on
    record["arrow_inner_voxels"] = args.arrow_inner
record["error_display"] = dict(vmax=args.err_vmax, cmap=args.err_cmap)
y, im = top_in, None
for k, r in enumerate(rows):
    name, c = r["name"], r["c"]
    gkw = dict(cmap="gray", vmin=0, vmax=r["vmax"], **IMKW)
    Z = ZOOMS[name]
    for j in range(ncol):
        x = j * (pw + gap)
        a = axes_at(x, y, pw, h)
        bL = axes_at(x, y + h + gap, half, eh)
        if j == 0:
            a.imshow(D(r["g2"]), **gkw)
            outline(a, Z)
            label(a, name, bold=True, fs=13)
            label(a, c, bold=True, line=1, fs=13, hh=h)
            if not args.role:
                label(a, f"AF {AF[(name, c)]:.2f}x", bold=True, line=2, fs=13, hh=h)
            bL.imshow(crop(D(r["g2"]), Z), **gkw)
            zframe(bL)
            if args.arrow:
                arrow_at(bL, r["arrow"], z=Z) if args.arrow_on == "zoom" else arrow_at(a, r["arrow"], True, D(r["g2"]).shape, Z)
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
        outline(a, Z)
        n = r["num"][lab]
        for i, (txt, met) in enumerate(((f"{n['psnr']:.2f} dB", "psnr"), (f"SSIM {n['ssim']:.4f}", "ssim"),
                                         (f"NMSE {n['nmse']:.5f}", "nmse"))):
            label(a, txt, bold=lab == r["best"][met], line=i, fs=10.5, hh=h, color=METRIC_COLOR)
        emap = D(np.abs(v - r["g2"]) / GMAX[c] * r["emask"])
        bL.imshow(crop(D(v), Z), **gkw)
        im = bR.imshow(crop(emap, Z), cmap=args.err_cmap, vmin=0.0, vmax=args.err_vmax, **IMKW)
        zframe(bL); zframe(bR)
        if args.arrow:
            pt = r["arrow_own"][lab][0] if "arrow_own" in r else r["arrow"]
            arrow_at(bL, pt, z=Z) if args.arrow_on == "zoom" else arrow_at(a, pt, True, D(v).shape, Z)
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
if args.save_data:
    dat = {}
    for r in rows:
        k = f"{r['name'].replace(' ', '_')}_{r['c']}"
        dat[f"{k}__GT"], dat[f"{k}__head_mask"], dat[f"{k}__error_region"] = r["g2"], r["m2"], r["emask"]
        for lab in LABELS:
            dat[f"{k}__{lab.replace(' ', '_')}"] = SL[(r["name"], r["c"])][lab][args.tra]
    for n_, _ in TOPICS:
        dat[f"{n_.replace(' ', '_')}__sampling_mask_plane"] = KMASK[n_]
    dat["meta_json"] = np.array(json.dumps(dict(
        note="magnitude slices in volume orientation (display: rot90(x.T), origin lower); error = |recon - GT| / gt_max "
             "inside error_region; metrics inside head_mask with PSNR peak gt_max",
        slice=args.tra, gt_max=GMAX, zoom_box=record["zoom_box"], zoom_display_px=ZOOMS, arrows=record.get("arrows"),
        error_display=record["error_display"], volumes=record["volumes"])))
    np.savez_compressed(dest / f"{stem}_data.npz", **dat)
    log("wrote", dest / f"{stem}_data.npz")
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
    man = dict(what=f"diff-mask TEST-SET study ({args.scan_name}, T1 + FLAIR, AF 16): uniform density and pseudo-radial at transverse "
                    f"{args.tra} ({args.what}) in the plain-scan / enhanced record format",
               note=args.note, rows="vd-Poisson / uniform density (new, 2026-11-12) / pseudo-radial (September result kept)",
               columns=["GT"] + LABELS, zoom_box=record["zoom_box"], topics=record["topics"], scan=args.scan_name,
               per_row=rows6, mcgs_first_on_every_row_and_metric=record["mcgs_first_on_every_row_and_metric"],
               volumes="every column's file and SHA-256: record_" + stem + ".json",
               regenerate=f"gpu.py run --cpu -- @python autoresearch/diffmask_af16_masks_20261111/{Path(__file__).name} "
                          f"--tra {args.tra} --box {args.box} "
                          + ("--box_select" if args.box_select else "--box_centres " + " ".join(str(v) for n, _ in TOPICS for v in CENTRES[n])
                             if args.box_centres else f"--center {J} {K}") + f" --png_dpi {args.png_dpi}"
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
