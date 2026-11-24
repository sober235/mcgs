#!/usr/bin/env python3
"""fig_lesion_plane_prepost.py -- one lesion-containing plane, pre and post, in the house style of the
T1+FLAIR AF12 / AF16 figures of record (Results_20260908/T1FLAIR_AF12_16_20_24_figures_hires, made by
fig_three_planes_inset.py / fig_three_planes_compact.py) and of their CE adaptation
(CE_retro_cssense_af16_vdpois_ar/scripts/fig_three_planes_compact.py).

Layout per figure (one plane): rows = pre image, pre error, post image, post error; columns = GT, ZF and the
ten methods in the house order (CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR,
GS-single, MC-GS). Error = |recon - GT| / max|GT| of the volume, jet [0, 0.1], inside the filled head only.
Left cell of each error row: the sampling mask in that plane (k-space). A 48 x 48 box, centred on the lesion
core in that plane, is outlined on every panel and magnified into a corner.

Numbers: each panel carries the frozen evaluator's lesion-plane PSNR / SSIM / NMSE (lesion_planes.json of
the given score directory: opponents = the frozen baseline, MC-GS = the scored candidate). The script also
recomputes every number from the volumes with the evaluator's plane_metrics arithmetic and stops if any differs,
so the drawn volumes are provably the scored ones. Best of each row in bold.

Read-only: reads the score directory, the candidate and opponent volumes listed in its input_hashes.json, the GT,
the 3-D mask and the lesion ROI; writes PNG (and optional PDF) + a JSON record into --out. No GT/ROI goes into any
reconstruction; the ROI only places the zoom box.

  python fig_lesion_plane_prepost.py --score-dir RUN/scores/s42_m0625 --label "C77 seed 42" --out DIR [--planes 1:130 1:127]
  python fig_lesion_plane_prepost.py --score-dir ... --label ... --out DIR --three 196 130 43    # three planes, pre/post

--topic TOPIC (instead of --score-dir): a CE retro topic that has no frozen lesion-plane evaluator, e.g. the
2026-09-18 CE_retro_cssense_af16 (BART Poisson-disc AF16). Volumes are the topic's registered arms (logs/summary.json,
as its scripts/arms.py; GS arms = recon/gs_{single,joint}_<scan>_mag.npy, as its fig_three_planes_compact.py), and
the panel numbers are computed here with the same arithmetic (head mask |GT| > 0.05 P99.9, PSNR at max|GT|, 7x7 SSIM
averaged over the head, head NMSE). Every drawn volume's SHA-256 goes into the record.
  python fig_lesion_plane_prepost.py --topic .../CE_retro_cssense_af16 --label "Sept recipe" --out DIR --three 201 131 48
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--score-dir", type=Path, default=None, help="evaluator output dir (objective.json, lesion_planes.json, input_hashes.json)")
ap.add_argument("--topic", type=Path, default=None, help="CE retro topic without an evaluator (see the docstring); needs --three or --planes")
ap.add_argument("--arm", action="append", default=[], metavar="LABEL=PATH",
                help="topic mode only: replace one arm's volume, PATH may contain {s} for the scan (pre/post), e.g. "
                     "'3D-DIP=/x/recon/dip_sup_b96_tv0.03_{s}_complex.npy'; repeatable; recorded in the JSON")
ap.add_argument("--mc_npz", type=Path, default=None,
                help="topic mode only: MC-GS = magnitude of a trainer volumes file (keys recT1 = pre, recFL = post), "
                     "e.g. a tuned run saved with --save_volumes; default = the topic's registered gs_joint volumes")
ap.add_argument("--label", required=True, help="what the MC-GS column is, for the title, e.g. 'C77 seed 42 (m625)'")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--planes", nargs="*", default=None, metavar="AXIS:INDEX",
                help="planes to draw (default: every plane where the candidate strictly wins all six conditions)")
ap.add_argument("--three", type=int, nargs=3, default=None, metavar=("TRA", "COR", "SAG"),
                help="one figure of three evaluator planes, rows top to bottom: pre transverse, post transverse, pre coronal, "
                     "post coronal, pre sagittal, post sagittal (each image row followed by its error row)")
ap.add_argument("--inset", type=int, default=48)
ap.add_argument("--inset_corner", default="auto", choices=["auto", "bl", "br", "tr"],
                help="corner of the magnified box (auto = the free corner farthest from the outlined box; tl holds the numbers)")
ap.add_argument("--err_vmax", type=float, default=0.1)
ap.add_argument("--err_cmap", default="jet", help="colour map of the error maps (jet = the record format; inferno = the article look)")
ap.add_argument("--arrow", action="store_true",
                help="article look (as fig_t1flair_af_record_format.py --arrow --arrow_on main): one arrow per row on every "
                     "slice image, at the same place along the row = the point of the outlined box where the error of the "
                     "other methods (median of the nine reconstructions other than ZF and MC-GS, 7x7 local mean) exceeds "
                     "that of MC-GS the most. Chosen with the GT, for display only")
ap.add_argument("--arrow_mode", choices=["gap", "edge", "lesion"], default="gap",
                help="edge = the arrow point is searched only among the tissue boundaries of the box (the fifth of the box "
                     "with the steepest GT gradient); lesion = only on the boundary of the lesion core in that plane "
                     "(ROI dilated 2 px minus eroded 2 px), at least --arrow_skull_px from the edge of the head, so it "
                     "never marks the skull (user, 2026-11-22)")
ap.add_argument("--arrow_skull_px", type=int, default=10, help="lesion mode: smallest distance of the arrow point from the head edge")
ap.add_argument("--arrow_on", choices=["main", "main+err", "zoom"], default="main",
                help="zoom = the arrow is drawn only in the magnified box of the image (left half of the zoom row), not on "
                     "the slice and not in the error map (user, 2026-11-22); main+err = the arrow is also drawn in the magnified error map of the row below (zoom layout), so it "
                     "marks the place on the error map where MC-GS is better (user, 2026-11-22)")
ap.add_argument("--arrow_color", default="red")
ap.add_argument("--role", default="", help="what the figure stands for in the article, put at the head of the title")
ap.add_argument("--save_data", action="store_true",
                help="also write <stem>_data.npz: every drawn slice (GT, all reconstructions, head mask, error region, "
                     "lesion core, sampling-mask plane) in volume orientation, with the box and arrow positions")
ap.add_argument("--err_layout", default="full", choices=["full", "zoom"],
                help="row under each image row: full = whole error map (house style); zoom = left half the magnified box "
                     "of the image, right half the magnified box of the error map (GT column: GT box | sampling mask)")
ap.add_argument("--cbar", default="per_row", choices=["per_row", "single"],
                help="per_row = a grey and an error bar beside every row pair (house style); single = one grey bar "
                     "(intensity as a fraction of each row's own window, 0-1) and one error bar, beside the first row pair")
ap.add_argument("--brightness", type=float, default=1.5, help="display only, as the CE three-plane figures (window top / brightness)")
ap.add_argument("--title_fs", type=float, default=22.0, help="first-row method names (22 = the hi-res figures of record)")
ap.add_argument("--png_dpi", type=int, default=200)
ap.add_argument("--pdf", action="store_true")
ap.add_argument("--pdf_dpi", type=int, default=300)
ap.add_argument("--tra_rot", type=int, default=1, choices=[0, 1, 2, 3], help="quarter turns of transverse panels (1 = eyes up)")
args = ap.parse_args()
assert (args.score_dir is None) != (args.topic is None), "give exactly one of --score-dir / --topic"
assert args.score_dir or args.three or args.planes, "--topic needs --three or --planes"
assert not (args.arm and args.score_dir), "--arm is for --topic mode (an evaluator's scored volumes are fixed)"

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
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy.ndimage import (binary_dilation, binary_erosion, binary_fill_holes, binary_opening, distance_transform_edt,  # noqa: E402
                           gaussian_filter, uniform_filter)  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402

TOPIC = args.topic or Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois_cmp")   # evaluate.py TOPIC
ROI = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz")
SCANS = ("pre", "post")
SCAN_NAME = {"pre": "Pre-contrast", "post": "Post-contrast"}
METRICS = ("psnr", "ssim", "nmse")
# house column order and the file-name pattern of each frozen opponent volume (evaluator input_hashes.json)
ARMS = [("ZF", r"zf_{s}_complex\.npy"), ("CG-SENSE", r"cgsense_{s}_complex\.npy"),
        ("CS-SENSE", r"cssense_o200_lam[0-9.]+_{s}_complex\.npy"), ("Joint SENSE", r"jcssense_o200_lam[0-9.]+_{s}_complex\.npy"),
        ("L1-SPIRiT", r"spirit_ks[0-9]+_lam[0-9.]+_{s}_complex\.npy"), ("ZS-SSDU", r"zsssdu_{s}_complex\.npy"),
        ("3D-DIP", r"dip_{s}_complex\.npy"), ("3D-INR", r"inr_{s}_complex\.npy"), ("MC-INR", r"mcinr_{s}_complex\.npy"),
        ("GS-single", r"gs_single_{s}_complex\.npy")]
MC = "MC-GS"
LABELS = [a for a, _ in ARMS] + [MC]
ANAT = {0: "Transverse", 1: "Coronal", 2: "Sagittal"}
METRIC_COLOR = "#FFC23D"
ZC, ZLW = METRIC_COLOR, 2.2
log = lambda *a: print(*a, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------- inputs: evaluator record, volumes, GT, mask, ROI ----------------
sd = args.score_dir
paths = {s: {} for s in SCANS}
if sd is not None:
    hashes = json.load(open(sd / "input_hashes.json"))
    lp = json.load(open(sd / "lesion_planes.json"))["planes"]
    PL = {(p["axis"], p["index"]): p for p in lp}
    if args.planes:
        want = [tuple(int(v) for v in s.split(":")) for s in args.planes]
    else:
        want = [(p["axis"], p["index"]) for p in lp if p["strict_success"]]
    assert want and all(w in PL for w in want), f"planes {want} not among the 59 evaluator planes"
    for s in SCANS:
        paths[s][MC] = Path(hashes["candidate"][s]["path"])
        recon = [Path(p) for p in hashes["files"] if p.startswith(str(TOPIC / "recon") + "/")]
        for lab, pat in ARMS:
            hit = [p for p in recon if re.fullmatch(pat.format(s=s), p.name)]
            assert len(hit) == 1, f"{lab} {s}: expected one frozen volume, got {hit}"
            paths[s][lab] = hit[0]
    for s in SCANS:            # the drawn files are the scored files (byte-identical)
        for lab, p in paths[s].items():
            want_sha = hashes["candidate"][s]["sha256"] if lab == MC else hashes["files"][str(p)]
            assert sha(p) == want_sha, f"hash mismatch {lab} {s} {p}"
    log("volumes verified against input_hashes.json")
else:                      # topic mode: the topic's registered arms (its scripts/arms.py resolution)
    PL = None
    want = [tuple(int(v) for v in s.split(":")) for s in args.planes] if args.planes else []
    summ = json.load(open(TOPIC / "logs/summary.json"))
    for s in SCANS:
        sel = {r["method"]: r for r in summ if r["contrast"] == s}
        R_ = TOPIC / "recon"
        paths[s] = {"ZF": R_ / f"zf_{s}_complex.npy", "CG-SENSE": R_ / f"cgsense_{s}_complex.npy",
                    "CS-SENSE": R_ / f"cssense_o200_lam{sel['cssense_o200']['lam']:g}_{s}_complex.npy",
                    "Joint SENSE": R_ / f"jcssense_o200_lam{sel['jcssense']['lam']:g}_{s}_complex.npy",
                    "L1-SPIRiT": R_ / f"spirit_ks{sel['spirit']['ksize']}_lam{sel['spirit']['lam']:g}_{s}_complex.npy",
                    "ZS-SSDU": R_ / f"zsssdu_{s}_complex.npy", "3D-DIP": R_ / f"dip_{s}_complex.npy",
                    "3D-INR": R_ / f"inr_{s}_complex.npy", "MC-INR": R_ / f"mcinr_{s}_complex.npy",
                    "GS-single": R_ / f"gs_single_{s}_mag.npy", MC: R_ / f"gs_joint_{s}_mag.npy"}
        for spec in args.arm:
            lab, _, pth = spec.partition("=")
            assert lab in paths[s] and lab != MC, f"--arm {spec}: unknown or not replaceable arm {lab!r}"
            paths[s][lab] = Path(pth.format(s=s))
        if args.mc_npz:
            paths[s][MC] = args.mc_npz
        missing = [str(p) for p in paths[s].values() if not p.exists()]
        assert not missing, f"missing volumes: {missing}"
    log(f"topic mode: {TOPIC}; panel numbers computed from the volumes")

gt, vol, gmax, head = {}, {s: {} for s in SCANS}, {}, {}
for s in SCANS:
    gt[s] = np.abs(np.load(TOPIC / f"recon/gt_{s}_complex.npy"))                 # float32, as evaluate.py
    gmax[s] = float(gt[s].max())
    head[s] = gt[s] > .05 * float(np.quantile(gt[s], .999))
    for lab, p in paths[s].items():
        if lab == MC and args.mc_npz:
            with np.load(p, allow_pickle=False) as z:
                vol[s][lab] = np.abs(z[{"pre": "recT1", "post": "recFL"}[s]])
            assert vol[s][lab].shape == gt[s].shape, f"--mc_npz {s}: shape {vol[s][lab].shape} != GT {gt[s].shape}"
        else:
            vol[s][lab] = np.abs(np.load(p))
with np.load(ROI, allow_pickle=False) as z:
    names = [str(n) for n in z["names"]]
    core = z[f"roi{names.index('病灶核心')}"] > 0
with h5py.File(TOPIC / "sim/mask/mask_af16_3d.h5", "r") as f:
    mask3 = np.array(f["mask"]) > 0
assert mask3.shape == gt["pre"].shape and np.all(mask3 == mask3[:1]), "expected a 3-D mask, readout (axis 0) fully sampled"
dens = uniform_filter(mask3[0].astype(float), size=9, mode="constant")
kcen = (mask3.shape[0] // 2, *(int(v) for v in np.unravel_index(np.argmax(dens), dens.shape)))
NPL, NSAMP = mask3.shape[1] * mask3.shape[2], int(mask3[0].sum())
AF_LABEL = f"nominal AF 16, actual {NPL / NSAMP:.2f}x"
DISK = np.hypot(*np.mgrid[-6:7, -6:7]) <= 6


def head_region(m):
    """display region of the error maps, as the house script: filled head without thin protrusions + 2-px rim"""
    return binary_dilation(binary_fill_holes(binary_opening(binary_fill_holes(m), structure=DISK)), iterations=2)


def D(x, axis):
    """house display transform: transverse transposed then --tra_rot quarter turns; coronal / sagittal as stored"""
    return np.rot90(x.T, args.tra_rot) if axis == 0 else x


def plane_metrics(rec, truth, mask, gm):
    """evaluate.plane_metrics arithmetic"""
    t, v = truth.astype(np.float64), rec.astype(np.float64)
    _, smap = structural_similarity(v, t, data_range=gm, full=True)
    return {"psnr": float(20 * np.log10(gm) - 10 * np.log10(np.mean((v[mask] - t[mask]) ** 2))),
            "ssim": float(smap[mask].mean()),
            "nmse": float(np.sum((v[mask] - t[mask]) ** 2) / np.sum(t[mask] ** 2))}


def frozen_numbers(p, s):
    """the evaluator's numbers for this plane and scan: opponents from the frozen baseline, MC-GS = candidate"""
    out = {lab: {} for lab in LABELS}
    for m in METRICS:
        c = p["comparisons"][f"{s}:{m}"]
        for lab in LABELS[:-1]:
            out[lab][m] = float(c["opponents"][lab])
        out[MC][m] = float(c["candidate"])
    return out


TOL = {"psnr": 1e-6, "ssim": 1e-9, "nmse": 1e-10}
args.out.mkdir(parents=True, exist_ok=True)
SAFE = re.sub(r"[^A-Za-z0-9]+", "_", args.label).strip("_")
if sd is not None:
    record = {"score_dir": str(sd), "label": args.label, "objective_sha256": sha(sd / "objective.json"),
              "lesion_planes_sha256": sha(sd / "lesion_planes.json"),
              "volumes": {s: {lab: str(p) for lab, p in paths[s].items()} for s in SCANS}, "figures": []}
else:
    record = {"topic": str(TOPIC), "label": args.label, "arm_overrides": args.arm,
              "mc_npz": (dict(path=str(args.mc_npz), keys={"pre": "recT1", "post": "recFL"}, magnitude=True)
                         if args.mc_npz else None),
              "numbers": "computed by this script (topic mode), not by a frozen evaluator",
              "volumes": {s: {lab: {"path": str(p), "sha256": sha(p)} for lab, p in paths[s].items()} for s in SCANS},
              "gt_sha256": {s: sha(TOPIC / f"recon/gt_{s}_complex.npy") for s in SCANS}, "figures": []}


def topic_plane(axis, idx):
    """topic mode: a plane record shaped like an evaluator one, numbers from plane_metrics"""
    sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
    comps, wins = {}, []
    for s in SCANS:
        num = {lab: plane_metrics(vol[s][lab][sl], gt[s][sl], head[s][sl], gmax[s]) for lab in LABELS}
        for m in METRICS:
            opp = {lab: num[lab][m] for lab in LABELS[:-1]}
            best = (min if m == "nmse" else max)(opp.values())
            win = num[MC][m] < best if m == "nmse" else num[MC][m] > best
            wins.append(win)
            comps[f"{s}:{m}"] = {"candidate": num[MC][m], "opponents": opp, "best_opponent": best,
                                 "margin": (best - num[MC][m]) if m == "nmse" else (num[MC][m] - best), "strict_win": win}
    return {"axis": axis, "index": idx, "core": int(core[sl].sum()), "strict_success": all(wins),
            "score_six": None, "comparisons": comps}


def plane_rows(axis, idx):
    """the two rows (pre, post) of one evaluator plane: data, evaluator numbers (checked), zoom box, window"""
    P = PL[(axis, idx)] if PL is not None else topic_plane(axis, idx)
    sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
    c2 = core[sl]                          # zoom box: centred on the lesion core of this plane, kept inside the slice
    assert core.shape == gt["pre"].shape and int(c2.sum()) == P["core"], "ROI does not reproduce the evaluator plane"
    assert P["core"] > 0, f"{ANAT[axis]} {idx} has no lesion core"
    rr, cc = np.nonzero(c2)
    hw = args.inset // 2
    ci = int(np.clip(round(rr.mean()), hw, c2.shape[0] - hw))
    cj = int(np.clip(round(cc.mean()), hw, c2.shape[1] - hw))
    box = np.zeros(c2.shape, bool); box[ci - hw:ci + hw, cj - hw:cj + hw] = True
    bm = D(box, axis); r_, c_ = np.nonzero(bm)
    zoom = dict(x0=int(c_.min()), y0=int(r_.min()), w=args.inset)
    Hd, Wd = bm.shape
    if args.inset_corner == "auto":        # origin="lower": y grows upward on screen; top-left holds the numbers
        bx, by = zoom["x0"] + hw, zoom["y0"] + hw
        corners = {"bl": (0, 0), "br": (Wd, 0), "tr": (Wd, Hd)}
        corner = max(corners, key=lambda k: np.hypot(bx - corners[k][0], by - corners[k][1]))
    else:
        corner = args.inset_corner
    out = []
    for s in SCANS:
        g2, m2 = gt[s][sl], head[s][sl]
        num = frozen_numbers(P, s)
        for lab in LABELS:                 # recompute and match the evaluator
            mine = plane_metrics(vol[s][lab][sl], g2, m2, gmax[s])
            for m in METRICS:
                d = abs(mine[m] - num[lab][m])
                assert d <= TOL[m] * max(1.0, abs(num[lab][m])), \
                    f"{ANAT[axis]} {idx} {s} {lab} {m}: drawn {mine[m]} vs evaluator {num[lab][m]}"
        best = {m: (min if m == "nmse" else max)(LABELS, key=lambda lab: num[lab][m]) for m in METRICS}
        out.append(dict(axis=axis, idx=idx, scan=s, sl=sl, g2=g2, emask=head_region(m2), num=num, best=best,
                        zoom=zoom, corner=corner, vmax=float(np.quantile(g2, 0.999)), med=float(np.median(g2[m2])),
                        P=P, zoom_centre=[ci, cj]))
        if args.arrow:                     # same rule as fig_t1flair_af_record_format.py
            r = out[-1]
            zs = (slice(zoom["y0"], zoom["y0"] + zoom["w"]), slice(zoom["x0"], zoom["x0"] + zoom["w"]))
            sm = {lab: uniform_filter(np.abs(vol[s][lab][sl] - g2) / gmax[s] * r["emask"], size=7, mode="nearest")
                  for lab in LABELS if lab != "ZF"}
            sc = D(np.median([sm[lab] for lab in sm if lab != MC], axis=0) - sm[MC], axis)[zs].copy()
            mg = 4 if args.arrow_mode == "lesion" else 7
            sc[:mg] = sc[-mg:] = -np.inf; sc[:, :mg] = sc[:, -mg:] = -np.inf
            if args.arrow_mode == "lesion":
                band = binary_dilation(c2, iterations=2) & ~binary_erosion(c2, iterations=2)
                inner = distance_transform_edt(r["emask"]) >= args.arrow_skull_px
                ok = D(band & inner, axis)[zs]
                assert ok[mg:-mg, mg:-mg].any(), f"{ANAT[axis]} {idx}: no lesion-boundary pixel inside the box away from the head edge"
                sc[~ok] = -np.inf
            if args.arrow_mode == "edge":
                gy, gx = np.gradient(gaussian_filter(g2 / gmax[s], 1.0))
                gm_ = D(np.hypot(gx, gy), axis)[zs]
                sc[gm_ < np.percentile(gm_, 80)] = -np.inf
            yy, xx = np.unravel_index(np.argmax(sc), sc.shape)
            r["arrow"], r["arrow_gap"] = (int(xx), int(yy)), float(sc[yy, xx])
            at = {lab: float(D(sm[lab], axis)[zs][yy, xx]) for lab in sm}
            r["arrow_worse"] = sorted(lab for lab in at if lab != MC and at[lab] > at[MC])
            assert r["arrow_gap"] > 0, f"{ANAT[axis]} {idx} {s}: no candidate point where MC-GS has the lower error"
            log(f"  arrow {ANAT[axis]} {idx} {s}: box pixel ({xx}, {yy}), error gap {r['arrow_gap']:.4f}, local error "
                f"MC-GS {at[MC]:.4f}, higher in {len(r['arrow_worse'])} of {len(at) - 1} other reconstructions "
                f"(not higher: {sorted(set(at) - set(r['arrow_worse']) - {MC})})")
    log(f"{ANAT[axis]} {idx}: all {len(LABELS) * 2 * 3} panel numbers match the evaluator; "
        f"zoom box centre ({ci}, {cj}), inset {corner}; strict {P['strict_success']}"
        + (f", S6 {P['score_six']:+.4f}" if P["score_six"] is not None else ""))
    return out


def set_windows(rows):
    """house grey windows: each row its slice P99.9; a sagittal row gets the in-head median / P99.9 ratio of the
    other planes of the same scan (bright lateral scalp fat would darken it); then / --brightness (display only)"""
    for s in SCANS:
        other = [r["med"] / r["vmax"] for r in rows if r["scan"] == s and r["axis"] != 2]
        for r in rows:
            if r["scan"] == s and r["axis"] == 2 and other:
                r["vmax"] = r["med"] / float(np.mean(other))
    for r in rows:
        r["vmax"] /= args.brightness


def render(rows, title, stem):
    """rows top to bottom, each an image row followed by its error row; house geometry"""
    if args.err_layout == "zoom":
        stem += "_zoomerr"
    stem += (f"_{args.err_cmap}" if args.err_cmap != "jet" else "") + (f"_err{args.err_vmax:g}" if args.err_vmax != 0.1 else "") \
        + (f"_arrow{args.arrow_mode}" if args.arrow else "") + ("_onerr" if args.arrow and args.arrow_on == "main+err" else "") \
        + ("_onzoom" if args.arrow and args.arrow_on == "zoom" else "")
    ncol = len(LABELS) + 1
    pw, gap = 2.6, 0.025
    CBTRIM, top_in, cbpad, cbw, cbroom = 0.10, 1.35, 0.06, 0.13, (0.95 if args.cbar == "single" else 0.60)
    top_in += 0.22 if args.arrow else 0.0      # room for the arrow line of the caption
    hs = []
    for r in rows:
        H0, W0 = D(r["g2"], r["axis"]).shape
        hs.append(pw * H0 / W0)
    ZOOMROW = args.err_layout == "zoom"
    zgap = 0.02
    half = (pw - zgap) / 2                      # zoom layout: two square halves per cell
    ehs = [half if ZOOMROW else h for h in hs]  # height of the row under each image row
    fw = ncol * pw + (ncol - 1) * gap + cbpad + cbw + cbroom
    fh = top_in + sum(hs) + sum(ehs) + (2 * len(rows) - 1) * gap + 0.16
    fig = plt.figure(figsize=(fw, fh))

    def crop(img, z):
        return img[z["y0"]:z["y0"] + z["w"], z["x0"]:z["x0"] + z["w"]]

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

    def put_inset(parent, img, r, x0_in, y0_in, hh, **imkw):
        z, corner = r["zoom"], r["corner"]
        parent.add_patch(Rectangle((z["x0"], z["y0"]), z["w"], z["w"], fill=False, ec=ZC, lw=ZLW))
        if ZOOMROW:                                 # zoom layout: the magnified box lives in the row below, no corner inset
            return
        side, pad = 0.38 * pw, 0.012 * pw
        left = x0_in + (pad if corner in ("bl", "tl") else pw - pad - side)
        top = y0_in + (pad if corner in ("tl", "tr") else hh - pad - side)
        a = fig.add_axes([left / fw, 1 - (top + side) / fh, side / fw, side / fh])
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_edgecolor(ZC); sp.set_linewidth(ZLW + 0.6)
        a.imshow(img[z["y0"]:z["y0"] + z["w"], z["x0"]:z["x0"] + z["w"]], origin="lower", aspect="auto", **imkw)

    def arrow_at(a, r, shape):
        """arrow head at the chosen point of the box, tail towards the centre of the slice image"""
        if not args.arrow or (args.arrow_on == "zoom" and ZOOMROW):
            return
        x_, y_ = r["arrow"][0] + r["zoom"]["x0"], r["arrow"][1] + r["zoom"]["y0"]
        v, L = np.array([shape[1] / 2 - x_, shape[0] / 2 - y_], float), 0.11 * shape[1]
        n = float(np.hypot(*v))
        v = v / n * L if n > 4 else np.array([L, L]) / 1.4
        a.annotate("", xy=(x_, y_), xytext=(x_ + v[0], y_ + v[1]),
                   arrowprops=dict(arrowstyle="-|>", color=args.arrow_color, lw=2.2, mutation_scale=15, shrinkA=0, shrinkB=1))

    def arrow_zoom(a, r, where="main+err"):
        """the same point in a magnified panel: head at the point, tail towards the centre of the box"""
        if not (args.arrow and args.arrow_on == where and ZOOMROW):
            return
        x_, y_, w_ = r["arrow"][0], r["arrow"][1], r["zoom"]["w"]
        v = np.array([w_ / 2 - x_, w_ / 2 - y_], float)
        n = float(np.hypot(*v))
        v = v / n * 13.0 if n > 4 else np.array([13.0, 13.0]) / 1.4
        a.annotate("", xy=(x_, y_), xytext=(x_ + v[0], y_ + v[1]),
                   arrowprops=dict(arrowstyle="-|>", color=args.arrow_color, lw=1.8, mutation_scale=13, shrinkA=0, shrinkB=1))

    y, im = top_in, None
    for k, (r, h, eh) in enumerate(zip(rows, hs, ehs)):
        s, axis = r["scan"], r["axis"]
        z = r["zoom"]
        for j in range(ncol):
            x = j * (pw + gap)
            a = axes_at(x, y, pw, h)
            if ZOOMROW:
                bL = axes_at(x, y + h + gap, half, eh)
                bR = axes_at(x + half + zgap, y + h + gap, half, eh)
            else:
                b = axes_at(x, y + h + gap, pw, h)
            if j == 0:
                gim = a.imshow(D(r["g2"], axis), cmap="gray", vmin=0, vmax=r["vmax"], origin="lower", aspect="auto")
                put_inset(a, D(r["g2"], axis), r, x, y, h, cmap="gray", vmin=0, vmax=r["vmax"])
                arrow_at(a, r, D(r["g2"], axis).shape)
                label(a, ANAT[axis], bold=True, fs=13)          # plane name only, as the house figures; indices in the title
                label(a, SCAN_NAME[s], bold=True, line=1, fs=13, hh=h)
                kplane = D(mask3.take(kcen[axis], axis=axis), axis)
                if ZOOMROW:
                    bL.imshow(crop(D(r["g2"], axis), z), cmap="gray", vmin=0, vmax=r["vmax"], origin="lower", aspect="auto")
                    zframe(bL)
                    arrow_zoom(bL, r, "zoom")
                    bR.set_facecolor("black")
                    bR.imshow(kplane, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal", interpolation="nearest")
                else:
                    b.imshow(kplane, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="auto", interpolation="nearest")
                if k == 0:
                    a.set_title("GT", fontsize=args.title_fs, pad=4)
                continue
            lab = LABELS[j - 1]
            v = vol[s][lab][r["sl"]]
            a.imshow(D(v, axis), cmap="gray", vmin=0, vmax=r["vmax"], origin="lower", aspect="auto")
            put_inset(a, D(v, axis), r, x, y, h, cmap="gray", vmin=0, vmax=r["vmax"])
            arrow_at(a, r, D(v, axis).shape)
            n = r["num"][lab]
            for i, (txt, met) in enumerate(((f"{n['psnr']:.2f} dB", "psnr"), (f"SSIM {n['ssim']:.4f}", "ssim"),
                                             (f"NMSE {n['nmse']:.5f}", "nmse"))):
                label(a, txt, bold=lab == r["best"][met], line=i, fs=10.5, hh=h, color=METRIC_COLOR)
            emap = D(np.abs(v - r["g2"]) / gmax[s] * r["emask"], axis)
            if ZOOMROW:
                bL.imshow(crop(D(v, axis), z), cmap="gray", vmin=0, vmax=r["vmax"], origin="lower", aspect="auto")
                im = bR.imshow(crop(emap, z), cmap=args.err_cmap, vmin=0.0, vmax=args.err_vmax, origin="lower", aspect="auto")
                zframe(bL); zframe(bR)
                arrow_zoom(bR, r)
                arrow_zoom(bL, r, "zoom")
            else:
                im = b.imshow(emap, cmap=args.err_cmap, vmin=0.0, vmax=args.err_vmax, origin="lower", aspect="auto")
                put_inset(b, emap, r, x, y + h + gap, h, cmap=args.err_cmap, vmin=0.0, vmax=args.err_vmax)
            if k == 0:
                a.set_title(lab if "SPIRiT" in lab else lab.upper(), fontsize=args.title_fs, pad=4)
        x_cb = (ncol * pw + (ncol - 1) * gap + cbpad) / fw
        e0 = y + h + gap + (0.04 if ZOOMROW else CBTRIM)
        if args.cbar == "single":
            if k == 0:     # one grey bar in units of each row's own window, one error bar (same scale in every row)
                gsm = ScalarMappable(norm=Normalize(0, 1), cmap="gray")
                for handle, t0, t1, tk, lab_ in ((gsm, y, y + h - CBTRIM, np.linspace(0, 1, 5), "intensity / row window"),
                                                 (im, e0, y + h + gap + eh, np.arange(0, 5) * args.err_vmax / 4, "|error| / max GT")):
                    cax = fig.add_axes([x_cb, 1 - t1 / fh, cbw / fw, (t1 - t0) / fh])
                    cb = fig.colorbar(handle, cax=cax, ticks=tk)
                    cb.ax.tick_params(labelsize=9.5)
                    cb.set_label(lab_, fontsize=10)
        else:
            for handle, t0, t1, tk in ((gim, y, y + h - CBTRIM, MaxNLocator(5)),
                                       (im, e0, y + h + gap + eh, np.arange(0, 5) * args.err_vmax / 4)):
                cax = fig.add_axes([x_cb, 1 - t1 / fh, cbw / fw, (t1 - t0) / fh])
                fig.colorbar(handle, cax=cax, ticks=tk).ax.tick_params(labelsize=9.5)
        y += h + eh + 2 * gap

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1 - 0.10 / fh, va="top")
    CN = {"bl": "bottom-left", "br": "bottom-right", "tr": "top-right"}
    corners = []
    for r in rows:
        t = f"{CN[r['corner']]} ({ANAT[r['axis']].lower()})"
        if t not in corners:
            corners.append(t)
    src = "the frozen evaluator's lesion-plane numbers" if sd is not None else "computed from the topic's registered volumes"
    cap = [f"head-masked slice PSNR / SSIM (7x7 window) / NMSE in each panel = {src}, "
           "the best of each row in bold",
           f"error |recon - GT| / max(GT), {args.err_cmap} [0, {args.err_vmax:g}]   (inside the head only)",
           ("row under each image row: left = the magnified lesion box of the image, right = the same box of the error map; "
            "GT column: GT box | sampling mask in that plane (k-space)") if ZOOMROW
           else "left of each error row: the sampling mask in that plane (k-space)",
           f"AF = {NPL} plane points / {NSAMP} sampled points, the same mask for pre and post",
           ("grey bar: intensity as a fraction of each row's own window (slice P99.9 / brightness)"
            if args.cbar == "single" else "grey bar right of each image row: that row's window")]
    cap2 = (f"in every panel, the {args.inset}x{args.inset} box centred on the lesion core of that plane is outlined on the "
            + ("slice and magnified in the row below" if ZOOMROW
               else "slice and magnified into the " + " / ".join(corners) + " corner"))
    if args.arrow:
        cap2 += ("\n" + ("arrow in each magnified image box" if args.arrow_on == "zoom" and ZOOMROW else "arrow on each slice")
                 + " (same place along a row): where the error of the other methods (median of the nine "
                 "reconstructions other than ZF and MC-GS, 7x7 mean) exceeds that of MC-GS the most inside the box"
                 + (", among its tissue boundaries (steepest fifth of the GT gradient)" if args.arrow_mode == "edge" else "")
                 + (", on the boundary of the lesion core" if args.arrow_mode == "lesion" else "")
                 + ("; also drawn in the magnified error map" if args.arrow_on == "main+err" and ZOOMROW else "")
                 + " -- chosen with the GT, for display only")
    fig.text(0.5, 1 - 0.40 / fh, ";   ".join(cap[:2]) + "\n" + ";   ".join(cap[2:]) + "\n" + cap2,
             ha="center", va="top", fontsize=11)
    fig.savefig(args.out / f"{stem}.png", dpi=args.png_dpi)
    if args.pdf:
        for im_ in fig.findobj(mimage.AxesImage):
            im_.set_rasterized(True)
        fig.savefig(args.out / f"{stem}.pdf", dpi=args.pdf_dpi)
    plt.close(fig)
    if args.save_data:
        dat = {}
        for r in rows:
            k = f"{ANAT[r['axis']].lower()}{r['idx']}_{r['scan']}"
            dat[f"{k}__GT"] = r["g2"]
            dat[f"{k}__head_mask"] = head[r["scan"]][r["sl"]]
            dat[f"{k}__error_region"] = r["emask"]
            dat[f"{k}__lesion_core"] = core[r["sl"]]
            for lab in LABELS:
                dat[f"{k}__{lab.replace(' ', '_')}"] = vol[r["scan"]][lab][r["sl"]]
            dat[f"{ANAT[r['axis']].lower()}_sampling_mask_plane"] = mask3.take(kcen[r["axis"]], axis=r["axis"])
        dat["meta_json"] = np.array(json.dumps(dict(
            note="magnitude slices in volume orientation (display: transverse = rot90(x.T, tra_rot), others as stored, origin "
                 "lower); error = |recon - GT| / gt_max inside error_region; metrics inside head_mask with PSNR peak gt_max; "
                 "zoom / arrow positions are in display pixels (arrow relative to the zoom box)",
            title=title, gt_max=gmax, tra_rot=args.tra_rot, error_display=dict(cmap=args.err_cmap, vmax=args.err_vmax),
            zoom_display_px={f"{ANAT[r['axis']]}_{r['idx']}_{r['scan']}": r["zoom"] for r in rows},
            arrows=({f"{ANAT[r['axis']]}_{r['idx']}_{r['scan']}": r["arrow"] for r in rows} if args.arrow else None),
            arrow_mode=args.arrow_mode if args.arrow else None,
            numbers={f"{ANAT[r['axis']]}_{r['idx']}_{r['scan']}": r["num"] for r in rows},
            volumes=record["volumes"]), ensure_ascii=False))
        np.savez_compressed(args.out / f"{stem}_data.npz", **dat)
        log(f"  wrote {args.out / stem}_data.npz  sha256 {sha(args.out / (stem + '_data.npz'))}")
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(args.out / f"{stem}.png") as I:
        I.convert("RGB").resize((2000, round(2000 * I.height / I.width)), Image.LANCZOS).save(
            args.out / f"{stem}_preview.jpg", quality=90)
    planes = []
    for r in rows:
        if (r["axis"], r["idx"]) not in [(p["axis"], p["index"]) for p in planes]:
            P = r["P"]
            planes.append(dict(axis=r["axis"], index=r["idx"], core=P["core"], strict_success=P["strict_success"],
                               score_six=P["score_six"], zoom_centre=r["zoom_centre"], inset_corner=r["corner"],
                               margins={c: P["comparisons"][c]["margin"] for c in P["comparisons"]}))
    record["figures"].append(dict(png=str(args.out / f"{stem}.png"), rows=[f"{r['scan']} {ANAT[r['axis']]} {r['idx']}" for r in rows],
                                  planes=planes, error_display=dict(cmap=args.err_cmap, vmax=args.err_vmax),
                                  arrows=([dict(row=f"{r['scan']} {ANAT[r['axis']]} {r['idx']}", box_xy=r["arrow"],
                                                gap=r["arrow_gap"], error_higher_than_mcgs_in=r["arrow_worse"]) for r in rows] if args.arrow else None),
                                  arrow_mode=args.arrow_mode if args.arrow else None, arrow_on=args.arrow_on if args.arrow else None,
                                  role=args.role or None,
                                  pdf_image_dpi=args.pdf_dpi if args.pdf else None, png_dpi=args.png_dpi))
    log(f"  wrote {args.out / stem}.png")


HEAD = (f"{args.role}: " if args.role else "") + f"UIH pre/post, CS-SENSE truth   {AF_LABEL}"
if args.three:
    rows = [r for axis, idx in enumerate(args.three) for r in plane_rows(axis, idx)]   # pre/post of each plane in turn
    set_windows(rows)
    t, c_, s_ = args.three
    render(rows, f"{HEAD}   --   Transverse {t} / Coronal {c_} / Sagittal {s_} (lesion-containing planes), pre and post"
                 f"   --   MC-GS = {args.label}",
           f"ce_af16_lesion_three_planes_{t}_{c_}_{s_}_prepost_{SAFE}")
else:
    for axis, idx in want:
        rows = plane_rows(axis, idx)
        set_windows(rows)
        render(rows, f"{ANAT[axis]} {idx} (lesion-containing plane, lesion core {rows[0]['P']['core']} px)   {HEAD}"
                     f"   --   MC-GS = {args.label}",
               f"ce_af16_lesion_{ANAT[axis].lower()}{idx}_prepost_{SAFE}")
(args.out / f"record_{SAFE}{'_three' if args.three else ''}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
log("DONE")
