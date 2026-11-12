#!/usr/bin/env python3
"""fig_t1flair_record_format.py -- T1 + FLAIR three-plane figure of a diff-mask topic (default: the uniform-density
mask, T1FLAIR_AF16_uniform_v3_jiangyizhou) in the format of the plain-scan / enhanced figure of record
(ce_af16_lesion_figs_20261107/fig_lesion_plane_prepost.py --three --err_layout zoom --cbar single), drawn from
WHATEVER ARMS HAVE FINISHED. Written 2026-11-12 to look at the uniform-mask study while its comparison arms were
still running: the topic has no logs/summary.json and no registered GS volumes before its finalize step, so its own
scripts/fig_three_planes_compact.py cannot run yet.

Format (as fig_peds_prepost_format.py): one figure for both contrasts; rows transverse T1, transverse FLAIR,
coronal T1, coronal FLAIR, sagittal T1, sagittal FLAIR; under each image row a zoom row (left = magnified box of
the image, right = the same box of the error map; GT column: GT box | sampling mask in that plane); head-masked
slice PSNR / SSIM / NMSE in every panel, best of each row in bold; one grey bar and one error bar.

This topic's own conventions (its scripts/fig_three_planes_compact.py): volume (256, 256, 176), axis 0 transverse,
1 coronal, 2 sagittal; transverse shown transposed and turned a quarter (eyes up), the other two as stored;
origin lower; grey window = slice P99.9, the sagittal row at the in-head median level of the other two planes;
head = |GT| > 0.05 P99.9 of the volume, PSNR at max|GT|, 7x7 SSIM map averaged over the head pixels, head NMSE.
Default slices 180 / 124 / 76 with a 48-voxel cube where they meet = the inset figure of finalize_diffmask.sh.

Where each column comes from
  ZF, CG-SENSE, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR   recon/<stem>_<c>_complex.npy
  CS-SENSE, Joint SENSE, L1-SPIRiT                the swept volume with the highest head PSNR in the topic's logs
                                                  (summarize.py's rule), marked when it sits on the edge of its grid
  GS-single, MC-GS                                results_arc/af16u_p3_*_full4k_volumes.npz, the channel that
                                                  add_gs_arm.py / finalize_diffmask.sh register (plain = recT1,
                                                  contrast = recFL); the npz's own GT must equal the topic's GT
A column is drawn only when BOTH contrasts have its volume; the others are listed in the title as pending.
With --require_all the script stops instead (use that for a final figure).

Read-only on the topic and the worktree; writes PNG, preview and a JSON record (every volume's SHA-256) into
MCGS_ARTIFACT_DIR. CPU only. Neither GT nor any ROI enters a reconstruction.

  gpu.py run --cpu -- @python autoresearch/uniform_af16_fig_20261112/fig_t1flair_record_format.py
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--kind", default="uniform", help="topic T1FLAIR_AF16_<kind>_v3_jiangyizhou")
ap.add_argument("--exp_prefix", default="af16u_p3", help="results_arc/<prefix>_{joint,singleT1,singleFLAIR}_full4k")
ap.add_argument("--slices", type=int, nargs=3, default=[180, 124, 76], metavar=("TRA", "COR", "SAG"))
ap.add_argument("--box", type=int, default=48, help="edge of the zoom cube in voxels, centred where the slices meet")
ap.add_argument("--err_vmax", type=float, default=0.1)
ap.add_argument("--brightness", type=float, default=1.0, help="display only: window top / brightness")
ap.add_argument("--title_fs", type=float, default=22.0)
ap.add_argument("--png_dpi", type=int, default=200)
ap.add_argument("--require_all", action="store_true", help="stop when a column has no volume yet")
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
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_opening, uniform_filter  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402

os.nice(10)
R = Path(f"/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_{args.kind}_v3_jiangyizhou")
ARC = Path("/data0/congcong/code/GS/mcgs_diffmask_wt/results_arc")
CONTRASTS = ("T1", "FLAIR")
MC = "MC-GS"
PLANES = [(0, "Transverse"), (1, "Coronal"), (2, "Sagittal")]
METRICS = ("psnr", "ssim", "nmse")
METRIC_COLOR = "#FFC23D"
ZC, ZLW = METRIC_COLOR, 2.2
ORDER = ["ZF", "CG-SENSE", "CS-SENSE", "Joint SENSE", "L1-SPIRiT", "ZS-SSDU", "3D-DIP", "3D-INR", "MC-INR", "GS-single", MC]
log = lambda *a: print(*a, flush=True)
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", "t1flair_record_format_out")).resolve()
art.mkdir(parents=True, exist_ok=True)
assert R.is_dir(), f"no topic {R}"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def D(x, axis):
    """the topic's display transform (fig_three_planes_compact.py, --tra_rot 1)"""
    return np.rot90(x.T, 1) if axis == 0 else x


# ---------------- which volume fills which column ----------------
def swept(c, pattern, contrast_in_row):
    """best row of a swept family by head PSNR (summarize.py's rule) -> (row, n rows, on the grid edge)"""
    cands = [r for f in sorted((R / "logs").glob(pattern)) for r in json.load(open(f)).get("rows", [])
             if (r.get("contrast", c) == c or not contrast_in_row) and r.get("head_psnr") is not None]
    cands = [r for r in cands if r.get("contrast", c) == c]
    if not cands:
        return None
    b = max(cands, key=lambda r: r["head_psnr"])
    lams = sorted({float(r["lam"]) for r in cands if r.get("lam") is not None})
    return b, len(cands), bool(lams) and float(b["lam"]) in (lams[0], lams[-1]), lams


def lam_path(prefix, lam, c):
    hit = [p for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy")
           if (m := re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name))
           and abs(float(m.group(1)) - float(lam)) < 1e-12]
    return hit[0] if len(hit) == 1 else None


def logged(stem, c, method):
    p = R / f"logs/{stem}_{c}.json"
    if not p.exists():
        return None
    return next((r.get("head_psnr") for r in json.load(open(p)).get("rows", []) if r.get("method") == method), None)


def sources(c):
    """label -> dict(path, [npz_key], logged head PSNR, note) for every column whose volume exists"""
    out = {}
    for lab, stem, logstem in (("ZF", "zf", "classical"), ("CG-SENSE", "cgsense", "classical"), ("ZS-SSDU", "zsssdu", "zsssdu"),
                               ("3D-DIP", "dip", "dip"), ("3D-INR", "inr", "inr"), ("MC-INR", "mcinr", "mcinr")):
        p = R / f"recon/{stem}_{c}_complex.npy"
        if p.exists():
            out[lab] = dict(path=p, logged=logged(logstem, c, stem), note="")
    for lab, pattern, prefix in (("CS-SENSE", f"classical_{c}_o200_lam*.json", "cssense_o200"),
                                 ("Joint SENSE", "jcssense_o200_lam*.json", "jcssense_o200")):
        s = swept(c, pattern, True)
        if s:
            b, n, edge, lams = s
            p = lam_path(prefix, b["lam"], c)
            if p is not None:
                out[lab] = dict(path=p, logged=b["head_psnr"], grid_edge=edge, n_grid=n,
                                note=f"lam {float(b['lam']):g} of {n}" + (" (grid edge)" if edge else ""))
    s = swept(c, f"spirit_{c}_ks*_lam*.json", True)
    if s:
        b, n, edge, lams = s
        p = R / f"recon/spirit_ks{b['ksize']}_lam{float(b['lam']):g}_{c}_complex.npy"
        if p.exists():
            out["L1-SPIRiT"] = dict(path=p, logged=b["head_psnr"], grid_edge=edge, n_grid=n,
                                    note=f"ks {b['ksize']} lam {float(b['lam']):g} of {n}" + (" (grid edge)" if edge else ""))
    # the GS arms, as add_gs_arm.py / finalize_diffmask.sh register them
    for lab, exp, ch in ((MC, f"{args.exp_prefix}_joint_full4k", "plain" if c == "T1" else "contrast"),
                         ("GS-single", f"{args.exp_prefix}_single{c}_full4k", "plain")):
        npz, sj = ARC / f"{exp}_volumes.npz", ARC / f"{exp}_score.json"
        if npz.exists() and sj.exists():
            d = json.load(open(sj))
            assert d.get(f"{ch}_name") == c, f"{exp}: channel {ch} is {d.get(f'{ch}_name')}, not {c}"
            out[lab] = dict(path=npz, npz_key={"plain": "recT1", "contrast": "recFL"}[ch],
                            gt_key={"plain": "gtT1", "contrast": "gtFL"}[ch], logged=d[ch].get("head_psnr"), note=exp)
    return out


SRC = {c: sources(c) for c in CONTRASTS}
LABELS = [lab for lab in ORDER if all(lab in SRC[c] for c in CONTRASTS)]
PENDING = [lab for lab in ORDER if lab not in LABELS]
log("columns drawn:", LABELS)
log("columns without a volume for both contrasts yet:", PENDING or "none")
assert MC in LABELS, "no MC-GS volumes yet"
assert not (args.require_all and PENDING), f"--require_all: missing {PENDING}"

# ---------------- volumes ----------------
gt, gmax, head, vol = {}, {}, {}, {c: {} for c in CONTRASTS}
record = dict(topic=str(R), format="fig_lesion_plane_prepost.py --three --err_layout zoom --cbar single",
              interim=bool(PENDING), drawn=LABELS, pending=PENDING,
              slices=dict(zip(("transverse", "coronal", "sagittal"), args.slices)),
              volumes={c: {} for c in CONTRASTS}, gt_sha256={}, rows=[])
for c in CONTRASTS:
    g = R / f"recon/gt_{c}_complex.npy"
    gt[c] = np.abs(np.load(g)).astype(np.float32)
    gmax[c] = float(gt[c].max())
    head[c] = gt[c] > 0.05 * float(np.quantile(gt[c], 0.999))
    record["gt_sha256"][c] = sha(g)
    for lab in LABELS:
        s = SRC[c][lab]
        if "npz_key" in s:
            with np.load(s["path"]) as z:
                v = np.asarray(z[s["npz_key"]], dtype=np.float32)
                dg = float(np.abs(np.asarray(z[s["gt_key"]], dtype=np.float32) - gt[c]).max())
            assert dg <= 1e-5 * gmax[c], f"{c} {lab}: the run's GT differs from the topic's GT by {dg} (max GT {gmax[c]})"
        else:
            v, dg = np.abs(np.load(s["path"])).astype(np.float32), None
        assert v.shape == gt[c].shape, (c, lab, v.shape)
        vol[c][lab] = v
        e = v[head[c]].astype(np.float64) - gt[c][head[c]].astype(np.float64)
        mine = float(20 * np.log10(gmax[c]) - 10 * np.log10(float((e * e).mean())))
        record["volumes"][c][lab] = dict(path=str(s["path"]), npz_key=s.get("npz_key"), sha256=sha(s["path"]), note=s["note"],
                                         grid_edge=s.get("grid_edge"), volume_head_psnr=mine, logged_head_psnr=s["logged"],
                                         gt_max_abs_diff_vs_topic=dg)
SHAPE = gt["T1"].shape
assert gt["FLAIR"].shape == SHAPE
log("\nvolume head PSNR (this script's head rule) | the topic's logged head PSNR | source")
worst = 0.0
for c in CONTRASTS:
    for lab in LABELS:
        r = record["volumes"][c][lab]
        d = None if r["logged_head_psnr"] is None else r["volume_head_psnr"] - r["logged_head_psnr"]
        worst = max(worst, abs(d or 0.0))
        log(f"  {c:<5} {lab:<12} {r['volume_head_psnr']:8.3f} | " + ("    n/a " if d is None else f"{r['logged_head_psnr']:8.3f} ({d:+.4f})")
            + f" | {Path(r['path']).name}" + (f"[{r['npz_key']}]" if r["npz_key"] else "") + (f"  {r['note']}" if r["note"] else ""))
record["max_abs_diff_vs_logged_head_psnr"] = worst
log(f"largest |difference| to the logged head PSNR: {worst:.4f} dB")

with h5py.File(R / "data/mask_af16_3d.h5", "r") as f:
    mask3 = np.array(f["mask"]) > 0
assert mask3.shape == SHAPE and np.all(mask3 == mask3[:1]), "expected a 3-D mask, readout (axis 0) fully sampled"
dens = uniform_filter(mask3[0].astype(float), size=9, mode="constant")
kcen = (mask3.shape[0] // 2, *(int(v) for v in np.unravel_index(np.argmax(dens), dens.shape)))
NPL, NSAMP = mask3.shape[1] * mask3.shape[2], int(mask3[0].sum())
AF = {c: json.load(open(R / "logs/prep_04_export.json"))["contrasts"][c]["AF_vs_full_rect"] for c in CONTRASTS}
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

# ---------------- rows ----------------
rows = []
for (axis, pname), idx in zip(PLANES, args.slices):
    sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
    rr, cc = np.nonzero(D(BOX[sl], axis))
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
for r in rows:
    r["vmax"] /= args.brightness

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
TITLE = (f"jiangyizhou T1 + FLAIR, uniform-density mask (vdPoisMex pp = 0) + 24 x 16 centre, nominal AF 16"
         if args.kind == "uniform" else f"jiangyizhou T1 + FLAIR, {args.kind} mask, nominal AF 16")
TITLE += f"   --   Transverse {t_} / Coronal {c_} / Sagittal {s_}"
if PENDING:
    TITLE += f"\nINTERIM: {len(LABELS)} of {len(ORDER)} columns; still running: " + ", ".join(PENDING)
win = "slice P99.9" + (f" / {args.brightness:g}" if args.brightness != 1 else "")
edge = sorted({lab for c in CONTRASTS for lab in LABELS if SRC[c][lab].get("grid_edge")})
CAP = ["head-masked slice PSNR / SSIM (7x7 window) / NMSE in each panel, the best of each row in bold",
       f"error |recon - GT| / max(GT), jet [0, {args.err_vmax:g}], inside the head only",
       "row under each image row: left = the magnified box of the image, right = the same box of the error map; "
       "GT column: GT box | sampling mask in that plane (k-space)",
       f"{NSAMP} of {NPL} plane points sampled: T1 {AF['T1']:.2f}x, FLAIR {AF['FLAIR']:.2f}x of the full plane",
       f"the {args.box}-voxel box where the three slices meet is outlined on every slice; grey bar: fraction of each row's own window ({win})"]
if edge:
    CAP.append("swept arms are shown at their best lambda by head PSNR; on the edge of its grid for a contrast: " + ", ".join(edge))


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
n_title = TITLE.count("\n") + 1
cap_y = 0.10 + n_title * 13 * 1.35 / 72 + 0.06
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


IMKW = dict(origin="lower", aspect="auto")
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

fig.suptitle(TITLE, fontsize=13, fontweight="bold", y=1 - 0.10 / fh, va="top")
fig.text(0.5, 1 - cap_y / fh, "\n".join(cap_lines), ha="center", va="top", fontsize=11, linespacing=1.32)

stem = f"t1flair_af16_{args.kind}_three_planes_{t_}_{c_}_{s_}_record_format" + ("_interim" if PENDING else "") + args.tag
fig.savefig(art / f"{stem}.png", dpi=args.png_dpi)
plt.close(fig)
from PIL import Image  # noqa: E402
Image.MAX_IMAGE_PIXELS = None
with Image.open(art / f"{stem}.png") as I:
    log("figure size (px):", I.size)
    I.convert("RGB").resize((2000, round(2000 * I.height / I.width)), Image.LANCZOS).save(art / f"{stem}_preview.jpg", quality=90)

log(f"\nMC-GS on each row (rank among the {len(LABELS)} drawn arms; margin over the best other drawn arm):")
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
(art / f"record_{stem}.json").write_text(json.dumps(record, indent=2))
log(f"MC-GS first on PSNR, SSIM and NMSE in all six rows (among the {len(LABELS)} drawn arms):", all_first)
log("wrote", art / f"{stem}.png")
log("DONE")
