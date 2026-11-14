#!/usr/bin/env python3
"""abl_fig.py -- result figures of the framework ablation (design: DESIGN.md, numbers: RESULTS.md).

One figure per task (ce, t1flair, t1t2) in the house three-plane style of fig_lesion_plane_prepost.py
(--err_layout zoom --cbar single), plus one overview with the transverse plane of all three tasks.

  rows     first contrast transverse, second contrast transverse, then coronal, then sagittal; under each
           image row: per column the magnified 48 x 48 box of the image (left) and of the error map (right)
  columns  GT, ZF, Full MC-GS, w/o anisotropy, w/o joint modelling, w/o intra-contrast priors
           (the order of the result tables)

Volumes are the ones the ablation runs saved (results_arc/fw_<task>_<kind>_volumes.npz, magnitude after hard
data consistency, trainer gauge). For every drawn volume the whole-head PSNR is recomputed with the trainer's
definition (head = |GT| > 0.05 P99.9, PSNR at max|GT|) and must equal the run's score JSON, so the drawn
volumes are the scored ones. Panel numbers are slice metrics with the house arithmetic (head-masked PSNR at
the volume's max|GT|, 7x7 SSIM map averaged over the head, head NMSE), all in the trainer gauge; the w/o
columns also carry the slice PSNR difference from Full.

Planes: ce = 201 / 131 / 48 (lesion planes, box centred on the lesion core of each plane, as the September
figure); t1flair = 180 / 124 / 76 (the planes of the T1+FLAIR figures of record, box where the three planes
meet); t1t2 = 150 / 113 / 77 (the topic's documented planes, box where the three planes meet).

CPU only, read-only on the ablation outputs and the topics. Neither GT nor the ROI enters any reconstruction;
the ROI only places the zoom box.

  gpu.py run --cpu -- @python autoresearch/ce_af16_ablation_20261110/abl_fig.py
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", nargs="+", default=["ce", "t1flair", "t1t2"], choices=["ce", "t1flair", "t1t2"])
ap.add_argument("--inset", type=int, default=48)
ap.add_argument("--err_vmax", type=float, default=0.1)
ap.add_argument("--brightness", type=float, default=1.5)
ap.add_argument("--png_dpi", type=int, default=200)
ap.add_argument("--pdf_dpi", type=int, default=600)
ap.add_argument("--tra_rot", type=int, default=1, choices=[0, 1, 2, 3])
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

A = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_framework_ablation_20261110")
RES = A / "trainer/results_arc"
OUT = A / "fig_ablation"
R0 = Path("/data0/congcong/code/GS/Results_20260908")
ROI = R0 / "CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz"
TASKS = {
    "ce": dict(names=("pre", "post"), show=("Pre-contrast", "Post-contrast"), short="CE, AF16",
               title="pre/post contrast, AF16 (September data)", planes=(201, 131, 48), box="lesion",
               mask=R0 / "CE_retro_cssense_af16/sim/mask/mask_af16_3d.h5"),
    "t1flair": dict(names=("T1", "FLAIR"), show=("T1", "FLAIR"), short="T1+FLAIR, AF16",
                    title="T1+FLAIR, AF16", planes=(180, 124, 76), box="meet",
                    mask=R0 / "T1FLAIR_AF16_v3_jiangyizhou/data/mask_af16_3d.h5"),
    "t1t2": dict(names=("T1", "T2"), show=("T1", "T2"), short="T1+T2, AF12",
                 title="T1+T2, AF12", planes=(150, 113, 77), box="meet",
                 mask=R0 / "T1T2_AF12_retro/data/mask_af12_3d.h5"),
}
FULL, ISO, NOJ, NOP = "Full MC-GS", "w/o anisotropy", "w/o joint modelling", "w/o intra-contrast priors"
LABELS = ["ZF", FULL, ISO, NOJ, NOP]
HEADER = {"ZF": "ZF", FULL: "Full MC-GS", ISO: "w/o anisotropy", NOJ: "w/o joint\nmodelling", NOP: "w/o intra-contrast\npriors"}
ANAT = {0: "Transverse", 1: "Coronal", 2: "Sagittal"}
METRICS = ("psnr", "ssim", "nmse")
METRIC_COLOR = "#FFC23D"
ZC, ZLW = METRIC_COLOR, 2.2
DISK = np.hypot(*np.mgrid[-6:7, -6:7]) <= 6
log = lambda *a: print(*a, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def head_region(m):
    """display region of the error maps, as the house script"""
    return binary_dilation(binary_fill_holes(binary_opening(binary_fill_holes(m), structure=DISK)), iterations=2)


def D(x, axis):
    """house display transform: transverse transposed then --tra_rot quarter turns; coronal / sagittal as stored"""
    return np.rot90(x.T, args.tra_rot) if axis == 0 else x


def plane_metrics(rec, truth, mask, gm):
    t, v = truth.astype(np.float64), rec.astype(np.float64)
    _, smap = structural_similarity(v, t, data_range=gm, full=True)
    return {"psnr": float(20 * np.log10(gm) - 10 * np.log10(np.mean((v[mask] - t[mask]) ** 2))),
            "ssim": float(smap[mask].mean()),
            "nmse": float(np.sum((v[mask] - t[mask]) ** 2) / np.sum(t[mask] ** 2))}


record = {"root": str(A), "out": str(OUT), "tasks": {}, "figures": []}


def task_rows(task):
    """every (plane, contrast) row of one task, with the slices, numbers, zoom box and sampling-mask plane"""
    T = TASKS[task]
    pn, cn = T["names"]
    files = {k: RES / f"fw_{task}_{k}_volumes.npz" for k in ("full", "iso", "noprior")}
    files.update({f"nojoint_{n}": RES / f"fw_{task}_nojoint_{n}_volumes.npz" for n in (pn, cn)})
    score = {k: json.load(open(str(p).replace("_volumes.npz", "_score.json"))) for k, p in files.items()}
    z = {k: np.load(p) for k, p in files.items()}
    with h5py.File(T["mask"], "r") as f:
        mask3 = np.array(f["mask"]) > 0
    assert np.all(mask3 == mask3[:1]), "expected a 3-D mask with the readout (axis 0) fully sampled"
    dens = uniform_filter(mask3[0].astype(float), size=9, mode="constant")
    kcen = (mask3.shape[0] // 2, *(int(v) for v in np.unravel_index(np.argmax(dens), dens.shape)))
    npl, nsamp = mask3.shape[1] * mask3.shape[2], int(mask3[0].sum())
    core = None
    if T["box"] == "lesion":
        with np.load(ROI, allow_pickle=False) as r:
            names = [str(n) for n in r["names"]]
            core = r[f"roi{names.index('病灶核心')}"] > 0
    rec = {"tasks": {}, "files": {k: {"path": str(p), "sha256": sha(p)} for k, p in files.items()},
           "mask": str(T["mask"]), "af_actual": npl / nsamp, "planes": list(T["planes"]), "contrasts": {}}
    per_contrast = {}
    for slot, name, show, suf in (("plain", pn, T["show"][0], "T1"), ("contrast", cn, T["show"][1], "FL")):
        gt = z["full"]["gt" + suf]
        assert gt.shape == mask3.shape
        gmax = float(gt.max())
        head = gt > .05 * float(np.quantile(gt, .999))
        nj = z[f"nojoint_{name}"]
        assert str(nj["contrast_name"]) == name
        vol = {"ZF": z["full"]["zf" + suf], FULL: z["full"]["rec" + suf], ISO: z["iso"]["rec" + suf],
               NOJ: nj["rec"], NOP: z["noprior"]["rec" + suf]}
        # one gauge for the whole task: every run built the same reference and the same zero-filled volume
        for k, g in (("iso", z["iso"]["gt" + suf]), ("noprior", z["noprior"]["gt" + suf]), ("nojoint", nj["gt"])):
            d = float(np.abs(g - gt).max()) / gmax
            assert d < 1e-5, f"{task} {name}: reference of the {k} run differs from the full run ({d:.2e} of max)"
        # the drawn volumes are the scored ones: whole-head PSNR, trainer definition, against the score JSONs
        want = {FULL: score["full"][slot]["head_psnr"], ISO: score["iso"][slot]["head_psnr"],
                NOJ: score[f"nojoint_{name}"]["plain"]["head_psnr"], NOP: score["noprior"][slot]["head_psnr"]}
        g64 = gt[head].astype(np.float64)
        whole = {}
        for lab in LABELS:
            whole[lab] = float(20 * np.log10(gmax) - 10 * np.log10(np.mean((vol[lab][head].astype(np.float64) - g64) ** 2)))
            if lab in want:
                assert abs(whole[lab] - want[lab]) < 1e-3, f"{task} {name} {lab}: volume {whole[lab]} vs score JSON {want[lab]}"
        log(f"{task} {name}: whole-head PSNR " + ", ".join(f"{lab} {whole[lab]:.3f}" for lab in LABELS) + "  (matches the score JSONs)")
        per_contrast[name] = dict(show=show, gt=gt, gmax=gmax, head=head, vol=vol, whole=whole)
        rec["contrasts"][name] = {"whole_head_psnr": whole, "gmax": gmax}
    rows = []
    for axis, idx in enumerate(T["planes"]):
        sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
        shape2 = per_contrast[pn]["gt"][sl].shape
        hw = args.inset // 2
        if core is not None:
            c2 = core[sl]
            assert c2.any(), f"{ANAT[axis]} {idx} has no lesion core"
            rr, cc = np.nonzero(c2)
            ci, cj = round(rr.mean()), round(cc.mean())
        else:
            ci, cj = [T["planes"][a] for a in range(3) if a != axis]
        ci = int(np.clip(ci, hw, shape2[0] - hw)); cj = int(np.clip(cj, hw, shape2[1] - hw))
        box = np.zeros(shape2, bool); box[ci - hw:ci + hw, cj - hw:cj + hw] = True
        r_, c_ = np.nonzero(D(box, axis))
        zoom = dict(x0=int(c_.min()), y0=int(r_.min()), w=args.inset)
        kplane = D(mask3.take(kcen[axis], axis=axis), axis)
        for name in (pn, cn):
            C = per_contrast[name]
            g2, m2 = C["gt"][sl], C["head"][sl]
            v2 = {lab: C["vol"][lab][sl] for lab in LABELS}
            num = {lab: plane_metrics(v2[lab], g2, m2, C["gmax"]) for lab in LABELS}
            best = {m: (min if m == "nmse" else max)(LABELS, key=lambda lab: num[lab][m]) for m in METRICS}
            rows.append(dict(task=task, short=T["short"], name=name, show=C["show"], axis=axis, idx=idx, g2=g2,
                             emask=head_region(m2), v2=v2, num=num, best=best, zoom=zoom, zoom_centre=[ci, cj],
                             gmax=C["gmax"], vmax=float(np.quantile(g2, 0.999)), med=float(np.median(g2[m2])),
                             kplane=kplane, whole=C["whole"], af=f"{npl} / {nsamp} = {npl / nsamp:.2f}x"))
            log(f"  {ANAT[axis]} {idx} {name}: " + ", ".join(f"{lab} {num[lab]['psnr']:.3f}" for lab in LABELS)
                + f"; box centre ({ci}, {cj})")
    # house grey windows: slice P99.9; a sagittal row takes the in-head median / P99.9 ratio of the other planes
    for name in (pn, cn):
        other = [r["med"] / r["vmax"] for r in rows if r["name"] == name and r["axis"] != 2]
        for r in rows:
            if r["name"] == name and r["axis"] == 2:
                r["vmax"] = r["med"] / float(np.mean(other))
    for r in rows:
        r["vmax"] /= args.brightness
    rec["slice_numbers"] = [dict(plane=f"{ANAT[r['axis']]} {r['idx']}", contrast=r["name"], zoom_centre=r["zoom_centre"],
                                 **{lab: r["num"][lab] for lab in LABELS}) for r in rows]
    record["tasks"][task] = rec
    for v in z.values():
        v.close()
    return rows


def render(rows, title, stem):
    ncol = len(LABELS) + 1
    pw, gap, zgap = 2.6, 0.025, 0.02
    cbpad, cbw, cbroom = 0.06, 0.13, 0.95
    half = (pw - zgap) / 2
    hs = [pw * D(r["g2"], r["axis"]).shape[0] / D(r["g2"], r["axis"]).shape[1] for r in rows]
    seen, whole_lines = [], []
    for r in rows:                              # one whole-head line per (task, contrast)
        if (r["task"], r["name"]) not in seen:
            seen.append((r["task"], r["name"]))
            whole_lines.append(f"{r['short']}, {r['show']}: " + " / ".join(f"{r['whole'][lab]:.3f}" for lab in LABELS[1:]))
    afs = []
    for r in rows:
        t = f"{r['short']}: {r['af']}"
        if t not in afs:
            afs.append(t)
    cap = ["columns: fully sampled reference (GT), zero-filled (ZF), the full model, and the three ablations, each "
           "compared with the same full model (one training run each, seed 42)",
           "in each panel: head-masked slice PSNR / SSIM (7x7 window) / NMSE, best of the row in bold; white line = "
           "slice PSNR minus that of Full MC-GS",
           "row under each image row: left = the magnified box of the image, right = the same box of the error map "
           f"|recon - GT| / max(GT), jet [0, {args.err_vmax:g}], head only;   GT column: GT box | sampling mask (k-space)",
           "whole-head PSNR in dB (the table of record), Full / w/o anisotropy / w/o joint modelling / w/o intra-contrast priors:"]
    for i in range(0, len(whole_lines), 2):
        cap.append(";     ".join(whole_lines[i:i + 2]))
    cap.append("sampling: plane points / sampled points = " + ";   ".join(afs)
               + ";     grey bar: intensity as a fraction of each row's own window")
    top_in = 0.50 + 0.19 * len(cap) + 0.66
    fw = ncol * pw + (ncol - 1) * gap + cbpad + cbw + cbroom
    fh = top_in + sum(hs) + len(rows) * half + (2 * len(rows) - 1) * gap + 0.16
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

    def label(a, text, bold=False, line=0, fs=12, hh=None, color="white", low=False, nlines=4):
        """text block in the top-left corner, or (low) in the bottom-left one when the zoom box sits top-left"""
        step = fs * 1.28 / 72 / hh if hh else 0.0
        yy, va = (0.025 + (nlines - 1 - line) * step, "bottom") if low else (0.975 - line * step, "top")
        t = a.text(0.025, yy, text, transform=a.transAxes, ha="left", va=va, color=color, fontsize=fs,
                   fontweight="bold" if bold else "normal")
        t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])

    y, im = top_in, None
    for k, (r, h) in enumerate(zip(rows, hs)):
        axis, z = r["axis"], r["zoom"]
        Hd, Wd = D(r["g2"], axis).shape
        low = z["x0"] < 0.50 * Wd and z["y0"] + z["w"] > 0.70 * Hd      # the box would sit under the numbers
        for j in range(ncol):
            x = j * (pw + gap)
            a = axes_at(x, y, pw, h)
            bL = axes_at(x, y + h + gap, half, half)
            bR = axes_at(x + half + zgap, y + h + gap, half, half)
            if j == 0:
                g = D(r["g2"], axis)
                a.imshow(g, cmap="gray", vmin=0, vmax=r["vmax"], origin="lower", aspect="auto")
                a.add_patch(Rectangle((z["x0"], z["y0"]), z["w"], z["w"], fill=False, ec=ZC, lw=ZLW))
                label(a, ANAT[axis], bold=True, fs=13, hh=h, low=low, nlines=3)
                label(a, r["show"], bold=True, line=1, fs=13, hh=h, low=low, nlines=3)
                label(a, r["short"], line=2, fs=13, hh=h, low=low, nlines=3)
                bL.imshow(crop(g, z), cmap="gray", vmin=0, vmax=r["vmax"], origin="lower", aspect="auto")
                zframe(bL)
                bR.set_facecolor("black")
                bR.imshow(r["kplane"], cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal", interpolation="nearest")
                if k == 0:
                    a.set_title("GT", fontsize=18, pad=5)
                continue
            lab = LABELS[j - 1]
            v = D(r["v2"][lab], axis)
            a.imshow(v, cmap="gray", vmin=0, vmax=r["vmax"], origin="lower", aspect="auto")
            a.add_patch(Rectangle((z["x0"], z["y0"]), z["w"], z["w"], fill=False, ec=ZC, lw=ZLW))
            n = r["num"][lab]
            for i, (txt, met) in enumerate(((f"{n['psnr']:.2f} dB", "psnr"), (f"SSIM {n['ssim']:.4f}", "ssim"),
                                             (f"NMSE {n['nmse']:.5f}", "nmse"))):
                label(a, txt, bold=lab == r["best"][met], line=i, fs=10.5, hh=h, color=METRIC_COLOR, low=low)
            if lab in (ISO, NOJ, NOP):
                label(a, f"{n['psnr'] - r['num'][FULL]['psnr']:+.2f} dB vs Full", line=3, fs=10.5, hh=h, low=low)
            emap = D(np.abs(r["v2"][lab] - r["g2"]) / r["gmax"] * r["emask"], axis)
            bL.imshow(crop(v, z), cmap="gray", vmin=0, vmax=r["vmax"], origin="lower", aspect="auto")
            im = bR.imshow(crop(emap, z), cmap="jet", vmin=0.0, vmax=args.err_vmax, origin="lower", aspect="auto")
            zframe(bL); zframe(bR)
            if k == 0:
                a.set_title(HEADER[lab], fontsize=18, pad=5, linespacing=1.0)
        if k == 0:
            x_cb = (ncol * pw + (ncol - 1) * gap + cbpad) / fw
            gsm = ScalarMappable(norm=Normalize(0, 1), cmap="gray")
            for handle, t0, t1, tk, lab_ in ((gsm, y, y + h - 0.10, np.linspace(0, 1, 5), "intensity / row window"),
                                             (im, y + h + gap + 0.04, y + h + gap + half, np.arange(0, 5) * args.err_vmax / 4,
                                              "|error| / max GT")):
                cax = fig.add_axes([x_cb, 1 - t1 / fh, cbw / fw, (t1 - t0) / fh])
                cb = fig.colorbar(handle, cax=cax, ticks=tk)
                cb.ax.tick_params(labelsize=9.5)
                cb.set_label(lab_, fontsize=10)
        y += h + half + 2 * gap

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1 - 0.10 / fh, va="top")
    fig.text(0.5, 1 - 0.42 / fh, "\n".join(cap), ha="center", va="top", fontsize=10.5, linespacing=1.28)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=args.png_dpi)
    for im_ in fig.findobj(mimage.AxesImage):
        im_.set_rasterized(True)
    fig.savefig(OUT / f"{stem}.pdf", dpi=args.pdf_dpi)
    plt.close(fig)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(OUT / f"{stem}.png") as I:
        size = I.size
        I.convert("RGB").resize((2000, round(2000 * I.height / I.width)), Image.LANCZOS).save(
            OUT / f"{stem}_preview.jpg", quality=90)
    record["figures"].append(dict(stem=stem, png=str(OUT / f"{stem}.png"), pdf=str(OUT / f"{stem}.pdf"), png_px=size,
                                  rows=[f"{r['task']} {r['name']} {ANAT[r['axis']]} {r['idx']}" for r in rows]))
    log(f"wrote {OUT / stem}.png  {size[0]} x {size[1]} px")


allrows = {}
for task in args.tasks:
    allrows[task] = task_rows(task)
    t, c_, s_ = TASKS[task]["planes"]
    render(allrows[task], f"MC-GS framework ablation   --   {TASKS[task]['title']}   --   "
                          f"Transverse {t} / Coronal {c_} / Sagittal {s_}",
           f"ablation_{task}_three_planes_{t}_{c_}_{s_}")
if len(args.tasks) > 1:
    rows = [r for task in args.tasks for r in allrows[task] if r["axis"] == 0]
    render(rows, "MC-GS framework ablation   --   transverse plane of each task:   "
                 + ";   ".join(f"{TASKS[t]['short']} (slice {TASKS[t]['planes'][0]})" for t in args.tasks),
           "ablation_overview_transverse")
(OUT / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))

art = Path(os.environ.get("MCGS_ARTIFACT_DIR", OUT / "artifacts"))
art.mkdir(parents=True, exist_ok=True)
shutil.copy2(OUT / "record.json", art / "record.json")
budget = 44 << 20                                # gpu.py fetch takes at most 50 MiB
for f in record["figures"]:                      # previews first, then the full PNGs while they fit
    p = OUT / f"{f['stem']}_preview.jpg"
    shutil.copy2(p, art / p.name); budget -= p.stat().st_size
for f in record["figures"][::-1]:                # overview first
    p = Path(f["png"])
    if p.stat().st_size <= budget:
        shutil.copy2(p, art / p.name); budget -= p.stat().st_size
    else:
        log(f"not copied to the artifacts (size): {p.name}")
log("DONE")
