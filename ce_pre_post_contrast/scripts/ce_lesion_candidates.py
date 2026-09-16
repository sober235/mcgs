#!/usr/bin/env python3
"""ce_lesion_candidates.py -- look for focal enhancement in the CE pair, as candidates for slices that show a lesion.

There is no annotation or clinical report for this exam, and normal structures enhance too (vessels, dural sinuses,
choroid plexus, pituitary). So this ranks candidates and renders them for a human decision; it does not diagnose.

  prep     : pre/post magnitudes of the chosen reconstruction -> NIfTI (1 mm, RAS affine with the axis facts below),
             SynthSeg-robust 2.0 on the pre-contrast T1 -> anatomical labels on this grid
  analyze  : pre/post alignment check (phase correlation), white-matter normalisation, enhancement map, connected
             blobs inside the brain parenchyma (ventricles and CSF excluded), shape and location features -> table
  render   : three planes through each top candidate: pre, post, post - pre, crosshair, crop around the blob
  slices   : the lesion seen by eye on the axial montage (image-left upper hemisphere, T1-dark with a darker core; the
             enhancement search above only returned vessels and sinuses): its darkest core inside --core_box, the
             axial/coronal/sagittal slice where that core is largest, pre/post figure with zoom -> lesion_cores_<recon>.json
  enhance  : what one static post-contrast scan shows -- barrier leakage and blood pool, not flow: three planes
             through the lesion, pre / post / post - pre, enhancing set and lesion core outlined, region means

Axis facts (checked on this data, 2026-09-15): volumes are (S-I 256, A-P 232, L-R 176); axis 0 grows superior,
A-P index 0 is anterior (the orbits sit at low indices). The patient's left/right along axis 2 is not recorded;
radiological convention is assumed (index grows toward patient left), which only affects SynthSeg's left/right names.

  python scripts/ce_lesion_candidates.py prep|analyze|render|slices [--recon cssense|cgsense|zsssdu ...]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

T = Path(__file__).resolve().parents[1]
BUNDLE = T / "export/ce_l1spirit_20260915"
OUT = T / "lesion_candidates"
FIGS = Path("/home/congcongliu/figs/ce_pair/lesion_candidates")
SYNTHSEG = Path("/home/congcongliu/src/SynthSeg/scripts/commands/SynthSeg_predict.py")
SYNTH_PY = "/home/congcongliu/anaconda3/envs/synthseg/bin/python"
# SynthSeg 2.0 labels
WM, GM_CTX = (2, 41), (3, 42)
VENTRICLES, CSF = (4, 5, 14, 15, 43, 44), (24,)
NAMES = {2: "L cerebral WM", 3: "L cortex", 4: "L lat ventricle", 5: "L inf lat vent", 7: "L cerebellum WM",
         8: "L cerebellum ctx", 10: "L thalamus", 11: "L caudate", 12: "L putamen", 13: "L pallidum", 14: "3rd ventricle",
         15: "4th ventricle", 16: "brainstem", 17: "L hippocampus", 18: "L amygdala", 24: "CSF", 26: "L accumbens",
         28: "L ventral DC", 41: "R cerebral WM", 42: "R cortex", 43: "R lat ventricle", 44: "R inf lat vent",
         46: "R cerebellum WM", 47: "R cerebellum ctx", 49: "R thalamus", 50: "R caudate", 51: "R putamen",
         52: "R pallidum", 53: "R hippocampus", 54: "R amygdala", 58: "R accumbens", 60: "R ventral DC"}


def affine_for(shape_lr_ap_si):
    """RAS affine for data stored as (L-R, A-P, S-I): x = -i, y = -j, z = +k, centred."""
    nx, ny, nz = shape_lr_ap_si
    A = np.diag([-1.0, -1.0, 1.0, 1.0])
    A[:3, 3] = [nx / 2, ny / 2, -nz / 2]
    return A


def load_mag(recon, scan):
    return np.abs(np.load(BUNDLE / f"volumes/{scan}/{recon}_{scan}_complex.npy")).astype(np.float32)


def prep(a):
    import nibabel as nib
    OUT.mkdir(parents=True, exist_ok=True)
    for scan in ("pre", "post"):
        v = load_mag(a.recon, scan)                                       # (S-I, A-P, L-R)
        d = np.transpose(v, (2, 1, 0))                                    # (L-R, A-P, S-I)
        d = d / np.percentile(d, 99.9)
        nib.save(nib.Nifti1Image(d.astype(np.float32), affine_for(d.shape)), OUT / f"{a.recon}_{scan}.nii.gz")
    cmd = [SYNTH_PY, str(SYNTHSEG), "--i", str(OUT / f"{a.recon}_pre.nii.gz"), "--o", str(OUT / f"{a.recon}_pre_seg.nii.gz"),
           "--robust", "--cpu", "--threads", "16", "--vol", str(OUT / f"{a.recon}_pre_volumes.csv")]
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_VISIBLE_DEVICES=""))
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    if r.returncode != 0:
        sys.exit("SynthSeg failed")


def seg_on_grid(a, shape_si_ap_lr):
    """SynthSeg labels resampled (nearest) onto the (S-I, A-P, L-R) array grid."""
    import nibabel as nib
    from scipy.ndimage import affine_transform
    img = nib.load(OUT / f"{a.recon}_pre_seg.nii.gz")
    seg = np.asarray(img.dataobj).astype(np.int16)
    nx, ny, nz = shape_si_ap_lr[2], shape_si_ap_lr[1], shape_si_ap_lr[0]
    A_data = affine_for((nx, ny, nz))
    M = np.linalg.inv(img.affine) @ A_data                                # data voxel (i,j,k) -> seg voxel
    lab = affine_transform(seg, M[:3, :3], offset=M[:3, 3], output_shape=(nx, ny, nz), order=0, mode="constant", cval=0)
    return np.transpose(lab, (2, 1, 0))                                    # (S-I, A-P, L-R)


def phase_corr_shift(a, b):
    """integer + parabolic sub-voxel shift s such that b(x) ~ a(x - s)."""
    Fa, Fb = np.fft.fftn(a), np.fft.fftn(b)
    R = Fb * np.conj(Fa)
    R /= np.abs(R) + 1e-12
    c = np.abs(np.fft.ifftn(R))
    idx = np.unravel_index(np.argmax(c), c.shape)
    shift = []
    for ax, i in enumerate(idx):
        n = c.shape[ax]
        im, ip = [list(idx) for _ in range(2)]
        im[ax], ip[ax] = (i - 1) % n, (i + 1) % n
        y0, ym, yp = c[idx], c[tuple(im)], c[tuple(ip)]
        frac = 0.5 * (ym - yp) / (ym - 2 * y0 + yp + 1e-12)
        s = i + frac
        shift.append(s - n if s > n / 2 else s)
    return np.array(shift)


def analyze(a):
    from scipy import ndimage as ndi
    pre, post = load_mag(a.recon, "pre"), load_mag(a.recon, "post")
    lab = seg_on_grid(a, pre.shape)
    brain = lab > 0
    wm = np.isin(lab, WM)
    report = dict(recon=a.recon, labels_present=int(len(np.unique(lab)) - 1), brain_voxels=int(brain.sum()))
    # alignment: phase correlation on the dilated brain region
    roi = ndi.binary_dilation(brain, iterations=6)
    sh = phase_corr_shift(pre * roi, post * roi)
    report["post_vs_pre_shift_voxels"] = sh.round(3).tolist()
    ncc = lambda u, v: float((u * v).sum() / np.sqrt((u * u).sum() * (v * v).sum()))
    report["ncc_brain_before"] = ncc(pre * brain, post * brain)
    if np.abs(sh).max() > 0.3:
        post = ndi.shift(post, -sh, order=1, mode="nearest")
        report["ncc_brain_after_shift"] = ncc(pre * brain, post * brain)
    # white-matter normalisation and enhancement
    pn, qn = pre / np.median(pre[wm]), post / np.median(post[wm])
    D = ndi.gaussian_filter(qn - pn, 1.0)
    Rr = ndi.gaussian_filter(qn, 1.0) / np.maximum(ndi.gaussian_filter(pn, 1.0), 0.05)
    parench = brain & ~np.isin(lab, VENTRICLES + CSF)
    excl = ndi.binary_dilation(np.isin(lab, VENTRICLES), iterations=2)   # choroid plexus sits in/at the ventricles
    core = parench & ~excl
    med = float(np.median(D[core]))
    mad = float(np.median(np.abs(D[core] - med))) * 1.4826
    thr = med + a.k * mad
    report.update(D_median=med, D_mad=mad, D_threshold=thr)
    cand = core & (D > thr) & (Rr > a.min_ratio)
    cc, n = ndi.label(cand, structure=np.ones((3, 3, 3)))
    surf = ndi.distance_transform_edt(brain)                              # mm to the brain edge
    vdist = ndi.distance_transform_edt(~np.isin(lab, VENTRICLES))
    rows = []
    for i, sl in enumerate(ndi.find_objects(cc), start=1):
        m = cc[sl] == i
        nvox = int(m.sum())
        if nvox < a.min_vox:
            continue
        pts = np.argwhere(m) + np.array([s.start for s in sl])
        cen = pts.mean(0)
        ev = np.sort(np.linalg.eigvalsh(np.cov((pts - cen).T) + 1e-6 * np.eye(3)))[::-1]
        elong = float(np.sqrt(ev[0] / ev[2]))
        vals_D, vals_R = D[tuple(pts.T)], Rr[tuple(pts.T)]
        labs = lab[tuple(pts.T)]
        top = np.bincount(labs[labs > 0]).argmax() if (labs > 0).any() else 0
        ci = tuple(int(round(c)) for c in cen)
        rows.append(dict(id=len(rows) + 1, voxels=nvox, centroid_si_ap_lr=[int(c) for c in ci],
                         bbox_extent=[int(s.stop - s.start) for s in sl], elongation=round(elong, 2),
                         D_mean=round(float(vals_D.mean()), 3), D_peak=round(float(vals_D.max()), 3),
                         ratio_mean=round(float(vals_R.mean()), 2), ratio_peak=round(float(vals_R.max()), 2),
                         label=NAMES.get(int(top), str(int(top))), depth_mm=round(float(surf[ci]), 1),
                         to_ventricle_mm=round(float(vdist[ci]), 1),
                         score=round(nvox ** (1 / 3) * float(vals_D.mean()) / max(elong, 1.0) ** 0.5, 3)))
    rows.sort(key=lambda r: -r["score"])
    for k, r in enumerate(rows, start=1):
        r["rank"] = k
    report["n_candidates"] = len(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(report=report, candidates=rows), open(OUT / f"{a.recon}_candidates.json", "w"), indent=1)
    np.save(OUT / f"{a.recon}_enhancement_D.npy", D.astype(np.float32))
    np.save(OUT / f"{a.recon}_labels.npy", lab)
    print(json.dumps(report, indent=1))
    print(f"{'rank':>4} {'vox':>6} {'centroid(SI,AP,LR)':>20} {'extent':>14} {'elong':>6} {'Dmean':>6} {'Rmean':>6} {'depth':>6} {'toV':>5}  label")
    for r in rows[:a.top]:
        print(f"{r['rank']:>4} {r['voxels']:>6} {str(r['centroid_si_ap_lr']):>20} {str(r['bbox_extent']):>14} {r['elongation']:>6} "
              f"{r['D_mean']:>6} {r['ratio_mean']:>6} {r['depth_mm']:>6} {r['to_ventricle_mm']:>5}  {r['label']}")


def blobs(a):
    """blob-shaped enhancement: scale-normalised Hessian of post - pre, bright in all three directions (tubes, i.e.
    vessels, have one near-zero eigenvalue and score ~0), inside parenchyma at least --depth mm from the brain edge."""
    from scipy import ndimage as ndi
    pre, post = load_mag(a.recon, "pre"), load_mag(a.recon, "post")
    lab = np.load(OUT / f"{a.recon}_labels.npy")
    brain, wm = lab > 0, np.isin(lab, WM)
    pn, qn = pre / np.median(pre[wm]), post / np.median(post[wm])
    D0 = qn - pn
    Dsm = ndi.gaussian_filter(D0, 1.0)
    Rr = ndi.gaussian_filter(qn, 1.0) / np.maximum(ndi.gaussian_filter(pn, 1.0), 0.05)
    parench = brain & ~np.isin(lab, VENTRICLES + CSF)
    core = parench & ~ndi.binary_dilation(np.isin(lab, VENTRICLES), iterations=2)
    depth = ndi.distance_transform_edt(brain)
    sel = core & (depth >= a.depth) & (Dsm > a.dmin)
    idx = np.nonzero(sel)
    best = np.zeros(idx[0].size, np.float32); best_s = np.zeros(idx[0].size, np.float32)
    for sig in [float(v) for v in a.scales.split(",")]:
        H = np.empty((idx[0].size, 3, 3), np.float64)
        for i in range(3):
            for j in range(i, 3):
                order = [0, 0, 0]; order[i] += 1; order[j] += 1
                h = ndi.gaussian_filter(D0, sig, order=order)[idx] * sig ** 2
                H[:, i, j] = h; H[:, j, i] = h
        ev = np.linalg.eigvalsh(H)                      # ascending: ev[:, 2] is the one closest to zero / largest
        score = np.where(ev[:, 2] < 0, -ev[:, 2], 0.0).astype(np.float32)
        better = score > best
        best[better], best_s[better] = score[better], sig
    B = np.zeros(D0.shape, np.float32); B[idx] = best
    S = np.zeros(D0.shape, np.float32); S[idx] = best_s
    peaks = (B == ndi.maximum_filter(B, size=a.nms)) & (B > a.bmin)
    vdist = ndi.distance_transform_edt(~np.isin(lab, VENTRICLES))
    rows = []
    for z, y, x in np.argwhere(peaks):
        rows.append(dict(centroid_si_ap_lr=[int(z), int(y), int(x)], blob=round(float(B[z, y, x]), 3), scale_vox=float(S[z, y, x]),
                         D=round(float(Dsm[z, y, x]), 3), ratio=round(float(Rr[z, y, x]), 2),
                         pre_wm_units=round(float(ndi.gaussian_filter(pn[z-3:z+4, y-3:y+4, x-3:x+4], 1.0)[3, 3, 3]), 2),
                         label=NAMES.get(int(lab[z, y, x]), str(int(lab[z, y, x]))), depth_mm=round(float(depth[z, y, x]), 1),
                         to_ventricle_mm=round(float(vdist[z, y, x]), 1),
                         score=round(float(B[z, y, x]) * float(Dsm[z, y, x]), 4)))
    rows.sort(key=lambda r: -r["score"])
    for k, r in enumerate(rows, start=1):
        r["rank"] = k
        r["voxels"], r["bbox_extent"], r["elongation"] = 0, [0, 0, 0], 0.0
        r["D_mean"], r["ratio_mean"] = r["D"], r["ratio"]
    json.dump(dict(report=dict(recon=a.recon, method="hessian blobs", scales=a.scales, depth_mm=a.depth, dmin=a.dmin,
                               bmin=a.bmin, n=len(rows), post_vs_pre_shift_voxels=[0, 0, 0]), candidates=rows),
              open(OUT / f"{a.recon}_blobs.json", "w"), indent=1)
    print(f"{len(rows)} blob peaks (depth >= {a.depth} mm, D > {a.dmin}, blob > {a.bmin})")
    print(f"{'rank':>4} {'centroid(SI,AP,LR)':>20} {'blob':>6} {'scale':>5} {'D':>6} {'post/pre':>8} {'pre':>5} {'depth':>6} {'toV':>5}  label")
    for r in rows[:a.top]:
        print(f"{r['rank']:>4} {str(r['centroid_si_ap_lr']):>20} {r['blob']:>6} {r['scale_vox']:>5} {r['D']:>6} {r['ratio']:>8} "
              f"{r['pre_wm_units']:>5} {r['depth_mm']:>6} {r['to_ventricle_mm']:>5}  {r['label']}")


def render(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    J = json.load(open(OUT / f"{a.recon}_{a.source}.json"))
    if a.points:
        J["candidates"] = [dict(rank=900 + k, centroid_si_ap_lr=[int(v) for v in p.split(",")], voxels=0, bbox_extent=[0, 0, 0],
                                elongation=0, D_mean="-", ratio_mean="-", depth_mm="-", label="given point")
                           for k, p in enumerate(a.points.split(";"))]
        a.ids = ",".join(str(900 + k) for k in range(len(a.points.split(";"))))
    pre, post = load_mag(a.recon, "pre"), load_mag(a.recon, "post")
    sh = np.array(J["report"]["post_vs_pre_shift_voxels"])
    if np.abs(sh).max() > 0.3:
        from scipy import ndimage as ndi
        post = ndi.shift(post, -sh, order=1, mode="nearest")
    lab = np.load(OUT / f"{a.recon}_labels.npy")
    wm = np.isin(lab, WM)
    pn, qn = pre / np.median(pre[wm]), post / np.median(post[wm])
    D = qn - pn
    FIGS.mkdir(parents=True, exist_ok=True)
    ids = [int(v) for v in a.ids.split(",")] if a.ids else [r["rank"] for r in J["candidates"][:a.top]]
    for rank in ids:
        r = next(c for c in J["candidates"] if c["rank"] == rank)
        z, y, x = r["centroid_si_ap_lr"]
        h = a.crop // 2
        planes = [("axial z=%d" % z, lambda v: v[z], (y, x)),
                  ("coronal y=%d" % y, lambda v: v[:, y, :][::-1], (255 - z, x)),
                  ("sagittal x=%d" % x, lambda v: v[:, :, x][::-1], (255 - z, y))]
        fig, axs = plt.subplots(3, 6, figsize=(18, 9.4))
        for pi, (pname, take, (cr, cc_)) in enumerate(planes):
            for col, (vol, ttl, cmap, lim) in enumerate([(pn, "pre", "gray", (0, 1.6)), (qn, "post", "gray", (0, 1.6)),
                                                          (D, "post - pre", "magma", (0, 0.8))]):
                im = take(vol)
                ax = axs[pi, 2 * col]
                ax.imshow(im, cmap=cmap, vmin=lim[0], vmax=lim[1]); ax.axhline(cr, color="c", lw=0.4); ax.axvline(cc_, color="c", lw=0.4)
                ax.set_title(f"{pname}  {ttl}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
                r0, c0 = max(cr - h, 0), max(cc_ - h, 0)
                ax2 = axs[pi, 2 * col + 1]
                ax2.imshow(im[r0:cr + h, c0:cc_ + h], cmap=cmap, vmin=lim[0], vmax=lim[1])
                ax2.plot([cc_ - c0], [cr - r0], "c+", ms=10, mew=0.8)
                ax2.set_title(f"zoom {ttl}", fontsize=8); ax2.set_xticks([]); ax2.set_yticks([])
        fig.suptitle(f"{a.recon} candidate rank {rank}: {r['voxels']} vox, extent {r['bbox_extent']}, elong {r['elongation']}, "
                     f"D {r['D_mean']}, post/pre {r['ratio_mean']}, depth {r['depth_mm']} mm, {r['label']}  "
                     f"(axial anterior up; coronal/sagittal superior up, sagittal anterior left)", fontsize=10)
        fig.tight_layout()
        p = FIGS / f"{a.recon}_{a.source}_r{rank:03d}_z{z}_y{y}_x{x}.png"
        fig.savefig(p, dpi=85); plt.close(fig)
        print(p)


def wm_units(a):
    """pre and post in white-matter units (median of the scan's own white matter = 1), plus the labels."""
    lab = np.load(OUT / f"{a.recon}_labels.npy")
    wm = np.isin(lab, WM)
    pre, post = load_mag(a.recon, "pre"), load_mag(a.recon, "post")
    return pre / np.median(pre[wm]), post / np.median(post[wm]), lab


def core_components(a, pn, lab):
    """dark components of the pre-contrast scan inside --core_box (the box is set by eye around the lesion)."""
    from scipy import ndimage as ndi
    box = np.zeros(pn.shape, bool)
    box[tuple(slice(*map(int, s.split(":"))) for s in a.core_box.split(","))] = True
    inner = ndi.binary_erosion(ndi.binary_fill_holes(lab > 0), iterations=5)
    core = box & inner & (ndi.gaussian_filter(pn, 1.0) < a.core_thr)
    cc, n = ndi.label(core, structure=np.ones((3, 3, 3)))
    sizes = ndi.sum(core, cc, range(1, n + 1))
    return cc, sizes, np.argsort(sizes)[::-1]


def slices(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle
    pn, qn, lab = wm_units(a)
    cc, sizes, order = core_components(a, pn, lab)
    rows = []
    for i in order[:6]:
        pts = np.argwhere(cc == i + 1)
        rows.append(dict(vox=int(sizes[i]), centroid=[round(float(v), 1) for v in pts.mean(0)],
                         z=[int(pts[:, 0].min()), int(pts[:, 0].max())], y=[int(pts[:, 1].min()), int(pts[:, 1].max())],
                         x=[int(pts[:, 2].min()), int(pts[:, 2].max())],
                         pre=round(float(pn[tuple(pts.T)].mean()), 3), post=round(float(qn[tuple(pts.T)].mean()), 3),
                         label=int(np.bincount(lab[tuple(pts.T)]).argmax())))
    m = cc == order[0] + 1
    area = dict(axial=m.sum((1, 2)), coronal=m.sum((0, 2)), sagittal=m.sum((0, 1)))
    z, y, x = (int(area[k].argmax()) for k in ("axial", "coronal", "sagittal"))
    json.dump(dict(recon=a.recon, core_box=a.core_box, core_thr=a.core_thr, slices=dict(axial_z=z, coronal_y=y, sagittal_x=x),
                   core_area_mm2={k: {int(i): int(v) for i, v in enumerate(area[k]) if v} for k in area}, cores=rows),
              open(OUT / f"lesion_cores_{a.recon}.json", "w"), indent=1)
    print(f"largest core {rows[0]}\nslices: axial z={z} ({area['axial'][z]} mm2)  coronal y={y} ({area['coronal'][y]} mm2)  "
          f"sagittal x={x} ({area['sagittal'][x]} mm2)")
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    z0, z1, y0, y1, x0, x1 = 105, 252, 25, 230, 0, 176                     # brain crop
    views = [("横断位 z=%d" % z, lambda v: v[z, y0:y1, x0:x1], (y - y0, x - x0)),
             ("冠状位 y=%d" % y, lambda v: v[z0:z1, y, x0:x1][::-1], (z1 - 1 - z, x - x0)),
             ("矢状位 x=%d" % x, lambda v: v[z0:z1, y0:y1, x][::-1], (z1 - 1 - z, y - y0))]
    h = 34                                                                  # zoom half-width, mm
    ratios = [(x1 - x0) / (y1 - y0), 1, (x1 - x0) / (z1 - z0), 1, (y1 - y0) / (z1 - z0), 1]   # equal panel heights
    fig, axs = plt.subplots(2, 6, figsize=(19, 7.0), gridspec_kw=dict(width_ratios=ratios), layout="compressed")
    for r, (vol, name) in enumerate([(pn, "平扫"), (qn, "增强")]):
        for c, (title, take, (cr, cc_)) in enumerate(views):
            im = take(vol)
            ax = axs[r, 2 * c]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1.5)
            ax.add_patch(Rectangle((cc_ - h, cr - h), 2 * h, 2 * h, fill=False, ec="#FFC23D", lw=1.0))
            ax.set_title(f"{name}  {title}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
            ax2 = axs[r, 2 * c + 1]
            ax2.imshow(im[max(cr - h, 0):cr + h, max(cc_ - h, 0):cc_ + h], cmap="gray", vmin=0, vmax=1.5)
            ax2.set_title("放大", fontsize=10); ax2.set_xticks([]); ax2.set_yticks([])
    cz, cy, cx = (int(round(v)) for v in rows[0]["centroid"])
    name = {"cssense": "CS-SENSE", "cgsense": "CG-SENSE", "zsssdu": "ZS-SSDU"}.get(a.recon, a.recon)
    fig.suptitle(f"CE 平扫/增强（{name} 重建）：过病灶的三轴面层（暗核心质心 z {cz}, y {cy}, x {cx}）\n"
                 "横断位前方朝上；冠状/矢状位上方朝上；矢状位前方朝左；图像左侧 = 左右轴低索引侧（患者左右未记录）", fontsize=11)
    p = FIGS / f"lesion_slices_z{z}_y{y}_x{x}.png"
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=110); plt.close(fig)
    print(p)


def enhance(a):
    """what one post-contrast scan does and does not show: leakage through a broken blood-brain barrier and the blood
    pool, nothing about flow. Three planes through the lesion, pre / post / post - pre, with the enhancing set and the
    lesion core outlined, plus the mean post-minus-pre of reference regions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from scipy import ndimage as ndi
    pn, qn, lab = wm_units(a)
    D = ndi.gaussian_filter(qn - pn, 1.0)
    cc, sizes, order = core_components(a, pn, lab)
    C = cc == order[0] + 1
    near = ndi.binary_dilation(lab > 0, iterations=4)              # dural sinuses sit just outside the SynthSeg mask
    away = ndi.binary_dilation(C, iterations=10)                   # keep the lesion out of the normal-tissue regions
    rois = [("病灶核心", C), ("核心外 0-3 mm", ndi.binary_dilation(C, iterations=3) & ~C),
            ("正常白质", np.isin(lab, WM) & ~away), ("正常皮层", np.isin(lab, GM_CTX) & ~away),
            ("血管/静脉窦", near & (qn > 2.0) & (D > 0.5)),         # blood pool: bright after, not before
            ("脉络丛", np.isin(lab, VENTRICLES) & (qn > 1.2))]      # no blood-brain barrier, enhances normally
    tab = []
    for name, m in rois:
        tab.append(dict(roi=name, vox=int(m.sum()), pre=round(float(pn[m].mean()), 3),
                        post=round(float(qn[m].mean()), 3), d=round(float((qn - pn)[m].mean()), 3)))
        print(f"{name:<14}{tab[-1]['vox']:>8} vox  pre {tab[-1]['pre']:>6.3f}  post {tab[-1]['post']:>6.3f}  post-pre {tab[-1]['d']:>+7.3f}")
    enh = (D > a.dthr) & near
    cz, cy, cx = enh.sum((1, 2)), enh.sum((0, 2)), enh.sum((0, 1))
    if a.point:
        z, y, x = (int(v) for v in a.point.split(","))
    else:
        pts = np.argwhere(C)                                       # the point in the lesion whose three planes carry
        s = cz[pts[:, 0]] / cz.max() + cy[pts[:, 1]] / cy.max() + cx[pts[:, 2]] / cx.max()   # the most enhancement
        z, y, x = (int(v) for v in pts[int(s.argmax())])
    print(f"point z={z} y={y} x={x}; enhancing area in its planes: {int(cz[z])}, {int(cy[y])}, {int(cx[x])} mm2")
    json.dump(dict(recon=a.recon, point=dict(z=z, y=y, x=x), dthr=a.dthr, rois=tab),
              open(OUT / f"enhance_rois_{a.recon}.json", "w"), indent=1)
    np.savez_compressed(OUT / f"enhance_rois_{a.recon}.npz", core=C, **{f"roi{i}": m for i, (_, m) in enumerate(rois)},
                        names=np.array([n for n, _ in rois]))          # ce_figure_paper.py --diff reads these
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    z0, z1, y0, y1, x0, x1 = 105, 252, 25, 230, 0, 176             # brain crop
    views = [(f"横断位 z={z}", lambda v: v[z, y0:y1, x0:x1]),
             (f"冠状位 y={y}", lambda v: v[z0:z1, y, x0:x1][::-1]),
             (f"矢状位 x={x}", lambda v: v[z0:z1, y0:y1, x][::-1, ::-1])]
    ratios = [(x1 - x0) / (y1 - y0), (x1 - x0) / (z1 - z0), (y1 - y0) / (z1 - z0)]
    fig, axs = plt.subplots(3, 3, figsize=(13, 9.6), gridspec_kw=dict(width_ratios=ratios), layout="compressed")
    for r, (vol, rlab, cmap, lim) in enumerate([(pn, "平扫", "gray", (0, 1.5)), (qn, "增强", "gray", (0, 1.5)),
                                                (D, "增强 − 平扫", "magma", (0, 0.6))]):
        for c, (title, take) in enumerate(views):
            ax = axs[r, c]
            ax.imshow(take(vol), cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax.contour(take(enh).astype(float), levels=[0.5], colors="#3BE8FF", linewidths=0.45)
            ax.contour(take(C).astype(float), levels=[0.5], colors="#FFC23D", linewidths=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(title, fontsize=11)
            if c == 0: ax.set_ylabel(rlab, fontsize=12)
    t = {r["roi"]: r["d"] for r in tab}
    nm = {"cssense": "CS-SENSE", "cgsense": "CG-SENSE", "zsssdu": "ZS-SSDU"}.get(a.recon, a.recon)
    fig.suptitle(f"一次静态增强能说明什么：漏出与血池，不是血流（{nm} 重建，各自按白质归一，白质 = 1）\n"
                 f"增强 − 平扫的均值：病灶核心 {t['病灶核心']:+.3f}，核心外 0–3 mm {t['核心外 0-3 mm']:+.3f}，"
                 f"正常白质 {t['正常白质']:+.3f}，正常皮层 {t['正常皮层']:+.3f}，"
                 f"血管/静脉窦 {t['血管/静脉窦']:+.3f}，脉络丛 {t['脉络丛']:+.3f}\n"
                 f"青线 = 强化区（增强−平扫 > {a.dthr}），橙线 = 病灶核心；横断位前方朝上，冠状/矢状位上方朝上、矢状位前方朝右",
                 fontsize=11)
    p = FIGS / f"enhancement_explained_z{z}_y{y}_x{x}.png"
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=120); plt.close(fig)
    print(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "analyze", "blobs", "render", "slices", "enhance"])
    ap.add_argument("--recon", default="cssense")
    ap.add_argument("--k", type=float, default=6.0, help="threshold = median + k * MAD of the enhancement map in parenchyma")
    ap.add_argument("--min_ratio", type=float, default=1.3)
    ap.add_argument("--min_vox", type=int, default=10)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--ids", default="")
    ap.add_argument("--crop", type=int, default=60)
    ap.add_argument("--source", default="candidates", choices=["candidates", "blobs"])
    ap.add_argument("--points", default="", help="render given points instead: 'z,y,x;z,y,x'")
    ap.add_argument("--scales", default="1.5,2.5,4")
    ap.add_argument("--depth", type=float, default=3.0)
    ap.add_argument("--dmin", type=float, default=0.25)
    ap.add_argument("--bmin", type=float, default=0.05)
    ap.add_argument("--nms", type=int, default=7)
    ap.add_argument("--core_box", default="180:224,75:155,22:82", help="slices: search box 'z0:z1,y0:y1,x0:x1' set by eye")
    ap.add_argument("--core_thr", type=float, default=0.32, help="slices: core = smoothed pre (WM units) below this")
    ap.add_argument("--dthr", type=float, default=0.3, help="enhance: a voxel counts as enhancing above this post - pre")
    ap.add_argument("--point", default="201,131,48", help="enhance: 'z,y,x' to draw; empty = the point in the lesion "
                                                          "whose three planes carry the most enhancement")
    a = ap.parse_args()
    {"prep": prep, "analyze": analyze, "blobs": blobs, "render": render, "slices": slices, "enhance": enhance}[a.stage](a)
