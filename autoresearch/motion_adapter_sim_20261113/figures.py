#!/usr/bin/env python3
"""Figures of the motion study (design 11.4). Runs on amax, CPU only; reads finished runs, writes PNG (+ PDF).

    figures.py --af af12 [--pdf] [--only threeplane_E2 threeplane_E4b lesions motion delta]

  fig_<af>_C0_T2_threeplane    GT, A0, A1, A2, A3, I-T2       no simulated motion: the adapters must not change the image
  fig_<af>_E2_T2_threeplane    GT, A0, A2, O, I-T2            three planes, zoom rows, one grey bar, one error bar
  fig_<af>_E4b_T2_threeplane   GT, A0, A3, O, I-T2            (record format of the plain-scan / enhanced figure)
  fig_<af>_E3_lesions          every inserted lesion, a T1 row and a T2 row, GT / A0 / A2 / O / independent
  fig_<af>_motion_recovery     estimated against true inter-scan pose; segment poses along the echo-train order
  fig_<af>_delta_psnr          normalised T2 delta-PSNR of every condition and arm

Volume axes 0 / 1 / 2 = S-I / A-P / L-R, 1 mm; index 0 of axis 0 is inferior, so the coronal and sagittal planes are
drawn flipped (vertex up). Panel numbers: head = |GT| > 0.05 P99.9 of the volume, PSNR at max|GT|, 7x7 SSIM map
averaged over the head pixels, head NMSE; error = |recon - GT| / max|GT|, jet [0, 0.1], inside the head.
Arms that model intra-scan motion are drawn with only the reference segment's lines inserted (design 11.1).
No clock value appears on any figure.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

import geom
from common import ARMS, CONDITIONS, DATA, NATIVE_SHIFT_MM, RES, ROOT, exp_name

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_opening
from skimage.metrics import structural_similarity

OUT = ROOT / "figures"
ZC, ZLW = "#ffd400", 1.6
PLANES = ((0, "Transverse"), (1, "Coronal"), (2, "Sagittal"))
NAMES = {"A0": "MC-GS", "A1": "MC-GS + rigid", "A2": "MC-GS + rigid + non-rigid", "A3": "MC-GS + rigid + segments",
         "A23": "MC-GS + rigid + non-rigid + segments", "O": "MC-GS, true motion (oracle)", "I-T1": "GS-single",
         "I-T2": "GS-single", "I-T2+S": "GS-single + segments"}
log = lambda *a: print(*a, flush=True)


def D(x, axis):
    """display orientation: transverse as stored (anterior up); coronal and sagittal with the vertex up"""
    return x if axis == 0 else x[::-1]


def load(af, cond, arm, contrast):
    """(reconstruction magnitude, GT magnitude) of one run and contrast, the evaluation volume of design 11.1."""
    exp = exp_name(af, cond, arm)
    with np.load(RES / f"{exp}_volumes.npz") as v:
        if ARMS[arm]["trainer"] == "single":
            rec, gt = v["rec"], v["gt"]
        else:
            rec, gt = (v["recT1"], v["gtT1"]) if contrast == "T1" else (v["recFL"], v["gtFL"])
    canon = RES / f"{exp}_canon.npz"
    if contrast == "T2" and CONDITIONS[cond]["intra"] and (ARMS[arm]["segments"] or ARMS[arm]["oracle"]) and canon.exists():
        with np.load(canon) as z:
            if "rec_ref_dc" in z.files:
                rec = np.abs(z["rec_ref_dc"]).astype(np.float32)
    return rec, gt


def plane_metrics(rec, truth, mask, gm):
    t, v = truth.astype(np.float64), rec.astype(np.float64)
    _, smap = structural_similarity(t, v, data_range=gm, full=True, win_size=7)
    e = v[mask] - t[mask]
    return dict(psnr=float(20 * np.log10(gm) - 10 * np.log10(float((e * e).mean()))), ssim=float(smap[mask].mean()),
                nmse=float((e * e).sum() / (t[mask] ** 2).sum()))


def head_region(m):
    disk = np.hypot(*np.mgrid[-2:3, -2:3]) <= 2
    return binary_dilation(binary_fill_holes(binary_opening(binary_fill_holes(m), structure=disk)), iterations=2)


def save(fig, name, pdf, record=None):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, facecolor=fig.get_facecolor())
    if pdf:
        fig.savefig(OUT / f"{name}.pdf", dpi=600, facecolor=fig.get_facecolor())
    plt.close(fig)
    if record is not None:
        (OUT / f"{name}.json").write_text(json.dumps(record, indent=1))
    art = os.environ.get("MCGS_ARTIFACT_DIR")
    for p in OUT.glob(f"{name}.*"):
        log(f"wrote {p.name} ({p.stat().st_size} bytes)")
        if art and p.suffix != ".pdf":
            shutil.copy(p, Path(art) / p.name)


# ------------------------------------------------------------------------------------------- three planes
def threeplane(af, cond, arms, pdf, contrast="T2", box=48):
    """Record format: per plane an image row and a zoom row (box of the image | box of the error map)."""
    vols = {}
    for arm in arms:
        if not (RES / f"{exp_name(af, cond, arm)}_volumes.npz").exists():
            log(f"threeplane {cond}: {arm} has no volumes, column left out")
            continue
        vols[arm] = load(af, cond, arm, contrast)
    if not vols:
        return
    gt = next(iter(vols.values()))[1]
    for arm, (_, g) in vols.items():
        assert np.allclose(g, gt, atol=1e-6 * float(gt.max())), f"{arm}: a different GT"
    gm = float(gt.max())
    head = gt > 0.05 * np.quantile(gt, 0.999)
    brain = np.load(DATA / (cond if (DATA / cond / "brain.npy").exists() else "C0") / "brain.npy").astype(bool)
    cen = [int(round(c)) for c in np.argwhere(brain).mean(0)]
    hb = box // 2
    cols = ["GT", *vols]
    rows = []
    for axis, pname in PLANES:
        sl = [slice(None)] * 3
        sl[axis] = cen[axis]
        sl = tuple(sl)
        bm = np.zeros(gt.shape, bool)
        bm[cen[0] - hb:cen[0] + hb, cen[1] - hb:cen[1] + hb, cen[2] - hb:cen[2] + hb] = True
        rr, cc = np.nonzero(D(bm[sl], axis))
        zoom = dict(x0=int(cc.min()), y0=int(rr.min()), w=int(np.ptp(cc)) + 1, h=int(np.ptp(rr)) + 1)
        g2, m2 = gt[sl], head[sl]
        num = {a: plane_metrics(vols[a][0][sl], g2, m2, gm) for a in vols}
        best = {m: (min if m == "nmse" else max)(num, key=lambda a: num[a][m]) for m in ("psnr", "ssim", "nmse")}
        rows.append(dict(axis=axis, pname=pname, idx=cen[axis], sl=sl, g2=g2, emask=head_region(m2), num=num, best=best,
                         zoom=zoom, vmax=float(np.quantile(g2, 0.999))))
    pw, gap, zgap, top_in, cbroom = 2.6, 0.025, 0.02, 0.75, 1.1
    half = (pw - zgap) / 2
    hs = [pw * D(r["g2"], r["axis"]).shape[0] / D(r["g2"], r["axis"]).shape[1] for r in rows]
    ehs = [half * r["zoom"]["h"] / r["zoom"]["w"] for r in rows]
    fw = len(cols) * pw + (len(cols) - 1) * gap + cbroom
    fh = top_in + sum(hs) + sum(ehs) + (2 * len(rows) - 1) * gap + 0.12
    fig = plt.figure(figsize=(fw, fh), facecolor="black")

    def axes_at(x0, y0, w, hh):
        a = fig.add_axes([x0 / fw, 1 - (y0 + hh) / fh, w / fw, hh / fh])
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        return a

    def label(a, text, bold=False, line=0, fs=11, hh=1.0):
        t = a.text(0.025, 0.975 - line * fs * 1.28 / 72 / hh, text, transform=a.transAxes, ha="left", va="top",
                   color="white", fontsize=fs, fontweight="bold" if bold else "normal")
        t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])

    crop = lambda img, z: img[z["y0"]:z["y0"] + z["h"], z["x0"]:z["x0"] + z["w"]]
    kw = dict(origin="upper", aspect="auto")
    y, im_g, im_e, first = top_in, None, None, None
    for r, h, eh in zip(rows, hs, ehs):
        z = r["zoom"]
        for j, col in enumerate(cols):
            x = j * (pw + gap)
            img = D(r["g2"] if col == "GT" else vols[col][0][r["sl"]], r["axis"])
            a = axes_at(x, y, pw, h)
            im_g = a.imshow(img / r["vmax"], cmap="gray", vmin=0, vmax=1, **kw)
            a.add_patch(Rectangle((z["x0"] - 0.5, z["y0"] - 0.5), z["w"], z["h"], fill=False, ec=ZC, lw=ZLW))
            if col == "GT":
                label(a, f"{r['pname']} {r['idx']}", bold=True, hh=h)
            else:
                n = r["num"][col]
                for i, (m, fmt) in enumerate((("psnr", "{:.2f} dB"), ("ssim", "{:.4f}"), ("nmse", "{:.5f}"))):
                    label(a, fmt.format(n[m]), bold=r["best"][m] == col, line=i, hh=h)
            if r is rows[0]:
                a.set_title("Reference" if col == "GT" else NAMES[col], color="white", fontsize=13, pad=6)
            za = axes_at(x, y + h + gap, half, eh)
            za.imshow(crop(img, z) / r["vmax"], cmap="gray", vmin=0, vmax=1, **kw)
            if col != "GT":
                err = np.where(r["emask"], np.abs(vols[col][0][r["sl"]] - r["g2"]) / gm, 0.0)
                ea = axes_at(x + half + zgap, y + h + gap, half, eh)
                im_e = ea.imshow(crop(D(err, r["axis"]), z), cmap="jet", vmin=0, vmax=0.1, **kw)
            if first is None:
                first = (y, h, eh)
        y += h + eh + 2 * gap
    x0 = len(cols) * pw + (len(cols) - 1) * gap + 0.08
    for im, yy, hh, lab in ((im_g, first[0], first[1], "intensity / row window"),
                            (im_e, first[0] + first[1] + gap, first[2], "|error| / max GT")):
        ca = fig.add_axes([x0 / fw, 1 - (yy + hh * 0.95) / fh, 0.13 / fw, hh * 0.9 / fh])
        cb = fig.colorbar(im, cax=ca)
        cb.ax.tick_params(colors="white", labelsize=9)
        cb.set_label(lab, color="white", fontsize=9)
    fig.suptitle(f"{cond}, {contrast}W, {af.upper()}: reference pose GT against each model", color="white", fontsize=15,
                 y=1 - 0.12 / fh)
    save(fig, f"fig_{af}_{cond}_{contrast}_threeplane", pdf,
         dict(cond=cond, af=af, contrast=contrast, centre_voxel=cen, box_voxels=box,
              panels={f"{r['pname']} {r['idx']}": r["num"] for r in rows}))


# ------------------------------------------------------------------------------------------- lesions
def lesions(af, pdf, cond="E3", half=20):
    truth = json.loads((DATA / cond / "motion_truth.json").read_text())
    masks = dict(T1=np.load(DATA / cond / "lesion_mask_canon.npy"), T2=np.load(DATA / cond / "lesion_mask_T2.npy"))
    arms = dict(T1=["A0", "A2", "O", "I-T1"], T2=["A0", "A2", "O", "I-T2"])
    vols = {c: {a: load(af, cond, a, c) for a in arms[c] if (RES / f"{exp_name(af, cond, a)}_volumes.npz").exists()}
            for c in ("T1", "T2")}
    if not vols["T2"]:
        return
    sites = truth["lesion_sites"]
    ncol, nrow, pw = 5, 2 * len(sites), 1.9
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * pw + 0.2, nrow * pw + 0.5), facecolor="black")
    for k, site in enumerate(sites, start=1):
        for ci, c in enumerate(("T1", "T2")):
            row = 2 * (k - 1) + ci
            cen = [int(round(x)) for x in np.argwhere(masks[c] == k).mean(0)]
            win = (cen[0], slice(cen[1] - half, cen[1] + half), slice(cen[2] - half, cen[2] + half))
            some = next(iter(vols[c].values()), None)
            for j, col in enumerate(["GT", *arms[c]]):
                a = axs[row, j]
                a.set_xticks([]); a.set_yticks([]); a.set_facecolor("black")
                if some is None or (col != "GT" and col not in vols[c]):
                    continue
                img = some[1][win] if col == "GT" else vols[c][col][0][win]
                a.imshow(img, cmap="gray", vmin=0, vmax=float(np.quantile(some[1][win], 0.999)) * 1.05, origin="upper")
                a.contour((masks[c][win] == k).astype(float), levels=[0.5], colors=[ZC], linewidths=0.7)
                if row == 0:
                    a.set_title("Reference" if col == "GT" else NAMES[col].replace(" + rigid + non-rigid", " + adapter")
                                .replace(", true motion (oracle)", " oracle"), color="white", fontsize=9)
                if j == 0:
                    tag = "inserted" if site["contrast"] == c else "must stay unchanged"
                    a.set_ylabel(f"L{k} {site['diameter_mm']:.0f} mm\n{c}: {tag}", color="white", fontsize=8)
    fig.suptitle(f"{cond}, {af.upper()}: lesions present in one contrast only (transverse plane through each centre, "
                 f"{2 * half} mm window; outline = lesion mask)", color="white", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.975), pad=0.3)
    save(fig, f"fig_{af}_{cond}_lesions", pdf)


# ------------------------------------------------------------------------------------------- motion
def _state(exp):
    p = RES / f"{exp}_motion.json"
    return json.loads(p.read_text())["final"] if p.exists() else None


def motion(af, pdf):
    inter = [(c, a) for c, a in (("E1a", "A1"), ("E1a", "A2"), ("E1b", "A1"), ("E2", "A2"), ("E3", "A2"), ("E4c", "A23"))
             if _state(exp_name(af, c, a))]
    intra = [(c, a) for c, a in (("E4a", "A3"), ("E4b", "A3"), ("E4c", "A23")) if _state(exp_name(af, c, a))]
    if not inter and not intra:
        return
    n = max(1, len(intra))
    fig = plt.figure(figsize=(13, 3.6 + 3.0 * n))
    gs = fig.add_gridspec(1 + n, 2, height_ratios=[1.15] + [1] * n)
    labels = ["turn", "tilt", "nod", "S-I", "A-P", "L-R"]
    for half, (title, unit, sl) in enumerate((("inter-scan rotation", "degrees", slice(0, 3)),
                                              ("inter-scan translation at the FOV centre", "mm", slice(3, 6)))):
        ax = fig.add_subplot(gs[0, half])
        w = 0.8 / max(1, len(inter))
        for i, (c, a) in enumerate(inter):
            st = _state(exp_name(af, c, a))
            tr = json.loads((DATA / c / "motion_truth.json").read_text())
            g = geom.Rigid.from_json(tr["inter"])
            if CONDITIONS[c]["intra"]:
                g = geom.Rigid.from_json(tr["segment_mean_poses"][tr["ref_seg"]]).compose(g)
            true = np.r_[np.rad2deg(geom.axis_angle(g.R)), g.t + g.R @ np.asarray(NATIVE_SHIFT_MM)]
            est = np.r_[st["theta_deg"], st["t_mm"]]
            x = np.arange(3) + (i - (len(inter) - 1) / 2) * w
            ax.bar(x, true[sl], w * 0.9, color=f"C{i}", alpha=0.35, label=f"{c} truth")
            ax.plot(x, est[sl], "o", color=f"C{i}", ms=6, mec="black", label=f"{c} {a} estimate")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(np.arange(3)); ax.set_xticklabels(labels[sl]); ax.set_ylabel(unit); ax.set_title(title)
        if half == 0:
            ax.legend(fontsize=7, ncol=2)
    for i, (c, a) in enumerate(intra):
        st = _state(exp_name(af, c, a))
        tr = json.loads((DATA / c / "motion_truth.json").read_text())
        ref_inv = geom.Rigid.from_json(tr["segment_mean_poses"][tr["ref_seg"]]).inverse()
        rel = [geom.Rigid.from_json(p).compose(ref_inv) for p in tr["train_poses"]]
        seg = np.asarray(tr["seg_of_train"])
        trains = np.arange(len(rel))
        for half, (unit, pick_true, pick_est, names) in enumerate((
                ("degrees", lambda p: np.rad2deg(geom.axis_angle(p.R)), lambda s: s["theta_deg"], labels[:3]),
                ("mm", lambda p: p.t, lambda s: s["t_mm"], labels[3:]))):
            ax = fig.add_subplot(gs[1 + i, half])
            true = np.stack([pick_true(p) for p in rel])
            est = np.stack([pick_est(st["segments"][k]) for k in seg])
            for d in range(3):
                ax.plot(trains, true[:, d], color=f"C{d}", lw=1.2, label=f"{names[d]} truth")
                ax.step(trains, est[:, d], where="mid", color=f"C{d}", lw=1.6, ls="--", label=f"{names[d]} estimate")
            ax.axvspan(*np.flatnonzero(seg == tr["ref_seg"])[[0, -1]], color="0.85", zorder=0)
            ax.set_xlabel("echo train (acquisition order)"); ax.set_ylabel(unit)
            ax.set_title(f"{c} {a}: pose within the T2W scan, relative to the reference segment (grey)")
            if i == 0:
                ax.legend(fontsize=7, ncol=3)
    fig.suptitle(f"Recovered motion, {af.upper()} (truth = simulated motion; translation truth includes the native "
                 f"T1-T2 offset estimated from the images)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, f"fig_{af}_motion_recovery", pdf)


# ------------------------------------------------------------------------------------------- delta PSNR
def delta(af, pdf):
    p = ROOT / "eval/summary.json"
    if not p.exists():
        log("delta: no summary.json (run evaluate.py --all first)")
        return
    s = json.loads(p.read_text()).get(af)
    if not s or not s["delta"]:
        return
    conds = [c for c in ("E1a", "E1b", "E2", "E3", "E4a", "E4b", "E4c") if c in s["delta"]]
    arms = [a for a in ("A0", "A1", "A2", "A3", "A23", "O") if any(a in s["delta"][c] for c in conds)]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), sharey=False)
    for ax, c_ in zip(axs, ("T2", "T1")):
        w = 0.8 / len(arms)
        for i, a in enumerate(arms):
            xs = [j + (i - (len(arms) - 1) / 2) * w for j, c in enumerate(conds) if s["delta"][c].get(a, {}).get(c_) is not None]
            ys = [s["delta"][c][a][c_] for c in conds if s["delta"][c].get(a, {}).get(c_) is not None]
            ax.bar(xs, ys, w * 0.92, label=NAMES[a], color=f"C{i}")
        ax.axhline(0, color="black", lw=0.6)
        ax.axhline(-0.1 if c_ == "T2" else -0.05, color="red", lw=0.8, ls=":")
        ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds)
        ax.set_ylabel("dB")
        ax.set_title(f"{c_}: PSNR(arm, condition) - PSNR(arm, C0)"
                     + (", minus the same difference of GS-single" if c_ == "T2" else " (T1 data unchanged except E3)"))
    axs[0].legend(fontsize=7)
    fig.suptitle(f"Head PSNR change caused by the simulated motion, {af.upper()} (dotted: criterion S2)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, f"fig_{af}_delta_psnr", pdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--af", default="af12")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    jobs = dict(threeplane_C0=lambda: threeplane(args.af, "C0", ["A0", "A1", "A2", "A3", "I-T2"], args.pdf),
                threeplane_E2=lambda: threeplane(args.af, "E2", ["A0", "A2", "O", "I-T2"], args.pdf),
                threeplane_E4b=lambda: threeplane(args.af, "E4b", ["A0", "A3", "O", "I-T2"], args.pdf),
                lesions=lambda: lesions(args.af, args.pdf), motion=lambda: motion(args.af, args.pdf),
                delta=lambda: delta(args.af, args.pdf))
    for name in (args.only or jobs):
        log(f"== {name}")
        jobs[name]()


if __name__ == "__main__":
    main()
