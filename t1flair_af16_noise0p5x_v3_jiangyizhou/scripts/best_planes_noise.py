#!/usr/bin/env python3
"""best_planes_noise.py -- the slices on which MC-GS leads every comparison column of the compact figure by the most.

Columns = fig_three_planes_compact.py with --with_zf --l1_arm spirit (ZF, CG-SENSE, CS-SENSE at its selected lambda,
Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR, GS-single) against MC-GS (gs_joint_p3). Same gauge as the
panels: head mask |GT| > 0.05 * P99.9 of the volume; PSNR at the volume's max|GT| with the MSE over the head pixels of the
slice; SSIM = 2-D map (7x7 uniform window, sample covariance, reflect border, data range max|GT|) averaged over the head
pixels. NMSE ranks like PSNR within a slice, so "first on PSNR and SSIM" means all three panel numbers bold.
Eligible: slices whose head area is at least --min_area of the largest along that axis (per contrast; both contrasts for
the paired pick). For every axis: the slices where MC-GS is first on PSNR and SSIM, ranked by its PSNR lead over the best
other column; and the paired pick, first on both contrasts, ranked by the smaller of the two leads.
Read-only on the volumes; writes logs/best_planes_noise.json.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

ap = argparse.ArgumentParser()
ap.add_argument("--min_area", type=float, default=0.5)
ap.add_argument("--gap", type=int, default=6, help="minimum distance between listed candidates")
ap.add_argument("--top", type=int, default=5)
ap.add_argument("--check", type=int, nargs=3, default=None, metavar=("TRA", "COR", "SAG"),
                help="print the panel numbers of these slices (to compare with a rendered figure)")
args = ap.parse_args()

R = Path(__file__).resolve().parents[1]
summary = json.load(open(R / "logs/summary.json"))
ANAT = ("transverse", "coronal", "sagittal")
MC = "MC-GS"


def columns(c):
    rows = {r["method"]: r for r in summary if r["contrast"] == c}

    def lam_file(key, prefix):
        lam = rows[key]["lam"]
        for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
            m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
            if m and abs(float(m.group(1)) - float(lam)) < 1e-12:
                return p
        raise FileNotFoundError(f"{prefix} {c} lam={lam}")

    sp = rows["spirit"]
    return [("ZF", R / f"recon/zf_{c}_complex.npy"),
            ("CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"),
            ("CS-SENSE", lam_file("cssense_o200", "cssense_o200")),
            ("Joint SENSE", lam_file("jcssense", "jcssense_o200")),
            ("L1-SPIRiT", R / f"recon/spirit_ks{sp['ksize']}_lam{sp['lam']:g}_{c}_complex.npy"),
            ("ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
            ("3D-DIP", R / f"recon/dip_{c}_complex.npy"),
            ("3D-INR", R / f"recon/inr_{c}_complex.npy"),
            ("MC-INR", R / f"recon/mcinr_{c}_complex.npy"),
            ("GS-single", R / "recon/gs_single_p3_{}_mag.npy".format(c)),
            (MC, R / "recon/gs_joint_p3_{}_mag.npy".format(c))]


def slice_ssim(x, y, axis, head, rng):
    """per-slice 2-D SSIM map averaged over the head pixels, all slices along `axis` at once"""
    size = [7, 7, 7]
    size[axis] = 1
    f = lambda a: uniform_filter(a, size=size, mode="reflect")
    cov = 49.0 / 48.0
    ux, uy = f(x), f(y)
    vx = cov * (f(x * x) - ux * ux)
    vy = cov * (f(y * y) - uy * uy)
    vxy = cov * (f(x * y) - ux * uy)
    c1, c2 = (0.01 * rng) ** 2, (0.03 * rng) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    red = tuple(a for a in range(3) if a != axis)
    return (s * head).sum(axis=red) / np.maximum(head.sum(axis=red), 1)


res = {}
for c in ("T1", "FLAIR"):
    gt32 = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
    gmax = float(gt32.max())
    head = gt32 > 0.05 * float(np.quantile(gt32, 0.999))
    gt = gt32.astype(np.float64)
    res[c] = {}
    for lab, path in columns(c):
        v = np.abs(np.load(path)).astype(np.float32).astype(np.float64)
        d = (v - gt) ** 2 * head
        res[c][lab] = {}
        for axis in range(3):
            red = tuple(a for a in range(3) if a != axis)
            n = head.sum(axis=red).astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                ps = 20 * np.log10(gmax) - 10 * np.log10(d.sum(axis=red) / n)
            res[c][lab][axis] = dict(psnr=ps, ssim=slice_ssim(gt, v, axis, head, gmax), area=n)
        print(f"  scored {c} {lab}", flush=True)


def distinct(order, k):
    keep = []
    for s in order:
        if all(abs(int(s) - q) >= args.gap for q in keep):
            keep.append(int(s))
        if len(keep) == k:
            break
    return keep


out = {"min_area": args.min_area, "gauge": "head-masked slice PSNR at volume max|GT|; 7x7 uniform SSIM over head pixels",
       "per_contrast": {}, "paired": {}}
lead = {}
for c in ("T1", "FLAIR"):
    out["per_contrast"][c] = {}
    for axis in range(3):
        area = res[c][MC][axis]["area"]
        elig = area >= args.min_area * area.max()
        others = [k for k in res[c] if k != MC]
        P = np.stack([res[c][k][axis]["psnr"] for k in others])
        S = np.stack([res[c][k][axis]["ssim"] for k in others])
        mp = res[c][MC][axis]["psnr"] - P.max(axis=0)
        ms = res[c][MC][axis]["ssim"] - S.max(axis=0)
        rp, rs = np.array(others)[P.argmax(axis=0)], np.array(others)[S.argmax(axis=0)]
        lead[(c, axis)] = (mp, ms, rp, rs, elig)
        win = np.flatnonzero(elig & (mp > 0) & (ms > 0))
        cand = distinct(win[np.argsort(-mp[win])], args.top)
        print(f"\n{c} {ANAT[axis]}: eligible {int(elig.sum())}, MC-GS first on PSNR {int((elig & (mp > 0)).sum())}, "
              f"on PSNR and SSIM {win.size}")
        for s in cand:
            print(f"  slice {s:3d}: PSNR {res[c][MC][axis]['psnr'][s]:.3f} lead {mp[s]:+.3f} over {rp[s]:12s}  "
                  f"SSIM {res[c][MC][axis]['ssim'][s]:.4f} lead {ms[s]:+.4f} over {rs[s]}")
        out["per_contrast"][c][ANAT[axis]] = dict(
            n_eligible=int(elig.sum()), n_first_psnr=int((elig & (mp > 0)).sum()), n_first_both=int(win.size),
            candidates=[dict(slice=s, psnr=round(float(res[c][MC][axis]["psnr"][s]), 4), psnr_lead=round(float(mp[s]), 4),
                             psnr_runner=str(rp[s]), ssim=round(float(res[c][MC][axis]["ssim"][s]), 5),
                             ssim_lead=round(float(ms[s]), 5), ssim_runner=str(rs[s])) for s in cand])
for axis in range(3):
    (mp1, ms1, rp1, _, e1), (mp2, ms2, rp2, _, e2) = lead[("T1", axis)], lead[("FLAIR", axis)]
    ok = e1 & e2 & (mp1 > 0) & (ms1 > 0) & (mp2 > 0) & (ms2 > 0)
    win = np.flatnonzero(ok)
    score = np.minimum(mp1, mp2)
    cand = distinct(win[np.argsort(-score[win])], args.top)
    print(f"\npaired {ANAT[axis]}: first on PSNR and SSIM in both contrasts on {win.size} slices")
    for s in cand:
        print(f"  slice {s:3d}: T1 lead {mp1[s]:+.3f} ({rp1[s]}), FLAIR lead {mp2[s]:+.3f} ({rp2[s]}), "
              f"SSIM leads {ms1[s]:+.4f} / {ms2[s]:+.4f}")
    out["paired"][ANAT[axis]] = dict(n_first_both=int(win.size), candidates=[
        dict(slice=s, t1_lead=round(float(mp1[s]), 4), flair_lead=round(float(mp2[s]), 4),
             t1_ssim_lead=round(float(ms1[s]), 5), flair_ssim_lead=round(float(ms2[s]), 5)) for s in cand])
if args.check:
    for c in ("T1", "FLAIR"):
        for axis, s in enumerate(args.check):
            print(f"check {c} {ANAT[axis]} {s}: " + "  ".join(
                f"{k} {res[c][k][axis]['psnr'][s]:.2f}/{res[c][k][axis]['ssim'][s]:.4f}" for k in res[c]))
p = R / "logs/best_planes_noise.json"
json.dump(out, open(p, "w"), indent=1)
print("wrote", p)
