#!/usr/bin/env python3
"""best_slice_mcgs.py -- the transverse slice on which MC-GS leads every other method of the row figure by the most.

Methods = the nine of fig_row_final.py (same volumes, same lambdas) and the same gauge: head mask |GT| > 0.05 * P99.9
of the volume, PSNR peak = max|GT| of the volume, MSE over the head voxels of the slice.
Eligible: axis-0 slices whose head area is at least --min_area of the largest slice, in both contrasts (keeps the ends
of the head out, where a handful of voxels decides the number).
margin(slice, contrast) = PSNR(MC-GS) - max over the other eight (GS-single included).
One slice for both contrasts: the eligible slice with the largest min(margin_T1, margin_FLAIR).
Writes logs/best_slice_mcgs.json.
"""
import argparse, json, re
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--min_area", type=float, default=0.5)
ap.add_argument("--ref", type=int, default=168, help="slice to report for comparison")
args = ap.parse_args()

R = Path(__file__).resolve().parents[1]
summary = json.load(open(R / "logs/summary.json"))


def arms(c):
    rows = {r["method"]: r for r in summary if r["contrast"] == c}

    def lam_file(row_key, prefix):
        lam = rows.get(row_key, {}).get("lam")
        for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
            m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
            if lam is not None and m and abs(float(m.group(1)) - float(lam)) < 1e-12:
                return p
        raise FileNotFoundError(f"{prefix} {c} lam={lam}")

    return [("CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"),
            ("CS-SENSE", lam_file("cssense_o200", "cssense_o200")),
            ("Joint SENSE", lam_file("jcssense", "jcssense_o200")),
            ("L1-ESPIRiT", lam_file("l1espirit", "l1espirit")),
            ("ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
            ("3D-DIP", R / f"recon/dip_{c}_complex.npy"),
            ("3D-INR", R / f"recon/inr_{c}_complex.npy"),
            ("GS-single", R / f"recon/gs_single_ar_{c}_mag.npy"),
            ("MC-GS", R / f"recon/gs_joint_ar_{c}_mag.npy")]


psnr, margin, runner, elig = {}, {}, {}, None
for c in ("T1", "FLAIR"):
    gt32 = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
    head = gt32 > 0.05 * float(np.quantile(gt32, 0.999))
    gt = gt32.astype(np.float64)
    n = head.sum(axis=(1, 2)).astype(np.float64)
    ok = n >= args.min_area * n.max()
    elig = ok if elig is None else elig & ok
    peak = 20 * np.log10(gt.max())
    psnr[c] = {}
    for lab, path in arms(c):
        v = np.abs(np.load(path)).astype(np.float64)
        se = (((v - gt) ** 2) * head).sum(axis=(1, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            psnr[c][lab] = peak - 10 * np.log10(se / n)
    others = [k for k in psnr[c] if k != "MC-GS"]
    stack = np.stack([psnr[c][k] for k in others])
    margin[c] = psnr[c]["MC-GS"] - stack.max(axis=0)
    runner[c] = np.array(others)[stack.argmax(axis=0)]

idx = np.flatnonzero(elig)
score = np.minimum(margin["T1"], margin["FLAIR"])
best = int(idx[np.argmax(score[idx])])
print(f"eligible slices (head area >= {args.min_area:g} of max, both contrasts): {idx.min()}..{idx.max()}  n={idx.size}")
for c in ("T1", "FLAIR"):
    own = int(idx[np.argmax(margin[c][idx])])
    print(f"{c:5s} MC-GS first on {int((margin[c][idx] > 0).sum())}/{idx.size} eligible slices; "
          f"own best slice {own}: margin {margin[c][own]:+.3f} dB over {runner[c][own]}")
print("top 8 slices for both contrasts (min of the two margins):")
for s in idx[np.argsort(-score[idx])][:8]:
    print(f"  slice {s}: T1 {margin['T1'][s]:+.3f} over {runner['T1'][s]:11s}  FLAIR {margin['FLAIR'][s]:+.3f} over {runner['FLAIR'][s]}")
for s, tag in ((best, "CHOSEN"), (args.ref, "reference")):
    print(f"{tag} slice {s}:")
    for c in ("T1", "FLAIR"):
        print(f"  {c:5s} " + "  ".join(f"{k} {psnr[c][k][s]:.3f}" for k in psnr[c]))
json.dump({"criterion": "max over eligible axis-0 slices of min(margin_T1, margin_FLAIR); margin = PSNR(MC-GS) - "
                        "max over the other eight methods; head-masked slice PSNR, volume peak",
           "min_area": args.min_area, "eligible": idx.tolist(), "best_slice": best,
           "psnr": {c: {k: np.round(v, 4).tolist() for k, v in psnr[c].items()} for c in psnr},
           "margin": {c: np.round(margin[c], 4).tolist() for c in margin},
           "runner_up": {c: runner[c].tolist() for c in runner}},
          open(R / "logs/best_slice_mcgs.json", "w"))
print("wrote", R / "logs/best_slice_mcgs.json")
