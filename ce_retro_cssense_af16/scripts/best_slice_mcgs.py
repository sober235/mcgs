#!/usr/bin/env python3
"""best_slice_mcgs.py -- the slices (transverse; --axis 1 coronal, --axis 2 sagittal) on which MC-GS leads every other method of the row figure by the most.

Methods = the nine of fig_row_final.py (same volumes, same lambdas) and the same gauge: head mask |GT| > 0.05 * P99.9
of the volume, PSNR peak = max|GT| of the volume, MSE over the head voxels of the slice.
Eligible: slices along --axis whose head area is at least --min_area of the largest slice, in both contrasts (keeps the ends
of the head out, where a handful of voxels decides the number).
margin(slice, contrast) = PSNR(MC-GS) - max over the other eight (GS-single included).
One slice for both contrasts: the eligible slice with the largest min(margin_T1, margin_FLAIR).
Also lists distinct candidates (at least --gap slices apart): by that score, and, among the slices where MC-GS is first
on both contrasts, by its lead over the best method outside the GS family (GS-single left out).
Writes logs/best_slice_mcgs.json (axis 0) or logs/best_slice_mcgs_axis<k>.json.

Review note (2026-09-17): arms(c) below covers 8 of this topic's 11 registered arms -- no zf, L1-SPIRiT (spirit)
or MC-INR row. Inherited from the jyz source (Step 7 only asked to drop its L1-ESPIRiT row); left as is.
"""
import argparse, json, re
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--min_area", type=float, default=0.5)
ap.add_argument("--axis", type=int, default=0, choices=[0, 1, 2], help="0 transverse, 1 coronal, 2 sagittal")
ap.add_argument("--ref", type=int, default=None, help="slice to report for comparison (168 on axis 0, else the middle)")
ap.add_argument("--gap", type=int, default=6, help="minimum distance between listed candidates")
ap.add_argument("--top", type=int, default=8)
ap.add_argument("--mcgs_key", default="gs_joint", help="registered joint arm to score as MC-GS")
ap.add_argument("--single_key", default="gs_single", help="registered single arm to score as GS-single")
ap.add_argument("--tag", default="", help="suffix for the output json, e.g. _p3")
args = ap.parse_args()
red = tuple(a for a in range(3) if a != args.axis)          # the two in-plane axes

R = Path(__import__("os").environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
out = R / (f"logs/best_slice_mcgs{args.tag}.json" if args.axis == 0 else f"logs/best_slice_mcgs_axis{args.axis}{args.tag}.json")
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
            ("ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
            ("3D-DIP", R / f"recon/dip_{c}_complex.npy"),
            ("3D-INR", R / f"recon/inr_{c}_complex.npy"),
            ("GS-single", R / f"recon/{args.single_key}_{c}_mag.npy"),
            ("MC-GS", R / f"recon/{args.mcgs_key}_{c}_mag.npy")]


psnr, margin, runner, lead, leader, elig = {}, {}, {}, {}, {}, None
for c in ("pre", "post"):
    gt32 = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
    head = gt32 > 0.05 * float(np.quantile(gt32, 0.999))
    gt = gt32.astype(np.float64)
    n = head.sum(axis=red).astype(np.float64)
    ok = n >= args.min_area * n.max()
    elig = ok if elig is None else elig & ok
    peak = 20 * np.log10(gt.max())
    psnr[c] = {}
    for lab, path in arms(c):
        v = np.abs(np.load(path)).astype(np.float64)
        se = (((v - gt) ** 2) * head).sum(axis=red)
        with np.errstate(divide="ignore", invalid="ignore"):
            psnr[c][lab] = peak - 10 * np.log10(se / n)
    others = [k for k in psnr[c] if k != "MC-GS"]
    stack = np.stack([psnr[c][k] for k in others])
    margin[c] = psnr[c]["MC-GS"] - stack.max(axis=0)
    runner[c] = np.array(others)[stack.argmax(axis=0)]
    rest = [k for k in others if k != "GS-single"]
    stack = np.stack([psnr[c][k] for k in rest])
    lead[c] = psnr[c]["MC-GS"] - stack.max(axis=0)
    leader[c] = np.array(rest)[stack.argmax(axis=0)]

idx = np.flatnonzero(elig)
score = np.minimum(margin["pre"], margin["post"])
best = int(idx[np.argmax(score[idx])])
both = idx[(margin["pre"][idx] > 0) & (margin["post"][idx] > 0)]
score_b = np.minimum(lead["pre"], lead["post"])


def distinct(sc, pool):
    keep = []
    for s in pool[np.argsort(-sc[pool])]:
        if all(abs(int(s) - k) >= args.gap for k in keep):
            keep.append(int(s))
        if len(keep) == args.top:
            break
    return keep


def line(s):
    return (f"  slice {s:3d}: over all others pre {margin['pre'][s]:+.3f} [{runner['pre'][s]}] post {margin['post'][s]:+.3f} "
            f"[{runner['post'][s]}] | over best non-GS pre {lead['pre'][s]:+.3f} [{leader['pre'][s]}] "
            f"post {lead['post'][s]:+.3f} [{leader['post'][s]}]")


print(f"axis {args.axis}: eligible slices (head area >= {args.min_area:g} of max, both contrasts): {idx.min()}..{idx.max()}  n={idx.size}")
for c in ("pre", "post"):
    own = int(idx[np.argmax(margin[c][idx])])
    print(f"{c:5s} MC-GS first on {int((margin[c][idx] > 0).sum())}/{idx.size} eligible slices; "
          f"own best slice {own}: margin {margin[c][own]:+.3f} dB over {runner[c][own]}")
print(f"MC-GS first on both contrasts on {both.size}/{idx.size} eligible slices")
cand_a, cand_b = distinct(score, idx), distinct(score_b, both)
print(f"distinct candidates (>= {args.gap} apart) by the smaller margin over all others:")
for s in cand_a:
    print(line(s))
print(f"distinct candidates among the both-first slices by the smaller lead over the best non-GS method:")
for s in cand_b:
    print(line(s))
ref = args.ref if args.ref is not None else (168 if args.axis == 0 else len(elig) // 2)
for s, tag in ((best, "CHOSEN"), (ref, "reference")):
    print(f"{tag} slice {s}:")
    for c in ("pre", "post"):
        print(f"  {c:5s} " + "  ".join(f"{k} {psnr[c][k][s]:.3f}" for k in psnr[c]))
json.dump({"axis": args.axis, "criterion": "max over eligible slices along the axis of min(margin_T1, margin_FLAIR); margin = PSNR(MC-GS) - "
                        "max over the other eight methods; head-masked slice PSNR, volume peak",
           "min_area": args.min_area, "eligible": idx.tolist(), "best_slice": best, "both_first": both.tolist(),
           "gap": args.gap, "candidates_all": cand_a, "candidates_non_gs": cand_b,
           "psnr": {c: {k: np.round(v, 4).tolist() for k, v in psnr[c].items()} for c in psnr},
           "margin": {c: np.round(margin[c], 4).tolist() for c in margin},
           "runner_up": {c: runner[c].tolist() for c in runner},
           "lead_non_gs": {c: np.round(lead[c], 4).tolist() for c in lead},
           "leader_non_gs": {c: leader[c].tolist() for c in leader}},
          open(out, "w"))
print("wrote", out)
