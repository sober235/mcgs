#!/usr/bin/env python3
"""rescore.py -- the primary table: every arm rescored on the frozen gauge, averaged over the head
inside the vendor's central 80 partitions (SPE 12..92). The secondary table is each arm's own JSON
over all 104 partitions; --full reproduces it here as the cross-check.
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_peds_wt")
from uih_mtp.rescore import partition_region, score_region

ap = argparse.ArgumentParser()
ap.add_argument("--lo", type=int, default=12)
ap.add_argument("--hi", type=int, default=92)
ap.add_argument("--full", action="store_true", help="score all 104 partitions (the cross-check)")
ap.add_argument("--out", default=None)
ap.add_argument("--jobs", type=int, default=10,
                help="arms scored in parallel; the 15^3 LoG convolution is single-threaded and dominates")
args = ap.parse_args()

R = Path(__file__).resolve().parents[1]
CONTRASTS = ("T1w", "T2s")
summary = json.load(open(R / "logs/summary.json"))


def lam_file(c, row_key, prefix):
    lam = next((r.get("lam") for r in summary if r["contrast"] == c and r["method"] == row_key), None)
    if lam is None:
        return None, None
    for p in sorted(R.glob(f"recon/{prefix}_lam*_{c}_complex.npy")):
        m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
        if m and abs(float(m.group(1)) - float(lam)) < 1e-12:
            return p, float(lam)
    return None, float(lam)


def arms(c):
    a = [("cgsense", "CG-SENSE", R / f"recon/cgsense_{c}_complex.npy", None)]
    for key, label, prefix in (("cssense_o200", "CS-SENSE", "cssense_o200"),
                               ("jcssense", "Joint SENSE", "jcssense_o200"),
                               ("l1espirit", "L1-ESPIRiT", "l1espirit")):
        p, lam = lam_file(c, key, prefix)
        a.append((key, label, p, lam))
    a += [("zsssdu", "ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy", None),
          ("dip", "3D-DIP", R / f"recon/dip_{c}_complex.npy", None),
          ("inr", "3D-INR", R / f"recon/inr_{c}_complex.npy", None),
          ("mcinr", "MC-INR (Niessen 2025)", R / f"recon/mcinr_{c}_complex.npy", None),
          ("gs_single_ar", "GS-single", R / f"recon/gs_single_ar_{c}_complex.npy", None),
          ("gs_joint_ar", "MC-GS", R / f"recon/gs_joint_ar_{c}_complex.npy", None)]
    # stage 3 (tuned on this subject): the joint arm on both contrasts, the single arm on T1w only --
    # T2s's own weights were not changed by the tuning, so gs_single_ar is already its tuned single
    for key, label in (("gs_joint_p3", "MC-GS (tuned)"), ("gs_single_p3", "GS-single (tuned)")):
        v = R / f"recon/{key}_{c}_complex.npy"
        if v.exists():
            a.append((key, label, v, None))
    return a


def score_one(job):
    """One (arm, contrast). Loads its own volumes so the work parallelises across processes: the
    frozen LoG convolution and the two SSIM maps are single-threaded and dominate the run time."""
    c, key, label, path, lam = job
    gt = np.load(R / f"recon/gt_{c}_complex.npy")
    region = None if args.full else partition_region(gt.shape, 2, args.lo, args.hi)
    row = score_region(np.load(path), gt, region)
    row.update(method=key, label=label, contrast=c, lam=lam, volume=str(Path(path).name))
    return f"{key}|{c}", row


jobs = []
for c in CONTRASTS:
    for key, label, path, lam in arms(c):
        assert path is not None and Path(path).exists(), f"missing volume for {label} {c}: {path}"
        jobs.append((c, key, label, path, lam))

out = {}
if args.jobs > 1:
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for k, row in ex.map(score_one, jobs):
            out[k] = row
            print(f"{row['contrast']} {row['label']:12s} head_psnr {row['head_psnr']:7.3f} "
                  f"ssim {row['head_ssim']:.4f} hfen {row['head_hfen']:.4f} psnr {row['psnr']:7.3f} "
                  f"nmse {row['nmse']:.5f} sharp {row['sharp']:.3f}", flush=True)
else:
    for job in jobs:
        k, row = score_one(job)
        out[k] = row
        print(f"{row['contrast']} {row['label']:12s} head_psnr {row['head_psnr']:7.3f} "
              f"ssim {row['head_ssim']:.4f} hfen {row['head_hfen']:.4f} psnr {row['psnr']:7.3f} "
              f"nmse {row['nmse']:.5f} sharp {row['sharp']:.3f}", flush=True)

p = Path(args.out) if args.out else (R / ("logs/rescore_full.json" if args.full else "logs/rescore.json"))
json.dump(dict(region=("full volume" if args.full else f"head inside partitions {args.lo}..{args.hi}"),
               lo=args.lo, hi=args.hi, rows=out), open(p, "w"), indent=2)
print("wrote", p)
