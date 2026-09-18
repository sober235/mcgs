#!/usr/bin/env python3
"""find_best_slice.py -- per-slice head-masked PSNR for every method, every plane, every slice.

Answers two different questions and keeps them apart:
  (a) where does MC-GS joint beat ALL SEVEN other arms?
  (b) where does MC-GS joint beat MC-GS single by the most? -- the apples-to-apples pair
      (same model class, same 600k budget, same loss; only the shared geometry differs)

Per-slice metric is the head-only PSNR of the frozen gauge, restricted to that slice:
    head = |GT| > 0.05 * P99.9(|GT|)     (global mask, frozen definition)
    psnr_i = 20 log10(max|GT|) - 10 log10( mean over head&slice of (rec - gt)^2 )
Slices whose head mask holds < 2000 voxels are skipped: a handful of voxels makes the number
meaningless, not favourable. Writes logs/slice_scores.json.

Review note (2026-09-17): METHODS is flat (not lambda-aware) and does not match this topic's swept CS-SENSE
filenames (recon/cssense_o200_lam<L>_<c>_complex.npy) -- as ported this will KeyError/ValueError once
gs_joint/gs_single/cssense are needed but "cssense" never resolves to a file. Inherited from the jyz source;
generalising the loader is future work, not done here.

Task 16 fix (2026-09-18): all eight methods are now needed, so the predicted KeyError above was hit. "cssense"
is resolved through logs/summary.json's "cssense_o200" row (the lambda summarize.py kept for that scan), the
same lam_file() pattern arms.py/best_slice_mcgs.py already use -- everything else in METHODS is unchanged.
"""
import json
from pathlib import Path

import numpy as np

R = Path(__import__("os").environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
METHODS = ["zf", "cgsense", "cssense", "zsssdu", "dip", "inr", "gs_single", "gs_joint"]
MIN_HEAD = 2000
SUMMARY = {(r["contrast"], r["method"]): r for r in json.load(open(R / "logs/summary.json"))}


def load(method, c):
    if method == "cssense":
        row = SUMMARY.get((c, "cssense_o200"))
        if row is None or row.get("lam") is None:
            return None
        p = R / f"recon/cssense_o200_lam{row['lam']:g}_{c}_complex.npy"
        return np.abs(np.load(p)).astype(np.float32) if p.exists() else None
    for suf in ("_complex.npy", "_mag.npy"):
        p = R / f"recon/{method}_{c}{suf}"
        if p.exists():
            return np.abs(np.load(p)).astype(np.float32)
    return None


def main():
    out = {}
    for c in ("pre", "post"):
        gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
        gmax = float(gt.max())
        head = gt > 0.05 * float(np.quantile(gt, 0.999))
        vols = {m: load(m, c) for m in METHODS}
        vols = {m: v for m, v in vols.items() if v is not None}
        rows = []
        for ax in range(3):
            n = gt.shape[ax]
            for i in range(n):
                sl = [slice(None)] * 3; sl[ax] = i; sl = tuple(sl)
                h = head[sl]
                nh = int(h.sum())
                if nh < MIN_HEAD:
                    continue
                g = gt[sl][h].astype(np.float64)
                psnr = {}
                for m, v in vols.items():
                    e = v[sl][h].astype(np.float64) - g
                    mse = float((e * e).mean())
                    psnr[m] = 20 * np.log10(gmax) - 10 * np.log10(mse) if mse > 0 else np.inf
                order = sorted(psnr, key=lambda m: -psnr[m])
                rows.append({"contrast": c, "axis": ax, "index": i, "n_head": nh,
                             "psnr": psnr, "rank_gs_joint": order.index("gs_joint") + 1,
                             "best": order[0],
                             "joint_minus_runnerup": psnr["gs_joint"] - max(
                                 p for m, p in psnr.items() if m != "gs_joint"),
                             "joint_minus_single": psnr["gs_joint"] - psnr["gs_single"],
                             "joint_minus_cssense": psnr["gs_joint"] - psnr["cssense"]})
        out[c] = rows
        won = [r for r in rows if r["rank_gs_joint"] == 1]
        print(f"\n=== {c}: {len(rows)} scored slices, MC-GS joint ranks 1st on {len(won)}")
        from collections import Counter
        print("   winner counts:", dict(Counter(r["best"] for r in rows).most_common()))
        print("   joint rank histogram:", dict(sorted(Counter(r["rank_gs_joint"] for r in rows).items())))
        for key, label in (("joint_minus_runnerup", "vs the best other arm"),
                           ("joint_minus_single", "vs MC-GS single"),
                           ("joint_minus_cssense", "vs CS-SENSE")):
            top = sorted(rows, key=lambda r: -r[key])[:5]
            print(f"   top-5 slices by joint {label}:")
            for r in top:
                print(f"      axis{r['axis']} idx {r['index']:3d}  d={r[key]:+.3f} dB  "
                      f"rank={r['rank_gs_joint']}  best={r['best']}  n_head={r['n_head']}")
    json.dump({c: [{k: (v if k != 'psnr' else {m: float(x) for m, x in v.items()})
                    for k, v in r.items()} for r in rows] for c, rows in out.items()},
              open(R / "logs/slice_scores.json", "w"))
    print(f"\nwrote {R/'logs/slice_scores.json'}")


if __name__ == "__main__":
    main()
