#!/usr/bin/env python3
"""complex_gauge.py -- head-only PSNR of every arm with a complex volume, on the strict complex gauge |rec - GT|
next to the magnitude gauge | |rec| - |GT| | used by every table and figure (the AF16 line's caveat, measured here
for AF12). Head mask and PSNR convention as fig_three_planes_compact.py: mask = |GT| > 0.05 * P99.9(|GT|), PSNR at
max|GT| over the head voxels. Also the magnitude-weighted phase agreement cos(angle(rec) - angle(GT)) inside the head.
Writes logs/complex_gauge.json and prints the table."""
import json, re
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]
rows = {(r["contrast"], r["method"]): r for r in json.load(open(R / "logs/summary.json"))}
out = {}
for c in ("T1", "FLAIR"):
    gt = np.load(R / f"recon/gt_{c}_complex.npy")
    ag = np.abs(gt); gmax = float(ag.max()); head = ag > 0.05 * float(np.quantile(ag, 0.999))
    arms = {"cgsense": f"recon/cgsense_{c}_complex.npy", "cssense_o200": None, "jcssense": None, "l1espirit": None,
            "zsssdu": f"recon/zsssdu_{c}_complex.npy", "dip": f"recon/dip_{c}_complex.npy", "inr": f"recon/inr_{c}_complex.npy",
            "ploraks": f"recon/ploraks_{c}_complex.npy", "gs_joint_ar": f"recon/gs_joint_ar_{c}_complex.npy"}
    for key, prefix in (("cssense_o200", "cssense_o200"), ("jcssense", "jcssense_o200"), ("l1espirit", "l1espirit")):
        lam = rows[(c, key)].get("lam")
        cands = [p for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy")
                 if abs(float(re.search(r"_lam([0-9.]+)_", p.name).group(1)) - float(lam)) < 1e-12]
        arms[key] = str(cands[0].relative_to(R)) if cands else (f"recon/{prefix}_{c}_complex.npy")
    res = {}
    for key, rel in arms.items():
        p = R / rel
        if not p.exists():
            continue
        x = np.load(p)
        e_c = np.abs(x - gt)[head]; e_m = np.abs(np.abs(x) - ag)[head]
        psnr = lambda e: float(20 * np.log10(gmax / np.sqrt(np.mean(e.astype(np.float64) ** 2))))
        w = ag[head]; cosph = float(np.sum(w * np.cos(np.angle(x[head]) - np.angle(gt[head]))) / np.sum(w))
        res[key] = dict(head_psnr_mag=psnr(e_m), head_psnr_cplx=psnr(e_c), cos_phase=cosph, file=rel)
    out[c] = res
    print(f"\n{c}: {'arm':14s} {'mag':>8s} {'complex':>8s} {'drop':>6s} {'cos_phase':>9s}")
    for key, r in sorted(res.items(), key=lambda kv: -kv[1]["head_psnr_cplx"]):
        print(f"    {key:14s} {r['head_psnr_mag']:8.3f} {r['head_psnr_cplx']:8.3f} {r['head_psnr_mag']-r['head_psnr_cplx']:6.3f} {r['cos_phase']:9.4f}")
json.dump(out, open(R / "logs/complex_gauge.json", "w"), indent=1)
