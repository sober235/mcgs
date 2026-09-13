#!/usr/bin/env python3
"""bar_complex_gauge.py -- head-only PSNR of every arm under BOTH gauges, and the competitor bar.

magnitude gauge: | |recon| - |GT| |, the one the table and every figure use.
complex gauge:   |recon - GT| on the complex volumes (needs recon/<arm>_<c>_complex.npy; the GS arms
                 got theirs from arc/cycle.py --from_ckpt --save_volumes).
Also reports the magnitude-weighted phase agreement inside the head. Writes logs/bar_complex_gauge.json:
per-arm numbers plus `bar`, the best of the seven comparison arms per gauge and contrast.
"""
import json, re
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
S = json.load(open(R / "logs/summary.json"))
COMPARISON = ["CG-SENSE", "CS-SENSE", "Joint SENSE", "L1-ESPIRiT", "ZS-SSDU", "3D-DIP", "3D-INR"]


def arms(c, rows):
    def lam(row_key, prefix):
        v = rows.get(row_key, {}).get("lam")
        for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
            m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
            if m and abs(float(m.group(1)) - float(v)) < 1e-12:
                return p
        raise FileNotFoundError(f"{prefix} {c}")
    return [("CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"),
            ("CS-SENSE", lam("cssense_o200", "cssense_o200")),
            ("Joint SENSE", lam("jcssense", "jcssense_o200")),
            ("L1-ESPIRiT", lam("l1espirit", "l1espirit")),
            ("ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
            ("3D-DIP", R / f"recon/dip_{c}_complex.npy"),
            ("3D-INR", R / f"recon/inr_{c}_complex.npy"),
            ("GS-single", R / f"recon/gs_single_ar_{c}_complex.npy"),
            ("MC-GS", R / f"recon/gs_joint_ar_{c}_complex.npy")]


out = {"gauges": {"magnitude": "| |recon| - |GT| | over the head mask, peak = max|GT|",
                  "complex": "|recon - GT| on the complex volumes, same mask and peak"},
       "head_mask": "|GT| > 0.05 * P99.9(|GT|)", "arms": {}, "bar": {}}
for c in ("T1", "FLAIR"):
    rows = {r["method"]: r for r in S if r["contrast"] == c}
    gtc = np.load(R / f"recon/gt_{c}_complex.npy")
    gt = np.abs(gtc).astype(np.float64)
    head = gt > 0.05 * np.quantile(gt, 0.999)
    peak = float(gt.max()); gh = gt[head]; gch = gtc[head].astype(np.complex128)
    psnr = lambda e: 20 * np.log10(peak) - 10 * np.log10(float((np.abs(e) ** 2).mean()))
    out["arms"][c] = {}
    for lab, p in arms(c, rows):
        vh = np.load(p)[head].astype(np.complex128)
        w = np.abs(vh) * gh
        out["arms"][c][lab] = dict(
            head_psnr=round(psnr(np.abs(vh) - gh), 4),
            head_psnr_cplx=round(psnr(vh - gch), 4),
            cos_phase=round(float((w * np.cos(np.angle(vh) - np.angle(gch))).sum() / w.sum()), 5))
    for key in ("head_psnr", "head_psnr_cplx"):
        best = max(COMPARISON, key=lambda a: out["arms"][c][a][key])
        out["bar"].setdefault(key, {})[c] = {"arm": best, "value": out["arms"][c][best][key]}
json.dump(out, open(R / "logs/bar_complex_gauge.json", "w"), indent=1)
print(json.dumps(out["bar"], indent=1))
for c in ("T1", "FLAIR"):
    print(c, {a: (v["head_psnr"], v["head_psnr_cplx"]) for a, v in out["arms"][c].items()})
