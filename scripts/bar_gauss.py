#!/usr/bin/env python3
"""bar_gauss.py -- the comparison bar of the Gaussian-mask autoresearch loop (autoresearch_gauss.md).

Reads the gauss topic's logs/summary.json (magnitude gauge: head PSNR / SSIM / HFEN / sharpness of every arm) and
the complex volumes it has, and writes logs/bar_gauss.json in the topic: per-arm numbers, and `bar` = the best
COMPARISON arm per cell. Comparison arms = every arm except MC-GS, INCLUDING GS-single at the frozen phase-3
recipe (user decision 2026-09-16: GS-single is a comparison arm and is not re-tuned) and excluding L1-ESPIRiT
(dropped 2026-09-15, not run on this mask). The loop's objective is MC-GS minus this bar, head PSNR, both contrasts.
"""
import json, re
from pathlib import Path
import numpy as np

R = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_gauss_v3_jiangyizhou")
S = json.load(open(R / "logs/summary.json"))
LABEL = {"zf": "ZF", "cgsense": "CG-SENSE", "cssense": "CS-SENSE (base)", "cssense_o200": "CS-SENSE",
         "jcssense": "Joint SENSE", "spirit": "L1-SPIRiT", "zsssdu": "ZS-SSDU", "dip": "3D-DIP", "inr": "3D-INR",
         "mcinr": "MC-INR", "gs_single_p3": "GS-single", "gs_joint_p3": "MC-GS"}
NOT_COMPARISON = {"MC-GS", "ZF"}


def cplx_path(c, key, row):
    lam = row.get("lam")
    if key == "cgsense":
        return R / f"recon/cgsense_{c}_complex.npy"
    if key == "cssense":
        return R / f"recon/cssense_{c}_complex.npy"
    if key == "cssense_o200":
        return next(p for p in R.glob(f"recon/cssense_o200_lam*_{c}_complex.npy")
                    if abs(float(re.search(r"lam([0-9.]+)_", p.name).group(1)) - float(lam)) < 1e-12)
    if key == "jcssense":
        return next(p for p in R.glob(f"recon/jcssense_o200_lam*_{c}_complex.npy")
                    if abs(float(re.search(r"lam([0-9.]+)_", p.name).group(1)) - float(lam)) < 1e-12)
    if key == "spirit":
        ks = row.get("ks", 5)
        return next(p for p in R.glob(f"recon/spirit_ks{ks}_lam*_{c}_complex.npy")
                    if abs(float(re.search(r"lam([0-9.]+)_", p.name).group(1)) - float(lam)) < 1e-12)
    if key in ("zsssdu", "dip", "inr"):
        return R / f"recon/{key}_{c}_complex.npy"
    if key in ("gs_single_p3", "gs_joint_p3"):
        return R / f"recon/{key}_{c}_complex.npy"
    return None


out = {"topic": str(R), "gauges": {"magnitude": "| |recon| - |GT| | over the head mask, peak = max|GT|",
                                   "complex": "|recon - GT| on the complex volumes, same mask and peak"},
       "head_mask": "|GT| > 0.05 * P99.9(|GT|)",
       "comparison_arms": "every arm except MC-GS and ZF; GS-single at the frozen phase-3 recipe counts (user 2026-09-16)",
       "arms": {}, "bar": {}}
for c in ("T1", "FLAIR"):
    rows = {r["method"]: r for r in S if r["contrast"] == c}
    gtc = np.load(R / f"recon/gt_{c}_complex.npy")
    gt = np.abs(gtc).astype(np.float64)
    head = gt > 0.05 * np.quantile(gt, 0.999)
    peak = float(gt.max()); gh = gt[head]; gch = gtc[head].astype(np.complex128)
    psnr = lambda e: 20 * np.log10(peak) - 10 * np.log10(float((np.abs(e) ** 2).mean()))
    out["arms"][c] = {}
    for key, row in rows.items():
        lab = LABEL.get(key, key)
        a = dict(head_psnr=round(row["head_psnr"], 4), head_ssim=round(row["head_ssim"], 4),
                 head_hfen=round(row["head_hfen"], 4), sharp=round(row["sharp"], 3), lam=row.get("lam"))
        p = cplx_path(c, key, row)
        if p is not None and p.exists():
            vh = np.load(p)[head].astype(np.complex128)
            w = np.abs(vh) * gh
            a["head_psnr_cplx"] = round(psnr(vh - gch), 4)
            a["cos_phase"] = round(float((w * np.cos(np.angle(vh) - np.angle(gch))).sum() / w.sum()), 5)
            mag_check = psnr(np.abs(vh) - gh)
            a["head_psnr_recomputed"] = round(mag_check, 4)
        out["arms"][c][lab] = a
    comp = [l for l in out["arms"][c] if l not in NOT_COMPARISON]
    for cell, better in (("head_psnr", max), ("head_ssim", max), ("head_hfen", min), ("sharp", max)):
        best = better(comp, key=lambda l: out["arms"][c][l][cell])
        out["bar"].setdefault(cell, {})[c] = {"arm": best, "value": out["arms"][c][best][cell]}
    compx = [l for l in comp if "head_psnr_cplx" in out["arms"][c][l]]
    best = max(compx, key=lambda l: out["arms"][c][l]["head_psnr_cplx"])
    out["bar"].setdefault("head_psnr_cplx", {})[c] = {"arm": best, "value": out["arms"][c][best]["head_psnr_cplx"]}
json.dump(out, open(R / "logs/bar_gauss.json", "w"), indent=1)
print(json.dumps(out["bar"], indent=1))
for c in ("T1", "FLAIR"):
    for l, a in out["arms"][c].items():
        print(c, f"{l:16s}", a["head_psnr"], a.get("head_psnr_cplx"), a["head_ssim"], a["head_hfen"], a["sharp"],
              "" if "head_psnr_recomputed" not in a else f"(mag recomputed {a['head_psnr_recomputed']})")
