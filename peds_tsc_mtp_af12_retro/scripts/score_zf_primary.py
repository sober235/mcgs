#!/usr/bin/env python3
"""score_zf_primary.py -- ZF scored on the frozen gauge in the primary region (head inside partitions 12..92), the one arm
scripts/rescore.py does not enumerate. Before writing, the gauge is cross-checked on CG-SENSE against logs/rescore.json
(the recorded primary row must be reproduced). Output: logs/rescore_zf_primary.json, read by scripts/plot_af12_methods.py.
Run: CUDA_VISIBLE_DEVICES= ./scripts/py scripts/score_zf_primary.py   (CPU, a few minutes)
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_peds_wt")
from uih_mtp.rescore import partition_region, score_region

R = Path(__file__).resolve().parents[1]
LO, HI = 12, 92
ref = json.load(open(R / "logs/rescore.json"))
assert ref["lo"] == LO and ref["hi"] == HI, (ref["lo"], ref["hi"])
out = {}
for c in ("T1w", "T2s"):
    gt = np.load(R / f"recon/gt_{c}_complex.npy")
    region = partition_region(gt.shape, 2, LO, HI)
    chk = score_region(np.load(R / f"recon/cgsense_{c}_complex.npy"), gt, region)
    worst = max(abs(chk[m] - ref["rows"][f"cgsense|{c}"][m]) for m in ("head_psnr", "head_ssim", "head_hfen"))
    assert worst < 1e-6, f"gauge mismatch on CG-SENSE {c}: {worst:.2e}"
    row = score_region(np.load(R / f"recon/zf_{c}_complex.npy"), gt, region)
    row.update(method="zf", label="ZF", contrast=c, lam=None, volume=f"zf_{c}_complex.npy")
    out[f"zf|{c}"] = row
    print(f"{c} gauge check on CG-SENSE: max |diff| {worst:.1e}  ->  ZF head_psnr {row['head_psnr']:.3f} "
          f"head_ssim {row['head_ssim']:.4f} head_hfen {row['head_hfen']:.4f}", flush=True)
p = R / "logs/rescore_zf_primary.json"
json.dump(dict(region=ref["region"], lo=LO, hi=HI, rows=out), open(p, "w"), indent=2)
print("wrote", p)
