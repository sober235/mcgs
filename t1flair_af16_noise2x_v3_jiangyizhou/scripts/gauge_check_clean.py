#!/usr/bin/env python3
"""gauge_check_clean.py -- spec section 7 item 4: with the measurements pointed at the CLEAN file, this topic's code
reproduces the zero-noise AF16 topic; without the override it fits data/meas_<c>.h5.

Run after
    T1T2_MEAS_<C>=<clean ksp> recon_classical.py --contrast <C> --gpu G --methods cgsense --suffix _gaugecheck_clean
with the same override set. Checks
  1. that recon_classical row against the AF16 topic's recorded cgsense row, and the scale;
  2. the saved check volume re-scored through this topic's Pair against the same row;
  3. without the override, Pair reads data/meas_<C>.h5, those measurements differ from the clean ones, and the
     ground truth and scale do not change.
Then moves logs/classical_<C>_gaugecheck_clean.json to logs/gauge_check/ and deletes the check volume, so nothing from
the check can be taken for a result.
usage: T1T2_MEAS_<C>=<clean ksp> gauge_check_clean.py --contrast <C> --gpu G
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", required=True, choices=["T1", "FLAIR"])
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--tol", type=float, default=2e-4)
a = ap.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
c = a.contrast
ENV = f"T1T2_MEAS_{c}"
if not os.environ.get(ENV):
    sys.exit(f"set {ENV} to the clean k-space for this check")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1t2_common as TC  # noqa: E402

H = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou")
KEYS = ("psnr", "ssim", "nmse", "hfen", "head_psnr", "head_ssim", "head_hfen", "sharp", "grain")
house = json.load(open(H / f"logs/classical_{c}.json"))
ref = next(r for r in house["rows"] if r["method"] == "cgsense")
mine_p = TC.R / f"logs/classical_{c}_gaugecheck_clean.json"
mine = json.load(open(mine_p))
row = next(r for r in mine["rows"] if r["method"] == "cgsense_gaugecheck_clean")
bad = []


def compare(tag, got):
    for k in KEYS:
        d = abs(got[k] - ref[k])
        ok = d <= a.tol * max(1.0, abs(ref[k]))
        if not ok:
            bad.append(f"{tag}:{k}")
        print(f"  {'OK ' if ok else 'BAD'} {tag:15s} {k:10s} house={ref[k]:.6f} here={got[k]:.6f} d={d:.2e}", flush=True)


compare("recon_classical", row)
srel = abs(mine["scale_p999"] - house["scale_p999"]) / house["scale_p999"]
print(f"  scale house={house['scale_p999']:.6e} here={mine['scale_p999']:.6e} rel={srel:.2e}")
if srel > 1e-5:
    bad.append("scale")
if mine.get("measurements") != os.environ[ENV]:
    bad.append("recon_classical did not read the override")

dev = torch.device("cuda:0")
vol_p = TC.R / f"recon/cgsense_gaugecheck_clean_{c}_complex.npy"
P = TC.Pair(c, dev)                                            # override active: the measurements are the clean file
compare("Pair.score", P.score(torch.from_numpy(np.load(vol_p)).to(dev), "cgsense"))
y_clean = P.y_full
os.environ.pop(ENV)
Q = TC.Pair(c, dev)                                            # the default: data/meas_<c>.h5
diff = float((Q.y_full - y_clean).abs().max())
same = bool(torch.allclose(Q.gt_c, P.gt_c, rtol=0, atol=1e-6)) and abs(Q.scale - P.scale) <= 1e-9 * P.scale
print(f"  default measurements {Q.meas_path}: max |noisy - clean| = {diff:.4e}; ground truth and scale unchanged: {same}")
if not (Q.meas_path.endswith(f"data/meas_{c}.h5") and diff > 0 and same):
    bad.append("default measurements")

(TC.R / "logs/gauge_check").mkdir(exist_ok=True)
shutil.move(str(mine_p), str(TC.R / "logs/gauge_check" / mine_p.name))
vol_p.unlink()
if bad:
    sys.exit(f"GAUGE CHECK FAILED ({c}): {bad}")
print(f"GAUGE CHECK OK ({c}): clean measurements reproduce T1FLAIR_AF16_v3_jiangyizhou; the default fits data/meas_{c}.h5")
