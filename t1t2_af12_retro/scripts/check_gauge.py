#!/usr/bin/env python3
"""check_gauge.py -- prove t1t2_common reproduces recon_classical's numbers.

Re-scores an already-exported classical volume through the shared module and compares against
the value recorded in logs/classical_<contrast>.json. Any drift means the two paths disagree and
the learned-method numbers would not be comparable with the classical ones.
"""
import argparse, json, os, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", default="T1", choices=["T1", "T2"])
ap.add_argument("--method", default="cgsense")
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--tol", type=float, default=2e-4)
args = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1t2_common import Pair, R

dev = torch.device("cuda:0")
p = Pair(args.contrast, dev)
x = torch.from_numpy(np.load(R / f"recon/{args.method}_{args.contrast}_complex.npy")).to(dev)
got = p.score(x, args.method)

ref = next(r for r in json.load(open(R / f"logs/classical_{args.contrast}.json"))["rows"]
           if r["method"] == args.method)
bad = []
for k in ("psnr", "ssim", "nmse", "hfen", "head_psnr", "head_ssim", "head_hfen", "sharp", "grain"):
    d = abs(got[k] - ref[k])
    flag = "OK " if d <= args.tol * max(1.0, abs(ref[k])) else "MISMATCH"
    if flag != "OK ":
        bad.append(k)
    print(f"  {flag} {k:11s} recon_classical={ref[k]:.6f}  t1t2_common={got[k]:.6f}  d={d:.2e}")
print(f"scale: recon_classical={json.load(open(R / f'logs/classical_{args.contrast}.json'))['scale_p999']:.6e}"
      f"  t1t2_common={p.scale:.6e}")
if bad:
    raise SystemExit(f"GAUGE MISMATCH on {bad}")
print("GAUGE OK -- the shared module reproduces recon_classical exactly")
