#!/usr/bin/env python3
"""score_volume.py -- re-score any saved volume through t1t2_common.Pair (the one gauge) and, with --check, compare against
the row an arm wrote itself. Every arm here scores through the same Pair.score, so a mismatch means a volume was written in
the wrong units or layout (the GS and MC-INR conversions are the places that can go wrong).
  python scripts/score_volume.py --scan pre --file recon/gs_joint_pre_complex.npy --method gs_joint --check logs/gs_joint_pre.json"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--scan", required=True, choices=["pre", "post"]); ap.add_argument("--file", required=True)
ap.add_argument("--method", required=True); ap.add_argument("--check", default=None); ap.add_argument("--gpu", type=int, default=-1)
ap.add_argument("--tol", type=float, default=2e-4)
a = ap.parse_args()
if a.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1t2_common import Pair, R   # noqa: E402

dev = torch.device("cpu") if a.gpu < 0 else torch.device("cuda:0")
P = Pair(a.scan, dev)
if a.gpu < 0:
    P.gt_c = P.gt_c.to(torch.complex128); P.gt = P.gt.to(torch.float64)     # CPU float32 norms drift ~1e-3 on 10M voxels
v = np.load(R / a.file if not Path(a.file).is_absolute() else a.file)
x = torch.from_numpy(v if np.iscomplexobj(v) else v.astype(np.complex64)).to(dev)
if a.gpu < 0:
    x = x.to(torch.complex128)
got = P.score(x, a.method)
print(json.dumps({k: got[k] for k in ("head_psnr", "head_ssim", "head_nmse", "head_hfen", "psnr_cplx", "psnr", "ssim", "nmse")}, indent=1))
if a.check:
    ref = next(r for r in json.load(open(R / a.check))["rows"] if r["method"] == a.method)
    bad = [k for k in ("psnr", "ssim", "nmse", "hfen", "head_psnr", "head_ssim", "head_nmse", "head_hfen") if abs(got[k] - ref[k]) > a.tol * max(1.0, abs(ref[k]))]
    for k in ("head_psnr", "head_ssim", "head_nmse", "head_hfen", "psnr"):
        print(f"  {'MISMATCH' if k in bad else 'OK      '} {k:10s} recorded={ref[k]:.6f} rescored={got[k]:.6f}")
    if bad:
        raise SystemExit(f"GAUGE MISMATCH on {bad}")
    print("GAUGE OK")
