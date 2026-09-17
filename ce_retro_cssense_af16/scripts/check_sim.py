#!/usr/bin/env python3
"""check_sim.py -- the operator gates of the spec (section 10, items 1-2), on the real synthetic data, one GPU.
  (a) noiseless full k-space: x_hat = sum_c conj(S_c) ifft3c(F(S_c x_GT)) / sum_c |S_c|^2 == x_GT inside the head (rel err < 1e-4)
  (b) CG-SENSE (Tikhonov 1e-6, 60 iters) on that noiseless full k-space: head PSNR > 60 dB
  (c) the noise actually written: std of (gs/<scan>_kspace.h5 - F(S x_GT)) within 2 % of meta sigma
  (d) for the record: ZF head PSNR of the AF16 noisy data
Writes logs/check_sim.json; exits non-zero on any failed gate.
  python scripts/check_sim.py --gpu 0
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, required=True); a = ap.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1t2_common import Pair, R, fft3c, ifft3c, head_mask   # noqa: E402

dev = torch.device("cuda:0")
meta = json.load(open(R / "sim/meta.json"))
out, ok = {}, True


def cg(applyA, b, iters):
    x = torch.zeros_like(b); r = b.clone(); p = r.clone(); rs = (r.conj() * r).sum().real
    for _ in range(iters):
        Ap = applyA(p); al = rs / (p.conj() * Ap).sum().real.clamp_min(1e-30)
        x = x + al * p; r = r - al * Ap; rs_new = (r.conj() * r).sum().real
        p = r + (rs_new / rs.clamp_min(1e-30)) * p; rs = rs_new
    return x


for scan in ("pre", "post"):
    P = Pair(scan, dev)
    head = torch.from_numpy(head_mask(P.gt.cpu().numpy())).to(dev)
    sos2 = (P.S.abs() ** 2).sum(0)
    adj = torch.zeros_like(P.gt_c); noise2 = 0.0; npts = 0
    for i in range(0, P.NC, P.chunk):
        k = fft3c(P.S[i:i + P.chunk] * P.gt_c)                        # noiseless, gauge units
        adj += (P.S[i:i + P.chunk].conj() * ifft3c(k)).sum(0)
        d = (P.y_full[i:i + P.chunk] - k)                             # the written noise (unmasked h5), gauge units
        noise2 += float(d.real.square().sum() + d.imag.square().sum()); npts += 2 * d.numel()
    xhat = adj / sos2.clamp_min(1e-6)
    # Step-7 permitted adaptation (task-5-brief.md): the raw (unregularised) adjoint identity is singular
    # wherever coil coverage sum|S|^2 is near zero, so the adjoint gate is restricted to head voxels with
    # sum|S|^2 > 0.25; recorded below (adjoint_head_voxels_*) for the record. CG/ZF gates keep the full head mask.
    head_adj = head & (sos2 > 0.25)
    rel = float(((xhat - P.gt_c)[head_adj].abs().norm() / P.gt_c[head_adj].abs().norm()))
    ones = torch.ones_like(P.m3)
    xcg = cg(lambda v: P.AHA(v, ones) + 1e-6 * v, adj, 60)
    mse = float(((xcg.abs() - P.gt)[head] ** 2).mean()); psnr_cg = float(20 * np.log10(P.gt.max().item()) - 10 * np.log10(mse))
    sigma_w = float(np.sqrt(noise2 / npts)) * P.scale; sigma_ref = meta["scans"][scan]["sigma"]
    zf = P.score(P.sense_adj(P.y), "zf")["head_psnr"]
    row = dict(adjoint_rel_err=rel, cg_noiseless_head_psnr=psnr_cg, sigma_written=sigma_w, sigma_meta=sigma_ref,
               sigma_rel_dev=abs(sigma_w / sigma_ref - 1), zf_af16_head_psnr=zf,
               adjoint_voxel_condition="head & (sum_c|S_c|^2 > 0.25)",
               adjoint_head_voxels_total=int(head.sum().item()), adjoint_head_voxels_used=int(head_adj.sum().item()),
               gates=dict(adjoint=rel < 1e-4, cg=psnr_cg > 60, sigma=abs(sigma_w / sigma_ref - 1) < 0.02))
    ok &= all(row["gates"].values()); out[scan] = row
    print(f"[{scan}] adjoint rel err {rel:.2e}  CG noiseless head PSNR {psnr_cg:.2f} dB  sigma {sigma_w:.4e} vs {sigma_ref:.4e}  ZF AF16 {zf:.2f} dB  gates {row['gates']}", flush=True)
    del P, adj, xhat, xcg; torch.cuda.empty_cache()
json.dump(out, open(R / "logs/check_sim.json", "w"), indent=1)
print("CHECK_SIM " + ("OK" if ok else "FAILED")); sys.exit(0 if ok else 1)
