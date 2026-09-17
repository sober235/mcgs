#!/usr/bin/env python3
"""recon_joint_cssense.py -- joint CS-SENSE: the two contrasts reconstructed together under a
vectorial (joint) 3-D total variation, the classical counterpart of "shared geometry".

Same ADMM as recon_classical.py's cssense arm (isotropic TV, lam, mu, 35 outer x 8 CG), except
that the shrinkage acts on the 6-vector (gradients of both contrasts) at every voxel: edges are
encouraged to sit at the same places in both images. The two data-fidelity CG steps are
separable (each contrast has its own maps; the mask is shared), so the joint model costs about
twice a single CS-SENSE run. Each contrast is normalised by its own P99.9(|GT|) before coupling,
as in the gauge. Note the standing warning (CRISP phase 0): at the SAME lambda a vectorial TV
is a weaker penalty per contrast than the scalar one, so lambda must be swept for both arms and
the best of each reported, never the same lambda for both.

usage: recon_joint_cssense.py --topic <dir> --contrasts T1,T2 --gpu N [--lam 1e-3 --mu 0.1 --suffix _lam1e-3]
Writes recon/jcssense<suffix>_<c>_complex.npy and logs/jcssense<suffix>.json.
"""
import argparse, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--topic", required=True)
ap.add_argument("--contrasts", required=True)
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--admm_outer", type=int, default=35)
ap.add_argument("--admm_cg", type=int, default=8)
ap.add_argument("--lam", type=float, default=1e-3)
ap.add_argument("--mu", type=float, default=0.1)
ap.add_argument("--suffix", default="")
args = ap.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import numpy as np
import torch

sys.path.insert(0, str(Path(args.topic) / "scripts"))
import t1t2_common as C                                            # noqa: E402  (this topic's gauge)

R = Path(args.topic)
dev = torch.device("cuda:0")
log = lambda *a: print(*a, flush=True)


def cg(applyA, b, iters, tol=1e-10):
    x = torch.zeros_like(b); r = b.clone(); p = r.clone()
    rs = (r.conj() * r).sum().real
    for _ in range(iters):
        Ap = applyA(p)
        a = rs / (p.conj() * Ap).sum().real.clamp_min(1e-30)
        x = x + a * p; r = r - a * Ap
        rs_new = (r.conj() * r).sum().real
        if rs_new.sqrt() < tol:
            break
        p = r + (rs_new / rs.clamp_min(1e-30)) * p
        rs = rs_new
    return x


def grad3(x):
    return torch.stack([torch.roll(x, -1, 0) - x, torch.roll(x, -1, 1) - x, torch.roll(x, -1, 2) - x], 0)


def div3(g):
    gx, gy, gz = g[0], g[1], g[2]
    return ((torch.roll(gx, 1, 0) - gx) + (torch.roll(gy, 1, 1) - gy) + (torch.roll(gz, 1, 2) - gz))


t0 = time.time()
cs = args.contrasts.split(",")
pairs = [C.Pair(c, dev) for c in cs]
for P in pairs:
    log(f"[{P.contrast}] coils={P.NC} vol={P.G} scale={P.scale:.4e}")
b0 = [P.sense_adj(P.y) for P in pairs]
x = [torch.zeros_like(b) for b in b0]
z = [grad3(xi) for xi in x]
u = [torch.zeros_like(zi) for zi in z]
for it in range(args.admm_outer):
    for k, P in enumerate(pairs):
        rhs = b0[k] + args.mu * div3(z[k] - u[k])
        x[k] = cg(lambda v, P=P: P.AHA(v) + args.mu * div3(grad3(v)), rhs, args.admm_cg)
    v = [grad3(x[k]) + u[k] for k in range(len(pairs))]
    nv = torch.sqrt(sum((vk.abs() ** 2).sum(0, keepdim=True) for vk in v) + 1e-12)   # joint norm over both contrasts
    shrink = torch.clamp(1.0 - args.lam / (args.mu * nv), min=0.0)
    for k in range(len(pairs)):
        z[k] = v[k] * shrink
        u[k] = v[k] - z[k]
    if it % 5 == 0 or it == args.admm_outer - 1:
        log(f"  iter {it + 1}/{args.admm_outer}  {time.time() - t0:.0f} s")
rows = []
for k, P in enumerate(pairs):
    row = P.score(x[k], "jcssense" + args.suffix, seconds=time.time() - t0, lam=args.lam, mu=args.mu, partner=[c for c in cs if c != P.contrast][0])
    rows.append(row)
    np.save(R / f"recon/jcssense{args.suffix}_{P.contrast}_complex.npy", x[k].cpu().numpy().astype(np.complex64))
    log(f"[{P.contrast}] jcssense{args.suffix} psnr={row['psnr']:.3f} ssim={row['ssim']:.4f} head_psnr={row['head_psnr']:.3f} "
        f"head_ssim={row['head_ssim']:.4f} head_hfen={row['head_hfen']:.4f} sharp={row['sharp']:.3f}")
p = R / f"logs/jcssense{args.suffix}.json"
json.dump({"contrasts": cs, "lam": args.lam, "mu": args.mu, "rows": rows, "total_seconds": time.time() - t0}, open(p, "w"), indent=2)
log(f"WROTE {p}  total {time.time() - t0:.0f} s")
