#!/usr/bin/env python3
"""recon_classical.py -- ZF / CG-SENSE / CS-SENSE(ADMM-TV) on the retrospective AF12 T1+FLAIR pair.

Same gauge as arc/cycle.py so the numbers are directly comparable with the MC-GS runs:
  GT      = sum_c conj(S_c) . ifft3c(y_full_c)          (the ESPIRiT-CSM self-consistent gauge)
  scale   = P99.9 of |GT|, applied to GT and to y
  metrics = paired_gs.metrics.evaluate_reconstruction (whole@max) + diag_artefact.score_headonly,
            with hfen / sharp_ratio / grain transcribed verbatim from arc/cycle.py.

Solvers (contract copied from Results_20260908/CE_pre_post_contrast/scripts/ce_classical.py):
  cgsense : CG on (A^H A + tik I) x = A^H y,  tik 0.01, 60 iters
  cssense : ADMM isotropic 3-D TV SENSE,      lam 1e-3, mu 0.1, 35 outer x 8 CG
"""
import argparse, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", required=True, choices=["T1w", "T2s"])
ap.add_argument("--methods", default="zf,cgsense,cssense")
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--cg_iters", type=int, default=60)
ap.add_argument("--tik", type=float, default=0.01)
ap.add_argument("--admm_outer", type=int, default=35)
ap.add_argument("--admm_cg", type=int, default=8)
ap.add_argument("--lam", type=float, default=1e-3)
ap.add_argument("--mu", type=float, default=0.1)
ap.add_argument("--suffix", default="", help="appended to the method key and file names (lambda sweeps); empty = the canonical arm")
args = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import h5py, numpy as np, torch, torch.fft as tfft
import torch.nn.functional as F

ROOT = Path("/data0/congcong/code/GS/mcgs_peds_wt")
sys.path.insert(0, str(ROOT / "paired_gs")); sys.path.insert(0, str(ROOT))
from metrics import evaluate_reconstruction
from diag_artefact.headonly import score_headonly

R = Path("/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro")
dev = torch.device("cuda:0")
log = lambda *a: print(*a, flush=True)


def _c(fn, x, dims):
    return tfft.fftshift(fn(tfft.ifftshift(x, dim=dims), dim=dims, norm="ortho"), dim=dims)
def fft3c(x):  return _c(tfft.fftn,  x, (-3, -2, -1))
def ifft3c(x): return _c(tfft.ifftn, x, (-3, -2, -1))


# ---- metrics transcribed verbatim from arc/cycle.py (do not "improve") ---------------------
def hfen(rec, gt):
    def lap(v):
        return (6.0 * v - torch.roll(v, 1, -3) - torch.roll(v, -1, -3)
                - torch.roll(v, 1, -2) - torch.roll(v, -1, -2)
                - torch.roll(v, 1, -1) - torch.roll(v, -1, -1))
    g = gt / (gt.max() + 1e-9); r = rec / (gt.max() + 1e-9); lg = lap(g)
    return float(torch.linalg.norm(lap(r) - lg) / (torch.linalg.norm(lg) + 1e-9))


def _gmag(v):
    gx = torch.zeros_like(v); gy = torch.zeros_like(v); gz = torch.zeros_like(v)
    gx[:-1] = v[1:] - v[:-1]; gy[:, :-1] = v[:, 1:] - v[:, :-1]; gz[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
    return torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)


def sharp_ratio(rec, gt):
    brain = gt > 0.05 * gt.max()
    return float(_gmag(rec)[brain].mean() / (_gmag(gt)[brain].mean() + 1e-9))


def _quantile(t, q):
    """torch.quantile caps at ~16M elements; fall back to numpy for larger tensors."""
    if t.numel() <= 16_000_000:
        return torch.quantile(t, q)
    return torch.tensor(float(np.quantile(t.detach().cpu().numpy(), q)), device=t.device, dtype=t.dtype)


def grain(rec, gt):
    g = gt / (gt.max() + 1e-9); r = rec / (gt.max() + 1e-9)
    sm = F.avg_pool3d(r[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    hp = r - sm
    brain = g > 0.05 * g.max()
    gg = _gmag(g)
    thr = _quantile(gg[brain], 0.30)
    flat = brain & (gg <= thr)
    return float(hp[flat].std())


# ---- data ---------------------------------------------------------------------------------
t0 = time.time()
with h5py.File(R / f"data/ksp_{args.contrast}.h5", "r") as f:
    y_full = torch.from_numpy(f["kspace"][:]).to(torch.complex64)
with h5py.File(R / f"data/csm_espirit_{args.contrast}.h5", "r") as f:
    S = torch.from_numpy(f["map"][:]).to(torch.complex64)
with h5py.File(R / "data/mask_af12_union_3d.h5", "r") as f:
    m3 = torch.from_numpy(f["mask"][:]).to(torch.float32)
assert y_full.shape == S.shape, (y_full.shape, S.shape)
NC = y_full.shape[0]
y_full = y_full.to(dev); S = S.to(dev); m3 = (m3 > 0.5).to(dev)
log(f"[{args.contrast}] coils={NC} vol={tuple(m3.shape)} "
    f"mask frac={m3.float().mean():.5f} AF_rect={1/m3.float().mean():.3f}")

CHUNK = 4
def sense_adj(k):                       # (C,X,Y,Z) -> (X,Y,Z)
    acc = torch.zeros(m3.shape, dtype=torch.complex64, device=dev)
    for i in range(0, NC, CHUNK):
        acc += (S[i:i + CHUNK].conj() * ifft3c(k[i:i + CHUNK])).sum(0)
    return acc

def AHA(x):
    acc = torch.zeros_like(x)
    for i in range(0, NC, CHUNK):
        acc += (S[i:i + CHUNK].conj() * ifft3c(fft3c(S[i:i + CHUNK] * x) * m3)).sum(0)
    return acc

gt_c = sense_adj(y_full)                                       # GT from the FULL k-space
mag = gt_c.abs().reshape(-1)
scale = float(np.quantile(mag.cpu().numpy(), 0.999))
gt_c = gt_c / scale; y_full = y_full / scale
gt = gt_c.abs()
log(f"[{args.contrast}] P99.9 |GT| scale = {scale:.6e}; |GT| max = {float(gt.max()):.4f}")

y_us = y_full * m3.unsqueeze(0)


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
    gx = torch.roll(x, -1, 0) - x; gy = torch.roll(x, -1, 1) - x; gz = torch.roll(x, -1, 2) - x
    return torch.stack([gx, gy, gz], 0)


def div3(g):                            # already the adjoint D^H (see CE gotchas)
    gx, gy, gz = g[0], g[1], g[2]
    return ((torch.roll(gx, 1, 0) - gx) + (torch.roll(gy, 1, 1) - gy) + (torch.roll(gz, 1, 2) - gz))


def run(method):
    t = time.time()
    if method == "zf":
        x = sense_adj(y_us)
    elif method == "cgsense":
        b = sense_adj(y_us)
        x = cg(lambda v: AHA(v) + args.tik * v, b, args.cg_iters)
    elif method == "cssense":
        b0 = sense_adj(y_us)
        x = torch.zeros_like(b0); z = grad3(x); u = torch.zeros_like(z)
        for _ in range(args.admm_outer):
            # CE gotcha: div3 as written IS the adjoint D^H in this sign convention, so the CG
            # operator is AHA + mu.div3(grad3(.)) (PSD: div3(grad3(x)) = 2x - roll(x,1) - roll(x,-1)
            # per axis). Negating that term makes it indefinite and CG returns NaN.
            rhs = b0 + args.mu * div3(z - u)
            x = cg(lambda v: AHA(v) + args.mu * div3(grad3(v)), rhs, args.admm_cg)
            v = grad3(x) + u
            nv = torch.sqrt((v.abs() ** 2).sum(0, keepdim=True) + 1e-12)
            z = v * torch.clamp(1.0 - args.lam / (args.mu * nv), min=0.0)
            u = v - z
    else:
        raise ValueError(method)
    rec = x.abs()
    met = evaluate_reconstruction(x, gt_c, compute_extended=False)
    ho = score_headonly(rec.cpu().numpy(), gt.cpu().numpy())
    out = dict(method=method + args.suffix, contrast=args.contrast, lam=args.lam, mu=args.mu, tik=args.tik,
               psnr=float(met["psnr"]), ssim=float(met["ssim"]), nmse=float(met["nmse"]),
               hfen=hfen(rec, gt), sharp=sharp_ratio(rec, gt), grain=grain(rec, gt),
               seconds=time.time() - t, **{k: float(v) for k, v in ho.items()})
    np.save(R / f"recon/{method}{args.suffix}_{args.contrast}_complex.npy", x.cpu().numpy().astype(np.complex64))
    log(f"[{args.contrast}] {method:8s} psnr={out['psnr']:.3f} ssim={out['ssim']:.4f} "
        f"nmse={out['nmse']:.5f} hfen={out['hfen']:.4f} head_psnr={out['head_psnr']:.3f} "
        f"head_ssim={out['head_ssim']:.4f} sharp={out['sharp']:.3f} grain={out['grain']:.5f} "
        f"({out['seconds']:.0f} s)")
    return out


rows = [run(m) for m in args.methods.split(",")]
if not args.suffix:
    np.save(R / f"recon/gt_{args.contrast}_complex.npy", gt_c.cpu().numpy().astype(np.complex64))
p = R / f"logs/classical_{args.contrast}{args.suffix}.json"
json.dump({"contrast": args.contrast, "scale_p999": scale, "coils": int(NC),
           "mask_frac": float(m3.float().mean()), "rows": rows,
           "total_seconds": time.time() - t0}, open(p, "w"), indent=2)
log(f"WROTE {p}  total {time.time()-t0:.0f} s")
