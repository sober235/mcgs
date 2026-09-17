#!/usr/bin/env python3
"""recon_classical.py -- ZF / CG-SENSE / CS-SENSE (ADMM isotropic 3-D TV) on the synthetic AF16 pre/post pair.

Solvers and defaults are the jyz AF16 ones (T1FLAIR_AF16_v3_jiangyizhou/scripts/recon_classical.py), the data and
gauge come from t1t2_common.Pair (the CS-SENSE truth). --gpu -1 runs on the CPU (tests, tiny data).
  cgsense : CG on (A^H A + tik I) x = A^H y,  tik 0.01, 60 iters
  cssense : ADMM isotropic 3-D TV SENSE,      lam, mu 0.1, admm_outer x admm_cg CG   (the registered arm: --admm_outer 200 with a lambda sweep)
"""
import argparse, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--scan", required=True, choices=["pre", "post"])
ap.add_argument("--methods", default="zf,cgsense,cssense")
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--cg_iters", type=int, default=60)
ap.add_argument("--tik", type=float, default=0.01)
ap.add_argument("--admm_outer", type=int, default=35)
ap.add_argument("--admm_cg", type=int, default=8)
ap.add_argument("--lam", type=float, default=1e-3)
ap.add_argument("--mu", type=float, default=0.1)
ap.add_argument("--suffix", default="", help="appended to the method key and file names (lambda sweeps); empty = the canonical arm")
args = ap.parse_args() if __name__ == "__main__" else ap.parse_args(["--scan", "pre", "--gpu", "-1"])

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch                                             # noqa: E402
from t1t2_common import Pair, R                                       # noqa: E402

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


def div3(g):                            # already the adjoint D^H (CE gotcha: do not negate)
    gx, gy, gz = g[0], g[1], g[2]
    return ((torch.roll(gx, 1, 0) - gx) + (torch.roll(gy, 1, 1) - gy) + (torch.roll(gz, 1, 2) - gz))


def run(P, method):
    t = time.time()
    if method == "zf":
        x = P.sense_adj(P.y)
    elif method == "cgsense":
        x = cg(lambda v: P.AHA(v) + args.tik * v, P.sense_adj(P.y), args.cg_iters)
    elif method == "cssense":
        b0 = P.sense_adj(P.y)
        x = torch.zeros_like(b0); z = grad3(x); u = torch.zeros_like(z)
        for _ in range(args.admm_outer):
            rhs = b0 + args.mu * div3(z - u)
            x = cg(lambda v: P.AHA(v) + args.mu * div3(grad3(v)), rhs, args.admm_cg)
            v = grad3(x) + u
            nv = torch.sqrt((v.abs() ** 2).sum(0, keepdim=True) + 1e-12)
            z = v * torch.clamp(1.0 - args.lam / (args.mu * nv), min=0.0)
            u = v - z
    else:
        raise ValueError(method)
    row = P.score(x, method + args.suffix, seconds=time.time() - t, lam=args.lam, mu=args.mu, tik=args.tik)
    np.save(R / f"recon/{method}{args.suffix}_{P.contrast}_complex.npy", x.cpu().numpy().astype(np.complex64))
    log(f"[{P.contrast}] {method + args.suffix:18s} head_psnr={row['head_psnr']:.3f} head_ssim={row['head_ssim']:.4f} "
        f"head_nmse={row['head_nmse']:.5f} head_hfen={row['head_hfen']:.4f} psnr={row['psnr']:.3f} ({row['seconds']:.0f} s)")
    return row


def main():
    t0 = time.time()
    dev = torch.device("cpu") if args.gpu < 0 else torch.device("cuda:0")
    (R / "recon").mkdir(exist_ok=True); (R / "logs").mkdir(exist_ok=True)
    P = Pair(args.scan, dev)
    log(f"[{args.scan}] coils={P.NC} vol={P.G} mask frac={P.m3.float().mean():.5f} AF_rect={1 / P.m3.float().mean():.3f} scale={P.scale:.4e}")
    rows = [run(P, m) for m in args.methods.split(",")]
    if not args.suffix:
        np.save(R / f"recon/gt_{args.scan}_complex.npy", P.gt_c.cpu().numpy().astype(np.complex64))
    p = R / f"logs/classical_{args.scan}{args.suffix}.json"
    json.dump({"contrast": args.scan, "scale_p999": P.scale, "coils": int(P.NC), "mask_frac": float(P.m3.float().mean()),
               "rows": rows, "total_seconds": time.time() - t0}, open(p, "w"), indent=2)
    log(f"WROTE {p}  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
