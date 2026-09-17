#!/usr/bin/env python3
"""ce_tv_sense.py -- converged CS-SENSE (isotropic 3-D TV) and Joint SENSE (vectorial TV over pre+post) on
the UIH prospective CE pair, one lambda after another in one process.

The ADMM is ce_classical.py's cssense arm unchanged (tik 0.01, mu 0.1, 8 warm-started CG steps per outer
iteration, backward-difference grad3 / adjoint div3), except
  * --outer defaults to 200: the jyz/hyw audit (T1FLAIR_AF12_retro README, 2026-09-10) found the 35 outer
    iterations inherited from this topic under-converged by 0.35-0.61 dB, flat from 100 to 200;
  * --mode joint couples the two scans in the shrinkage only: at each voxel the 6-vector of both scans'
    gradients is soft-thresholded as one vector, so edges are encouraged to coincide. Each scan keeps its own
    maps, data term and normalisation (its own P99.9 |ZF|, the value ce_classical.py used);
  * --mask fit trains on the 95% lines of heldout_split/split_95_5_seed0.npz, --mask full on every line.
Lambda is chosen on the held-out residual of --mask fit runs (ce_select_lambda.py), then refit with --mask full.

usage: ce_tv_sense.py --mode cs --scan pre --mask fit --lams 2.5e-4,5e-4 --gpu 0
       ce_tv_sense.py --mode joint --mask full --lams 1.41e-3 --gpu 0
Writes <out>/<tag>_complex.npy (alpha-scaled onto the measured data, as every column of this topic) and
<out>/<tag>_meta.json, tag = {cssense|jsense}_o<outer>_lam<lam>_<mask>_<scan>. Existing metas are skipped.
"""
import argparse
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["cs", "joint"])
ap.add_argument("--scan", choices=["pre", "post"], help="cs mode only")
ap.add_argument("--mask", required=True, choices=["fit", "full"])
ap.add_argument("--lams", required=True, help="comma-separated")
ap.add_argument("--gpu", type=int, required=True, help="-1 = CPU (smoke tests only)")
ap.add_argument("--outer", type=int, default=200)
ap.add_argument("--cg", type=int, default=8)
ap.add_argument("--mu", type=float, default=0.1)
ap.add_argument("--tik", type=float, default=0.01)
ap.add_argument("--out", required=True)
args = ap.parse_args()
if args.mode == "cs" and not args.scan:
    ap.error("--scan is required in cs mode")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.gpu >= 0 else ""
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ce_common as C  # noqa: E402

dev = torch.device("cuda:0" if args.gpu >= 0 else "cpu")
torch.zeros(1, device=dev)            # hold a CUDA context from the start (other schedulers read it as "card busy")
if args.gpu < 0:
    torch.set_num_threads(16)
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)
scans = [args.scan] if args.mode == "cs" else list(C.SCANS)
lams = [float(v) for v in args.lams.split(",")]
name = "cssense" if args.mode == "cs" else "jsense"
CHUNK = 14


def tag(lam, scan):
    return f"{name}_o{args.outer}_lam{lam:g}_{args.mask}_{scan}"


todo = [lam for lam in lams if not all((OUT / f"{tag(lam, s)}_meta.json").exists() for s in scans)]
if not todo:
    log("nothing to do"); sys.exit(0)

t0 = time.time()
fit_plane, ho_plane = C.load_split(dev)
D = {}
for s in scans:
    y, S, m2 = C.load_scan(s, dev)
    plane = fit_plane if args.mask == "fit" else m2
    scale = C.classical_scale(s)
    yn = (y * plane) / scale
    D[s] = dict(y=y, S=S, m2=m2, plane=plane, scale=scale, NC=y.shape[0])
    log(f"[{s}] coils {y.shape[0]} plane {args.mask} {int(plane.sum())} lines  scale {scale:.4e}  ({time.time() - t0:.0f}s)")

    def AH_k(k, plane=plane, S=S):
        acc = torch.zeros(k.shape[1:], dtype=k.dtype, device=dev)
        for i in range(0, k.shape[0], CHUNK):
            acc += (S[i:i + CHUNK].conj() * C.ifft3c(k[i:i + CHUNK] * plane)).sum(0)
        return acc

    def AHA(x, plane=plane, S=S):
        acc = torch.zeros_like(x)
        for i in range(0, S.shape[0], CHUNK):
            kk = C.fft3c(S[i:i + CHUNK] * x) * plane
            acc += (S[i:i + CHUNK].conj() * C.ifft3c(kk)).sum(0)
        return acc

    D[s]["AHA"] = AHA
    D[s]["rhs0"] = AH_k(yn)
    del yn


def cg(applyA, b, x0=None, iters=60, tol=1e-10):
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - applyA(x); p = r.clone()
    rs = torch.vdot(r.flatten(), r.flatten()).real
    for _ in range(iters):
        Ap = applyA(p)
        a = rs / torch.vdot(p.flatten(), Ap.flatten()).real.clamp_min(1e-30)
        x = x + a * p; r = r - a * Ap
        rs_new = torch.vdot(r.flatten(), r.flatten()).real
        if rs_new.sqrt() < tol: break
        p = r + (rs_new / rs.clamp_min(1e-30)) * p; rs = rs_new
    return x


def grad3(x):
    return torch.stack([x - torch.roll(x, 1, 0), x - torch.roll(x, 1, 1), x - torch.roll(x, 1, 2)])


def div3(g):
    return sum(g[d] - torch.roll(g[d], -1, d) for d in range(3))


for lam in todo:
    t1 = time.time()
    MU, TIK = args.mu, args.tik
    X, Z, U = {}, {}, {}
    for s in scans:
        d = D[s]
        X[s] = cg(lambda v, d=d: d["AHA"](v) + TIK * v, d["rhs0"], iters=args.cg)
        Z[s] = grad3(X[s]); U[s] = torch.zeros_like(Z[s])
    for it in range(args.outer):
        G = {}
        for s in scans:
            d = D[s]
            rhs = d["rhs0"] + MU * div3(Z[s] - U[s])
            X[s] = cg(lambda v, d=d: d["AHA"](v) + TIK * v + MU * div3(grad3(v)), rhs, x0=X[s], iters=args.cg)
            G[s] = grad3(X[s]) + U[s]
        mag = torch.sqrt(sum(G[s].abs().square().sum(0) for s in scans))      # isotropic; joint over scans
        shrink = (1.0 - (lam / MU) / mag.clamp_min(1e-12)).clamp_min(0.0)
        for s in scans:
            Z[s] = G[s] * shrink
            U[s] = G[s] - Z[s]
        del G, mag, shrink
        if (it + 1) % 25 == 0 or it == 0:
            log(f"  [{name} lam {lam:g} {args.mask}] outer {it + 1}/{args.outer}  "
                + " ".join(f"{s}|x|max={X[s].abs().max().item():.4f}" for s in scans) + f"  {time.time() - t1:.0f}s")
    for s in scans:
        d = D[s]
        x = X[s] * d["scale"]
        r = C.kres_metrics(x, d["y"], d["S"],
                           C.drop_foreign_planes({"full": d["m2"], "fit": fit_plane, "ho": ho_plane}, d["m2"]))
        alpha = r["full"]["alpha"]
        np.save(OUT / f"{tag(lam, s)}_complex.npy", (alpha * x).cpu().numpy().astype(np.complex64))
        meta = dict(method=name, mode=args.mode, scan=s, mask=args.mask, lam=lam, outer=args.outer, cg=args.cg,
                    mu=MU, tik=TIK, scale=d["scale"], alpha_re=float(np.real(alpha)), alpha_im=float(np.imag(alpha)),
                    kres=r["full"]["kres"], kres_fit=r.get("fit", {}).get("kres"), kres_heldout=r.get("ho", {}).get("kres"),
                    heldout_is_unseen=args.mask == "fit", shape=list(x.shape),
                    seconds=time.time() - t1, torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(0) if args.gpu >= 0 else "cpu")
        C.dump_json(meta, OUT / f"{tag(lam, s)}_meta.json")
        fmt = lambda v: "-" if v is None else f"{v:.5f}"
        log(f"[{tag(lam, s)}] kres full {meta['kres']:.5f} fit {fmt(meta['kres_fit'])} held-out {fmt(meta['kres_heldout'])}"
            f"  |alpha| {abs(alpha):.4f}  {meta['seconds']:.0f}s")
    del X, Z, U
    torch.cuda.empty_cache()
log(f"DONE {time.time() - t0:.0f}s")
