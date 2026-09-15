#!/usr/bin/env python3
"""ce_spirit.py -- L1-SPIRiT (Lustig & Pauly, MRM 2010) on the UIH prospective CE pair (2026-09-15).

The method is the jyz topics' registered L1-SPIRiT arm, reused without change: every function between the two
"verbatim" markers below is copied byte for byte from
/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/scripts/recon_spirit.py (md5 6c25087d20de,
identical in the AF12/AF16/AF20 topics). That port was aligned with Lustig's MATLAB toolbox to 1e-12 in double
precision (spirit_align_check.py). It runs a damped proximal-gradient / FISTA solver on the pocsSPIRiT objective,
because pocsSPIRiT itself diverges on that data (see that topic's README).

CE wiring, the only new code:
  * hybrid (x, ky, kz) space along this data's fully sampled readout (axis 0, 256 positions), one 2D problem per
    readout position, as in ce_classical.py's ZS-SSDU arm;
  * k-space divided by ce_classical.py's P99.9 |ZF| scale, completed coil k-space combined with this topic's coil
    maps (the maps every column uses), then scored with ce_common.kres_metrics like every other arm;
  * --mask fit fills only the 95% lines of heldout_split/split_95_5_seed0.npz (the data-consistency set);
    --mask full uses every acquired line;
  * the kernel is calibrated from the fully sampled 24 x 24 ACS block of the FULL acquired data in both cases. The
    95% mask shatters that block (its largest centred fully sampled block is 18 x 4), and the coil maps every arm
    uses were likewise estimated from the full data. 26 of the 576 ACS lines (4.5%) are held-out lines: they
    inform the kernel, never the data-consistency term;
  * all lambdas of one run share one calibration per readout chunk (the kernel does not depend on lambda).

usage: ce_spirit.py --scan pre --mask fit --ksize 5 --lams 0,5e-4,1e-3,2e-3,4e-3,8e-3 --gpu 0 --out spirit
Writes <out>/spirit_ks<k>_i<iters>_lam<lam>_<mask>_<scan>_{complex.npy,meta.json}; existing metas are skipped.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.fft as tfft


def _c(fn, x, dims):
    return tfft.fftshift(fn(tfft.ifftshift(x, dim=dims), dim=dims, norm="ortho"), dim=dims)


def fft2c(x):  return _c(tfft.fftn,  x, (-2, -1))
def ifft2c(x): return _c(tfft.ifftn, x, (-2, -1))
def ifft1c(x, d): return _c(tfft.ifftn, x, (d,))


# ======================== verbatim from recon_spirit.py (md5 6c25087d20de), lines 70-283 ========================
# ---- calibration: dat2AtA + calibrate, verbatim ----------------------------------------------
def build_AtA(acs, ks):
    """acs (B, NC, Ca, Cb) -> (B, D, D), D = NC * ks * ks. Flattening order (coil, k1, k2) with k2
    fastest; any consistent order gives the same kernel, this one matches the reshape below."""
    B, NC = acs.shape[:2]
    p = acs.unfold(2, ks, 1).unfold(3, ks, 1)                  # (B, NC, na, nb, ks, ks)
    p = p.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, NC * ks * ks)
    return p.conj().transpose(1, 2) @ p


def calib_kernels(AtA, NC, ks, tyk):
    """calibrate.m for every target coil: drop that coil's centre tap, solve the normal equations
    with lambda = ||AtA||_F / rows * tyk, put a zero back at the dropped tap."""
    B, D = AtA.shape[0], AtA.shape[-1]
    c = ks // 2
    ker = torch.zeros(B, NC, NC, ks, ks, dtype=AtA.dtype, device=AtA.device)   # [in, out, k1, k2]
    eye = torch.eye(D - 1, dtype=AtA.dtype, device=AtA.device)
    for n in range(NC):
        iy = n * ks * ks + c * ks + c
        idx = torch.cat([torch.arange(iy, device=AtA.device), torch.arange(iy + 1, D, device=AtA.device)])
        M = AtA[:, idx][:, :, idx]
        b = AtA[:, idx, iy]
        lam = torch.linalg.matrix_norm(M, ord="fro").real / M.shape[-1] * tyk
        sol = torch.linalg.solve(M + lam[:, None, None] * eye, b.unsqueeze(-1)).squeeze(-1)
        full = torch.zeros(B, D, dtype=AtA.dtype, device=AtA.device)
        full[:, idx] = sol
        ker[:, :, n] = full.view(B, NC, ks, ks)
    return ker


def image_kernels(ker, ny, nz):
    """SPIRiT.m 'fft': flip both kernel axes, scale by sqrt(ny*nz), zero-pad centred on (ny//2,
    nz//2) -- MATLAB zpad's centre -- then ifft2c. -> (B, NC_in, NC_out, ny, nz)."""
    B, NC, _, ks, _ = ker.shape
    K = torch.zeros(B, NC, NC, ny, nz, dtype=ker.dtype, device=ker.device)
    y0, z0 = ny // 2 - ks // 2, nz // 2 - ks // 2
    K[..., y0:y0 + ks, z0:z0 + ks] = torch.flip(ker, dims=(-2, -1)) * (ny * nz) ** 0.5
    return ifft2c(K)


def gop(x, K):
    """mtimes.m 'fft', forward: returns (G - I) x. x (B, NC, ny, nz). K is [in, out, y, z]."""
    return fft2c(torch.einsum("bcyz,bcnyz->bnyz", ifft2c(x), K)) - x


def gop_adj(x, K):
    """mtimes.m 'fft', adjoint branch: (G - I)^H x."""
    return fft2c(torch.einsum("bnyz,bcnyz->bcyz", ifft2c(x), K.conj())) - x


def normal_matrix(K):
    """(G - I)^H (G - I) as one per-pixel matrix instead of two operator applications.

    (G - I) = fft2c . (M - I) . ifft2c, with M the image-domain kernel: per pixel a NC x NC matrix
    A[n, c] = K[c, n]. So the gradient of ||(G-I)x||^2 is fft2c . P . ifft2c with P = (A-I)^H (A-I),
    one FFT pair and one batched matmul per iteration. Laid out contiguously as [pixel, c, c'] --
    einsum over the coil axes of a (B, NC, NC, ny, nz) tensor re-permutes 6 GB every iteration and
    is ~50x slower, which is what made the first sweep crawl.
    """
    B, NC, _, ny, nz = K.shape
    A = K.permute(0, 3, 4, 2, 1).reshape(-1, NC, NC).contiguous()      # [pixel, out n, in c]
    A -= torch.eye(NC, dtype=K.dtype, device=K.device)
    return (A.conj().transpose(-2, -1) @ A).contiguous()               # [pixel, c, c']


def applyP(x, P):
    """x (B, NC, ny, nz) -> P x, with P from normal_matrix."""
    B, NC, ny, nz = x.shape
    v = x.permute(0, 2, 3, 1).reshape(-1, NC, 1)
    return (P @ v).reshape(B, ny, nz, NC).permute(0, 3, 1, 2)


def grad(x, P):
    """gradient of ||(G - I) x||^2 / 2."""
    return fft2c(applyP(ifft2c(x), P))


def lipschitz(P, unacq, shape, iters=15):
    """largest eigenvalue of (G-I)^H (G-I) restricted to the holes, by power iteration."""
    v = torch.randn(shape, dtype=P.dtype, device=P.device) * unacq
    lam = 0.0
    for _ in range(iters):
        v = v / torch.linalg.vector_norm(v).clamp_min(1e-30)
        w = grad(v, P) * unacq
        lam = float(torch.linalg.vector_norm(w))
        v = w
    return lam


# ---- periodised orthogonal DWT (Daubechies 4-tap), replacing Wavelab FWT2_PO ------------------
# MakeONFilter('Daubechies', 4), the 12 digits Wavelab carries. float64 so that casting to
# complex128 in the alignment check does not smuggle in single-precision rounding.
DEC_LO = torch.tensor([0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551],
                      dtype=torch.float64)
DEC_HI = torch.tensor([-0.129409522551, -0.224143868042, 0.836516303738, -0.482962913145],
                      dtype=torch.float64)


def _idx(n, taps, dev):
    k = torch.arange(n // 2, device=dev)
    return (2 * k[:, None] + torch.arange(taps, device=dev)[None, :]) % n


def _dwt1(x, lo, hi, ix):
    p = x[..., ix]                                             # (..., n/2, taps)
    return (p * lo).sum(-1), (p * hi).sum(-1)


def _idwt1(a, d, lo, hi, ix, n):
    contrib = (a.unsqueeze(-1) * lo + d.unsqueeze(-1) * hi).flatten(-2)      # (..., n/2 * taps)
    out = torch.zeros(*a.shape[:-1], n, dtype=a.dtype, device=a.device)
    return out.scatter_add_(-1, ix.reshape(-1).expand(contrib.shape), contrib)


def dwt2(x, levels, lo, hi):
    """Mallat layout in place: the LL block stays in the top-left corner."""
    out = x.clone()
    n = x.shape[-1]
    for _ in range(levels):
        ix = _idx(n, lo.shape[-1], x.device)
        sub = out[..., :n, :n]
        a, d = _dwt1(sub, lo, hi, ix)                           # along the last axis
        sub = torch.cat([a, d], dim=-1)
        a, d = _dwt1(sub.transpose(-1, -2), lo, hi, ix)
        out[..., :n, :n] = torch.cat([a, d], dim=-1).transpose(-1, -2)
        n //= 2
    return out


def idwt2(c, levels, lo, hi):
    out = c.clone()
    ns = [c.shape[-1] // 2 ** k for k in range(levels)][::-1]
    for n in ns:
        ix = _idx(n, lo.shape[-1], c.device)
        sub = out[..., :n, :n].transpose(-1, -2)
        sub = _idwt1(sub[..., :n // 2], sub[..., n // 2:], lo, hi, ix, n)
        sub = sub.transpose(-1, -2)
        out[..., :n, :n] = _idwt1(sub[..., :n // 2], sub[..., n // 2:], lo, hi, ix, n)
    return out


def joint_soft(c, t):
    """softThresh from pocsSPIRiT.m: joint sparsity across the coil axis (dim 1)."""
    a = (c.conj() * c).real.sum(1, keepdim=True).sqrt()
    return c / (a + 1e-30) * torch.clamp(a - t, min=0.0)


# ---- POCS ------------------------------------------------------------------------------------
def fista(data, P, unacq, iters, lam, levels, lo, hi, ss, step, trace=None):
    """L1-SPIRiT by damped proximal gradient with FISTA momentum:

        min_x  ||(G - I) x||^2 + lam * ||W F^-1 x||_1        s.t.  x = data on the mask

    Same objective as pocsSPIRiT.m, same wavelet prox (joint over coils). The difference is the
    smooth step: POCS takes x <- G x, which is Richardson with step 1 on an operator that is not a
    contraction here, and it DIVERGES on this data (verified: Lustig's own MATLAB diverges the same
    way, peak 5.97 at 10 iterations vs 277.7 at 60, and this port tracks it to 7e-11). A step of
    1 / ||(G-I)^H (G-I)|| converges to the same stationary point.
    """
    ny, nz = data.shape[-2:]
    y0, z0 = (ss - ny) // 2, (ss - nz) // 2
    x = data.clone()
    z, t = x, 1.0
    for _ in range(iters):
        prev = x
        g = grad(z, P)
        x = torch.where(unacq, z - step * g, data)
        if lam > 0:
            X = torch.zeros(*x.shape[:-2], ss, ss, dtype=x.dtype, device=x.device)
            X[..., y0:y0 + ny, z0:z0 + nz] = ifft2c(x)
            X = idwt2(joint_soft(dwt2(X, levels, lo, hi), lam * step), levels, lo, hi)
            x = torch.where(unacq, fft2c(X[..., y0:y0 + ny, z0:z0 + nz]), data)
        t2 = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
        z = x + ((t - 1) / t2) * (x - prev)
        t = t2
        if trace is not None:
            trace.append(float(torch.linalg.vector_norm(x - prev) / torch.linalg.vector_norm(x)))
    return x


def pocs(data, K, unacq, iters, lam, levels, lo, hi, ss, trace=None):
    """pocsSPIRiT.m. data (B, NC, ny, nz) with zeros off the mask; unacq = ~mask.
    trace: if a list is passed, the relative change per iteration is appended to it."""
    ny, nz = data.shape[-2:]
    y0, z0 = (ss - ny) // 2, (ss - nz) // 2
    x = data.clone()
    for _ in range(iters):
        if trace is not None:
            prev = x
        x = torch.where(unacq, x + gop(x, K), data)            # (x + (G-I)x) on the holes
        if lam > 0:
            X = torch.zeros(*x.shape[:-2], ss, ss, dtype=x.dtype, device=x.device)
            X[..., y0:y0 + ny, z0:z0 + nz] = ifft2c(x)
            X = idwt2(joint_soft(dwt2(X, levels, lo, hi), lam), levels, lo, hi)
            x = torch.where(unacq, fft2c(X[..., y0:y0 + ny, z0:z0 + nz]), data)
        if trace is not None:
            trace.append(float(torch.linalg.vector_norm(x - prev) / torch.linalg.vector_norm(x)))
    return x


def acs_box(m2):
    """largest fully sampled block centred on the k-space origin, grown one ring at a time in each
    axis independently. This mask's block is 24 x 16, not the 24 x 24 the generator's calib=24
    suggests: vdPoisMex's calib is not the same number of lines in ky and kz."""
    ny, nz = m2.shape
    y0, z0, h, w = ny // 2 - 1, nz // 2 - 1, 2, 2
    grew = True
    while grew:
        grew = False
        if y0 > 0 and y0 + h < ny and bool(m2[y0 - 1:y0 + h + 1, z0:z0 + w].all()):
            y0, h, grew = y0 - 1, h + 2, True
        if z0 > 0 and z0 + w < nz and bool(m2[y0:y0 + h, z0 - 1:z0 + w + 1].all()):
            z0, w, grew = z0 - 1, w + 2, True
    return h, w, y0, z0


# ======================== end of verbatim block ==================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True, choices=["pre", "post"])
    ap.add_argument("--mask", required=True, choices=["fit", "full"])
    ap.add_argument("--ksize", type=int, required=True)
    ap.add_argument("--lams", required=True, help="comma-separated wavelet thresholds")
    ap.add_argument("--tyk", type=float, default=0.01)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=12, help="readout positions per GPU batch")
    ap.add_argument("--gpu", type=int, required=True, help="-1 = CPU (smoke test only)")
    ap.add_argument("--smoke", default="", help="a:b -- only these readout positions, nothing written")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.gpu >= 0 else ""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ce_common as C

    dev = torch.device("cuda:0" if args.gpu >= 0 else "cpu")
    torch.zeros(1, device=dev)          # hold a CUDA context from the start (other schedulers read it as "card busy")
    if args.gpu < 0:
        torch.set_num_threads(16)
    log = lambda *a: print(*a, flush=True)
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    lams = [float(v) for v in args.lams.split(",")]
    tag = lambda lam: f"spirit_ks{args.ksize}_i{args.iters}_lam{lam:g}_{args.mask}_{args.scan}"
    todo = [lam for lam in lams if args.smoke or not (OUT / f"{tag(lam)}_meta.json").exists()]
    if not todo:
        log("nothing to do"); return

    t0 = time.time()
    y, S, m2 = C.load_scan(args.scan)                                  # CPU; y has zeros off the acquired mask
    fit_plane, ho_plane = C.load_split()
    scale = C.classical_scale(args.scan)
    NC, NX, NY, NZ = y.shape
    plane = fit_plane if args.mask == "fit" else m2
    ch, cw, cy0, cz0 = acs_box(m2)                                     # calibration block: FULL acquired data
    npatch, D = (ch - args.ksize + 1) * (cw - args.ksize + 1), NC * args.ksize ** 2
    log(f"[{args.scan}] coils={NC} grid={(NX, NY, NZ)} mask={args.mask} ({int(plane.sum())} lines) ACS={ch}x{cw} at "
        f"({cy0},{cz0}) ksize={args.ksize} lams={todo} tyk={args.tyk} iters={args.iters} scale={scale:.4e}")
    log(f"[{args.scan}] calibration matrix {npatch} x {D}{'  (under-determined, Tikhonov carries it)' if npatch < D else ''}")
    assert min(ch, cw) >= args.ksize + 2

    Sd = S.to(dev)
    yh = torch.empty((NC, NX, NY, NZ), dtype=torch.complex64, device=dev)      # hybrid space, normalised
    for i in range(NC):
        yh[i] = ifft1c(y[i].to(dev) / scale, 0)
    acs_all = yh[:, :, cy0:cy0 + ch, cz0:cz0 + cw].clone()                     # full-data ACS, every readout position
    pl = plane.to(dev)
    yh *= pl.to(yh.dtype)                                                      # data-consistency set only
    unacq = (~pl).view(1, 1, NY, NZ)
    ss = 1 << max(NY - 1, NZ - 1).bit_length()
    lo = DEC_LO.to(dev).to(torch.complex64)
    hi = DEC_HI.to(dev).to(torch.complex64)

    outs = {lam: torch.zeros((NX, NY, NZ), dtype=torch.complex64, device=dev) for lam in todo}
    t1 = time.time()
    steps, traces = [], {}
    lo_x, hi_x = (int(v) for v in args.smoke.split(":")) if args.smoke else (0, NX)
    for a in range(lo_x, hi_x, args.chunk):
        b = min(a + args.chunk, hi_x)
        data = yh[:, a:b].permute(1, 0, 2, 3).contiguous()                     # (B, NC, NY, NZ)
        acs = acs_all[:, a:b].permute(1, 0, 2, 3).contiguous()
        K = image_kernels(calib_kernels(build_AtA(acs, args.ksize), NC, args.ksize, args.tyk), NY, NZ)
        PM = normal_matrix(K); del K
        step = 0.9 / lipschitz(PM, unacq.to(PM.dtype), data.shape)
        steps.append(step)
        Sb = Sd[:, a:b].permute(1, 0, 2, 3).conj()
        for lam in todo:
            tr = [] if a == lo_x else None
            kf = fista(data, PM, unacq, args.iters, lam, args.levels, lo, hi, ss, step, trace=tr)
            if tr:
                traces[lam] = tr
                sel = tr[:3] + tr[len(tr) // 2 - 1:len(tr) // 2 + 1] + tr[-2:]
                log(f"  lam {lam:g} step={step:.4f} relative change (first 3 / middle 2 / last 2): "
                    f"{' '.join(f'{t:.2e}' for t in sel)}")
            outs[lam][a:b] = (Sb * ifft2c(kf)).sum(1)
            del kf
        del PM, data, acs, Sb
        if a % (args.chunk * 4) == 0:
            log(f"  readout {b}/{NX}  {time.time() - t1:.0f}s")
    secs = time.time() - t1
    if args.smoke:
        for lam in todo:
            v = outs[lam][lo_x:hi_x]
            log(f"smoke lam {lam:g}: |x| max {v.abs().max().item():.4f} mean {v.abs().mean().item():.5f} finite {bool(torch.isfinite(torch.view_as_real(v)).all())}")
        log(f"smoke done {secs:.0f}s"); return
    del yh, acs_all
    torch.cuda.empty_cache()

    yd = y.to(dev)
    for lam in todo:
        x = outs[lam] * scale
        r = C.kres_metrics(x, yd, Sd, {"full": m2.to(dev), "fit": fit_plane.to(dev), "ho": ho_plane.to(dev)})
        alpha = r["full"]["alpha"]
        np.save(OUT / f"{tag(lam)}_complex.npy", (alpha * x).cpu().numpy().astype(np.complex64))
        C.dump_json(dict(method="spirit", scan=args.scan, mask=args.mask, ksize=args.ksize, lam=lam, tyk=args.tyk,
                         iters=args.iters, levels=args.levels, solver="fista", acs=f"{ch}x{cw}", calib="full-data ACS",
                         npatch=npatch, ncols=D, scale=scale, alpha_re=float(np.real(alpha)), alpha_im=float(np.imag(alpha)),
                         kres=r["full"]["kres"], kres_fit=r["fit"]["kres"], kres_heldout=r["ho"]["kres"],
                         heldout_is_unseen=args.mask == "fit", step_median=float(np.median(steps)),
                         relchange_first_chunk=traces.get(lam, [])[-1:] or None,
                         seconds_all_lams=secs, torch=torch.__version__, gpu=torch.cuda.get_device_name(0)),
                    OUT / f"{tag(lam)}_meta.json")
        log(f"[{tag(lam)}] kres full {r['full']['kres']:.5f} fit {r['fit']['kres']:.5f} held-out {r['ho']['kres']:.5f}"
            f"  |alpha| {abs(alpha):.4f}")
    log(f"DONE {time.time() - t0:.0f}s (solver {secs:.0f}s for {len(todo)} lambdas)")


if __name__ == "__main__":
    main()
