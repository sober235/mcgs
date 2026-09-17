#!/usr/bin/env python3
"""recon_spirit.py -- L1-SPIRiT (Lustig & Pauly, MRM 2010) on this topic's data and gauge.

A k-space parallel-imaging arm: unlike SENSE/ESPIRiT the unknown is the missing k-space, and the
prior is the SPIRiT self-consistency x = G x, with G calibrated from the fully sampled ACS
(24 x 16 here -- measured by acs_box(), not the generator's nominal calib=24). GRAPPA proper is not applicable to this mask (variable-density Poisson disc: every hole has
a different neighbourhood, and a GRAPPA kernel is fitted per fixed geometry). SPIRiT is the
k-space method that is defined for arbitrary Cartesian sampling.

Transcribed from the reference MATLAB implementation in /data0/congcong/code/ESPIRiT
(Lustig's ESPIRiT/SPIRiT toolbox), function by function:
  SPIRiT_code/calibSPIRiT.m + utils/dat2AtA.m + utils/calibrate.m   -> calib_kernels()
  @SPIRiT/SPIRiT.m ('fft' method) + @SPIRiT/mtimes.m                -> image_kernels() / gop()
  SPIRiT_code/pocsSPIRiT.m (wavWeight > 0 branch)                   -> pocs()
The only deliberate departure is the wavelet: Wavelab's FWT2_PO is replaced by a periodised
orthogonal DWT written here with the same filter (Daubechies 4-tap = db2) and the same number of
levels. The soft threshold is applied to every coefficient with one threshold, so the coefficient
LAYOUT cannot change the result -- only the filter and orthogonality can, and both match.
scripts/spirit_align_check.py verifies kernels and POCS output against the MATLAB original.

The readout is fully sampled and the mask is constant along it, so the problem factorises into NX
independent 2-D problems in hybrid (x, ky, kz) space -- exact, not an approximation, the same
factorisation recon_zsssdu.py uses. Calibration is per readout slice, as the coil sensitivities
vary along x.

The completed multi-coil k-space is combined with the topic's ESPIRiT map (sum_c conj(S_c) . ) --
the same operator the ground truth is built with -- so the result lands in the shared gauge.
RSS would not: it loses the phase and lands on a different scale.

usage: recon_spirit.py --scan pre --gpu N [--lam 0.002 --ksize 3 --iters 150 --suffix _ks3_lam0.002]
Writes recon/spirit<suffix>_<c>_complex.npy and logs/spirit_<c><suffix>.json.
"""
import argparse, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--scan", dest="contrast", required=True, choices=["pre", "post"])
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--lam", type=float, default=0.002, help="wavelet joint soft-threshold (wavWeight)")
ap.add_argument("--ksize", type=int, default=5, help="SPIRiT kernel side")
ap.add_argument("--tyk", type=float, default=0.01, help="Tikhonov in the calibration (CalibTyk)")
ap.add_argument("--iters", type=int, default=150, help="solver iterations")
ap.add_argument("--solver", default="fista", choices=["fista", "pocs"],
                help="fista: damped proximal gradient on the L1-SPIRiT objective (converges here). "
                     "pocs: pocsSPIRiT.m verbatim -- kept for the alignment check, it diverges on this data")
ap.add_argument("--levels", type=int, default=4, help="wavelet decomposition levels")
ap.add_argument("--chunk", type=int, default=8, help="readout slices per GPU batch")
ap.add_argument("--suffix", default="")
ap.add_argument("--calib", default="acquired", choices=["acquired", "full24"],
                help="where the SPIRiT kernel is calibrated from. acquired: the block this mask actually "
                     "samples (24 x 16 here). full24: the centre 24 x 24 of the FULL k-space, which is the "
                     "region `bart ecalib -r 24` took the topic's ESPIRiT maps from -- so every other arm, "
                     "and the ground truth itself, already rests on that block. Reported both ways.")
ap.add_argument("--save", type=int, default=1, help="0: score only, do not write the volume")
args = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1t2_common import Pair, R, fft2c, ifft2c, ifft1c                # noqa: E402

dev = torch.device("cuda:0")
log = lambda *a: print(*a, flush=True)


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


def main():
    t0 = time.time()
    P = Pair(args.contrast, dev)
    NC, (NX, NY, NZ) = P.NC, P.G
    ch, cw, cy0, cz0 = acs_box(P.m2)
    log(f"[{args.contrast}] coils={NC} grid={P.G} ACS={ch}x{cw} at ({cy0},{cz0}) ksize={args.ksize} "
        f"lam={args.lam} tyk={args.tyk} iters={args.iters} scale={P.scale:.4e}")
    assert min(ch, cw) >= args.ksize + 2, f"ACS {ch}x{cw} too small for a {args.ksize} kernel"
    npatch = (ch - args.ksize + 1) * (cw - args.ksize + 1)
    D = NC * args.ksize ** 2
    log(f"[{args.contrast}] calibration matrix {npatch} x {D}"
        f"{'  (under-determined, Tikhonov carries it)' if npatch < D else ''}")

    yh = torch.empty_like(P.y)                                  # hybrid space: ifft along readout
    for i in range(0, NC, P.chunk):
        yh[i:i + P.chunk] = ifft1c(P.y[i:i + P.chunk], -3)
    if args.calib == "full24":                                  # the ecalib -r 24 block, see --calib
        ch = cw = 24
        cy0, cz0 = NY // 2 - 12, NZ // 2 - 12
        ca = torch.empty_like(P.y_full[:, :, cy0:cy0 + ch, cz0:cz0 + cw])
        for i in range(0, NC, P.chunk):
            ca[i:i + P.chunk] = ifft1c(P.y_full[i:i + P.chunk], -3)[:, :, cy0:cy0 + ch, cz0:cz0 + cw]
        npatch = (ch - args.ksize + 1) * (cw - args.ksize + 1)
        log(f"[{args.contrast}] calib from the FULL k-space centre {ch}x{cw}; matrix {npatch} x {D}")
    unacq = (~P.m2).view(1, 1, NY, NZ)
    ss = 1 << max(NY - 1, NZ - 1).bit_length()
    lo = DEC_LO.to(dev).to(torch.complex64)
    hi = DEC_HI.to(dev).to(torch.complex64)

    out = torch.zeros(P.G, dtype=torch.complex64, device=dev)
    t1 = time.time()
    for a in range(0, NX, args.chunk):
        b = min(a + args.chunk, NX)
        data = yh[:, a:b].permute(1, 0, 2, 3).contiguous()      # (B, NC, NY, NZ)
        acs = (ca[:, a:b].permute(1, 0, 2, 3).contiguous() if args.calib == "full24"
               else data[..., cy0:cy0 + ch, cz0:cz0 + cw])
        K = image_kernels(calib_kernels(build_AtA(acs, args.ksize), NC, args.ksize, args.tyk), NY, NZ)
        tr = [] if a == 0 else None
        if args.solver == "fista":
            PM = normal_matrix(K); del K            # PM, not P: P is the Pair
            step = 0.9 / lipschitz(PM, unacq.to(PM.dtype), data.shape)
            kf = fista(data, PM, unacq, args.iters, args.lam, args.levels, lo, hi, ss, step, trace=tr)
            del PM
        else:
            step = float("nan")
            kf = pocs(data, K, unacq, args.iters, args.lam, args.levels, lo, hi, ss, trace=tr)
        if tr:
            sel = tr[:3] + tr[len(tr) // 2 - 1:len(tr) // 2 + 1] + tr[-2:]
            log(f"  step={step:.4f}  relative change (first 3 / middle 2 / last 2): "
                f"{' '.join(f'{t:.2e}' for t in sel)}")
        out[a:b] = (P.S[:, a:b].permute(1, 0, 2, 3).conj() * ifft2c(kf)).sum(1)
        del kf
        if a % (args.chunk * 8) == 0:
            log(f"  readout {b}/{NX}  {time.time() - t1:.0f}s")
    secs = time.time() - t1

    row = P.score(out, "spirit" + args.suffix, seconds=secs, lam=args.lam, ksize=args.ksize, solver=args.solver,
                  tyk=args.tyk, iters=args.iters, levels=args.levels, acs=f"{ch}x{cw}", calib=args.calib,
                  npatch=npatch, ncols=D)
    if args.save:
        np.save(R / f"recon/spirit{args.suffix}_{args.contrast}_complex.npy",
                out.cpu().numpy().astype(np.complex64))
    p = R / f"logs/spirit_{args.contrast}{args.suffix}.json"
    json.dump({"contrast": args.contrast, "rows": [row], **{k: getattr(args, k) for k in
              ("lam", "ksize", "tyk", "iters", "levels", "solver", "calib")}}, open(p, "w"), indent=2)
    log(f"[{args.contrast}] spirit{args.suffix} psnr={row['psnr']:.3f} ssim={row['ssim']:.4f} "
        f"head_psnr={row['head_psnr']:.3f} head_ssim={row['head_ssim']:.4f} "
        f"head_hfen={row['head_hfen']:.4f} sharp={row['sharp']:.3f} ({secs:.0f} s, total {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
