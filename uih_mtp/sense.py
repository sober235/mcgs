"""SENSE combination and CG-SENSE for DC-centred Cartesian k-space (coil, RO, PE, SPE) with coil maps of
the same layout. Transforms are centred and orthonormal: F(x) = fftshift(fftn(ifftshift(x))) / sqrt(N).
"""
from __future__ import annotations

import numpy as np
import torch


def fft3c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=(-3, -2, -1)), dim=(-3, -2, -1), norm="ortho"), dim=(-3, -2, -1))


def ifft3c(k: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(k, dim=(-3, -2, -1)), dim=(-3, -2, -1), norm="ortho"), dim=(-3, -2, -1))


def sense_combine(k: torch.Tensor, maps: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """x = sum_c conj(S_c) * ifft3c(k_c), coil by coil; returns (X, Y, Z) complex64 on the CPU."""
    device = device or k.device
    x = None
    for c in range(k.shape[0]):
        img = ifft3c(k[c].to(device=device, dtype=torch.complex64)) * maps[c].to(device=device, dtype=torch.complex64).conj()
        x = img if x is None else x + img
    return x.to("cpu", torch.complex64)


def cg_sense(y: torch.Tensor, maps: torch.Tensor, mask: torch.Tensor, iters: int = 30, lam: float = 0.0,
             device: torch.device | None = None, coil_chunk: int = 8, history: list | None = None) -> torch.Tensor:
    """Conjugate-gradient solution of (A^H A + lam I) x = A^H y with A = mask . F . S.

    ``y`` (C, X, Y, Z) undersampled k-space, ``maps`` (C, X, Y, Z), ``mask`` (Y, Z) or (X, Y, Z) bool.
    Returns x (X, Y, Z) complex64 on the CPU. ``history`` receives the relative residual per iteration.
    """
    device = device or y.device
    C = y.shape[0]
    m = mask.to(device)
    if m.ndim == 2:
        m = m[None]                                                        # broadcast along RO
    m = m.to(torch.complex64)
    S = maps.to(device=device, dtype=torch.complex64)

    def normal(x):                                                         # A^H A x + lam x, coil-chunked
        out = torch.zeros_like(x)
        for c0 in range(0, C, coil_chunk):
            Sc = S[c0:c0 + coil_chunk]
            kc = fft3c(Sc * x[None]) * m[None]
            out += (Sc.conj() * ifft3c(kc)).sum(0)
        return out + lam * x if lam else out

    b = torch.zeros(y.shape[1:], dtype=torch.complex64, device=device)
    for c0 in range(0, C, coil_chunk):
        b += (S[c0:c0 + coil_chunk].conj() * ifft3c(y[c0:c0 + coil_chunk].to(device=device, dtype=torch.complex64) * m[None])).sum(0)
    x = torch.zeros_like(b)
    r = b.clone(); p = r.clone()
    rr = torch.vdot(r.flatten(), r.flatten()).real
    b_norm = torch.sqrt(rr).item() + 1e-30
    for _ in range(iters):
        Ap = normal(p)
        alpha = rr / torch.vdot(p.flatten(), Ap.flatten()).real
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = torch.vdot(r.flatten(), r.flatten()).real
        if history is not None:
            history.append(float(torch.sqrt(rr_new).item() / b_norm))
        if rr_new.item() < 1e-20 * b_norm ** 2:
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x.to("cpu", torch.complex64)
