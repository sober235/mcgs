"""Wavelet-sparsity penalty on rendered complex volumes: the classical compressed-sensing prior
behind L1-wavelet ESPIRiT (BART ``pics -l1``), usable as a regulariser on the MC-GS render.

    R_wav(x) = (1 / N) * sum over detail sub-bands and coefficients of  rho(|W x|)

W is a 3-D discrete wavelet transform (ptwt, periodic boundary, ``levels`` levels). For a complex
volume W is real and linear, so W x = W Re(x) + i W Im(x) and |W x| is the modulus of that pair --
the same complex L1 that BART soft-thresholds. rho is Charbonnier (eps = 0 gives plain L1). The
approximation band is not penalised. N is the voxel count of one volume, so the value is a
per-voxel average and the terms of several volumes add up the way the TV terms do
(l_geo = TV_plain + TV_contrast).

Cycle spinning (BART's default): pass ``shifts`` -- a random offset in [0, 2^levels) per axis,
redrawn at every call -- and the volumes are circularly shifted before the transform, so the
penalty is not tied to a single dyadic grid.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import ptwt
import torch


def _rho(mag2: torch.Tensor, eps: float) -> torch.Tensor:
    """Charbonnier of a modulus given its square: sqrt(m^2 + eps^2) - eps for eps > 0, else m.
    The 1e-12 floor only keeps the gradient finite where a coefficient is exactly zero."""
    if eps > 0.0:
        return torch.sqrt(mag2 + eps * eps) - eps
    return torch.sqrt(mag2 + 1e-12)


def wavelet_l1(vols: Sequence[torch.Tensor], wavelet: str = "db2", levels: int = 3, eps: float = 0.0,
               shifts: Optional[Iterable[int]] = None, on: str = "complex",
               weights: Optional[Sequence[float]] = None) -> torch.Tensor:
    """Sum over ``vols`` (each (D, H, W), complex or real, same shape) of R_wav.

    on = "complex":   modulus of the complex wavelet coefficients (BART's penalty).
    on = "magnitude": wavelet L1 of |vol| (phase ignored, like the project's TV terms).
    weights: optional per-volume multipliers, giving sum_v w_v R_wav(v) -- a per-contrast strength,
             the way L1-ESPIRiT's lambda is chosen per contrast. None keeps the unweighted sum,
             computed exactly as before.
    All volumes go through one batched transform.
    """
    if on not in ("complex", "magnitude"):
        raise ValueError(f"on must be 'complex' or 'magnitude', got {on!r}")
    if weights is not None and len(weights) != len(vols):
        raise ValueError(f"need one weight per volume, got {len(weights)} for {len(vols)}")
    parts = []
    for v in vols:
        if on == "complex":
            vc = v if torch.is_complex(v) else torch.complex(v, torch.zeros_like(v))
            parts += [vc.real, vc.imag]
        else:
            parts.append(v.abs())
    x = torch.stack(parts, 0)                                     # (B, D, H, W)
    if shifts is not None:
        x = torch.roll(x, shifts=tuple(int(s) for s in shifts), dims=(1, 2, 3))
    coeffs = ptwt.wavedec3(x, wavelet, level=levels, mode="periodic")
    per = 2 if on == "complex" else 1
    nv = len(vols)
    if weights is None:
        total = x.new_zeros(())
        for detail in coeffs[1:]:                                 # coeffs[0] is the approximation band
            for band in detail.values():                          # (B, d1, d2, d3)
                b = band.reshape(nv, per, *band.shape[1:])
                total = total + _rho((b * b).sum(1), eps).sum()
        return total / vols[0].numel()
    per_vol = x.new_zeros(nv)
    for detail in coeffs[1:]:
        for band in detail.values():
            b = band.reshape(nv, per, *band.shape[1:])
            per_vol = per_vol + _rho((b * b).sum(1), eps).reshape(nv, -1).sum(1)
    w = torch.as_tensor([float(v) for v in weights], dtype=per_vol.dtype, device=per_vol.device)
    return (per_vol * w).sum() / vols[0].numel()


def wavelet_l21(vols: Sequence[torch.Tensor], wavelet: str = "db2", levels: int = 3, eps: float = 0.0,
                shifts: Optional[Iterable[int]] = None, on: str = "complex",
                weights: Optional[Sequence[float]] = None) -> torch.Tensor:
    """Joint (group) sparsity of several co-registered volumes -- the multi-contrast CS prior:

        R = (1 / N) * sum over detail coefficients of rho( sum_v w_v^2 |W x_v|^2 )

    i.e. an L2 norm across the volumes at each wavelet coefficient and an L1 norm over coefficients.
    A coefficient that is large in one contrast makes the same coefficient cheap in the other, so
    edges both contrasts share are kept together while isolated coefficients (noise, aliasing) are
    shrunk as before. With one volume zero it is that weight times the plain L1 of the other
    (eps = 0), and it never exceeds the separate weighted sum of ``wavelet_l1``.
    Arguments as in ``wavelet_l1``; weights default to one per volume.
    """
    if on not in ("complex", "magnitude"):
        raise ValueError(f"on must be 'complex' or 'magnitude', got {on!r}")
    if weights is None:
        weights = [1.0] * len(vols)
    if len(weights) != len(vols):
        raise ValueError(f"need one weight per volume, got {len(weights)} for {len(vols)}")
    parts = []
    for v in vols:
        if on == "complex":
            vc = v if torch.is_complex(v) else torch.complex(v, torch.zeros_like(v))
            parts += [vc.real, vc.imag]
        else:
            parts.append(v.abs())
    x = torch.stack(parts, 0)
    if shifts is not None:
        x = torch.roll(x, shifts=tuple(int(s) for s in shifts), dims=(1, 2, 3))
    coeffs = ptwt.wavedec3(x, wavelet, level=levels, mode="periodic")
    per = 2 if on == "complex" else 1
    nv = len(vols)
    w2 = torch.as_tensor([float(w) ** 2 for w in weights], dtype=x.dtype, device=x.device).view(nv, 1, 1, 1)
    total = x.new_zeros(())
    for detail in coeffs[1:]:
        for band in detail.values():
            b = band.reshape(nv, per, *band.shape[1:])
            total = total + _rho(((b * b).sum(1) * w2).sum(0), eps).sum()
    return total / vols[0].numel()


def wavelet_l1_guided(vol: torch.Tensor, guide: torch.Tensor, wavelet: str = "db2", levels: int = 3, eps: float = 0.0,
                      shifts: Optional[Iterable[int]] = None, on: str = "complex", q: float = 0.85,
                      w_floor: float = 0.1) -> torch.Tensor:
    """One-way guided wavelet sparsity: R_wav of ``vol`` with each detail coefficient weighted by

        w = clamp(1 / (1 + (|W guide| / kappa)^2), w_floor, 1),   kappa = q-quantile of the guide's detail magnitudes

    (guide detached, same transform and cycle-spin shift as ``vol``). The weight only drops where the guide has a strong
    coefficient, so the guide's structure can relax ``vol``'s shrinkage there and nowhere raises it; the guide's own
    penalty is untouched (unlike ``wavelet_l21``). A flat guide gives exactly ``wavelet_l1([vol])``; kappa has a floor of
    1e-6 * max|guide| so round-off in a flat guide's coefficients stays harmless, and w is invariant to the guide's scale.
    """
    if on not in ("complex", "magnitude"):
        raise ValueError(f"on must be 'complex' or 'magnitude', got {on!r}")

    def parts(v):
        if on == "complex":
            vc = v if torch.is_complex(v) else torch.complex(v, torch.zeros_like(v))
            return [vc.real, vc.imag]
        return [v.abs()]

    gd = guide.detach()
    x = torch.stack(parts(vol) + parts(gd), 0)
    if shifts is not None:
        x = torch.roll(x, shifts=tuple(int(s) for s in shifts), dims=(1, 2, 3))
    coeffs = ptwt.wavedec3(x, wavelet, level=levels, mode="periodic")
    per = 2 if on == "complex" else 1
    bands = [band for detail in coeffs[1:] for band in detail.values()]
    with torch.no_grad():
        mags = [(b[per:].detach() * b[per:].detach()).sum(0).sqrt() for b in bands]
        flat = torch.cat([m.reshape(-1) for m in mags])
        sub = flat[:: max(1, flat.numel() // 1_000_000)]
        kappa = torch.maximum(torch.quantile(sub.double(), q).to(flat.dtype), 1e-6 * gd.abs().max().to(flat.dtype))
    total = x.new_zeros(())
    for b, m in zip(bands, mags):
        w = torch.clamp(1.0 / (1.0 + (m / kappa) ** 2), min=w_floor, max=1.0)
        total = total + (w * _rho((b[:per] * b[:per]).sum(0), eps)).sum()
    return total / vol.numel()
