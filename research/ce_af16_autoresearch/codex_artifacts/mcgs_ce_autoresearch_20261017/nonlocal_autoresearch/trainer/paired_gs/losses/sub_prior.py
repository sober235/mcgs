"""Subtraction (enhancement-sparsity) prior for a pre / post-contrast pair (CE autoresearch phase 6, 2026-10-10).

Physics. The plain (pre) and contrast (post) channels of this study are the same T1 GRE before and after
gadolinium. Outside enhancing tissue the two magnitudes agree up to one global scale s, so the enhancement

    d = |x_post| - s * |x_pre|

is sparse: large in vessels, plexus and lesion rim, small in cortex, ~0 in white matter, CSF and background.
A single-contrast reconstruction cannot use this; the joint one can. Every cross-contrast term tried before
(common edge, vector TV, directional TV, edge gate) couples only edge LOCATIONS; this one couples intensities.

Penalty on d over the head support, in data units of the contrast channel (so lambda is on the scale of
lambda_geo), delta = the transition between the quadratic and the flat / linear regime:

    cauchy : (delta / 2) * log(1 + (d / delta)^2)   quadratic for |d| << delta, logarithmic beyond: real
                                                      enhancement (|d| >> delta) is barely shrunk
    charb  : sqrt(d^2 + delta^2) - delta              L1-like: shrinks every enhancement by about lambda
    neg    : charb on min(d, 0)                       only 'enhancement is non-negative'

s and delta are re-estimated from the detached renders (robust medians over the bright head voxels), so they
need no tuning; sub_delta_rel sets delta relative to the median tissue magnitude of the contrast channel.
"""
from __future__ import annotations

import torch

_SUB_N = 2_000_000          # quantile / median on at most this many voxels (torch.quantile caps at 16M)


def _sub(t: torch.Tensor) -> torch.Tensor:
    return t[:: max(1, t.numel() // _SUB_N)]


def robust_scale(mag_p: torch.Tensor, mag_c: torch.Tensor, support: torch.Tensor,
                 rel_floor: float = 0.1) -> tuple[float, float]:
    """(s, tissue): s = median(|x_c| / |x_p|) and tissue = median(|x_c|) over the support voxels where |x_p| is
    above rel_floor x its P99 inside the support (dark voxels give unstable ratios). Both detached floats."""
    p = mag_p[support].float()
    c = mag_c[support].float()
    thr = rel_floor * torch.quantile(_sub(p), 0.99)
    keep = p > thr
    r = c[keep] / p[keep]
    return float(torch.median(_sub(r))), float(torch.median(_sub(c[keep])))


def penalty(d: torch.Tensor, delta: float, mode: str) -> torch.Tensor:
    if mode == "cauchy":
        return 0.5 * delta * torch.log1p((d / delta) ** 2)
    if mode == "charb":
        return torch.sqrt(d * d + delta * delta) - delta
    if mode == "neg":
        dn = torch.clamp(d, max=0.0)
        return torch.sqrt(dn * dn + delta * delta) - delta
    raise ValueError(f"sub_mode must be cauchy, charb or neg, got {mode!r}")


def sub_loss(vol_p: torch.Tensor, vol_c: torch.Tensor, s: float, delta: float, support: torch.Tensor,
             mode: str = "cauchy", detach_plain: bool = False) -> torch.Tensor:
    """sum over the support of penalty(|x_c| - s |x_p|), divided by the number of voxels of the volume (the same
    normalisation as the TV terms, which are means over the volume)."""
    mp = vol_p.abs()
    if detach_plain:
        mp = mp.detach()
    d = (vol_c.abs() - s * mp)[support]
    return penalty(d, delta, mode).sum() / vol_c.numel()
