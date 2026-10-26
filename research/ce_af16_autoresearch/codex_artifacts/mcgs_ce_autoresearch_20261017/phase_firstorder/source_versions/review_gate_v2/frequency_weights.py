"""Training-free frequency flattening preserving acquired squared-weight mass."""
from __future__ import annotations

import math
from numbers import Real

import torch


def _rms(values):
    """Float64 RMS with scale-first squares, avoiding unnecessary overflow."""
    scale = values.max()
    return scale * (values / scale).square().mean().sqrt()


def flatten_acquired_rms(weights: torch.Tensor, mask: torch.Tensor, t: float) -> torch.Tensor:
    """Flatten original residual weights while keeping acquired mean weight².

    For acquired RMS c, interpolate v=(1-t)w+t*c and return v*c/RMS_m(v).
    The source loss multiplies the residual by these weights BEFORE squaring.
    Only original weights, a same-shape binary acquired mask and t enter this
    function. Inputs are never mutated; t=0 returns the original Tensor object.
    """
    if isinstance(t, bool) or not isinstance(t, Real) or not math.isfinite(t) or not 0. <= t <= 1.:
        raise ValueError('t must be a finite real scalar in [0,1]')
    if not isinstance(weights, torch.Tensor) or not weights.is_floating_point():
        raise ValueError('weights must be a real floating Tensor')
    if not isinstance(mask, torch.Tensor) or mask.is_complex():
        raise ValueError('mask must be a real binary Tensor')
    if weights.shape != mask.shape or weights.device != mask.device:
        raise ValueError('weights and mask must have identical shape and device')
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError('weights must be finite and nonnegative')
    if not bool(torch.isfinite(mask).all()) or not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError('mask must contain only exact binary values')
    acquired = mask.bool()
    if not bool(acquired.any()) or not bool((weights[acquired] > 0).any()):
        raise ValueError('mask must be nonempty with positive acquired RMS')
    # Do not even multiply by one or compute an RMS in the exact identity path.
    if t == 0.:
        return weights
    original = weights.to(torch.float64)
    c = _rms(original[acquired])
    if t == 1.:
        result = c.to(weights.dtype).expand_as(weights).clone()
    else:
        v = (1. - float(t)) * original + float(t) * c
        result = (v * (c / _rms(v[acquired]))).to(weights.dtype)
    if not bool(torch.isfinite(result).all()) or not bool((result[acquired] > 0).any()):
        raise ValueError('flattened weights are not representable with finite positive RMS in the source dtype')
    return result
