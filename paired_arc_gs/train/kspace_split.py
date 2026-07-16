"""P1 self-supervised frequency extrapolation — Θ/Λ k-space split (design §4).

Hold out a ρ fraction of the SAMPLED mid/high-frequency phase-plane locations as Λ
(the extrapolation target); the ACS disc and the remaining sampled locations form Θ
(the fit set). The readout axis is fully sampled and excluded from the radius.
"""
from __future__ import annotations
from typing import Tuple
import torch


def frequency_holdout_split(mask: torch.Tensor, phase_axes: Tuple[int, int],
                            r_acs: float, rho: float,
                            generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    dev, dt = mask.device, mask.dtype
    ax0, ax1 = phase_axes
    n0, n1 = mask.shape[ax0], mask.shape[ax1]
    c0, c1 = n0 // 2, n1 // 2
    s0 = [1, 1, 1]; s0[ax0] = n0
    s1 = [1, 1, 1]; s1[ax1] = n1
    coord0 = (torch.arange(n0) - c0).float().view(s0)
    coord1 = (torch.arange(n1) - c1).float().view(s1)
    r = torch.sqrt(coord0 ** 2 + coord1 ** 2).expand(mask.shape)          # (D,H,W) on CPU
    sampled_cpu = (mask.detach().cpu() > 0)
    highfreq = sampled_cpu & (r >= float(r_acs))
    # one random value PER PHASE-PLANE LOCATION, broadcast along the (fully sampled)
    # readout axis, so an ENTIRE readout line is held out together — Λ mirrors the real
    # missing-data structure (a phase location is fully sampled or fully absent).
    plane_shape = [1, 1, 1]; plane_shape[ax0] = n0; plane_shape[ax1] = n1
    u = torch.rand(plane_shape, generator=generator).expand(mask.shape)     # CPU draw (reproducible)
    lam_cpu = highfreq & (u < float(rho))
    mask_lambda = lam_cpu.to(dtype=dt, device=dev)
    mask_theta = (sampled_cpu & (~lam_cpu)).to(dtype=dt, device=dev)
    return mask_theta, mask_lambda
