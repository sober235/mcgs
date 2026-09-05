"""Known-zero k-space region Z and coil-noise prewhitening W — the Phase 1 physics levers
(docs/superpowers/specs/2026-09-05-af12-to-af10-quality-design.md §4).

Everything here is estimated from the ACQUIRED samples of one scan (kspace already multiplied by the
mask). Nothing reads the full k-space or the ground truth. Pure functions; the trainer only wires the
outputs in. All rule constants are fixed a priori (spec §4.1) and are NOT tuning knobs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

RING_WIDTH = 0.05        # ring / band width in normalised radius
MARGIN = 1.5             # a ring is "at the floor" if E_r <= MARGIN * n0
CLIFF_RATIO = 5.0        # the energy just inside Z must exceed the Z energy by this factor
CLIFF_SPAN = 0.15        # ... within this radial distance
MIN_RINGS = 2            # Z must span at least this many rings
MIN_RING_POINTS = 30     # PE rings with fewer sampled points merge into the inner neighbour
MIN_NOISE_SAMPLES = 2000 # per-coil samples needed for a noise covariance


@dataclass
class KnownZeroInfo:
    dc: tuple
    ro_axis: int
    rho_c: Optional[float]
    kro_c: Optional[float]
    n0_pe: float
    n0_ro: float
    frac_z: float
    frac_sampled_in_z: float
    pe_profile: list
    ro_profile: list


def _profile(e_mean: np.ndarray, coord: np.ndarray, sampled: np.ndarray, width: float, min_points: int) -> list:
    """[(coord_lo, median energy, n_points)] over sampled entries, ring by ring from the centre outward.
    A ring with fewer than `min_points` sampled entries is merged into the previous (inner) ring."""
    if not sampled.any():
        return []
    edges = np.arange(0.0, float(coord[sampled].max()) + width, width)
    out: list = []
    for lo in edges:
        sel = sampled & (coord >= lo) & (coord < lo + width)
        vals = e_mean[sel]
        if vals.size == 0:
            continue
        if vals.size < min_points and out:
            out[-1][1] = np.concatenate([out[-1][1], vals])
        else:
            out.append([float(lo), vals])
    return [(lo, float(np.median(v)), int(v.size)) for lo, v in out]


def _cutoff(profile: list, *, margin: float, cliff_ratio: float, cliff_span: float, min_rings: int):
    """The cliff rule (spec §4.1 step 4). Returns (cutoff or None, n0)."""
    if len(profile) < min_rings + 1:
        return None, (float(min(p[1] for p in profile)) if profile else 0.0)
    lo = np.array([p[0] for p in profile]); E = np.array([p[1] for p in profile])
    n0 = float(E.min())
    thr = margin * n0
    k = len(E)
    while k > 0 and E[k - 1] <= thr:          # maximal suffix at the floor, walking inward from the outermost ring
        k -= 1
    if len(E) - k < min_rings or k == 0:
        return None, n0
    rho_in = float(lo[k])
    before = E[(lo >= rho_in - cliff_span) & (lo < rho_in)]
    if before.size == 0:
        return None, n0
    e_z = float(E[k:].max())
    cliff = (float(before.max()) >= cliff_ratio * e_z) if e_z > 0 else (float(before.max()) > 0)
    if not cliff:
        return None, n0
    return rho_in, n0


def pe_radius_map(shape: tuple, dc: tuple, ro_axis: int, device) -> torch.Tensor:
    """Normalised PE-plane radius about `dc`, broadcast along the readout axis. (D,H,W) float32."""
    pe = [a for a in range(3) if a != ro_axis]
    n0, n1 = shape[pe[0]], shape[pe[1]]
    i = torch.arange(n0, device=device, dtype=torch.float32)
    j = torch.arange(n1, device=device, dtype=torch.float32)
    rho2 = torch.sqrt(((i - dc[pe[0]]) / (n0 / 2)).view(-1, 1) ** 2 + ((j - dc[pe[1]]) / (n1 / 2)).view(1, -1) ** 2)
    rho = rho2.unsqueeze(-1).expand(n0, n1, shape[ro_axis])
    return torch.movedim(rho, -1, ro_axis).contiguous()


def estimate_known_zero(kspace: torch.Tensor, mask: torch.Tensor, ro_axis: int, *,
                        ring_width: float = RING_WIDTH, margin: float = MARGIN, cliff_ratio: float = CLIFF_RATIO,
                        cliff_span: float = CLIFF_SPAN, min_rings: int = MIN_RINGS,
                        min_points: int = MIN_RING_POINTS):
    """Z from acquired samples only. kspace (C,D,H,W) complex, already masked; mask (D,H,W) in {0,1}."""
    assert kspace.ndim == 4 and mask.ndim == 3 and tuple(kspace.shape[1:]) == tuple(mask.shape)
    dev = kspace.device
    Cn = kspace.shape[0]
    m = mask > 0.5
    e = (kspace.real.square() + kspace.imag.square()).sum(0) * m            # sampled energy, (D,H,W)
    e_r = torch.movedim(e, ro_axis, -1); m_r = torch.movedim(m, ro_axis, -1)  # (P0,P1,R)
    P0, P1, Rn = e_r.shape
    # --- PE plane: DC = ACS energy peak of the sampled data
    e_pe = e_r.sum(-1)
    flat = int(torch.argmax(e_pe)); dc0, dc1 = flat // P1, flat % P1
    n_ro = m_r.sum(-1)                                                     # sampled readout entries per PE point
    sampled_pe = (n_ro > 0).cpu().numpy()
    e_mean_pe = (e_pe / (Cn * n_ro.clamp_min(1))).cpu().numpy()           # per (coil, sample)
    ii, jj = np.meshgrid(np.arange(P0), np.arange(P1), indexing="ij")
    rho = np.sqrt(((ii - dc0) / (P0 / 2)) ** 2 + ((jj - dc1) / (P1 / 2)) ** 2)
    pe_profile = _profile(e_mean_pe, rho, sampled_pe, ring_width, min_points)
    rho_c, n0_pe = _cutoff(pe_profile, margin=margin, cliff_ratio=cliff_ratio, cliff_span=cliff_span, min_rings=min_rings)
    # --- readout axis: each index already averages thousands of PE samples, so no merging (min_points=1)
    e_ro = e_r.sum((0, 1))
    n_pe = m_r.sum((0, 1))
    dc2 = int(torch.argmax(e_ro))
    e_mean_ro = (e_ro / (Cn * n_pe.clamp_min(1))).cpu().numpy()
    kro = np.abs(np.arange(Rn) - dc2) / (Rn / 2)
    ro_profile = _profile(e_mean_ro, kro, (n_pe > 0).cpu().numpy(), ring_width, 1)
    kro_c, n0_ro = _cutoff(ro_profile, margin=margin, cliff_ratio=cliff_ratio, cliff_span=cliff_span, min_rings=min_rings)
    # --- assemble Z
    Z_r = torch.zeros((P0, P1, Rn), dtype=torch.bool, device=dev)
    if rho_c is not None:
        Z_r |= torch.from_numpy(rho >= rho_c).to(dev)[:, :, None]
    if kro_c is not None:
        Z_r |= torch.from_numpy(kro >= kro_c).to(dev)[None, None, :]
    Z = torch.movedim(Z_r, -1, ro_axis).contiguous()
    dc = [0, 0, 0]
    pe = [a for a in range(3) if a != ro_axis]
    dc[pe[0]], dc[pe[1]], dc[ro_axis] = int(dc0), int(dc1), int(dc2)
    info = KnownZeroInfo(dc=tuple(dc), ro_axis=int(ro_axis), rho_c=rho_c, kro_c=kro_c,
                         n0_pe=float(n0_pe), n0_ro=float(n0_ro), frac_z=float(Z.float().mean()),
                         frac_sampled_in_z=float((Z & m).sum().item() / max(m.sum().item(), 1)),
                         pe_profile=pe_profile, ro_profile=ro_profile)
    return Z, info
