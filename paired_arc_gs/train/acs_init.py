"""§13 Stage 1 — ACS initialization of the shared anatomy Gaussian set.

Spec §13-S1:
  1. reconstruct low-resolution complex images x_m^LR from each contrast's ACS region,
  2. build a contrast-normalized structural map per contrast:
         E_m = ||∇|x_m^LR||| / (median(||∇|x_m^LR|||) + ε)                (§13-S1)
  3. combine into a joint structural map:
         E_A = sqrt(E_1^2 + E_2^2)                                        (§13-S1)
  4. initialize shared Gaussian POSITIONS by importance-sampling E_A (structure-dense
     regions get more Gaussians), and SCALES from local structure.

The shared geometry initialized here is what §13-S2 then holds fixed while fitting the
response decoders.
"""
from __future__ import annotations

from typing import Tuple
import torch


def acs_lowres_image(kspace_multicoil: torch.Tensor, csm: torch.Tensor,
                     readout_axis: int, volume_shape: Tuple[int, int, int],
                     acs_lines: int, ifft3c) -> torch.Tensor:
    """Low-resolution complex image x^LR = Σ_c conj(S_c) · iFFT(ACS-masked k-space).

    Uses a centered ACS window on the two phase axes (full readout), matching the
    dataset's ACS convention. Returns a (D,H,W) complex volume.
    """
    acs_mask = torch.zeros_like(kspace_multicoil, dtype=torch.float32)
    index = [slice(None)] * kspace_multicoil.ndim
    for axis in range(len(volume_shape)):
        t_axis = axis + 1
        if axis == readout_axis:
            index[t_axis] = slice(None)
        else:
            size = volume_shape[axis]
            c = size // 2; half = acs_lines // 2
            index[t_axis] = slice(max(0, c - half), min(size, c + half))
    acs_mask[tuple(index)] = 1.0
    images_acs = ifft3c(kspace_multicoil * acs_mask)          # (C,D,H,W)
    return torch.sum(torch.conj(csm) * images_acs, dim=0)     # coil-combined (D,H,W)


def structural_map(x_lr: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """E_m = ||∇|x^LR||| / (median(||∇|x^LR|||) + ε)   (§13-S1). Returns (D,H,W) real >=0."""
    mag = x_lr.abs()
    grad_sq = torch.zeros_like(mag)
    for ax in range(mag.ndim):
        g = torch.zeros_like(mag)
        d = torch.diff(mag, dim=ax)
        sl = [slice(None)] * mag.ndim; sl[ax] = slice(1, None)
        g[tuple(sl)] = d
        grad_sq = grad_sq + g ** 2
    gnorm = torch.sqrt(grad_sq + eps)
    med = torch.median(gnorm)
    return gnorm / (med + eps)


def joint_structural_map(x1_lr: torch.Tensor, x2_lr: torch.Tensor) -> torch.Tensor:
    """E_A = sqrt(E_1^2 + E_2^2)   (§13-S1)."""
    E1 = structural_map(x1_lr)
    E2 = structural_map(x2_lr)
    return torch.sqrt(E1 ** 2 + E2 ** 2)


def init_gaussians_from_structure(E_A: torch.Tensor, num_points: int,
                                  base_scale_vox: float = 1.5,
                                  device=None, seed: int = 42
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Importance-sample `num_points` positions ∝ (E_A + uniform floor), map to [-1,1]^3.

    Returns (positions [N,3] in [-1,1], scales [N,3] in NORMALIZED units). A uniform floor
    is mixed in so low-structure regions still get some coverage (avoids holes). Scales are
    isotropic; `base_scale_vox` is a target extent in VOXELS, converted to the voxelizer's
    normalized convention (scale_vox = scale_norm · shape · 0.5, so scale_norm =
    base_scale_vox · 2 / shape). Later freed in §13-S4. CPU-safe multinomial sampling.
    """
    if device is None:
        device = E_A.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    D, H, W = E_A.shape
    prob = E_A.detach().flatten().float().cpu()
    prob = prob / prob.sum().clamp_min(1e-8)
    floor = torch.full_like(prob, 1.0 / prob.numel())
    prob = 0.7 * prob + 0.3 * floor
    prob = prob / prob.sum()
    idx = torch.multinomial(prob, num_points, replacement=True, generator=g)
    zi = (idx // (H * W)); rem = idx % (H * W); yi = rem // W; xi = rem % W
    # voxel centers -> normalized [-1,1] (matches voxelizer center=(p+1)*0.5*shape-0.5)
    px = (xi.float() + 0.5) / W * 2 - 1
    py = (yi.float() + 0.5) / H * 2 - 1
    pz = (zi.float() + 0.5) / D * 2 - 1
    positions = torch.stack([pz, py, px], dim=1).to(device)   # axis order (axis0,axis1,axis2)
    # scales in NORMALIZED units, per-axis so target ≈ base_scale_vox voxels on each axis
    scale_norm = torch.tensor([base_scale_vox * 2.0 / D,
                               base_scale_vox * 2.0 / H,
                               base_scale_vox * 2.0 / W], device=device)
    scales = scale_norm.unsqueeze(0).expand(num_points, 3).contiguous()
    # jitter positions within a voxel to avoid exact grid stacking
    jitter = (torch.rand(num_points, 3, generator=g).to(device) - 0.5) * (2.0 / max(D, H, W))
    positions = (positions + jitter).clamp(-1.0, 1.0)
    return positions, scales
