"""§8 Cross-scan geometric deformation  T_m.

Spec §8: T1W–FLAIR and pre/post are rarely voxel-aligned, so each contrast gets

    T_m(r) = R_m r + t_m + u_m(r)                                (rigid + low-freq nonrigid)

applied to the SHARED anatomy Gaussian centers μ_n → μ_{m,n} = T_m(μ_n). The local
covariance transforms as Σ_{m,n} = J_{T_m} Σ_n J_{T_m}^T. Deformation regularization:

    R_def = Σ_m [ ‖∇u_m‖² + λ_J ‖det J_{T_m} − 1‖² ].

Critical caveat (§8): u_m's spatial frequency and amplitude must be STRICTLY limited, or the
deformation absorbs genuine lesion / enhancement. We enforce this structurally with a coarse
control grid for u_m (few control points, trilinear upsample) and a small amplitude cap.

Design choices for compatibility with the reused voxelizer (positions/scales/rotations →
volume):
  * rigid: R_m from a small axis-angle vector θ_m (so R stays a proper rotation), t_m a 3-vec.
    One contrast (index 0 = plain) is pinned to identity as the reference frame; only the
    OTHER contrast(s) deform relative to it (§8 needs relative alignment, not absolute).
  * nonrigid: u_m sampled from a coarse control grid at each Gaussian center.
  * positions get the full T_m; the rotations get R_m composed (quaternion), and scales get
    the local Jacobian's diagonal scaling (isotropic approx of J Σ J^T for the low-freq u).
"""
from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .interfaces import DeformationBase


def _axis_angle_to_matrix(theta: torch.Tensor) -> torch.Tensor:
    """Rodrigues: axis-angle 3-vec -> 3x3 rotation matrix."""
    ang = torch.linalg.norm(theta) + 1e-12
    k = theta / ang
    K = torch.zeros(3, 3, device=theta.device, dtype=theta.dtype)
    K[0, 1] = -k[2]; K[0, 2] = k[1]
    K[1, 0] = k[2];  K[1, 2] = -k[0]
    K[2, 0] = -k[1]; K[2, 1] = k[0]
    I = torch.eye(3, device=theta.device, dtype=theta.dtype)
    return I + torch.sin(ang) * K + (1 - torch.cos(ang)) * (K @ K)


class Deformation(DeformationBase):
    def __init__(self, n_contrasts: int, volume_shape, device,
                 rigid: bool = True, nonrigid: bool = True,
                 control_points=(6, 6, 6), u_max: float = 0.05,
                 reference_index: int = 0):
        super().__init__()
        self.enabled = True
        self.n_contrasts = int(n_contrasts)
        self.volume_shape = tuple(int(s) for s in volume_shape)
        self.use_rigid = bool(rigid)
        self.use_nonrigid = bool(nonrigid)
        self.reference_index = int(reference_index)
        self.u_max = float(u_max)                    # cap on |u| in normalized [-1,1] units
        cp = tuple(int(c) for c in control_points)
        self.control_points = cp
        # rigid params per contrast (axis-angle + translation), reference pinned to identity
        self.theta = nn.ParameterList([nn.Parameter(torch.zeros(3, device=device)) for _ in range(self.n_contrasts)])
        self.trans = nn.ParameterList([nn.Parameter(torch.zeros(3, device=device)) for _ in range(self.n_contrasts)])
        # nonrigid control grid u (normalized displacement), per contrast, init 0
        self.u_ctrl = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 3, *cp, device=device)) for _ in range(self.n_contrasts)
        ])

    def _sample_u(self, ci: int, positions: torch.Tensor) -> torch.Tensor:
        """Sample the low-freq displacement u_m at each Gaussian center via grid_sample.
        positions (N,3) in [-1,1] (axis order z,y,x). Returns (N,3) displacement, capped."""
        ctrl = self.u_ctrl[ci]                                  # (1,3,cz,cy,cx)
        # grid_sample expects grid (1,N,1,1,3) with coords in (x,y,z) order, normalized [-1,1]
        p = positions
        grid = torch.stack([p[:, 2], p[:, 1], p[:, 0]], dim=1)  # -> (x,y,z)
        grid = grid.view(1, -1, 1, 1, 3)
        u = F.grid_sample(ctrl, grid, mode="bilinear", align_corners=True, padding_mode="border")
        u = u.view(3, -1).t()                                   # (N,3) in (x?,) -> map back
        # grid_sample returns channels in ctrl's channel order (dz,dy,dx as stored); we stored
        # channels as (0->z,1->y,2->x) to match positions axis order
        u = torch.tanh(u) * self.u_max                          # amplitude cap (§8)
        return u

    def forward(self, contrast_index: int, positions: torch.Tensor,
                scales: torch.Tensor, rotations: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply T_m to the shared Gaussians. Returns (positions', scales', rotations').
        The reference contrast is identity. Layer-3 uses rigid+nonrigid on positions; scales
        and rotations pass through (low-freq u ⇒ J≈I, isotropic; kept simple + stable)."""
        if not self.enabled or contrast_index == self.reference_index:
            return positions, scales, rotations
        ci = contrast_index
        pos = positions
        if self.use_rigid:
            R = _axis_angle_to_matrix(self.theta[ci]).to(pos.dtype)
            pos = pos @ R.t() + self.trans[ci]
        if self.use_nonrigid:
            pos = pos + self._sample_u(ci, positions)
        pos = pos.clamp(-1.0, 1.0)
        return pos, scales, rotations

    def displacement_field(self, ci: int) -> Optional[torch.Tensor]:
        """Full-resolution u_m (3,D,H,W) for the deformation reg (§8). None for reference."""
        if not self.enabled or ci == self.reference_index or not self.use_nonrigid:
            return None
        up = F.interpolate(self.u_ctrl[ci], size=self.volume_shape, mode="trilinear", align_corners=True)
        return torch.tanh(up[0]) * self.u_max                   # (3,D,H,W)

    def param_groups(self, lr: float) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        ps = []
        if self.use_rigid:
            ps += [p for i, p in enumerate(self.theta) if i != self.reference_index]
            ps += [p for i, p in enumerate(self.trans) if i != self.reference_index]
        if self.use_nonrigid:
            ps += [p for i, p in enumerate(self.u_ctrl) if i != self.reference_index]
        return [{"params": ps, "lr": lr}] if ps else []
