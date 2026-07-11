"""§7 Contrast-specific change branch  G_{Δ,m}  +  §10 measurement-audited routing.

§7: a per-contrast set of change Gaussians

    G_{Δ,m} = { μ_{Δ,m,l}, Σ_{Δ,m,l}, α_{Δ,m,l}, p_{Δ,m,l} }_{l=1..N_{Δ,m}}

that represents ONLY signal not explainable by the shared anatomy + conditioned response
(FLAIR-only white-matter hyperintensity; post-contrast enhancement). The full image is

    x_m = x_m^A + x_m^Δ,   x_m^Δ = V(G_{Δ,m}).

§10: the danger of a free private branch is that it fits undersampling aliasing. So change
Gaussians are NOT accepted from an image residual — they must pass a measurement audit:
  §10.2 propose candidates from the FIT-set k-space residual backprojection e_m,
  §10.3 accept a candidate only if it REDUCES error on the disjoint AUDIT set (ΔE > τ_m).

This module owns the change-Gaussian parameters and renders x_m^Δ. Proposal and audit are
driven by the trainer (which holds the data, masks, and renderer) via the helper functions
at the bottom. The branch starts EMPTY (no change Gaussians) — birth happens at stage
s5_change_birth.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn

from .interfaces import ChangeBranchBase


class ChangeBranch(ChangeBranchBase):
    """Per-contrast change Gaussians. Geometry mirrors the anatomy model's conventions
    (positions in [-1,1], scales in normalized units, quaternion rotations); density is a
    learnable complex amplitude per Gaussian. Confidence p is stored as a logit."""

    def __init__(self, n_contrasts: int, volume_shape, device,
                 max_per_contrast: int = 20000, init_scale_vox: float = 1.5):
        super().__init__()
        self.enabled = True
        self.n_contrasts = int(n_contrasts)
        self.volume_shape = tuple(volume_shape)
        self.device = device
        self.max_per_contrast = int(max_per_contrast)
        D, H, W = self.volume_shape
        self._base_scale = torch.tensor([init_scale_vox * 2.0 / D,
                                         init_scale_vox * 2.0 / H,
                                         init_scale_vox * 2.0 / W], device=device)
        # per-contrast parameter lists; each starts empty (0 change Gaussians)
        self.positions = nn.ParameterList([nn.Parameter(torch.zeros(0, 3, device=device)) for _ in range(self.n_contrasts)])
        self.scales    = nn.ParameterList([nn.Parameter(torch.zeros(0, 3, device=device)) for _ in range(self.n_contrasts)])
        self.rotations = nn.ParameterList([nn.Parameter(torch.zeros(0, 4, device=device)) for _ in range(self.n_contrasts)])
        self.density   = nn.ParameterList([nn.Parameter(torch.zeros(0, 2, device=device)) for _ in range(self.n_contrasts)])
        self.pi_logit  = nn.ParameterList([nn.Parameter(torch.zeros(0, device=device)) for _ in range(self.n_contrasts)])

    def count(self, ci: int) -> int:
        return self.positions[ci].shape[0]

    def get_scale_values(self, ci: int) -> torch.Tensor:
        return torch.exp(self.scales[ci])

    def get_density(self, ci: int) -> torch.Tensor:
        t = self.density[ci]
        return torch.complex(t[:, 0], t[:, 1])

    def confidence(self, ci: int) -> torch.Tensor:
        return torch.sigmoid(self.pi_logit[ci])

    @torch.no_grad()
    def add_gaussians(self, ci: int, positions: torch.Tensor, scales_norm: Optional[torch.Tensor] = None,
                      density: Optional[torch.Tensor] = None):
        """Append k new change Gaussians to contrast ci (used by the birth step). positions
        (k,3) in [-1,1]; scales_norm (k,3) normalized (default isotropic base); density (k,2)
        real/imag (default near-zero)."""
        k = positions.shape[0]
        if k == 0:
            return
        cap = self.max_per_contrast - self.count(ci)
        if cap <= 0:
            return
        if k > cap:
            positions = positions[:cap]; k = cap
            if scales_norm is not None: scales_norm = scales_norm[:cap]
            if density is not None: density = density[:cap]
        if scales_norm is None:
            scales_norm = self._base_scale.unsqueeze(0).expand(k, 3).contiguous()
        log_scales = torch.log(torch.clamp(scales_norm, min=1e-8))
        rot = torch.zeros(k, 4, device=self.device); rot[:, 0] = 1.0
        dens = density if density is not None else torch.randn(k, 2, device=self.device) * 1e-3
        pil = torch.zeros(k, device=self.device)
        def cat(param_list, idx, new):
            old = param_list[idx].data
            param_list[idx] = nn.Parameter(torch.cat([old, new.to(self.device)], dim=0))
        cat(self.positions, ci, positions.detach())
        cat(self.scales, ci, log_scales.detach())
        cat(self.rotations, ci, rot)
        cat(self.density, ci, dens.detach())
        cat(self.pi_logit, ci, pil)

    @torch.no_grad()
    def prune(self, ci: int, keep_mask: torch.Tensor):
        """Keep only change Gaussians where keep_mask is True (§13-S7)."""
        for pl in (self.positions, self.scales, self.rotations, self.density, self.pi_logit):
            pl[ci] = nn.Parameter(pl[ci].data[keep_mask])

    def forward(self, contrast_index: int, voxelizer, volume_shape) -> Optional[torch.Tensor]:
        """Render x_m^Δ = V(G_{Δ,m}); returns None when this contrast has no change Gaussians."""
        ci = contrast_index
        if not self.enabled or self.count(ci) == 0:
            return None
        vol = voxelizer(positions=self.positions[ci], scales=self.get_scale_values(ci),
                        rotations=self.rotations[ci], density=self.get_density(ci))
        if isinstance(vol, tuple):
            vol = torch.complex(vol[0], vol[1])
        return vol

    def param_groups(self, lr: float) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        ps = []
        for pl in (self.positions, self.scales, self.rotations, self.density, self.pi_logit):
            ps += [p for p in pl if p.numel() > 0]
        return [{"params": ps, "lr": lr}] if ps else []
