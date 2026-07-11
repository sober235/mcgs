"""§9 Low-frequency per-contrast phase field  φ_m.

Spec §9: each contrast's complex image is x_m(r) = m_m(r) · e^{jφ_m(r)}, with φ_m a LOW-
FREQUENCY field realized by a 3D B-spline or a low-frequency coordinate grid:

    φ_m(r) = P_{ω_m}(r).

Regularization is wrap-aware (§9): R_φ = Σ_m ‖ w_m ⊙ ∇ e^{jφ_m} ‖_1 — the gradient is taken
on the complex phasor e^{jφ}, NOT the angle map, because a plain TV on the angle penalizes
the −π↔π branch cut spuriously. (The reg itself lives in losses/arc_losses.phase_reg; this
module just produces the field e^{jφ_m}.)

Implementation: a coarse control grid of phase values (few control points per axis) is
trilinearly upsampled to the full volume. Keeping the control grid coarse (e.g. 8³) enforces
the low-frequency constraint structurally — the field cannot represent sharp phase features
that would otherwise soak up magnitude structure. Per-contrast independent grids.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .interfaces import PhaseFieldBase


class PhaseField(PhaseFieldBase):
    def __init__(self, n_contrasts: int, volume_shape, device,
                 control_points=(8, 8, 8), init_std: float = 1e-2):
        super().__init__()
        self.enabled = True
        self.n_contrasts = int(n_contrasts)
        self.volume_shape = tuple(int(s) for s in volume_shape)
        cp = tuple(int(c) for c in control_points)
        self.control_points = cp
        # per-contrast coarse phase control grids φ (radians), init near 0 (identity phase)
        self.phi_ctrl = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, *cp, device=device) * init_std)
            for _ in range(self.n_contrasts)
        ])

    def phase_map(self, contrast_index: int) -> torch.Tensor:
        """Upsample the coarse control grid to a full-resolution phase map φ_m (D,H,W)."""
        ctrl = self.phi_ctrl[contrast_index]
        up = F.interpolate(ctrl, size=self.volume_shape, mode="trilinear", align_corners=True)
        return up[0, 0]

    def forward(self, contrast_index: int, volume_shape) -> Optional[torch.Tensor]:
        """Return the complex phasor e^{jφ_m} (D,H,W complex), or None when disabled."""
        if not self.enabled:
            return None
        phi = self.phase_map(contrast_index)
        return torch.exp(1j * phi.to(torch.float32)).to(torch.complex64)

    def param_groups(self, lr: float) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        return [{"params": list(self.parameters()), "lr": lr}]
