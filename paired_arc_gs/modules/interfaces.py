"""Module interfaces for Paired-ARC-GS (spec §4 decomposition).

The spec composes each contrast image as (§4):

    x_m = exp(j*phi_m) * [ V(T_m(G_A), D_m(z_A)) + V(G_change_m, alpha_change_m) ]

Four pluggable modules realize this, each with a narrow interface so layers can be built
and validated one at a time:

    ResponseDecoder   §6   z_A (+ condition e_m)          -> per-Gaussian complex density
    ChangeBranch      §7   (own Gaussian set)             -> per-contrast change volume
    PhaseField        §9   coords                         -> low-freq phase phi_m(r)
    Deformation       §8   shared Gaussian centers/cov    -> T_m(mu), J Sigma J^T

Every module is an nn.Module exposing:
    * forward(...) with the signature documented on its ABC,
    * param_groups(lr) -> list of optim param-group dicts (may be empty),
    * enabled: bool (a disabled module is an identity/no-op so the same forward path
      works whether or not the layer is built yet).

The concrete implementations land per layer; these ABCs + the NoOp defaults let the
model, trainer, scheduler, and losses all be written and import-tested now.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


class PairedModule(nn.Module):
    """Common base: an optionally-disabled module that contributes optim param groups."""
    enabled: bool = False

    def param_groups(self, lr: float) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        params = [p for p in self.parameters() if p.requires_grad]
        return [{"params": params, "lr": lr}] if params else []


class ResponseDecoderBase(PairedModule):
    """§6: map anatomy latent z_A (+ imaging-condition code e_m) to per-Gaussian complex
    density for contrast m. Concrete impl is a small 2-layer MLP with per-contrast heads
    (§6.2). A disabled decoder means the model uses independent density tables (baseline)."""

    def forward(self, z_A: torch.Tensor, contrast_index: int) -> torch.Tensor:  # -> (N,) complex
        raise NotImplementedError


class ChangeBranchBase(PairedModule):
    """§7: a per-contrast, sparse, measurement-born Gaussian set G_change_m. Renders ONLY
    into contrast m. Candidates are proposed from fit-residual and accepted by audit-gain
    (§10.3); this ABC just defines the render + bookkeeping contract."""

    def forward(self, contrast_index: int, voxelizer, volume_shape) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def num_primitives(self, contrast_index: int) -> int:
        return 0


class PhaseFieldBase(PairedModule):
    """§9: low-frequency phase field phi_m(r). forward returns exp(j*phi_m) as a complex
    volume the same shape as the image, applied multiplicatively (§4)."""

    def forward(self, contrast_index: int, volume_shape) -> Optional[torch.Tensor]:
        raise NotImplementedError


class DeformationBase(PairedModule):
    """§8: per-contrast deformation T_m = R_m r + t_m + u_m(r). Transforms shared Gaussian
    centers and (Jacobian-)transforms covariances. Low-frequency, low-amplitude (§19.2.3)."""

    def forward(self, contrast_index: int, positions: torch.Tensor,
                scales: torch.Tensor, rotations: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


# ------------------------- NoOp defaults (identity / empty) ---------------------------
class NoOpPhase(PhaseFieldBase):
    enabled = False
    def forward(self, contrast_index, volume_shape):
        return None  # -> trainer treats None as phase == 0 (exp(j*0)=1)


class NoOpDeformation(DeformationBase):
    enabled = False
    def forward(self, contrast_index, positions, scales, rotations):
        return positions, scales, rotations  # identity


class NoOpChange(ChangeBranchBase):
    enabled = False
    def forward(self, contrast_index, voxelizer, volume_shape):
        return None
    def num_primitives(self, contrast_index): return 0
    def count(self, contrast_index): return 0
