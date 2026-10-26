"""Counterfactual-safe loss (Phase 12 CF-08).

L^cf_safe = Σ_{c,r} H̄^cf_c(r) · |x_c(r) − stopgrad[x^dc_c(r)]|²

Forces the refined reconstruction toward a hard-data-consistency solution at
high counterfactual-risk voxels (anti-hallucination guard). The per-voxel risk
maps H̄^cf_c come from Phase 11 ``cf_results/H_bar_cf_{plain,contrast}.npy``.

Stop-grad audit (PITFALLS C-06 / CF-08 SC#1)
--------------------------------------------
Every attribution tensor (``H̄^cf_p``, ``H̄^cf_c``) is wrapped as an
``nn.Module.register_buffer`` with an explicit ``.detach().clone()`` at the
constructor — buffers are already immune to autograd, but the explicit detach
is the CF-08 SC#1 contract (defends against a caller that ``.requires_grad_``s
the source). The internal Landweber hard-DC step ``x^dc = x − τ·A*(Ax − y)`` is
``.detach()``ed at the USE site before the subtraction, so backward flows ONLY
through the live ``x_p`` / ``x_c`` tensors. A NaN injected into any attribution
buffer post-construction therefore CANNOT pollute ``model.parameters()`` grads.

DC operator band (CF-08 — tr∪fb, NOT fb-only)
---------------------------------------------
The Landweber step uses the FULL training-available operator
``A_c = M^{tr∪fb} · F · S_c`` (PDF §1.12). The held-out test mask M^te is NEVER
folded in (EA-07 isolation), so "full A_c" == tr∪fb here. The caller is
responsible for passing the tr∪fb union mask as ``mask_trfb_*``.
FIX (CF-08, commit 8b11386): an earlier draft used the fb band only — that
undercorrected the DC target; the union is required, not optional.

References
----------
- PDF §1.12 (refinement objective + Landweber hard-DC formula)
- .planning/research/PITFALLS.md C-06 (gradient pollution prevention)
- .planning/phases/12-risk-aware-refinement/12-CONTEXT.md decision A
- .planning/phases/12-risk-aware-refinement/12-VALIDATION.md CF-08
"""
from __future__ import annotations

import torch
import torch.nn as nn

from data.transforms import fft3c, ifft3c


__all__ = ["CounterfactualSafeLoss"]


class CounterfactualSafeLoss(nn.Module):
    """L^cf_safe with explicit stop-grad on H̄^cf_p / H̄^cf_c.

    Parameters
    ----------
    H_bar_cf_p, H_bar_cf_c : torch.Tensor
        Per-voxel CF risk maps from Plan 11 ``cf_results/H_bar_cf_*.npy``.
        Shape ``(D, H, W)`` real32. Wrapped as ``register_buffer`` (no autograd).
    tau : float, default 1.0
        Landweber relaxation parameter. Default 1.0 matches the Phase 8
        ``_kspace_dc_loss`` convention.
    """

    def __init__(
        self,
        H_bar_cf_p: torch.Tensor,
        H_bar_cf_c: torch.Tensor,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        # Defensive .detach().clone() — register_buffer alone already blocks
        # autograd, but explicit per the CF-08 SC#1 stop-grad contract.
        self.register_buffer("H_bar_cf_p", H_bar_cf_p.detach().clone())
        self.register_buffer("H_bar_cf_c", H_bar_cf_c.detach().clone())
        self.tau = float(tau)

    def forward(
        self,
        x_p: torch.Tensor,
        x_c: torch.Tensor,
        csm_p: torch.Tensor,
        csm_c: torch.Tensor,
        kspace_p: torch.Tensor,
        kspace_c: torch.Tensor,
        mask_trfb_p: torch.Tensor,
        mask_trfb_c: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L^cf_safe = Σ_{c,r} H̄^cf_c(r) · |x_c(r) − stopgrad[x^dc_c(r)]|².

        ``x_p`` / ``x_c`` are the live, grad-attached current reconstructions
        ``(D, H, W)`` complex. The DC target ``x^dc`` is a single Landweber step
        over the FULL acquired operator (tr∪fb union mask).
        """
        # Hard-DC Landweber step (one iteration) on A_c = M^{tr∪fb} · F · S_c.
        x_dc_p = self._landweber_step(x_p, csm_p, kspace_p, mask_trfb_p)
        x_dc_c = self._landweber_step(x_c, csm_c, kspace_c, mask_trfb_c)
        # stop-grad on x_dc; x_p / x_c remain grad-attached.
        diff_p = (x_p - x_dc_p.detach()).abs().pow(2)
        diff_c = (x_c - x_dc_c.detach()).abs().pow(2)
        # CF-08 SC#1 stop-grad audit: the attribution buffers are PURE per-voxel
        # coefficients multiplying the grad-bearing diff. register_buffer +
        # .detach() blocks autograd ENTANGLEMENT (PITFALLS Pitfall 1), but a
        # non-finite coefficient would still arithmetically poison the live
        # diff's grad (∂loss/∂diff = H̄^cf). nan_to_num sanitizes the weights
        # (non-finite risk → 0 weight) so a NaN injected into any attribution
        # buffer CANNOT reach model.parameters() grads. The .detach() keeps the
        # sanitized weight off the grad graph; finite inputs are unchanged.
        w_p = torch.nan_to_num(self.H_bar_cf_p, nan=0.0, posinf=0.0, neginf=0.0).detach()
        w_c = torch.nan_to_num(self.H_bar_cf_c, nan=0.0, posinf=0.0, neginf=0.0).detach()
        return (w_p * diff_p).sum() + (w_c * diff_c).sum()

    def _landweber_step(
        self,
        x: torch.Tensor,
        csm: torch.Tensor,
        kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """x^dc = x − τ · A*((A x − y) ⊙ M); centered FFT (CLAUDE.md mandate).

        A(x) = M ⊙ F ⊙ S(x); A*(r) = S*(IFFT(M ⊙ r)) summed over coils.
        Uses ONLY ``data.transforms.fft3c`` / ``ifft3c`` — never raw torch.fft.
        """
        pred_ksp = fft3c(x.unsqueeze(0) * csm)            # (C, D, H, W)
        residual = (pred_ksp - kspace) * mask.unsqueeze(0)
        backproj = (torch.conj(csm) * ifft3c(residual)).sum(dim=0)  # (D, H, W)
        return x - self.tau * backproj
