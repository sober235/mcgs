"""Counterfactual damping loss (Phase 12 CF-09).

L_cf-damp = Σ_{n,c} stopgrad[C^final_{n,c}] · |m_{n,c}|²

PDF-faithful (§1.12): bare ``C^final · |m|²``, **NO** ``S_cross`` factor. Lesion
preservation is delegated UPSTREAM to ``C^final`` itself — CF-04's DC-safeguard
term ``exp(-γ_D[ΔD]_+)`` shrinks ``C^final`` for any primitive whose removal
breaks k-space data consistency, so real FLAIR-only lesions (which ARE k-space
supported) receive low ``C^final`` and are damped little. (Decision 2026-05-29:
an earlier plan multiplied by ``stopgrad[S_cross(μ_n)]`` as a lesion guard; that
factor is NOT in the PDF and was reverted to keep the locked formula. See
REQUIREMENTS L72.)

Per-Gaussian damping shrinks the posterior-mean amplitudes ``m_{n,c} =
density_real_* + i·density_imag_*`` of high-``C^final`` primitives (likely
hallucinations) toward zero. ``m`` is the live grad path; ``C^final`` is a frozen
attribution buffer (register_buffer + explicit ``.detach()``), so a NaN injected
into the buffer post-construction cannot pollute ``model.parameters()`` grads.

References
----------
- PDF §1.12 (refinement objective) — bare C^final · |m|²
- .planning/research/PITFALLS.md C-05 (lesion preservation carried by CF-04 C^final)
- .planning/phases/12-risk-aware-refinement/12-CONTEXT.md decision C
- .planning/phases/12-risk-aware-refinement/12-VALIDATION.md CF-09
"""
from __future__ import annotations

import torch
import torch.nn as nn


__all__ = ["CounterfactualDampingLoss"]


class CounterfactualDampingLoss(nn.Module):
    """L_cf-damp with stop-grad on C^final (PDF §1.12 — no S_cross factor).

    Parameters
    ----------
    C_final : torch.Tensor
        Per-Gaussian per-contrast attribution from Plan 11. Shape ``(N, 2)``
        float32. Registered as a buffer (no autograd). Lesion preservation is
        carried by ``C^final`` itself (CF-04 DC-safeguard) — see module docstring.
    """

    def __init__(self, C_final: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("C_final", C_final.detach().clone())

    @staticmethod
    def _plain_real_imag(model: nn.Module):
        """Resolve the plain-channel (density_real, density_imag) live tensors.

        The real ``(Bayesian)JointGaussianModel3D`` stores the PLAIN channel as
        ``density_real`` / ``density_imag`` (the ``density_plain`` property wraps
        them); the contrast channel is ``density_real_contrast`` /
        ``density_imag_contrast``. Test substitutes may instead expose
        ``density_real_plain`` / ``density_imag_plain`` (the plan's interface
        name). Accept either so cf_damp works against the real model AND fixtures.
        """
        if hasattr(model, "density_real_plain"):
            return model.density_real_plain, model.density_imag_plain
        return model.density_real, model.density_imag

    def forward(self, model: nn.Module) -> torch.Tensor:
        """Compute L_cf-damp = Σ_{n,c} C^final · |m|² with live grad on model.density_*."""
        # m_{n, plain} amplitude squared (live grad path through density_*).
        dr_p, di_p = self._plain_real_imag(model)
        m_plain_sq = dr_p.pow(2) + di_p.pow(2)
        # m_{n, contrast} amplitude squared (gate on has_contrast for Method A).
        if getattr(model, "has_contrast", True) and hasattr(
            model, "density_real_contrast"
        ):
            m_contrast_sq = (
                model.density_real_contrast.pow(2)
                + model.density_imag_contrast.pow(2)
            )
        else:
            m_contrast_sq = torch.zeros_like(m_plain_sq)

        # Damping coefficients per Gaussian per channel (PDF §1.12):
        # C_final[:,0] = plain attribution; C_final[:,1] = contrast attribution.
        # NO S_cross factor — lesion protection is delegated to C^final (CF-04).
        # CF-09 stop-grad audit: C_final is a frozen coefficient. register_buffer
        # + .detach() blocks autograd entanglement; nan_to_num sanitizes the
        # coefficient (non-finite C^final → 0) so a NaN injected into the buffer
        # cannot arithmetically poison the live m_*_sq grad. Finite C^final is
        # unchanged (bare PDF formula preserved).
        coef = torch.nan_to_num(self.C_final, nan=0.0, posinf=0.0, neginf=0.0).detach()
        coef_p = coef[:, 0]
        coef_c = coef[:, 1]
        return (coef_p * m_plain_sq).sum() + (coef_c * m_contrast_sq).sum()
