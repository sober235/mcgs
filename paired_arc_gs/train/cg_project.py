"""§13 Stage 10 — CG data-consistency projection.

Spec §13-S10: replace the original per-coil hard k-space replacement with a regularized
data-consistency solve, per contrast:

    x_m* = argmin_x  ½‖x − x̃_m‖²  +  (η_m/2)‖A_m x − y_m‖².

Setting the gradient to zero gives the normal equation

    (I + η A^H A) x = x̃ + η A^H y,

solved by conjugate gradient. A_m = M_m F S_m (SENSE encoding: coil-multiply → 3D FFT →
sampling mask). A^H = S^H F^H M (conj-coil combine of masked iFFT). This is a linear,
positive-definite system (I + η A^H A ≻ 0), so CG converges; it enforces measured k-space
softly (weight η) instead of overwriting it, which is gentler on noise and avoids the
hard-replacement discontinuity at the sampling boundary.
"""
from __future__ import annotations
from typing import Callable
import torch


def _AHA(x: torch.Tensor, csm: torch.Tensor, mask: torch.Tensor, fft3c, ifft3c) -> torch.Tensor:
    """A^H A x = S^H F^H M F S x  (returns image-space (D,H,W) complex)."""
    k = fft3c(csm * x.unsqueeze(0))            # F S x            (C,D,H,W)
    m = mask if mask.ndim == k.ndim else mask.unsqueeze(0)
    k = m * k                                  # M F S x
    return (ifft3c(k) * csm.conj()).sum(0)     # S^H F^H (·)


def _AHy(ksp: torch.Tensor, csm: torch.Tensor, mask: torch.Tensor, ifft3c) -> torch.Tensor:
    """A^H y = S^H F^H M y  (the masked, coil-combined measurement backprojection)."""
    m = mask if mask.ndim == ksp.ndim else mask.unsqueeze(0)
    return (ifft3c(m * ksp) * csm.conj()).sum(0)


@torch.no_grad()
def cg_data_consistency(x_tilde: torch.Tensor, csm: torch.Tensor, ksp: torch.Tensor,
                        mask: torch.Tensor, fft3c, ifft3c,
                        eta: float = 1.0, n_iters: int = 10, tol: float = 1e-6) -> torch.Tensor:
    """Solve (I + η A^H A) x = x̃ + η A^H y by conjugate gradient. Returns x_m* (D,H,W)
    complex. x_tilde is the model reconstruction (the CG anchor); the solve pulls it toward
    data consistency with the acquired samples without overwriting them."""
    def applyM(v):                              # M(v) = (I + η A^H A) v
        return v + eta * _AHA(v, csm, mask, fft3c, ifft3c)
    b = x_tilde + eta * _AHy(ksp, csm, mask, ifft3c)
    x = x_tilde.clone()
    r = b - applyM(x)
    p = r.clone()
    rs = torch.vdot(r.flatten(), r.flatten()).real
    rs0 = rs.clamp_min(1e-30)
    for _ in range(n_iters):
        Ap = applyM(p)
        alpha = rs / torch.vdot(p.flatten(), Ap.flatten()).real.clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.vdot(r.flatten(), r.flatten()).real
        if (rs_new / rs0) < tol * tol:
            break
        p = r + (rs_new / rs.clamp_min(1e-30)) * p
        rs = rs_new
    return x
