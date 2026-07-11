"""§10 Measurement-audited change routing — proposal (§10.2) + audit acceptance (§10.3).

These are pure functions driven by the trainer. They never touch model parameters directly;
they take rendered volumes / k-space and return candidate positions (proposal) or an accept
decision (audit). The anti-aliasing guarantee lives entirely in §10.3: a candidate is judged
only by whether it reduces error on the HELD-OUT audit measurements.
"""
from __future__ import annotations

from typing import Tuple
import torch


def fit_residual_evidence(x_anat: torch.Tensor, csm: torch.Tensor, ksp: torch.Tensor,
                          mask_fit: torch.Tensor, fft3c, ifft3c) -> torch.Tensor:
    """§10.2 image-space evidence map from the FIT-set k-space residual.

        r_fit = M_fit [ F(S x_A) - y ]                      (fit residual, only anatomy)
        e     = | A_fit^H r_fit |  = | Σ_c conj(S_c) iFFT(r_fit_c) |

    High e = signal the anatomy+response branch cannot explain from the fit measurements.
    Returns a (D,H,W) real >=0 evidence volume.
    """
    k_model = fft3c(csm * x_anat.unsqueeze(0))                       # (C,D,H,W)
    m = mask_fit if mask_fit.ndim == k_model.ndim else mask_fit.unsqueeze(0)
    r_fit = m * (k_model - ksp)
    backproj = (ifft3c(r_fit) * csm.conj()).sum(0)                   # A^H r  (D,H,W) complex
    return backproj.abs()


def propose_from_evidence(evidence: torch.Tensor, n_candidates: int,
                          frac_threshold: float = 0.9, seed: int = 42
                          ) -> torch.Tensor:
    """Pick up to n_candidates voxel positions (in [-1,1]^3) from the high-evidence tail.

    Candidates are drawn from voxels whose evidence exceeds the `frac_threshold` quantile,
    sampled ∝ evidence (so the strongest unexplained regions are proposed first). Returns
    positions (k,3) in the voxelizer's normalized axis order (axis0,axis1,axis2).
    """
    D, H, W = evidence.shape
    e = evidence.detach().flatten().float()
    if e.numel() == 0 or float(e.max()) <= 0:
        return torch.zeros(0, 3, device=evidence.device)
    thr = torch.quantile(e[e > 0], frac_threshold) if (e > 0).any() else e.max()
    cand = (e >= thr)
    idx_pool = torch.nonzero(cand, as_tuple=False).flatten()
    if idx_pool.numel() == 0:
        return torch.zeros(0, 3, device=evidence.device)
    w = e[idx_pool]
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = min(n_candidates, idx_pool.numel())
    sel = torch.multinomial(w.cpu() / w.cpu().sum(), k, replacement=False, generator=g)
    idx = idx_pool[sel.to(idx_pool.device)]
    zi = idx // (H * W); rem = idx % (H * W); yi = rem // W; xi = rem % W
    pz = (zi.float() + 0.5) / D * 2 - 1
    py = (yi.float() + 0.5) / H * 2 - 1
    px = (xi.float() + 0.5) / W * 2 - 1
    return torch.stack([pz, py, px], dim=1)


def audit_error(x: torch.Tensor, csm: torch.Tensor, ksp: torch.Tensor,
                mask_audit: torch.Tensor, fft3c) -> torch.Tensor:
    """‖ M_audit (F(S x) - y) ‖_2^2  on the held-out audit set (§10.3)."""
    k_model = fft3c(csm * x.unsqueeze(0))
    m = mask_audit if mask_audit.ndim == k_model.ndim else mask_audit.unsqueeze(0)
    return (m * (k_model - ksp)).abs().pow(2).sum()


def audit_gain(x_before: torch.Tensor, x_after: torch.Tensor, csm: torch.Tensor,
               ksp: torch.Tensor, mask_audit: torch.Tensor, fft3c) -> torch.Tensor:
    """ΔE = ‖A_audit x - y_audit‖² - ‖A_audit x⁺ - y_audit‖²   (§10.3).

    Positive ΔE ⇒ adding the change reduced error on measurements that did NOT generate it
    ⇒ it is a real contrast-specific change, not fitted aliasing. Accept iff ΔE > τ.
    """
    e_before = audit_error(x_before, csm, ksp, mask_audit, fft3c)
    e_after = audit_error(x_after, csm, ksp, mask_audit, fft3c)
    return e_before - e_after
