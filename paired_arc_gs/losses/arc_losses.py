"""§12 optimization objective for Paired-ARC-GS.

    L = L_DC + λ_Δ R_Δ + λ_route R_route + λ_geo R_geo + λ_def R_def + λ_φ R_φ   (§12)

Each term below carries its exact spec formula. Two MC-GS mechanisms are DELIBERATELY
NOT here, per spec:
  * §12.1 — no fixed centre-0.3 / high-2.2 k-space frequency weighting. It "changes the
            statistical meaning of measurement noise". DC is instead noise-normalized by
            coil pre-whitening and divided by |Ω_fit|.
  * §12.4 — no image-space TV(|x1|)+TV(|x2|) masquerading as geometry regularization.
            Geometry reg acts DIRECTLY on Gaussian parameters (scale/aniso/overlap).

All terms return real scalars; complex data uses squared modulus. Terms are additively
combined by `total_loss` using the stage's active-term set (from the scheduler).
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import torch


# --------------------------------------------------------------------------------------
# §12.1  Noise-normalized data consistency
# --------------------------------------------------------------------------------------
def prewhiten(y: torch.Tensor, S: torch.Tensor,
              L_inv: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply coil-noise pre-whitening ỹ = Ψ^{-1/2} y, S̃ = Ψ^{-1/2} S  (§12.1).

    Parameters
    ----------
    y : (C, D, H, W) complex     multi-coil k-space (or its masked subset)
    S : (C, D, H, W) complex     coil sensitivities
    L_inv : (C, C) complex or None
        Ψ^{-1/2} — the inverse matrix square-root of the coil noise covariance Ψ.
        When None (noise cov unavailable), returns (y, S) unchanged (whitening is a
        no-op) so the term still runs; the DC then equals an unweighted LS fit.

    The whitening mixes coils: ỹ[c'] = Σ_c L_inv[c',c] y[c]. Implemented as an einsum
    over the coil axis.
    """
    if L_inv is None:
        return y, S
    yt = torch.einsum("ij,j...->i...", L_inv, y)
    St = torch.einsum("ij,j...->i...", L_inv, S)
    return yt, St


def data_consistency(x: torch.Tensor, S: torch.Tensor, y: torch.Tensor,
                     mask_fit: torch.Tensor, fft3c, L_inv: Optional[torch.Tensor] = None
                     ) -> torch.Tensor:
    """L_DC for ONE contrast (§12.1):

        || M_fit [ F_3D(S̃ x) - ỹ ] ||_2^2  /  |Ω_fit|

    x : (D,H,W) complex image ;  S : (C,D,H,W) ;  y : (C,D,H,W) k-space ;
    mask_fit : (D,H,W) or (C,D,H,W) {0,1} fit-set sampling mask ;
    fft3c : centered 3D FFT callable (reused from paired_gs.data.transforms).
    Divides by |Ω_fit| = number of sampled fit locations (per §12.1) so the term is
    invariant to sampling density and comparable across the two contrasts.
    """
    yt, St = prewhiten(y, S, L_inv)
    k_model = fft3c(St * x.unsqueeze(0))                 # (C,D,H,W)
    m = mask_fit if mask_fit.ndim == k_model.ndim else mask_fit.unsqueeze(0)
    resid = m * (k_model - yt)
    omega = m.sum().clamp_min(1.0)                        # |Ω_fit| (counts coils×locs)
    return (resid.abs() ** 2).sum() / omega


# --------------------------------------------------------------------------------------
# §12.4  Direct Gaussian geometry regularization  (NOT image-space TV)
# --------------------------------------------------------------------------------------
def geometry_reg(scale_values: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    """R_geo = R_scale + λ_a R_aniso + λ_o R_overlap   (§12.4).

    scale_values : (N, 3) POSITIVE per-Gaussian scales (voxels) — i.e. exp(log-scales).

    R_scale  = Σ_n [ max(0, s_n - s_max)^2 + max(0, s_min - s_n)^2 ]   (hinge on out-of-range scales)
    R_aniso  = Σ_n Var_axis(s_n) / mean_axis(s_n)^2   (penalize extreme axis anisotropy)
    R_overlap= (optional, λ_o default 0) mean squared inverse-scale, a cheap proxy that
               discourages many tiny overlapping Gaussians; full pairwise overlap is left
               to the pruning stage (§13-S7).
    """
    geo = cfg.get("geo", {})
    s_min = float(geo.get("scale_min", 0.5))
    s_max = float(geo.get("scale_max", 20.0))
    lam_a = float(geo.get("lambda_aniso", 0.1))
    lam_o = float(geo.get("lambda_overlap", 0.0))

    over = torch.clamp(scale_values - s_max, min=0.0) ** 2
    under = torch.clamp(s_min - scale_values, min=0.0) ** 2
    r_scale = (over + under).sum()

    mean_ax = scale_values.mean(dim=1, keepdim=True).clamp_min(1e-6)
    r_aniso = (scale_values.var(dim=1, unbiased=False) / (mean_ax.squeeze(1) ** 2)).sum()

    r_overlap = torch.zeros((), device=scale_values.device, dtype=scale_values.dtype)
    if lam_o > 0:
        r_overlap = (1.0 / scale_values.clamp_min(1e-3)).pow(2).mean()

    return r_scale + lam_a * r_aniso + lam_o * r_overlap


# --------------------------------------------------------------------------------------
# §12.2  Change-branch sparsity
# --------------------------------------------------------------------------------------
def change_sparsity(change_densities: torch.Tensor,
                    p_change: Optional[torch.Tensor] = None) -> torch.Tensor:
    """R_Δ = Σ_{m,l} p_{Δ,l} ||α_{Δ,l}||_2   (§12.2, group-L2 over each primitive).

    change_densities : (L,) complex OR (L, K) complex amplitudes of change primitives.
    p_change : (L,) optional per-primitive confidence weight p_{Δ,l} (default 1).
    Group-L2 (not L1 of scalars) so each primitive's complex amplitude is one group —
    drives whole change Gaussians to zero rather than shrinking real/imag independently.
    """
    if change_densities.numel() == 0:
        return torch.zeros((), device=change_densities.device)
    a = change_densities
    if a.ndim == 1:
        group_norm = a.abs()                              # |α_l| per primitive
    else:
        group_norm = torch.sqrt((a.abs() ** 2).sum(dim=1) + 1e-12)
    if p_change is not None:
        group_norm = p_change * group_norm
    return group_norm.sum()


# --------------------------------------------------------------------------------------
# §12.3  Shared–change exclusivity (routing)
# --------------------------------------------------------------------------------------
def route_exclusivity(a_shared: torch.Tensor, a_change: torch.Tensor) -> torch.Tensor:
    """R_route = Σ_m ∫ a_m^A(r) a_m^Δ(r) dr   (§12.3).

    a_shared, a_change : (D,H,W) NON-NEGATIVE spatial activation maps (e.g. |x^A|, |x^Δ|)
    for one contrast. The integral becomes a sum over voxels. Penalizes spatial overlap so
    the same content is not explained by BOTH branches (which would make the A/Δ split
    non-identifiable, §19.2.1)."""
    return (a_shared * a_change).sum()


# --------------------------------------------------------------------------------------
# P3  Cross-contrast local low-rank (nuclear-norm) prior  (self-contained, design §5)
# --------------------------------------------------------------------------------------
def cross_contrast_llr(x_p: torch.Tensor, x_c: torch.Tensor,
                       block: int = 8, stride: int = 4, eps: float = 1e-8) -> torch.Tensor:
    """L_LLR = Σ_b ||M_b||_*  where M_b ∈ R^{block^3 × 2} stacks the two contrasts'
    magnitudes over sliding block b (design §5).

    For an N×2 matrix the nuclear norm is σ₁+σ₂ = √λ₊ + √λ₋, the sqrt-eigenvalues of the
    2×2 Gram G_b = M_bᵀM_b = [[p, q],[q, r]] with p=Σa², r=Σb², q=Σab (a=|x_p|, b=|x_c| on
    the block). λ± = (p+r)/2 ± √(((p−r)/2)² + q²). The three block sums are STRIDED BOX
    FILTERS (avg_pool3d × block³), so the whole penalty is fully vectorized — no per-block
    loop — and differentiable. Magnitudes are used to avoid phase coupling."""
    import torch.nn.functional as F
    a = x_p.abs() if x_p.is_complex() else x_p
    b = x_c.abs() if x_c.is_complex() else x_c
    n = float(block ** 3)

    def boxsum(v: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(v[None, None], kernel_size=block, stride=stride)[0, 0] * n

    p = boxsum(a * a)
    r = boxsum(b * b)
    q = boxsum(a * b)
    half = 0.5 * (p + r)
    disc = torch.sqrt(torch.clamp((0.5 * (p - r)) ** 2 + q * q, min=0.0) + eps)
    lam_hi = torch.clamp(half + disc, min=0.0)
    lam_lo = torch.clamp(half - disc, min=0.0)
    return (torch.sqrt(lam_hi + eps) + torch.sqrt(lam_lo + eps)).sum()


# --------------------------------------------------------------------------------------
# §9 / §8  phase and deformation regularizers (used when those layers are on)
# --------------------------------------------------------------------------------------
def phase_reg(phase_complex: torch.Tensor, weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
    """R_φ = || w ⊙ ∇ e^{jφ} ||_1   (§9, wrap-aware — gradient taken on e^{jφ}, NOT on the
    angle map, so the ±π wrap is not penalized).

    phase_complex : (D,H,W) complex = e^{jφ_m}. Finite-difference gradient along each axis;
    L1 of the complex gradient magnitude, optionally weighted by w (edge-aware)."""
    e = phase_complex
    total = torch.zeros((), device=e.device)
    for ax in range(e.ndim):
        g = torch.diff(e, dim=ax)
        gm = g.abs()
        if weight_map is not None:
            sl = [slice(None)] * e.ndim
            sl[ax] = slice(1, None)
            gm = weight_map[tuple(sl)] * gm
        total = total + gm.sum()
    return total


def deformation_reg(u: torch.Tensor, detJ: Optional[torch.Tensor], lambda_jacobian: float) -> torch.Tensor:
    """R_def = ||∇u||_2^2 + λ_J ||det J - 1||_2^2   (§8).

    u : (..., 3) nonrigid displacement field ;  detJ : (...) Jacobian determinant (or None).
    Keeps deformation low-frequency/small so it cannot absorb real change/enhancement (§19.2.3)."""
    reg = torch.zeros((), device=u.device)
    # ||∇u||^2 : finite diff of each displacement component along spatial axes
    if u.ndim >= 2:
        for ax in range(u.ndim - 1):
            reg = reg + (torch.diff(u, dim=ax) ** 2).sum()
    if detJ is not None:
        reg = reg + lambda_jacobian * ((detJ - 1.0) ** 2).sum()
    return reg


# --------------------------------------------------------------------------------------
# Aggregator
# --------------------------------------------------------------------------------------
def total_loss(terms: Dict[str, torch.Tensor], active: Tuple[str, ...],
               weights: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Sum the active terms with their λ weights. `terms` maps term-key -> scalar tensor;
    `active` is the stage's active-term set (from StageState.loss_terms); `weights` maps
    term-key -> λ (dc has implicit weight 1). Returns (loss, per-term float log)."""
    lam = {
        "dc": 1.0,
        "geo": weights.get("lambda_geo", 1e-3),
        "change_sparse": weights.get("lambda_change", 1e-3),
        "route": weights.get("lambda_route", 1e-2),
        "phase": weights.get("lambda_phase", 1e-3),
        "def": weights.get("lambda_def", 1e-2),
        "llr": weights.get("lambda_llr", 0.0),
        "freqextrap": weights.get("lambda_freqextrap", 0.0),
    }
    device = next((t.device for t in terms.values() if torch.is_tensor(t)), torch.device("cpu"))
    loss = torch.zeros((), device=device)
    log: Dict[str, float] = {}
    for key in active:
        if key in terms and torch.is_tensor(terms[key]):
            contrib = lam.get(key, 1.0) * terms[key]
            loss = loss + contrib
            log[key] = float(contrib.detach())
    log["total"] = float(loss.detach())
    return loss, log
