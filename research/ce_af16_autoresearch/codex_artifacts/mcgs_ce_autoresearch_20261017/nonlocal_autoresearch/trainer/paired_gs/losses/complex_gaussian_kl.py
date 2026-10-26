"""Complex Gaussian KL closed-form (PDF §1.4 / BP-10).

For q(a) = CN(m, s²), p(a) = CN(m_0, σ²_p), both isotropic complex Gaussian:
    KL(q || p) = (s² + |m - m_0|²) / σ²_p - 1 + log(σ²_p / s²)

Notes
-----
- `s² > 0` parameterized as `s² = σ²_min + softplus(log_s2)`
  (A-02 / A-03 mitigation — prevents s² → 0 from sending KL → +inf).
- `log(σ²_p / s²)` clamped to [LOG_CLAMP_MIN, LOG_CLAMP_MAX] = [-30, +30]
  to prevent fp32 overflow on extreme posterior/prior width mismatches
  (PITFALLS A-02).
- `s²` is the total variance of a complex circular Gaussian; real and imag
  components each have variance s²/2.

References
----------
- PDF §1.4 "Bayesian Posterior"
- /data0/congcong/code/GS/3DGS/.planning/research/PITFALLS.md A-02, A-03
- Plan: .planning/phases/08-bayesian-elbo-bayesian-posterior-elbo/
  08-01-bayesian-model-PLAN.md (BP-10)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# PITFALLS A-02: clamp log(σ²_p / s²) to a finite range so fp32 cannot
# return ±inf even when posterior and prior widths differ by ~1e20.
LOG_CLAMP_MIN: float = -30.0
LOG_CLAMP_MAX: float = 30.0


def complex_gaussian_kl_closed_form(
    m_real: torch.Tensor,                # (N,) real fp32 — posterior mean (real part)
    m_imag: torch.Tensor,                # (N,) real fp32 — posterior mean (imag part)
    log_s2: torch.Tensor,                # (N,) real fp32 — posterior parameterization
    m_0_real: torch.Tensor,              # (N,) prior mean (real part)
    m_0_imag: torch.Tensor,              # (N,) prior mean (imag part)
    log_sigma2_p_param: torch.Tensor,    # (N,) eta — prior precision parameterization
    *,
    sigma2_min: float = 1e-8,
) -> torch.Tensor:
    """Per-Gaussian closed-form KL of two isotropic complex Gaussians, shape (N,).

    Decoding (PITFALLS A-02 / A-03):
        s²    = sigma2_min + softplus(log_s2)
        σ²_p  = sigma2_min + softplus(log_sigma2_p_param)

    Parameters
    ----------
    m_real, m_imag : (N,) real tensors
        Posterior mean of q(a) = CN(m, s²) split into real and imaginary parts.
    log_s2 : (N,) real tensor
        Posterior variance parameterization. ``s² = σ²_min + softplus(log_s2)``.
    m_0_real, m_0_imag : (N,) real tensors
        Prior mean of p(a) = CN(m_0, σ²_p) (no grad expected — usually a
        snapshot of the v1 deterministic warm-start, per Plan BP-11).
    log_sigma2_p_param : (N,) real tensor
        Prior variance parameterization (η in Empirical Bayes BP-09).
        ``σ²_p = σ²_min + softplus(log_sigma2_p_param)``.
    sigma2_min : float
        Numerical floor on both s² and σ²_p (PITFALLS A-03). Default 1e-8.

    Returns
    -------
    kl : (N,) real tensor
        Per-Gaussian KL(q || p). Always finite; gradients propagate through
        ``m_real``, ``m_imag``, ``log_s2``, and ``log_sigma2_p_param``.
    """
    s2 = sigma2_min + F.softplus(log_s2)                       # (N,) > 0
    sigma2_p = sigma2_min + F.softplus(log_sigma2_p_param)     # (N,) > 0

    d_real = m_real - m_0_real
    d_imag = m_imag - m_0_imag
    dist_sq = d_real.pow(2) + d_imag.pow(2)                    # |m - m_0|² (real)

    ratio = (s2 + dist_sq) / sigma2_p

    # log term — clamp for fp32 safety. We compute log(σ²_p) - log(s²) so the
    # clamp lives directly on the log-ratio. The sigma2_min floor ensures the
    # log arguments are strictly positive even when softplus saturates near 0.
    log_s2_safe = torch.log(s2.clamp_min(1e-38))
    log_sp2_safe = torch.log(sigma2_p.clamp_min(1e-38))
    log_ratio = (log_sp2_safe - log_s2_safe).clamp(LOG_CLAMP_MIN, LOG_CLAMP_MAX)

    return ratio - 1.0 + log_ratio                              # (N,)
