"""ElboLoss (PDF §1.5 / BP-04 + BP-12 + BP-13 weighting).

L_Bayes = L_dc^tr + β_a · L_KL^a + β_μ · L_KL^μ + β_ℓ · L_KL^ℓ
        + λ_geo · R_geo + λ_sp · R_sp

Phase 7 INF-05 contract (CONTEXT.md decision H — flat 6-item ratios)
-------------------------------------------------------------------
``loss/dc + loss/R_geo + loss/R_sp + kl/a + kl/mu + kl/ell == 1.0`` (tol 1e-3)

B-4 fix per revision loop 1 (2026-05-27)
----------------------------------------
``ElboLoss.forward`` inspects ``model.has_plain`` / ``model.has_contrast`` and
only accumulates active-channel contributions in ``L_KL^a`` and ``R_sp``. This
is required for Method A single-channel mode (08-04) where the inactive
channel's ``nn.Parameter`` (``density_*_contrast`` or ``density_*``) is NOT
registered on the model — accessing it would AttributeError.

``r_geo_tv`` accepts ``None`` for either side; single-channel callers pass
``vol_plain_for_geo=None`` (or ``vol_contrast_for_geo=None``) so the inactive
side contributes 0 (no empty-tensor traversal).

PITFALLS mitigation
-------------------
- A-01 posterior collapse → β_a warmup (caller-driven via ``beta_a_override``)
                            + free-bits clip (optional, ``enable_free_bits``)
- A-04 weight imbalance   → 6-item flat ``ratios`` dict for trainer
                            balance-violation TB scalar
- D-04 11+ loss tuning    → 6 flat scalars (drop v1 L_diff / L_enh per
                            A-LOSS-DROP / Phase 7 INF-05)

References
----------
- PDF §1.5 "ELBO" / §1.4 "Bayesian posterior" / §1.6 "Posterior uncertainty"
- .planning/phases/08-bayesian-elbo-bayesian-posterior-elbo/08-CONTEXT.md
  decisions A (geometry shared), G (η init), H (flat 6-item ratios)
- .planning/phases/08-bayesian-elbo-bayesian-posterior-elbo/08-02-elbo-loss-PLAN.md
"""
from __future__ import annotations

from typing import Dict, Any, Optional, TYPE_CHECKING

import torch
import torch.nn as nn

from .complex_gaussian_kl import complex_gaussian_kl_closed_form

if TYPE_CHECKING:
    # Avoid runtime circular import: gaussian.bayesian_joint_model imports
    # losses.complex_gaussian_kl, and losses/__init__.py re-exports ElboLoss.
    # ElboLoss only needs duck-typed access to model.has_plain / has_contrast /
    # density_* / log_s2_* / eta_* / num_points / sigma2_min / kl_geometry.
    from gaussian.bayesian_joint_model import BayesianJointGaussianModel3D


# --------------------------------------------------------------------------- #
# R_geo + R_sp utility functions (BP-12)                                     #
# --------------------------------------------------------------------------- #


def r_geo_tv(
    vol_plain: Optional[torch.Tensor],
    vol_contrast: Optional[torch.Tensor],
) -> torch.Tensor:
    """Total-variation regularizer on rendered volumes (BP-12 ``R_geo``).

    ``TV(|x|) = mean |∂_d x| + mean |∂_h x| + mean |∂_w x|`` summed over
    active channels (None on either side is skipped — B-4 fix for
    single-channel Method A mode).

    Parameters
    ----------
    vol_plain, vol_contrast : Optional[Tensor]
        Each either ``None`` (channel inactive in single-channel mode) or
        a real / complex (D, H, W) tensor.

    Returns
    -------
    Tensor
        Scalar TV value; ``0`` if both inputs are None.
    """
    def _tv(vol: torch.Tensor) -> torch.Tensor:
        # _to_magnitude analogue: complex -> abs, 2-channel real -> sqrt(re² + im²).
        if torch.is_complex(vol):
            mag = vol.abs()
        elif vol.ndim >= 1 and vol.shape[0] == 2:
            mag = torch.sqrt(vol[0] ** 2 + vol[1] ** 2 + 1e-12)
        else:
            mag = vol
        gx = (mag[1:] - mag[:-1]).abs().mean()
        gy = (mag[:, 1:] - mag[:, :-1]).abs().mean()
        gz = (mag[:, :, 1:] - mag[:, :, :-1]).abs().mean()
        return gx + gy + gz

    if vol_plain is None and vol_contrast is None:
        return torch.zeros(())
    device = (vol_plain if vol_plain is not None else vol_contrast).device
    total = torch.zeros((), device=device)
    if vol_plain is not None:
        total = total + _tv(vol_plain)
    if vol_contrast is not None:
        total = total + _tv(vol_contrast)
    return total


def r_sp_amplitude(model: BayesianJointGaussianModel3D) -> torch.Tensor:
    """Sparsity prior on ``|m_{n,c}|²`` (BP-12 ``R_sp``).

    ``R_sp = sum_{active c} mean_n(m_real_c² + m_imag_c²)``

    Note v1 ``density_real`` / ``density_imag`` ARE the posterior mean ``m``
    in the Bayesian subclass (no rename per 08-01 design decision).

    B-4 fix: introspects ``model.has_plain`` / ``model.has_contrast`` to avoid
    AttributeError when single-channel mode UNREGISTERed opposite-side params.

    Returns
    -------
    Tensor
        Scalar sum-of-channel-means. Carries grad through ``density_*``.
    """
    if not (model.has_plain or model.has_contrast):
        # Defensive — shouldn't happen given model __init__ validation.
        return torch.zeros(())
    # Derive device from an active parameter.
    if model.has_plain:
        device = model.density_real.device
    else:
        device = model.density_real_contrast.device
    sq_sum = torch.zeros((), device=device)
    if model.has_plain:
        sq_sum = sq_sum + model.density_real.pow(2).mean() \
                        + model.density_imag.pow(2).mean()
    if model.has_contrast:
        sq_sum = sq_sum + model.density_real_contrast.pow(2).mean() \
                        + model.density_imag_contrast.pow(2).mean()
    return sq_sum


# --------------------------------------------------------------------------- #
# ElboLoss                                                                    #
# --------------------------------------------------------------------------- #


class ElboLoss(nn.Module):
    """Composite ELBO loss for Bayesian Joint Gaussian Splatting MRI.

    Designed for ``BayesianJointTrainer`` (08-03) consumption: returns a dict
    so the trainer can write each component to TensorBoard + assert the 6-key
    ratios sum to 1.0 (Phase 7 INF-05 balance-violation contract).

    Parameters
    ----------
    beta_a, beta_mu, beta_ell : float
        KL weighting coefficients (BP-13). The trainer may pass per-iter
        warmup values via ``forward(beta_a_override=...)``.
    lambda_geo, lambda_sp : float
        BP-12 default 5e-4 (R_geo) / 5e-6 (R_sp). Set 0 to disable.
    sigma2_p_mu, sigma2_p_ell : float
        Geometry prior variance (CONTEXT.md decision A — shared across
        channels). Default 1e-3.
    enable_free_bits : bool
        If True, per-channel ``mean_n(KL_n)`` is clamped to >= ``free_bits_kappa``
        before optional normalization, preventing A-01 posterior collapse. Default
        OFF (per BP-13 — only enable if collapse monitor fires in retro).
    free_bits_kappa : float
        Free-bits floor (nats per Gaussian). Default 0.02.
    kl_normalization : {"sum", "mean"}
        ``sum`` preserves the original ``N * mean_n(KL_n)`` scaling. ``mean``
        uses per-Gaussian mean KL so large warm-start checkpoints do not make
        KL dominate purely because they contain more Gaussians.
    eps : float
        Numerical floor for ratios denominator.
    """

    def __init__(
        self,
        beta_a: float = 1.0,
        beta_mu: float = 0.1,
        beta_ell: float = 0.1,
        lambda_geo: float = 5e-4,
        lambda_sp: float = 5e-6,
        sigma2_p_mu: float = 1e-3,
        sigma2_p_ell: float = 1e-3,
        enable_free_bits: bool = False,
        free_bits_kappa: float = 0.02,
        kl_normalization: str = "sum",
        eps: float = 1e-12,
    ):
        super().__init__()
        self.beta_a = float(beta_a)
        self.beta_mu = float(beta_mu)
        self.beta_ell = float(beta_ell)
        self.lambda_geo = float(lambda_geo)
        self.lambda_sp = float(lambda_sp)
        self.sigma2_p_mu = float(sigma2_p_mu)
        self.sigma2_p_ell = float(sigma2_p_ell)
        self.enable_free_bits = bool(enable_free_bits)
        self.free_bits_kappa = float(free_bits_kappa)
        if kl_normalization not in {"sum", "mean"}:
            raise ValueError(f"kl_normalization must be 'sum' or 'mean', got {kl_normalization!r}")
        self.kl_normalization = kl_normalization
        self.eps = float(eps)

    # ------------------------------------------------------------------ #
    # L_KL^a (BP-04 amplitude KL with optional free-bits floor)          #
    # ------------------------------------------------------------------ #

    def _kl_amp_with_free_bits(
        self,
        model: BayesianJointGaussianModel3D,
        m_0: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """``L^a_KL`` summed over active channels (B-4 fix).

        Per active channel:
          - compute per-Gaussian closed-form KL (BP-10 complex_gaussian_kl)
          - if free-bits enabled, ``KL_n = max(KL_n, κ_freebits)`` elementwise
          - take ``mean_n`` and either keep it as a mean or rescale by ``N``
            depending on ``kl_normalization``

        Returns
        -------
        Tensor
            Scalar ``= sum_{active c} (N * mean_n(KL_n))``. Carries grad
            through posterior + EB-prior parameters.
        """
        if model.has_plain:
            device = model.density_real.device
        else:
            device = model.density_real_contrast.device
        total = torch.zeros((), device=device)
        scale = float(model.num_points) if self.kl_normalization == "sum" else 1.0

        if model.has_plain:
            kl_p_per = complex_gaussian_kl_closed_form(
                model.density_real, model.density_imag, model.log_s2_plain,
                m_0["m_0_real_plain"], m_0["m_0_imag_plain"], model.eta_plain,
                sigma2_min=model.sigma2_min,
            )
            if self.enable_free_bits:
                kl_p_per = kl_p_per.clamp_min(self.free_bits_kappa)
            total = total + scale * kl_p_per.mean()
        if model.has_contrast:
            kl_c_per = complex_gaussian_kl_closed_form(
                model.density_real_contrast, model.density_imag_contrast,
                model.log_s2_contrast,
                m_0["m_0_real_contrast"], m_0["m_0_imag_contrast"],
                model.eta_contrast,
                sigma2_min=model.sigma2_min,
            )
            if self.enable_free_bits:
                kl_c_per = kl_c_per.clamp_min(self.free_bits_kappa)
            total = total + scale * kl_c_per.mean()
        return total

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        l_dc_mc: torch.Tensor,
        model: BayesianJointGaussianModel3D,
        m_0: Dict[str, torch.Tensor],
        vol_plain_for_geo: Optional[torch.Tensor] = None,
        vol_contrast_for_geo: Optional[torch.Tensor] = None,
        *,
        beta_a_override: Optional[float] = None,
        beta_mu_override: Optional[float] = None,
        beta_ell_override: Optional[float] = None,
        disable_kl: bool = False,
        # NEW Phase 12 kwargs (CF-10 + INF-05 6→8 ratio migration).
        cf_safe_value: Optional[torch.Tensor] = None,
        cf_damp_value: Optional[torch.Tensor] = None,
        lambda_safe: float = 0.0,
        lambda_cf: float = 0.0,
    ) -> Dict[str, Any]:
        """Compose ELBO. B-4 fix: works for both dual + single-channel models.

        Parameters
        ----------
        l_dc_mc : Tensor (scalar)
            MC-averaged data-consistency loss ``L_dc^tr`` (computed by trainer
            from ``model.sample(K=K_train)`` MC samples — see PLAN 08-03).
        model : BayesianJointGaussianModel3D
            Live Bayesian model. ``has_plain`` / ``has_contrast`` introspected.
        m_0 : Dict[str, Tensor]
            Prior-mean snapshot from warm-start (``init_posterior_from_v1_checkpoint``).
            Single-channel callers pass only the active-channel keys.
        vol_plain_for_geo, vol_contrast_for_geo : Optional[Tensor]
            (D, H, W) volumes for R_geo TV regularizer. None disables that
            side (B-4 fix for single-channel mode).
        beta_*_override : Optional[float]
            Per-iter β override (used by BayesianJointTrainer for β_a warmup
            BP-13 schedule).
        disable_kl : bool
            BP-14 / D-01 diagnostic — first 200 iters K=1 + posterior frozen
            sanity baseline runs with all KL terms zeroed.
        cf_safe_value, cf_damp_value : Optional[Tensor]
            Phase 12 CF-10 counterfactual terms (scalar, grad-attached). When
            BOTH are ``None`` (Phase 7/8/9 callers) the return is byte-identical
            to the Phase 8 6-ratio schema. When supplied, the ``total`` gains
            ``lambda_safe·cf_safe`` / ``lambda_cf·cf_damp`` and the ``ratios``
            dict grows the matching ``cf/safe`` / ``cf/damp`` key(s). The kwargs
            are NOT ``.detach()``ed — grad flows through ``total.backward()``.
        lambda_safe, lambda_cf : float
            Phase 12 λ weights (default 0.0). Multiply the corresponding cf term.

        Returns
        -------
        Dict[str, Any]
            ``{'total': scalar with grad, 'l_dc_mc', 'kl_a', 'kl_mu', 'kl_ell',
                'r_geo', 'r_sp', 'cf_safe', 'cf_damp',
                'values': {per-key (β·KL or λ·R / λ·cf) products},
                'ratios': {6-, 7-, or 8-key flat ratios sum=1.0 ± 1e-6 — gains
                           cf/safe / cf/damp keys ONLY when the matching cf value
                           is supplied (INF-05 inherits; CONTEXT decision A)},
                'beta_a_applied', 'beta_mu_applied', 'beta_ell_applied'}``
        """
        device = l_dc_mc.device
        beta_a = self.beta_a if beta_a_override is None else float(beta_a_override)
        beta_mu = self.beta_mu if beta_mu_override is None else float(beta_mu_override)
        beta_ell = self.beta_ell if beta_ell_override is None else float(beta_ell_override)

        # --- KL terms ---
        if disable_kl:
            kl_a = torch.zeros((), device=device)
            kl_mu = torch.zeros((), device=device)
            kl_ell = torch.zeros((), device=device)
        else:
            kl_a = self._kl_amp_with_free_bits(model, m_0)
            # Geometry KL — always shared across channels (CONTEXT.md decision A:
            # single-channel mode still has geometry posterior; KL counted once,
            # not zero, not doubled). See 08-01 model.kl_geometry doc.
            kl_mu, kl_ell = model.kl_geometry(
                sigma2_p_mu=self.sigma2_p_mu, sigma2_p_ell=self.sigma2_p_ell,
            )

        # --- R_geo (TV) ---
        if (vol_plain_for_geo is not None or vol_contrast_for_geo is not None) \
                and self.lambda_geo > 0:
            r_geo = r_geo_tv(vol_plain_for_geo, vol_contrast_for_geo)
        else:
            r_geo = torch.zeros((), device=device)

        # --- R_sp (amplitude sparsity) ---
        r_sp = r_sp_amplitude(model)

        # --- Composite ---
        total = l_dc_mc \
            + beta_a * kl_a \
            + beta_mu * kl_mu \
            + beta_ell * kl_ell \
            + self.lambda_geo * r_geo \
            + self.lambda_sp * r_sp

        # --- Phase 12 cf terms (CF-10) — additive; zero when not supplied. ---
        # grad flows through total.backward(): cf_*_value is NOT detached here.
        if cf_safe_value is not None:
            total = total + lambda_safe * cf_safe_value
        if cf_damp_value is not None:
            total = total + lambda_cf * cf_damp_value

        # --- 6-item flat ratios (Phase 7 INF-05 schema; CONTEXT decision H) ---
        comp = {
            "loss/dc":    float(l_dc_mc.detach().item()),
            "loss/R_geo": float((self.lambda_geo * r_geo).detach().item()),
            "loss/R_sp":  float((self.lambda_sp * r_sp).detach().item()),
            "kl/a":       float((beta_a * kl_a).detach().item()),
            "kl/mu":      float((beta_mu * kl_mu).detach().item()),
            "kl/ell":     float((beta_ell * kl_ell).detach().item()),
        }
        # Phase 12: emit cf keys ONLY when the matching value is supplied —
        # preserves the Phase 8 6-key schema byte-identically when None. The
        # ``denom = sum(comp) + eps`` flow below then yields 6/7/8 keys whose
        # ratios sum to 1.0 ± 1e-6 (INF-05 inherits; CONTEXT decision A locked).
        if cf_safe_value is not None:
            comp["cf/safe"] = float((lambda_safe * cf_safe_value).detach().item())
        if cf_damp_value is not None:
            comp["cf/damp"] = float((lambda_cf * cf_damp_value).detach().item())
        denom = sum(comp.values()) + self.eps
        ratios = {k: v / denom for k, v in comp.items()}

        return {
            "total": total,
            "l_dc_mc": l_dc_mc,
            "kl_a": kl_a, "kl_mu": kl_mu, "kl_ell": kl_ell,
            "r_geo": r_geo, "r_sp": r_sp,
            # NEW: surface raw cf tensors for trainer-side TB writing (zero-tensor
            # placeholder when not supplied so callers can index unconditionally).
            "cf_safe": cf_safe_value if cf_safe_value is not None
                       else torch.zeros((), device=device),
            "cf_damp": cf_damp_value if cf_damp_value is not None
                       else torch.zeros((), device=device),
            "values": comp,
            "ratios": ratios,
            "beta_a_applied": beta_a,
            "beta_mu_applied": beta_mu,
            "beta_ell_applied": beta_ell,
        }
