"""BayesianJointGaussianModel3D (PDF §1.4 / BP-01..03 / BP-09).

Subclass of JointGaussianModel3D adding Bayesian posterior parameters:
  - q(a_{n,c}) = CN(m_{n,c}, s²_{n,c})    # complex amplitude posterior (BP-02)
  - q(δμ_n)   = N(0, σ²_{μ,n} I)          # geometry center perturbation (BP-03)
  - q(δℓ_n)   = N(0, σ²_{ℓ,n} I)          # log-scale perturbation (BP-03)
  - σ²_{p,n,c} = σ²_min + softplus(η_{n,c})  # Empirical Bayes prior (BP-09)

Parameter sharing protocol (v1 backward-compat)
-----------------------------------------------
The v1 ``density_real`` / ``density_imag`` (and their ``_contrast`` siblings)
are REPURPOSED as the posterior mean ``m``. No new ``m_real`` / ``m_imag``
fields are added — they ARE the v1 ``density_*`` tensors.

Single-channel mode (B-1 fix per revision loop 1 — 2026-05-27)
--------------------------------------------------------------
``single_contrast_channel: Optional[Literal['plain', 'contrast']] = None``
  - ``None``      → dual-channel joint (Method B; CONTEXT.md decision A — "ours")
  - ``'plain'``   → only plain channel registered (Method A T1 run).
                    DELATTR + UNREGISTER ``density_real_contrast``,
                    ``density_imag_contrast``, ``log_s2_contrast``,
                    ``eta_contrast`` — these ``nn.Parameter`` do NOT EXIST.
  - ``'contrast'``→ only contrast channel registered (Method A FLAIR run, mirror).

Method A protocol: each contrast trained with N independent Gaussians, two
runs produce a total of 2N Gaussians (CONTEXT.md decision A locked: 2N total,
NOT a freeze-hack on a dual model). Module attributes ``has_plain`` /
``has_contrast`` are set per instance and consumed by Phase 8 ElboLoss
(B-4 fix in 08-02).

A-LOSS-DROP deprecation (W-1 fix per revision loop 1)
-----------------------------------------------------
``amplitude_difference()`` override raises ``DeprecationWarning`` at call time —
v2 uses ``q(a_{n,c})`` posterior KL via ``ElboLoss._kl_amp_with_free_bits``.

Five-or-six NEW ``nn.Parameter`` groups (per-Gaussian, real fp32)
----------------------------------------------------------------
Dual mode:
  log_s2_plain        — q(a_plain) variance posterior parameterization
  log_s2_contrast     — q(a_contrast) variance posterior parameterization
  log_sigma2_mu       — q(δμ) variance posterior parameterization (shared)
  log_sigma2_ell      — q(δℓ) variance posterior parameterization (shared)
  eta_plain           — Empirical Bayes prior precision (plain channel)
  eta_contrast        — Empirical Bayes prior precision (contrast channel)

Single-channel modes register only the geometry posteriors plus the active
channel's amp posterior + EB prior parameter.

PITFALLS mitigation
-------------------
- A-02 / A-03: ``log_s2`` parameterization with ``s² = σ²_min + softplus(log_s2)``
- A-06 + INF-06: subclass overrides ``_update_params`` to sync new fields during
  densification — picked up transparently by ``trainer._transplant_adam_state``
  (Phase 7 INF-06 contract).
- D-02: ``load_state_dict(strict=False)`` for v1 ckpt warm-start.
"""
from __future__ import annotations

from typing import Tuple, List, Dict, Any, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .joint_gaussian_model import JointGaussianModel3D
from losses.complex_gaussian_kl import complex_gaussian_kl_closed_form


# CONTEXT.md decision G: η init -1.0 -> σ²_p_init ≈ 0.313 (same order as s²_init,
# prevents A-04 sham convergence by preventing prior from collapsing).
DEFAULT_LOG_ETA_INIT: float = -1.0

# CONTEXT.md decision D: σ²_min = 1e-8 (PITFALLS A-02 / A-03 floor).
DEFAULT_SIGMA2_MIN: float = 1e-8

# Reparam: s = sqrt(s²).clamp_min(σ_min_clip) (PITFALLS A-03 NaN gate).
DEFAULT_SIGMA_MIN_CLIP: float = 1e-5

# s²_init = log(σ² - σ²_min) -> softplus_inv ≈ -6.91 for σ²_init = 1e-3.
DEFAULT_SIGMA2_INIT: float = 1e-3


def _softplus_inv(y: torch.Tensor | float) -> torch.Tensor:
    """Invert softplus: solve ``softplus(x) = y`` for x. ``log(exp(y) - 1)``."""
    if isinstance(y, (int, float)):
        y = torch.tensor(float(y))
    return torch.log(torch.expm1(y).clamp_min(1e-30))


class BayesianJointGaussianModel3D(JointGaussianModel3D):
    """v1 dual-density subclass with complex-amplitude + geometry posteriors + EB prior.

    Supports single-channel mode (``single_contrast_channel='plain'`` or ``'contrast'``)
    for Method A baseline (B-1 revision-loop-1 fix).
    """

    def __init__(
        self,
        num_points: int,
        volume_shape: Tuple[int, int, int],
        device: torch.device,
        initial_positions: Optional[torch.Tensor] = None,
        initial_densities_plain: Optional[torch.Tensor] = None,
        initial_densities_contrast: Optional[torch.Tensor] = None,
        initial_scales: Optional[torch.Tensor] = None,
        sigma2_init: float = DEFAULT_SIGMA2_INIT,
        log_eta_init: float = DEFAULT_LOG_ETA_INIT,
        sigma2_min: float = DEFAULT_SIGMA2_MIN,
        single_contrast_channel: Optional[Literal['plain', 'contrast']] = None,
    ):
        # ----- single-channel mode validation (B-1 revision-loop-1 fix) -----
        if single_contrast_channel not in (None, 'plain', 'contrast'):
            raise ValueError(
                f"single_contrast_channel must be None / 'plain' / 'contrast', "
                f"got {single_contrast_channel!r}"
            )
        # Set introspection booleans BEFORE super().__init__ so any parent hook
        # that asks could read them — parent doesn't, but defensive ordering.
        self_has_plain = single_contrast_channel in (None, 'plain')
        self_has_contrast = single_contrast_channel in (None, 'contrast')

        # Parent always initializes both density_*  / density_*_contrast. If
        # initial densities for the inactive side aren't provided, mirror the
        # active side to avoid AttributeError inside the v1 init code.
        plain_init = initial_densities_plain
        contrast_init = initial_densities_contrast
        if not self_has_plain and contrast_init is None:
            # contrast-only mode: still need a placeholder for plain to satisfy
            # the v1 constructor signature; it gets unregistered immediately.
            plain_init = None
        if not self_has_contrast and plain_init is None:
            contrast_init = None

        super().__init__(
            num_points=num_points,
            volume_shape=volume_shape,
            device=device,
            initial_positions=initial_positions,
            initial_densities_plain=plain_init,
            initial_densities_contrast=contrast_init,
            initial_scales=initial_scales,
        )

        # Now nn.Module is set up; record final introspection state.
        self.has_plain: bool = self_has_plain
        self.has_contrast: bool = self_has_contrast
        self._channel_mode = single_contrast_channel  # for debugging / state_dict logs

        # ----- single-channel UNREGISTER opposite-side density_* (B-1 fix) -----
        if not self.has_contrast:
            # Method A T1 run: *_contrast nn.Parameter must NOT exist.
            for attr in ("density_real_contrast", "density_imag_contrast"):
                if attr in self._parameters:
                    del self._parameters[attr]
                if attr in self.__dict__:
                    del self.__dict__[attr]
        if not self.has_plain:
            for attr in ("density_real", "density_imag"):
                if attr in self._parameters:
                    del self._parameters[attr]
                if attr in self.__dict__:
                    del self.__dict__[attr]

        self.sigma2_min = float(sigma2_min)
        log_s2_init_value = _softplus_inv(
            max(sigma2_init - sigma2_min, 1e-30)
        ).item()
        ones = torch.ones(self.num_points, device=device)

        # Geometry posteriors — always registered (shared across channels,
        # CONTEXT.md decision A: even single-channel runs have geometry KL).
        self.log_sigma2_mu = nn.Parameter(ones.clone() * log_s2_init_value)
        self.log_sigma2_ell = nn.Parameter(ones.clone() * log_s2_init_value)

        # Amplitude posteriors + EB prior — ONLY for active channels.
        if self.has_plain:
            self.log_s2_plain = nn.Parameter(ones.clone() * log_s2_init_value)
            self.eta_plain = nn.Parameter(ones.clone() * float(log_eta_init))
        if self.has_contrast:
            self.log_s2_contrast = nn.Parameter(ones.clone() * log_s2_init_value)
            self.eta_contrast = nn.Parameter(ones.clone() * float(log_eta_init))

        # Build instance _EXTRA_DENSIFY_FIELDS based on active channels.
        extra: List[str] = ["log_sigma2_mu", "log_sigma2_ell"]
        if self.has_plain:
            extra += ["log_s2_plain", "eta_plain"]
        if self.has_contrast:
            extra += ["log_s2_contrast", "eta_contrast"]
        self._EXTRA_DENSIFY_FIELDS = tuple(extra)

    # --------------------------------------------------------------------- #
    # Helpers                                                                #
    # --------------------------------------------------------------------- #

    def _decode_s2(self, log_s2: torch.Tensor) -> torch.Tensor:
        """``s² = σ²_min + softplus(log_s2)`` (A-02 / A-03)."""
        return self.sigma2_min + F.softplus(log_s2)

    # --------------------------------------------------------------------- #
    # BP-09: Empirical Bayes prior width                                     #
    # --------------------------------------------------------------------- #

    def prior_sigma2_p(self, channel: str) -> torch.Tensor:
        """``σ²_p = σ²_min + softplus(η)`` for the requested channel."""
        if channel == "plain":
            if not self.has_plain:
                raise ValueError("plain channel not active in this model instance")
            eta = self.eta_plain
        elif channel == "contrast":
            if not self.has_contrast:
                raise ValueError("contrast channel not active in this model instance")
            eta = self.eta_contrast
        else:
            raise ValueError(f"unknown channel {channel!r}")
        return self.sigma2_min + F.softplus(eta)

    # --------------------------------------------------------------------- #
    # BP-02 + BP-03: reparam sampling                                        #
    # --------------------------------------------------------------------- #

    def sample(
        self,
        K: int,
        *,
        generator: Optional[torch.Generator] = None,
        disable_mc_for_init: bool = False,
        sigma_min_clip: float = DEFAULT_SIGMA_MIN_CLIP,
    ) -> Dict[str, torch.Tensor]:
        """Draw K Monte-Carlo reparam samples.

        Returns a dict with keys only for active channels (B-1 fix):
          - dual mode:    ``{'a_plain', 'a_contrast', 'delta_mu', 'delta_log_scales'}``
          - plain mode:   ``{'a_plain', 'delta_mu', 'delta_log_scales'}``
          - contrast mode:``{'a_contrast', 'delta_mu', 'delta_log_scales'}``

        ``disable_mc_for_init=True`` (BP-14 / D-01 sanity baseline):
            - K forced to 1
            - eps_scale forced to 0 (s = sigma_min; delta = 0)
            Result is deterministic — two consecutive calls return identical tensors.
        """
        device = self.positions.device
        N = self.positions.shape[0]
        if disable_mc_for_init:
            K = 1
            eps_scale = 0.0
        else:
            eps_scale = 1.0

        # Geometry posteriors — always present.
        s2_mu = self._decode_s2(self.log_sigma2_mu).clamp_min(self.sigma2_min)
        s2_ell = self._decode_s2(self.log_sigma2_ell).clamp_min(self.sigma2_min)
        s_mu = s2_mu.sqrt().clamp_min(sigma_min_clip)
        s_ell = s2_ell.sqrt().clamp_min(sigma_min_clip)
        eps_mu = torch.randn(K, N, 3, device=device, generator=generator)
        eps_ell = torch.randn(K, N, 3, device=device, generator=generator)
        out: Dict[str, torch.Tensor] = {
            "delta_mu": eps_scale * s_mu.view(1, N, 1) * eps_mu,
            "delta_log_scales": eps_scale * s_ell.view(1, N, 1) * eps_ell,
        }

        # Per-channel amplitude posteriors.
        sqrt_half = 0.5 ** 0.5  # complex circular: each real/imag component has variance s²/2
        if self.has_plain:
            s2_plain = self._decode_s2(self.log_s2_plain).clamp_min(self.sigma2_min)
            s_plain = s2_plain.sqrt().clamp_min(sigma_min_clip)
            eps_re = torch.randn(K, N, device=device, generator=generator) * sqrt_half
            eps_im = torch.randn(K, N, device=device, generator=generator) * sqrt_half
            a_re = self.density_real.unsqueeze(0) + eps_scale * s_plain.unsqueeze(0) * eps_re
            a_im = self.density_imag.unsqueeze(0) + eps_scale * s_plain.unsqueeze(0) * eps_im
            out["a_plain"] = torch.complex(a_re, a_im)
        if self.has_contrast:
            s2_c = self._decode_s2(self.log_s2_contrast).clamp_min(self.sigma2_min)
            s_c = s2_c.sqrt().clamp_min(sigma_min_clip)
            eps_re = torch.randn(K, N, device=device, generator=generator) * sqrt_half
            eps_im = torch.randn(K, N, device=device, generator=generator) * sqrt_half
            a_re = self.density_real_contrast.unsqueeze(0) + eps_scale * s_c.unsqueeze(0) * eps_re
            a_im = self.density_imag_contrast.unsqueeze(0) + eps_scale * s_c.unsqueeze(0) * eps_im
            out["a_contrast"] = torch.complex(a_re, a_im)
        return out

    # --------------------------------------------------------------------- #
    # BP-04 + BP-10: amplitude KL                                            #
    # --------------------------------------------------------------------- #

    def kl_amplitude(
        self,
        m_0_real_plain: torch.Tensor, m_0_imag_plain: torch.Tensor,
        m_0_real_contrast: torch.Tensor, m_0_imag_contrast: torch.Tensor,
    ) -> torch.Tensor:
        """``L^a_KL`` summed over active channels — scalar tensor with grad.

        Each channel contributes ``N * mean_n(KL_n)`` (PITFALLS A-01 loss-reduction
        calibration: per-Gaussian average × N keeps physical magnitude under
        densification while staying β-stable).

        NOTE: kept as a backward-compat convenience. ElboLoss in 08-02 will call
        ``complex_gaussian_kl_closed_form`` directly with per-channel branching
        guided by ``has_plain`` / ``has_contrast`` for sharper diagnostics.
        """
        total = torch.zeros((), device=self.positions.device)
        if self.has_plain:
            kl_p = complex_gaussian_kl_closed_form(
                self.density_real, self.density_imag, self.log_s2_plain,
                m_0_real_plain, m_0_imag_plain, self.eta_plain,
                sigma2_min=self.sigma2_min,
            )
            total = total + float(self.num_points) * kl_p.mean()
        if self.has_contrast:
            kl_c = complex_gaussian_kl_closed_form(
                self.density_real_contrast, self.density_imag_contrast,
                self.log_s2_contrast,
                m_0_real_contrast, m_0_imag_contrast, self.eta_contrast,
                sigma2_min=self.sigma2_min,
            )
            total = total + float(self.num_points) * kl_c.mean()
        return total

    def kl_geometry(
        self,
        sigma2_p_mu: float = 1e-3,
        sigma2_p_ell: float = 1e-3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``L^μ_KL`` + ``L^ℓ_KL`` with isotropic prior ``N(0, σ²_p I)``.

        Geometry posterior is SHARED across channels (CONTEXT.md decision A —
        single-channel runs still account for geometry KL once, not zero, not
        double). Posterior mean is zero (perturbations δμ / δℓ are centered).

        Returns ``(kl_mu, kl_ell)`` — both scalar tensors with grad.
        """
        s2_mu = self._decode_s2(self.log_sigma2_mu)
        s2_ell = self._decode_s2(self.log_sigma2_ell)
        # 3D isotropic Gaussian KL with q-mean = 0:
        # (3/2) * (s²/σ²_p − 1 + log(σ²_p / s²))
        log_sigma2_p_mu = torch.log(torch.tensor(sigma2_p_mu, device=s2_mu.device))
        log_sigma2_p_ell = torch.log(torch.tensor(sigma2_p_ell, device=s2_ell.device))
        kl_mu_per = 1.5 * (
            s2_mu / sigma2_p_mu - 1.0
            + log_sigma2_p_mu - torch.log(s2_mu.clamp_min(1e-38))
        )
        kl_ell_per = 1.5 * (
            s2_ell / sigma2_p_ell - 1.0
            + log_sigma2_p_ell - torch.log(s2_ell.clamp_min(1e-38))
        )
        return (
            float(self.num_points) * kl_mu_per.mean(),
            float(self.num_points) * kl_ell_per.mean(),
        )

    # --------------------------------------------------------------------- #
    # B decision: load_state_dict tolerates v1 ckpts                          #
    # --------------------------------------------------------------------- #

    def load_state_dict(self, state_dict, strict=False):
        """Override forces ``strict=False`` by default (B decision step 1).

        - v1 ckpt missing 6 v2 ``nn.Parameter`` → kept at ``__init__`` defaults,
          surfaced via ``_IncompatibleKeys.missing_keys``.
        - Single-channel-mode load of a dual-mode ckpt → opposite-side
          ``*_contrast`` / ``*_plain`` fields land in ``unexpected_keys`` and
          are silently ignored (no raise).
        """
        return super().load_state_dict(state_dict, strict=strict)

    # --------------------------------------------------------------------- #
    # W-1 fix per revision loop 1: amplitude_difference deprecation          #
    # --------------------------------------------------------------------- #

    def amplitude_difference(self) -> torch.Tensor:
        """v1 A-LOSS-DROP method, REMOVED in Phase 7 INF-05.

        v2 models cross-contrast amplitude relationships via the posterior
        ``q(a_{n,c}) = CN(m, s²)`` divergence (see
        ``ElboLoss._kl_amp_with_free_bits`` in 08-02). Calling this directly
        is a programming error and is surfaced loudly.
        """
        raise DeprecationWarning(
            "v2 A-LOSS-DROP: amplitude_difference() removed in Phase 7 INF-05. "
            "v2 uses posterior q(a_{n,c}) = CN(m, s^2) divergence via "
            "ElboLoss._kl_amp_with_free_bits — see 08-02 ElboLoss."
        )

    # --------------------------------------------------------------------- #
    # PITFALLS A-06 + Phase 7 INF-06: densification sync                     #
    # --------------------------------------------------------------------- #

    def _update_params(
        self,
        mask: torch.Tensor,
        new_pos: Optional[torch.Tensor] = None,
        new_scale: Optional[torch.Tensor] = None,
        new_rot: Optional[torch.Tensor] = None,
        new_dr: Optional[torch.Tensor] = None,
        new_di: Optional[torch.Tensor] = None,
        is_prune: bool = False,
    ):
        # Snapshot extra-field data BEFORE the parent reassigns nn.Parameters.
        # Parent's _update_params uses ``mask`` as the keep-mask; in split path
        # the appended children come from rows at ``~mask`` (the split parents),
        # each parent contributing two children (matching parent's two-shift
        # ``parent_positions + global_shift`` / ``- global_shift`` cat).
        old_extra = {
            name: getattr(self, name).data.clone()
            for name in self._EXTRA_DENSIFY_FIELDS
        }

        # Run the v1 parent path — syncs shared geometry, plain density, and
        # contrast density. This subclass only mirrors the v2 posterior fields.
        super()._update_params(mask, new_pos, new_scale, new_rot, new_dr, new_di, is_prune=is_prune)

        # ---- v2 extra posteriors ----
        for field in self._EXTRA_DENSIFY_FIELDS:
            old_data = old_extra[field]
            if is_prune:
                new_data = old_data[mask]
            else:
                kept = old_data[mask]
                if (~mask).any():
                    parent_vals = old_data[~mask]
                    children = parent_vals.repeat_interleave(2)
                else:
                    children = old_data.new_empty(0)
                n_appended = new_pos.shape[0] if new_pos is not None else 0
                children = _align_children(children, n_appended)
                new_data = torch.cat([kept, children], dim=0)
            setattr(self, field, nn.Parameter(new_data))

    # --------------------------------------------------------------------- #
    # F decision: Adam param groups (active channels only)                   #
    # --------------------------------------------------------------------- #

    def get_optimizable_params(
        self,
        lr_position: float = 1e-4,
        lr_density: float = 1e-3,
        lr_scale: float = 5e-4,
        lr_rotation: float = 1e-4,
    ) -> List[Dict[str, Any]]:
        """Active-channel-aware param-group list.

        Parent ``JointGaussianModel3D.get_optimizable_params`` returns 7 groups
        but unconditionally accesses ``density_real_contrast`` /
        ``density_imag_contrast`` — single-channel mode UNREGISTERed them so
        the parent call would AttributeError. We rebuild the v1 group list
        manually here based on active channels.

        Returns:
            dual:           13 groups (7 v1 + 6 v2)
            single 'plain': 9  groups (5 v1 + 4 v2)
            single 'contrast': 9 groups (5 v1 + 4 v2)
        """
        v1_groups: List[Dict[str, Any]] = [
            {"params": [self.positions], "lr": lr_position, "name": "positions"},
            {"params": [self.scales], "lr": lr_scale, "name": "scales"},
            {"params": [self.rotations], "lr": lr_rotation, "name": "rotations"},
        ]
        if self.has_plain:
            v1_groups += [
                {"params": [self.density_real], "lr": lr_density, "name": "density_real"},
                {"params": [self.density_imag], "lr": lr_density, "name": "density_imag"},
            ]
        if self.has_contrast:
            v1_groups += [
                {"params": [self.density_real_contrast], "lr": lr_density,
                 "name": "density_real_contrast"},
                {"params": [self.density_imag_contrast], "lr": lr_density,
                 "name": "density_imag_contrast"},
            ]
        v2_groups: List[Dict[str, Any]] = [
            {"params": [self.log_sigma2_mu], "lr": lr_position, "name": "log_sigma2_mu"},
            {"params": [self.log_sigma2_ell], "lr": lr_position, "name": "log_sigma2_ell"},
        ]
        if self.has_plain:
            v2_groups += [
                {"params": [self.log_s2_plain], "lr": lr_density, "name": "log_s2_plain"},
                # CONTEXT.md decision F: EB prior eta uses lr_density * 0.1.
                {"params": [self.eta_plain], "lr": lr_density * 0.1, "name": "eta_plain"},
            ]
        if self.has_contrast:
            v2_groups += [
                {"params": [self.log_s2_contrast], "lr": lr_density,
                 "name": "log_s2_contrast"},
                {"params": [self.eta_contrast], "lr": lr_density * 0.1,
                 "name": "eta_contrast"},
            ]
        return v1_groups + v2_groups


# --------------------------------------------------------------------------- #
# Internal helpers (module-private)                                          #
# --------------------------------------------------------------------------- #


def _align_children(children: torch.Tensor, n_appended: int) -> torch.Tensor:
    """Pad-or-truncate ``children`` to exactly ``n_appended`` rows."""
    if children.shape[0] == n_appended:
        return children
    if children.shape[0] < n_appended:
        pad_shape = list(children.shape)
        pad_shape[0] = n_appended - children.shape[0]
        pad = torch.zeros(
            pad_shape, device=children.device, dtype=children.dtype,
        )
        return torch.cat([children, pad], dim=0)
    return children[:n_appended]
