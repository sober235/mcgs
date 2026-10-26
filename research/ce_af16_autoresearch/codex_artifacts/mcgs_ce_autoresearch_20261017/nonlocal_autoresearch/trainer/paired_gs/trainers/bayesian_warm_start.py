"""Bayesian warm-start helper (BP-11 / B decision / A-DET-WARM).

Two-step explicit protocol per CONTEXT.md decision B:

Step 1 — load v1 ckpt with ``strict=False``
    ``model.load_state_dict(torch.load(ckpt_path), strict=False)``
    v1 geometry + amplitude byte-identical load. The 6 v2 ``nn.Parameter`` keys
    are absent from the v1 state_dict → captured as ``missing_keys`` by
    ``BayesianJointGaussianModel3D.load_state_dict`` (which defaults to
    ``strict=False``).

Step 2 — overwrite ``log_s2_*`` / ``log_sigma2_*`` to make the posterior std
    concentrate around the prior mean ``m_0 = â^det_{n,c}``:
    ``log_s2 = softplus_inv(sigma2_init - σ²_min)``
    so ``s² = σ²_min + softplus(log_s2) ≈ sigma2_init``.

The Empirical Bayes prior parameter ``η`` is NOT touched here — it keeps its
``__init__`` default (G decision, ``log_η_init = -1.0``). Without this gap
between ``s²_init`` (~1e-3) and ``σ²_p_init`` (~0.313) the KL term would
vanish at init, and the EB prior would never learn (A-04 sham convergence).

Single-channel mode support (B-1 联动 fix per revision loop 1)
-----------------------------------------------------------
The helper inspects ``model.has_plain`` and ``model.has_contrast`` and only
writes ``log_s2_*`` fields that actually exist on the model. The returned
``m_0`` dict carries snapshots for the active channels only.

BP-11 mapping (dual-channel)
----------------------------
    m_0_real_plain     = ckpt['density_real']              (loaded by strict=False)
    m_0_imag_plain     = ckpt['density_imag']              (loaded by strict=False)
    m_0_real_contrast  = ckpt['density_real_contrast']     (loaded by strict=False)
    m_0_imag_contrast  = ckpt['density_imag_contrast']     (loaded by strict=False)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Union

import torch

from gaussian.bayesian_joint_model import (
    BayesianJointGaussianModel3D,
    DEFAULT_SIGMA2_MIN,
    _softplus_inv,
)


def init_posterior_from_v1_checkpoint(
    model: BayesianJointGaussianModel3D,
    ckpt_path: Union[str, Path],
    *,
    sigma2_init: float = 1e-3,
) -> Dict[str, Any]:
    """Two-step warm-start of a Bayesian model from a v1 deterministic ckpt.

    Step 1: ``model.load_state_dict(torch.load(ckpt_path), strict=False)``
    Step 2: rewrite ``log_s2_*`` so posterior std equals ``sigma2_init`` at init
            (only for active channels — B-1 联动 fix).

    Parameters
    ----------
    model : BayesianJointGaussianModel3D
        The target model. Single-channel modes are supported.
    ckpt_path : str | Path
        Path to a v1 ``.pth`` file (a dict with ``model_state`` /
        ``gaussian_state`` key, or a raw state_dict).
    sigma2_init : float
        Desired posterior variance at init. Default 1e-3.

    Returns
    -------
    dict with keys:
        ``m_0`` : Dict[str, Tensor]
            Detached snapshots of the loaded v1 density tensors. Keys are
            ``m_0_real_plain`` / ``m_0_imag_plain`` for plain channel and
            ``m_0_real_contrast`` / ``m_0_imag_contrast`` for contrast channel
            — only the active channels are populated (B-1 联动).
        ``missing_keys`` / ``unexpected_keys`` : List[str]
            For debugging; reflects what the v1 ckpt did / did not provide.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"warm-start ckpt not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # Accept three layouts: {model_state: ...}, {gaussian_state: ...}, or raw state_dict.
    state_dict = ckpt.get("model_state", ckpt.get("gaussian_state", ckpt))

    # Step 1: strict=False load (B decision step 1)
    incompat = model.load_state_dict(state_dict, strict=False)

    # Step 2: overwrite log_s2_* + log_sigma2_* for active channels (B-1 联动)
    log_s2_value = _softplus_inv(
        max(sigma2_init - model.sigma2_min, 1e-30)
    ).item()
    with torch.no_grad():
        if hasattr(model, "log_s2_plain"):
            model.log_s2_plain.data.fill_(log_s2_value)
        if hasattr(model, "log_s2_contrast"):
            model.log_s2_contrast.data.fill_(log_s2_value)
        # Geometry posteriors are always present (CONTEXT.md decision A).
        model.log_sigma2_mu.data.fill_(log_s2_value)
        model.log_sigma2_ell.data.fill_(log_s2_value)

    # Snapshot m_0 — only active channels (B-1 联动)
    m_0: Dict[str, torch.Tensor] = {}
    if model.has_plain:
        m_0["m_0_real_plain"] = model.density_real.detach().clone()
        m_0["m_0_imag_plain"] = model.density_imag.detach().clone()
    if model.has_contrast:
        m_0["m_0_real_contrast"] = model.density_real_contrast.detach().clone()
        m_0["m_0_imag_contrast"] = model.density_imag_contrast.detach().clone()

    return {
        "m_0": m_0,
        "missing_keys": list(getattr(incompat, "missing_keys", [])),
        "unexpected_keys": list(getattr(incompat, "unexpected_keys", [])),
    }
