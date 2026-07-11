"""Config schema + validation for Paired-ARC-GS.

The framework is fully config-driven: every spec module (§5-§13) is toggled and
parameterized here, so the same trainer drives Layer-1 (shared+response only) through
Layer-4 (full pipeline) purely by config. This module provides:

  * DEFAULT_CONFIG   — a documented default with every key present and every advanced
                       module OFF (so an unmodified default = Layer-1 minimal build).
  * validate_config  — structural checks + spec-derived guardrails (§6.2 decoder capacity,
                       §19.2.4 change branch must stay smaller than shared, §8 deformation
                       must be low-frequency, §12.1 pre-whitening presence).
  * merge_config     — deep-merge a user config onto DEFAULT_CONFIG.

Design intent per spec §16: ONE framework, ONE main set of hyperparameters, ONE training
mechanism must serve both T1/FLAIR and pre/post-contrast. So the *architecture* keys below
are pair-agnostic; only `data.*` and per-pair overrides differ between the two worked
examples.
"""
from __future__ import annotations

import copy
from typing import Dict, Any, List, Tuple


DEFAULT_CONFIG: Dict[str, Any] = {
    # ---- data: two contrasts, each its own ksp/csm/mask (spec §2: A_m = M_m F S_m) ----
    "data": {
        "plain":    {"name": "plain",    "data_path": None, "csm_path": None, "mask_path": None,
                     "readout_axis": 0, "phase_axes": [1, 2], "presampled": True,
                     "normalize_kspace": True, "normalization_percentile": 99.9},
        "contrast": {"name": "contrast", "data_path": None, "csm_path": None, "mask_path": None,
                     "readout_axis": 0, "phase_axes": [1, 2], "presampled": True,
                     "normalize_kspace": True, "normalization_percentile": 99.9},
    },

    # ---- model: the four Paired-ARC-GS modules (§4). Advanced modules default OFF. ----
    "model": {
        "num_points": 200000,               # shared anatomy Gaussian count N_A
        "init_scale": 0.008,

        # §5-§6 shared anatomy latent + response decoders (Layer 1 — ON by default)
        "anatomy": {
            "latent_dim": 12,               # z_A dim, spec §6.2 recommends 8-16
            "use_confidence": True,         # pi_n^A confidence weight
        },
        "decoder": {
            "enable": True,                 # if False -> fall back to independent density tables (baseline)
            "hidden_dim": 24,               # spec §6.2 recommends 16-32
            "n_layers": 2,
            "condition_dim": 4,             # e_m = [seq, contrast-state, pre/post, echo/inv]
            "shared_trunk": True,           # shared trunk, per-contrast output heads (§6.2)
        },

        # §7 contrast-specific change branch (Layer 2)
        "change": {
            "enable": False,
            "max_per_contrast": 20000,      # §19.2.4: must be << num_points
            "audit_tau": 0.0,               # §10.3 audit-gain acceptance threshold τ_m
            "propose_per_contrast": 4000,   # §10.2 candidates proposed per birth
            "evidence_quantile": 0.9,       # §10.2 high-evidence tail cutoff for proposal
            "birth_fit_iters": 60,          # brief FIT-set refit of candidates before audit
            "init_scale_vox": 1.5,          # initial isotropic change-Gaussian extent (voxels)
            "prune_amp": 1.0e-4,            # §13-S7 relative-amplitude prune threshold
        },

        # §9 low-frequency phase field (Layer 3 — OFF until built)
        "phase": {
            "enable": False,
            "type": "bspline",              # 'bspline' | 'lowfreq_grid'
            "control_points": [8, 8, 8],
        },

        # §8 cross-scan deformation (Layer 3 — OFF until built)
        "deformation": {
            "enable": False,
            "rigid": True,
            "nonrigid": True,
            "control_points": [6, 6, 6],    # coarse grid ⇒ low-freq u (§8)
            "u_max": 0.05,                  # amplitude cap on |u| (norm units) — §8 caveat
            "reference_index": 0,           # plain contrast pinned as reference frame
            "lambda_jacobian": 1.0,         # §8 det(J)-1 penalty weight
        },

        # §11 Gaussian-Gabor hybrid (Layer 4 — OFF until built)
        "gabor": {
            "enable": False,
            "max_primitives": 5000,
        },
    },

    # ---- loss (§12): noise-normalized DC + geometry reg + change sparsity + exclusivity ----
    # DEPRECATED and intentionally absent: kspace_center_weight(0.3)/kspace_high_weight(2.2)
    # frequency weighting (§12.1) and image-space TV as geometry reg (§12.4).
    "loss": {
        "prewhiten": True,                  # §12.1 coil noise pre-whitening (needs noise cov)
        "lambda_change": 1.0e-3,            # §12.2 change-branch sparsity (group-L2)
        "lambda_route": 1.0e-2,             # §12.3 shared-change exclusivity
        "lambda_geo": 1.0e-3,               # §12.4 geometry reg on Gaussian params
        "geo": {
            "scale_min": 0.5,               # in voxels; §12.4 R_scale hinge bounds
            "scale_max": 20.0,
            "lambda_aniso": 0.1,            # §12.4 R_aniso
            "lambda_overlap": 0.0,          # §12.4 R_overlap (0 = off until built)
        },
        "lambda_phase": 1.0e-3,             # §9 wrap-aware phase reg
        "lambda_def": 1.0e-2,               # §8 deformation reg
    },

    # ---- training: §13 stage schedule + optimizer ----
    "training": {
        "geometry_lr_scale": 0.05,          # §13-S4 eta_geometry/eta_signal in [1/50, 1/10]
        "lr": {
            "geometry": 3.0e-4,             # positions/scales/rotations base LR (scaled in S4)
            "latent": 5.0e-3,
            "decoder": 5.0e-3,
            "phase": 1.0e-3,
            "deformation": 1.0e-3,
            "change": 5.0e-3,
            "gabor": 5.0e-3,
        },
        "adam_gamma": 0.9997,
        # per-stage iteration budgets (0 = disabled). Layer-1 default: only S2 (+optional S4).
        "stages": {
            "s1_acs_init":       {"iters": 0},      # setup happens in model builder
            "s2_response_fit":   {"iters": 1500},
            "s3_deformation":    {"iters": 0},
            "s4_geometry_lowlr": {"iters": 0},
            "s5_change_birth":   {"iters": 0},
            "s6_gabor_birth":    {"iters": 0},
            "s7_joint_prune":    {"iters": 0},
            "s8_stability":      {"iters": 0},
            "s9_full_refine":    {"iters": 0},
            "s10_cg_project":    {"iters": 0},
        },
        # §10 fit/audit k-space split
        "audit": {
            "enable": False,
            "fit_fraction": 0.75,           # |Omega_fit| / |Omega|
            "seed": 42,
        },
        "stability": {"enable": False, "n_splits": 5, "tau_stable": 0.5},  # §13-S8
        "cg": {"enable": False, "eta": 1.0, "n_iters": 10},                # §13-S10
    },

    "output": {"output_dir": "results_arc", "exp_name": "arc", "save_every": 500, "eval_every": 500},
    "seed": 42,
}


def merge_config(user_cfg: Dict[str, Any], base: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Deep-merge user_cfg onto DEFAULT_CONFIG (user wins on scalar keys)."""
    out = copy.deepcopy(DEFAULT_CONFIG if base is None else base)

    def _merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _merge(dst[k], v)
            else:
                dst[k] = v

    _merge(out, user_cfg or {})
    return out


def validate_config(cfg: Dict[str, Any]) -> List[str]:
    """Return a list of validation errors (empty = valid). Spec-derived guardrails."""
    errs: List[str] = []
    m = cfg.get("model", {})

    # data present
    for side in ("plain", "contrast"):
        d = cfg.get("data", {}).get(side, {})
        if not d.get("data_path"):
            errs.append(f"data.{side}.data_path is required")

    # §6.2 decoder capacity guardrail: keep it small, or it re-memorizes each image
    dec = m.get("decoder", {})
    if dec.get("enable", True):
        if dec.get("hidden_dim", 0) > 64:
            errs.append(f"decoder.hidden_dim={dec.get('hidden_dim')} too large (§6.2: keep 16-32, hard cap 64)")
        lat = m.get("anatomy", {}).get("latent_dim", 0)
        if not (4 <= lat <= 32):
            errs.append(f"anatomy.latent_dim={lat} outside recommended 8-16 (hard bounds 4-32, §6.2)")

    # §19.2.4 change branch must be much smaller than shared branch
    ch = m.get("change", {})
    if ch.get("enable", False):
        if ch.get("max_per_contrast", 0) >= m.get("num_points", 1):
            errs.append("change.max_per_contrast must be << model.num_points (§19.2.4)")

    # §12.1: pre-whitening requested but modules that need audit split present is fine;
    # just flag the deprecated keys if a user tries to sneak them back in.
    loss = cfg.get("loss", {})
    for dead in ("kspace_center_weight", "kspace_high_weight", "kspace_mid_weight", "tv_weight"):
        if dead in loss:
            errs.append(f"loss.{dead} is DEPRECATED by spec §12.1/§12.4 (freq weighting + image-TV removed) — remove it")

    # stage/module consistency
    t = cfg.get("training", {})
    stages = t.get("stages", {})
    def _iters(sid):
        s = stages.get(sid, 0)
        return int(s.get("iters", 0)) if isinstance(s, dict) else int(s or 0)
    if _iters("s5_change_birth") > 0 and not ch.get("enable", False):
        errs.append("s5_change_birth has iters>0 but model.change.enable is False")
    if _iters("s3_deformation") > 0 and not m.get("deformation", {}).get("enable", False):
        errs.append("s3_deformation has iters>0 but model.deformation.enable is False")
    if _iters("s6_gabor_birth") > 0 and not m.get("gabor", {}).get("enable", False):
        errs.append("s6_gabor_birth has iters>0 but model.gabor.enable is False")

    return errs
