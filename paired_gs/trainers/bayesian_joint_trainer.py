"""BayesianJointTrainer (Phase 8 / BP-05/06/13/14).

Bayesian variational training for joint T1+FLAIR MC-GS:
  - K_train=2 MC reparam forward + per-sample backward + mean (BP-05)
  - β_a KL warmup linear 0→1.0 over warmup_kl steps after sanity window (BP-13)
  - Posterior collapse monitor TB scalars + 200-iter rolling warning (BP-13)
  - K_eval=16 inference path for posterior mean + variance (BP-06)
  - Diagnostic flag disable_mc_for_init: K=1 + frozen posterior first 200 iter (BP-14)
  - Phase 7 INF-05 schema migration (B-2/B-5 fix per revision loop 1):
    emit 6-item ratios (dc + R_geo + R_sp + kl/a + kl/mu + kl/ell). Note:
    v1 JointTrainer keeps 3-item schema (dc_p + dc_c + R_geo) — they are
    independent emit paths and tested separately.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from data.dataset import MRIDataset
from data.transforms import fft3c, ifft3c
from gaussian.bayesian_joint_model import BayesianJointGaussianModel3D
from gaussian.tile_voxelizer import TileVoxelizer
from gaussian.voxelizer import Voxelizer
from losses import ElboLoss
from trainers.bayesian_warm_start import init_posterior_from_v1_checkpoint
from trainers.voxelizer_config import apply_deterministic_voxelizer_reference


# ----------------------------------------------------------------------
def _load_dataset_split(data_cfg: Dict[str, Any]) -> MRIDataset:
    """Phase 7 INF-02: 默认 enable_mask_split=True for Phase 8."""
    return MRIDataset(
        data_path=data_cfg["data_path"],
        mask_path=data_cfg.get("mask_path"),
        csm_path=data_cfg.get("csm_path"),
        acceleration_factor=data_cfg.get("acceleration_factor"),
        mask_type=data_cfg.get("mask_type", "external"),
        readout_axis=data_cfg.get("readout_axis", 0),
        phase_axes=tuple(data_cfg.get("phase_axes", [1, 2])),
        normalize_kspace=data_cfg.get("normalize_kspace", True),
        normalization_percentile=data_cfg.get("normalization_percentile", 99.9),
        skip_readout_check=data_cfg.get("skip_readout_check", True),
        presampled=data_cfg.get("presampled", True),
        enable_mask_split=bool(data_cfg.get("enable_mask_split", True)),
        mask_split_seed=int(data_cfg.get("mask_split_seed", 42)),
        mask_split_out_dir=data_cfg.get("mask_split_out_dir"),
    )


# ----------------------------------------------------------------------
class BayesianJointTrainer:
    """Bayesian variational joint plain+contrast trainer (Phase 8 / BP-05..14).

    Distinct from v1 ``JointTrainer`` (which keeps deterministic 3-ratio
    schema). This class consumes ``BayesianJointGaussianModel3D`` +
    ``ElboLoss`` and emits the Phase 8 6-ratio schema directly from
    ``ElboLoss`` output (B-2/B-5 fix per revision loop 1).
    """

    def __init__(self, config: Dict[str, Any], device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._validate_config()
        voxelizer_note = apply_deterministic_voxelizer_reference(self.config)
        if voxelizer_note:
            print(f"[DeterministicVoxelizer] {voxelizer_note}")
        self._setup_seeds()
        self._resolve_adaptive_strategy()
        self._setup_data()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_output()
        self._setup_collapse_monitor_state()
        # Phase 9 GNG-01 Layer 2: emit runtime fairness fingerprint sidecar.
        try:
            fingerprint_path = self.out_dir / "phase9_fairness_fingerprint.json"
            self.dump_fairness_fingerprint(fingerprint_path)
            print(f"[phase9] runtime fingerprint dumped: {fingerprint_path}")
        except Exception as e:
            print(f"[phase9] WARN: fingerprint dump failed: {e}")

    # ------------------------------------------------------------------ #
    def _validate_config(self):
        bcfg = self.config.get("bayesian", {})
        assert bcfg.get("K_train", 2) >= 1, "K_train >= 1"
        assert bcfg.get("warmup_kl", 500) > 0, "warmup_kl > 0"

    def _setup_seeds(self):
        train_cfg = self.config.get("training", {})
        seed = int(train_cfg.get("seed", 42))
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _resolve_adaptive_strategy(self):
        adaptive_cfg = self.config.setdefault("adaptive_control", {})
        strategy = adaptive_cfg.get("strategy", "long_axis")
        if strategy == "auto":
            data_cfg = self.config.get("data", {})
            probe_cfg = data_cfg.get("plain", data_cfg)
            acc = int(probe_cfg.get("acceleration_factor", 0))
            threshold = int(adaptive_cfg.get("high_acceleration_threshold", 10))
            self._resolved_use_long_axis = acc >= threshold
        elif strategy == "long_axis":
            self._resolved_use_long_axis = True
        else:
            self._resolved_use_long_axis = False
        # Bayesian posterior fields are synchronized by _update_params, which
        # split/prune use. The inherited clone path bypasses _update_params.
        self._resolved_use_cloning = False

    # ------------------------------------------------------------------ #
    def _setup_data(self):
        """Phase 7 INF-02 三拆 mask;trainer 只用 M^tr 的 DC loss。"""
        data_cfg = self.config["data"]
        self.ds_p = _load_dataset_split(data_cfg["plain"])
        self.ds_c = _load_dataset_split(data_cfg["contrast"])
        assert self.ds_p.volume_shape == self.ds_c.volume_shape, (
            f"plain/contrast volume_shape mismatch: "
            f"{self.ds_p.volume_shape} vs {self.ds_c.volume_shape}"
        )
        self.volume_shape = self.ds_p.volume_shape
        self.kspace_p = self.ds_p.kspace_undersampled_complex.to(self.device)
        self.kspace_c = self.ds_c.kspace_undersampled_complex.to(self.device)
        self.csm_p = self.ds_p.sensitivity_maps.to(self.device)
        self.csm_c = self.ds_c.sensitivity_maps.to(self.device)
        if self.ds_p.mask_split:
            self.mask_tr_p = self.ds_p.mask_split["tr"].to(self.device)
            self.mask_fb_p = self.ds_p.mask_split["fb"].to(self.device)
            self.mask_te_p = self.ds_p.mask_split["te"].to(self.device)
            self.mask_tr_c = self.ds_c.mask_split["tr"].to(self.device)
            self.mask_fb_c = self.ds_c.mask_split["fb"].to(self.device)
            self.mask_te_c = self.ds_c.mask_split["te"].to(self.device)
            self._mask_te_locked_for_trainer = True
        else:
            self.mask_tr_p = self.ds_p.mask.to(self.device)
            self.mask_tr_c = self.ds_c.mask.to(self.device)
            self.mask_fb_p = self.mask_fb_c = self.mask_te_p = self.mask_te_c = None
            self._mask_te_locked_for_trainer = False
        with torch.no_grad():
            self.zf_p = torch.sum(torch.conj(self.csm_p) * ifft3c(self.kspace_p), dim=0)
            self.zf_c = torch.sum(torch.conj(self.csm_c) * ifft3c(self.kspace_c), dim=0)

    # ------------------------------------------------------------------ #
    def _setup_model(self):
        gcfg = self.config["gaussian"]
        bcfg = self.config["bayesian"]
        warm_path = bcfg.get("warm_start", {}).get("ckpt_path") or gcfg.get("warm_start_checkpoint")
        if warm_path and Path(warm_path).exists():
            ckpt = torch.load(warm_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state", ckpt.get("gaussian_state", ckpt))
            n_init = sd["positions"].shape[0]
            positions = sd["positions"].to(self.device)
            scales_log = sd["scales"].to(self.device)
            rotations = sd["rotations"].to(self.device)
            d_r = sd["density_real"].to(self.device)
            d_i = sd["density_imag"].to(self.device)
            d_rc = sd.get("density_real_contrast", d_r).to(self.device)
            d_ic = sd.get("density_imag_contrast", d_i).to(self.device)
            self.model = BayesianJointGaussianModel3D(
                num_points=n_init,
                volume_shape=self.volume_shape,
                device=self.device,
                initial_positions=positions,
                initial_densities_plain=torch.complex(d_r, d_i),
                initial_densities_contrast=torch.complex(d_rc, d_ic),
                initial_scales=torch.exp(scales_log),
                sigma2_init=float(bcfg.get("sigma2_init", 1e-3)),
                log_eta_init=float(bcfg.get("log_eta_init", -1.0)),
                sigma2_min=float(bcfg.get("sigma2_min", 1e-8)),
            ).to(self.device)
            with torch.no_grad():
                self.model.rotations.copy_(rotations)
            ws_info = init_posterior_from_v1_checkpoint(
                self.model, warm_path,
                sigma2_init=float(
                    bcfg.get("warm_start", {}).get(
                        "sigma2_init", bcfg.get("sigma2_init", 1e-3)
                    )
                ),
            )
            self.m_0 = {k: v.to(self.device) for k, v in ws_info["m_0"].items()}
            for v in self.m_0.values():
                v.requires_grad_(False)
        else:
            n_init = gcfg.get("initial_num_points", 200000)
            d, h, w = self.volume_shape
            positions = torch.rand(n_init, 3, device=self.device) * 2 - 1
            with torch.no_grad():
                ix = ((positions[:, 0] + 1) * 0.5 * d).clamp(0, d - 1).long()
                iy = ((positions[:, 1] + 1) * 0.5 * h).clamp(0, h - 1).long()
                iz = ((positions[:, 2] + 1) * 0.5 * w).clamp(0, w - 1).long()
                d_p = self.zf_p[ix, iy, iz].to(torch.complex64) * 0.1
                d_c = self.zf_c[ix, iy, iz].to(torch.complex64) * 0.1
            base_scale = gcfg.get("init_scale", 2.0 / max(d, h, w))
            scales = torch.ones(n_init, 3, device=self.device) * base_scale
            self.model = BayesianJointGaussianModel3D(
                num_points=n_init,
                volume_shape=self.volume_shape,
                device=self.device,
                initial_positions=positions,
                initial_densities_plain=d_p,
                initial_densities_contrast=d_c,
                initial_scales=scales,
                sigma2_init=float(bcfg.get("sigma2_init", 1e-3)),
                log_eta_init=float(bcfg.get("log_eta_init", -1.0)),
                sigma2_min=float(bcfg.get("sigma2_min", 1e-8)),
            ).to(self.device)
            self.m_0 = {
                "m_0_real_plain": self.model.density_real.detach().clone(),
                "m_0_imag_plain": self.model.density_imag.detach().clone(),
                "m_0_real_contrast": self.model.density_real_contrast.detach().clone(),
                "m_0_imag_contrast": self.model.density_imag_contrast.detach().clone(),
            }
            for v in self.m_0.values():
                v.requires_grad_(False)
        # voxelizer (same convention as v1)
        vcfg = self.config.get("voxelizer", {})
        vtype = vcfg.get("type", "tile_cuda")
        if vtype in {"tile", "tile_cuda"}:
            self.voxelizer = TileVoxelizer(
                volume_shape=self.volume_shape,
                tile_size=vcfg.get("tile_size", 8),
                max_radius=vcfg.get("max_radius", 20),
                use_cuda=(vtype == "tile_cuda"),
                strict_cuda=bool(self.config.get("mode", {}).get("strict_cuda", False))
                            and vtype == "tile_cuda",
                device=str(self.device),
            ).to(self.device)
        elif vtype == "chunk":
            self.voxelizer = Voxelizer(
                volume_shape=self.volume_shape,
                device=str(self.device),
                max_radius=vcfg.get("max_radius", 20),
            ).to(self.device)
        else:
            raise ValueError(f"unsupported voxelizer.type: {vtype!r}")

    # ------------------------------------------------------------------ #
    def _setup_loss(self):
        bcfg = self.config["bayesian"]
        beta_a_final = float(bcfg.get("beta_a", 1.0))
        self.elbo_loss = ElboLoss(
            beta_a=beta_a_final,
            beta_mu=beta_a_final * float(bcfg.get("beta_mu_ratio", 0.1)),
            beta_ell=beta_a_final * float(bcfg.get("beta_ell_ratio", 0.1)),
            lambda_geo=float(bcfg.get("lambda_geo", 5e-4)),
            lambda_sp=float(bcfg.get("lambda_sp", 5e-6)),
            sigma2_p_mu=float(bcfg.get("sigma2_p_mu", 1e-3)),
            sigma2_p_ell=float(bcfg.get("sigma2_p_ell", 1e-3)),
            enable_free_bits=bool(bcfg.get("enable_free_bits", False)),
            free_bits_kappa=float(bcfg.get("free_bits_kappa", 0.02)),
            kl_normalization=str(bcfg.get("kl_normalization", "sum")),
        )

    # ------------------------------------------------------------------ #
    def _setup_optimizer(self):
        gcfg = self.config["gaussian"]
        params = self.model.get_optimizable_params(
            lr_position=gcfg.get("position_lr", 1e-4),
            lr_density=gcfg.get("density_lr", 1e-3),
            lr_scale=gcfg.get("scale_lr", 5e-4),
            lr_rotation=gcfg.get("rotation_lr", 1e-4),
        )
        self.optimizer = torch.optim.Adam(params)
        sched_cfg = self.config.get("training", {}).get("lr_scheduler", {})
        if sched_cfg.get("type") == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=sched_cfg.get("gamma", 0.9995)
            )
        else:
            self.scheduler = None

    # ------------------------------------------------------------------ #
    def _setup_output(self):
        out_cfg = self.config.get("output", {})
        out_root = out_cfg.get("output_dir", "./results")
        exp = out_cfg.get("exp_name", "hyw_bayesian_v2")
        self.out_dir = Path(out_root) / exp
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        use_tb = bool(out_cfg.get("use_tensorboard", True))
        self.writer = SummaryWriter(log_dir=str(self.out_dir / "logs")) if use_tb else None

    # ------------------------------------------------------------------ #
    def _setup_collapse_monitor_state(self):
        self._collapse_window_p: List[int] = []
        self._collapse_window_c: List[int] = []
        mcfg = self.config["bayesian"]["collapse_monitor"]
        self._collapse_log_every = int(mcfg.get("log_every", 100))
        self._collapse_warn_p50_factor = float(mcfg.get("warn_p50_factor", 1.05))
        self._collapse_warn_rolling_iters = int(mcfg.get("warn_rolling_iters", 200))

    # ------------------------------------------------------------------ #
    # Phase 9 GNG-01 Layer-2 runtime fairness fingerprint                #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def dump_fairness_fingerprint(self, out_path) -> None:
        """Phase 9 GNG-01 Layer-2 runtime fairness check.

        Serializes the instantiated optimizer's param_groups (sorted by group
        name) + key Bayesian hyperparameters to JSON. The Phase 9 GNG runner
        consumes this file from every run's output_dir and cross-checks across
        methods.
        """
        import json
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        groups: List[Dict[str, Any]] = []
        for g in self.optimizer.param_groups:
            groups.append({
                "name": g.get("name", "<unnamed>"),
                "lr": float(g["lr"]),
                "betas": list(g.get("betas", (0.9, 0.999))),
                "weight_decay": float(g.get("weight_decay", 0.0)),
                "n_params": sum(p.numel() for p in g["params"]),
            })
        groups.sort(key=lambda g: g["name"])
        record = {
            "phase9_fingerprint_schema_version": 1,
            "model_class": type(self.model).__name__,
            "trainer_class": type(self).__name__,
            "n_params_total": sum(g["n_params"] for g in groups),
            "optimizer_param_groups": groups,
            "K_train": int(self.config.get("bayesian", {}).get("K_train", -1)),
            "max_iterations": int(self.config["training"]["max_iterations"]),
            "seed": int(self.config["training"]["seed"]),
            "beta_a_target": float(self.config.get("bayesian", {}).get("beta_a", float("nan"))),
            "warmup_kl": int(self.config.get("bayesian", {}).get("warmup_kl", -1)),
            "kl_normalization": str(
                self.config.get("bayesian", {}).get("kl_normalization", "sum")
            ),
            "stopping": dict(self.config.get("stopping", {})),
        }
        with open(out_path, "w") as fh:
            json.dump(record, fh, indent=2)

    # ------------------------------------------------------------------ #
    # Phase 9 plan 09-02 Task 1: CONTEXT.md decision H — merge Phase 8   #
    # SC#2/#3 closure into Phase 9 Method B runs by dumping the same     #
    # 13-field health record produced by scripts/phase8_smoke_train.py.  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate_for_phase8_sc(self, out_path=None) -> Dict[str, Any]:
        """Phase 9 CONTEXT.md decision H: dump Phase 8 SC#2/#3 health fields.

        Reuses ``scripts/phase8_smoke_train.py::_compute_phase8_health_record``
        so Method B Phase 9 runs (5 seeds × 16000 iter) auto-emit the same
        13-field schema as the Phase 8 standalone smoke script. Phase 9 GNG
        analyzer then cross-aggregates the 5 seed reports and retro-updates
        Phase 8 VERIFICATION.md SC#2/#3 status: ``human_needed → passed``.

        Parameters
        ----------
        out_path : str | Path | None, optional
            Destination JSON path. Defaults to
            ``<out_dir>/phase8_smoke_report.json``.

        Returns
        -------
        Dict[str, Any]
            The 13-field record (also written to ``out_path``).
        """
        import json
        if out_path is None:
            out_path = self.out_dir / "phase8_smoke_report.json"
        else:
            out_path = Path(out_path)
        # Late import to avoid circular: scripts/ is sibling of trainers/.
        import sys
        scripts_dir = (Path(__file__).resolve().parent.parent / "scripts")
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from phase8_smoke_train import _compute_phase8_health_record
        tb_log_dir = Path(self.out_dir) / "logs"
        ckpt_path = Path(self.out_dir) / "checkpoints" / "latest.pth"
        warmup_kl = int(self.config.get("bayesian", {}).get("warmup_kl", 500))
        diagnostics = self.config.get("bayesian", {}).get("diagnostics", {})
        sanity_iters = (
            int(diagnostics.get("sanity_iters", 0))
            if diagnostics.get("disable_mc_for_init", False)
            else 0
        )
        record = _compute_phase8_health_record(tb_log_dir, ckpt_path, warmup_kl, sanity_iters)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(record, fh, indent=2)
        return record

    # ------------------------------------------------------------------ #
    # MC sampling + rendering helpers                                    #
    # ------------------------------------------------------------------ #

    def _render_sample(self, samples: Dict[str, torch.Tensor], k: int, channel: str
                       ) -> torch.Tensor:
        """Voxelize one MC sample for ``channel`` in {'plain', 'contrast'}.

        ``samples`` keys (post-08-01 sample()): ``a_plain`` / ``a_contrast`` (if
        active), ``delta_mu``, ``delta_log_scales`` — all leading dim K.

        ``model.scales`` is log-parameterized (GaussianModel3D._init_parameters);
        we add the log-space perturbation then ``.exp()`` to recover linear
        scale before passing to the voxelizer (matches v1 convention).
        """
        delta_mu_k = samples["delta_mu"][k]
        delta_log_scales_k = samples["delta_log_scales"][k]
        positions_k = self.model.positions + delta_mu_k
        scales_k = (self.model.scales + delta_log_scales_k).exp()
        key = "a_plain" if channel == "plain" else "a_contrast"
        density_k = samples[key][k]
        vol = self.voxelizer(
            positions=positions_k,
            scales=scales_k,
            rotations=self.model.rotations,
            density=density_k,
        )
        if isinstance(vol, tuple):
            return torch.complex(vol[0], vol[1])
        return vol

    def _kspace_dc_loss(
        self,
        vol: torch.Tensor,
        csm: torch.Tensor,
        ksp_meas: torch.Tensor,
        mask_tr: torch.Tensor,
    ) -> torch.Tensor:
        """``||(A(x) - y) ⊙ M^tr||² / |M^tr|`` — Phase 8 / Phase 7 INF-02 protocol."""
        pred_ksp = fft3c(vol.unsqueeze(0) * csm)
        diff = (pred_ksp - ksp_meas) * mask_tr.unsqueeze(0)
        return (diff.real.pow(2) + diff.imag.pow(2)).sum() / max(mask_tr.sum().item(), 1.0)

    def _hard_dc(
        self,
        vol: torch.Tensor,
        csm: torch.Tensor,
        ksp_meas: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Hard data consistency on acquired samples."""
        pred_ksp = fft3c(vol.unsqueeze(0) * csm)
        mask_c = mask.unsqueeze(0)
        ksp_dc = pred_ksp * (1 - mask_c) + ksp_meas * mask_c
        coil = ifft3c(ksp_dc)
        return torch.sum(torch.conj(csm) * coil, dim=0)

    def _beta_a_schedule(self, it: int, warmup_kl: int, beta_a_final: float) -> float:
        """Linear ramp 0 → beta_a_final over warmup_kl active-KL iter (BP-13)."""
        if warmup_kl <= 0:
            return beta_a_final
        return min(beta_a_final, beta_a_final * float(max(it, 0)) / float(warmup_kl))

    # ------------------------------------------------------------------ #
    # Adaptive density control for Bayesian trainers                     #
    # ------------------------------------------------------------------ #

    def _bayesian_density_magnitude(self) -> torch.Tensor:
        mags: List[torch.Tensor] = []
        if getattr(self.model, "has_plain", False):
            mags.append(torch.complex(self.model.density_real, self.model.density_imag).abs())
        if getattr(self.model, "has_contrast", False):
            mags.append(torch.complex(
                self.model.density_real_contrast,
                self.model.density_imag_contrast,
            ).abs())
        if not mags:
            return torch.zeros(self.model.num_points, device=self.device)
        return torch.stack(mags, dim=0).amax(dim=0)

    def _adaptive_density_control(self, iteration: int) -> Dict[str, int]:
        adaptive_cfg = self.config.get("adaptive_control", {})
        if not adaptive_cfg.get("enable", False):
            return {"split": 0, "clone": 0, "prune": 0}
        start_iter = int(adaptive_cfg.get("densify_from_iter", 100))
        end_iter = int(adaptive_cfg.get("densify_until_iter", 2500))
        interval = int(adaptive_cfg.get("densify_every", 100))
        first_event_iter = (
            interval
            if start_iter <= 0
            else ((start_iter + interval - 1) // interval) * interval
        )
        if (
            interval <= 0
            or iteration < first_event_iter
            or iteration > end_iter
            or iteration % interval != 0
        ):
            return {"split": 0, "clone": 0, "prune": 0}

        gaussian_cfg = self.config.get("gaussian", {})
        configured_max_points = gaussian_cfg.get("max_num_points")
        if configured_max_points is None:
            # Warm-start checkpoints can already exceed the legacy 400k cap.
            # In that case, default to 2x current size so adaptive control
            # remains active instead of silently disabling split/prune.
            max_num_points = max(400000, int(self.model.num_points * 2))
        else:
            max_num_points = int(configured_max_points)

        stats = {"split": 0, "clone": 0, "prune": 0}
        n_before = self.model.num_points
        did_modify = False
        densify_step_index = (iteration - first_event_iter) // interval
        is_split_iter = densify_step_index % 2 == 0

        if is_split_iter:
            if self.model.positions.grad is None:
                return stats
            grad_norm = torch.norm(self.model.positions.grad.detach(), dim=-1)
            grad_threshold = float(adaptive_cfg.get("grad_threshold", 0.01))
            scale_threshold = float(adaptive_cfg.get("scale_threshold", 0.01))
            max_split_fraction = float(adaptive_cfg.get("max_split_fraction", 0.25))
            remaining_slots = max_num_points - self.model.num_points
            if remaining_slots <= 0:
                return stats
            max_split_fraction = min(
                max_split_fraction,
                remaining_slots / max(float(self.model.num_points), 1.0),
            )
            long_axis_offset_factor = float(
                self.config.get("gaussian", {}).get("long_axis_offset_factor", 1.0)
            )
            stats["split"] = self.model.densify_and_split(
                grads=grad_norm,
                grad_threshold=grad_threshold,
                scale_threshold=scale_threshold,
                use_long_axis_splitting=self._resolved_use_long_axis,
                long_axis_offset_factor=long_axis_offset_factor,
                max_split_fraction=max_split_fraction,
            )
            did_modify = stats["split"] > 0
        else:
            opacity_threshold = float(adaptive_cfg.get("opacity_threshold", 0.001))
            max_scale_limit = float(adaptive_cfg.get("max_scale", 0.5))
            scales = self.model.get_scale_values()
            max_scale = scales.max(dim=-1)[0]
            density_mag = self._bayesian_density_magnitude()
            prune_mask = (density_mag < opacity_threshold) | (max_scale > max_scale_limit)
            keep_mask = ~prune_mask
            if keep_mask.sum() >= 100 and prune_mask.sum() > 0:
                self.model._update_params(keep_mask, is_prune=True)
                stats["prune"] = int(prune_mask.sum().item())
                did_modify = True

        if did_modify:
            self._sync_m0_after_density_control()
            self._rebuild_optimizer()
            n_after = self.model.num_points
            if self.writer is not None:
                self.writer.add_scalar("gaussian/num_points", n_after, iteration)
                self.writer.add_scalar("gaussian/n_split_per_event", stats["split"], iteration)
                self.writer.add_scalar("gaussian/n_prune_per_event", stats["prune"], iteration)
            print(
                f"[Bayesian Densify iter={iteration}] Gaussians: "
                f"{n_before} -> {n_after}, split={stats['split']}, prune={stats['prune']}"
            )
        return stats

    def _sync_m0_after_density_control(self) -> None:
        if not hasattr(self, "m_0"):
            return
        keep_mask = getattr(self.model, "_last_densify_keep_mask", None)
        is_prune = getattr(self.model, "_last_densify_is_prune", False)
        if keep_mask is None:
            return

        synced: Dict[str, torch.Tensor] = {}
        for key, value in self.m_0.items():
            value = value.to(device=keep_mask.device)
            if is_prune:
                new_value = value[keep_mask]
            else:
                kept = value[keep_mask]
                parents = value[~keep_mask]
                n_appended = int(self.model.num_points - int(keep_mask.sum().item()))
                children = parents.repeat_interleave(2, dim=0)
                if children.shape[0] != n_appended:
                    children = children[:n_appended]
                new_value = torch.cat([kept, children], dim=0)
            new_value = new_value.detach().clone()
            new_value.requires_grad_(False)
            if new_value.shape[0] != self.model.num_points:
                raise RuntimeError(
                    f"m_0 sync failed for {key}: {new_value.shape[0]} "
                    f"!= model.num_points {self.model.num_points}"
                )
            synced[key] = new_value
        self.m_0 = synced

    def _rebuild_optimizer(self):
        gcfg = self.config["gaussian"]
        params = self.model.get_optimizable_params(
            lr_position=gcfg.get("position_lr", 1e-4),
            lr_density=gcfg.get("density_lr", 1e-3),
            lr_scale=gcfg.get("scale_lr", 5e-4),
            lr_rotation=gcfg.get("rotation_lr", 1e-4),
        )
        current_lrs = [group["lr"] for group in self.optimizer.param_groups]
        old_optimizer = self.optimizer
        self.optimizer = torch.optim.Adam(params)
        for group, lr in zip(self.optimizer.param_groups, current_lrs):
            group["lr"] = lr
            group["initial_lr"] = lr

        keep_mask = getattr(self.model, "_last_densify_keep_mask", None)
        old_params_map = getattr(self.model, "_last_densify_old_params", None)
        is_prune = getattr(self.model, "_last_densify_is_prune", False)
        if keep_mask is not None and old_params_map is not None:
            from trainers.trainer import GaussianTrainer
            GaussianTrainer._transplant_adam_state(
                old_optimizer, old_params_map, keep_mask, is_prune, self.optimizer
            )
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p in self.optimizer.state and "exp_avg" in self.optimizer.state[p]:
                        assert self.optimizer.state[p]["exp_avg"].shape == p.shape
                        assert self.optimizer.state[p]["exp_avg_sq"].shape == p.shape
            self.model._last_densify_old_params = None
            self.model._last_densify_keep_mask = None

        sched_cfg = self.config.get("training", {}).get("lr_scheduler", {})
        if sched_cfg.get("type") == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=sched_cfg.get("gamma", 0.9995)
            )
        elif sched_cfg:
            self.scheduler = None

    # ------------------------------------------------------------------ #
    # Phase 9 B-1: plateau-eval helper (PSNR on posterior mean K_eval=16) #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _eval_psnr_for_plateau(self, K_eval: int = 16) -> float:
        """Phase 9 B-1: posterior-mean PSNR for MetricPlateauStopper.

        Method B (dual channel) average PSNR(plain) and PSNR(contrast) -- a
        single scalar is enough for plateau detection (gradients of the two
        channels are tightly coupled by the shared geometry).

        Method A (BayesianSingleTrainer subclass) overrides this with a
        single-channel variant.
        """
        from metrics import compute_psnr
        xs_p: List[torch.Tensor] = []
        xs_c: List[torch.Tensor] = []
        for _ in range(K_eval):
            samples_k = self.model.sample(K=1, disable_mc_for_init=False)
            vol_p_k = self._render_sample(samples_k, 0, "plain")
            vol_c_k = self._render_sample(samples_k, 0, "contrast")
            xs_p.append(vol_p_k.detach())
            xs_c.append(vol_c_k.detach())
        x_bar_p = torch.stack(xs_p, dim=0).mean(dim=0)
        x_bar_c = torch.stack(xs_c, dim=0).mean(dim=0)
        psnr_p = compute_psnr(torch.abs(x_bar_p), torch.abs(self.zf_p))
        psnr_c = compute_psnr(torch.abs(x_bar_c), torch.abs(self.zf_c))
        return float(0.5 * (psnr_p + psnr_c))

    # ------------------------------------------------------------------ #
    # Training loop                                                      #
    # ------------------------------------------------------------------ #

    def train(self):
        bcfg = self.config["bayesian"]
        train_cfg = self.config["training"]
        max_iter = train_cfg.get("max_iterations", 4000)
        log_every = train_cfg.get("log_every", 50)
        save_every = train_cfg.get("save_every", 500)
        K_train = int(bcfg.get("K_train", 2))
        warmup_kl = int(bcfg.get("warmup_kl", 500))
        beta_a_final = float(bcfg.get("beta_a", 1.0))
        beta_mu_ratio = float(bcfg.get("beta_mu_ratio", 0.1))
        beta_ell_ratio = float(bcfg.get("beta_ell_ratio", 0.1))
        sanity_iters = int(bcfg["diagnostics"].get("sanity_iters", 200))
        disable_mc_for_init = bool(bcfg["diagnostics"].get("disable_mc_for_init", True))
        grad_clip_amp = float(train_cfg.get("grad_clip_per_param_amp", 100.0))
        max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        # Phase 9 B-1: instantiate MetricPlateauStopper from config.stopping schema.
        stop_cfg = self.config.get("stopping", {})
        from metrics import MetricPlateauStopper
        if stop_cfg:
            self._stopper = MetricPlateauStopper(
                patience_evals=int(stop_cfg.get("patience_evals", 4)),
                min_iterations_before_stop=int(stop_cfg.get("min_iterations_before_stop", 1000)),
                metric_min_delta=dict(stop_cfg.get("metric_min_delta", {"psnr": 0.01})),
                best_metric=str(stop_cfg.get("best_metric", "psnr")),
            )
            print(
                f"[phase9] MetricPlateauStopper armed: patience_evals="
                f"{self._stopper.patience_evals}, "
                f"min_iter={self._stopper.min_iterations_before_stop}, "
                f"delta={self._stopper.metric_min_delta}"
            )
        else:
            self._stopper = None
        early_stop_iter: Optional[int] = None
        plateau_eval_every = int(stop_cfg.get("plateau_eval_every", 1000)) if stop_cfg else 1000
        K_eval_plateau = int(bcfg.get("K_eval", 16))
        log_path = self.out_dir / "bayesian_train.log"
        log_f = open(log_path, "w")
        t0 = time.time()
        best_loss = float("inf")
        for it in range(1, max_iter + 1):
            in_sanity_window = disable_mc_for_init and (it <= sanity_iters)
            K_this_iter = 1 if in_sanity_window else K_train
            if in_sanity_window:
                beta_a_now = 0.0
            else:
                beta_iter = it - sanity_iters if disable_mc_for_init else it
                beta_a_now = self._beta_a_schedule(beta_iter, warmup_kl, beta_a_final)
            beta_mu_now = beta_a_now * beta_mu_ratio
            beta_ell_now = beta_a_now * beta_ell_ratio
            self.optimizer.zero_grad()
            samples = self.model.sample(K=K_this_iter, disable_mc_for_init=in_sanity_window)
            # B-2/B-5 fix: sum dc_p + dc_c into a single l_dc_mc scalar fed to ElboLoss,
            # which emits 6-ratio schema (dc + R_geo + R_sp + kl/a + kl/mu + kl/ell).
            l_dc_total = torch.zeros((), device=self.device)
            vol_p_mean_acc = torch.zeros_like(self.zf_p)
            vol_c_mean_acc = torch.zeros_like(self.zf_c)
            for k in range(K_this_iter):
                vol_p_k = self._render_sample(samples, k, "plain")
                vol_c_k = self._render_sample(samples, k, "contrast")
                l_dc_p_k = self._kspace_dc_loss(vol_p_k, self.csm_p, self.kspace_p, self.mask_tr_p)
                l_dc_c_k = self._kspace_dc_loss(vol_c_k, self.csm_c, self.kspace_c, self.mask_tr_c)
                l_dc_total = l_dc_total + (l_dc_p_k + l_dc_c_k)
                vol_p_mean_acc = vol_p_mean_acc + vol_p_k.detach()
                vol_c_mean_acc = vol_c_mean_acc + vol_c_k.detach()
            l_dc_mc = l_dc_total / float(K_this_iter)  # B-5: dc_p + dc_c merged single scalar
            vol_p_for_geo = vol_p_mean_acc / float(K_this_iter)
            vol_c_for_geo = vol_c_mean_acc / float(K_this_iter)
            elbo_out = self.elbo_loss(
                l_dc_mc, self.model, self.m_0, vol_p_for_geo, vol_c_for_geo,
                beta_a_override=beta_a_now,
                beta_mu_override=beta_mu_now,
                beta_ell_override=beta_ell_now,
                disable_kl=in_sanity_window,
            )
            total = elbo_out["total"]
            ratios = elbo_out["ratios"]   # 6-item: dc + R_geo + R_sp + kl/a + kl/mu + kl/ell
            # Phase 7 INF-05 assertion extended to 6-ratio (B-2/B-5 fix):
            sum_ratios = sum(ratios.values())
            balance_violation = abs(sum_ratios - 1.0) > 1e-3
            if balance_violation and it == 1:
                raise ValueError(
                    f"[BP/INF-05] loss-balance violated at iter 1: "
                    f"sum(ratios)={sum_ratios:.6f}, ratios={ratios}"
                )
            elif balance_violation:
                warnings.warn(
                    f"[BP/INF-05] loss-balance violated mid-train at iter {it}: "
                    f"sum(ratios)={sum_ratios:.6f}", stacklevel=2
                )
            total.backward()
            # PITFALLS A-03: per-param L∞ clip on amplitude posterior + global L2 clip
            with torch.no_grad():
                amp_param_names = (
                    "density_real", "density_imag",
                    "density_real_contrast", "density_imag_contrast",
                    "log_s2_plain", "log_s2_contrast",
                )
                for pname in amp_param_names:
                    if not hasattr(self.model, pname):
                        continue
                    amp_p = getattr(self.model, pname)
                    if amp_p.grad is not None:
                        amp_p.grad.data.clamp_(-grad_clip_amp, grad_clip_amp)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self._adaptive_density_control(it)
            # ---------- TB scalars ----------
            if self.writer is not None and (it == 1 or it % log_every == 0):
                for k_, v_ in elbo_out["values"].items():
                    self.writer.add_scalar(f"{k_}/value", v_, it)
                for k_, v_ in ratios.items():
                    self.writer.add_scalar(f"{k_}/ratio", v_, it)
                self.writer.add_scalar(
                    "loss/balance_violation", 1.0 if balance_violation else 0.0, it
                )
                self.writer.add_scalar("loss/total", float(total.item()), it)
                self.writer.add_scalar("kl_schedule/beta_a", beta_a_now, it)
                self.writer.add_scalar("kl_schedule/beta_mu", beta_mu_now, it)
                self.writer.add_scalar("kl_schedule/beta_ell", beta_ell_now, it)
            if it % self._collapse_log_every == 0:
                self._log_collapse_monitor(it)
            if it == 1 or it % log_every == 0:
                msg = (
                    f"[iter {it:5d}/{max_iter}] total={total.item():.4e} "
                    f"dc={elbo_out['values']['loss/dc']:.4e} "
                    f"kl/a={elbo_out['values']['kl/a']:.4e} "
                    f"β_a={beta_a_now:.3f} K={K_this_iter} sanity={in_sanity_window}"
                )
                print(msg)
                log_f.write(msg + "\n")
                log_f.flush()
            if it % save_every == 0 or it == max_iter:
                ckpt = {
                    "iter": it,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "m_0": self.m_0,
                }
                ckpt_dir = self.out_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                torch.save(ckpt, ckpt_dir / "latest.pth")
                if total.item() < best_loss:
                    best_loss = total.item()
                    torch.save(ckpt, ckpt_dir / "best.pth")
            # Phase 9 B-1: plateau eval every plateau_eval_every iter (K_eval=16).
            if (
                self._stopper is not None
                and (it % plateau_eval_every == 0)
                and (it >= self._stopper.min_iterations_before_stop)
            ):
                psnr_value = self._eval_psnr_for_plateau(K_eval=K_eval_plateau)
                if self._stopper.update(it, {"psnr": psnr_value}):
                    early_stop_iter = it
                    print(
                        f"[phase9] Plateau detected at iter={it}; "
                        f"PSNR={psnr_value:.3f}; early-stopping"
                    )
                    log_f.write(f"[phase9] early_stop_iter={it}\n")
                    log_f.flush()
                    break
                else:
                    if self.writer is not None:
                        self.writer.add_scalar("plateau/psnr_eval", psnr_value, it)
        print(f"BayesianJointTrainer done in {time.time() - t0:.1f}s")
        log_f.close()
        # Phase 9 B-1: emit early_stop.json sidecar (Phase 9 GNG runner consumer).
        import json
        with open(self.out_dir / "early_stop.json", "w") as fh:
            json.dump(
                {"early_stop_iter": early_stop_iter, "max_iterations": max_iter},
                fh,
            )
        if self.writer is not None:
            self.writer.close()

    # ------------------------------------------------------------------ #
    # Posterior collapse monitor (A-01 mitigation + BP-13)               #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _log_collapse_monitor(self, it: int):
        """每 100 iter dump P5/P50/P95(s²_*) → TB,并维护 200-iter rolling collapse_warn.

        W-2 fix per revision loop 1: variable name ``warm_thresh`` -> ``warn_thresh``.
        """
        if self.writer is None:
            return
        s2_plain = self.model._decode_s2(self.model.log_s2_plain)
        s2_contrast = self.model._decode_s2(self.model.log_s2_contrast)
        s2_mu = self.model._decode_s2(self.model.log_sigma2_mu)
        for name, t in (
            ("s2_plain", s2_plain),
            ("s2_contrast", s2_contrast),
            ("log_sigma2_mu", s2_mu),
        ):
            p05 = torch.quantile(t.detach(), 0.05).item()
            p50 = torch.quantile(t.detach(), 0.50).item()
            p95 = torch.quantile(t.detach(), 0.95).item()
            self.writer.add_scalar(f"posterior/{name}/p05", p05, it)
            self.writer.add_scalar(f"posterior/{name}/p50", p50, it)
            self.writer.add_scalar(f"posterior/{name}/p95", p95, it)
        warn_thresh = self._collapse_warn_p50_factor * self.model.sigma2_min  # W-2 fix: warm_thresh -> warn_thresh
        p50_p = torch.quantile(s2_plain.detach(), 0.5).item()
        p50_c = torch.quantile(s2_contrast.detach(), 0.5).item()
        self._collapse_window_p.append(1 if p50_p < warn_thresh else 0)
        self._collapse_window_c.append(1 if p50_c < warn_thresh else 0)
        window_size = max(self._collapse_warn_rolling_iters // self._collapse_log_every, 1)
        self._collapse_window_p = self._collapse_window_p[-window_size:]
        self._collapse_window_c = self._collapse_window_c[-window_size:]
        warn_p = 1 if len(self._collapse_window_p) == window_size and all(self._collapse_window_p) else 0
        warn_c = 1 if len(self._collapse_window_c) == window_size and all(self._collapse_window_c) else 0
        self.writer.add_scalar("collapse_warn/s2_plain", warn_p, it)
        self.writer.add_scalar("collapse_warn/s2_contrast", warn_c, it)
        if warn_p == 1 or warn_c == 1:
            warnings.warn(
                f"[BP/A-01] Posterior collapse suspected at iter {it}: "
                f"P50(s²_plain)={p50_p:.3e}, P50(s²_contrast)={p50_c:.3e}, "
                f"threshold={warn_thresh:.3e} sustained {self._collapse_warn_rolling_iters} iter",
                stacklevel=2,
            )

    # ------------------------------------------------------------------ #
    # Inference / export (BP-06: posterior mean + variance K_eval=16)    #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def export_results(self):
        import scipy.io as sio
        out_dir = self.out_dir / "test_results_best"
        out_dir.mkdir(exist_ok=True)
        K_eval = int(self.config["bayesian"].get("K_eval", 16))
        xs_p: List[torch.Tensor] = []
        xs_c: List[torch.Tensor] = []
        for k in range(K_eval):
            samples_k = self.model.sample(K=1, disable_mc_for_init=False)
            vol_p_k = self._render_sample(samples_k, 0, "plain")
            vol_c_k = self._render_sample(samples_k, 0, "contrast")
            xs_p.append(vol_p_k.cpu())
            xs_c.append(vol_c_k.cpu())
        X_p = torch.stack(xs_p, dim=0)
        X_c = torch.stack(xs_c, dim=0)
        x_bar_p = X_p.mean(dim=0)
        x_bar_c = X_c.mean(dim=0)
        U_p = ((X_p - x_bar_p.unsqueeze(0)).abs().pow(2)).sum(dim=0) / max(K_eval - 1, 1)
        U_c = ((X_c - x_bar_c.unsqueeze(0)).abs().pow(2)).sum(dim=0) / max(K_eval - 1, 1)
        recon_p = self._hard_dc(
            x_bar_p.to(self.device), self.csm_p, self.kspace_p, self.mask_tr_p,
        ).cpu()
        recon_c = self._hard_dc(
            x_bar_c.to(self.device), self.csm_c, self.kspace_c, self.mask_tr_c,
        ).cpu()
        sio.savemat(
            str(out_dir / "reconstruction_plain.mat"),
            {"reconstruction": x_bar_p.numpy(), "source": "posterior_mean_raw"},
        )
        sio.savemat(
            str(out_dir / "reconstruction_contrast.mat"),
            {"reconstruction": x_bar_c.numpy(), "source": "posterior_mean_raw"},
        )
        sio.savemat(
            str(out_dir / "reconstruction_plain_harddc.mat"),
            {"reconstruction": recon_p.numpy(), "dc_mask": "tr"},
        )
        sio.savemat(
            str(out_dir / "reconstruction_contrast_harddc.mat"),
            {"reconstruction": recon_c.numpy(), "dc_mask": "tr"},
        )
        np.save(out_dir / "U_p.npy", U_p.numpy())
        np.save(out_dir / "U_c.npy", U_c.numpy())
