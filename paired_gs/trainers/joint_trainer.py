"""JointTrainer: shared-3DGS-geometry plain+contrast joint reconstruction.

Goal: reconstruct plain + contrast simultaneously with shared Gaussian geometry,
dual per-Gaussian amplitudes. Joint k-space supervision doubly constrains the
geometry, suppressing aliasing artifacts in both reconstructions.

Loss (v2 / INF-05 / A-LOSS-DROP — REQUIREMENTS.md §🔒 Locked Assumptions):
    L_dc_plain    = ‖A_plain    · x_plain    - y_plain‖²
    L_dc_contrast = ‖A_contrast · x_contrast - y_contrast‖²
    R_geo         = TV(|x_plain|) + TV(|x_contrast|)   (renamed from v1 L_tv;
                                                         PDF §1.5 geometry
                                                         regularizer role)

v1 priors REMOVED here (replaced by q(a_{n,c}) posterior in v2 Phase 8
BayesianJointGaussianModel3D / BP-04 ElboLoss):
    L_diff        = ‖|alpha_c| - |alpha_p|‖_1         [DELETED, A-LOSS-DROP]
    L_enh         = Σ ReLU(|alpha_p| - |alpha_c|)²    [DELETED, A-LOSS-DROP]

TensorBoard scalar group prefixes (INF-05):
    loss/*   — populated by Phase 7 (this module): loss/total, loss/dc_p,
               loss/dc_c, loss/R_geo, loss/balance_violation
    kl/*     — reserved for Phase 8 (BP-04 ElboLoss): kl/a, kl/mu, kl/ell
    audit/*  — reserved for Phase 10 (EA): U_c_p99, R_c_p99, H_c_p99
    cf/*     — reserved for Phase 11 (CF): topM_score_mean, topM_score_max

Loss-balance ratio assertion (CONTEXT.md 决策 D):
    iter == 1 + |sum(ratios) - 1| > 1e-3 → raise ValueError (config sanity)
    iter >  1 + |sum(ratios) - 1| > 1e-3 → warnings.warn + TB scalar = 1.0

This trainer reuses GaussianTrainer plumbing as much as possible by composing two
single-scan datasets (plain + contrast) and rendering twice through the shared
voxelizer.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from data.dataset import MRIDataset
from data.transforms import fft3c, ifft3c
from gaussian.joint_gaussian_model import JointGaussianModel3D
from gaussian.tile_voxelizer import TileVoxelizer
from gaussian.voxelizer import Voxelizer
from trainers.voxelizer_config import apply_deterministic_voxelizer_reference


# ----------------------------------------------------------------------
def _load_dataset(data_cfg: Dict[str, Any]) -> MRIDataset:
    return MRIDataset(
        data_path=data_cfg["data_path"],
        mask_path=data_cfg.get("mask_path"),
        csm_path=data_cfg.get("csm_path"),
        acceleration_factor=data_cfg.get("acceleration_factor"),
        mask_type=data_cfg.get("mask_type", "external"),
        readout_axis=data_cfg.get("readout_axis", 0),
        phase_axes=tuple(data_cfg.get("phase_axes", [1, 2])),
        use_acs=data_cfg.get("use_acs", False),
        acs_lines=data_cfg.get("acs_lines", 0),
        normalize_kspace=data_cfg.get("normalize_kspace", True),
        normalization_percentile=data_cfg.get("normalization_percentile", 99.9),
        skip_readout_check=data_cfg.get("skip_readout_check", True),
        center_kspace_dc=data_cfg.get("center_kspace_dc", False),
        presampled=data_cfg.get("presampled", True),
    )


# ----------------------------------------------------------------------
class JointTrainer:
    """Shared-geometry plain+contrast joint reconstructor."""

    def __init__(self, config: Dict[str, Any], device: torch.device | None = None):
        self.config = config
        # Config-driven contrast labels: the pair config's data.plain / data.contrast
        # subdicts may carry an optional `name`. These drive log strings only (plain is
        # the high-SNR guide, contrast is the guided channel); the mechanism is unchanged.
        self.name_plain = config.get("data", {}).get("plain", {}).get("name", "plain")
        self.name_contrast = config.get("data", {}).get("contrast", {}).get("name", "contrast")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # D-06 (CB-02): Fail fast on the silent warmup-tying inert-prior bug
        # (CONCERNS sec 2.4). Setting shared_warmup_iters >= max_iterations * 0.5
        # means the train loop spends most/all of the run in the tying branch
        # (joint_trainer.py:284-289), so lambda_diff and lambda_enh are silently
        # ignored. The CE 8-min budget x default warmup=1000 reproduced this
        # exactly (autoresearch v3 cycles 1-5 wasted on this; C6 50<-500 was the
        # single biggest win). Catch it here, before the slow data load.
        train_cfg = config.get("training", {})
        max_iter = train_cfg.get("max_iterations", 8000)
        warmup = train_cfg.get("shared_warmup_iters", 1000)
        if warmup >= max_iter * 0.5:
            raise ValueError(
                f"shared_warmup_iters ({warmup}) must be strictly less than "
                f"max_iterations * 0.5 ({max_iter * 0.5}) "
                f"[max_iterations={max_iter}]; otherwise the joint trainer "
                f"never exits the warmup-tying branch and lambda_diff / "
                f"lambda_enh are silently inert (CONCERNS sec 2.4, D-06)."
            )

        # D-08 (CB-03): Defensive re-seed for direct instantiation paths.
        # train_joint.py entry already seeds via train.set_seed; this guards
        # against callers (autoresearch loops, notebooks, ipython) that bypass
        # the entry script. Reuses train_cfg already read above.
        _seed = int(train_cfg.get("seed", 42))
        _deterministic = bool(train_cfg.get("deterministic", False))
        import random as _random
        import numpy as _np
        _random.seed(_seed)
        _np.random.seed(_seed)
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
            if _deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.use_deterministic_algorithms(True)
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        print(f"[JointTrainer] re-seed: seed={_seed}, deterministic={_deterministic}")
        voxelizer_note = apply_deterministic_voxelizer_reference(self.config)
        if voxelizer_note:
            print(f"[DeterministicVoxelizer] {voxelizer_note}")
        self._position_grads_for_densify: torch.Tensor | None = None

        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_output()

    # -- data ----------------------------------------------------------
    def _setup_data(self):
        data_cfg = self.config["data"]
        print(">>> loading PLAIN")
        self.ds_p = _load_dataset(data_cfg["plain"])
        print(">>> loading CONTRAST")
        self.ds_c = _load_dataset(data_cfg["contrast"])

        # both datasets must share volume_shape (same subject, same geometry)
        assert self.ds_p.volume_shape == self.ds_c.volume_shape, (
            f"plain {self.ds_p.volume_shape} != contrast {self.ds_c.volume_shape}"
        )
        self.volume_shape = self.ds_p.volume_shape
        print(f"shared volume_shape = {self.volume_shape}")

        # move to device
        self.kspace_p = self.ds_p.kspace_undersampled_complex.to(self.device)   # (C,D,H,W) complex
        self.csm_p = self.ds_p.sensitivity_maps.to(self.device)
        self.mask_p = self.ds_p.mask.to(self.device)                             # (D,H,W) float

        self.kspace_c = self.ds_c.kspace_undersampled_complex.to(self.device)
        self.csm_c = self.ds_c.sensitivity_maps.to(self.device)
        self.mask_c = self.ds_c.mask.to(self.device)

        # zero-filled SENSE init for amplitude warm-start
        with torch.no_grad():
            zf_p = torch.sum(torch.conj(self.csm_p) * ifft3c(self.kspace_p), dim=0)
            zf_c = torch.sum(torch.conj(self.csm_c) * ifft3c(self.kspace_c), dim=0)
        self.zf_p = zf_p
        self.zf_c = zf_c
        print(f"|zf_p| max {zf_p.abs().max().item():.4f}, |zf_c| max {zf_c.abs().max().item():.4f}")

    # -- model ---------------------------------------------------------
    def _setup_model(self):
        gcfg = self.config["gaussian"]
        d, h, w = self.volume_shape

        warm_path = gcfg.get("warm_start_checkpoint")
        if warm_path and Path(warm_path).exists():
            print(f"warm-starting Gaussian geometry from {warm_path}")
            ckpt = torch.load(warm_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("gaussian_state", ckpt)
            positions = sd["positions"].to(self.device)
            scales_log = sd["scales"].to(self.device)
            rotations = sd["rotations"].to(self.device)
            d_real = sd["density_real"].to(self.device)
            d_imag = sd["density_imag"].to(self.device)
            n_init = positions.shape[0]
            print(f"  loaded {n_init} Gaussians from V2 checkpoint")
            scales = torch.exp(scales_log)
            d_p = torch.complex(d_real, d_imag)
            d_c = d_p.clone()    # initialize contrast = plain; trained to differ
            self.model = JointGaussianModel3D(
                num_points=n_init,
                volume_shape=self.volume_shape,
                device=self.device,
                initial_positions=positions,
                initial_densities_plain=d_p,
                initial_densities_contrast=d_c,
                initial_scales=scales,
            ).to(self.device)
            with torch.no_grad():
                self.model.rotations.copy_(rotations)
        else:
            n_init = gcfg.get("initial_num_points", 200000)
            positions = torch.rand(n_init, 3, device=self.device) * 2 - 1
            with torch.no_grad():
                ix = ((positions[:, 0] + 1) * 0.5 * d).clamp(0, d - 1).long()
                iy = ((positions[:, 1] + 1) * 0.5 * h).clamp(0, h - 1).long()
                iz = ((positions[:, 2] + 1) * 0.5 * w).clamp(0, w - 1).long()
                d_p = self.zf_p[ix, iy, iz].to(torch.complex64) * 0.1
                d_c = self.zf_c[ix, iy, iz].to(torch.complex64) * 0.1
            base_scale = gcfg.get("init_scale", 2.0 / max(d, h, w))
            scales = torch.ones(n_init, 3, device=self.device) * base_scale
            self.model = JointGaussianModel3D(
                num_points=n_init,
                volume_shape=self.volume_shape,
                device=self.device,
                initial_positions=positions,
                initial_densities_plain=d_p,
                initial_densities_contrast=d_c,
                initial_scales=scales,
            ).to(self.device)
        print(f"JointGaussianModel3D init: {self.model.parameters_summary()}")

        # Alignment-aware contrast-private residual (MC-GS): attach a separate Gaussian set that
        # renders ONLY into the contrast channel. Init geometry from a deterministic stride
        # subset of the shared cloud (co-located with anatomy, convention-correct), and a small
        # fraction of the shared contrast amplitude so the residual starts near-zero yet its
        # geometry receives gradient immediately. Trains on the contrast DC term alone.
        n_priv = int(gcfg.get("num_private_points", 0))
        if n_priv > 0:
            fac = float(gcfg.get("private_density_init_factor", 0.1))
            priv_warm = gcfg.get("private_warm_start_checkpoint")
            with torch.no_grad():
                if priv_warm and Path(priv_warm).exists():
                    # contrast-specific geometry init: the residual's Gaussians start from a
                    # contrast-tuned cloud, not a stride subset of the plain-dominated shared cloud.
                    pck = torch.load(priv_warm, map_location="cpu", weights_only=False)
                    psd = pck.get("gaussian_state", pck)
                    ppos = psd["positions"].to(self.device)
                    pscl = psd["scales"].to(self.device)
                    prot = psd["rotations"].to(self.device)
                    pdr = psd["density_real"].to(self.device) * fac
                    pdi = psd["density_imag"].to(self.device) * fac
                    if n_priv < ppos.shape[0]:
                        idx = torch.linspace(0, ppos.shape[0] - 1, n_priv, device=self.device).long()
                        ppos, pscl, prot, pdr, pdi = ppos[idx], pscl[idx], prot[idx], pdr[idx], pdi[idx]
                    src = f"{self.name_contrast}-cloud {Path(priv_warm).parts[-3]}"
                else:
                    # fallback: stride subset of the shared cloud
                    n_sh = self.model.positions.shape[0]
                    n_take = min(n_priv, n_sh)
                    idx = torch.linspace(0, n_sh - 1, n_take, device=self.device).long()
                    ppos = self.model.positions[idx]
                    pscl = self.model.scales[idx]
                    prot = self.model.rotations[idx]
                    pdr = self.model.density_real_contrast[idx] * fac
                    pdi = self.model.density_imag_contrast[idx] * fac
                    src = "shared stride-subset"
                self.model.add_private_residual(
                    positions=ppos.clone(), scales_log=pscl.clone(), rotations=prot.clone(),
                    density_real=pdr.clone(), density_imag=pdi.clone(),
                )
            self.model.to(self.device)
            print(f"[{self.name_contrast}-private] attached {self.model.num_private} private residual Gaussians "
                  f"(init {src}, factor {fac}); summary {self.model.parameters_summary()}")

        # voxelizer (one instance, called twice)
        vcfg = self.config.get("voxelizer", {})
        vtype = vcfg.get("type", "tile_cuda")
        if vtype in {"tile", "tile_cuda"}:
            self.voxelizer = TileVoxelizer(
                volume_shape=self.volume_shape,
                tile_size=vcfg.get("tile_size", 8),
                max_radius=vcfg.get("max_radius", 20),
                use_cuda=(vtype == "tile_cuda"),
                strict_cuda=bool(self.config.get("mode", {}).get("strict_cuda", False)) and vtype == "tile_cuda",
                device=str(self.device),
            ).to(self.device)
        elif vtype == "chunk":
            self.voxelizer = Voxelizer(
                volume_shape=self.volume_shape,
                device=str(self.device),
                max_radius=vcfg.get("max_radius", 20),
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported voxelizer.type: {vtype!r}")

    # -- optimizer -----------------------------------------------------
    def _optimizer_param_groups(self):
        gcfg = self.config["gaussian"]
        return self.model.get_optimizable_params(
            lr_position=gcfg.get("position_lr", 1.5e-3),
            lr_density=gcfg.get("density_lr", 1.5e-2),
            lr_scale=gcfg.get("scale_lr", 8e-3),
            lr_rotation=gcfg.get("rotation_lr", 1.5e-3),
            lr_private_position=gcfg.get("private_position_lr"),
            lr_private_density=gcfg.get("private_density_lr"),
            lr_private_scale=gcfg.get("private_scale_lr"),
            lr_private_rotation=gcfg.get("private_rotation_lr"),
        )

    def _setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self._optimizer_param_groups())
        sched_cfg = self.config.get("training", {}).get("lr_scheduler", {})
        if sched_cfg.get("type") == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=sched_cfg.get("gamma", 0.9997)
            )
        else:
            self.scheduler = None

    def _rebuild_optimizer(self):
        current_lrs = [group["lr"] for group in self.optimizer.param_groups]
        old_optimizer = self.optimizer
        self.optimizer = torch.optim.Adam(self._optimizer_param_groups())
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
                self.optimizer, gamma=sched_cfg.get("gamma", 0.9997)
            )
        else:
            self.scheduler = None

    # -- output --------------------------------------------------------
    def _setup_output(self):
        out_cfg = self.config.get("output", {})
        out_root = out_cfg.get("output_dir", "./results")
        exp = out_cfg.get("exp_name", "ce_joint_v3")
        self.out_dir = Path(out_root) / exp
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # save resolved config
        with open(self.out_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        self._emit_run_info()
        print(f"output dir: {self.out_dir}")

        # INF-05: TensorBoard SummaryWriter (mirrors v1 trainer.py:295).
        # Gated on output.use_tensorboard (default True). Phase 7/8/10/11
        # populate scalar groups: loss/* (this phase), kl/* (Phase 8 BP-04),
        # audit/* (Phase 10 EA), cf/* (Phase 11 CF).
        use_tb = bool(out_cfg.get("use_tensorboard", True))
        if use_tb:
            self.writer = SummaryWriter(log_dir=str(self.out_dir / "logs"))
            print(f"[INF-05] TensorBoard writer: {self.out_dir / 'logs'}")
        else:
            self.writer = None
            print("[INF-05] TensorBoard writer disabled (output.use_tensorboard=False)")

    def _emit_run_info(self) -> None:
        """Write <out_dir>/run_info.json (D-13, CB-03 reproducibility provenance).

        Mirrors GaussianTrainer._emit_run_info; see GS_TMI/REPRODUCIBILITY.md
        for the schema and key semantics.
        """
        import datetime as _dt
        import hashlib as _hashlib
        import json as _json
        gm_path = Path(__file__).resolve().parent.parent / "gaussian" / "gaussian_model.py"
        try:
            gm_sha = _hashlib.sha256(gm_path.read_bytes()).hexdigest()
        except OSError:
            gm_sha = ""
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        try:
            pci_idx = int(cvd.split(",")[0]) if cvd else -1
        except ValueError:
            pci_idx = -1
        train_cfg = self.config.get("training", {})
        run_info = {
            "seed": int(train_cfg.get("seed", 42)),
            "deterministic": bool(train_cfg.get("deterministic", False)),
            "gpu_pci_index": pci_idx,
            "cuda_visible_devices": cvd,
            "config_snapshot": str(self.out_dir / "config.yaml"),
            "gaussian_model_sha256": gm_sha,
            "timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
        }
        with open(self.out_dir / "run_info.json", "w") as _f:
            _json.dump(run_info, _f, indent=2)

    # -- rendering -----------------------------------------------------
    def _render(self, channel: str) -> torch.Tensor:
        density = self.model.get_density(channel)
        scales = self.model.get_scale_values()
        vol = self.voxelizer(
            positions=self.model.positions,
            scales=scales,
            rotations=self.model.rotations,
            density=density,
        )
        if isinstance(vol, tuple):
            vol = torch.complex(vol[0], vol[1])
        # Alignment-aware contrast-private residual: a separate Gaussian set rendered ONLY into
        # the contrast channel, added on top of the shared-geometry contrast render.
        if channel == "contrast" and self.model.has_private:
            pv = self.voxelizer(
                positions=self.model.private_positions,
                scales=self.model.get_private_scale_values(),
                rotations=self.model.private_rotations,
                density=self.model.get_private_density(),
            )
            if isinstance(pv, tuple):
                pv = torch.complex(pv[0], pv[1])
            vol = vol + pv
        return vol

    # -- losses --------------------------------------------------------
    def _kspace_loss(self, vol: torch.Tensor, csm: torch.Tensor, ksp_meas: torch.Tensor,
                     mask: torch.Tensor, freq_weight: torch.Tensor | None) -> torch.Tensor:
        pred_ksp = fft3c(vol.unsqueeze(0) * csm)
        diff = (pred_ksp - ksp_meas) * mask.unsqueeze(0)
        if freq_weight is not None:
            diff = diff * freq_weight.unsqueeze(0)
        return (diff.real.pow(2) + diff.imag.pow(2)).sum() / max(mask.sum().item(), 1.0)

    def _tv_loss(self, vol: torch.Tensor) -> torch.Tensor:
        m = vol.abs()
        gx = m[1:] - m[:-1]
        gy = m[:, 1:] - m[:, :-1]
        gz = m[:, :, 1:] - m[:, :, :-1]
        return gx.abs().mean() + gy.abs().mean() + gz.abs().mean()

    def _dtv_loss_directional(self, vol_c: torch.Tensor, vol_p: torch.Tensor,
                              eta_scale: float, w_floor: float) -> torch.Tensor:
        """plain-guided directional TV on the contrast magnitude.

        The high-SNR plain reconstruction (detached, guidance only) supplies a
        per-axis edge map. The contrast smoothness penalty is relaxed toward
        ``w_floor`` at shared anatomical edges (where |grad plain| is large) and
        enforced (w -> 1) in plain-flat regions. This injects plain edge geometry
        into the contrast's zero-space so boundaries form at the correct location
        (targets the diagnosed ~77% contrast error in the 6-voxel boundary shell)
        while suppressing non-anatomical aliasing/grain in flat regions.
        ``w_floor`` keeps a residual penalty at plain edges so genuinely
        contrast-private structure is not forced to vanish.
        """
        mc = vol_c.abs()
        mp = vol_p.abs().detach()
        gpx = (mp[1:] - mp[:-1]).abs()
        gpy = (mp[:, 1:] - mp[:, :-1]).abs()
        gpz = (mp[:, :, 1:] - mp[:, :, :-1]).abs()
        eta = eta_scale * (gpx.mean() + gpy.mean() + gpz.mean()) / 3.0 + 1e-8

        def edge_w(g: torch.Tensor) -> torch.Tensor:
            return w_floor + (1.0 - w_floor) * torch.exp(-g / eta)

        gcx = (mc[1:] - mc[:-1]).abs()
        gcy = (mc[:, 1:] - mc[:, :-1]).abs()
        gcz = (mc[:, :, 1:] - mc[:, :, :-1]).abs()
        return ((edge_w(gpx) * gcx).mean()
                + (edge_w(gpy) * gcy).mean()
                + (edge_w(gpz) * gcz).mean())

    # back-compat alias (external callers of the old name still work)
    _dtv_loss_flair = _dtv_loss_directional

    def _edge_align_loss(self, vol_p: torch.Tensor, vol_c: torch.Tensor,
                         p: float, kappa: float) -> torch.Tensor:
        """Symmetric common-edge cross-gradient prior (M=2 case of the locked
        RPairCommonEdgeAlign formula in losses/pair_prior.py). Penalizes MISaligned
        gradient DIRECTIONS between the two contrasts, gated to voxels where BOTH have
        edges: ell = 1 - <g_p,g_c>^2 / (||g_p||^2 ||g_c||^2). Unlike the plain->contrast
        directional TV (_dtv_loss_directional), this couples BOTH contrasts symmetrically
        (differentiable through each), so shared anatomical boundaries reinforce in T1 AND
        FLAIR. Edge-gate weight is detached (stopgrad) per the locked formula."""
        def grad3(x):
            gd = (torch.roll(x, -1, 0) - torch.roll(x, 1, 0)) * 0.5
            gh = (torch.roll(x, -1, 1) - torch.roll(x, 1, 1)) * 0.5
            gw = (torch.roll(x, -1, 2) - torch.roll(x, 1, 2)) * 0.5
            return torch.stack([gd, gh, gw], dim=0)
        gp, gc = grad3(vol_p.abs()), grad3(vol_c.abs())
        gpm, gcm = gp.norm(dim=0), gc.norm(dim=0)
        # subsampled quantile threshold (full 14.4M-vox per-iter quantile is slow; a
        # ~0.9M stride sample gives an essentially identical tau)
        tau_p = torch.quantile(gpm.flatten()[::16], p)
        tau_c = torch.quantile(gcm.flatten()[::16], p)
        a_p = torch.sigmoid(kappa * (gpm - tau_p))
        a_c = torch.sigmoid(kappa * (gcm - tau_c))
        w = (a_p * a_c).detach()
        dot = (gp * gc).sum(dim=0)
        ell = 1.0 - dot.pow(2) / ((gp * gp).sum(dim=0) * (gc * gc).sum(dim=0) + 1e-8)
        return (w * ell).sum() / (w.sum() + 1e-8)

    def _joint_vtv_loss(self, vol_p: torch.Tensor, vol_c: torch.Tensor) -> torch.Tensor:
        """Vectorial (joint) TV: mean of sqrt(|grad_p|^2 + |grad_c|^2) over the 3 axes.
        Couples edge LOCATIONS symmetrically — one shared boundary costs less than two
        independent ones, so both contrasts place edges at the same voxels. Magnitude-based
        (bounded gradient; STABLE, unlike the normalized directional _edge_align_loss)."""
        mp, mc = vol_p.abs(), vol_c.abs()
        eps = 1e-6
        gpx, gcx = mp[1:] - mp[:-1], mc[1:] - mc[:-1]
        gpy, gcy = mp[:, 1:] - mp[:, :-1], mc[:, 1:] - mc[:, :-1]
        gpz, gcz = mp[:, :, 1:] - mp[:, :, :-1], mc[:, :, 1:] - mc[:, :, :-1]
        return (torch.sqrt(gpx**2 + gcx**2 + eps).mean()
                + torch.sqrt(gpy**2 + gcy**2 + eps).mean()
                + torch.sqrt(gpz**2 + gcz**2 + eps).mean())

    def _build_freq_weight(self, high_override: float | None = None) -> torch.Tensor | None:
        """Optional frequency-adaptive weighting (mirrors V2 detail config).

        ``high_override`` lets a contrast use a different high-band weight than the shared
        config (per-contrast sharpness decoupling: plain wants more high-k emphasis = sharper,
        contrast wants less = cleaner). When None, uses loss.kspace_high_weight.
        """
        loss_cfg = self.config.get("loss", {})
        if not loss_cfg.get("kspace_freq_adaptive", False):
            return None
        d, h, w = self.volume_shape
        zd = torch.linspace(-1, 1, d, device=self.device).view(d, 1, 1)
        zh = torch.linspace(-1, 1, h, device=self.device).view(1, h, 1)
        zw = torch.linspace(-1, 1, w, device=self.device).view(1, 1, w)
        r = torch.sqrt(zd * zd + zh * zh + zw * zw)
        cf = loss_cfg.get("kspace_center_fraction", 0.05)
        wc = loss_cfg.get("kspace_center_weight", 0.3)
        wm = loss_cfg.get("kspace_mid_weight", 1.0)
        wh_ = float(high_override) if high_override is not None else loss_cfg.get("kspace_high_weight", 4.0)
        weight = torch.full_like(r, wm)
        weight[r < cf] = wc
        weight[r > 0.5] = wh_
        return weight

    def _freq_weight_curriculum(self, it: int, max_iter: int) -> torch.Tensor | None:
        """JSG-GS progressive k-space frequency fitting (coarse->fine): the high-freq
        window grows with training. Early iters fit only central k (large-scale anatomy);
        the cutoff radius expands to full k by `kspace_curriculum_frac * max_iter`. This
        coarse-to-fine schedule suppresses early over-fitting of high-freq aliasing.
        Config-gated: returns the static freq weight when disabled (legacy behavior)."""
        cfg = self.config.get("loss", {})
        if not cfg.get("kspace_freq_curriculum", False):
            return getattr(self, "_static_freq_w", None)
        if not hasattr(self, "_kr_grid"):
            d, h, w = self.volume_shape
            zd = torch.linspace(-1, 1, d, device=self.device).view(d, 1, 1)
            zh = torch.linspace(-1, 1, h, device=self.device).view(1, h, 1)
            zw = torch.linspace(-1, 1, w, device=self.device).view(1, 1, w)
            self._kr_grid = torch.sqrt(zd * zd + zh * zh + zw * zw)
        frac = float(cfg.get("kspace_curriculum_frac", 0.7))
        rc0 = float(cfg.get("kspace_curriculum_rc_start", 0.12))
        tail = float(cfg.get("kspace_curriculum_tail_w", 0.05))
        progress = min(1.0, (it / max_iter) / max(frac, 1e-6))
        rc = rc0 + (1.0 - rc0) * progress
        weight = torch.full_like(self._kr_grid, tail)
        weight[self._kr_grid <= rc] = 1.0
        return weight

    def _joint_density_magnitude(self) -> torch.Tensor:
        density_mag = torch.maximum(
            self.model.density_plain.abs(),
            self.model.density_contrast.abs(),
        )
        return density_mag

    def _evidence_energy_score(self) -> torch.Tensor | None:
        """E-MC-GS evidence-guided growth score (prototype P1).

        Per-Gaussian backprojected multi-contrast k-space residual ENERGY, sampled at the
        Gaussian centres. Grows primitives where the ACQUIRED measurement is still unexplained
        (combined over plain+contrast), instead of where the image-space position gradient is large
        (which at high AF tends to chase aliasing). e_c = A_c^H (A_c x_c - y_c); score_k =
        (|e_plain|^2 + |e_contrast|^2) sampled at mu_k.
        """
        with torch.no_grad():
            def energy(vol, csm, y, mask):
                m = (mask if mask.ndim == 3 else mask[0]).unsqueeze(0)
                r = m * fft3c(vol.unsqueeze(0) * csm) - y           # residual on sampled set
                e = (ifft3c(m * r) * csm.conj()).sum(0)             # backproject to image (D,H,W)
                return e.real ** 2 + e.imag ** 2
            E = (energy(self._render("plain"), self.csm_p, self.kspace_p, self.mask_p)
                 + energy(self._render("contrast"), self.csm_c, self.kspace_c, self.mask_c))
            pos = self.model.positions.detach()                     # (N,3) in [-1,1], (D,H,W) order
            grid = pos[:, [2, 1, 0]].reshape(1, -1, 1, 1, 3)        # -> grid_sample (x=W, y=H, z=D)
            Eg = E.reshape(1, 1, *E.shape).float()
            s = torch.nn.functional.grid_sample(Eg, grid, mode="bilinear",
                                                padding_mode="border", align_corners=True)
            return s.reshape(-1)

    def _adaptive_density_control(self, iteration: int) -> Dict[str, int]:
        adaptive_cfg = self.config.get("adaptive_control", {})
        if not adaptive_cfg.get("enable", False):
            return {"split": 0, "clone": 0, "prune": 0}
        start_iter = int(adaptive_cfg.get("densify_from_iter", 100))
        end_iter = int(adaptive_cfg.get("densify_until_iter", 2500))
        interval = int(adaptive_cfg.get("densify_every", 100))
        if interval <= 0 or iteration < start_iter or iteration > end_iter or iteration % interval != 0:
            return {"split": 0, "clone": 0, "prune": 0}

        max_num_points = int(
            self.config.get("gaussian", {}).get("max_num_points", self.model.num_points * 2)
        )
        if self.model.num_points >= max_num_points:
            return {"split": 0, "clone": 0, "prune": 0}

        stats = {"split": 0, "clone": 0, "prune": 0}
        n_before = self.model.num_points
        did_modify = False
        step_index = (iteration - start_iter) // interval
        is_split_iter = (step_index % 2 == 0)

        if is_split_iter:
            split_score = str(adaptive_cfg.get("split_score", "gradient"))
            split_quantile = float(adaptive_cfg.get("split_quantile", 0.0))
            if split_score == "evidence_energy":
                score = self._evidence_energy_score()              # E-MC-GS evidence-guided growth
            else:
                grads = self._position_grads_for_densify
                if grads is None:
                    return stats
                score = torch.norm(grads, dim=-1)
            if score is None:
                return stats
            grad_norm = score
            if split_quantile > 0.0:                               # top-(1-q) by score (scale-robust)
                grad_threshold = float(torch.quantile(score, split_quantile))
            else:
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
            use_long_axis = bool(adaptive_cfg.get("use_long_axis_splitting", True))
            long_axis_offset_factor = float(
                self.config.get("gaussian", {}).get("long_axis_offset_factor", 1.0)
            )
            stats["split"] = self.model.densify_and_split(
                grads=grad_norm,
                grad_threshold=grad_threshold,
                scale_threshold=scale_threshold,
                use_long_axis_splitting=use_long_axis,
                long_axis_offset_factor=long_axis_offset_factor,
                max_split_fraction=max_split_fraction,
            )
            did_modify = stats["split"] > 0

            if not did_modify and bool(adaptive_cfg.get("use_cloning", False)):
                stats["clone"] = self.model.densify_and_clone(
                    grad_norm,
                    grad_threshold,
                    scale_threshold,
                )
                did_modify = stats["clone"] > 0
        else:
            opacity_threshold = float(adaptive_cfg.get("opacity_threshold", 0.001))
            max_scale_limit = float(adaptive_cfg.get("max_scale", 0.5))
            max_scale = self.model.get_scale_values().max(dim=-1)[0]
            prune_mask = (self._joint_density_magnitude() < opacity_threshold) | (max_scale > max_scale_limit)
            keep_mask = ~prune_mask
            if keep_mask.sum() >= 100 and prune_mask.sum() > 0:
                self.model._update_params(keep_mask, is_prune=True)
                stats["prune"] = int(prune_mask.sum().item())
                did_modify = True

        if did_modify:
            self._rebuild_optimizer()
            n_after = self.model.num_points
            if self.writer is not None:
                self.writer.add_scalar("gaussian/num_points", n_after, iteration)
                self.writer.add_scalar("gaussian/n_split_per_event", stats["split"], iteration)
                self.writer.add_scalar("gaussian/n_clone_per_event", stats["clone"], iteration)
                self.writer.add_scalar("gaussian/n_prune_per_event", stats["prune"], iteration)
            print(
                f"[Joint Densify iter={iteration}] Gaussians: {n_before} -> {n_after}, "
                f"split={stats['split']} clone={stats['clone']} prune={stats['prune']}"
            )
        return stats

    # -- train loop ----------------------------------------------------
    def train(self):
        train_cfg = self.config["training"]
        loss_cfg = self.config["loss"]
        max_iter = train_cfg.get("max_iterations", 8000)
        warmup = train_cfg.get("shared_warmup_iters", 1000)
        log_every = train_cfg.get("log_every", 50)
        save_every = train_cfg.get("save_every", 1000)
        eval_every = train_cfg.get("eval_every", 200)

        # INF-05 / A-LOSS-DROP (REQUIREMENTS.md §🔒 Locked Assumptions):
        # v1 lambda_diff (sparsity of |α_c|-|α_p|) and lambda_enh (Gd uptake
        # ReLU monotonicity) have been DELETED — v2 q(a_{n,c}) posterior in
        # BayesianJointGaussianModel3D (Phase 8) fully replaces these
        # hand-written cross-contrast priors. v1 tv_weight has been renamed
        # to lambda_geo to reflect its ELBO role as a geometry regularizer
        # (PDF §1.5: L_Bayes = ... + λ_geo R_geo + λ_sp R_sp).
        if "tv_weight" in loss_cfg and "lambda_geo" not in loss_cfg:
            warnings.warn(
                "loss.tv_weight is deprecated (INF-05 / A-LOSS-DROP); use "
                "loss.lambda_geo instead. Reading tv_weight as lambda_geo for "
                "backward compat with v1 configs.",
                DeprecationWarning,
                stacklevel=2,
            )
            lambda_geo = float(loss_cfg["tv_weight"])
        else:
            lambda_geo = float(loss_cfg.get("lambda_geo", 5e-4))
        if "lambda_diff" in loss_cfg or "lambda_enh" in loss_cfg:
            warnings.warn(
                "loss.lambda_diff / loss.lambda_enh are silently ignored "
                "(INF-05 / A-LOSS-DROP). v2 q(a_{n,c}) posterior replaces "
                "these hand-written priors.",
                stacklevel=2,
            )
        # R_sp: L1 sparsity on the contrast-private residual amplitude (PDF §1.5),
        # keeping the residual localized to genuine contrast-specific deviations.
        lambda_private_sp = float(loss_cfg.get("lambda_private_sp", 0.0))
        # R_dtv: plain-guided directional TV on contrast (this work). Default 0 => no-op.
        lambda_dtv = float(loss_cfg.get("lambda_dtv", 0.0))
        dtv_eta_scale = float(loss_cfg.get("dtv_eta_scale", 1.0))
        dtv_w_floor = float(loss_cfg.get("dtv_w_floor", 0.1))
        # R_edge: symmetric common-edge cross-gradient prior (this work). Default 0 => no-op.
        lambda_edge = float(loss_cfg.get("lambda_edge", 0.0))
        edge_p = float(loss_cfg.get("edge_p", 0.75))
        edge_kappa = float(loss_cfg.get("edge_kappa", 10.0))
        # R_vtv: vectorial (joint) TV — symmetric edge-LOCATION coupling (this work). Default 0.
        lambda_vtv = float(loss_cfg.get("lambda_vtv", 0.0))
        loss_type = loss_cfg.get("loss_type", "l2")
        del loss_type   # placeholder if l1 added later

        self._static_freq_w = self._build_freq_weight()
        freq_w = self._static_freq_w
        # Per-contrast sharpness decoupling: if loss.kspace_high_weight_plain is set, the plain
        # channel uses its own high-band weight (sharper) while the contrast keeps the shared one.
        _khp = self.config.get("loss", {}).get("kspace_high_weight_plain")
        self._static_freq_w_plain = self._build_freq_weight(high_override=_khp) if _khp is not None else None

        log_path = self.out_dir / "joint_train.log"
        log_f = open(log_path, "w")

        # INF-05 NOTE: `warmup` parameter (shared_warmup_iters) is still
        # validated upstream by the D-06 / CB-02 guard at L75-82, but the
        # original use-site (warmup-tying branch with l_diff/l_enh) was
        # deleted in this restructure because A-LOSS-DROP replaces both
        # priors with q(a_{n,c}) posterior. The guard is retained as
        # defensive validation for legacy configs.
        del warmup

        t0 = time.time()
        best_loss = float("inf")
        for it in range(1, max_iter + 1):
            self.optimizer.zero_grad()
            freq_w = self._freq_weight_curriculum(it, max_iter)   # JSG-GS progressive freq (static if disabled)
            vol_p = self._render("plain")
            vol_c = self._render("contrast")

            # k-space DC (plain may use a decoupled higher high-band weight = sharper plain)
            freq_w_p = self._static_freq_w_plain if self._static_freq_w_plain is not None else freq_w
            l_dc_p = self._kspace_loss(vol_p, self.csm_p, self.kspace_p, self.mask_p, freq_w_p)
            l_dc_c = self._kspace_loss(vol_c, self.csm_c, self.kspace_c, self.mask_c, freq_w)
            loss = l_dc_p + l_dc_c

            # R_geo (renamed from v1 L_tv per INF-05 / A-LOSS-DROP; PDF §1.5
            # casts this term as a geometry regularizer in the ELBO).
            if lambda_geo > 0:
                l_geo = self._tv_loss(vol_p) + self._tv_loss(vol_c)
                loss = loss + lambda_geo * l_geo
            else:
                l_geo = torch.tensor(0.0, device=self.device)

            # R_dtv: plain-guided directional TV on contrast (this work). Injects plain
            # edge geometry into the contrast's zero-space and suppresses non-anatomical
            # aliasing/grain in plain-flat regions.
            if lambda_dtv > 0:
                l_dtv = self._dtv_loss_directional(vol_c, vol_p, dtv_eta_scale, dtv_w_floor)
                loss = loss + lambda_dtv * l_dtv
            else:
                l_dtv = torch.tensor(0.0, device=self.device)

            # R_edge: symmetric common-edge cross-gradient prior — couples BOTH contrasts'
            # edge directions (helps T1 and FLAIR, vs dtv's one-way plain->contrast relax).
            if lambda_edge > 0:
                l_edge = self._edge_align_loss(vol_p, vol_c, edge_p, edge_kappa)
                loss = loss + lambda_edge * l_edge
            else:
                l_edge = torch.tensor(0.0, device=self.device)

            # R_vtv: vectorial (joint) TV — symmetric edge-location coupling (stable).
            if lambda_vtv > 0:
                l_vtv = self._joint_vtv_loss(vol_p, vol_c)
                loss = loss + lambda_vtv * l_vtv
            else:
                l_vtv = torch.tensor(0.0, device=self.device)

            # R_sp: keep the contrast-private residual amplitude sparse (localized).
            if lambda_private_sp > 0 and self.model.has_private:
                loss = loss + lambda_private_sp * self.model.get_private_density().abs().mean()

            # INF-05 NOTE: v1 L_diff / L_enh DELETED per A-LOSS-DROP locked
            # assumption (REQUIREMENTS.md §🔒). v2 q(a_{n,c}) posterior fully
            # replaces these hand-written cross-contrast priors. KL terms
            # (kl/a, kl/mu, kl/ell) will be wired in Phase 8 via BP-04
            # ElboLoss with β_a / β_μ / β_ℓ schedules.

            # === INF-05 loss-balance ratio assertion (CONTEXT.md 决策 D) ===
            # Components are the lambda-weighted scalar contributions to the
            # total objective. By construction ratios always sum to 1 unless
            # a numerical / configuration error caused a component to be NaN
            # / Inf / a sign change. Tolerance 1e-3 is 200x the empirical
            # float32 6-component sum rounding error (~5e-7).
            components = {
                "loss/dc_p":  float(l_dc_p.detach().item()),
                "loss/dc_c":  float(l_dc_c.detach().item()),
                "loss/R_geo": float((lambda_geo * l_geo).detach().item()),
                "loss/R_dtv": float((lambda_dtv * l_dtv).detach().item()),
                "loss/R_edge": float((lambda_edge * l_edge).detach().item()),
                "loss/R_vtv": float((lambda_vtv * l_vtv).detach().item()),
            }
            total_loss_value = sum(components.values()) + 1e-12
            ratios = {k: v / total_loss_value for k, v in components.items()}
            sum_ratios = sum(ratios.values())
            balance_violation = abs(sum_ratios - 1.0) > 1e-3
            if balance_violation:
                violation_msg = (
                    f"[INF-05] loss-balance violated at iter {it}: "
                    f"sum(ratios)={sum_ratios:.6f}, "
                    f"deviation={abs(sum_ratios - 1.0):.3e} (tol=1e-3). "
                    f"components_ratio={ratios}"
                )
                if it == 1:
                    # First-iter hard-error catches obvious config misalignments
                    # before a multi-hour run wastes wall-clock (CONTEXT 决策 D).
                    raise ValueError(violation_msg)
                # Mid-train: warn + TB scalar 1.0 (does NOT abort long runs).
                warnings.warn(violation_msg, stacklevel=2)

            # === INF-05 TensorBoard scalars — 4 group-prefix scheme ===
            # Phase 7 populates `loss/*` only; `kl/* audit/* cf/*` are
            # reserved for Phase 8 (BP) / Phase 10 (EA) / Phase 11 (CF).
            if self.writer is not None and (it % log_every == 0 or it == 1):
                self.writer.add_scalar("loss/total",
                                       float(loss.item()), it)
                self.writer.add_scalar("loss/dc_p",
                                       components["loss/dc_p"], it)
                self.writer.add_scalar("loss/dc_c",
                                       components["loss/dc_c"], it)
                self.writer.add_scalar("loss/R_geo",
                                       components["loss/R_geo"], it)
                self.writer.add_scalar("loss/balance_violation",
                                       1.0 if balance_violation else 0.0, it)
                # === Phase 8 BP / Phase 10 EA / Phase 11 CF reserved groups ===
                # Phase 8 (BP-04 ElboLoss) will add: kl/a, kl/mu, kl/ell.
                # Phase 10 (EA-04 / EA-08) will add: audit/U_c_p99,
                #     audit/R_c_p99, audit/H_c_p99.
                # Phase 11 (CF-07) will add: cf/topM_score_mean,
                #     cf/topM_score_max.

            loss.backward()
            if self.model.positions.grad is not None:
                self._position_grads_for_densify = self.model.positions.grad.detach().clone()
            else:
                self._position_grads_for_densify = None
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           train_cfg.get("max_grad_norm", 1.0))
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self._adaptive_density_control(it)

            if it % log_every == 0 or it == 1:
                msg = (f"[iter {it:5d}/{max_iter}] loss={loss.item():.4e} "
                       f"dc_p={l_dc_p.item():.4e} dc_c={l_dc_c.item():.4e} "
                       f"R_geo={l_geo.item():.4e} R_dtv={l_dtv.item():.4e} "
                       f"R_edge={l_edge.item():.4e} R_vtv={l_vtv.item():.4e} "
                       f"balance_violation={balance_violation}")
                print(msg)
                log_f.write(msg + "\n"); log_f.flush()

            if it % eval_every == 0:
                with torch.no_grad():
                    snr_p = vol_p.abs().mean().item() / max(vol_p.abs().std().item(), 1e-8)
                    snr_c = vol_c.abs().mean().item() / max(vol_c.abs().std().item(), 1e-8)
                    print(f"  eval @ iter {it}: |x_p|.mean/std={snr_p:.3f}, |x_c|.mean/std={snr_c:.3f}")

            if it % save_every == 0 or it == max_iter:
                ckpt_dir = self.out_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                ckpt = {
                    "iter": it,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                }
                torch.save(ckpt, ckpt_dir / "latest.pth")
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(ckpt, ckpt_dir / "best.pth")

        elapsed = time.time() - t0
        print(f"joint training done in {elapsed:.1f}s")
        log_f.close()
        if self.writer is not None:
            # INF-05: flush + close TB writer to prevent file-handle leak on
            # long-running training (especially when called repeatedly from
            # autoresearch loops without process recycle).
            self.writer.close()

    # -- post-train: render + hard DC + save .mat ----------------------
    @torch.no_grad()
    def export_results(self):
        import scipy.io as sio
        out_dir = self.out_dir / "test_results_best"
        out_dir.mkdir(exist_ok=True)

        vol_p = self._render("plain")
        vol_c = self._render("contrast")

        # hard data consistency: replace measured ksp at sampled locations
        def _hard_dc(vol, csm, ksp_meas, mask):
            pred_ksp = fft3c(vol.unsqueeze(0) * csm)
            ksp_dc = pred_ksp * (1 - mask.unsqueeze(0)) + ksp_meas
            coil = ifft3c(ksp_dc)
            return torch.sum(torch.conj(csm) * coil, dim=0)

        vol_p_dc = _hard_dc(vol_p, self.csm_p, self.kspace_p, self.mask_p)
        vol_c_dc = _hard_dc(vol_c, self.csm_c, self.kspace_c, self.mask_c)

        sio.savemat(str(out_dir / "reconstruction_plain.mat"),
                    {"reconstruction": vol_p_dc.cpu().numpy()})
        sio.savemat(str(out_dir / "reconstruction_contrast.mat"),
                    {"reconstruction": vol_c_dc.cpu().numpy()})
        sio.savemat(str(out_dir / "zero_filled_plain.mat"),
                    {"zero_filled": self.zf_p.cpu().numpy()})
        sio.savemat(str(out_dir / "zero_filled_contrast.mat"),
                    {"zero_filled": self.zf_c.cpu().numpy()})

        # save enhancement difference map.
        #
        # NOTE (INF-05): amplitude_difference per-Gaussian map is retained
        # purely as a diagnostic export for v1 backward compat (e.g. compare
        # against v1 joint results, regression-test joint trainer
        # outputs). v1 L_diff / L_enh losses driving this signal were deleted
        # per A-LOSS-DROP. In v2 BayesianJointTrainer (Phase 8) this is
        # superseded by the contrast-specific posterior mean |m_{n,c}|; the
        # corresponding export will switch to dumping `m_real` / `m_imag` per
        # contrast rather than this scalar difference.
        amp_diff_per_gauss = self.model.amplitude_difference()
        np.save(out_dir / "amplitude_difference.npy",
                amp_diff_per_gauss.detach().cpu().numpy())

        # absolute metrics
        try:
            from metrics import evaluate_reconstruction_absolute, print_metrics
            mp = evaluate_reconstruction_absolute(vol_p_dc.abs().cpu().numpy())
            mc = evaluate_reconstruction_absolute(vol_c_dc.abs().cpu().numpy())
            with open(out_dir / "metrics.yaml", "w") as f:
                yaml.dump({"plain": mp, "contrast": mc}, f)
            print_metrics(mp, prefix="  plain    ")
            print_metrics(mc, prefix="  contrast ")
        except Exception as e:
            print(f"metrics export failed: {e}")

        print(f"results saved to {out_dir}")
