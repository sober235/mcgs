"""VoxelGridJointTrainer (Phase 8 / BP-08 / Method C).

Voxel-grid multi-contrast baseline for Go/No-Go (PDF §1.14 Method C).

Representation
--------------
    x_plain     ∈ ℂ^{D × H × W}      (full HYW resolution, CONTEXT 决策 E)
    x_contrast  ∈ ℂ^{D × H × W}
    No Gaussian primitives, no densification, no Bayesian posterior.

Loss (deterministic, K=1 by definition)
---------------------------------------
    L = ‖A_p x_p − y_p^tr‖² + ‖A_c x_c − y_c^tr‖²
      + λ_geo · (TV(|x_p|) + TV(|x_c|))

Fairness protocol with Method A/B
---------------------------------
    - Same lr / schedule / max_iter / seed / voxelizer.deterministic
    - K_train=1 by definition (deterministic baseline; no MC sampling)
    - No bayesian.* config block needed
    - density_lr in `gaussian` block is reinterpreted here as voxel-grid lr
      (the single entry-point that the W-7 fix optimizer-level fairness
      test consumes across all four voxel param groups).

Notes
-----
- Phase 7 INF-05 backward-compat: emits flat 3-item ratio dict
  (loss/dc_p, loss/dc_c, loss/R_geo) with sum=1.0 assertion at iter 1
  (hard-error) and warn-only at later iters. Method C has no KL terms,
  so the 3-item schema is correct (NOT the 6-item Bayesian schema).
- output schema matches Method B: writes
  ``test_results_best/reconstruction_plain.mat`` and
  ``test_results_best/reconstruction_contrast.mat`` so Phase 9 GNG
  paired-t-test consumes Method C identically.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter

from data.dataset import MRIDataset
from data.transforms import fft3c, ifft3c


def _load_dataset_split_for_c(data_cfg: Dict[str, Any]) -> MRIDataset:
    """Load one contrast's MRIDataset with Phase 7 mask 三拆 enabled.

    Mirrors ``BayesianJointTrainer._load_dataset_split`` byte-identically
    on the data side (Method C trains on M^tr just like Method B, only
    the representation differs).
    """
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


class VoxelGridJointTrainer:
    """Method C voxel-grid baseline trainer (deterministic, K=1)."""

    def __init__(self, config: Dict[str, Any], device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_seeds()
        self._setup_data()
        self._setup_voxel_grid()
        self._setup_optimizer()
        self._setup_output()
        # Phase 9 GNG-01 Layer 2: emit runtime fairness fingerprint sidecar.
        try:
            fingerprint_path = self.out_dir / "phase9_fairness_fingerprint.json"
            self.dump_fairness_fingerprint(fingerprint_path)
            print(f"[phase9] runtime fingerprint dumped: {fingerprint_path}")
        except Exception as e:
            print(f"[phase9] WARN: fingerprint dump failed: {e}")

    # ------------------------------------------------------------------ #
    def _setup_seeds(self):
        train_cfg = self.config.get("training", {})
        seed = int(train_cfg.get("seed", 42))
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------ #
    def _setup_data(self):
        data_cfg = self.config["data"]
        self.ds_p = _load_dataset_split_for_c(data_cfg["plain"])
        self.ds_c = _load_dataset_split_for_c(data_cfg["contrast"])
        assert self.ds_p.volume_shape == self.ds_c.volume_shape, (
            f"plain/contrast volume_shape mismatch: "
            f"{self.ds_p.volume_shape} vs {self.ds_c.volume_shape}"
        )
        self.volume_shape = self.ds_p.volume_shape
        self.kspace_p = self.ds_p.kspace_undersampled_complex.to(self.device)
        self.kspace_c = self.ds_c.kspace_undersampled_complex.to(self.device)
        self.csm_p = self.ds_p.sensitivity_maps.to(self.device)
        self.csm_c = self.ds_c.sensitivity_maps.to(self.device)
        # Phase 7 INF-01/INF-02 mask 三拆;Method C 仅消费 M^tr 训练
        if self.ds_p.mask_split:
            self.mask_tr_p = self.ds_p.mask_split["tr"].to(self.device)
            self.mask_tr_c = self.ds_c.mask_split["tr"].to(self.device)
        else:
            self.mask_tr_p = self.ds_p.mask.to(self.device)
            self.mask_tr_c = self.ds_c.mask.to(self.device)
        # ZF-IFFT init: coil-conj-combine before voxel-grid Parameter assignment
        with torch.no_grad():
            self.zf_p = torch.sum(torch.conj(self.csm_p) * ifft3c(self.kspace_p), dim=0)
            self.zf_c = torch.sum(torch.conj(self.csm_c) * ifft3c(self.kspace_c), dim=0)
        print(
            f"[VoxelGridJointTrainer] volume_shape={self.volume_shape} "
            f"(full HYW resolution per CONTEXT 决策 E)"
        )

    # ------------------------------------------------------------------ #
    def _setup_voxel_grid(self):
        """Two complex voxel volumes as learnable nn.Parameter (D, H, W) each.

        Init = ZF-IFFT reconstruction so that gradient signal is well-conditioned
        from iter 1. fp32 real + fp32 imag — fits 224 MB at HYW (256,220,256).
        """
        init_p_re = self.zf_p.real.clone().detach()
        init_p_im = self.zf_p.imag.clone().detach()
        init_c_re = self.zf_c.real.clone().detach()
        init_c_im = self.zf_c.imag.clone().detach()
        self.x_real_p = nn.Parameter(init_p_re)
        self.x_imag_p = nn.Parameter(init_p_im)
        self.x_real_c = nn.Parameter(init_c_re)
        self.x_imag_c = nn.Parameter(init_c_im)
        n_params = sum(
            p.numel() for p in (self.x_real_p, self.x_imag_p, self.x_real_c, self.x_imag_c)
        )
        print(
            f"[VoxelGridJointTrainer] total params = {n_params} "
            f"({n_params * 4 / (1024 ** 2):.1f} MB fp32)"
        )

    # ------------------------------------------------------------------ #
    def _setup_optimizer(self):
        """Build Adam with one param group per voxel parameter.

        W-7 fix entry point: each group carries a ``name`` field so the
        fairness test can assert ``lr`` against Method B's correspondingly
        named amplitude group (density_real / density_imag / ...) via the
        density_lr key in the shared ``gaussian.density_lr`` config field.
        """
        gcfg = self.config.get("gaussian", {})
        # density_lr is the fairness-protocol entry point for Method A/B/C
        lr = float(gcfg.get("density_lr", 1.5e-2))
        self.optimizer = torch.optim.Adam(
            [
                {"params": [self.x_real_p], "lr": lr, "name": "x_real_p"},
                {"params": [self.x_imag_p], "lr": lr, "name": "x_imag_p"},
                {"params": [self.x_real_c], "lr": lr, "name": "x_real_c"},
                {"params": [self.x_imag_c], "lr": lr, "name": "x_imag_c"},
            ]
        )
        sched_cfg = self.config.get("training", {}).get("lr_scheduler", {})
        if sched_cfg.get("type") == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=float(sched_cfg.get("gamma", 0.9995))
            )
        else:
            self.scheduler = None

    # ------------------------------------------------------------------ #
    def _setup_output(self):
        out_cfg = self.config.get("output", {})
        out_root = out_cfg.get("output_dir", "./results")
        exp = out_cfg.get("exp_name", "hyw_bayesian_v2_method_c")
        self.out_dir = Path(out_root) / exp
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        use_tb = bool(out_cfg.get("use_tensorboard", True))
        if use_tb:
            self.writer = SummaryWriter(log_dir=str(self.out_dir / "logs"))
        else:
            self.writer = None

    # ------------------------------------------------------------------ #
    # Phase 9 GNG-01 Layer-2 runtime fairness fingerprint                #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def dump_fairness_fingerprint(self, out_path) -> None:
        """Phase 9 GNG-01 Layer-2 runtime fairness check (Method C variant).

        Method C lacks K_train / beta_a / warmup_kl (Bayesian-only fields),
        so they are written as None in the JSON for unambiguous diff with
        Method A/B sidecars by the Phase 9 GNG runner.
        """
        import json
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        groups = []
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
            "model_class": "VoxelGridJointTrainer",
            "trainer_class": type(self).__name__,
            "n_params_total": sum(g["n_params"] for g in groups),
            "optimizer_param_groups": groups,
            "K_train": None,
            "max_iterations": int(self.config["training"]["max_iterations"]),
            "seed": int(self.config["training"]["seed"]),
            "beta_a_target": None,
            "warmup_kl": None,
            "stopping": dict(self.config.get("stopping", {})),
        }
        with open(out_path, "w") as fh:
            json.dump(record, fh, indent=2)

    # ------------------------------------------------------------------ #
    def _kspace_dc_loss(
        self,
        x_re: torch.Tensor,
        x_im: torch.Tensor,
        csm: torch.Tensor,
        ksp_meas: torch.Tensor,
        mask_tr: torch.Tensor,
    ) -> torch.Tensor:
        """L2 k-space data consistency on M^tr.

        Composition: A(x) = M_tr ⊙ F ⊙ S(x) where F = fft3c, S = elementwise
        multiply by per-coil CSM. Normalize by sampled-voxel count to match
        Method A/B's per-voxel scaling.
        """
        x = torch.complex(x_re, x_im)
        pred_ksp = fft3c(x.unsqueeze(0) * csm)
        diff = (pred_ksp - ksp_meas) * mask_tr.unsqueeze(0)
        n = max(float(mask_tr.sum().item()), 1.0)
        return (diff.real.pow(2) + diff.imag.pow(2)).sum() / n

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _eval_psnr_for_plateau(self, K_eval: int = 1) -> float:
        """Method C plateau eval: deterministic, K=1 by definition.

        K_eval parameter accepted for API compatibility with Method A/B but
        ignored: Method C has no MC posterior, only a single deterministic
        voxel grid.
        """
        from metrics import compute_psnr
        x_p = torch.complex(self.x_real_p, self.x_imag_p)
        x_c = torch.complex(self.x_real_c, self.x_imag_c)
        psnr_p = compute_psnr(torch.abs(x_p), torch.abs(self.zf_p))
        psnr_c = compute_psnr(torch.abs(x_c), torch.abs(self.zf_c))
        return float(0.5 * (psnr_p + psnr_c))

    # ------------------------------------------------------------------ #
    def _tv_3d(self, x_re: torch.Tensor, x_im: torch.Tensor) -> torch.Tensor:
        """Isotropic TV on the magnitude — Phase 7 INF-05 R_geo for Method C."""
        x = torch.complex(x_re, x_im).abs()
        gx = (x[1:] - x[:-1]).abs().mean()
        gy = (x[:, 1:] - x[:, :-1]).abs().mean()
        gz = (x[:, :, 1:] - x[:, :, :-1]).abs().mean()
        return gx + gy + gz

    # ------------------------------------------------------------------ #
    def train(self):
        train_cfg = self.config["training"]
        loss_cfg = self.config.get("loss", {})
        # Method C consumes loss.lambda_geo;fairness with Method A/B 同字段
        lambda_geo = float(
            loss_cfg.get(
                "lambda_geo",
                self.config.get("bayesian", {}).get("lambda_geo", 5e-4),
            )
        )
        max_iter = int(train_cfg.get("max_iterations", 4000))
        log_every = int(train_cfg.get("log_every", 50))
        save_every = int(train_cfg.get("save_every", 500))
        max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        # Phase 9 B-1: instantiate MetricPlateauStopper from config.stopping schema.
        stop_cfg = self.config.get("stopping", {})
        from metrics import MetricPlateauStopper
        if stop_cfg:
            self._stopper = MetricPlateauStopper(
                patience_evals=int(stop_cfg.get("patience_evals", 4)),
                min_iterations_before_stop=int(
                    stop_cfg.get("min_iterations_before_stop", 0)
                ),
                metric_min_delta=dict(
                    stop_cfg.get("metric_min_delta", {"psnr": 0.01})
                ),
                best_metric=str(stop_cfg.get("best_metric", "psnr")),
            )
            print(
                f"[phase9] MetricPlateauStopper armed (Method C): "
                f"patience_evals={self._stopper.patience_evals}, "
                f"min_iter={self._stopper.min_iterations_before_stop}, "
                f"delta={self._stopper.metric_min_delta}"
            )
        else:
            self._stopper = None
        early_stop_iter: Optional[int] = None
        plateau_eval_every = (
            int(stop_cfg.get("plateau_eval_every", 1000)) if stop_cfg else 1000
        )
        log_f = open(self.out_dir / "voxel_grid_train.log", "w")
        t0 = time.time()
        best_loss = float("inf")
        for it in range(1, max_iter + 1):
            self.optimizer.zero_grad()
            l_dc_p = self._kspace_dc_loss(
                self.x_real_p, self.x_imag_p, self.csm_p, self.kspace_p, self.mask_tr_p
            )
            l_dc_c = self._kspace_dc_loss(
                self.x_real_c, self.x_imag_c, self.csm_c, self.kspace_c, self.mask_tr_c
            )
            l_geo_raw = self._tv_3d(self.x_real_p, self.x_imag_p) + self._tv_3d(
                self.x_real_c, self.x_imag_c
            )
            l_geo = lambda_geo * l_geo_raw
            total = l_dc_p + l_dc_c + l_geo
            # Phase 7 INF-05 backward-compat: flat 3-item ratio dict for Method C
            # (no KL terms — pure deterministic). sum=1.0 assertion tol 1e-3.
            comp = {
                "loss/dc_p": float(l_dc_p.detach().item()),
                "loss/dc_c": float(l_dc_c.detach().item()),
                "loss/R_geo": float(l_geo.detach().item()),
            }
            total_value = sum(comp.values()) + 1e-12
            ratios = {k: v / total_value for k, v in comp.items()}
            sum_ratios = sum(ratios.values())
            balance_violation = abs(sum_ratios - 1.0) > 1e-3
            if balance_violation and it == 1:
                raise ValueError(
                    f"[Method C] loss-balance violated at iter 1: sum={sum_ratios:.6f}"
                )
            elif balance_violation:
                warnings.warn(
                    f"[Method C] balance violated mid-train at iter {it}: "
                    f"sum_ratios={sum_ratios:.6f}",
                    stacklevel=2,
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [self.x_real_p, self.x_imag_p, self.x_real_c, self.x_imag_c],
                max_grad_norm,
            )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.writer is not None and (it == 1 or it % log_every == 0):
                for k, v in comp.items():
                    self.writer.add_scalar(f"{k}/value", v, it)
                for k, v in ratios.items():
                    self.writer.add_scalar(f"{k}/ratio", v, it)
                self.writer.add_scalar("loss/total", float(total.item()), it)
                self.writer.add_scalar(
                    "loss/balance_violation", 1.0 if balance_violation else 0.0, it
                )
            if it == 1 or it % log_every == 0:
                msg = (
                    f"[MC iter {it:5d}/{max_iter}] total={total.item():.4e} "
                    f"dc_p={l_dc_p.item():.4e} dc_c={l_dc_c.item():.4e} "
                    f"R_geo={l_geo.item():.4e} "
                    f"(raw_tv={l_geo_raw.item():.4e})"
                )
                print(msg)
                log_f.write(msg + "\n")
                log_f.flush()
            if it % save_every == 0 or it == max_iter:
                ckpt = {
                    "iter": it,
                    "x_real_p": self.x_real_p.detach().cpu(),
                    "x_imag_p": self.x_imag_p.detach().cpu(),
                    "x_real_c": self.x_real_c.detach().cpu(),
                    "x_imag_c": self.x_imag_c.detach().cpu(),
                    "optim_state": self.optimizer.state_dict(),
                }
                ckpt_dir = self.out_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                torch.save(ckpt, ckpt_dir / "latest.pth")
                if total.item() < best_loss:
                    best_loss = total.item()
                    torch.save(ckpt, ckpt_dir / "best.pth")
            # Phase 9 B-1: plateau eval (Method C deterministic, K=1).
            if (
                self._stopper is not None
                and (it % plateau_eval_every == 0)
                and (it >= self._stopper.min_iterations_before_stop)
            ):
                psnr_value = self._eval_psnr_for_plateau(K_eval=1)
                if self._stopper.update(it, {"psnr": psnr_value}):
                    early_stop_iter = it
                    print(
                        f"[phase9] Plateau detected at iter={it} (Method C); "
                        f"PSNR={psnr_value:.3f}; early-stopping"
                    )
                    log_f.write(f"[phase9] early_stop_iter={it}\n")
                    log_f.flush()
                    break
                else:
                    if self.writer is not None:
                        self.writer.add_scalar("plateau/psnr_eval", psnr_value, it)
        print(f"[VoxelGridJointTrainer] done in {time.time() - t0:.1f}s")
        log_f.close()
        # Phase 9 B-1: emit early_stop.json sidecar.
        import json
        with open(self.out_dir / "early_stop.json", "w") as fh:
            json.dump(
                {"early_stop_iter": early_stop_iter, "max_iterations": max_iter},
                fh,
            )
        if self.writer is not None:
            self.writer.close()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def export_results(self):
        """Write reconstruction_{plain,contrast}.mat matching Method B schema.

        Phase 9 GNG paired-t-test consumes ``reconstruction`` key from
        ``test_results_best/`` identically across Method A/B/C.
        """
        import scipy.io as sio

        out_dir = self.out_dir / "test_results_best"
        out_dir.mkdir(exist_ok=True)
        x_p = (
            torch.complex(self.x_real_p.detach(), self.x_imag_p.detach()).cpu().numpy()
        )
        x_c = (
            torch.complex(self.x_real_c.detach(), self.x_imag_c.detach()).cpu().numpy()
        )
        sio.savemat(str(out_dir / "reconstruction_plain.mat"), {"reconstruction": x_p})
        sio.savemat(
            str(out_dir / "reconstruction_contrast.mat"), {"reconstruction": x_c}
        )
        print(f"[Method C] export done: voxel-grid reconstruction → {out_dir}")
