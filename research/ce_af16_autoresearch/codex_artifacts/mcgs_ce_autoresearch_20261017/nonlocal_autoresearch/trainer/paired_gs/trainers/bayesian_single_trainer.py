"""BayesianSingleTrainer (Phase 8 / BP-07 Method A / GNG-02).

Independent single-contrast Bayesian GS training, for Go/No-Go Method A.
Total parameter count = 2N (N Gaussians per contrast x 2 contrasts, run twice).

B-1 fix per revision loop 1 (2026-05-27):
    Method A 真 2N total 协议 -- uses BayesianJointGaussianModel3D's new
    ``single_contrast_channel`` parameter (08-01 revision-loop-1 fix) to
    UNREGISTER opposite-side *_contrast / *_plain nn.Parameter. No freeze-grad
    hack on dual model; each Method A run truly has its own N Gaussians.

Fairness protocol with BayesianJointTrainer (Method B):
  Same lr / sanity-offset schedule / max_iter / init / K_train=2 / beta_a /
  seed / voxelizer / deterministic flag / ELBO weights -- ONLY differences are:
    - data block (single contrast vs joint)
    - geometry NOT shared (each contrast has its own N Gaussians; total 2N)
    - opposite-side density_* / log_s2_* / eta_* nn.Parameter unregistered
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch

from data.transforms import fft3c, ifft3c
from gaussian.bayesian_joint_model import BayesianJointGaussianModel3D
from gaussian.tile_voxelizer import TileVoxelizer
from gaussian.voxelizer import Voxelizer
from trainers.bayesian_joint_trainer import BayesianJointTrainer, _load_dataset_split
from trainers.bayesian_warm_start import init_posterior_from_v1_checkpoint


class BayesianSingleTrainer(BayesianJointTrainer):
    """Independent single-contrast Bayesian (Method A) -- true 2N total per B-1 fix.

    Subclass of BayesianJointTrainer: inherits _setup_loss / _setup_optimizer /
    _setup_output / _setup_collapse_monitor_state / _setup_seeds / _validate_config /
    _render_sample / _kspace_dc_loss / _beta_a_schedule unchanged (fairness protocol).

    Overrides:
      - _setup_data: load only ONE contrast (T1 OR FLAIR by config)
      - _setup_model: instantiate BayesianJointGaussianModel3D(single_contrast_channel=...)
                     -- opposite-side density_* nn.Parameter is UNREGISTERED (B-1 fix)
      - train: render only ACTIVE channel; dc_loss only for ACTIVE channel;
               ElboLoss with vol_*_for_geo=None for inactive channel (B-4 fix consumer)
      - _log_collapse_monitor: only inspect active channel's s2_*
      - export_results: dump single reconstruction.mat (no _plain/_contrast suffix)
    """

    # ------------------------------------------------------------------ #
    def _setup_data(self):
        """Method A: load only ONE contrast.

        Config layout: ``data`` dict has either ``plain`` key (T1 run) OR
        ``contrast`` key (FLAIR run), not both. The channel name drives
        ``single_contrast_channel`` for the model.
        """
        data_cfg = self.config["data"]
        if "plain" in data_cfg:
            single_cfg = data_cfg["plain"]
            self._channel_name = "plain"
        elif "contrast" in data_cfg:
            single_cfg = data_cfg["contrast"]
            self._channel_name = "contrast"
        else:
            # Flat layout (no plain/contrast sub-key) -- default to plain.
            single_cfg = data_cfg
            self._channel_name = "plain"
        self.ds_p = _load_dataset_split(single_cfg)
        self.ds_c = None
        self.volume_shape = self.ds_p.volume_shape
        self.kspace_p = self.ds_p.kspace_undersampled_complex.to(self.device)
        self.csm_p = self.ds_p.sensitivity_maps.to(self.device)
        if self.ds_p.mask_split:
            self.mask_tr_p = self.ds_p.mask_split["tr"].to(self.device)
            self.mask_fb_p = self.ds_p.mask_split["fb"].to(self.device)
            self.mask_te_p = self.ds_p.mask_split["te"].to(self.device)
            self._mask_te_locked_for_trainer = True
        else:
            self.mask_tr_p = self.ds_p.mask.to(self.device)
            self.mask_fb_p = self.mask_te_p = None
            self._mask_te_locked_for_trainer = False
        # Inactive channel slots -- None placeholders so subclass attribute
        # access patterns from parent stay safe in case any override probes.
        self.mask_tr_c = self.mask_fb_c = self.mask_te_c = None
        self.kspace_c = self.csm_c = None
        with torch.no_grad():
            self.zf_p = torch.sum(
                torch.conj(self.csm_p) * ifft3c(self.kspace_p), dim=0
            )
        self.zf_c = None
        print(
            f"[BayesianSingleTrainer] Method A single channel = {self._channel_name},"
            f" volume_shape={self.volume_shape},"
            f" N_kspace={self.kspace_p.shape[0]} coils"
        )

    # ------------------------------------------------------------------ #
    def _setup_model(self):
        """B-1 fix: instantiate model in single-channel mode.

        The unused channel's density_* nn.Parameter will NOT be registered
        (BayesianJointGaussianModel3D UNREGISTER logic), so no freeze-grad
        hack needed; total parameters per Method A run = N independent Gaussians.
        """
        gcfg = self.config["gaussian"]
        bcfg = self.config["bayesian"]
        warm_path = (
            bcfg.get("warm_start", {}).get("ckpt_path")
            or gcfg.get("warm_start_checkpoint")
        )
        d, h, w = self.volume_shape

        if warm_path and Path(warm_path).exists():
            ckpt = torch.load(warm_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state", ckpt.get("gaussian_state", ckpt))
            n_init = sd["positions"].shape[0]
            positions = sd["positions"].to(self.device)
            scales_log = sd["scales"].to(self.device)
            rotations = sd["rotations"].to(self.device)
            # Method A uses v1 SINGLE-contrast warm-start ckpt (hyw_t1_v2 or
            # hyw_flair_v2), which has only density_real/density_imag (no
            # *_contrast). For channel='contrast' we reinterpret those tensors
            # as the contrast init.
            d_r = sd["density_real"].to(self.device)
            d_i = sd["density_imag"].to(self.device)
            init_density = torch.complex(d_r, d_i)
            self.model = BayesianJointGaussianModel3D(
                num_points=n_init,
                volume_shape=self.volume_shape,
                device=self.device,
                initial_positions=positions,
                initial_densities_plain=(
                    init_density if self._channel_name == "plain" else None
                ),
                initial_densities_contrast=(
                    init_density if self._channel_name == "contrast" else None
                ),
                initial_scales=torch.exp(scales_log),
                sigma2_init=float(bcfg.get("sigma2_init", 1e-3)),
                log_eta_init=float(bcfg.get("log_eta_init", -1.0)),
                sigma2_min=float(bcfg.get("sigma2_min", 1e-8)),
                single_contrast_channel=self._channel_name,  # B-1 fix
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
            # B-1 联动: helper now returns m_0 dict with only active-channel keys.
            self.m_0 = {k: v.to(self.device) for k, v in ws_info["m_0"].items()}
            for v in self.m_0.values():
                v.requires_grad_(False)
        else:
            # Cold start.
            n_init = gcfg.get("initial_num_points", 200000)
            positions = torch.rand(n_init, 3, device=self.device) * 2 - 1
            with torch.no_grad():
                ix = ((positions[:, 0] + 1) * 0.5 * d).clamp(0, d - 1).long()
                iy = ((positions[:, 1] + 1) * 0.5 * h).clamp(0, h - 1).long()
                iz = ((positions[:, 2] + 1) * 0.5 * w).clamp(0, w - 1).long()
                d_seed = self.zf_p[ix, iy, iz].to(torch.complex64) * 0.1
            base_scale = gcfg.get("init_scale", 2.0 / max(d, h, w))
            scales = torch.ones(n_init, 3, device=self.device) * base_scale
            self.model = BayesianJointGaussianModel3D(
                num_points=n_init,
                volume_shape=self.volume_shape,
                device=self.device,
                initial_positions=positions,
                initial_densities_plain=(
                    d_seed if self._channel_name == "plain" else None
                ),
                initial_densities_contrast=(
                    d_seed if self._channel_name == "contrast" else None
                ),
                initial_scales=scales,
                sigma2_init=float(bcfg.get("sigma2_init", 1e-3)),
                log_eta_init=float(bcfg.get("log_eta_init", -1.0)),
                sigma2_min=float(bcfg.get("sigma2_min", 1e-8)),
                single_contrast_channel=self._channel_name,  # B-1 fix
            ).to(self.device)
            # Cold-start m_0: snapshot current active-channel density_*.
            if self._channel_name == "plain":
                self.m_0 = {
                    "m_0_real_plain": self.model.density_real.detach().clone(),
                    "m_0_imag_plain": self.model.density_imag.detach().clone(),
                }
            else:
                self.m_0 = {
                    "m_0_real_contrast":
                        self.model.density_real_contrast.detach().clone(),
                    "m_0_imag_contrast":
                        self.model.density_imag_contrast.detach().clone(),
                }
            for v in self.m_0.values():
                v.requires_grad_(False)
        print(
            f"[BayesianSingleTrainer] N={self.model.num_points},"
            f" channel={self._channel_name},"
            f" has_plain={self.model.has_plain},"
            f" has_contrast={self.model.has_contrast}"
        )
        # Voxelizer (same convention as joint).
        vcfg = self.config.get("voxelizer", {})
        vtype = vcfg.get("type", "tile_cuda")
        if vtype in {"tile", "tile_cuda"}:
            self.voxelizer = TileVoxelizer(
                volume_shape=self.volume_shape,
                tile_size=vcfg.get("tile_size", 8),
                max_radius=vcfg.get("max_radius", 20),
                use_cuda=(vtype == "tile_cuda"),
                strict_cuda=bool(
                    self.config.get("mode", {}).get("strict_cuda", False)
                ) and vtype == "tile_cuda",
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
    @torch.no_grad()
    def _eval_psnr_for_plateau(self, K_eval: int = 16) -> float:
        """Method A plateau eval: PSNR on the single active channel."""
        from metrics import compute_psnr
        xs = []
        for _ in range(K_eval):
            samples_k = self.model.sample(K=1, disable_mc_for_init=False)
            vol_k = self._render_sample(samples_k, 0, self._channel_name)
            xs.append(vol_k.detach())
        x_bar = torch.stack(xs, dim=0).mean(dim=0)
        ref = self.zf_p  # single-channel mode always uses the _p slot
        return float(compute_psnr(torch.abs(x_bar), torch.abs(ref)))

    # ------------------------------------------------------------------ #
    def train(self):
        """Method A train loop: render + dc_loss only for ACTIVE channel.

        ElboLoss inspects model.has_plain / model.has_contrast (B-4 fix) and
        only computes the active channel's KL + R_sp contributions. Pass the
        active channel's volume as the corresponding vol_*_for_geo; None for
        the inactive.
        """
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
        disable_mc_for_init = bool(
            bcfg["diagnostics"].get("disable_mc_for_init", True)
        )
        grad_clip_amp = float(train_cfg.get("grad_clip_per_param_amp", 100.0))
        max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        # Phase 9 B-1: instantiate MetricPlateauStopper from config.stopping schema.
        stop_cfg = self.config.get("stopping", {})
        from metrics import MetricPlateauStopper
        if stop_cfg:
            self._stopper = MetricPlateauStopper(
                patience_evals=int(stop_cfg.get("patience_evals", 4)),
                min_iterations_before_stop=int(
                    stop_cfg.get("min_iterations_before_stop", 1000)
                ),
                metric_min_delta=dict(
                    stop_cfg.get("metric_min_delta", {"psnr": 0.01})
                ),
                best_metric=str(stop_cfg.get("best_metric", "psnr")),
            )
            print(
                f"[phase9] MetricPlateauStopper armed (Method A "
                f"channel={self._channel_name}): "
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
        K_eval_plateau = int(bcfg.get("K_eval", 16))
        log_f = open(self.out_dir / "bayesian_single_train.log", "w")
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
            samples = self.model.sample(
                K=K_this_iter, disable_mc_for_init=in_sanity_window
            )
            l_dc_total = torch.zeros((), device=self.device)
            vol_mean_acc = torch.zeros_like(self.zf_p)
            for k in range(K_this_iter):
                # Method A: only render ACTIVE channel.
                vol_k = self._render_sample(samples, k, self._channel_name)
                l_dc_k = self._kspace_dc_loss(
                    vol_k, self.csm_p, self.kspace_p, self.mask_tr_p
                )
                l_dc_total = l_dc_total + l_dc_k
                vol_mean_acc = vol_mean_acc + vol_k.detach()
            l_dc_mc = l_dc_total / float(K_this_iter)
            vol_for_geo = vol_mean_acc / float(K_this_iter)
            vol_p_for_geo = vol_for_geo if self._channel_name == "plain" else None
            vol_c_for_geo = (
                vol_for_geo if self._channel_name == "contrast" else None
            )
            elbo_out = self.elbo_loss(
                l_dc_mc, self.model, self.m_0, vol_p_for_geo, vol_c_for_geo,
                beta_a_override=beta_a_now,
                beta_mu_override=beta_mu_now,
                beta_ell_override=beta_ell_now,
                disable_kl=in_sanity_window,
            )
            total = elbo_out["total"]
            ratios = elbo_out["ratios"]

            # Phase 7 INF-05 / B-2 6-ratio balance check (same as Method B).
            sum_ratios = sum(ratios.values())
            balance_violation = abs(sum_ratios - 1.0) > 1e-3
            if balance_violation and it == 1:
                raise ValueError(
                    f"[Method A/INF-05] loss-balance violated at iter 1: "
                    f"sum(ratios)={sum_ratios:.6f}, ratios={ratios}"
                )
            elif balance_violation:
                warnings.warn(
                    f"[Method A/INF-05] loss-balance violated mid-train at "
                    f"iter {it}: sum(ratios)={sum_ratios:.6f}",
                    stacklevel=2,
                )

            total.backward()

            # PITFALLS A-03 per-param L-infinity clip on amplitude posterior,
            # restricted to existing params (single-channel mode only has the
            # active side; the absent attributes must not be touched).
            with torch.no_grad():
                active_amp_params: List[torch.Tensor] = []
                if self.model.has_plain:
                    active_amp_params += [
                        self.model.density_real,
                        self.model.density_imag,
                        self.model.log_s2_plain,
                        self.model.eta_plain,
                    ]
                if self.model.has_contrast:
                    active_amp_params += [
                        self.model.density_real_contrast,
                        self.model.density_imag_contrast,
                        self.model.log_s2_contrast,
                        self.model.eta_contrast,
                    ]
                for amp_p in active_amp_params:
                    if amp_p.grad is not None:
                        amp_p.grad.data.clamp_(-grad_clip_amp, grad_clip_amp)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self._adaptive_density_control(it)

            if self.writer is not None and (it == 1 or it % log_every == 0):
                for k_, v_ in elbo_out["values"].items():
                    self.writer.add_scalar(f"{k_}/value", v_, it)
                for k_, v_ in ratios.items():
                    self.writer.add_scalar(f"{k_}/ratio", v_, it)
                self.writer.add_scalar(
                    "loss/balance_violation",
                    1.0 if balance_violation else 0.0,
                    it,
                )
                self.writer.add_scalar("loss/total", float(total.item()), it)
                self.writer.add_scalar("kl_schedule/beta_a", beta_a_now, it)
                self.writer.add_scalar("kl_schedule/beta_mu", beta_mu_now, it)
                self.writer.add_scalar("kl_schedule/beta_ell", beta_ell_now, it)

            if it % self._collapse_log_every == 0:
                self._log_collapse_monitor(it)

            if it == 1 or it % log_every == 0:
                msg = (
                    f"[MA iter {it:5d}/{max_iter} ch={self._channel_name}] "
                    f"total={total.item():.4e} "
                    f"dc={elbo_out['values']['loss/dc']:.4e} "
                    f"kl/a={elbo_out['values']['kl/a']:.4e} "
                    f"beta_a={beta_a_now:.3f} K={K_this_iter} "
                    f"sanity={in_sanity_window}"
                )
                print(msg)
                log_f.write(msg + "\n")
                log_f.flush()

            if it % save_every == 0 or it == max_iter:
                ckpt = {
                    "iter": it,
                    "channel": self._channel_name,
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
            # Phase 9 B-1: plateau eval (single-channel PSNR).
            if (
                self._stopper is not None
                and (it % plateau_eval_every == 0)
                and (it >= self._stopper.min_iterations_before_stop)
            ):
                psnr_value = self._eval_psnr_for_plateau(K_eval=K_eval_plateau)
                if self._stopper.update(it, {"psnr": psnr_value}):
                    early_stop_iter = it
                    print(
                        f"[phase9] Plateau detected at iter={it} "
                        f"(channel={self._channel_name}); "
                        f"PSNR={psnr_value:.3f}; early-stopping"
                    )
                    log_f.write(f"[phase9] early_stop_iter={it}\n")
                    log_f.flush()
                    break
                else:
                    if self.writer is not None:
                        self.writer.add_scalar("plateau/psnr_eval", psnr_value, it)
        print(
            f"[BayesianSingleTrainer] done in {time.time() - t0:.1f}s "
            f"(channel={self._channel_name})"
        )
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
    def _log_collapse_monitor(self, it: int):
        """Single-channel collapse monitor (inspect ACTIVE channel(s) only)."""
        if self.writer is None:
            return
        warn_thresh = self._collapse_warn_p50_factor * self.model.sigma2_min
        s2_mu = self.model._decode_s2(self.model.log_sigma2_mu)
        self.writer.add_scalar(
            "posterior/log_sigma2_mu/p05",
            torch.quantile(s2_mu.detach(), 0.05).item(), it,
        )
        self.writer.add_scalar(
            "posterior/log_sigma2_mu/p50",
            torch.quantile(s2_mu.detach(), 0.50).item(), it,
        )
        self.writer.add_scalar(
            "posterior/log_sigma2_mu/p95",
            torch.quantile(s2_mu.detach(), 0.95).item(), it,
        )
        window_size = max(
            self._collapse_warn_rolling_iters // self._collapse_log_every, 1
        )
        if self.model.has_plain:
            s2 = self.model._decode_s2(self.model.log_s2_plain)
            p50_p = torch.quantile(s2.detach(), 0.50).item()
            self.writer.add_scalar(
                "posterior/s2_plain/p05",
                torch.quantile(s2.detach(), 0.05).item(), it,
            )
            self.writer.add_scalar("posterior/s2_plain/p50", p50_p, it)
            self.writer.add_scalar(
                "posterior/s2_plain/p95",
                torch.quantile(s2.detach(), 0.95).item(), it,
            )
            self._collapse_window_p.append(1 if p50_p < warn_thresh else 0)
            self._collapse_window_p = self._collapse_window_p[-window_size:]
            warn_p = 1 if (
                len(self._collapse_window_p) == window_size
                and all(self._collapse_window_p)
            ) else 0
            self.writer.add_scalar("collapse_warn/s2_plain", warn_p, it)
            if warn_p == 1:
                warnings.warn(
                    f"[Method A/A-01] Posterior collapse suspected at iter {it}: "
                    f"P50(s2_plain)={p50_p:.3e}, threshold={warn_thresh:.3e} "
                    f"sustained {self._collapse_warn_rolling_iters} iter",
                    stacklevel=2,
                )
        if self.model.has_contrast:
            s2 = self.model._decode_s2(self.model.log_s2_contrast)
            p50_c = torch.quantile(s2.detach(), 0.50).item()
            self.writer.add_scalar(
                "posterior/s2_contrast/p05",
                torch.quantile(s2.detach(), 0.05).item(), it,
            )
            self.writer.add_scalar("posterior/s2_contrast/p50", p50_c, it)
            self.writer.add_scalar(
                "posterior/s2_contrast/p95",
                torch.quantile(s2.detach(), 0.95).item(), it,
            )
            self._collapse_window_c.append(1 if p50_c < warn_thresh else 0)
            self._collapse_window_c = self._collapse_window_c[-window_size:]
            warn_c = 1 if (
                len(self._collapse_window_c) == window_size
                and all(self._collapse_window_c)
            ) else 0
            self.writer.add_scalar("collapse_warn/s2_contrast", warn_c, it)
            if warn_c == 1:
                warnings.warn(
                    f"[Method A/A-01] Posterior collapse suspected at iter {it}: "
                    f"P50(s2_contrast)={p50_c:.3e}, threshold={warn_thresh:.3e} "
                    f"sustained {self._collapse_warn_rolling_iters} iter",
                    stacklevel=2,
                )

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def export_results(self):
        """Method A export: channel-suffixed reconstruction file (B-3 fix).

        Writes ``reconstruction_plain.mat`` (when channel='plain') OR
        ``reconstruction_contrast.mat`` (when channel='contrast') -- NOT a
        unified ``reconstruction.mat``. Phase 9 GNG runner globs
        ``test_results_best/reconstruction_{plain,contrast}.mat`` uniformly
        across Method A / B / C output dirs.

        Also writes ``U_p.npy`` (channel='plain') or ``U_c.npy``
        (channel='contrast') to align with Method B output schema.

        B-3 fix per revision-loop-1 fixes_to_apply.
        """
        import scipy.io as sio
        out_dir = self.out_dir / "test_results_best"
        out_dir.mkdir(exist_ok=True)
        K_eval = int(self.config["bayesian"].get("K_eval", 16))
        xs: List[torch.Tensor] = []
        for _ in range(K_eval):
            samples_k = self.model.sample(K=1, disable_mc_for_init=False)
            vol_k = self._render_sample(samples_k, 0, self._channel_name)
            xs.append(vol_k.cpu())
        X = torch.stack(xs, dim=0)
        x_bar = X.mean(dim=0)
        U = ((X - x_bar.unsqueeze(0)).abs().pow(2)).sum(dim=0) / max(K_eval - 1, 1)
        recon = self._hard_dc(
            x_bar.to(self.device), self.csm_p, self.kspace_p, self.mask_tr_p,
        ).cpu()
        # B-3 fix: channel-suffixed reconstruction filename + U sidecar.
        suffix = "_plain" if self._channel_name == "plain" else "_contrast"
        recon_filename = f"reconstruction{suffix}.mat"
        harddc_filename = f"reconstruction{suffix}_harddc.mat"
        u_filename = f"U_{'p' if self._channel_name == 'plain' else 'c'}.npy"
        sio.savemat(
            str(out_dir / recon_filename),
            {
                "reconstruction": x_bar.numpy(),
                "channel": self._channel_name,
                "source": "posterior_mean_raw",
            },
        )
        sio.savemat(
            str(out_dir / harddc_filename),
            {
                "reconstruction": recon.numpy(),
                "channel": self._channel_name,
                "dc_mask": "tr",
            },
        )
        np.save(out_dir / u_filename, U.numpy())
        print(
            f"[Method A] export done: K_eval={K_eval}, "
            f"channel={self._channel_name} -> {out_dir}/{recon_filename}, {u_filename}"
        )
