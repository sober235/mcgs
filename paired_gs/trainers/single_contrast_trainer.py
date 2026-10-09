"""SingleContrastTrainer: one contrast, its own Gaussian geometry, in its own process.

The GS-single arm of the shared-geometry 2x2 ablation (docs/experiments/2026-10-09-mcgs-shared-geometry-2x2-ablation.md).
It reuses JointTrainer's data loading, model construction (init modes, 3-NN scales), voxelizer, optimiser,
checkpointing and logging, and replaces two things:

  _setup_data  loads data.plain only; every `*_c` attribute JointTrainer exposes is an alias of the plain one so the
               inherited helpers (_setup_model's union_magnitude(zf_p, zf_c), the scorer's tr.kspace_c, ...) see one
               channel. Nothing of a second contrast is read.
  train        the plain half of JointTrainer.train's objective:
                   L = L_dc(plain) + lambda_geo * R_geo(plain) + lambda_wav_plain * R_wav(plain) + lambda_ptv * R_ptv(plain)
               with the same edge-TV kappa rule, the same wavelet cycle-spin generator and the same clip / scheduler /
               checkpoint schedule. The contrast density parameters that JointGaussianModel3D allocates are never
               rendered and never receive a gradient (asserted in tests/test_sg2x2_single_trainer.py).

Every cross-contrast weight, the private residual, densification, a file guide, a warm start and a data.contrast block
are refused at construction: this trainer exists to be a clean single-contrast control, not a general one.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import asdict
from typing import Any, Dict

import torch

from data.known_zero import build_fit_tensors, estimate_known_zero, estimate_noise_cov, known_zero_from_support, \
    load_support_map, whitening_matrix
from data.transforms import ifft3c
from losses.edge_tv import edge_preserving_tv, resolve_edge_kappa, tv_magnitude
from trainers.joint_trainer import JointTrainer, _load_dataset

_REFUSED_LOSS_WEIGHTS = ("lambda_dtv", "lambda_edge", "lambda_vtv", "lambda_dtv_proj", "edge_tv_guide_alpha",
                         "ptv_guide_alpha", "lambda_nl", "lambda_nl_plain", "lambda_nl_contrast", "lambda_private_sp")
_REFUSED_LOSS_KEYS = ("guide_path_plain", "guide_path_contrast", "kspace_high_weight_plain")


class SingleContrastTrainer(JointTrainer):
    """One contrast per process; see the module docstring."""

    def __init__(self, config: Dict[str, Any], device: torch.device | None = None):
        self._refuse_out_of_scope(config)
        super().__init__(config, device)

    # -- scope guard ---------------------------------------------------
    @staticmethod
    def _refuse_out_of_scope(config: Dict[str, Any]) -> None:
        data = config.get("data", {})
        if "contrast" in data:
            raise ValueError("SingleContrastTrainer takes data.plain only; remove the data.contrast block")
        if "plain" not in data:
            raise ValueError("SingleContrastTrainer needs data.plain")
        loss = config.get("loss", {})
        for k in _REFUSED_LOSS_WEIGHTS:
            if float(loss.get(k, 0.0) or 0.0) > 0:
                raise ValueError(f"loss.{k} is a cross-contrast / out-of-scope term; it must be 0 in a single-contrast run")
        for k in _REFUSED_LOSS_KEYS:
            if loss.get(k) not in (None, "", False):
                raise ValueError(f"loss.{k} is not allowed in a single-contrast run")
        if bool(config.get("adaptive_control", {}).get("enable", False)):
            raise ValueError("adaptive_control.enable must be false in a single-contrast run (fixed capacity)")
        g = config.get("gaussian", {})
        if int(g.get("num_private_points", 0)) > 0:
            raise ValueError("gaussian.num_private_points must be 0 in a single-contrast run")
        if g.get("warm_start_checkpoint"):
            raise ValueError("gaussian.warm_start_checkpoint is not allowed in a single-contrast run")

    # -- data: plain only, contrast attributes alias plain ---------------
    def _setup_data(self):
        data_cfg = self.config["data"]
        print(">>> loading PLAIN (single-contrast run)")
        self.ds_p = _load_dataset(data_cfg["plain"])
        self.ds_c = self.ds_p
        self.volume_shape = self.ds_p.volume_shape
        print(f"volume_shape = {self.volume_shape}")
        self.kspace_p = self.ds_p.kspace_undersampled_complex.to(self.device)
        self.csm_p = self.ds_p.sensitivity_maps.to(self.device)
        self.mask_p = self.ds_p.mask.to(self.device)
        self.kspace_c, self.csm_c, self.mask_c = self.kspace_p, self.csm_p, self.mask_p
        with torch.no_grad():
            self.zf_p = torch.sum(torch.conj(self.csm_p) * ifft3c(self.kspace_p), dim=0)
        self.zf_c = self.zf_p
        print(f"|zf_p| max {self.zf_p.abs().max().item():.4f}")

        # physics levers for the one channel, exactly as JointTrainer._setup_data does for tag "p"
        self.mask_fit_p, self.kspace_dc_p, self.kspace_fit_p, self.csm_fit_p = self.mask_p, self.kspace_p, self.kspace_p, self.csm_p
        self._dc_norm_p = max(self.mask_p.sum().item(), 1.0)
        self.physics_info: Dict[str, Any] = {}
        sub = data_cfg["plain"]
        use_kz = bool(sub.get("known_zero", False))
        use_pw = bool(sub.get("prewhiten", self.config.get("loss", {}).get("prewhiten", False)))
        if use_kz or use_pw:
            y, m, S = self.kspace_p, self.mask_p, self.csm_p
            sup_path = sub.get("known_zero_support")
            if sup_path:
                Z, info = known_zero_from_support(load_support_map(sup_path), y, m, int(self.ds_p.readout_axis))
            else:
                Z, info = estimate_known_zero(y, m, int(self.ds_p.readout_axis))
            W, winfo = None, {}
            if use_pw:
                Sigma, winfo = estimate_noise_cov(y, m, Z, info)
                W = whitening_matrix(Sigma) if Sigma is not None else None
            if not use_kz:
                Z = torch.zeros_like(m, dtype=torch.bool)
            self.mask_fit_p, self.kspace_dc_p, self.kspace_fit_p, self.csm_fit_p = build_fit_tensors(y, m, S, Z, W)
            rec = {"known_zero": use_kz, "prewhiten": bool(use_pw and W is not None), **asdict(info), **winfo,
                   "frac_z_used": float(Z.float().mean())}
            self.physics_info["p"] = rec
            print(f"[physics:plain] known_zero={use_kz} support={sup_path or 'estimated'} cuts={info.cuts} "
                  f"ellipse={info.ellipse} frac_z_used={rec['frac_z_used']:.4f} "
                  f"zero_lines={info.n_zero_lines}/{info.n_sampled_lines} prewhiten={rec['prewhiten']}")
        self.mask_fit_c, self.kspace_dc_c, self.kspace_fit_c, self.csm_fit_c = \
            self.mask_fit_p, self.kspace_dc_p, self.kspace_fit_p, self.csm_fit_p
        self._dc_norm_c = self._dc_norm_p

    # -- rendering: one channel -----------------------------------------
    def _render(self, channel: str) -> torch.Tensor:
        if channel != "plain":
            raise RuntimeError("SingleContrastTrainer has no contrast channel")
        return super()._render("plain")

    # -- train loop: the plain half of JointTrainer.train -----------------
    def train(self):
        train_cfg = self.config["training"]
        loss_cfg = self.config["loss"]
        max_iter = train_cfg.get("max_iterations", 8000)
        log_every = train_cfg.get("log_every", 50)
        save_every = train_cfg.get("save_every", 1000)
        eval_every = train_cfg.get("eval_every", 200)

        if "tv_weight" in loss_cfg and "lambda_geo" not in loss_cfg:
            warnings.warn("loss.tv_weight is deprecated; read as lambda_geo", DeprecationWarning, stacklevel=2)
            lambda_geo = float(loss_cfg["tv_weight"])
        else:
            lambda_geo = float(loss_cfg.get("lambda_geo", 5e-4))
        lambda_ptv = float(loss_cfg.get("lambda_ptv", 0.0))
        if lambda_ptv > 0:
            if not bool(loss_cfg.get("edge_tv", False)):
                raise ValueError("loss.lambda_ptv reuses the edge-TV kappa / w_floor; enable loss.edge_tv")
            from losses.phase_tv import phase_tv
            print(f"[phase-tv] lambda_ptv={lambda_ptv} eps={float(loss_cfg.get('tv_eps', 0.0))}")
        lambda_wav = float(loss_cfg.get("lambda_wav", 0.0))
        wav_lam_p = float(loss_cfg.get("lambda_wav_plain", lambda_wav))
        if lambda_wav > 0:
            from losses.wavelet_l1 import wavelet_l1
            wav_name = str(loss_cfg.get("wav_name", "db2"))
            wav_levels = int(loss_cfg.get("wav_levels", 3))
            wav_eps = float(loss_cfg.get("wav_eps", 0.0))
            wav_spin = bool(loss_cfg.get("wav_cycle_spin", True))
            wav_on = str(loss_cfg.get("wav_on", "complex"))
            wav_gen = torch.Generator().manual_seed(int(train_cfg.get("seed", 42)) + 7919)
            print(f"[wavelet] lambda={lambda_wav} (plain {wav_lam_p}) wavelet={wav_name} levels={wav_levels} "
                  f"eps={wav_eps} cycle_spin={wav_spin} on={wav_on}")

        self._static_freq_w = self._build_freq_weight()
        self._static_freq_w_plain = None
        self._edge_tv = bool(loss_cfg.get("edge_tv", False))
        self._edge_tv_wfloor = float(loss_cfg.get("edge_tv_wfloor", 0.1))
        self._tv_eps = float(loss_cfg.get("tv_eps", 0.0))
        self._tv_vector = bool(loss_cfg.get("tv_vector", False))
        if self._tv_eps > 0 or self._tv_vector:
            print(f"[tv_shape] eps={self._tv_eps} vector={self._tv_vector}")
        _kappa_refresh = 0
        if self._edge_tv:
            _kq = float(loss_cfg.get("edge_tv_kappa_q", 0.75))
            _ksrc = str(loss_cfg.get("edge_tv_kappa_source", "render"))
            _kappa_refresh = int(loss_cfg.get("edge_tv_kappa_refresh_iter", 0))
            with torch.no_grad():
                if _ksrc == "render":
                    self._edge_kappa_p = resolve_edge_kappa(self._render("plain"), _kq)
                elif _ksrc == "zf":
                    self._edge_kappa_p = resolve_edge_kappa(self.zf_p, _kq)
                else:
                    raise ValueError(f"loss.edge_tv_kappa_source must be 'render' or 'zf', got {_ksrc!r}")
            print(f"[edge_tv] kappa_q={_kq} source={_ksrc} refresh_iter={_kappa_refresh} -> "
                  f"kappa_plain={self._edge_kappa_p:.5f} w_floor={self._edge_tv_wfloor}")
        print(f"[single-contrast] lambda_geo={lambda_geo} edge_tv={self._edge_tv} lambda_wav_plain={wav_lam_p} "
              f"lambda_ptv={lambda_ptv} (every cross-contrast weight 0 by construction)")

        log_f = open(self.out_dir / "joint_train.log", "w")
        zero = torch.tensor(0.0, device=self.device)
        t0 = time.time()
        best_loss = float("inf")
        for it in range(1, max_iter + 1):
            self.optimizer.zero_grad()
            freq_w = self._freq_weight_curriculum(it, max_iter)
            vol_p = self._render("plain")
            if self._edge_tv and _kappa_refresh > 0 and it == _kappa_refresh:
                with torch.no_grad():
                    self._edge_kappa_p = resolve_edge_kappa(vol_p, float(loss_cfg.get("edge_tv_kappa_q", 0.75)))
                print(f"[edge_tv] kappa refreshed at iter {it}: kappa_plain={self._edge_kappa_p:.5f}")
            l_dc_p = self._kspace_loss(vol_p, self.csm_fit_p, self.kspace_fit_p, self.mask_fit_p, freq_w, norm=self._dc_norm_p)
            loss = l_dc_p
            if lambda_geo > 0:
                if self._edge_tv:
                    l_geo = edge_preserving_tv(vol_p, self._edge_kappa_p, self._edge_tv_wfloor, self._tv_eps, self._tv_vector)
                else:
                    l_geo = tv_magnitude(vol_p, self._tv_eps, self._tv_vector)
                loss = loss + lambda_geo * l_geo
            else:
                l_geo = zero
            if lambda_ptv > 0:
                l_ptv = phase_tv(vol_p, self._edge_kappa_p, self._edge_tv_wfloor, self._tv_eps)
                loss = loss + lambda_ptv * l_ptv
            else:
                l_ptv = zero
            if lambda_wav > 0:
                sh = tuple(torch.randint(0, 2 ** wav_levels, (3,), generator=wav_gen).tolist()) if wav_spin else None
                l_wav = wavelet_l1([vol_p], wav_name, wav_levels, wav_eps, sh, wav_on)
                loss = loss + wav_lam_p * l_wav
            else:
                l_wav = zero

            if self.writer is not None and (it % log_every == 0 or it == 1):
                self.writer.add_scalar("loss/total", float(loss.item()), it)
                self.writer.add_scalar("loss/dc_p", float(l_dc_p.item()), it)
                self.writer.add_scalar("loss/R_geo", float((lambda_geo * l_geo).item()), it)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg.get("max_grad_norm", 1.0))
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            if it % log_every == 0 or it == 1:
                msg = (f"[iter {it:5d}/{max_iter}] loss={loss.item():.4e} dc_p={l_dc_p.item():.4e} dc_c=0.0000e+00 "
                       f"R_geo={l_geo.item():.4e} R_dtv=0.0000e+00 R_edge=0.0000e+00 R_vtv=0.0000e+00 R_dtvp=0.0000e+00 "
                       f"R_wav={l_wav.item():.4e} R_ptv={l_ptv.item():.4e} R_nl=0.0000e+00 balance_violation=False")
                print(msg)
                log_f.write(msg + "\n"); log_f.flush()
            if it % eval_every == 0:
                with torch.no_grad():
                    snr_p = vol_p.abs().mean().item() / max(vol_p.abs().std().item(), 1e-8)
                    print(f"  eval @ iter {it}: |x_p|.mean/std={snr_p:.3f}")
            if it % save_every == 0 or it == max_iter:
                ckpt_dir = self.out_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                ckpt = {"iter": it, "model_state": self.model.state_dict(), "optim_state": self.optimizer.state_dict()}
                torch.save(ckpt, ckpt_dir / "latest.pth")
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(ckpt, ckpt_dir / "best.pth")
        print(f"single-contrast training done in {time.time() - t0:.1f}s")
        log_f.close()
        if self.writer is not None:
            self.writer.close()
