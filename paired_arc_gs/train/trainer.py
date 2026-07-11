"""PairedARCTrainer — driver for the §13 staged Paired-ARC-GS pipeline.

Layer 1 scope (this file): Stage 1 (ACS structural init) + Stage 2 (fixed-geometry response
fit), plus a general scheduler-driven optimization loop that later layers extend (change
birth / phase / deformation / gabor / CG hook points are present but no-op until their
modules are enabled).

The trainer is a THIN driver: the model (PairedARCModel) holds parameters, the scheduler
(StagePlan) decides per-iteration which param groups are live and which loss terms are
active, and the loss module (arc_losses) computes each term. Geometry freezing is done by
zeroing a param group's LR (not by requires_grad toggling) so Adam moment buffers survive
across stage boundaries.

Reuses from paired_gs: MRIDataset (loader), TileVoxelizer (renderer), fft3c/ifft3c.
"""
from __future__ import annotations

import os
from typing import Dict, Any, List, Optional, Tuple
import torch

# paired_gs shared pieces (framework root + paired_gs both on sys.path — see entry script)
from data.dataset import MRIDataset
from data.transforms import fft3c, ifft3c
from gaussian.tile_voxelizer import TileVoxelizer
from gaussian.voxelizer import Voxelizer

from paired_arc_gs.model.paired_arc_model import PairedARCModel
from paired_arc_gs.modules.interfaces import NoOpPhase, NoOpDeformation, NoOpChange
from paired_arc_gs.train.stages import StagePlan
from paired_arc_gs.train import acs_init
from paired_arc_gs.losses import arc_losses as L


def _load_dataset(dcfg: Dict[str, Any], device: str) -> MRIDataset:
    return MRIDataset(
        data_path=dcfg["data_path"], csm_path=dcfg.get("csm_path"), mask_path=dcfg.get("mask_path"),
        acceleration_factor=dcfg.get("acceleration_factor"),
        mask_type=dcfg.get("mask_type", "external"),
        readout_axis=dcfg.get("readout_axis", 0),
        phase_axes=tuple(dcfg.get("phase_axes", [1, 2])),
        use_acs=dcfg.get("use_acs", False), acs_lines=dcfg.get("acs_lines", 24),
        normalize_kspace=dcfg.get("normalize_kspace", True),
        normalization_percentile=dcfg.get("normalization_percentile", 99.9),
        skip_readout_check=dcfg.get("skip_readout_check", True),
        presampled=dcfg.get("presampled", True), device=device,
    )


class PairedARCTrainer:
    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        self.cfg = config
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.plan = StagePlan.from_config(config)
        print(f"[PairedARC] {self.plan.summary()}")

        # ---- data: two contrasts (§2) ----
        dcfg = config["data"]
        self.name_p = dcfg["plain"].get("name", "plain")
        self.name_c = dcfg["contrast"].get("name", "contrast")
        self.ds_p = _load_dataset(dcfg["plain"], str(self.device))
        self.ds_c = _load_dataset(dcfg["contrast"], str(self.device))
        self.volume_shape = self.ds_p.volume_shape
        assert self.ds_c.volume_shape == self.volume_shape, \
            f"contrast shape {self.ds_c.volume_shape} != plain {self.volume_shape}"

        self._pull_tensors()
        self._build_model()
        self._build_modules()
        self._build_voxelizer()
        self._build_optimizer()
        self._audit_split()

    # ------------------------------------------------------------------ data tensors
    def _pull_tensors(self):
        gp, gc = self.ds_p.get_data(), self.ds_c.get_data()
        self.csm_p = self.ds_p.sensitivity_maps.to(self.device)
        self.csm_c = self.ds_c.sensitivity_maps.to(self.device)
        self.ksp_p = self.ds_p.kspace_undersampled_complex.to(self.device)
        self.ksp_c = self.ds_c.kspace_undersampled_complex.to(self.device)
        self.mask_p = self.ds_p.mask.to(self.device)
        self.mask_c = self.ds_c.mask.to(self.device)
        self.gt_p = gp["ground_truth"].to(self.device)
        self.gt_c = gc["ground_truth"].to(self.device)
        # coil-noise pre-whitening matrices (§12.1): None unless supplied -> DC = unweighted LS
        self.Linv_p = None
        self.Linv_c = None

    # ------------------------------------------------------------------ §13-S1 ACS init
    def _build_model(self):
        mcfg = self.cfg["model"]; n = int(mcfg.get("num_points", 200000))
        ro = self.cfg["data"]["plain"].get("readout_axis", 0)
        acs_lines = self.cfg["data"]["plain"].get("acs_lines", 24)
        x1 = acs_init.acs_lowres_image(self.ds_p.kspace_multicoil.to(self.device), self.csm_p,
                                       ro, self.volume_shape, acs_lines, ifft3c)
        x2 = acs_init.acs_lowres_image(self.ds_c.kspace_multicoil.to(self.device), self.csm_c,
                                       ro, self.volume_shape, acs_lines, ifft3c)
        E_A = acs_init.joint_structural_map(x1, x2)
        pos, scl = acs_init.init_gaussians_from_structure(
            E_A, n, base_scale_vox=1.5, device=self.device, seed=self.cfg.get("seed", 42))
        acfg = mcfg.get("anatomy", {}); dcfg = mcfg.get("decoder", {})
        self.model = PairedARCModel(
            num_points=n, volume_shape=self.volume_shape, device=self.device,
            latent_dim=acfg.get("latent_dim", 12), n_contrasts=2,
            use_confidence=acfg.get("use_confidence", True),
            decoder_enable=dcfg.get("enable", True), decoder_hidden=dcfg.get("hidden_dim", 24),
            decoder_layers=dcfg.get("n_layers", 2), condition_dim=dcfg.get("condition_dim", 4),
            shared_trunk=dcfg.get("shared_trunk", True),
            initial_positions=pos, initial_scales=scl,
        ).to(self.device)
        print(f"[PairedARC] model init from ACS structure: {self.model.parameters_summary()}")

    def _build_modules(self):
        # later layers replace these NoOps with real Phase/Deformation modules
        self.phase = NoOpPhase().to(self.device)
        self.deform = NoOpDeformation().to(self.device)
        ccfg = self.cfg["model"].get("change", {})
        if ccfg.get("enable", False):
            from paired_arc_gs.modules.change_branch import ChangeBranch
            self.change = ChangeBranch(n_contrasts=2, volume_shape=self.volume_shape,
                                       device=self.device,
                                       max_per_contrast=int(ccfg.get("max_per_contrast", 20000)),
                                       init_scale_vox=float(ccfg.get("init_scale_vox", 1.5))).to(self.device)
            print(f"[PairedARC] change branch ENABLED (max {ccfg.get('max_per_contrast', 20000)}/contrast)")
        else:
            self.change = NoOpChange().to(self.device)

    def _build_voxelizer(self):
        vcfg = self.cfg.get("voxelizer", {})
        try:
            self.voxelizer = TileVoxelizer(volume_shape=self.volume_shape,
                                           tile_size=vcfg.get("tile_size", 8),
                                           max_radius=vcfg.get("max_radius", 20)).to(self.device)
        except Exception as e:
            print(f"[PairedARC] TileVoxelizer unavailable ({e}); using fallback Voxelizer")
            self.voxelizer = Voxelizer(volume_shape=self.volume_shape).to(self.device)

    def _build_optimizer(self):
        lr_cfg = self.cfg["training"].get("lr", {})
        groups = self.model.param_groups(lr_cfg)
        self._group_index: Dict[str, List[int]] = {}
        flat: List[Dict[str, Any]] = []
        for gname, glist in groups.items():
            self._group_index[gname] = []
            for g in glist:
                self._group_index[gname].append(len(flat)); flat.append(g)
        # module param groups (phase/deform/change) appended, indexed by their names
        for mod, nm in ((self.phase, "phase"), (self.deform, "deformation"), (self.change, "change")):
            pgs = mod.param_groups(lr_cfg.get(nm, 1e-3))
            if pgs:
                self._group_index.setdefault(nm, [])
                for g in pgs:
                    self._group_index[nm].append(len(flat)); flat.append(g)
        self.optimizer = torch.optim.Adam(flat)
        self._base_lrs = [g["lr"] for g in self.optimizer.param_groups]

    def _audit_split(self):
        """§10 fit/audit split of the SAMPLED k-space. Layer 1: audit disabled -> fit=all."""
        acfg = self.cfg["training"].get("audit", {})
        # audit split is REQUIRED by change routing (§10); auto-enable when change is on
        change_on = bool(self.cfg["model"].get("change", {}).get("enable", False))
        self.audit_enabled = bool(acfg.get("enable", False)) or change_on
        # Held-out TEST split (§18 eval): always reserve a fraction of ACQUIRED samples that
        # are NEVER fit and NEVER audited — used only for the GT-free held-out k-space NRMSE,
        # the primary metric for clinical data (e.g. CE) that has no fully-sampled reference.
        ecfg = self.cfg["training"].get("eval", {})
        test_frac = float(ecfg.get("holdout_fraction", 0.10))
        seed = int(acfg.get("seed", 42))
        gg = torch.Generator(device="cpu").manual_seed(seed)

        def three_way(mask):
            m = mask.detach().cpu()
            samp = (m > 0)
            r = torch.rand(m.shape, generator=gg)
            test = (samp & (r < test_frac))                       # held-out (never fit/audit)
            rest = (samp & (r >= test_frac))
            if self.audit_enabled:
                fit_frac = float(acfg.get("fit_fraction", 0.75))
                # split `rest` into fit/audit by a second uniform on the same r rescaled
                r2 = (r - test_frac) / max(1e-6, 1.0 - test_frac)
                fit = (rest & (r2 < fit_frac)).float().to(self.device)
                aud = (rest & (r2 >= fit_frac)).float().to(self.device)
            else:
                fit = rest.float().to(self.device)
                aud = None
            return fit, aud, test.float().to(self.device)

        self.mask_fit_p, self.mask_audit_p, self.mask_test_p = three_way(self.mask_p)
        self.mask_fit_c, self.mask_audit_c, self.mask_test_c = three_way(self.mask_c)

    # ------------------------------------------------------------------ rendering (§4)
    def render(self, contrast_index: int) -> torch.Tensor:
        """x_m = e^{jφ_m} [ V(T_m(G_A), D_m(z_A)) + change_m ]  (§4). Layer 1: φ, T, change
        are NoOps, so this reduces to the shared anatomy-response render."""
        pos, scl, rot = self.deform(contrast_index, self.model.positions,
                                    self.model.get_scale_values(), self.model.rotations)
        dens = self.model.get_density(contrast_index)
        vol = self.voxelizer(positions=pos, scales=scl, rotations=rot, density=dens)
        if isinstance(vol, tuple):
            vol = torch.complex(vol[0], vol[1])
        chg = self.change(contrast_index, self.voxelizer, self.volume_shape)
        if chg is not None:
            vol = vol + chg
        ph = self.phase(contrast_index, self.volume_shape)
        if ph is not None:
            vol = vol * ph
        return vol

    # ------------------------------------------------------------------ loss (§12)
    def compute_losses(self, active_terms: Tuple[str, ...]) -> Tuple[torch.Tensor, Dict[str, float]]:
        lcfg = self.cfg["loss"]
        x_p = self.render(0)
        x_c = self.render(1)
        terms: Dict[str, torch.Tensor] = {}
        # §12.1 noise-normalized DC (both contrasts, summed) on the FIT set
        dc_p = L.data_consistency(x_p, self.csm_p, self.ksp_p, self.mask_fit_p, fft3c, self.Linv_p)
        dc_c = L.data_consistency(x_c, self.csm_c, self.ksp_c, self.mask_fit_c, fft3c, self.Linv_c)
        terms["dc"] = dc_p + dc_c
        if "geo" in active_terms:
            terms["geo"] = L.geometry_reg(self.model.get_scale_values(), lcfg)
        if "phase" in active_terms and self.phase.enabled:
            pass  # phase reg added when phase module is real (Layer 3)
        weights = {k: lcfg.get(k) for k in ("lambda_geo", "lambda_change", "lambda_route",
                                            "lambda_phase", "lambda_def")}
        return L.total_loss(terms, active_terms, weights)

    # ------------------------------------------------------------------ scheduler-driven loop
    def _apply_stage_freezing(self, stage):
        """Set each optim group's LR: base LR if its group is in stage.opt_groups (scaled for
        geometry in S4), else 0 (frozen but Adam state preserved)."""
        live = set(stage.opt_groups)
        for gname, indices in self._group_index.items():
            for i in indices:
                base = self._base_lrs[i]
                if gname in live:
                    lr = base * (stage.geometry_lr_scale if gname == "geometry" else 1.0)
                else:
                    lr = 0.0
                self.optimizer.param_groups[i]["lr"] = lr

    # ------------------------------------------------------------------ §10 stage actions
    def _run_stage_actions(self, stage, history):
        acts = set(stage.actions)
        if "propose_change" in acts and "audit_accept" in acts and self.change.enabled:
            self._change_birth(history)
        if "prune" in acts and self.change.enabled:
            self._change_prune(history)

    @torch.no_grad()
    def _render_anatomy_only(self, ci: int) -> torch.Tensor:
        """x_m^A = V(T_m(G_A), D_m(z_A)) with the change branch OFF (for §10.2 evidence)."""
        pos, scl, rot = self.deform(ci, self.model.positions,
                                    self.model.get_scale_values(), self.model.rotations)
        vol = self.voxelizer(positions=pos, scales=scl, rotations=rot,
                             density=self.model.get_density(ci))
        if isinstance(vol, tuple):
            vol = torch.complex(vol[0], vol[1])
        ph = self.phase(ci, self.volume_shape)
        return vol * ph if ph is not None else vol

    def _change_birth(self, history):
        """§10.2 propose + §10.3 audit-accept change Gaussians, per contrast.

        For each contrast: render anatomy-only x_A, backproject the fit-set residual to an
        evidence map, propose candidate positions from the high-evidence tail, add them,
        briefly fit ONLY the change-branch density against the FIT set, then compute the
        audit gain ΔE on the disjoint AUDIT set. Accept the whole birth only if ΔE > τ_m;
        otherwise roll it back (anti-aliasing guard). Candidates that survive stay in the
        model and are refined jointly in later stages.
        """
        from paired_arc_gs.train import routing as Rt
        ccfg = self.cfg["model"].get("change", {})
        n_cand = int(ccfg.get("propose_per_contrast", 4000))
        # §10.3 acceptance threshold τ_m: prefer training.audit.tau, else model.change.audit_tau
        tau = float(self.cfg["training"].get("audit", {}).get(
            "tau", ccfg.get("audit_tau", 0.0)))
        fit_iters = int(ccfg.get("birth_fit_iters", 60))
        birth_lr = float(self.cfg["training"].get("lr", {}).get("change", 1e-2))
        report = {}
        for ci, (name, csm, ksp, mfit, maud) in enumerate([
            (self.name_p, self.csm_p, self.ksp_p, self.mask_fit_p, self.mask_audit_p),
            (self.name_c, self.csm_c, self.ksp_c, self.mask_fit_c, self.mask_audit_c)]):
            if maud is None:
                print(f"[birth] {name}: audit disabled -> skip (change routing needs audit set)")
                continue
            x_A = self._render_anatomy_only(ci)
            evid = Rt.fit_residual_evidence(x_A, csm, ksp, mfit, fft3c, ifft3c)
            cand = Rt.propose_from_evidence(evid, n_cand,
                                            frac_threshold=float(ccfg.get("evidence_quantile", 0.9)),
                                            seed=self.cfg.get("seed", 42))
            if cand.shape[0] == 0:
                print(f"[birth] {name}: no candidates proposed"); continue
            n_before = self.change.count(ci)
            e_aud_before = Rt.audit_error(self.render(ci), csm, ksp, maud, fft3c).item()
            self.change.add_gaussians(ci, cand.to(self.device))
            # briefly fit ONLY this contrast's change density against the FIT set
            new_params = [self.change.density[ci], self.change.positions[ci], self.change.scales[ci]]
            for p in new_params:
                p.requires_grad_(True)
            opt = torch.optim.Adam([{"params": new_params, "lr": birth_lr}])
            for _ in range(fit_iters):
                opt.zero_grad(set_to_none=True)
                x_plus = self.render(ci)
                res = fft3c(csm * x_plus.unsqueeze(0))
                m = mfit if mfit.ndim == res.ndim else mfit.unsqueeze(0)
                dc = (m * (res - ksp)).abs().pow(2).sum() / m.sum().clamp_min(1.0)
                spars = self.change.get_density(ci).abs().sum() * float(self.cfg["loss"].get("lambda_change", 1e-3))
                (dc + spars).backward()
                opt.step()
            e_aud_after = Rt.audit_error(self.render(ci), csm, ksp, maud, fft3c).item()
            dE = e_aud_before - e_aud_after
            # §10.3 acceptance; audit.bypass=True accepts unconditionally (the w/o-audit
            # ablation arm — the naive private branch §10 warns fits aliasing)
            bypass = bool(self.cfg["training"].get("audit", {}).get("bypass", False))
            accept = True if bypass else (dE > tau)
            if not accept:
                # roll back the whole birth (prune the newly-added block)
                keep = torch.zeros(self.change.count(ci), dtype=torch.bool, device=self.device)
                keep[:n_before] = True
                self.change.prune(ci, keep)
            report[name] = {"proposed": int(cand.shape[0]), "audit_gain": float(dE),
                            "tau": tau, "accepted": bool(accept),
                            "n_change_after": self.change.count(ci)}
            print(f"[birth] {name}: proposed={cand.shape[0]} ΔE_audit={dE:.4g} τ={tau} "
                  f"-> {'ACCEPT' if accept else 'REJECT'} (change count={self.change.count(ci)})")
        # change params changed -> rebuild optimizer so new params get moment buffers
        self._build_optimizer()
        if history is not None:
            history.append({"event": "change_birth", **{f"{k}_{kk}": vv for k, d in report.items() for kk, vv in d.items()}})
        self._last_birth_report = report

    def _change_prune(self, history):
        """§13-S7 prune change Gaussians with negligible amplitude."""
        thr = float(self.cfg["model"].get("change", {}).get("prune_amp", 1e-4))
        for ci in range(2):
            if self.change.count(ci) == 0:
                continue
            amp = self.change.get_density(ci).abs()
            keep = amp > (thr * amp.max().clamp_min(1e-12))
            n0 = self.change.count(ci)
            self.change.prune(ci, keep)
            print(f"[prune] contrast{ci}: {n0} -> {self.change.count(ci)} change Gaussians")
        self._build_optimizer()

    def train(self) -> Dict[str, Any]:
        total = self.plan.total_iters
        log_every = self.cfg["output"].get("eval_every", 500)
        history: List[Dict[str, float]] = []
        for it in range(total):
            stage = self.plan.at(it)
            if stage is None:
                break
            if stage.is_stage_start:
                print(f"[PairedARC] --- {stage.stage_id} (iters {stage.stage_iters}) "
                      f"opt={stage.opt_groups} loss={stage.loss_terms} act={stage.actions} ---")
                self._run_stage_actions(stage, history)
            self._apply_stage_freezing(stage)
            self.optimizer.zero_grad(set_to_none=True)
            loss, log = self.compute_losses(stage.loss_terms)
            loss.backward()
            self.optimizer.step()
            if it % log_every == 0 or stage.is_stage_end:
                log["iter"] = it; log["stage"] = stage.stage_id
                history.append(log)
                print(f"[PairedARC] it{it} {stage.stage_id} " +
                      " ".join(f"{k}={v:.4g}" for k, v in log.items() if isinstance(v, float)))
        return {"history": history, "final_iter": total}

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def holdout_kspace_nrmse(self) -> Dict[str, Dict[str, float]]:
        """GT-FREE metric (§18): held-out k-space NRMSE.

            NRMSE_ho = || M_test (F(S x) - y) || / || M_test y ||    (per contrast)

        M_test is the reserved held-out sampling set (never fit, never audited), so this
        cannot be gamed by hard data-consistency replacement. A null model scores ~1.0; a
        perfect model scores 0. This is the PRIMARY metric for clinical data (CE) that has
        no fully-sampled reference — and a valid consistency check on T1/FLAIR too.
        """
        out = {}
        pairs = [(0, self.name_p, self.csm_p, self.ksp_p, self.mask_test_p),
                 (1, self.name_c, self.csm_c, self.ksp_c, self.mask_test_c)]
        for ci, name, csm, ksp, mtest in pairs:
            x = self.render(ci)
            k_model = fft3c(csm * x.unsqueeze(0))
            m = mtest if mtest.ndim == k_model.ndim else mtest.unsqueeze(0)
            num = (m * (k_model - ksp)).abs().pow(2).sum().sqrt()
            den = (m * ksp).abs().pow(2).sum().sqrt().clamp_min(1e-12)
            out[["plain", "contrast"][ci]] = {"name": name, "holdout_nrmse": float(num / den)}
        return out

    @torch.no_grad()
    def evaluate(self, ref_ok: bool = True) -> Dict[str, Dict[str, float]]:
        """Alpha-fit PSNR/SSIM vs GT for both contrasts (matches the MC-GS scoring口径).

        VALID only when GT is a fully-sampled reference (T1/FLAIR). For clinical data whose
        'GT' is the zero-filled recon of the acquired data (CE), full-reference PSNR is
        circular (rewards a null model) — use holdout_kspace_nrmse() instead. `ref_ok=False`
        flags the returned scores as reference-invalid.
        """
        from skimage.metrics import structural_similarity as sk_ssim
        import numpy as np
        out = {}
        for ci, (name, gt, csm, ksp, mask) in enumerate([
            (self.name_p, self.gt_p, self.csm_p, self.ksp_p, self.mask_p),
            (self.name_c, self.gt_c, self.csm_c, self.ksp_c, self.mask_c)]):
            x = self.render(ci)
            # hard DC + SENSE combine for a fair score against GT
            k_model = fft3c(csm * x.unsqueeze(0))
            m = mask if mask.ndim == k_model.ndim else mask.unsqueeze(0)
            k_dc = m * ksp + (1 - m) * k_model
            x_dc = (ifft3c(k_dc) * csm.conj()).sum(0)
            rec = x_dc.abs().cpu().numpy().astype("float64")
            g = gt.abs().cpu().numpy().astype("float64")
            alpha = float((rec * g).sum() / (rec * rec).sum())
            r = rec * alpha; mx = float(g.max())
            psnr = 10.0 * np.log10(mx ** 2 / np.mean((r - g) ** 2))
            ssim = sk_ssim(g, r, data_range=mx)
            out[["plain", "contrast"][ci]] = {"name": name, "psnr": float(psnr),
                                              "ssim": float(ssim), "ref_valid": bool(ref_ok)}
        return out
