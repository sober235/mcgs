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
        # later layers replace these NoOps with real Phase/Deformation/Change modules
        self.phase = NoOpPhase().to(self.device)
        self.deform = NoOpDeformation().to(self.device)
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
        self.audit_enabled = bool(acfg.get("enable", False))
        if not self.audit_enabled:
            self.mask_fit_p, self.mask_fit_c = self.mask_p, self.mask_c
            self.mask_audit_p = self.mask_audit_c = None
            return
        frac = float(acfg.get("fit_fraction", 0.75)); seed = int(acfg.get("seed", 42))
        g = torch.Generator(device="cpu").manual_seed(seed)
        def split(mask):
            m = mask.detach().cpu()
            samp = (m > 0)
            r = torch.rand(m.shape, generator=g)
            fit = (samp & (r < frac)).float().to(self.device)
            aud = (samp & (r >= frac)).float().to(self.device)
            return fit, aud
        self.mask_fit_p, self.mask_audit_p = split(self.mask_p)
        self.mask_fit_c, self.mask_audit_c = split(self.mask_c)

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
                      f"opt={stage.opt_groups} loss={stage.loss_terms} ---")
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
    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """Alpha-fit PSNR/SSIM vs GT for both contrasts (matches the MC-GS scoring口径)."""
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
            out[["plain", "contrast"][ci]] = {"name": name, "psnr": float(psnr), "ssim": float(ssim)}
        return out
