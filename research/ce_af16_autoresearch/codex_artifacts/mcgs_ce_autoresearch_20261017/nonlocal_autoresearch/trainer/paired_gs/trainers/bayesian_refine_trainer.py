"""BayesianRefineTrainer (Phase 12 CF-11).

Subclass of :class:`BayesianJointTrainer` that consumes a 3-source warmstart and
refines the posterior under risk-aware counterfactual signals:

  - Phase 8 ELBO checkpoint  (model_state + m_0 prior snapshot)
  - Phase 10 ``audit_results/<run_id>/``  (H_p, H_c, S_cross, U_*, R_* npy)
  - Phase 11 ``cf_results/<run_id>/``       (C_final, H_bar_cf_{plain,contrast} npy)

Five method customizations (RESEARCH §A inheritance shape):
  1. ``__init__``                  — pre-super densify-OFF override (layer 1) +
                                      pre-super warmstart load (so ``_setup_loss``
                                      can wire cf modules without ordering bugs,
                                      RESEARCH Pitfall #4) + post-super guards
                                      (densify-OFF assertion + plateau-off).
  2. ``_load_warmstart_artifacts``  — 3-path resolver + sha256 cross-link
                                      (fail-fast on mismatch, warn-then-continue
                                      on missing field per W-05) + B-2 CF-03/CF-07
                                      remediation-provenance gate.
  3. ``_setup_loss`` (override)      — ``super()._setup_loss()`` builds ElboLoss;
                                      then attach CounterfactualSafeLoss +
                                      CounterfactualDampingLoss(C_final only;
                                      CF-09 reverted 2026-05-29 — no S_cross).
  4. ``_setup_optimizer`` (override) — pre-multiply lr/3 on all 4 param groups
                                      (single source of truth so the Phase 9
                                      fairness fingerprint reports the refined lr).
  5. ``train`` (full override)        — bare loop mirroring the parent shape
                                      WITHOUT MetricPlateauStopper; hard-cap
                                      early-stop at ``max_iter // 4``; densify
                                      runtime assert. Plan 12-02 extends the loop
                                      with the per-iter cf_safe / cf_damp calls.

Densification OFF (3 paranoia layers); lr/3 (pre-multiply); early-stop hard cap
at ``max_iter // 4`` (no MetricPlateauStopper); plateau-eval block skipped
entirely by overriding ``train()`` — do NOT call ``super().train()``.

sha256 cross-link verification (CONTEXT decision B fail-fast):
  - elbo_ckpt content sha    == audit_summary.ckpt_sha256
  - audit_summary content sha == cf_summary.audit_summary_sha256
  Missing field → RuntimeWarning + continue (W-05 honest legacy support).

B-2 — Production-vs-phantom gating (CF-03/CF-07 remediation precondition):
  cf_results that predate the Phase 11 CF-03/CF-07 remediation rerun
  (``.planning/phases/11-counterfactual-attribution/11-CF03-CF07-REMEDIATION.md``)
  carry a rendered-magnitude proxy ΔH (CF-03) and a C^final reflecting it (CF-07
  pre-rerun). Synthetic phantom pytest is value-agnostic and SAFE; PRODUCTION
  refinement on real ``cf_results/*`` is BLOCKED until the remediation lands.
  This trainer RECORDS that dependency (it does NOT execute the remediation):
  the gate reads ``cf_summary.cf03_cf07_remediated`` (written in Plan 12-00) and
  warns by default / raises when ``require_remediated_cf=True`` (production).

References
----------
- .planning/phases/12-risk-aware-refinement/12-CONTEXT.md decisions A/B/C
- .planning/phases/12-risk-aware-refinement/12-RESEARCH.md §A inheritance shape
- .planning/phases/12-risk-aware-refinement/12-VALIDATION.md CF-11
- .planning/research/PITFALLS.md C-04/C-05/C-06 prevention
- PDF §1.12 (refinement objective formula)
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from trainers.bayesian_joint_trainer import BayesianJointTrainer


__all__ = ["BayesianRefineTrainer"]


def _sha256_file(path: Path) -> str:
    """Content-addressed file hash for warmstart cross-link verification."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class BayesianRefineTrainer(BayesianJointTrainer):
    """Refinement trainer; densify OFF; lr/3; early-stop ≤ ``max_iter // 4``.

    NOT exported from ``trainers/__init__.py`` (mirrors the BayesianJointTrainer
    convention; consumers import via the direct module path).
    """

    def __init__(self, config: Dict[str, Any], device: torch.device | None = None):
        # ---- LAYER 1: densify-OFF yaml override (CONTEXT B) ----
        # Forced BEFORE super().__init__ which would otherwise honor a truthy
        # yaml ``adaptive_control.enable``.
        if "adaptive_control" not in config:
            config["adaptive_control"] = {}
        config["adaptive_control"]["enable"] = False

        # Route the Phase 8 ELBO ckpt through the parent warm-start path so
        # ``_setup_model`` initializes the posterior from it (avoid a yaml fork).
        warmstart_cfg = config["warmstart"]
        config.setdefault("bayesian", {}).setdefault("warm_start", {})
        config["bayesian"]["warm_start"]["ckpt_path"] = warmstart_cfg["elbo_ckpt"]

        # B-2: thread the CF-03/CF-07 remediation gate into the warmstart cfg so
        # the loader can warn (synthetic pytest) or fail-fast (production) on
        # pre-rerun cf_results. Production yaml/dispatcher sets
        # ``require_remediated_cf: true``; test fixtures leave it false.
        warmstart_cfg["require_remediated_cf"] = bool(
            config.get("require_remediated_cf", False)
        )

        # ---- PRE-SUPER: load warmstart artifacts FIRST ----
        # So the _setup_loss override can wire cf modules without ordering bugs
        # (RESEARCH Pitfall #4). The loader ALSO enforces the B-2 provenance gate.
        # ``self.config`` is not assigned until super().__init__ runs — the
        # loader is a pure function of its argument.
        self._warmstart = self._load_warmstart_artifacts(warmstart_cfg)

        # super() runs _validate_config → _setup_seeds → _setup_data
        # → _setup_model (reads bayesian.warm_start.ckpt_path) → _setup_loss
        # (OURS) → _setup_optimizer (OURS) → _setup_output
        # → _setup_collapse_monitor_state → fairness fingerprint dump.
        super().__init__(config, device)

        # ---- LAYER 2: post-init densify-OFF assertion ----
        assert self.config["adaptive_control"]["enable"] is False, (
            "BayesianRefineTrainer: densify must be OFF (CONTEXT B); "
            "post-init guard failed."
        )

        # ---- LAYER 3: plateau stopper disabled (PITFALL #5) ----
        # The parent instantiates ``_stopper`` inside train(), not __init__; our
        # train() override never instantiates it. Set None defensively here for
        # any introspection between construction and train().
        self._stopper = None

    # ------------------------------------------------------------------ #
    # Warmstart loader                                                    #
    # ------------------------------------------------------------------ #
    def _load_warmstart_artifacts(
        self, warmstart_cfg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve 3 paths + verify sha256 cross-link + load npy/json.

        Pure function of ``warmstart_cfg`` — does NOT touch ``self.config``
        (which is not assigned until ``super().__init__`` runs).
        """
        elbo_ckpt = Path(warmstart_cfg["elbo_ckpt"]).resolve()
        audit_dir = Path(warmstart_cfg["audit_dir"]).resolve()
        cf_dir = Path(warmstart_cfg["cf_dir"]).resolve()

        for path, slot in [
            (elbo_ckpt, "warmstart.elbo_ckpt"),
            (audit_dir, "warmstart.audit_dir"),
            (cf_dir, "warmstart.cf_dir"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{slot} not found: {path!r}")

        # ---- JSON summaries ----
        with open(audit_dir / "audit_summary.json") as fh:
            audit_summary = json.load(fh)
        with open(cf_dir / "cf_summary.json") as fh:
            cf_summary = json.load(fh)

        # ---- sha256 3-way cross-link ----
        ckpt_sha = _sha256_file(elbo_ckpt)
        audit_summary_sha = _sha256_file(audit_dir / "audit_summary.json")

        audit_recorded_ckpt_sha = audit_summary.get("ckpt_sha256")
        cf_recorded_audit_sha = cf_summary.get("audit_summary_sha256")

        if audit_recorded_ckpt_sha is None:
            warnings.warn(
                "audit_summary.json lacks ckpt_sha256 field — Plan 12-00 Phase "
                "10 patch not applied; skipping ckpt sha cross-link verification "
                "(W-05 honest legacy support).",
                RuntimeWarning,
                stacklevel=2,
            )
        elif ckpt_sha != audit_recorded_ckpt_sha:
            raise ValueError(
                f"warmstart sha mismatch: elbo_ckpt content sha={ckpt_sha} "
                f"!= audit_summary.ckpt_sha256={audit_recorded_ckpt_sha}; "
                f"refinement aborted (CONTEXT B fail-fast)."
            )

        if cf_recorded_audit_sha is None:
            warnings.warn(
                "cf_summary.json lacks audit_summary_sha256 field — Plan 12-00 "
                "Phase 11 patch not applied; skipping audit sha cross-link "
                "verification (W-05).",
                RuntimeWarning,
                stacklevel=2,
            )
        elif audit_summary_sha != cf_recorded_audit_sha:
            raise ValueError(
                f"warmstart sha mismatch: audit_summary content sha="
                f"{audit_summary_sha} != cf_summary.audit_summary_sha256="
                f"{cf_recorded_audit_sha}; refinement aborted (CONTEXT B fail-fast)."
            )

        # ---------------------------------------------------------- #
        # B-2 CF-03/CF-07 REMEDIATION PROVENANCE GATE                #
        # Records the dependency; does NOT execute the remediation.  #
        # cf_results predating the 11-CF03-CF07-REMEDIATION.md rerun #
        # carry proxy ΔH (CF-03 drift) + proxy-derived C^final, so   #
        # PRODUCTION refinement on them silently corrupts the head-  #
        # line NMSE/lesion claims. Phantom pytest builds its own     #
        # synthetic C^final and is value-agnostic → SAFE; production #
        # runs set require_remediated_cf=True.                       #
        # ---------------------------------------------------------- #
        cf_remediated = bool(cf_summary.get("cf03_cf07_remediated", False))
        require_remediated = bool(warmstart_cfg.get("require_remediated_cf", False))
        if not cf_remediated:
            msg = (
                f"cf_dir {cf_dir} carries cf03_cf07_remediated="
                f"{cf_summary.get('cf03_cf07_remediated')!r} (proxy ΔH / "
                f"pre-rerun C^final per 11-CF03-CF07-REMEDIATION.md). PRODUCTION "
                f"refinement on this attribution is BLOCKED; phantom pytest "
                f"(synthetic C^final) is SAFE."
            )
            if require_remediated:
                raise ValueError(
                    msg + " (require_remediated_cf=True → fail-fast; run the "
                    "Phase 11 CF-03/CF-07 remediation rerun first.)"
                )
            warnings.warn(
                msg + " (require_remediated_cf=False → warn-only; acceptable for "
                "synthetic-fixture pytest only.)",
                RuntimeWarning,
                stacklevel=2,
            )

        # ---- npy arrays (value-agnostic; shapes verified by tests) ----
        H_p = np.load(audit_dir / "H_p.npy")
        H_c = np.load(audit_dir / "H_c.npy")
        S_cross = np.load(audit_dir / "S_cross.npy")
        C_final = np.load(cf_dir / "C_final.npy")
        H_bar_cf_p = np.load(cf_dir / "H_bar_cf_plain.npy")
        H_bar_cf_c = np.load(cf_dir / "H_bar_cf_contrast.npy")

        return {
            "elbo_ckpt": elbo_ckpt,
            "audit_dir": audit_dir,
            "cf_dir": cf_dir,
            "H_p": H_p, "H_c": H_c, "S_cross": S_cross,
            "C_final": C_final,
            "H_bar_cf_p": H_bar_cf_p, "H_bar_cf_c": H_bar_cf_c,
            "audit_summary": audit_summary,
            "cf_summary": cf_summary,
            "cf03_cf07_remediated": cf_remediated,
            "elbo_ckpt_sha256": ckpt_sha,
            "audit_summary_sha256": audit_summary_sha,
        }

    # ------------------------------------------------------------------ #
    # _setup_loss override — attach cf modules after parent ElboLoss build #
    # ------------------------------------------------------------------ #
    def _setup_loss(self):
        """Build ElboLoss (parent) then attach the two cf modules.

        Plan 12-02 wires the per-iter cf_safe / cf_damp calls into the loop; this
        plan only constructs the modules + persists the locked λ values so the
        train() loop has a fully-wired entry point.
        """
        # 1. Parent builds self.elbo_loss = ElboLoss(...) — unchanged.
        super()._setup_loss()

        # 2. Lazy import to avoid any import-order surprises at module load.
        from losses.cf_safe import CounterfactualSafeLoss
        from losses.cf_damp import CounterfactualDampingLoss

        warm = self._warmstart

        # 3. CounterfactualSafeLoss — H̄^cf maps registered as buffers (detached
        #    inside the module). tau from yaml (Landweber relaxation).
        self.cf_safe_loss = CounterfactualSafeLoss(
            H_bar_cf_p=torch.from_numpy(warm["H_bar_cf_p"]).to(self.device),
            H_bar_cf_c=torch.from_numpy(warm["H_bar_cf_c"]).to(self.device),
            tau=float(self.config.get("cf_safe", {}).get("tau", 1.0)),
        ).to(self.device)

        # 4. CounterfactualDampingLoss — C^final (N, 2) registered as buffer.
        #    PDF §1.12: bare C^final·|m|², NO S_cross factor (CF-09 reverted
        #    2026-05-29). Lesion protection is carried by C^final itself via
        #    CF-04's DC-safeguard — so NO S_cross trilinear sampling is needed.
        C_final = torch.from_numpy(warm["C_final"]).to(self.device)  # (N, 2)
        self.cf_damp_loss = CounterfactualDampingLoss(
            C_final=C_final,
        ).to(self.device)

        # 5. Persist locked λ values (single source for the Plan 12-02 wire-up).
        self.lambda_safe = float(self.config.get("cf_safe", {}).get("lambda", 1e-2))
        self.lambda_cf = float(self.config.get("cf_damp", {}).get("lambda", 1e-2))

    # ------------------------------------------------------------------ #
    # _setup_optimizer override — lr/3 pre-multiply                       #
    # ------------------------------------------------------------------ #
    def _setup_optimizer(self):
        """Pre-multiply lr_position/density/scale/rotation by 1/3.

        Single source of truth: the /3 is applied BEFORE
        ``get_optimizable_params`` so the eta_* cascade (lr_density * 0.1)
        inherits the refined lr automatically AND the Phase 9 fairness
        fingerprint (dumped after this method) reports the refined lr.
        """
        gcfg = self.config["gaussian"]
        params = self.model.get_optimizable_params(
            lr_position=gcfg.get("position_lr", 1e-4) / 3.0,
            lr_density=gcfg.get("density_lr", 1e-3) / 3.0,
            lr_scale=gcfg.get("scale_lr", 5e-4) / 3.0,
            lr_rotation=gcfg.get("rotation_lr", 1e-4) / 3.0,
        )
        self.optimizer = torch.optim.Adam(params)

        sched_cfg = self.config.get("training", {}).get("lr_scheduler", {})
        if sched_cfg.get("type") == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=sched_cfg.get("gamma", 0.9995),
            )
        else:
            self.scheduler = None

    # ------------------------------------------------------------------ #
    # train override — no plateau; hard-cap early-stop                    #
    # ------------------------------------------------------------------ #
    def train(self):
        """Refinement training loop (full override — do NOT call super().train()).

        Key differences from BayesianJointTrainer.train():
        - NO MetricPlateauStopper instantiation (PITFALL #5 prevention); the
          plateau-eval block is dropped entirely.
        - Hard early-stop at ``max_iter // 4`` (CONTEXT B; CF-11 SC#3).
        - Per-iter densify runtime assert (LAYER 3 paranoia).
        - DC term uses the tr∪fb union mask (CF-10; PDF §1.12 L_dc^{tr∪fb}).
        - cf_safe / cf_damp loss terms are wired in Plan 12-02; THIS loop runs
          ONLY the parent ELBO terms (DC + KLs + R_geo + R_sp). The cf modules
          (self.cf_safe_loss / self.cf_damp_loss / self.lambda_*) are already
          constructed so Plan 12-02 only extends the ElboLoss kwargs + TB group.
        """
        from torch.utils.tensorboard import SummaryWriter

        train_cfg = self.config.get("training", {})
        bcfg = self.config.get("bayesian", {})
        max_iter = int(train_cfg.get("max_iterations", 16000))
        early_stop_max_iter = max_iter // 4   # CONTEXT B hard cap
        K_train = int(bcfg.get("K_train", 1))
        max_grad_norm = float(train_cfg.get("max_grad_norm",
                                            train_cfg.get("grad_clip", 1.0)))
        log_every = int(train_cfg.get("log_every", 50))
        save_every = int(train_cfg.get("save_every", 500))

        # LAYER 3: explicit plateau disable — never instantiate the stopper.
        self._stopper = None

        writer = SummaryWriter(log_dir=str(self.out_dir / "logs")) \
            if self.config.get("output", {}).get("use_tensorboard", True) \
            else None

        print(
            f"[refine] starting refinement: max_iter={max_iter}, "
            f"early_stop_max_iter={early_stop_max_iter} (= max_iter // 4), "
            f"lr_position={self.optimizer.param_groups[0]['lr']:.3e}, "
            f"densify={self.config['adaptive_control']['enable']}"
        )

        generator = torch.Generator(device=self.device).manual_seed(
            int(train_cfg.get("seed", 42))
        )

        # Pre-refinement metrics (computed ONCE before the loop, cached for
        # export_results). Production HYW has no GT so this stays {} unless a
        # subclass (e.g. the e2e phantom test) overrides _compute_pre_metrics.
        self._pre_refinement_metrics = self._compute_pre_refinement_metrics()

        # cf ratio running-average accumulators (refinement_summary.json).
        self._cf_safe_ratio_sum = 0.0
        self._cf_damp_ratio_sum = 0.0
        self._cf_log_count = 0
        self._actual_early_stop_iter = 0

        t0 = time.time()
        last_it = 0
        for it in range(1, max_iter + 1):
            # LAYER 3 runtime assertion (densify OFF, every iter).
            assert self.config["adaptive_control"]["enable"] is False, (
                f"[refine iter {it}] densify guard violated"
            )

            # Hard-cap early-stop (CONTEXT B; CF-11 SC#3).
            if it > early_stop_max_iter:
                print(
                    f"[refine] early-stop hit at iter {it} "
                    f"(cap = max_iter // 4 = {early_stop_max_iter})"
                )
                break
            last_it = it
            self._actual_early_stop_iter = it

            self.optimizer.zero_grad()

            # MC sample posterior (K_train samples) + DC term over tr∪fb (CF-10).
            samples = self.model.sample(K=K_train, generator=generator)
            (
                l_dc_mc, vol_p_for_geo, vol_c_for_geo, vol_p_grad, vol_c_grad
            ) = self._render_and_dc_terms(samples, K_train)

            # ---- Phase 12-02 cf loss terms (Plan 12-00 modules) ----
            # vol_p_grad / vol_c_grad are (D, H, W) complex tensors WITH grad
            # path (cf_safe needs the live reconstruction; the detached MC-mean
            # vol_*_for_geo would block the cf_safe grad). PDF §1.12: x^dc uses
            # the full training-available operator A_c == tr∪fb (M^te stays
            # isolated per EA-07). FIX (CF-08): pass the union, NOT fb-only.
            mask_trfb_p = self._union_mask(self.mask_tr_p, self.mask_fb_p)
            mask_trfb_c = self._union_mask(self.mask_tr_c, self.mask_fb_c)
            l_cf_safe = self.cf_safe_loss(
                vol_p_grad, vol_c_grad,
                self.csm_p, self.csm_c,
                self.kspace_p, self.kspace_c,
                mask_trfb_p, mask_trfb_c,
            )
            l_cf_damp = self.cf_damp_loss(self.model)

            # ELBO total + ratios (parent ElboLoss) extended with the cf terms
            # (INF-05 8-ratio schema, gated by the kwargs). β_a warmup BP-13.
            beta_a, beta_mu, beta_ell = self._beta_warmup_schedule(it)
            elbo_out = self.elbo_loss(
                l_dc_mc=l_dc_mc,
                model=self.model,
                m_0=self.m_0,
                vol_plain_for_geo=vol_p_for_geo,
                vol_contrast_for_geo=vol_c_for_geo,
                beta_a_override=beta_a,
                beta_mu_override=beta_mu,
                beta_ell_override=beta_ell,
                cf_safe_value=l_cf_safe,
                cf_damp_value=l_cf_damp,
                lambda_safe=self.lambda_safe,
                lambda_cf=self.lambda_cf,
            )

            total = elbo_out["total"]
            # INF-05 ratio sanity (CONTEXT decision A locked tolerance 1e-6).
            # Now an 8-key dict (adds cf/safe + cf/damp); the sum-to-1.0 identity
            # is preserved by ElboLoss's comp/(sum+eps) normalization.
            ratios = elbo_out["ratios"]
            sum_ratios = sum(ratios.values())
            assert abs(sum_ratios - 1.0) < 1e-6, (
                f"[refine iter {it}] INF-05 ratio sum != 1.0 (locked 1e-6); got "
                f"{sum_ratios:.8f}"
            )
            # Running averages for refinement_summary.json (cf ratio means).
            self._cf_safe_ratio_sum += float(ratios.get("cf/safe", 0.0))
            self._cf_damp_ratio_sum += float(ratios.get("cf/damp", 0.0))
            self._cf_log_count += 1

            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # TB scalars — the values/ratios dicts already carry the 8 keys
            # (loss/* + kl/* + cf/safe + cf/damp), each emitted as "<k>/value"
            # and "<k>/ratio" (so e.g. cf/safe -> "cf/safe/value" + "cf/safe/
            # ratio"). CF-10 TB grouping requirement satisfied without a
            # separate cf branch.
            if writer is not None and (it % log_every == 0 or it == 1):
                for k_, v_ in elbo_out["values"].items():
                    writer.add_scalar(f"{k_}/value", v_, it)
                for k_, r_ in ratios.items():
                    writer.add_scalar(f"{k_}/ratio", r_, it)
                writer.add_scalar("loss/total", float(total.detach().item()), it)
                writer.add_scalar("kl_schedule/beta_a", beta_a, it)

            if it % save_every == 0:
                self._save_checkpoint(tag="latest")

        # Final ckpt dump (best/latest convention).
        self._save_checkpoint(tag="latest")
        if writer is not None:
            writer.close()
        print(
            f"[refine] training complete in {time.time() - t0:.1f}s; "
            f"last iter = {last_it} (cap = {early_stop_max_iter})"
        )

    # ------------------------------------------------------------------ #
    # Helpers — reuse parent render + DC arithmetic byte-identically      #
    # (the ONLY deliberate change is the CF-10 tr∪fb union mask)          #
    # ------------------------------------------------------------------ #
    def _render_and_dc_terms(self, samples, K_train: int):
        """Render plain + contrast MC volumes and accumulate the DC loss.

        Mirrors BayesianJointTrainer.train()'s per-sample render + DC block
        (bayesian_joint_trainer.py L511-524) verbatim, EXCEPT the DC term goes
        through ``_kspace_dc_loss_helper`` which uses the tr∪fb union mask
        (CF-10) instead of the parent's mask_tr-only DC.

        Returns ``(l_dc_mc, vol_p_for_geo, vol_c_for_geo, vol_p_grad,
        vol_c_grad)``:
          - ``l_dc_mc``            : MC-averaged DC scalar (grad-attached).
          - ``vol_p_for_geo`` / ``vol_c_for_geo`` : detached MC-mean volumes for
            the R_geo TV term (parent contract; passed to ElboLoss).
          - ``vol_p_grad`` / ``vol_c_grad`` : the LAST MC sample's GRAD-ATTACHED
            volume (CF-08 — cf_safe_loss needs a grad path through the live
            reconstruction; the detached MC-mean would block the cf_safe grad).
            With the locked ``K_train=1`` these equal the single rendered sample.
        """
        l_dc_total = torch.zeros((), device=self.device)
        vol_p_mean_acc = torch.zeros_like(self.zf_p) if self.zf_p is not None else None
        vol_c_mean_acc = torch.zeros_like(self.zf_c) if self.zf_c is not None else None
        vol_p_grad = vol_c_grad = None

        for k in range(K_train):
            vol_p_k = vol_c_k = None
            if getattr(self.model, "has_plain", True):
                vol_p_k = self._render_sample(samples, k, "plain")
            if getattr(self.model, "has_contrast", True):
                vol_c_k = self._render_sample(samples, k, "contrast")
            l_dc_total = l_dc_total + self._kspace_dc_loss_helper(vol_p_k, vol_c_k)
            # Keep the last sample's grad-attached volume for cf_safe (CF-08).
            vol_p_grad = vol_p_k if vol_p_k is not None else vol_p_grad
            vol_c_grad = vol_c_k if vol_c_k is not None else vol_c_grad
            if vol_p_k is not None and vol_p_mean_acc is not None:
                vol_p_mean_acc = vol_p_mean_acc + vol_p_k.detach()
            if vol_c_k is not None and vol_c_mean_acc is not None:
                vol_c_mean_acc = vol_c_mean_acc + vol_c_k.detach()

        l_dc_mc = l_dc_total / float(K_train)
        vol_p_for_geo = (
            vol_p_mean_acc / float(K_train) if vol_p_mean_acc is not None else None
        )
        vol_c_for_geo = (
            vol_c_mean_acc / float(K_train) if vol_c_mean_acc is not None else None
        )
        return l_dc_mc, vol_p_for_geo, vol_c_for_geo, vol_p_grad, vol_c_grad

    def _union_mask(self, mask_tr, mask_fb):
        """Build the M^tr ∪ M^fb union mask (CF-10).

        Masks are stored as 0/1 float tensors (data/dataset.py three-way split).
        Union = clamp(tr + fb, max=1). When ``mask_fb`` is None (no three-way
        split, e.g. single-channel fallback), the union degenerates to mask_tr.
        """
        if mask_fb is None:
            return mask_tr
        return (mask_tr + mask_fb).clamp(max=1.0)

    def _kspace_dc_loss_helper(self, vol_p, vol_c):
        """Centered-FFT DC loss on M^tr ∪ M^fb — PDF §1.12 L_dc^{tr∪fb}.

        FIX (CF-10): the refinement DC term MUST fold M^fb back in. We build the
        tr∪fb union mask and pass it where the parent passes mask_tr; the
        centered-FFT DC arithmetic is reused verbatim from the parent
        ``_kspace_dc_loss`` (no FFT-math change — ONLY the mask argument differs).
        """
        total = torch.zeros((), device=self.device)
        if vol_p is not None:
            mask_trfb_p = self._union_mask(self.mask_tr_p, self.mask_fb_p)
            total = total + self._kspace_dc_loss(
                vol_p, self.csm_p, self.kspace_p, mask_trfb_p
            )
        if vol_c is not None:
            mask_trfb_c = self._union_mask(self.mask_tr_c, self.mask_fb_c)
            total = total + self._kspace_dc_loss(
                vol_c, self.csm_c, self.kspace_c, mask_trfb_c
            )
        return total

    def _beta_warmup_schedule(self, it: int):
        """β_a linear warmup 0→target over warmup_kl iter (parent BP-13).

        Returns ``(beta_a, beta_mu, beta_ell)`` mirroring the parent's
        ``_beta_a_schedule`` + the beta_mu/beta_ell ratio cascade.
        """
        bcfg = self.config["bayesian"]
        warmup_kl = int(bcfg.get("warmup_kl", 1000))
        beta_a_target = float(bcfg.get("beta_a", 1.0))
        beta_mu_ratio = float(bcfg.get("beta_mu_ratio", 0.1))
        beta_ell_ratio = float(bcfg.get("beta_ell_ratio", 0.1))
        beta_a = self._beta_a_schedule(it, warmup_kl, beta_a_target)
        return beta_a, beta_a * beta_mu_ratio, beta_a * beta_ell_ratio

    def _save_checkpoint(self, tag: str = "latest"):
        """Save model_state to ``<out_dir>/checkpoints/<tag>.pth``.

        Records the warmstart sha cross-link so Phase 13 calibration can verify
        the refined ckpt's provenance back to the Phase 8/10/11 chain.
        """
        ckpt_dir = self.out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optim_state": self.optimizer.state_dict(),
                "m_0": self.m_0,
                "warmstart_elbo_ckpt_sha256": self._warmstart["elbo_ckpt_sha256"],
                "warmstart_audit_summary_sha256": self._warmstart["audit_summary_sha256"],
                "cf03_cf07_remediated": self._warmstart["cf03_cf07_remediated"],
                "config": self.config,
            },
            str(ckpt_dir / f"{tag}.pth"),
        )

    # ------------------------------------------------------------------ #
    # Plan 12-02 — pre/post metrics + run_id + refinement_summary.json    #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _voxelize_posterior_mean(self, channel: str) -> Optional[torch.Tensor]:
        """Render the posterior-MEAN reconstruction for ``channel`` (no MC noise).

        Used for pre/post NMSE/PSNR/SSIM. Returns ``None`` when the channel is
        inactive (single-channel Method A). The posterior mean is the model's
        ``density_*`` parameters directly (m = posterior mean per 08-01).
        """
        has = getattr(self.model, f"has_{channel}", True)
        if not has:
            return None
        if channel == "plain":
            density = self.model.density_plain
        else:
            density = torch.complex(
                self.model.density_real_contrast, self.model.density_imag_contrast
            )
        vol = self.voxelizer(
            positions=self.model.positions,
            scales=self.model.scales.exp(),
            rotations=self.model.rotations,
            density=density,
        )
        if isinstance(vol, tuple):
            vol = torch.complex(vol[0], vol[1])
        return vol

    @torch.no_grad()
    def _reference_metrics(self) -> Dict[str, Any]:
        """NMSE/PSNR/SSIM of the posterior-mean recon vs the ZF baseline.

        Production HYW has no paired GT, so the reference is the zero-filled
        reconstruction (``self.zf_*``) — a self-consistency signal, NOT a GT
        claim (the headline GT/lesion validation is the phantom e2e test + the
        Phase 13 real-data deferral per CONTEXT <deferred>). A subclass with a
        real GT volume can override this.
        """
        from metrics.metrics import compute_nmse, compute_psnr, compute_ssim

        out: Dict[str, Any] = {}
        for ch, zf in (("plain", self.zf_p), ("contrast", self.zf_c)):
            vol = self._voxelize_posterior_mean(ch)
            if vol is None or zf is None:
                continue
            pred = vol.abs()
            ref = zf.abs()
            out[ch] = {
                "nmse": compute_nmse(pred, ref),
                "psnr": compute_psnr(pred, ref),
                "ssim": compute_ssim(pred, ref),
            }
        return out

    def _compute_pre_refinement_metrics(self) -> Dict[str, Any]:
        """Cache the pre-refinement reference metrics (called once before loop)."""
        return self._reference_metrics()

    def _compute_post_refinement_metrics(self) -> Dict[str, Any]:
        """Post-refinement reference metrics (called from export_results)."""
        return self._reference_metrics()

    def _build_run_id(self) -> str:
        """Canonical Phase 12 run_id (mirrors RefineSpec.run_id in the dispatcher).

        ``<method>_<channel?>_seed<S>_beta<beta_a>_gamma_d<gamma_d>``
        ``_lambda_safe<lambda_safe>_lambda_cf<lambda_cf>``

        Method/channel/beta_cross/gamma_d are parsed from the warmstart
        cf_summary.json / audit_summary.json ``run_id`` field when present
        (dispatcher writes them); otherwise a yaml ``refine.run_id`` override or
        a stable ``refine_<exp_name>`` fallback is used so the e2e fixture (no
        dispatcher provenance) still produces a deterministic dir.
        """
        # 1. Explicit yaml override (highest priority).
        explicit = self.config.get("refine", {}).get("run_id")
        if explicit:
            return str(explicit)
        # 2. Inherit the audit run_id (carries method/seed/beta) + append λ.
        audit_run_id = (
            self._warmstart.get("audit_summary", {}).get("run_id")
            or self._warmstart.get("cf_summary", {}).get("audit_run_id")
        )
        if audit_run_id:
            return (
                f"{audit_run_id}"
                f"_lambda_safe{self.lambda_safe:g}_lambda_cf{self.lambda_cf:g}"
            )
        # 3. Fallback — stable per-exp dir name.
        exp = self.config.get("output", {}).get("exp_name", "refine")
        return (
            f"refine_{exp}"
            f"_lambda_safe{self.lambda_safe:g}_lambda_cf{self.lambda_cf:g}"
        )

    def export_results(self):
        """Write refinement_summary.json (13 fields) under refinement_results/<run_id>/.

        Schema (CF-10 + Plan 12 RefineSpec naming):
          lambda_safe, lambda_cf, max_iter_used, early_stop_iter_actual,
          pre_refinement_metrics, post_refinement_metrics,
          lesion_ssim_pre_post, nmse_improvement_pct,
          warmstart_elbo_ckpt_sha256, warmstart_audit_summary_sha256,
          warmstart_cf_dir_sha256, cf_safe_ratio_mean, cf_damp_ratio_mean

        ``lesion_ssim_pre_post`` + ``nmse_improvement_pct`` are populated by the
        phantom e2e test (which has a GT FLAIR + lesion mask); production HYW
        leaves them ``None`` (CONTEXT <deferred> — Phase 13 real-data block).
        """
        import json
        from datetime import datetime, timezone

        from evaluation.output_contract import resolve_subdir

        # Guard: train() may not have run (defensive defaults).
        pre_metrics = getattr(self, "_pre_refinement_metrics", {})
        cf_log_count = max(1, int(getattr(self, "_cf_log_count", 0)))

        summary = {
            "phase": "12-risk-aware-refinement",
            "lambda_safe": float(self.lambda_safe),
            "lambda_cf": float(self.lambda_cf),
            "max_iter_used": int(self.config["training"]["max_iterations"]),
            "early_stop_iter_actual": int(getattr(self, "_actual_early_stop_iter", 0)),
            "pre_refinement_metrics": pre_metrics,
            "post_refinement_metrics": self._compute_post_refinement_metrics(),
            "lesion_ssim_pre_post": getattr(self, "_lesion_ssim_pre_post", None),
            "nmse_improvement_pct": getattr(self, "_nmse_improvement_pct", None),
            "warmstart_elbo_ckpt_sha256": self._warmstart["elbo_ckpt_sha256"],
            "warmstart_audit_summary_sha256": self._warmstart["audit_summary_sha256"],
            "warmstart_cf_dir_sha256": _sha256_file(
                self._warmstart["cf_dir"] / "cf_summary.json"
            ),
            "cf_safe_ratio_mean": float(
                getattr(self, "_cf_safe_ratio_sum", 0.0) / cf_log_count
            ),
            "cf_damp_ratio_mean": float(
                getattr(self, "_cf_damp_ratio_sum", 0.0) / cf_log_count
            ),
            "cf03_cf07_remediated": bool(self._warmstart["cf03_cf07_remediated"]),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        refine_root = resolve_subdir(
            Path(self.config["output"]["output_dir"]), "refinement_results"
        )
        run_dir = refine_root / self._build_run_id()
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "refinement_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        # Also drop the refined ckpt into the run dir so Phase 13 calibration
        # consumes refinement_results/<run_id>/checkpoints/best.pth.
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        src = self.out_dir / "checkpoints" / "latest.pth"
        if src.exists():
            import shutil
            shutil.copy2(str(src), str(ckpt_dir / "best.pth"))
        return run_dir
