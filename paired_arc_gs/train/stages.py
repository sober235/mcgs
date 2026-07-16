"""§13 stage scheduler for Paired-ARC-GS.

The spec (MS_GS/Paired_ARC_GS_general_framework.md §13) defines a 10-stage training
pipeline. Each stage decides:
  * which parameter groups are optimized (geometry / latent+decoder / phase / deformation
    / change / gabor),
  * which loss terms are active,
  * whether change-birth / gabor-birth / pruning / stability-selection fire.

This module encodes that schedule as an explicit, iteration-indexed state machine so the
trainer stays a thin driver. A `StagePlan` is built from config; `plan.at(global_iter)`
returns the `StageState` in force at that iteration. Layers are delivered incrementally:
a stage whose module is not yet implemented is simply skipped (its `active` flag off and
its iteration budget collapsed to 0), so the same scheduler drives Layer-1 through
Layer-4 builds without change.

Stage map (spec §13):
    S1  ACS init                         (setup, 0 iters — done in model builder)
    S2  fixed-geometry response fit      (opt: z_A, D1, D2, phi)         [geometry FROZEN]
    S3  low-freq deformation estimate    (opt: T_m)                       [change OFF]
    S4  low-LR shared geometry           (opt: mu,Sigma  at eta/10..50)
    S5  measurement-audited change birth (fit residual -> propose; audit-gain accept)
    S6  audit-supported gabor birth      (high-freq held-out residual)
    S7  joint optimize + pruning
    S8  stability selection (B fit/audit splits)
    S9  full-measurement refine          (audit k-space returned to DC; topology frozen)
    S10 CG data-consistency projection   (post-hoc; replaces per-coil k-space replacement)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional


# Canonical stage identifiers (spec §13). Order is load-bearing.
STAGE_IDS: Tuple[str, ...] = (
    "s1_acs_init",
    "s2_response_fit",
    "s3_deformation",
    "s4_geometry_lowlr",
    "s5_change_birth",
    "s6_gabor_birth",
    "s7_joint_prune",
    "s8_stability",
    "s9_full_refine",
    "s10_cg_project",
)

# Which optimizable parameter groups each stage TOUCHES (unfreezes). A group absent here
# is frozen for that stage. Group names must match PairedARCModel.param_group_names().
STAGE_OPT_GROUPS: Dict[str, Tuple[str, ...]] = {
    "s1_acs_init":      (),                                   # setup only
    "s2_response_fit":  ("latent", "decoder", "phase"),       # geometry frozen
    "s3_deformation":   ("deformation",),                     # change off
    "s4_geometry_lowlr":("geometry", "latent", "decoder", "phase", "deformation"),
    "s5_change_birth":  ("latent", "decoder", "change", "phase", "deformation"),
    "s6_gabor_birth":   ("latent", "decoder", "change", "gabor", "phase"),
    "s7_joint_prune":   ("geometry", "latent", "decoder", "change", "gabor", "phase", "deformation"),
    "s8_stability":     ("latent", "decoder", "change"),
    "s9_full_refine":   ("latent", "decoder", "change", "gabor", "phase"),   # topology frozen
    "s10_cg_project":   (),                                   # post-hoc CG, no SGD
}

# Loss terms active per stage (names match the loss module's term keys). L_DC is always on.
STAGE_LOSS_TERMS: Dict[str, Tuple[str, ...]] = {
    "s1_acs_init":      (),
    "s2_response_fit":  ("dc", "geo", "phase", "llr", "freqextrap"),
    "s3_deformation":   ("dc", "def"),
    "s4_geometry_lowlr":("dc", "geo", "def", "phase", "llr", "freqextrap"),
    "s5_change_birth":  ("dc", "geo", "change_sparse", "route", "def", "phase", "llr", "freqextrap"),
    "s6_gabor_birth":   ("dc", "geo", "change_sparse", "route", "phase"),
    "s7_joint_prune":   ("dc", "geo", "change_sparse", "route", "def", "phase", "llr", "freqextrap"),
    "s8_stability":     ("dc", "change_sparse", "route"),
    "s9_full_refine":   ("dc", "geo", "change_sparse", "route", "phase", "llr", "freqextrap"),
    "s10_cg_project":   (),
}

# Special one-shot / recurring actions a stage triggers.
STAGE_ACTIONS: Dict[str, Tuple[str, ...]] = {
    "s5_change_birth":  ("propose_change", "audit_accept"),
    "s6_gabor_birth":   ("propose_gabor",),
    "s7_joint_prune":   ("prune",),
    "s8_stability":     ("stability_select",),
    "s9_full_refine":   ("restore_audit_kspace", "freeze_topology"),
    "s10_cg_project":   ("cg_project",),
}


@dataclass
class StageState:
    """Resolved schedule state at a given global iteration."""
    stage_id: str
    stage_index: int
    local_iter: int                     # iteration within this stage
    stage_iters: int                    # total iters allotted to this stage
    opt_groups: Tuple[str, ...]         # parameter groups to optimize (others frozen)
    loss_terms: Tuple[str, ...]         # active loss-term keys
    actions: Tuple[str, ...]            # special actions this stage runs
    geometry_lr_scale: float = 1.0      # §13-S4: eta_geometry << eta_signal
    is_stage_start: bool = False        # True on the first iter of the stage
    is_stage_end: bool = False          # True on the last iter of the stage


@dataclass
class StagePlan:
    """Iteration-indexed plan over the enabled stages.

    Built from config: each stage gets an iteration budget (0 disables it). Disabled
    modules collapse their stage to 0 iters so unimplemented layers are transparently
    skipped. `at(i)` maps a global iteration to its StageState.
    """
    stage_iters: Dict[str, int]
    geometry_lr_scale: float = 0.05      # spec §13-S4 recommends 1/10..1/50
    _boundaries: List[Tuple[int, int, str]] = field(default_factory=list)  # (start, end, id)
    total_iters: int = 0

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "StagePlan":
        """Read `training.stages.<stage_id>.iters` (default 0 = disabled).

        A stage is enabled iff its iters > 0 AND its module flag (if any) is on.
        Module gating: change stages need `model.change.enable`; gabor needs
        `model.gabor.enable`; deformation needs `model.deformation.enable`; phase needs
        `model.phase.enable`; cg needs `training.cg.enable`.
        """
        tcfg = cfg.get("training", {})
        scfg = tcfg.get("stages", {}) or {}
        mcfg = cfg.get("model", {})

        def _module_on(stage_id: str) -> bool:
            if stage_id == "s3_deformation":
                return bool(mcfg.get("deformation", {}).get("enable", False))
            if stage_id in ("s5_change_birth",):
                return bool(mcfg.get("change", {}).get("enable", False))
            if stage_id == "s6_gabor_birth":
                return bool(mcfg.get("gabor", {}).get("enable", False))
            if stage_id == "s8_stability":
                return bool(mcfg.get("change", {}).get("enable", False)) and \
                       bool(tcfg.get("stability", {}).get("enable", False))
            if stage_id == "s10_cg_project":
                return bool(tcfg.get("cg", {}).get("enable", False))
            return True

        iters: Dict[str, int] = {}
        for sid in STAGE_IDS:
            requested = int(scfg.get(sid, {}).get("iters", 0)) if isinstance(scfg.get(sid), dict) else int(scfg.get(sid, 0) or 0)
            iters[sid] = requested if _module_on(sid) else 0

        plan = cls(stage_iters=iters, geometry_lr_scale=float(tcfg.get("geometry_lr_scale", 0.05)))
        plan._build_boundaries()
        return plan

    def _build_boundaries(self) -> None:
        self._boundaries = []
        cursor = 0
        for sid in STAGE_IDS:
            n = int(self.stage_iters.get(sid, 0))
            if n <= 0:
                continue
            self._boundaries.append((cursor, cursor + n, sid))
            cursor += n
        self.total_iters = cursor

    def at(self, global_iter: int) -> Optional[StageState]:
        """Return the StageState at `global_iter`, or None if past the plan end."""
        for idx, (start, end, sid) in enumerate(self._boundaries):
            if start <= global_iter < end:
                local = global_iter - start
                lr_scale = self.geometry_lr_scale if sid == "s4_geometry_lowlr" else 1.0
                return StageState(
                    stage_id=sid,
                    stage_index=idx,
                    local_iter=local,
                    stage_iters=end - start,
                    opt_groups=STAGE_OPT_GROUPS[sid],
                    loss_terms=STAGE_LOSS_TERMS[sid],
                    actions=STAGE_ACTIONS.get(sid, ()),
                    geometry_lr_scale=lr_scale,
                    is_stage_start=(local == 0),
                    is_stage_end=(local == end - start - 1),
                )
        return None

    def enabled_stages(self) -> List[str]:
        return [sid for _, _, sid in self._boundaries]

    def summary(self) -> str:
        parts = [f"{sid}:{end - start}" for start, end, sid in self._boundaries]
        return f"StagePlan(total={self.total_iters} iters | " + " ".join(parts) + ")"
