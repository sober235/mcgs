# Retrospective nominal AF 16 on the 3D T1 + FLAIR pair, jiangyizhou -- Gaussian mask (diff-mask study)

Mask-pattern study (2026-09-15). Same subject, CFL k-space, ESPIRiT maps, ground truth, gauge, arms and frozen AF16
phase-3 MC-GS recipe as `../T1FLAIR_AF16_v3_jiangyizhou`; **exactly one thing differs: the mask pattern** (2-D Gaussian
variable-density random sampling instead of the house vd-Poisson disc). Acceleration is matched to that topic's actual
T1 acceleration under the user's definition of 2026-09-14 (PE-plane points / sampled points with data): T1 at 15.20x
(2964 of 45056 points; the vd-Poisson AF16 mask: 2966, 15.19x); FLAIR shares the mask. **Labelled AF 16 (nominal)**
everywhere, as the AF16 topic is. Twin topic: `../T1FLAIR_AF16_radial_v3_jiangyizhou` (pseudo-radial spokes).

Code: worktree `/data0/congcong/code/GS/mcgs_diffmask_wt`, branch `jyz-af16-diffmask` (from `autoresearch-jyz-af24`
13b7bce). Mask generator `scripts/make_masks_af16_diffmask.py` (seeded, regenerable), inspection figures
`scripts/fig_masks_af16_diffmask.py`; the mask read-out is `logs/make_masks_af16_diffmask.json`.

## Status

2026-09-15: masks built for the user's inspection; **no reconstruction has run** (user gate: the masks are approved first).
