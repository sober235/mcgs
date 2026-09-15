# Retrospective nominal AF 16 on the 3D T1 + FLAIR pair, jiangyizhou -- 2D-CAIPIRINHA mask (diff-mask study)

Mask-pattern study (2026-09-15). Same subject, CFL k-space, ESPIRiT maps, ground truth, gauge, arms and frozen AF16
phase-3 MC-GS recipe as `../T1FLAIR_AF16_v3_jiangyizhou`; **exactly one thing differs: the mask pattern** (a regular
4 x 3 phase-encode lattice with a CAIPIRINHA shift of 2, instead of the house vd-Poisson disc; the shift is chosen by
CAIPI's own criterion -- 4x3(2) is the only candidate at this budget whose shift widens the minimum aliasing-replica
separation, 0.3005 FOV against 0.2500 unshifted). The lattice plus the ACS overshoots the point budget and the excess
is removed at the largest normalised radius, i.e. elliptical scanning, cutting at rn 0.941 with the lattice intact
inside. Acceleration is matched to the AF16 topic's actual T1
acceleration under the user's definition of 2026-09-14 (PE-plane points / sampled points with data): T1 at 15.20x
(2964 of 45056 points; the vd-Poisson AF16 mask: 2966, 15.19x); FLAIR shares the mask. Twin topics:
`../T1FLAIR_AF16_gauss_v3_jiangyizhou` (Gaussian random), `../T1FLAIR_AF16_radial_v3_jiangyizhou` (pseudo-radial).

Code: worktree `/data0/congcong/code/GS/mcgs_diffmask_wt`, branch `jyz-af16-diffmask`. Generator
`scripts/make_masks_af16_diffmask.py --kinds caipi`, read-out `logs/make_masks_af16_diffmask.json`.

## Status

2026-09-15: mask built for the user's inspection; **no reconstruction has run**.
