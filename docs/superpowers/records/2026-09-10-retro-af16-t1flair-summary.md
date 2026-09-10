# Retrospective AF 16 on the true 3D T1 + FLAIR pairs — session record, 2026-09-09 / 10

Same data, maps, ground truth, gauge, eight arms and hyper-parameters as the AF 12 studies
(`2026-09-09-retro-af12-t1t2-t1flair-summary.md`); exactly one thing changed, the mask. Archives:
`Results_20260908/{T1FLAIR_AF16_retro, T1FLAIR_AF16_v3_jiangyizhou}` on amax, mirrored here as
`t1flair_af16_retro/` and `t1flair_af16_v3_jiangyizhou/` (README, scripts, figures; no volumes).
Each README carries the full tables and the regeneration commands.

## Setup

- **Mask**: house `vdPoisMex(256,176,256,176, accelx 4, accely 4, calib 24, ellipse 0, pp 1)` —
  2966 / 45056 PE points (6.58 %), **AF 15.19** on the full plane (the AF 12 mask: 3847 points,
  11.71). Against FLAIR's elliptical shutter: 12.49 (AF 12: 9.74). One mask for both contrasts and
  both subjects.
- **Isolation**: a concurrent autoresearch loop was editing the trainer in the main checkout, so
  this study ran in a git worktree at `30e6775` (`/data0/congcong/code/GS/mcgs_af16_wt`, branch
  `t1flair-af16-retro`) with its own `results_arc/` and its own JIT extension cache; inputs are
  symlinks into the AF 12 topics and the export script asserts them instead of rewriting.
- **Compute**: GPUs 0 and 1 only, shared with another user's ~33 GB training job until 01:29.
  `sched_af16.py` (strict priority, per-job VRAM need, two jobs per card, drain-and-fill,
  restart-safe) ran the thirty jobs; all thirty returned rc 0. Wall times are not comparable with
  AF 12 (MC-GS joint 137 min shared vs 29 min alone; ZS-SSDU T1 87 epochs / 5 h on one subject).

## Results — head-only PSNR, dB (AF 12 value of the same arm in brackets)

| method | hyw T1 | hyw FLAIR | jyz T1 | jyz FLAIR |
|---|---|---|---|---|
| ZS-SSDU | **33.67** (34.13) | **32.69** (33.27) | 34.15 (35.72) | **34.01** (35.03) |
| CS-SENSE | 33.08 (34.61) | 31.97 (33.25) | **34.41** (35.76) | 33.81 (35.08) |
| MC-GS joint | 32.03 (33.18) | 30.74 (31.81) | 32.89 (34.16) | 32.40 (33.39) |
| MC-GS single | 31.98 (33.30) | 30.48 (31.76) | 32.33 (33.80) | 32.14 (33.41) |
| CG-SENSE | 30.55 (31.94) | 31.05 (32.19) | 32.35 (33.51) | 32.83 (33.96) |
| 3D-DIP | 29.91 (30.11) | 29.07 (29.40) | 30.99 (31.36) | 30.65 (30.81) |
| 3D-INR | 29.28 (30.84) | 29.17 (29.94) | 30.38 (31.64) | 30.26 (31.19) |
| ZF | 23.42 (23.70) | 26.43 (26.74) | 25.84 (26.13) | 27.88 (28.19) |

Gauge proof (`check_gauge.py`, CG-SENSE T1) passed in both topics.

## Reading

1. **Sixteen-fold costs every regularised arm 1.0 to 1.6 dB inside the head.** The exception is
   ZS-SSDU on huangyuwei (−0.47 / −0.58), which takes first place on three of the four columns;
   CS-SENSE keeps jiangyizhou T1. Whoever leads, the lead over the other of the two is 0.2 to 0.7 dB.
2. **MC-GS joint is third or fourth of eight on every column**, 1.5 to 1.9 dB behind the leader
   (AF 12: 1.4 to 1.7). It degrades less than its own single arm and than CS-SENSE on all four
   columns (−1.15 / −1.06 / −1.27 / −0.99 dB), so the gap to CS-SENSE narrows (hyw 1.43 → 1.05 T1,
   1.44 → 1.23 FLAIR) while the gap to ZS-SSDU widens.
3. **The shared geometry gains inside the head in all four channel-pairings at AF 16** —
   joint − single head-only = +0.05 (hyw T1), +0.27 (hyw FLAIR), +0.56 (jyz T1), +0.27 (jyz FLAIR)
   dB, head SSIM +0.003 to +0.021 — against two of four at AF 12 (−0.11, +0.05, +0.36, −0.02). The
   whole-volume gain is unchanged in kind (+0.5 to +1.0 dB, background suppression). This is the one
   qualitatively new fact: the value of the shared geometry grows as the data get sparser. One seed
   per arm; the only seed repeat measured on this data family (the autoresearch loop's
   `ar0_A_base` vs `ar0_A_seed43`, hyw T1+T2, 1500 iterations) differs by 0.003 dB head-only, so
   the FLAIR and jiangyizhou gains are far outside it and the hyw T1 gain (+0.05) is marginal.
4. Per-slice: MC-GS joint wins 12 / 556 (hyw T1), 0 / 570 (hyw FLAIR), 5 / 600 (jyz T1) and
   0 / 606 (jyz FLAIR) scored slices; CS-SENSE and ZS-SSDU split almost every slice. Same picture
   as AF 12.
5. Every MC-GS run here is still a **cold start** at the frozen working point. The autoresearch loop
   running in parallel on the main checkout is the place where that assumption is being lifted;
   its cold-start levers (head-support init, ZF-sourced edge-TV kappa, densification) were not used
   here, so these MC-GS numbers are the same recipe as the AF 12 topics, nothing more.

## Gotchas recorded along the way

- A greedy "first job that fits" scheduler starves heavy runs when light jobs keep arriving;
  strict priority with an explicit drain (fillers only if they finish first) fixed it.
- Another user's job on the same card is a 2 to 5x slowdown; `nvidia-smi` reads low between two
  restarts of a foreign job — assume a floor for other users' memory, retry a failed job once.
- ZS-SSDU T1 no longer early-stops at ~23 epochs (validation creeps down by 1e-4 per epoch); it ran
  87 epochs. Do not change the criterion — the AF 12 numbers used the same one.
- Symlinked inputs: the parent export script rewrites `ksp_*.h5`; the clone asserts the symlink.
- The agent harness kills its own background tasks on low MemFree even at 850 GB available; run
  the scheduler and the post-processing under `setsid nohup` and watch the log.
- The AF 12 topics' `summary.json` now carry extra autoresearch rows (`gs_joint_v2`); the AF 16
  comparison ranks among the canonical eight arms only.
