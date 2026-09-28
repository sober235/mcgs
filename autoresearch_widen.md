# Widen the MC-GS margin over every comparison arm, jiangyizhou T1+FLAIR, AF 16 / 20 / 24 (from 2026-09-24)

Branch `widen-mcgs-af16-20-24`, worktree `/data0/congcong/code/GS/mcgs_widen_wt`, cut from `coupling-attribution` 81fe792.

## Rules (user decisions 2026-09-24)
- **Choice A.** Only the cross-contrast (MC-only) part of MC-GS is tuned for the margin. Any per-channel knob (edge-TV,
  wavelet, phase TV, capacity, iterations) is a general improvement: it is run on BOTH arms, and each arm keeps its own
  best setting of that symmetric sweep.
- **Choice (1).** GS-single = the single arm at lambda_edge 0 (the common-edge prior hurts the single arm: coupling
  attribution Stage 1, -0.041 / -0.018 dB at AF16). Coupling knobs (lambda_edge, lambda_dtv, edge_tv_guide_alpha) are
  never switched on in GS-single.
- GS-single is the strongest comparison arm at every AF (jyz comparison conventions), so margin = MC-GS - GS-single,
  head-only PSNR from the run's score json; SSIM must not fall.
- One trainer per card; only empty cards; never collide with another session's queue (09-24 12:00 collision with the
  coupling-attribution Stage 4 queue, five of our runs stopped at 13:47 and re-run as `_r`).

## Run
`./run_wd_mcgs.sh EXP GPU ITERS AF ARM LAMBDA_EDGE [cycle.py args]`; queue `scripts/wd_queue.sh scripts/wd_queue_r0b.txt`
(tmux `wd_sched`, `ALLOWED`, `MAXRUN`, pause file `results_arc/.wd_queue_pause`); ledger `python scripts/wd_ledger.py`.
New knob: `--set loss.edge_tv_guide_alpha=A` (each channel's edge-TV weight also spares the other contrast's edges;
tests `tests/test_edge_tv_guided.py`).

## Starting point (fair baseline, head PSNR T1 / FLAIR)
| AF | GS-single (lambda_edge 0) | MC-GS phase-3 | margin | published margin (GS-single at 0.001) |
|---|---|---|---|---|
| 16 | 35.599 / 34.904 | 35.706 / 34.928 | +0.107 / +0.024 | +0.148 / +0.042 |
| 20 | 34.715 / 34.150 | 34.832 / 34.179 (registered) | +0.117 / +0.029 | +0.165 / +0.052 |
| 24 | 33.641 / 33.137 | 33.766 / 33.182 (re-run, registered 33.767 / 33.182) | +0.126 / +0.045 | +0.179 / +0.071 |

## Round log
- **Round 0** (AF16 unless noted, lambda_edge 0.001, margin vs the fair baseline): dtv 0.01 +0.108 / +0.061,
  dtv 0.03 +0.108 / +0.059, guide 0.25 +0.131 / +0.033, guide 1.0 +0.096 / +0.007, lambda_edge 0.002 +0.116 / +0.031.
  AF20 lambda_edge 0.002 +0.128 / +0.042, 0.003 +0.100 / +0.015. AF24 lambda_edge 0.002 +0.140 / +0.065, 0.003 +0.118 / +0.045.
- Stage 4 (other session, GS-single AF16, lambda_edge 0): phase TV 0.01 +0.082 / +0.041 and T1 wavelet 0.18 +0.060 on
  the single arm -> round 1 runs them symmetrically on the joint arm.
- **Round 1** (symmetric per-channel sweep, AF16): phase TV 0.01 + T1 wavelet 0.18 is best for BOTH arms; GS-single gains
  more on T1 (35.599 -> 35.744) than MC-GS (35.706 -> 35.817), so the fair T1 margin first shrinks. phase TV 0.003 is worse.
- **Round 2**: dtv 0.01 + guide 0.25 add up (FLAIR from dtv, T1 from the guide); guide 0.1 ~ 0.25 > 0.5 > 1.0.
- **Round 3**: on the new per-channel base (`dgpw` = dtv 0.01 + edge guide 0.25 + phase TV 0.01 + T1 wavelet 0.18), and
  AF20 / AF24 single arms re-run at that base: lambda_edge 0.001 beats 0.002 at every AF; dtv 0.03 and guide 0.1 are level.
- **Round 4-5**: new knob `loss.ptv_guide_alpha` (the other contrast's edges also spare the phase-TV penalty): alpha 1.0 is
  best on T1, 0.25 is the balanced choice; phase TV 0.02 with the guide trades T1 for FLAIR. Same ordering at all three AF.

## Result (2026-09-25 07:00): one recipe for AF 16 / 20 / 24
Recipe = phase-3 recipe + phase TV 0.01 + T1 wavelet 0.18 (per-channel, both arms) + lambda_dtv 0.01 +
edge_tv_guide_alpha 0.25 + ptv_guide_alpha 0.25 (coupling, MC-GS only), lambda_edge 0.001, 3.6M points, 4000 iterations.
GS-single = its best single run of the symmetric sweep (phase TV 0.01, T1 wavelet 0.18, lambda_edge 0).

| AF | MC-GS (head PSNR / SSIM) T1 ; FLAIR | GS-single T1 ; FLAIR | fair margin now | fair margin at start | published |
|---|---|---|---|---|---|
| 12 | 37.040 / 0.9530 ; 36.364 / 0.9416 | 36.942 / 0.9516 ; 36.318 / 0.9411 | **+0.098 / +0.021** (FLAIR vs the P14-settings single, the best single at AF12; +0.046 vs the one-recipe single) | +0.109 / +0.017 (P14 recipe) | +0.150 / +0.036 (P14) |
| 16 | 35.850 / 0.9412 ; 35.009 / 0.9260 | 35.744 / 0.9394 ; 34.945 / 0.9249 | **+0.106 / +0.064** | +0.107 / +0.024 | +0.148 / +0.042 |
| 20 | 34.975 / 0.9306 ; 34.247 / 0.9135 | 34.863 / 0.9285 ; 34.178 / 0.9124 | **+0.112 / +0.069** | +0.117 / +0.029 | +0.165 / +0.052 |
| 24 | 33.897 / 0.9148 ; 33.239 / 0.8950 | 33.781 / 0.9122 ; 33.150 / 0.8935 | **+0.117 / +0.089** | +0.126 / +0.045 | +0.179 / +0.071 |

- FLAIR margin x2-2.7 at every AF; T1 margin level with the fair start (-0.001 / -0.005 / -0.009, inside the 0.02 dB band).
  With ptv_guide_alpha 1.0 instead, T1 +0.110 / +0.116 / +0.119 and FLAIR +0.061 / +0.065 / +0.084.
**AF12 added (2026-09-27, user request "要补充 for jyz 数据 with af=12").** Same recipe and fair GS-single on the AF12 topic's
mask (3847 points, 11.71x; configs `t1flair_v3_af12_jyz*.yaml` = byte-identical copies of the AF12 line's hyw-named files,
which read the jiangyizhou AF12 topic). MC-GS first on PSNR / SSIM / NMSE in all three planes of both contrasts; over the
best non-GS arm +0.878 (CS-SENSE, TV converged) / +0.739 (Joint SENSE); the one recipe edges the AF12 line's own P14
(37.015 / 36.360). Fair margin across the four AF, T1 / FLAIR (rule A, best single run): +0.098 / +0.021, +0.106 / +0.064, +0.112 / +0.068,
+0.117 / +0.090 (full precision); against the one-recipe single at AF12 the FLAIR figure is +0.046. The T1 margin grows
monotonically with acceleration; FLAIR grows from 15.19 on. P14 note: P14 used
`gaussian.rotation_lr 0.008`, the one recipe keeps the config's 0.0003 (not tested in this loop). The P14 joint re-run in this code tree
(`wd_af12_p14_j`) reproduces the registered 37.0147 / 36.3597 to +0.0003 / -0.0003 dB; its fair start uses the P14
singles at lambda_edge 0 (`wd_af12_p14_s_t1_e0`, `wd_af12_p14_s_fl_e0b`; the first FLAIR attempt was stopped after another
user's job landed on the same card).
- MC-GS first on head PSNR and SSIM in all six cells; every non-GS arm of the AF topics is >= 0.6 dB behind (unchanged).
- The margin still grows with AF on both contrasts. Single runs, one seed (floor 0.001 dB from the AF16 topic).
- Not done: volumes of the winners (re-run with SAVE_VOLUMES=1 for figures), registration in the AF topics, other subjects.

## Open after the 2026-09-28 tidy-up

- **rotation_lr is not swept symmetrically.** The P14-settings single (rotation_lr 0.008) is the best FLAIR single at
  AF12 (36.343 vs 36.318 for the one-recipe single; the two differ only in rotation_lr), and rule A takes it, which is
  why AF12 FLAIR reads +0.021. MC-GS with the new coupling has never run at rotation_lr 0.008, at any AF, nor have the
  AF16-24 singles. Rule A asks for both arms at both values (12 runs); until then every fair margin is provisional on
  this knob.
- Display rule (user 2026-09-28): every shown result carries every comparison method. The asymmetric-probe figures and
  the three summary charts contain only the GS arms and are labelled as GS-model analyses on the viewing page.
- Chinese summary: `docs/2026-09-28-widen-mcgs-summary.md`; ledger snapshot: `docs/superpowers/records/2026-09-28-widen-ledger.md`.
