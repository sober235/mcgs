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
- **Round 1** (queued): symmetric per-channel sweep, phase TV 0.01 / 0.003, T1 wavelet 0.18.
- **Round 2** (queued): dtv 0.01 + guide 0.25, at lambda_edge 0.001 / 0.002, with phase TV 0.01; AF20/24 transfer; guide 0.1 / 0.5.
