# MC-GS hyper-parameter sensitivity: shared Gaussian count and private ratio — design (2026-09-16)

Branch `hparam-sensitivity`, worktree `mcgs_hparam_wt`, cut from `origin/autoresearch-jyz-af24` 5ea6189.
Status of this document: approved by the user on 2026-09-16 ("同意" to the four decisions listed in `STATUS.md`).

## 1. Purpose

Two small sensitivity curves for the paper, measured on the line's figure-of-record setting (jiangyizhou T1+FLAIR,
nominal AF 16 house vd-Poisson mask) with everything except one knob frozen at the phase-3 recipe:

- **Group 1, capacity.** Shared Gaussian count N_s. Question: does MC-GS depend on a particular count, and does
  performance saturate as capacity grows?
- **Group 2, private ratio.** Private-to-shared ratio N_p / N_s at N_s = 3.6M. Question: does adding a
  contrast-private Gaussian set change the result beyond adding parameters?

The paper shows two panels: N_s -> head PSNR / SSIM, and N_p / N_s -> head PSNR / SSIM / NMSE. No 5x5 grid, no repeat
at other accelerations.

## 2. Decisions taken (user 2026-09-16)

1. **N_s grid re-centred on the frozen recipe, factor-2 steps: 0.9M, 1.8M, 3.6M, 7.2M.** The received design's
   50k-400k is 9-72x below the operating point (volume 256 x 256 x 176 = 11.5M voxels; the recipe's 3.6M is
   0.31 Gaussians per voxel; the line never ran below 0.6M).
2. **The published MC-GS has N_p = 0 and keeps it.** Group 2 adds a FLAIR-private residual set on top of the recipe,
   N_p / N_s in {0, 5 %, 10 %} = {0, 180k, 360k}. The 0 % arm is the same run as group 1's 3.6M arm.
3. **AF 16, the existing house mask.** `T1FLAIR_AF16_v3_jiangyizhou/data/mask_af16_3d.h5`
   (md5 d08179b8e74f35846455983dbe299294; T1 15.19x, FLAIR 16.49x under the user's AF definition). No new mask.
4. **No lesion CNR.** The T1+FLAIR subjects are healthy volunteers. Group 2 reports PSNR / SSIM / NMSE like group 1.
   Subject: jiangyizhou. huangyuwei (`T1FLAIR_AF16_retro`, byte-identical mask file) is a possible later extension and
   is not part of this design.

## 3. Fixed factors

- Code: this branch. The recipe block of the launcher is byte-identical to `run_af24_mcgs.sh` (the frozen phase-3
  recipe): P6 cold start (init_mode head, 3-NN scales, edge-TV threshold from ZF), edge-TV lambda_geo 0.10 /
  kappa_q 0.85 / w_floor 0.10 / eps 0.04, kspace_high 2.2, warm-up 500, wavelet db4 on its finest level with
  lambda_wav T1 0.12 / FLAIR 0.10, common-edge 0.001, phase TV 0.03; 4000 iterations; seed 42 with
  `deterministic=False` (as every registered run, `arc/cycle.py:273`).
- Base config `configs/t1flair_v3_af16_jyz.yaml` (joint arm, known_zero on FLAIR only).
- Only the `--set` values appended after the recipe block differ between runs. `--set` is applied in argv order, so
  the appended value overrides the block's `gaussian.initial_num_points=3600000` (`arc/cycle.py:197`).
- The point count is fixed during a run: `adaptive_control.enable` is absent from the config
  (`paired_gs/trainers/joint_trainer.py:672`), and every registered log ends with `N=` equal to the initial count.
- Private set initialisation: trainer default — a stride subset of the shared cloud, amplitude x 0.1, geometry and
  amplitude trained on the FLAIR data term only (`joint_trainer.py:304-339`); the same setting as the AF16 phase-3
  test `p3_r2_priv03`.

## 4. Runs

| exp | initial_num_points | num_private_points | role |
|---|---|---|---|
| hp_n09 | 900 000 | 0 | group 1 |
| hp_n18 | 1 800 000 | 0 | group 1 |
| hp_n36 | 3 600 000 | 0 | group 1, group 2 (0 %), regression anchor |
| hp_n72 | 7 200 000 | 0 | group 1 |
| hp_p05 | 3 600 000 | 180 000 | group 2 (5 %) |
| hp_p10 | 3 600 000 | 360 000 | group 2 (10 %) |

All six save volumes (`SAVE_VOLUMES=1`) for the later figures and head-NMSE.

**Regression anchor.** hp_n36 repeats the registered `gs_joint_p3` (AF16 topic README: head PSNR 35.706 / 34.928).
Gate: |delta| <= 0.02 dB on both contrasts' head PSNR. If it fails, the curve is still internally valid (one code state),
but nothing is compared to the registered numbers until the drift between the phase-3 tip and this branch is found.

## 5. Metrics

From `arc/cycle.py`'s score json, per contrast: `head_psnr`, `head_ssim`, `head_hfen` (the line's main gauge), plus
whole-volume `psnr`, `ssim`, `nmse`, and `head_psnr_cplx` as the complex-gauge footnote. Head NMSE is recomputed from
the saved volumes with the scorer's head mask by the summariser (the json has no head_nmse). Elapsed time and VRAM are
reported per run because capacity is a cost as well as a knob.

## 6. Scheduling and the do-not-disturb rules

- Launcher `run_hp_mcgs.sh EXP GPU ITERS [extra args]`: EXP must start with `hp_`; refuses to overwrite an existing
  log / score / run directory; `TORCH_EXTENSIONS_DIR=/data0/congcong/code/GS/mcgs_hparam_wt.torch_ext` (its own build
  cache, never the AF24 worktree's); appends one line per run to `results_arc/hp_runs.log`.
- Queue `scripts/hp_queue.sh scripts/hp_queue.txt`: cards from `HP_CARDS` (default `1 2`); a card is free by the
  gauss scheduler's rule (memory < 1500 MiB, no compute process, no live `.gpulock.<card>` owner under
  `Results_20260908/Cardiac_FreeRunning_MCGS/work/`); the lock is held for the run's lifetime; one trainer per card;
  pause file `results_arc/.hp_queue_pause`. Queue order: hp_n36, hp_n18, hp_n09, hp_n72, hp_p05, hp_p10 (anchor first).
  Launched detached (`setsid nohup`) from a bash script so the session's memory watchdog cannot kill it.
- Nothing in any other worktree or in any existing `Results_20260908/` topic is written. Outputs live only in this
  worktree's `results_arc/`.
- Budget: six runs at 6-7.5 h each (7.2M longer, ~50 % more VRAM), about 45 GPU-hours; two cards -> about one day.

## 7. Outputs and later steps

- `results_arc/hp_<exp>.log`, `hp_<exp>_score.json`, `hp_<exp>_volumes.npz`, `hp_runs.log`.
- `scripts/hp_summarize.py` -> `results_arc/hp_summary.md` and `.csv`: one row per run (N_s, N_p, ratio, per-contrast
  head PSNR / SSIM / HFEN / NMSE, whole-volume PSNR / SSIM / NMSE, elapsed, VRAM), deltas against hp_n36, the anchor
  check. Unit test on synthetic score jsons.
- After the runs (next session): the two-panel figure (dataviz conventions of the trend topic), the topic folder
  `Results_20260908/T1FLAIR_AF16_hparam_v3_jiangyizhou/` (README, scripts, logs, figures; `data/` symlinked to the
  AF16 topic), a dated record in `docs/superpowers/records/`.

## 8. Reading the result (pre-registered)

- Group 1: reported as measured. "Saturation" means the 3.6M -> 7.2M step is within the anchor tolerance (0.02 dB) on
  both contrasts. A T1 loss with FLAIR gain at higher capacity would repeat the phase-3 finding (3.6M / 4.8M / 6.0M:
  T1 35.706 / 35.697 / 35.675, FLAIR 34.928 / 34.962 / 34.977) and is not a defect.
- Group 2: 5 % and 10 % within +-0.02 dB of 0 % on both contrasts = null, consistent with p3_r2_priv03
  (+0.005 / +0.015). Anything larger is reported as measured. PSNR / SSIM / NMSE cannot support a claim that the private
  set "preserves contrast-specific information"; that claim needs lesion data, which this design excludes.

## 9. Out of scope

An R = 6 mask; lesion CNR; the classical comparison arms (unchanged by these knobs); a second subject; any change to
the recipe.
