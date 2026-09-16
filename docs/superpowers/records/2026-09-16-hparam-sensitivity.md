# Record — MC-GS hyper-parameter sensitivity (shared Gaussian count, private ratio), jiangyizhou AF16 (2026-09-16)

Branch `hparam-sensitivity` (worktree `mcgs_hparam_wt`, cut from `origin/autoresearch-jyz-af24` 5ea6189). Spec
`docs/superpowers/specs/2026-09-16-hparam-sensitivity-design.md`, plan `docs/superpowers/plans/2026-09-16-hparam-sensitivity.md`.
Topic `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_hparam_v3_jiangyizhou/` (mirror `t1flair_af16_hparam_v3_jiangyizhou/`).

## 1. What ran

Six MC-GS runs on jiangyizhou T1+FLAIR, the AF16 house vd-Poisson mask of `T1FLAIR_AF16_v3_jiangyizhou` (md5 d08179b8…,
T1 15.19x / FLAIR 16.49x), the frozen phase-3 recipe (`run_hp_mcgs.sh`, recipe block byte-identical to `run_af24_mcgs.sh`),
4000 iterations, seed 42. Only the appended `--set` differs per run:

| exp | initial_num_points | num_private_points | card | start | end | wall |
|---|---|---|---|---|---|---|
| hp_n36 | 3 600 000 | 0 | 1 | 18:27 | 19:33 | 66 min |
| hp_n18 | 1 800 000 | 0 | 2 | 18:29 | 19:22 | 53 min |
| hp_n09 | 900 000 | 0 | 2 | 19:23 | 20:11 | 47 min |
| hp_n72 | 7 200 000 | 0 | 1 | 19:34 | 20:56 | 82 min |
| hp_p05 | 3 600 000 | 180 000 | 3 | 19:40 | 20:46 | 66 min |
| hp_p10 | 3 600 000 | 360 000 | 4 | 19:41 | 20:49 | 68 min |

Queue runner `scripts/hp_queue.sh` (empty cards only, shared `.gpulock.<card>`, one trainer per card); a second instance on
the idle cards 3 and 4 was started at 19:39 once the gauss and peds lines had finished (their queues were checked: nothing
pending). The user's constraint "do not disturb other experiments" held: the three gauss-loop trainers that ended during the
session ended with DONE and saved scores (18:10, 18:28, 18:29); host memory never went below 560 GB free.

## 2. The table (`logs/hp_summary.md`, head-masked metrics, T1 / FLAIR)

| exp | N_s | N_p/N_s | T1 head PSNR / SSIM / HFEN / NMSE | FLAIR head PSNR / SSIM / HFEN / NMSE | vs 3.6M (dB) |
|---|---|---|---|---|---|
| hp_n09 | 0.9M | 0 | 35.357 / 0.9376 / 0.1142 / 0.01404 | 34.135 / 0.9165 / 0.1176 / 0.01647 | −0.349 / −0.793 |
| hp_n18 | 1.8M | 0 | 35.592 / 0.9391 / 0.1133 / 0.01330 | 34.664 / 0.9227 / 0.1129 / 0.01458 | −0.114 / −0.264 |
| hp_n36 | 3.6M | 0 | 35.706 / 0.9396 / 0.1139 / 0.01295 | 34.928 / 0.9261 / 0.1121 / 0.01372 | anchor |
| hp_n72 | 7.2M | 0 | 35.670 / 0.9392 / 0.1148 / 0.01306 | 34.980 / 0.9271 / 0.1126 / 0.01356 | −0.036 / +0.052 |
| hp_p05 | 3.6M | 5 % | 35.706 / 0.9396 / 0.1139 / 0.01295 | 34.933 / 0.9262 / 0.1122 / 0.01370 | +0.000 / +0.005 |
| hp_p10 | 3.6M | 10 % | 35.708 / 0.9396 / 0.1140 / 0.01295 | 34.938 / 0.9263 / 0.1122 / 0.01369 | +0.002 / +0.010 |

Whole-volume PSNR T1 / FLAIR: 38.951 / 37.867, 39.159 / 38.337, 39.268 / 38.585, 39.222 / 38.628, 39.268 / 38.591,
39.270 / 38.598. Complex-gauge head PSNR at 3.6M: 32.497 / 31.883. VRAM 32.3–33.8 GB.

## 3. Anchor verdict

hp_n36 vs the registered `gs_joint_p3` (35.706 / 34.928): delta 0.000 / 0.000 dB — PASS. Same seed, same code path, the
run reproduces the AF16 record to three decimals, so this table is directly comparable with the phase-3 record.

## 4. Readings (pre-registered in spec section 8, applied as written)

- **Group 1 (capacity).** Monotone improvement from 0.9M to 3.6M on every head metric (PSNR T1 +0.349, FLAIR +0.793 dB).
  3.6M -> 7.2M is **a trade, not saturation** under the pre-registered criterion: T1 −0.036 dB (outside the 0.02 dB
  tolerance), FLAIR +0.052 dB; SSIM and NMSE move the same way. This is the phase-3 finding again (3.6M / 4.8M / 6.0M:
  T1 35.706 / 35.697 / 35.675, FLAIR 34.928 / 34.962 / 34.977): T1 peaks at 3.6M, FLAIR keeps gaining with capacity.
  On "does MC-GS depend on a particular count": between 1.8M and 7.2M, T1 moves by at most 0.11 dB and FLAIR by at most
  0.27 dB; 0.9M is clearly under-provisioned (FLAIR −0.8 dB).
- **Group 2 (private ratio).** 5 % and 10 % against 0 %: +0.000 / +0.005 and +0.002 / +0.010 dB, every other cell unchanged
  to three or four decimals — **null**, consistent with `p3_r2_priv03` (0.3M private, +0.005 / +0.015). The shared cloud
  already carries what FLAIR needs; the private set adds parameters, not information. PSNR / SSIM / NMSE cannot support a
  claim that the private set preserves contrast-specific information; that needs lesion data, which this study excludes.
- **Cost.** 0.9M 47 min / 32.3 GB; 3.6M 66 min / 32.9 GB; 7.2M 82 min / 33.8 GB (one trainer per A100 80 GB).

## 5. Caveats

One subject, one mask draw, magnitude gauge (the complex-gauge numbers are in the table), the private set is FLAIR-only
(asymmetric by construction), no lesion metric, recipe and classical lambdas were selected on this subject (fair, not held
out). The received design's 50k–400k grid and "R = 6" were not run (user decision 2026-09-16, reasons in STATUS.md).
Smoke runs `hp_smoke_*` (60 iterations) are feasibility checks, never results.
