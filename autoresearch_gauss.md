# autoresearch — MC-GS on the Gaussian variable-density random mask (jiangyizhou T1+FLAIR, nominal AF 16): +0.5 dB over every comparison arm

Started 2026-09-16 (user: "新开一个分支, 在新开分支上调参 for gaussian mask using karpathy autoresearch, 使得提出的 mc-gs
超过所有对比方法 0.5 dB 以上"). Branch `autoresearch-gauss-af16`, worktree `mcgs_gauss_wt`, cut from `jyz-af16-diffmask` 889448a.

## Objective

MC-GS joint at least **0.5 dB above the strongest comparison arm on head-only PSNR (magnitude gauge) on BOTH contrasts**,
with all six head cells (PSNR / SSIM / HFEN x T1 / FLAIR) first. Comparison arms = every arm of the gauss topic except
MC-GS: CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR **and GS-single at the frozen
AF16 phase-3 recipe** (user 2026-09-16: GS-single counts as a comparison arm and is not re-tuned — that is the
"适当调差" this line implements; no worse settings are hand-picked for it). L1-ESPIRiT is out (user 2026-09-15).

Bar (`Results_20260908/T1FLAIR_AF16_gauss_v3_jiangyizhou/logs/bar_gauss.json`, built by `scripts/bar_gauss.py`): the
strongest comparison arm is GS-single on all six cells.

| head cell | T1 bar (arm) | FLAIR bar (arm) | strongest non-GS |
|---|---|---|---|
| PSNR | 34.771 (GS-single) | 34.294 (GS-single) | ZS-SSDU 34.375 / 33.972 |
| SSIM | 0.9305 (GS-single) | 0.9202 (GS-single) | ZS-SSDU 0.9273 / 0.9196 |
| HFEN | 0.1412 (GS-single) | 0.1295 (GS-single) | L1-SPIRiT 0.1653 / 0.1360 |

**Targets: T1 head PSNR >= 35.271, FLAIR >= 34.794.** Parent (below) is 34.909 / 34.317 → +0.362 / +0.477 still to find.
Ledger objective: maximise `min_margin_mag` = min(dT1_mag, dFLAIR_mag) over the bar, subject to `six_first`.
Forecast written before the first run: the four earlier loops of this project gained +0.10 to +0.19 dB each over their
parent by parameters; 0.5 dB over a frozen GS-single is beyond that record. The loop reports where it lands.

## Parent

The AF16 phase-3 recipe as it ran frozen on this mask (`mcgs_diffmask_wt/results_arc/af16g_p3_joint_full4k`, 4000 it):
head PSNR **34.909 / 34.317**, SSIM 0.9333 / 0.9210, HFEN 0.1344 / 0.1278, sharp 0.796 / 0.759; margins over the bar
+0.138 / +0.023. Recipe: P6 repaired cold start (init_mode head, 3-NN scales, edge-TV threshold from ZF), 3.6M points,
edge-TV lambda_geo 0.10 / kappa_q 0.85 / w_floor 0.10 / eps 0.04, kspace_high 2.2, warmup 500, wavelet db4 finest
level only T1 0.12 / FLAIR 0.10, common-edge 0.001, phase TV 0.03. Baked into `autoresearch_gauss.sh` (recipe block
byte-identical to `run_diffmask_mcgs.sh`, checked with diff). Candidates differ from the parent only by their extra args.

## Isolation

`g_` names only; configs `configs/t1flair_v3_af16g_jyz*.yaml` selected by name (`SINGLE=T1|FLAIR`); rows to
`autoresearch_gauss.tsv` via `scripts/autoresearch_gauss_row.py`; runs in this worktree's `results_arc/`; CUDA extension
cache `mcgs_gauss_wt.torch_ext`. The gauss topic's data and mask are read, never written. No code from other lines is merged.

## Protocol

Screen 1500 it, promote 4000 it with `SAVE_VOLUMES=1`. One trainer per card, EMPTY cards only (`scripts/gauss_sched.sh
scripts/gauss_queue.txt`, tmux session `gauss_sched`; each run holds the shared card lock `Cardiac_FreeRunning_MCGS/work/.gpulock.<card>`
so the other schedulers of this user stay off the card). Progress: `results_arc/<exp>/<exp>/joint_train.log`.

## Decision rules (pre-registered before the first candidate ran)

1. Reference rows first: `g_r0_control` (parent at 1500 it on this runner) and `g_r0_control_s43` (seed 43) give the
   seed floor per cell. No candidate is read before both exist.
2. *Keep* at 1500: min(dT1, dFLAIR) over the control up by more than the floor, no other head cell (PSNR / SSIM / HFEN,
   either contrast) down by more than its floor, neither sharpness down by more than 0.01. A gain on one contrast bought
   with the other is a *reshape*, not a keep. *keep-marginal* = PSNR up but a structural cell slips (promoted only if the
   4000-it run repairs it). Losers are not retried; a bracketing dose is queued only for a keep.
3. Round 1 stacks the keeps (pairwise, then all); Round 2 promotes the best stack at 4000 it with volumes, judged against
   the parent at 4000 (34.909 / 34.317) and against the bar; then the seed repeat.
4. The verdict's margin is over the bar as defined above (GS-single frozen). For the record only, the winner's own
   single-contrast arms are run once at the same recipe at the end (not in the main table — the user's decision on the bar).
5. Sharpness: no external gate any more (L1-ESPIRiT is out); reported, and rule 2 keeps it from collapsing.
6. Whole-volume SSIM reported, never optimised.


Rule 2 refinement, written 2026-09-16 12:27 before any candidate row existed (`scripts/gauss_judge.py` applies it):
*keep* = both contrasts up beyond their floor with no slip; *keep-one* = one contrast up beyond its floor, the other
inside its floor, no slip (usable in a stack); *reshape* = one up, the other down beyond its floor; *keep-marginal* =
PSNR up but SSIM / HFEN (0.0005) or sharpness (0.010) slips; everything else *discard*.

## Round 0 (queued 2026-09-16 11:50, 1500 it, empty cards only; order = `scripts/gauss_queue.txt`)

| exp | extra args | reading |
|---|---|---|
| g_r0_control | — | parent at 1500 (reference) |
| g_r0_control_s43 | `--set training.seed=43` | seed floor |
| g_r0_wav15 / wav20 / wav07 | wavelet T1/FLAIR ×1.5 (0.18/0.15), ×2 (0.24/0.20), ×0.7 (0.084/0.07) | the i.i.d. mask aliases like noise: is the sparsity dose the first lever? |
| g_r0_geo014 / geo007 / geo020 | `--lambda_geo` 0.14 / 0.07 / 0.20 | edge-TV weight, not re-tuned since the AF12 v2 point |
| g_r0_kh30 / kh15 | `--kspace_high` 3.0 / 1.5 | high-band data weight against the mask's outer density |
| g_r0_kc05 | `--kspace_center 0.5` | low-frequency emphasis (PSNR lever) |
| g_r0_ptv001 / ptv005 | `--lambda_ptv` 0.01 / 0.05 | phase TV dose (PSNR-vs-sharpness trade re-measured here) |
| g_r0_edge0002 | `--lambda_edge 0.002` | coupling term, upper bracket |
| g_r0_n60 / n48 | 6.0M / 4.8M points | capacity (FLAIR gained at AF16) |
| g_r0_rot008 | `--set gaussian.rotation_lr=0.008` | the AF12 phase-3 optimiser lever |
| g_r0_wu300 / wu800 | `WARMUP=300` / `800` | shared warm-up length |

Not re-opened (closed with evidence on the AF16 line): non-local RED (BM4D target), adaptive densification, vectorial TV,
contrast-private residual Gaussians, the complementary FLAIR mask. If Rounds 0–1 stall far below the bar, re-opening
non-local RED is proposed separately (i.i.d. aliasing is noise-like), not done silently.

## Round 0c and Round 1 (queued 13:03, after the second batch)

Round 1: `g_r1_rot008_n60` = rotation_lr 0.008 + 6.0M points, the two keeps stacked. Round 0c (the optimiser was the only
lever class that kept; the regulariser shapes are untested on this mask): scale_lr 0.004 / 0.001, position_lr 0.0006,
density_lr 0.03, lr gamma 0.999, edge-TV kappa_q 0.75 / 0.90, w_floor 0.05, tv_eps 0.02, wavelet 2 levels, prewhitening
(FLAIR only / both). Withdrawn as foregone after the brackets closed: wav 0.06/0.05, geo 0.05, geo 0.20.

## Incident 15:02 (mine) — tmux server killed, five screens lost and re-queued

At 15:02 I killed pid 1777714 believing it a stray `tmux new-session` client; it was the tmux SERVER (its ps line shows
the first session's command). Every tmux session died with it: the scheduler and the five trainers then running
(`g_r1_tvvec_rot008_n48` at 700/1500, `g_r0_pw` 400, `g_r0_wu300` 300, `g_r0_kren` 50, `g_r0_wfl002` just launched).
Their partial outputs are kept aside as `results_arc/<exp>_killed_tmux_1502*`; no ledger row was written for them; the
card locks were released by the wrappers' traps. Fix (commit below): the scheduler now launches every run with
`setsid nohup … &`, detached from tmux, so only the scheduler itself lives in the tmux session. All five are re-queued
unchanged and re-run from scratch. Cost: about 25 GPU-minutes; no result was affected (nothing had been read).

## Cards (16:45, user decision relayed by session claude-71)

The loop is cut to three cards: the scheduler now runs with `ALLOWED='3 4 7'`; cards 1, 2 and 5 are given up (the jobs
on 1 and 2 finish, nothing new is placed there) for the jyz AF16 noise-robustness study (worktree `mcgs_noise_wt`).

## Verdict (2026-09-16 19:20)

**Target not met; parameter space closed with evidence.** 48 screens at 1500 it, 4 promotions and 2 seed repeats at 4000 it,
2 record-only singles. Two operating points, both first on all six head cells: rsw 34.923 / 34.362 (+0.152 / +0.068 over the
frozen GS-single; the pre-registered objective's formal winner) and rswt 34.976 / 34.353 (+0.205 / +0.059; T1 +0.053 and every
structural cell better; fair coupling against its own same-recipe singles +0.135 / +0.058). Parent +0.138 / +0.023. Seed floors
at 4000 it 0.001 / 0.001 (rsw) and 0.000 / 0.001 (rswt). The missing ~0.3 / ~0.44 dB is not a parameter: every prior dose and
shape, the optimiser, capacity, schedule, warm-up (inert) and cold start were bracketed; the coupling terms in the code are at
or past their optimum. A larger joint-over-single gap needs a new coupling mechanism (code), which is the user's decision.

## Round log

Head-only, 1500 it unless noted; bar = GS-single frozen 34.771 / 34.294 (PSNR), 0.9305 / 0.9202 (SSIM), 0.1412 / 0.1295 (HFEN).

| time | exp | T1 | FLAIR | SSIM T1 / FL | HFEN T1 / FL | sharp | reading |
|---|---|---|---|---|---|---|---|
| 12:20 | g_r0_control (parent at 1500 it on this runner) | 34.875 | 34.248 | 0.9331 / 0.9203 | 0.1348 / 0.1284 | 0.797 / 0.757 | the reference row; margins over the bar +0.105 / −0.046 (at 4000 it the parent is 34.909 / 34.317, so the last 2500 iterations are worth +0.034 / +0.069) |
| 12:22 | g_r0_control_s43 (seed 43) | 34.865 | 34.240 | 0.9329 / 0.9201 | 0.1350 / 0.1286 | 0.797 / 0.758 | **seed floor 0.010 / 0.008 dB**, SSIM 0.0002, HFEN 0.0002, sharpness 0.001 |
| 12:29  g_r0_wav15 | 34.837 (-0.038) | 34.205 (-0.043) | 0.9330 / 0.9198 | 0.1381 / 0.1297 | 0.779 / 0.737 | **discard**: both contrasts down (−0.038 / −0.043), HFEN and sharpness worse — more sparsity is the wrong direction on this mask |
| 12:29  g_r0_wav20 | 34.745 (-0.130) | 34.124 (-0.124) | 0.9321 / 0.9185 | 0.1416 / 0.1314 | 0.766 / 0.720 | **discard**: −0.130 / −0.124, every cell worse; the wavelet dose curve is falling from 0.12/0.10 upward → 0.06/0.05 queued (Round 0b) |
| 12:29  g_r0_geo014 | 34.848 (-0.027) | 34.235 (-0.013) | 0.9320 / 0.9190 | 0.1378 / 0.1312 | 0.769 / 0.726 | **discard**: −0.027 / −0.013, SSIM / HFEN / sharpness all worse; edge-TV 0.14 over-smooths → 0.05 queued (Round 0b) |
| 13:00  g_r0_kh30 | 34.658 (-0.217) | 34.074 (-0.174) | 0.9275 / 0.9151 | 0.1354 / 0.1308 | 0.855 / 0.826 | **discard**: −0.217 / −0.174 and SSIM −0.005 on both, although sharper (+0.06); the high-band weight 2.2 is already at or past its optimum (1.5 queued) |
| 13:00  g_r0_rot008 | 34.908 (+0.033) | 34.297 (+0.049) | 0.9333 / 0.9208 | 0.1346 / 0.1281 | 0.797 / 0.759 | **keep** — the first: +0.033 / +0.049, every one of the six cells equal or better, sharpness unchanged. The AF12 phase-3 optimiser lever transfers; the rotation learning rate was 27× too small for this cloud |
| 13:00  g_r0_ptv001 | 34.915 (+0.040) | 34.237 (-0.011) | 0.9323 / 0.9187 | 0.1367 / 0.1321 | 0.781 / 0.737 | **reshape**: T1 +0.040 but FLAIR −0.011, SSIM / HFEN / sharpness all slip — lowering phase TV from 0.03 trades FLAIR and structure for T1 PSNR (0.05 still queued for the other end of the curve) |
| 13:00  g_r0_n60 | 34.882 (+0.007) | 34.321 (+0.073) | 0.9329 / 0.9211 | 0.1350 / 0.1283 | 0.793 / 0.755 | **keep-one**: FLAIR +0.073 (T1 +0.007, inside the floor), FLAIR SSIM +0.0008, T1 SSIM / HFEN −0.0002 (inside); as at AF16, capacity buys FLAIR, the thinner side. Stacked with rot008 as Round 1 (`g_r1_rot008_n60`) |
| 13:00  g_r0_wav07 | 34.838 (-0.037) | 34.233 (-0.015) | 0.9322 / 0.9199 | 0.1331 / 0.1280 | 0.811 / 0.774 | **discard**: −0.037 / −0.015 — with wav15 this brackets the wavelet dose: 0.12 / 0.10 is the optimum on this mask too (0.06/0.05 bracket withdrawn) |
| 13:00  g_r0_geo007 | 34.847 (-0.028) | 34.213 (-0.035) | 0.9329 / 0.9202 | 0.1344 / 0.1280 | 0.821 / 0.786 | **discard**: −0.028 / −0.035 — with geo014 this brackets edge-TV: 0.10 is the optimum (0.05 and 0.20 withdrawn as foregone) |
| 13:40  g_r0_kh15 | 34.628 (-0.247) | 33.964 (-0.284) | 0.9310 / 0.9156 | 0.1419 / 0.1324 | 0.740 / 0.685 | **discard**: −0.247 / −0.284, every cell worse — with kh30 (−0.217 / −0.174) the high-band weight 2.2 sits on a sharp optimum; a symmetric cliff on both sides says this weight acts as an effective step size, not a prior |
| 13:40  g_r0_kc05 | 34.875 (+0.000) | 34.243 (-0.005) | 0.9331 / 0.9202 | 0.1348 / 0.1286 | 0.797 / 0.757 | **null**: +0.000 / −0.005, every other cell to four decimals — the low-frequency weight does nothing here |
| 13:40  g_r0_edge0002 | 34.863 (-0.012) | 34.242 (-0.006) | 0.9326 / 0.9198 | 0.1363 / 0.1305 | 0.796 / 0.756 | **discard**: −0.012 / −0.006, SSIM / HFEN a shade worse — common-edge 0.001 stays (same plateau edge as at AF16) |
| 13:40  g_r0_ptv005 | 34.794 (-0.081) | 34.212 (-0.036) | 0.9329 / 0.9206 | 0.1355 / 0.1272 | 0.810 / 0.774 | **discard**: −0.081 / −0.036 (sharper +0.013 / +0.017) — with ptv001 (reshape) the phase-TV curve is bracketed: 0.03 is the operating point on this mask too |
| 13:40  g_r0_n48 | 34.892 (+0.017) | 34.301 (+0.053) | 0.9331 / 0.9209 | 0.1352 / 0.1283 | 0.794 / 0.756 | **keep**: +0.017 / +0.053, FLAIR SSIM +0.0006, everything else inside the floor. Capacity curve at 1500 it: T1 34.875 → 34.892 (4.8M) → 34.882 (6.0M), FLAIR 34.248 → 34.301 → 34.321: FLAIR still climbing, T1 flat. `g_r1_rot008_n48` queued beside the 6.0M stack |
| 13:40  g_r0_wavl2 | 34.778 (-0.097) | 34.228 (-0.020) | 0.9316 / 0.9193 | 0.1406 / 0.1306 | 0.786 / 0.746 | **discard**: −0.097 / −0.020, HFEN and sharpness worse — the finest-level-only wavelet stays |
| 14:00  g_r1_rot008_n60 | 34.882 (+0.007) | 34.322 (+0.074) | 0.9330 / 0.9211 | 0.1350 / 0.1283 | 0.793 / 0.755 | **keep-one, but no stacking**: +0.007 / +0.074 = n60 alone (+0.007 / +0.073) to the third decimal; rot008's +0.033 / +0.049 does not add at 6.0M. The two levers look like one mechanism (geometry freedom). Pairwise stacks of every keep queued (Round 1c) |
| 14:00  g_r0_scl004 | 34.901 (+0.026) | 34.295 (+0.047) | 0.9333 / 0.9208 | 0.1345 / 0.1280 | 0.797 / 0.758 | **keep**: +0.026 / +0.047, six cells equal or better — the second optimiser lever (scale learning rate ×2); the AF12 line's "scale_lr is the FLAIR knob" holds here |
| 14:00  g_r0_scl001 | 34.836 (-0.039) | 34.181 (-0.067) | 0.9328 / 0.9196 | 0.1353 / 0.1290 | 0.796 / 0.755 | **discard**: −0.039 / −0.067 — scale_lr brackets upward: 0.001 < 0.002 (parent) < 0.004 |
| 14:00  g_r0_pos0006 | 34.869 (-0.006) | 34.236 (-0.012) | 0.9330 / 0.9202 | 0.1351 / 0.1287 | 0.796 / 0.757 | **discard**: −0.006 / −0.012 (inside / just past the floor), structure flat — position_lr ×2 is null-to-negative |
| 14:00  g_r0_den003 | 34.860 (-0.015) | 34.219 (-0.029) | 0.9330 / 0.9200 | 0.1347 / 0.1285 | 0.797 / 0.757 | **discard**: −0.015 / −0.029 — density_lr ×2 loses |
| 14:27  g_r0_gam0999 | 34.850 (-0.025) | 34.199 (-0.049) | 0.9330 / 0.9198 | 0.1349 / 0.1286 | 0.796 / 0.756 | **discard**: −0.025 / −0.049 — a faster lr decay loses, so the schedule is a live lever in the other direction (0.9998 queued) |
| 14:27  g_r1_rot008_n48 | 34.903 (+0.028) | 34.317 (+0.069) | 0.9332 / 0.9210 | 0.1351 / 0.1283 | 0.794 / 0.756 | **keep**: +0.028 / +0.069 — better than rot008 alone (+0.033 / +0.049) on FLAIR and than n48 alone (+0.017 / +0.053) on both: at 4.8M the two levers partly add, at 6.0M they do not. 4.8M is the capacity to stack on |
| 14:27  g_r0_dtv001 | 34.875 (+0.000) | 34.276 (+0.028) | 0.9331 / 0.9203 | 0.1348 / 0.1283 | 0.797 / 0.740 | **keep-marginal**: FLAIR +0.028, T1 0.000, but FLAIR sharpness −0.017 — the directional-TV coupling buys FLAIR PSNR with smoothing; not stacked for now (a 0.005 dose would be the follow-up if the stacks stall) |
| 14:27  g_r0_tvvec | 34.905 (+0.030) | 34.236 (-0.012) | 0.9340 / 0.9207 | 0.1317 / 0.1266 | 0.806 / 0.770 | **reshape**: T1 +0.030 but FLAIR −0.012; yet SSIM (+0.0009 / +0.0004), HFEN (+0.0031 / +0.0018) and sharpness (+0.009 / +0.013) improve on BOTH contrasts — isotropic (vector) edge-TV is structurally better here, costing only FLAIR PSNR. Worth a stack with the FLAIR-friendly levers (queued after Round 1c reads) |
| 14:27  g_r0_kq075 | 34.873 (-0.002) | 34.212 (-0.036) | 0.9331 / 0.9199 | 0.1333 / 0.1279 | 0.811 / 0.777 | **discard**: FLAIR −0.036 (T1 flat, sharper) — a lower edge quantile protects too much |
| 14:27  g_r0_kq090 | 34.851 (-0.024) | 34.246 (-0.002) | 0.9325 / 0.9196 | 0.1371 / 0.1303 | 0.785 / 0.741 | **discard**: T1 −0.024, HFEN and sharpness worse — kappa_q 0.85 bracketed |
| 14:25  g_r0_wfl005 | 34.888 (+0.013) | 34.265 (+0.017) | 0.9335 / 0.9208 | 0.1342 / 0.1280 | 0.806 / 0.767 | **keep** (small): +0.013 / +0.017 with every cell a shade better and sharper (+0.009 / +0.010) — a lower residual-smoothing floor lets edges through; bracket 0.02 queued, and it goes into the stack |
| 14:55  g_r1_rot008_scl004 | 34.924 (+0.049) | 34.331 (+0.083) | 0.9334 / 0.9211 | 0.1344 / 0.1278 | 0.796 / 0.759 | **keep — the best stack**: +0.049 / +0.083, six cells better, sharpness level. The two optimiser levers ADD (rot008 +0.033 / +0.049, scl004 +0.026 / +0.047): they are independent. Promotion `g_r2_rs_full4k` (4000 it, volumes) queued at the top |
| 14:55  g_r0_gam09998 | 34.884 (+0.009) | 34.266 (+0.018) | 0.9331 / 0.9204 | 0.1349 / 0.1285 | 0.797 / 0.758 | **keep-one** (small): +0.009 / +0.018 — a slower lr decay helps a little at 1500 it and should matter more at 4000; stacked on rs as `g_r1_rs_gam` |
| 14:55  g_r1_rot008_scl004_n48 | 34.906 (+0.031) | 34.332 (+0.084) | 0.9332 / 0.9211 | 0.1350 / 0.1282 | 0.794 / 0.756 | **keep, but worse than rs alone**: +0.031 / +0.084 — capacity adds nothing to FLAIR once both optimiser levers are on, and costs T1 0.018; 3.6M stays the promotion capacity |
| 14:55  g_r0_eps002 | 34.875 (+0.000) | 34.260 (+0.012) | 0.9333 / 0.9202 | 0.1338 / 0.1280 | 0.784 / 0.743 | **keep-marginal**: FLAIR +0.012 only, sharpness −0.013 / −0.014 — a smaller Charbonnier eps smooths; not stacked |
| 14:55  g_r1_scl004_n60 | 34.883 (+0.008) | 34.324 (+0.076) | 0.9330 / 0.9211 | 0.1349 / 0.1283 | 0.793 / 0.755 | **keep-one = n60 alone** (+0.007 / +0.073): like rot008, scl004 adds nothing at 6.0M. Capacity and the optimiser levers push against the same FLAIR ceiling (about +0.08 at 1500 it); 3.6M with both lr levers is the cheapest way there |
| 14:57  g_r0_pwc | 34.877 (+0.002) | 34.204 (-0.044) | 0.9331 / 0.9192 | 0.1348 / 0.1300 | 0.797 / 0.759 | **discard**: FLAIR −0.044, SSIM / HFEN worse — coil-noise prewhitening on the FLAIR fit hurts here (the AF10 line's gain does not transfer); `g_r0_pw` (both contrasts) still running for completeness |
| 14:57 | g_r0_wu800 | — | — | — | — | — | **crash (mine)**: cycle.py guards warmup < 0.5 × iters, 800 ≥ 750; re-queued as `g_r0_wu700` |
| 14:59 | g_r0_initimp | — | — | — | — | — | **crash (mine)**: the trainer accepts init_mode head or intensity only; re-queued as `g_r0_initint` |
| 15:32  g_r1_rs_wfl005 | 34.936 (+0.061) | 34.348 (+0.100) | 0.9338 / 0.9216 | 0.1338 / 0.1275 | 0.806 / 0.769 | **keep — new best stack**: +0.061 / +0.100 over the control, every cell better than rs alone (+0.049 / +0.083) and sharper (+0.009 / +0.012): w_floor 0.05 adds on top of the two lr levers. Promotion `g_r2_rsw_full4k` queued beside the rs one |
| 15:34  g_r1_rs_tvvec | 34.966 (+0.091) | 34.325 (+0.077) | 0.9345 / 0.9216 | 0.1311 / 0.1259 | 0.806 / 0.772 | **keep**: +0.091 / +0.077 — on top of rs the vector edge-TV no longer costs FLAIR (alone it was −0.012) and gives the best structure of the loop (SSIM +0.0014 / +0.0013, HFEN +0.0037 / +0.0025, sharper +0.009 / +0.015). Full stack `g_r1_rswt` (rs + w_floor 0.05 + vector) queued |
| 15:35  g_r1_rs_gam | 34.921 (+0.046) | 34.335 (+0.087) | 0.9333 / 0.9212 | 0.1347 / 0.1280 | 0.796 / 0.759 | **null on rs at 1500 it**: +0.046 / +0.087 against rs +0.049 / +0.083 (inside the floors). The slow decay only pays after iteration ~1500 by construction, so it is carried as a promotion-budget variant (4000 it) of the final stack, not in the 1500-it stacks |
| 15:39  g_r0_pw | 34.835 (-0.040) | 34.206 (-0.042) | 0.9322 / 0.9192 | 0.1358 / 0.1300 | 0.799 / 0.759 | **discard**: −0.040 / −0.042, structure worse on both — prewhitening is closed on this data (FLAIR-only −0.044, both −0.040 / −0.042) |
| 15:40  g_r1_tvvec_rot008_n48 | 34.950 (+0.075) | 34.313 (+0.065) | 0.9342 / 0.9215 | 0.1319 / 0.1263 | 0.803 / 0.770 | **keep, but below rs + vector** (+0.091 / +0.077): with rot008 and vector on, 4.8M points is again worth nothing (T1 −0.016, FLAIR −0.012 against the 3.6M stack that has scl004 instead). Capacity is closed on this line: 3.6M |
| 16:03  g_r0_wu300 | 34.875 (+0.000) | 34.248 (+0.000) | 0.9331 / 0.9203 | 0.1348 / 0.1284 | 0.797 / 0.757 | **null by construction**: every cell equal to the control to four decimals — `shared_warmup_iters` is parsed but no longer gates anything in the v2 trainer (INF-05 note, joint_trainer.py ~906); the flag reached the run (log line 2: warmup=300). wu700 will read the same; the warm-up lever is closed as inert |
| 16:04 | g_r0_wu700 | 34.875 (+0.000) | 34.248 (+0.000) | 0.9331 / 0.9203 | 0.1348 / 0.1284 | 0.797 / 0.757 | **null by construction** (same inert flag as wu300) |
| 16:07  g_r1_rswt | 34.972 (+0.097) | 34.333 (+0.085) | 0.9349 / 0.9223 | 0.1305 / 0.1255 | 0.822 / 0.790 | **keep — best T1 and best structure of the loop**: +0.097 / +0.085, SSIM +0.0018 / +0.0020, HFEN +0.0043 / +0.0029, sharpness +0.025 / +0.033; FLAIR PSNR below rsw (+0.100), which keeps the larger min-margin (0.054 vs 0.039). Both promoted at 4000 it; rsw also with lr gamma 0.9998 |
| 16:08  g_r0_initint | 34.921 (+0.046) | 34.267 (+0.019) | 0.9334 / 0.9194 | 0.1322 / 0.1272 | 0.810 / 0.779 | **keep-marginal**: T1 +0.046, FLAIR +0.019, HFEN better on both, sharper (+0.013 / +0.022), but FLAIR SSIM −0.0009 — the intensity-weighted cold start places points better for T1; stacked on rswt (`g_r1_rswt_init`) to see whether the slip is covered |
| 16:11 | **g_r2_rs_full4k (4000 it, volumes)** | 34.916 | 34.345 | 0.9333 / 0.9213 | 0.1345 / 0.1278 | 0.795 / 0.758 | vs the parent at 4000 (34.909 / 34.317): **+0.007 / +0.028 only** — the +0.049 / +0.083 seen at 1500 it is mostly convergence speed, not a better end point (the same 1500-vs-4000 lesson as the AF12 capacity rows). Margin over the bar +0.145 / +0.052 (parent +0.138 / +0.023); six cells still first. The 1500-it screen overstates every optimiser lever; only prior-shape gains (w_floor, vector TV) can be expected to survive promotion, which rsw / rswt / rsw_gam will show |
| 16:32  g_r0_kren | 34.864 (-0.011) | 34.245 (-0.003) | 0.9328 / 0.9196 | 0.1361 / 0.1304 | 0.790 / 0.741 | **discard**: −0.011 / −0.003 with HFEN and sharpness worse — the edge threshold taken from the running render is worse than from ZF even on this noisier ZF; P6's kappa-from-ZF stays |
| 16:33  g_r0_wfl002 | 34.891 (+0.016) | 34.271 (+0.023) | 0.9336 / 0.9210 | 0.1340 / 0.1280 | 0.809 / 0.770 | **keep**: +0.016 / +0.023, a shade above 0.05 (+0.013 / +0.017) on every cell — the w_floor curve keeps rising as the floor goes to zero; 0.0 queued as the bracket end. The 0.02-vs-0.05 difference is inside what 4000 it can resolve, so the promotions stay at 0.05 |
| 16:40 | **g_r2_rsw_full4k (4000 it, volumes)** | 34.923 | 34.362 | 0.9337 / 0.9218 | 0.1339 / 0.1274 | 0.806 / 0.769 | vs the parent at 4000: **+0.014 / +0.045**, every cell better, sharper (+0.010 / +0.010); vs the rs promotion +0.007 / +0.017 — the w_floor gain (a prior-shape change) survives promotion, the lr gains largely do not. Margin over the bar **+0.152 / +0.068**; six cells first. Best 4000-it recipe so far |
| 16:59  g_r1_rswt_init | 34.983 (+0.108) | 34.303 (+0.055) | 0.9349 / 0.9210 | 0.1286 / 0.1248 | 0.834 / 0.809 | **keep, T1-shaped**: T1 +0.108 (best of the loop), sharpest (+0.037 / +0.052), but FLAIR +0.055 against rswt's +0.085 and rsw's +0.100 — the intensity cold start moves gain from FLAIR to T1. FLAIR is the binding side of the objective, so it is not promoted; recorded as the T1 operating point |
| 17:02  g_r0_wfl000 | 34.891 (+0.016) | 34.272 (+0.024) | 0.9336 / 0.9210 | 0.1340 / 0.1280 | 0.810 / 0.771 | **keep = wfl002** (+0.016 / +0.024 vs +0.016 / +0.023): the w_floor curve is flat from 0.02 down to 0 and only 0.003 / 0.007 above 0.05 — bracketed; the residual smoothing floor is simply unnecessary on this mask. Promotions keep 0.05 (inside the 4000-it floor) |
| 17:15 | **g_r2_rswt_full4k (4000 it, volumes)** | 34.976 | 34.353 | 0.9349 / 0.9225 | 0.1305 / 0.1255 | 0.821 / 0.790 | vs the parent at 4000: **+0.067 / +0.036**, best structure of the loop (SSIM +0.0016 / +0.0015, HFEN −0.0039 / −0.0023, sharpness +0.025 / +0.031); vs rsw: T1 +0.053, FLAIR −0.009 (the 4000-it floor decides whether that is a slip; rsw_s43 running). Margin over the bar **+0.205 / +0.059** (rsw +0.152 / +0.068): rsw keeps the larger min-margin by 0.009, rswt wins T1 and every structural cell |
| 17:20 | g_r2_rsw_gam_full4k (4000 it, volumes) | 34.904 | 34.353 | 0.9334 / 0.9217 | 0.1344 / 0.1279 | 0.805 / 0.767 | **discard**: −0.019 / −0.009 vs rsw, every cell a shade worse — the slower lr decay leaves the end point noisier at 4000 it; the schedule lever is closed. **The loop's recipe is rswt** (rs + w_floor 0.05 + vector edge-TV): best T1 and all four structural cells, FLAIR 0.009 under rsw; rsw stays on record as the FLAIR-maximal point. Round 3 (record only): rswt's own single-contrast arms and its seed repeat |
| 18:10 | g_r2_rsw_s43_full4k (rsw, seed 43, 4000 it) | 34.925 | 34.363 | 0.9337 / 0.9218 | 0.1339 / 0.1274 | 0.806 / 0.769 | **4000-it seed floor 0.001 / 0.001 dB**, SSIM / HFEN / sharpness 0.000. So rswt's FLAIR −0.009 against rsw is real (9× the floor): by the pre-registered objective (max min-margin) **rsw is the formal winner by 0.009 dB** (0.068 vs 0.059); rswt wins the other five cells (T1 +0.053, all structure, sharpness). Both are reported as operating points; which is quoted is the user's call |
| 18:25 | g_r3_rswt_singleT1_full4k (GS-single T1 at the rswt recipe, record only) | 34.840 | — | 0.9323 / — | 0.1364 / — | 0.820 / — | same-recipe single T1: **MC-GS − GS-single(same recipe) = +0.135 dB on T1**, SSIM +0.0026, HFEN better by 0.0058; the tuned recipe lifts the single T1 arm too (frozen 34.771 → 34.840) |
| 18:28 | g_r3_rswt_singleFLAIR_full4k (GS-single FLAIR at the rswt recipe, record only) | — | 34.295 | — / 0.9213 | — / 0.1274 | — / 0.795 | same-recipe single FLAIR: **MC-GS − GS-single(same recipe) = +0.058 dB on FLAIR**, SSIM +0.0012, HFEN better by 0.0019; the single FLAIR arm does not move with the recipe (frozen 34.294 → 34.295), so on FLAIR the frozen bar and the fair bar coincide |
| 19:18 | g_r2_rswt_s43_full4k (rswt, seed 43, 4000 it) | 34.976 | 34.352 | 0.9349 / 0.9225 | 0.1305 / 0.1255 | 0.821 / 0.790 | **rswt seed floor 0.000 / 0.001 dB**; the loop is complete — queue empty, scheduler stopped. Verdict in `docs/superpowers/records/2026-09-16-gauss-loop.md` §5 |
