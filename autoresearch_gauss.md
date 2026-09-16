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

## Round log

Head-only, 1500 it unless noted; bar = GS-single frozen 34.771 / 34.294 (PSNR), 0.9305 / 0.9202 (SSIM), 0.1412 / 0.1295 (HFEN).

| time | exp | T1 | FLAIR | SSIM T1 / FL | HFEN T1 / FL | sharp | reading |
|---|---|---|---|---|---|---|---|
