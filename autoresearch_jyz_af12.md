# autoresearch — MC-GS joint on the jiangyizhou T1 + FLAIR AF12 pair: widen the lead over every arm

_Opened 2026-09-11 19:40. Karpathy-style loop: propose one change → run at a fixed budget → read one
number → keep or discard → record → repeat. Runner `autoresearch_jyz_af12.sh`, ledger
`autoresearch_jyz_af12.tsv` (one row per experiment, append-only), branch `autoresearch-jyz-af12`
(= `autoresearch-jyz-af16` at 3a44aba, the code that produced P6), worktree
`/data0/congcong/code/GS/mcgs_af12_wt`. Cards: 6 and 7, both shared with another user's light job
(2–3x slower than a free card; user-approved), one trainer per card; card 0 belongs to the AF16
phase-2 loop (mcgs_af16_wt), cards 1–5 to another user. Move to a free card from 1–5 when one appears._

## Objective (user decision 2026-09-11)

The user marked the P6 recipe's result as the provisional best 12x reconstruction (CLAUDE.md, "Best 12x
reconstruction so far") and asked for a loop that raises MC-GS joint's quality and **widens its lead over
every comparison arm on every metric** on `Results_20260908/T1FLAIR_AF12_v3_jiangyizhou`. Arena =
jiangyizhou AF12 only (one vd-Poisson mask, AF 11.71 on T1 / 9.74 on FLAIR against their own support).
Levers use the measured data only (ZF support, acquired k-space) — no other method's output, no GT, no
learned prior. The model stays MC-GS joint: shared geometry, dual complex density, hard-DC scoring.

**The number the loop reads:** `min_margin = min(T1 head PSNR − bar_T1, FLAIR head PSNR − bar_FLAIR)`,
bar = the best non-MC-GS arm per contrast in the topic's `logs/summary.json` (row script; `--print_best`
shows it). **Six head cells** are recorded per row: per contrast the PSNR, SSIM and HFEN margins over the
best competitor of that cell (HFEN margin = competitor − ours, so every margin is "higher is better").

**Decision rules (pre-registered 2026-09-11, before the first candidate ran):**
1. *Keep* at the screening budget (1500 iterations): `min_margin` rises above the parent's at the same
   budget by more than the seed floor at that budget (`ar1_P6_i1500` vs `ar1_P6_i1500_s43`), AND none of
   the other five cells falls below the parent's by more than its own floor, AND sharpness of neither
   contrast drops more than 0.01 below the parent's. A knob that raises PSNR by trading SSIM or HFEN away
   is recorded as `reshape`, not kept.
2. *Promote*: kept knobs (singly, then the combination) re-run at 4000 iterations with `SAVE_VOLUMES=1`;
   promotion applies rule 1 against the 4000-iteration parent (`ar0_P6_parent_full4k`) and the bar. The
   promoted run becomes the new parent; the runner's baked recipe is only changed by a commit that says so.
3. *The bar moves once*, when the classical λ re-sweep on this subject lands
   (`T1FLAIR_AF12_v3_jiangyizhou/scripts/run_classical_resweep_af12.sh`: converged TV, joint TV,
   L1-ESPIRiT, λ swept against the ground truth on jiangyizhou — the same privilege this loop has;
   `summarize.py` keeps the best λ per family and contrast). Ledger rows store absolute numbers and the
   bar they were read against (note column); margins of earlier rows are re-read from the absolute columns.
4. *Two-seed average* is a separate row at 2x compute, never folded into the recipe.
5. *Single-contrast arms* (GS-single) are run for the final recipe only and reported, not optimised.
6. Whole-volume SSIM is reported, never optimised (air-noise gauge; see the topic READMEs).

**Where we start** (P6 unchanged at AF12, `ar0_P6_parent_full4k` = copy of the AF16 loop's
`ar17_P6_tf_jyz_af12_full4k`, 4000 iterations; bar = the huangyuwei-selected λ):

| head cell | T1: P6 / bar (arm) | margin | FLAIR: P6 / bar (arm) | margin |
|---|---|---|---|---|
| PSNR | 36.753 / 36.334 (L1-ESPIRiT) | +0.419 | 36.165 / 35.720 (L1-ESPIRiT) | +0.446 |
| SSIM | 0.9480 / 0.9459 (converged TV) | +0.0020 | 0.9374 / 0.9368 (L1-ESPIRiT) | +0.0006 |
| HFEN | 0.1009 / 0.1034 (L1-ESPIRiT) | +0.0025 | 0.1002 / 0.1002 (L1-ESPIRiT) | −0.0000 |

The thin cells are FLAIR head HFEN (level), FLAIR head SSIM (+0.0006) and T1 head SSIM (+0.0020); the
PSNR lead is +0.42 / +0.45 dB. Sharpness 0.857 / 0.832 (L1-ESPIRiT 0.906 / 0.809). The re-sweep can
only raise the bar.

## Parent = P6, baked into the runner

Frozen working point (edge-TV, κ_q 0.85, w_floor 0.10, Charbonnier eps 0.04, λ_geo 0.10, kspace_high
2.2) + `gaussian.init_mode=head`, `gaussian.init_scale_mode=3nn`, `loss.edge_tv_kappa_source=zf`,
`gaussian.initial_num_points=3600000`, `loss.lambda_wav=0.07` (FLAIR), `loss.lambda_wav_plain=0.10`
(T1); config `configs/t1flair_v3_af12_hyw.yaml` (named after the pairing, points at the jyz AF12 topic
data). An experiment changes it only through the extra args it passes (cycle.py flags: last occurrence
wins; `--set`: last wins). Reference rows: `ar0_P6_parent_full4k` (4000 it), `ar1_P6_i1500` (screening
baseline), `ar1_P6_i1500_s43` (seed floor at the screening budget).

## Fixed — never touched, never used for fitting or selection

Data, mask (`mask_af12_3d.h5`), ESPIRiT maps, ground truth, scorer, gauge (magnitude), the competitors'
numbers except through rule 3. FLAIR known zeros as in ar17 (the GS arms fit the measured shutter as
zeros; the comparison arms use the mask only — the documented asymmetry, unchanged here). Code changes
go in with a unit test and must leave the default path bit-identical.

## Already settled — do not re-run expecting a gain

hyw loop (`autoresearch_t1t2.md`): intensity init; densification; position_lr; T1 high-band weight
1.0 / 3.0; kappa refresh at 300; κ_q 0.75; 3-NN multiplier 1.5; wavelet replacing TV; wavelet on the
magnitude; points-per-support-voxel rule. jyz AF16 loop (`autoresearch_jyz_af16.md`): FLAIR
pre-whitening; λ_dtv under the wavelet prior (reverses sign); λ_geo 0.16/0.20 under the wavelet prior
(0.10 bracketed); FLAIR wavelet above 0.07 at AF16 (peaks); 0.9M points; FLAIR high-band 2.6; gamma
0.999. AF16 phase 2, round 1 (the parallel loop, 2026-09-11): kspace_high 3.0 −0.8 dB, frequency
curriculum −1.5 dB, 5.0M points flat at 1500; wavelet T1 0.14 / FLAIR 0.10 +0.28/+0.31 dB on the
*complex* gauge but only +0.03–0.05 on the magnitude gauge.

## Allowed and queued (single knobs against the parent; AF12 is denser than AF16, so the prior
strengths selected at AF16 are the first suspects — both directions)

| knob | how | round |
|---|---|---|
| seed floor at 1500 | `--set training.seed=43` | 0 |
| wavelet down: T1 0.07 / FLAIR 0.05 | `--set loss.lambda_wav=0.05 --set loss.lambda_wav_plain=0.07` | 1 |
| wavelet up: T1 0.14 / FLAIR 0.10 (peer's lever, low expectation on magnitude) | `--set loss.lambda_wav=0.10 --set loss.lambda_wav_plain=0.14` | 1 |
| edge-TV λ_geo 0.07 / 0.13 | `--lambda_geo` | 1–2 |
| κ_q 0.80 / 0.90 | `--edge_tv_kappa_q` | 2 |
| points 2.4M / 4.8M | `--set gaussian.initial_num_points=` | 2 |
| budget 2500 / 6000 (the budget is a knob: at AF16 T1 overfit at 4000 until the wavelet term) | ITERS | 2–3 |
| tv_eps 0.03; w_floor 0.2 | `--tv_eps`, `--edge_tv_wfloor` | 3 (low) |
| two-seed average of the best recipe | separate 2x row | after promotion |

## Budget and protocol

Screening at 1500 iterations (time on a shared card measured by `ar1_P6_i1500`), promotion at 4000 with
volumes. One trainer per card; the classical chains (10 GB) may share card 6 with nothing of ours but
the other user's job. Launch everything detached from a bash script file (`setsid nohup … &`) — the
session watchdog kills foreground background jobs. Every ledger row carries the commit and the exact
args. Waiting: poll `results_arc/<exp>_score.json`, never `pgrep -f` from the tool shell.

## Round log

### Round 0 — 2026-09-11 19:44
- Card 7: `ar1_P6_i1500` (parent, seed 42) then `ar1_P6_i1500_s43` (floor), `scripts/round0_gpu7.sh`.
- Card 6 + CPU: classical λ re-sweep (`run_classical_resweep_af12.sh`): converged TV T1 {0.0005, 0.0007,
  0.0014, 0.002} (0.001 exists), FLAIR {0.0004, 0.0005, 0.001, 0.0014} (0.0007 exists); joint TV {0.0005,
  0.0007, 0.0014, 0.002} (0.001 exists); L1-ESPIRiT T1 {0.0007, 0.001, 0.0025, 0.004} (0.0015 exists),
  FLAIR {0.003, 0.007, 0.01} (0.005 exists), BART on CPU, nice 10, 32 threads.
- 19:59 card 5 came free (clean card, ~4x the shared-card speed: the card-7 parent run logs 12.5 it/min, so a
  1500-iteration screening run takes ~2 h there). `scripts/chain_gpu5.sh` on card 5: the seed floor
  `ar1_P6_i1500_s43` first, then `ar2_wav_down`, `ar2_geo007`, `ar2_kq090`, `ar2_n48` (4.8M points). The card-6/7
  queues skip anything that already exists (the runner refuses to overwrite), so no run is duplicated.
- Classical re-sweep so far: L1-ESPIRiT T1 rises through 0.0007 → 0.001 → 0.0015 (35.226 → 35.917 → 36.334);
  converged TV T1 0.0005 = 35.970 < 0.001's 36.144. Weaker regularisation does not help either arm on T1.
- 20:10 rule 3 sharpened before any candidate row exists: the bar per cell is the best value over every
  non-MC-GS row of the table AND every λ of the three swept families (not only the head-PSNR-best λ that
  `summarize.py` keeps per family). A competitor tuned for SSIM or HFEN therefore counts for those cells.
  With L1-ESPIRiT T1 λ 0.0025 = 36.374 / 0.9473 / 0.1056 (0.0015: 36.334 / 0.9443 / 0.1034) the T1 bar is
  now PSNR 36.374 (L1@0.0025), SSIM 0.9473 (L1@0.0025), HFEN 0.1034 (L1@0.0015).
- 20:07 cards 1–4 came free (the other user's jobs ended). Proposed split with the AF16 phase-2 loop: it keeps
  0 and takes 1–2; this loop takes 3, 4 and 5. The card-7 parent run (150/1500 after 12 min) and the card-6/7
  round-1 waiters were stopped (runner killed before the trainer, no ledger row, partial outputs removed) and
  the same experiments relaunched on clean cards: card 3 `ar1_P6_i1500` → `ar2_wav_down` → `ar2_geo007`;
  card 4 `ar2_wav_up` → `ar2_geo013` → `ar2_kq090` → `ar2_n48`; card 5 after the seed floor (flag-gated in
  case the AF16 loop still wants the card) `ar2_n24` → `ar3_i2500` (budget knob) → `ar2_eps003` → `ar2_wfloor02`.
  Card 6 keeps only the classical GPU chain; card 7 is released.

### Round 0 read — 20:36
Parent P6 at 1500 iterations (`ar1_P6_i1500`, card 3, 26 min): head PSNR 36.704 / 36.072, SSIM 0.9480 / 0.9367,
HFEN 0.1012 / 0.1008, sharp 0.854 / 0.830. Seed 43 (`ar1_P6_i1500_s43`, card 5, 26 min): 36.698 / 36.068.
**Seed floor at the screening budget: PSNR 0.006 / 0.004 dB; SSIM 0.0000 / 0.0001; HFEN 0.0001 / 0.0002;
sharp 0.001 / 0.001.** Rule 1 threshold for a keep: `min_margin` up by more than 0.006 dB over the parent's
+0.329 (i.e. above +0.335 on the current bar), no other cell down by more than its floor, sharpness not
below 0.844 / 0.820. 1500 iterations already sit 0.05 / 0.10 dB under the 4000-iteration P6, so the
screening budget is sensitive enough.

### Round 1 — candidates at 1500 iterations against `ar1_P6_i1500` (36.704 / 36.072; floor 0.006 / 0.004)
- `ar2_wav_up` (wavelet T1 0.14 / FLAIR 0.10, the AF16 phase-2 lever; card 4, 26 min): 36.708 / 36.101 —
  T1 +0.004 (inside the floor), FLAIR +0.029; but FLAIR head HFEN 0.1038 vs 0.1008 (worse by 0.0030, 15x its
  floor) and FLAIR SSIM +0.0008. `min_margin` +0.334 vs +0.329: the rise (+0.005) is under the floor and a cell
  regresses → **discard (reshape)**: the stronger wavelet buys FLAIR PSNR with FLAIR high-frequency fidelity, as
  the peer's magnitude-gauge caveat predicted (+0.03–0.05 there).
  Full six cells (view script): T1 HFEN margin −0.0032 vs the parent's +0.0022 (worse by 0.0054), sharpness
  0.834 / 0.808 vs 0.855 / 0.829 (−0.021, over the 0.01 allowance). Three gates fail, not one.
