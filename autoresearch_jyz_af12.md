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
- 20:44 the L1-ESPIRiT T1 λ curve is complete (100 FISTA, BART): 0.0007 35.226, 0.001 35.917, 0.0015 36.334,
  **0.002 36.426**, 0.0025 36.374, 0.003 36.255, 0.004 35.934 (head PSNR). Best SSIM 0.9473 at 0.0025, best
  HFEN 0.1034 at 0.0015. T1 bar now: PSNR 36.426 (L1@0.002), SSIM 0.9473, HFEN 0.1034. The converged-TV T1
  curve peaks at 0.0007 = 36.162 (0.0005 35.970, 0.001 36.144, 0.0014 35.931, 0.002 35.501), below L1.
- `ar2_n24` (2.4M points instead of 3.6M; card 5, 23 min): 36.557 / 35.821 — **−0.147 / −0.251 dB** under the
  parent, FLAIR SSIM −0.0021, sharpness 0.849 / 0.820. **Discard.** Capacity is not a free knob downward at AF12
  either; the 4.8M run (`ar2_n48`, card 4, queued) tests the other direction.
- 20:50 L1-ESPIRiT FLAIR refinement: λ 0.004 = 35.757 / 0.9368 / 0.1000 beats 0.005 (35.720 / 0.9368 / 0.1002).
  FLAIR bar now: PSNR 35.757 (L1@0.004), SSIM 0.9368, HFEN 0.1000. Parent margins at 1500 iterations on the
  current bar: T1 +0.278 / FLAIR +0.315.
- 20:56 the L1-ESPIRiT FLAIR λ curve is complete: 0.003 35.689, **0.004 35.757**, 0.005 35.720, 0.006 35.630,
  0.007 35.513, 0.01 35.095; SSIM best 0.9368 (0.004 and 0.005), HFEN best 0.1000 (0.004). L1-ESPIRiT re-sweep
  finished on both contrasts; converged TV FLAIR and joint TV still running on card 6.
- `ar2_wav_down` (wavelet T1 0.07 / FLAIR 0.05; card 3, 26 min): 36.550 / 35.960 — **−0.154 / −0.112 dB** under the
  parent, head SSIM −0.0025 / −0.0019, sharpness 0.860 / 0.831 (+0.005 / +0.002). **Discard.** With `ar2_wav_up`
  this brackets the wavelet strengths at AF12: weaker loses PSNR and SSIM, stronger buys FLAIR PSNR with HFEN and
  sharpness; P6's 0.10 / 0.07 sits at the optimum for this budget. Round-2 option (low priority): FLAIR-only
  0.085, since the two channels' strengths were independent at AF16.
- `ar2_geo013` (edge-TV λ_geo 0.13; card 4, 26 min): 36.697 / 36.061 — −0.007 / −0.011 dB (at the floor), head
  HFEN worse by 0.0030 / 0.0031, sharpness 0.829 / 0.799 (−0.026 / −0.030). **Discard**: more edge-TV smooths
  without buying PSNR. `ar2_geo007` (card 3) tests the other direction.

### Round 2 queue — pre-registered 21:05, before round 1 finished (knobs the previous loops never touched)
Every one a single change against P6 at 1500 iterations; waiters start them when the round-1 chain on that card ends.

| knob | how | card | why |
|---|---|---|---|
| wavelet depth 4 / 2 levels (default 3) | `--set loss.wav_levels=4` / `=2` | 3 / 4 | the prior's scale range; never varied |
| k-space centre weight 0.5 / 0.15 (default 0.3) | `--kspace_center` | 3 / 4 | low-band DC emphasis; the hyw loop only varied the high band |
| shared warm-up 1000 / 250 (default 500) | `--warmup` | 3 / 5 | schedule; never varied |
| FLAIR-only wavelet 0.085 (T1 stays 0.10) | `--set loss.lambda_wav=0.085` | 4 | between the bracketed 0.07 and 0.10 on the channel that gained in `ar2_wav_up` |
| wavelet family db4 (default db2) | `--set loss.wav_name=db4` | 5 | smoother basis; never varied |
| k-space mid weight 0.8 (default 1.0) | `--kspace_mid` | 5 | band balance |
- `ar2_geo007` (edge-TV λ_geo 0.07; card 3, 25 min): 36.638 / 36.013 — **−0.066 / −0.059 dB** under the parent,
  head SSIM −0.0007 / −0.0007, head HFEN better by 0.0022 / 0.0018, sharpness up. **Discard (reshape)**: less
  edge-TV sharpens but loses PSNR and SSIM. With `ar2_geo013`, edge-TV is bracketed at P6's 0.10 on this budget.
- `ar2_kq090` (κ_q 0.90; card 4, 26 min): 36.714 / 36.078 — +0.010 / +0.006 dB over the parent (`min_margin`
  +0.011, just over the 0.006 floor), but head HFEN worse by 0.0022 / 0.0027 and head SSIM −0.0002 / −0.0006,
  sharpness down. **Discard (reshape)**: the same smoothing trade the AF16 loop saw for κ_q 0.90.
- `ar3_i2500` (P6 at 2500 iterations; card 5, 40 min): 36.734 / 36.134 — +0.030 / +0.062 over 1500, −0.019 /
  −0.031 under 4000; SSIM and HFEN track the parent. The budget curve 1500 → 2500 → 4000 is still rising with
  diminishing steps, so **`ar3_i6000`** (P6 at 6000 iterations, volumes saved) is queued on card 5 ahead of its
  round-2 knobs: if it beats the 4000-iteration parent by more than the floor with no cell regressing, the
  promotion budget becomes 6000 and that run is the new parent (rule 2).
- 21:42 converged-TV FLAIR curve complete: 0.0004 35.462, 0.0005 35.575, **0.0007 35.581**, 0.001 35.400,
  0.0014 35.075 — the huangyuwei-selected 0.0007 was already the peak; TV stays 0.18 dB under L1-ESPIRiT on
  FLAIR and 0.26 dB on T1. Joint TV (4 λ) is the last family still running on card 6.

### Round 2 read
- `ar4_wavlev4` (wavelet 4 levels; card 3, 25 min): 36.699 / 36.070 — −0.005 / −0.002 dB, every cell inside its
  floor. **Discard (null)**: the fourth decomposition level adds nothing; the prior's work is in the fine scales.
- `ar2_n48` (4.8M points; card 4, 28 min, 33.1 GB): 36.736 / 36.157 — **+0.032 / +0.085 dB** over the parent
  (`min_margin` +0.310, 5x the floor), FLAIR SSIM +0.0005; but T1 SSIM −0.0003, head HFEN worse by 0.0010 (T1) /
  0.0004 (FLAIR) — above their 0.0001–0.0002 floors, an order of magnitude smaller than the reshapes above.
  Recorded **keep-marginal**: the promotion run decides (`ar5_n48_full4k`, 4000 iterations with volumes, card 3
  right after the current run), judged by rule 2 against the 4000-iteration parent. Capacity is the first lever
  with a real PSNR gain at AF12; `ar2_n60` (6.0M) is queued on card 4 after its round-2 chain.
- `ar2_eps003` (Charbonnier eps 0.03; card 5, 26 min): 36.701 / 36.071 — −0.003 / −0.001 dB, HFEN −0.0010 both.
  **Discard (null)**, as on hyw.
- `ar4_kcenter05` (k-space centre weight 0.5; card 3, 25 min): 36.702 / 36.065 — −0.002 / −0.007 dB, other cells
  inside their floors. **Discard (null)**: with hard data consistency the low-band weight barely matters.
- `ar4_wavlev2` (wavelet 2 levels; card 4, 26 min): 36.719 / 36.076 — **+0.015 / +0.004 dB** (`min_margin` +0.293,
  2.5x the floor), head SSIM +0.0002 / +0.0001, head HFEN better by 0.0010 / 0.0005, sharpness +0.002 / +0.001.
  **Keep — every cell holds or improves.** Restricting the L1 wavelet prior to the two finest scales keeps the
  fine-detail sparsity and drops the coarse-scale penalty that was slightly biasing the anatomy.
  Promotion `ar5_wavlev2_full4k` (4000 it, volumes) queued on card 4 after its current run; the combination with
  the other keep, `ar5_n48_wavlev2_full4k` (4.8M + 2 levels, 4000 it, volumes), on card 3 after the 4.8M promotion;
  `ar4_wavlev1` (1 level) as the follow-up single knob on card 4.
- `ar2_wfloor02` (edge-TV w_floor 0.20; card 5, 26 min): 36.681 / 36.025 — −0.023 / −0.047 dB, HFEN worse by
  0.0028 / 0.0031, sharpness −0.023 / −0.024. **Discard.**
- `ar4_kcenter015` (k-space centre weight 0.15; card 4, 26 min): 36.703 / 36.074 — −0.001 / +0.002 dB, every cell
  inside its floor. **Discard (null)**; with 0.5 also null, the centre weight is settled (does nothing under hard DC).

### Promotion 1 — 4.8M points at 4000 iterations (23:25): not promoted
`ar5_n48_full4k` (card 3, 67 min, 33.1 GB): 36.739 / 36.196 against the parent's 36.753 / 36.165 — **T1 −0.014,
FLAIR +0.031**; T1 head SSIM −0.0006, T1 head HFEN worse by 0.0013, FLAIR head HFEN worse by 0.0006, sharpness
level. `min_margin` (set by T1) falls from +0.327 to +0.314. **Discard (reshape at 4000)**: at the full budget the
extra capacity moves quality from T1 to FLAIR, the AF16 pattern (T1 over-fits the sampled k-space when capacity
grows; the wavelet term only partly holds it). The 1500-iteration screen missed this because T1's over-fit sets in
late — capacity decisions need the promotion budget.
Consequences: the combination `ar5_n48_wavlev2_full4k` was cancelled at start (rule 2 combines kept knobs only;
partial outputs removed, no ledger row). Card 3 now runs `ar4_warmup1000` then **`ar5_n42_full4k`** (4.2M at 4000,
volumes): the capacity curve at the promotion budget between 3.6M and 4.8M, pre-registered keep rule = T1 not below
the parent beyond the floor, FLAIR above it, no cell regressing.
- `ar4_warmup1000` crashed at start: cycle.py guards `warmup < 0.5 * iters`, so 1000 is illegal at 1500 iterations
  (ledger row `crash`, nothing trained). Re-queued as `ar4_warmup700` on card 3 after `ar5_n42_full4k`.

### Promotion 2 — wavelet prior on 2 levels at 4000 iterations (23:55): **promoted, P7**
`ar5_wavlev2_full4k` (card 4, 61 min, volumes saved): **36.765 / 36.168** against the parent's 36.753 / 36.165
(+0.012 / +0.003), head SSIM +0.0002 / +0.0001, head HFEN better by 0.0010 / 0.0005, sharpness level. Every cell
holds or improves at the promotion budget too. Six-cell margins on the current bar: PSNR +0.339 / +0.411, SSIM
+0.0009 / +0.0007, HFEN +0.0035 / **+0.0003** — the FLAIR HFEN cell, level with L1-ESPIRiT under P6, is now ahead.
**P7 = P6 + `loss.wav_levels=2`** is the parent from here: the runner bakes it (this commit); every experiment
launched after this commit sits on P7 (its `args` column shows `--set loss.wav_levels=2` from the runner), and
the screening baseline is re-measured as `ar1_P7_i1500` (card 5, right after the 6000-iteration run). Runs already
in flight (`ar5_n42_full4k` on card 3) are P6-based and are read against P6's rows. The seed floor measured on P6
(0.006 / 0.004 dB at 1500) is carried over.
- `ar3_i6000` (P6 at 6000 iterations; card 5, 89 min): 36.760 / 36.179 — +0.007 / +0.014 over 4000, HFEN better
  by 0.0004 / 0.0002, SSIM level. The budget curve is still rising at 6000 but the step is at the floor on T1 and
  costs 1.5x. **Decision (pre-registered now): the promotion budget stays 4000; the final recipe is run once at 6000
  as a 1.5x-compute row, like the two-seed average, not folded into screening.**
- `ar4_wavC085` (FLAIR-only wavelet 0.085, P6-based — launched before the P7 commit; card 4, 26 min): 36.702 /
  36.103 against P6@1500 36.704 / 36.072 — T1 null, FLAIR +0.031 with FLAIR head HFEN worse by 0.0012 and FLAIR
  SSIM +0.0006. `min_margin` unchanged (T1 binds). **Discard (reshape)**: the FLAIR wavelet curve at AF12 is the
  same trade at every strength above 0.07 — PSNR for high-frequency fidelity — and FLAIR is not the binding channel.

### Round 3 queue — pre-registered 00:20, on P7, T1 is the binding channel
| knob | how | card | why |
|---|---|---|---|
| T1 wavelet 0.12 (FLAIR stays 0.07) | `--set loss.lambda_wav_plain=0.12` | 4 | the 2-level prior is weaker overall; T1 may take a little more strength without the 0.14 reshape |
| soft L1 on the wavelet coefficients, eps 1e-3 | `--set loss.wav_eps=0.001` | 4 | pure L1 shrinks every small coefficient; a Charbonnier floor spares them |
| density lr 0.02 / 0.01 (default 0.015) | `--set gaussian.density_lr=` | 4 / 3 | the hyw loop settled position_lr only |
| scale lr 0.003 (default 0.002) | `--set gaussian.scale_lr=0.003` | 3 | never varied |
| init head fraction 0.03 (default 0.05) | `--set gaussian.init_head_frac=0.03` | 3 | a wider initial support; never varied |
| P7 seed floor | `ar1_P7_i1500_s43` | 5 | after the card-5 round-2 knobs |
Still in flight from earlier rounds: `ar5_n42_full4k` (card 3, P6-based, 4000 it), `ar4_wavlev1` → `ar2_n60`
(card 4, P7-based), `ar1_P7_i1500` → `ar4_db4` → `ar4_warmup250` → `ar4_kmid08` (card 5, P7-based).

### Round 3 baseline — 00:42
`ar1_P7_i1500` (card 5, 26 min): 36.719 / 36.075, SSIM 0.9482 / 0.9368, HFEN 0.1001 / 0.1003, sharp 0.857 / 0.830 —
reproduces `ar4_wavlev2` (36.719 / 36.076) to 0.001 dB. Round-3 candidates are read against this row with the P6
floor (0.006 / 0.004 dB); `ar1_P7_i1500_s43` (queued on card 5) re-measures the floor on P7.
- `ar5_n42_full4k` (4.2M points, P6-based, 4000 it; card 3, 64 min): 36.749 / 36.188 — T1 −0.004 (inside the floor),
  FLAIR +0.023; T1 head HFEN worse by 0.0007, T1 SSIM −0.0003. `min_margin` does not rise. **Discard (reshape).**
  Capacity curve at the promotion budget: 3.6M 36.753 / 36.165 → 4.2M 36.749 / 36.188 → 4.8M 36.739 / 36.196 —
  T1 falls monotonically as FLAIR rises. **Capacity is settled at 3.6M** while T1 is the binding channel; `ar2_n60`
  (1500 it, card 4) still runs as the last point of the screening curve but cannot be promoted on this evidence.
- `ar4_wavlev1` (wavelet prior on 1 level only, on P7; card 4, 25 min): 36.756 / 36.061 against P7@1500 36.719 /
  36.075 — **T1 +0.037**, FLAIR −0.014; T1 SSIM +0.0003, FLAIR SSIM level, head HFEN better by **0.0049 (T1) /
  0.0024 (FLAIR)**. `min_margin` rises from +0.293 to +0.304 (now set by FLAIR), but FLAIR's PSNR cell falls
  beyond its floor → **keep-marginal**, promotion decides: `ar5_wavlev1_full4k` (4000 it, volumes) on card 3 after
  the warm-up run. The finest scale alone carries the T1 gain and the HFEN gain on both channels; FLAIR gives back
  0.014 dB of PSNR. Pre-registered follow-up on card 4: `ar6_wavlev1_wavC085` (1 level + FLAIR wavelet 0.085) to
  see whether FLAIR's PSNR recovers while its HFEN stays ahead of the parent.
- `ar4_db4` (wavelet family db4 instead of db2, on P7; card 5, 28 min): **36.767 / 36.095** against P7@1500 36.719 /
  36.075 — **+0.048 / +0.020 dB**, head SSIM +0.0005 / +0.0004, FLAIR head HFEN better by 0.0007, T1 head HFEN
  −0.0002 (at the floor), sharpness up. `min_margin` +0.337 vs +0.293. **Keep** — the largest single gain so far:
  the longer db4 filters (8 taps, 4 vanishing moments) sparsify the anatomy better than db2's 4 taps.
  Promotion `ar5_db4_full4k` (4000 it, volumes) on card 5 right after the running warm-up knob. Pre-registered
  follow-ups on card 4 after the 6.0M point: `ar6_db4_lev1` (db4 + 1 level, the two T1 levers together),
  `ar6_db6`, `ar6_sym4` (family curve), then the rest of round 3.
- `ar4_warmup700` (shared warm-up 700 instead of 500, on P7; card 3, 25 min): 36.719 / 36.076 — identical to the
  baseline to 0.001 dB in every cell. **Discard (null)**; the warm-up length does nothing in this range.
- `ar2_n60` (6.0M points, on P7; card 4, 30 min, 33.4 GB): 36.714 / 36.188 — T1 −0.005, FLAIR +0.113; T1 head SSIM
  −0.0013, T1 head HFEN worse by 0.0022. **Discard (reshape)**: the screening curve 3.6M → 4.8M → 6.0M moves FLAIR
  up and T1 down already at 1500 iterations. Capacity stays 3.6M (settled at both budgets).
- `ar4_warmup250` (shared warm-up 250, on P7; card 5, 26 min): 36.719 / 36.076 — identical to the baseline.
  **Discard (null)**; warm-up 250 / 500 / 700 all read the same, the knob is settled.

### The bar (final) — classical λ re-sweep complete, 01:32
Joint TV: 0.0005 35.584 / 35.383, 0.0007 35.936 / 35.606, **0.001 36.070 / 35.624**, 0.0014 35.986 / 35.464,
0.002 lower — peak at the huangyuwei λ. Table rows now carry the best λ per family (`summarize.py`, mirrored):
L1-ESPIRiT T1 λ 0.002 (36.426), FLAIR λ 0.004 (35.757); converged TV 0.0007 / 0.0007; joint TV 0.001. Bar per
cell (best over every swept λ): T1 PSNR 36.426 / SSIM 0.9473 / HFEN 0.1034 — all L1-ESPIRiT; FLAIR PSNR 35.757 /
SSIM 0.9368 / HFEN 0.1000 — all L1-ESPIRiT. Card 6 is released (nothing of this loop runs there any more).
- `ar6_db4_lev1` (db4 + 1 level, on P7; card 4, 27 min): **36.812 / 36.080** against P7@1500 36.719 / 36.075 —
  **T1 +0.093**, FLAIR +0.005; head SSIM +0.0008 / +0.0003, head HFEN better by **0.0050 / 0.0030**, sharpness up.
  `min_margin` +0.323 (set by FLAIR). **Keep — every cell improves**, and the two T1 levers add up (db4 alone
  +0.048, 1 level alone +0.037; together +0.093 on T1 while FLAIR returns to baseline instead of −0.014).
  Promotion `ar5_db4_lev1_full4k` (4000 it, volumes) on card 4 after the running 1-level + FLAIR-0.085 screen; the
  single-knob promotions `ar5_wavlev1_full4k` (card 3) and `ar5_db4_full4k` (card 5) keep running as the
  single-lever references at 4000.

### Promotion 3 — wavelet on 1 level at 4000 iterations (02:15): not promoted on its own
`ar5_wavlev1_full4k` (card 3, 59 min, volumes): 36.791 / 36.146 against P7's 36.765 / 36.168 — **T1 +0.026,
FLAIR −0.022**; head HFEN better by 0.0047 / 0.0022, SSIM level, sharpness up. `min_margin` rises to +0.365 (T1 now
the smaller margin) but FLAIR's PSNR cell falls beyond the floor → by rule 1 a **reshape at 4000**, not a promotion.
The decision passes to the combination `ar5_db4_lev1_full4k` (card 4, ~03:10), which at 1500 iterations held FLAIR
at the baseline while keeping the T1 gain.
- `ar6_wavlev1_wavC085` (1 level + FLAIR wavelet 0.085, on P7; card 4, 26 min): 36.757 / 36.118 against P7@1500
  36.719 / 36.075 — **T1 +0.038, FLAIR +0.043**; head SSIM +0.0003 / +0.0010, head HFEN better by 0.0048 / 0.0029,
  sharpness up. **Keep — every cell improves.** Under the 1-level prior the FLAIR strength 0.085 no longer costs
  HFEN (it did with 3 levels: `ar4_wavC085`), so FLAIR gets PSNR and fidelity together. Pre-registered three-way
  combination `ar6_db4_lev1_C085` (db4 + 1 level + FLAIR 0.085) queued on card 3 after the running density-lr knob.

### Promotion 4 — db4 wavelet family at 4000 iterations (02:25): **promoted, P8**
`ar5_db4_full4k` (card 5, 64 min, volumes): **36.819 / 36.189** against P7's 36.765 / 36.168 — **+0.054 / +0.021**,
head SSIM +0.0005 / +0.0004, head HFEN better by 0.0001 / 0.0008, sharpness level. Every cell holds or improves.
Six-cell margins on the bar: PSNR **+0.394 / +0.432**, SSIM +0.0014 / +0.0011, HFEN +0.0036 / +0.0011.
**P8 = P7 + `loss.wav_name=db4`** is the parent from here (baked into the runner by this commit). Its screening
baseline is `ar4_db4` (36.767 / 36.095, the same recipe run as a candidate on P7). Runs launched before this
commit (`ar6_density_lr01` on card 3, `ar5_db4_lev1_full4k` on card 4 — which already carries db4 explicitly) are
read against the rows they were launched on; the combination's promotion is judged against P8's 4000 row.
Against P6 (the user's provisional best): +0.066 / +0.024 dB head PSNR, HFEN better by 0.0011 / 0.0013 — two
promotions in, both from the shape of the wavelet prior.
- `ar6_density_lr01` (density lr 0.01, P7-based; card 3, 25 min): 36.716 / 36.068 against P7@1500 36.719 / 36.075 —
  −0.003 / −0.007, HFEN +0.0003 / +0.0001, SSIM level. **Discard (null)**; the 0.02 point (card 4) completes the pair.
- `ar4_kmid08` (k-space mid-band weight 0.8, P8-based; card 5, 26 min): 36.612 / 36.034 against P8@1500 (`ar4_db4`)
  36.767 / 36.095 — **−0.155 / −0.061**, head HFEN worse by 0.010 / 0.009, SSIM down. **Discard**: the mid band
  carries the detail the prior cannot supply; down-weighting it costs fidelity everywhere. Band weights settled.
- `ar6_db4_lev1_C085` (db4 + 1 level + FLAIR wavelet 0.085; card 3, 26 min): **36.812 / 36.135** against P8@1500
  (`ar4_db4`) 36.767 / 36.095 — **+0.045 / +0.040**, head SSIM +0.0003 / +0.0010, head HFEN better by 0.0052 /
  0.0028, sharpness up. **Keep — every cell improves**, and against `ar6_db4_lev1` it holds T1 and adds +0.055 on
  FLAIR with better FLAIR HFEN. Best screening row so far (`min_margin` +0.378). Promotion
  `ar5_db4_lev1_C085_full4k` (4000 it, volumes) queued on card 3 after the running scale-lr knob.

### Promotion 5 — db4 + 1 level at 4000 iterations (03:10): not promoted on its own
`ar5_db4_lev1_full4k` (card 4, 63 min, volumes): 36.853 / 36.165 against P8's 36.819 / 36.189 — **T1 +0.034,
FLAIR −0.024**; head HFEN better by 0.0049 / 0.0019, T1 SSIM +0.0002, FLAIR SSIM −0.0002, sharpness up.
`min_margin` rises to +0.408 (FLAIR now the smaller margin), but FLAIR's PSNR cell falls beyond the floor →
**reshape at 4000**, as with 1 level alone on P7 (−0.022). The 1-level prior consistently gives FLAIR back ~0.02 dB
at the full budget; the three-way run with FLAIR 0.085 (`ar5_db4_lev1_C085_full4k`, card 3, ~04:30) tests whether
the FLAIR strength buys that back without the HFEN cost it had under 3 levels.
- `ar1_P8_i1500_s43` (card 5, 27 min): 36.761 / 36.093 against seed 42's 36.767 / 36.095 — **P8 seed floor 0.006 /
  0.002 dB**, SSIM 0.0000 / 0.0000, HFEN 0.0001 / 0.0001; the P6 floor carries over unchanged.

### Round 4 queue — pre-registered 03:30, single steps beyond the best screening row (db4 + 1 level + FLAIR 0.085)
| knob | how | card |
|---|---|---|
| FLAIR wavelet 0.10 under the 1-level prior (0.085 → 0.10) | `ar6_db4_lev1_C10` | 5 |
| T1 wavelet 0.12 on the three-way row (the 1-level prior has fewer coefficients, T1 may take more strength) | `ar6_db4_lev1_C085_T112` | 5 |
- `ar6_scale_lr003` (scale lr 0.003 instead of 0.002, P8-based; card 3, 27 min): **36.793 / 36.141** against P8@1500
  36.767 / 36.095 — **+0.026 / +0.046**, head SSIM +0.0001 / +0.0004, head HFEN +0.0001 / +0.0002, sharpness level.
  **Keep — every cell holds or improves.** An optimiser lever orthogonal to the prior's shape: the Gaussians' scales
  adapt faster. Follow-ups pre-registered (card 4 after db6): `ar6_slr005` (0.005) and
  `ar6_db4_lev1_C085_slr003` (the best three-way row + scale lr 0.003); promotion of scale lr 0.003 alone
  (`ar5_slr003_full4k`, 4000 it, volumes) on card 5 after its round-4 steps.
- `ar6_db4_lev1_C10` (db4 + 1 level + FLAIR wavelet 0.10; card 5, 27 min): 36.812 / **36.160** — against the 0.085
  row T1 level, FLAIR +0.025, FLAIR SSIM +0.0006, FLAIR HFEN +0.0002; against P8@1500 +0.045 / +0.065 with every
  cell up. **Keep.** The FLAIR strength curve under the 1-level prior: 0.07 36.080, 0.085 36.135, 0.10 36.160 —
  still rising with shrinking steps; `ar6_db4_lev1_C12` queued (card 5), then the promotion of the 0.10 row
  (`ar5_db4_lev1_C10_full4k`) and of scale lr 0.003.
- `ar6_db6` (wavelet family db6, on P8's 2 levels; card 4, 32 min): 36.750 / 36.090 against P8@1500 (db4) 36.767 /
  36.095 — −0.017 / −0.005, T1 head HFEN worse by 0.0012, slower (12-tap filters). **Discard**: the family curve
  db2 36.719 → db4 36.767 → db6 36.750 peaks at db4; `ar6_sym4` (same length as db4, symmetric) still queued.
- `ar6_db4_lev1_C085_T112` (three-way row + T1 wavelet 0.12; card 5, 27 min): **36.885 / 36.134** — against the
  0.085 row **T1 +0.073**, FLAIR level, T1 head SSIM +0.0012, HFEN level, sharpness level. **Keep.** Under the
  1-level db4 prior the T1 channel takes more strength cleanly (with 3-level db2, 0.14 was a reshape). Card 4's
  queue re-cut (pre-registered): `ar6_best_C10_T112` (db4 + 1 level + FLAIR 0.10 + T1 0.12, the union of the kept
  steps), `ar6_best_C10_T114` (T1 0.14), `ar6_best_C10_T112_slr003` (scale lr 0.003 on top), then sym4, wav_eps,
  density lr 0.02. `ar6_wavT1_012` on P8 is dropped (superseded by the T1-0.12 result on the 1-level row).
- `ar6_slr005` (scale lr 0.005, P8-based; card 4, 26 min): **36.812 / 36.183** against P8@1500 36.767 / 36.095 —
  **+0.045 / +0.088**, head SSIM level / +0.0007, head HFEN +0.0001 / +0.0004, sharpness level. **Keep**, above the
  0.003 point (+0.026 / +0.046): the scale-lr curve 0.002 → 0.003 → 0.005 is still rising. Queues re-cut: `ar6_slr008`
  next on card 4; the union-row combination uses 0.005 (`ar6_best_C10_T112_slr005`); the promotion on card 5 is
  `ar5_slr005_full4k` instead of 0.003.

### Promotion 6 — db4 + 1 level + FLAIR wavelet 0.085 at 4000 iterations (04:20): **promoted, P9**
`ar5_db4_lev1_C085_full4k` (card 3, 62 min, volumes): **36.853 / 36.234** against P8's 36.819 / 36.189 — **+0.034 /
+0.045**, head SSIM +0.0002 / +0.0011, head HFEN better by **0.0048 / 0.0028**, sharpness up. Every cell improves;
the FLAIR strength 0.085 more than pays back what the 1-level prior alone gave away on FLAIR (−0.024).
Six-cell margins on the bar: PSNR **+0.427 / +0.477**, SSIM +0.0016 / +0.0022, HFEN +0.0084 / +0.0039.
**P9 = P8 + `loss.wav_levels=1` + `loss.lambda_wav=0.085`** (T1 keeps `lambda_wav_plain` 0.10) is the parent from
here (baked by this commit); screening baseline `ar6_db4_lev1_C085` (36.812 / 36.135). Against the user's P6:
**+0.100 / +0.069 dB** head PSNR, HFEN better by 0.0059 / 0.0041, SSIM +0.0009 / +0.0016.
Queued and still valid on P9: `ar5_db4_lev1_C10_full4k` (FLAIR 0.10, card 5) and the union row with T1 0.12 /
scale lr 0.005 (card 4) — each is one or two steps beyond P9 and is read against it.
- `ar6_db4_lev1_C12` (FLAIR wavelet 0.12 under the 1-level db4 prior; card 5, 26 min): 36.810 / 36.163 — against
  the 0.10 row FLAIR +0.003 (at the floor), FLAIR SSIM +0.0004, FLAIR HFEN −0.0002. **Plateau**: the FLAIR curve
  0.085 36.135 → 0.10 36.160 → 0.12 36.163 has flattened; 0.10 is taken as the FLAIR strength (its promotion is
  running on card 5), 0.12 is not pursued.
- `ar6_best_C10_T112` (db4 + 1 level + FLAIR 0.10 + T1 0.12 = P9 + two steps; card 4, 26 min): **36.885 / 36.159** —
  T1 equals the T1-0.12 row, FLAIR equals the FLAIR-0.10 row: the two strengths are independent, as at AF16.
  Against P9@1500 36.812 / 36.135: +0.073 / +0.024, head SSIM +0.0012 / +0.0006, HFEN level. **Keep.** Promotion of
  the union (`ar5_best_C10_T112_full4k`, 4000 it, volumes) replaces the T1-only promotion on card 3; the FLAIR-only
  promotion (`ar5_db4_lev1_C10_full4k`) keeps running on card 5 as the single-step reference.
- `ar6_headfrac03` (init head fraction 0.03 instead of 0.05, on P9; card 3, 28 min): 36.629 / 35.945 against
  P9@1500 36.812 / 36.135 — **−0.183 / −0.190**, HFEN worse by 0.009 / 0.006. **Discard**: a wider initial support
  spends Gaussians outside the head. Settled.
- `ar6_slr008` (scale lr 0.008, on P9; card 4, 26 min): **36.854 / 36.253** against P9@1500 36.812 / 36.135 —
  **+0.042 / +0.118**, FLAIR head SSIM +0.0010, FLAIR HFEN +0.0007, T1 cells level. **Keep.** The scale-lr curve
  keeps rising (0.002 → 0.003 → 0.005 on P8; 0.008 on P9). Queues re-cut: `ar6_slr012` next on card 4, the union
  combination takes 0.008 (`ar6_best_C10_T112_slr008`), the card-5 promotion becomes `ar5_slr008_full4k`.
- `ar6_best_C10_T114` (union row with T1 0.14; card 4, 26 min): 36.918 / 36.158 — against the T1-0.12 union row
  **T1 +0.033**, FLAIR level, T1 head SSIM +0.0008, T1 head HFEN −0.0004 (4x its floor), sharpness −0.003.
  `min_margin` unchanged (FLAIR binds at +0.401). **Keep-marginal on the non-binding channel**: the T1 strength
  curve under the 1-level db4 prior is 0.10 36.812 → 0.12 36.885 → 0.14 36.918, flattening, and HFEN starts to
  give. `ar6_best_C10_T116` queued as the end-point; the promotion in flight uses 0.12, a 0.14 step at 4000 follows
  only if 0.16 confirms the plateau.
  Correction (05:30, on the view's sharpness column): against the parent P9@1500 (0.865 / 0.829) the T1-0.14 row's
  T1 sharpness is 0.848, **−0.017, beyond the 0.01 allowance** → **discard (reshape)**, not keep-marginal. The
  T1-0.12 union row sits at 0.856 / 0.820 (−0.009 / −0.009), inside the allowance but at its edge; its promotion
  at 4000 is judged on sharpness against P9's 0.867 / 0.832 as well.

### Promotion 7 — FLAIR wavelet 0.10 on P9 at 4000 iterations (05:35): Pareto keep, folded into the next parent
`ar5_db4_lev1_C10_full4k` (card 5, 63 min, volumes): 36.852 / **36.268** against P9's 36.853 / 36.234 — T1 level,
**FLAIR +0.034**, FLAIR head SSIM +0.0007, FLAIR head HFEN +0.0003, T1 cells level, sharpness level. No cell
regresses; `min_margin` does not move because T1 now binds at 4000 (+0.426 vs FLAIR +0.511).
Rule clarification (recorded now, applies from here): a run whose cells all hold within their floors and at least
one improves is a **Pareto keep** even when `min_margin` is unchanged — the objective is the gap on every metric,
and the binding channel may swap. FLAIR 0.10 therefore enters the next parent together with whatever the union
promotion (`ar5_best_C10_T112_full4k`, T1 0.12 + FLAIR 0.10, card 3, ~05:55) decides for T1.
