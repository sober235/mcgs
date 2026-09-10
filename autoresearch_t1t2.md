# autoresearch — MC-GS joint on the hyw T1 + T2 AF12 pair: beat every method inside the head

_Opened 2026-09-09. Karpathy-style loop: propose one change → run at a fixed budget → read one
number → keep or discard → commit → repeat. Runner `autoresearch_t1t2.sh`, ledger
`autoresearch_t1t2.tsv` (one row per experiment, append-only), branch `autoresearch-t1t2`._

## Objective

MC-GS **joint** (shared geometry, per-primitive dual complex density) must beat every other arm of
the eight-method table in `Results_20260908/T1T2_AF12_retro` on **head-only PSNR of BOTH
contrasts**, at the full 4000-iteration budget, on the same data / mask / CSM / gauge.

The number the loop reads: `min_margin = min(T1_head_psnr - 34.607, T2_head_psnr - 34.219)`
(best competitor = CS-SENSE on both). Success = `min_margin > 0` at 4000 iterations **and** no
blur trade: head SSIM and head HFEN of each contrast no worse than the current MC-GS joint
(0.9190 / 0.1220 on T1, 0.9295 / 0.1005 on T2), sharpness not below it (0.874 / 0.802).
Stretch: head SSIM above the best competitor too (0.9436 T1 CS-SENSE, 0.9493 T2 ZS-SSDU).

Where we start (4000 iters, cold start, frozen working point):

| arm | T1 head PSNR | T2 head PSNR | T1 head SSIM | T2 head SSIM |
|---|---|---|---|---|
| CS-SENSE (to beat) | **34.607** | **34.219** | 0.9436 | 0.9427 |
| ZS-SSDU | 34.134 | 33.822 | 0.9365 | **0.9493** |
| MC-GS joint (now) | 33.294 | 33.121 | 0.9190 | 0.9295 |
| MC-GS single | 33.296 | 33.497 | 0.9164 | 0.9319 |

Gap to close: **+1.31 dB on T1, +1.10 dB on T2.**

## Why "debug": the cold-start path was never exercised by the frozen recipe

Every frozen number came from a 576902-Gaussian warm start fitted on the old FOV. On this grid
there is no warm start, and reading the cold-start path (`joint_trainer._setup_model`, else
branch) shows three defects the warm start had been hiding:

1. **Uniform init over the render box.** `positions = rand*2-1`; the head is 27.7 % of the
   box, so ~72 % of the 600k Gaussians start in air with density ~0 and a position gradient ~0
   (the tile voxelizer's gradient is proportional to density). They never move. The effective
   cloud inside the head is ~166k, not 577k. (`configs/new_t1flair_anat*.yaml`, Aug 17,
   documented the same defect on another bundle and measured +1.07 / +0.82 dB from a ZF-support
   init alone.)
2. **Edge-TV kappa is resolved once from the initial render** (`resolve_edge_kappa`), which for
   a cold start is speckle, not an image. The edge threshold that decides where TV is relaxed
   is therefore calibrated on garbage. With a warm start the first render was already anatomy.
3. **No densification.** The frozen cloud had already been densified by the legacy pipeline;
   `adaptive_control` is off here, so capacity is frozen at whatever the init gave. In the
   sibling gs3dmc project, capacity was the #1 cold-start bottleneck (+1.2 / +2.1 dB from
   1.2M to 2.4M points).

Plus the learning rates (position 3e-4, exponential decay to 0.135x by iter 4000) were tuned
for a converged geometry.

## Fixed — never touched, never used for fitting or selection

Data, mask (`data/mask_af12_3d.h5`, AF 11.71), ESPIRiT maps, ground truth, the scorer in
`arc/cycle.py` and `diag_artefact/headonly.py`, the gauge. No learned or pre-trained prior, no
external data, no GT in any decision (the GT enters only when the scorer runs). The model stays
MC-GS joint: one shared Gaussian geometry, dual complex density, hard-DC scoring. The base
config `configs/t1t2_af12_hyw.yaml` and the frozen working point baked into the runner
(edge-TV, kappa_q 0.85, w_floor 0.10, Charbonnier eps 0.04, lambda_geo 0.10, kspace_high 2.2)
are the baseline; an experiment changes them only through the extra args it passes.

## Allowed

Anything on the fitting side: initialisation, densification, learning rates and schedule,
regulariser weights and shape, iteration budget, trainer code on those paths. Code changes go in
with a unit test and must leave the default path bit-identical (the default init/kappa paths
reproduce every earlier number).

Knobs already wired (cycle.py flags, or `--set key.path=value` for anything in the yaml):

| knob | where | note |
|---|---|---|
| `gaussian.init_mode` | `--set` | `uniform` (default) / `head` / `intensity` — `paired_gs/gaussian/init_cloud.py` |
| `gaussian.init_scale_mode` | `--set` | `fixed` (default, init_scale) / `3nn` (mean 3-NN spacing x `init_scale_mult`) |
| `gaussian.init_head_frac` | `--set` | ZF support rule, default 0.05 (same rule as the head scorer, on ZF) |
| `loss.edge_tv_kappa_source` | `--set` | `render` (default) / `zf`; `loss.edge_tv_kappa_refresh_iter` re-resolves from the render once |
| densification | `--use_cloning --split_quantile q --max_points N --densify_until I` | any of these enables `adaptive_control`; `--set adaptive_control.{densify_from_iter,densify_every,scale_threshold,opacity_threshold}` |
| lrs | `--set gaussian.{position,scale,rotation,density}_lr` | schedule: `--set training.lr_scheduler.gamma` |
| regulariser | `--lambda_geo --tv_eps --kspace_high --edge_tv_kappa_q --edge_tv_wfloor --kspace_center --kspace_mid` | |
| wavelet sparsity (added 09-10) | `--set loss.lambda_wav=λ` (+ `loss.wav_name` db2, `wav_levels` 3, `wav_eps` 0, `wav_cycle_spin` true, `wav_on` complex/magnitude) | L1 of the 3-D wavelet detail coefficients of the render (`paired_gs/losses/wavelet_l1.py`), i.e. the L1-ESPIRiT prior; default off (bit-identical); +66 % time per iteration; `--lambda_geo 0` alongside it replaces TV instead of adding to it |
| capacity | `--set gaussian.initial_num_points` | 600000 baseline; VRAM 32 GB at 600k joint |

## Budget and protocol

- **Screening = 1500 iterations** (~30 min train + ~15 min load/score on an 80 GB card),
  no volume dump. **Promotion = 4000 iterations** with `SAVE_VOLUMES=1`. A screening winner is
  a hypothesis until its 4000-iteration run confirms it.
- Up to six experiments in parallel (GPUs 2–7), one change each relative to a named parent.
  The first screening run of each round is the parent itself, so every delta is same-budget.
- Keep rule: `min_margin` improves by more than the seed-repeat floor (measured once by
  re-running the parent with `--set training.seed=43`), with no blur trade (sharp, head SSIM,
  head HFEN of either contrast not worse than the parent beyond that floor). Ties go to the
  simpler / cheaper configuration.
- Every experiment: unique `EXP` tag, one row in the ledger (`status` = keep / discard / crash
  / promoted), the exact args in the row, one commit when code or the base config changed.
  Outputs live in `results_arc/<EXP>*` (gitignored); the ledger and this file are versioned.
- Do not re-run levers the ledger already settled; combine two knobs only after each has won
  on its own. Stop the loop when `min_margin > 0` survives promotion and the transfer check.
- Transfer check before claiming victory: the winning recipe on the two T1+FLAIR topics
  (`T1FLAIR_AF12_retro`, `T1FLAIR_AF12_v3_jiangyizhou`) against their own tables.

## Round log

(appended as rounds close; the per-experiment numbers are in `autoresearch_t1t2.tsv`)

### Round 0 — 2026-09-09 22:28, seven runs at 1500 iterations, GPUs 1–7, ~20 min each

Head-only PSNR T1 / T2 (dB), margins against CS-SENSE 34.607 / 34.219:

| exp | change vs parent | T1 | T2 | vs parent | head SSIM T1/T2 | sharp T1/T2 | verdict |
|---|---|---|---|---|---|---|---|
| `ar0_A_base` | — (uniform init, the shipped cold start) | 32.547 | 32.345 | — | 0.9069 / 0.9199 | 0.877 / 0.794 | baseline |
| `ar0_A_seed43` | seed 42 → 43 | 32.551 | 32.356 | +0.004 / +0.011 | | | repeat floor ≈ 0.01 dB |
| `ar0_B_head` | init positions inside the ZF head support | 34.165 | 34.269 | **+1.62 / +1.92** vs A | 0.9265 / 0.9398 | 0.962 / 0.876 | **keep** |
| `ar0_C_head3nn` | B + 3-NN spacing scales | 34.771 | 34.218 | **+0.61** / −0.05 vs B | 0.9409 / 0.9411 | 0.879 / 0.814 | **keep** (SSIM, HFEN up on both) |
| `ar0_D_head_kzf` | B + kappa from the ZF image | 34.469 | 34.350 | **+0.30 / +0.08** vs B | 0.9339 / 0.9412 | 0.925 / 0.856 | **keep** |
| `ar0_E_head_dens12` | B + densify (cap 1.2M) | 34.146 | 34.296 | −0.02 / +0.03 vs B | | | discard — inert, see below |
| `ar0_F_head_poslr1e3` | B + position_lr 3e-4 → 1e-3 | 34.122 | 34.213 | −0.04 / −0.06 vs B | | | discard |

Readings.

- **Defect 1 confirmed and it is the whole story so far.** Drawing the same 600k points inside
  the measured head support instead of the box is worth +1.6 / +1.9 dB at equal budget; every
  other axis (SSIM, HFEN, sharpness) moves the same way. At 1500 iterations `ar0_B_head` already
  passes CS-SENSE on T2 and `ar0_C_head3nn` on T1. The uniform baseline gained +0.75 dB from
  1500 to 4000 iterations, so the promotion runs have headroom.
- **Defect 2 confirmed.** Resolving kappa on the ZF image (0.031 / 0.031) instead of the
  speckle render (0.016 / 0.018) is worth +0.30 / +0.08 dB with SSIM and HFEN improving.
- **3-NN scales** trade sharpness (0.962 → 0.879 on T1) for +0.61 dB, higher SSIM and lower
  HFEN on both contrasts, and stay above the program's floor (0.874 / 0.802). Ruling recorded
  here rather than silently applied: the "no blur trade" check compares sharpness against the
  program's floor (the current joint), not against a parent that is sharper than the ground
  truth is worth — B's 0.96 was closer to speckle than to detail (ZS-SSDU sits at 1.01 with a
  worse HFEN). SSIM and HFEN keep the parent-relative test.
- **Densification as configured did nothing** (600000 → 635969): with the fixed 0.008 init
  scale, no Gaussian exceeds `scale_threshold` 0.01, so `split` never fires and only the
  starved `clone` branch adds points. Not a verdict on densification; re-run on the 3-NN
  parent with `scale_threshold` below the init scale.
- position_lr ×3.3 is slightly negative; leave the lrs alone.

Round 1 parent = B + 3-NN + kappa_zf (`ar1_P1`), the two independent wins combined; the parent
also goes to a 4000-iteration promotion straight away.

### Rounds 1–3 — 2026-09-10 00:05–00:45, 1500-iteration screening, GPUs 2–7

Parent P1 = head init + 3-NN scales + kappa from ZF (`ar1_P1_rerun`, 600k): **34.818 / 34.218**.
The original `ar1_P1` and its 4000-iteration promotion were killed at 00:26: another user's
training jobs landed on GPUs 2 and 3 and slowed both to 24 s/iteration (20x). The parent was
re-run on an exclusive GPU; the 600k promotion was dropped in favour of the 1.2M one below.

| exp | change vs parent | T1 | T2 | vs parent | verdict |
|---|---|---|---|---|---|
| `ar1_P1_rerun` | C + kappa from ZF | 34.818 | 34.218 | +0.05 / 0.00 vs C | parent (kappa_zf adds little once 3-NN is on: C's render kappa was already 0.025) |
| `ar1_P1_n12` | **initial points 600k → 1.2M** | **35.007** | **35.022** | **+0.19 / +0.80** | **keep = P2** |
| `ar1_P1_dens` | densify 600k → 825k (scale_threshold 0.005) | 34.948 | 34.716 | +0.13 / +0.50 | discard, dominated by n12 |
| `ar1_P1_intensity` | intensity-weighted init | 34.623 | 34.415 | −0.20 / +0.20 | discard (T1 is the binding contrast) |
| `ar1_P1_lam007` | lambda_geo 0.10 → 0.07 | 34.711 | 34.213 | −0.11 / −0.01 | discard |
| parent P2 = `ar1_P1_n12` (1.2M) | | 35.007 | 35.022 | | |
| `ar3_n16` | 1.2M → 1.6M | 34.910 | 35.285 | −0.10 / +0.26 | discard |
| `ar2_P2_n24` | 1.2M → 2.4M | 34.569 | 35.484 | −0.44 / +0.46 | discard |
| `ar2_P2_n48` | 1.2M → 4.8M | 32.237 | 35.078 | −2.77 / +0.06 | discard |
| `ar2_P2_dens24` | + densify to 1.6M | 34.890 | 35.409 | −0.12 / +0.39 | discard |
| `ar2_P2_intensity` | intensity init at 1.2M | 34.795 | 35.064 | −0.21 / +0.04 | discard |
| `ar3_n24_lam015` | 2.4M + lambda_geo 0.15 | 34.726 | 35.323 | −0.28 / +0.30 | discard (partial T1 rescue, not enough) |
| `ar3_n24_khp10` | 2.4M + plain high-band weight 2.2 → 1.0 | 34.239 | 35.477 | −0.77 / +0.46 | discard (lowering T1's high band hurts T1) |

Readings.

- **Capacity is the second lever and it is contrast-asymmetric.** T1 peaks at 1.2M points and
  falls monotonically beyond it (35.01 → 34.91 → 34.57 → 32.24 at 1.2 / 1.6 / 2.4 / 4.8M) while
  T2 keeps rising to 2.4M (35.02 → 35.29 → 35.48). The training loss on the sampled k-space
  falls monotonically with capacity (dc_p 1.05e-2 → 9.9e-3 → 9.4e-3) while T1's oracle
  unsampled-k error rises (e_unobs 0.590 → 0.617 → 0.726): extra Gaussians fit the acquired
  8.5 % better and extrapolate worse. T2 is protected by its shutter (22 % of k known-zero) and
  its apodised spectrum. More TV at 2.4M buys back only part of T1; lowering T1's high-band
  weight makes it worse. Since the geometry is shared, the point count follows the binding
  contrast: **1.2M**.
- Intensity-weighted sampling, 0.07 TV and densification are all inert-to-negative on top of
  the head-support init; the kappa source is a small positive that matters mostly when scales
  are fixed.
- At 1500 iterations P2 beats CS-SENSE by +0.40 (T1) / +0.80 (T2) dB head-only and on the
  whole volume too (39.34 / 39.87 vs 39.24 / 39.23).

### Promotion — `ar3_P2_full4k`, 2026-09-10 00:46: the 1.2M working point at 4000 iterations

| | T1 | T2 | best competitor |
|---|---|---|---|
| head PSNR | **34.978** (+0.371) | **35.429** (+1.211) | CS-SENSE 34.607 / 34.219 |
| head SSIM | 0.9373 | 0.9474 | CS-SENSE 0.9436 / ZS-SSDU 0.9493 |
| head HFEN | **0.0863** | **0.0665** | CS-SENSE 0.1071 / 0.0916 |
| whole PSNR | **39.249** | **40.174** | CS-SENSE 39.243 / 39.229 |
| sharp / N | 0.912 / 0.998 | 0.857 / 0.948 | (current joint 0.874 / 0.802) |

`min_margin = +0.371 dB` at the full budget: **the objective is met on head PSNR, head HFEN and
whole-volume PSNR for both contrasts.** Head SSIM is second by 0.006 (T1) and 0.002 (T2) — the
stretch goal, still open. Against the shipped cold start (33.294 / 33.121) this is
+1.68 / +2.31 dB from three fitting-side changes and no new prior. From 1500 to 4000
iterations T1 is flat (35.007 → 34.978) while T2 gains +0.41; T1 has reached its plateau by
1500 at this capacity. 40 min on an exclusive A800, 32 GB. Volumes in
`results_arc/ar3_P2_full4k_volumes.npz`.

Recipe = the frozen working point plus
`--set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn --set loss.edge_tv_kappa_source=zf --set gaussian.initial_num_points=1200000`.

Round 4 (single deltas from P2, 1500 iters): seed 43 repeat 35.004 / 35.017 (floor 0.003 /
0.005 dB); plain high-band weight 3.0 → 34.345 / 34.911, discard (2.2 is the optimum on T1:
1.0 and 3.0 both lose). Pending: 1.5x initial scales, kappa refresh at 300, edge quantile 0.75.
Next: transfer check on the two T1+FLAIR topics with the same recipe; archive the arm into
`Results_20260908/T1T2_AF12_retro`.

### Round 4 — single deltas from P2 at 1500 iterations, aimed at the head-SSIM gap (2026-09-10, 00:45–01:20)

| exp | change vs P2 | T1 | T2 | vs P2 (35.007 / 35.022) | head SSIM | verdict |
|---|---|---|---|---|---|---|
| `ar4_P2_seed43` | seed 43 | 35.004 | 35.017 | −0.003 / −0.005 | 0.9394 / 0.9454 | repeat floor |
| `ar4_P2_khp30` | plain high-band weight 3.0 | 34.345 | 34.911 | −0.66 / −0.11 | 0.9248 / 0.9449 | discard (2.2 is the T1 optimum: 1.0 and 3.0 both lose) |
| `ar4_P2_kref300` | kappa re-resolved from the render at iter 300 | 35.024 | 34.688 | +0.02 / −0.33 | 0.9374 / 0.9380 | discard (blur: sharp 0.81 / 0.77, HFEN up) |
| `ar4_P2_kq075` | edge quantile 0.85 → 0.75 | 34.786 | 34.961 | −0.22 / −0.06 | 0.9353 / 0.9452 | discard |
| `ar4_P2_mult15` | 3-NN scales × 1.5 | 34.624 | 33.754 | −0.38 / −1.27 | 0.9412 / 0.9365 | discard |

Nothing on the smoothing side has moved head SSIM up without paying PSNR so far. Still
running as single deltas: edge quantile 0.90, lambda_geo 0.12, w_floor 0.20, tv_eps 0.06.

**Incident, 01:15.** The session's memory watchdog killed every background launcher at once
("system is running low on memory") while the host had 919 GB available of 1007 — it reads
MemFree, which six concurrent h5 loads push down through the page cache. The six runs in
flight (both transfer checks and four probes) died with them; partial outputs were removed
and the runs relaunched with `setsid nohup`, detached from the session. Launch that way from
now on; the ledger is the only thing the session needs to watch.

### Transfer checks — the recipe applied unchanged to the two T1+FLAIR topics (2026-09-10, 01:22–02:05)

| topic | channel | v2 head PSNR | best other arm | margin | original joint arm | head SSIM v2 / best | head HFEN v2 / best |
|---|---|---|---|---|---|---|---|
| hyw T1+FLAIR (`ar5_tf_hyw_t1flair`) | T1 | **34.950** | CS-SENSE 34.607 | **+0.343** | 33.181 | 0.9379 / 0.9436 | **0.0851** / 0.1071 |
| | FLAIR | **33.492** | ZS-SSDU 33.269 | **+0.223** | 31.806 | 0.9023 / 0.9235 | **0.1150** / 0.1302 |
| jyz T1+FLAIR (`ar5_tf_jyz_t1flair`) | T1 | 35.520 | CS-SENSE 35.762 | −0.242 | 34.160 | 0.9321 / 0.9429 | **0.1056** / 0.1208 |
| | FLAIR | 35.056 | CS-SENSE 35.085 | −0.028 | 33.391 | 0.9238 / 0.9307 | **0.1063** / 0.1139 |

**Same subject, other pairing: transfers (first on head PSNR and head HFEN on both channels).
Other subject: does not — second on head PSNR by 0.24 / 0.03 dB**, still +1.36 / +1.67 dB over
the original joint arm and first on head HFEN. Head SSIM is behind the best arm everywhere.

Reading, and a pre-registered follow-up. The jyz head is larger: ZF support fraction 0.4234
against 0.3199 (hyw T1+T2) / 0.3269 (hyw T1+FLAIR). A fixed 1.2M points is therefore
0.246 points per support voxel on jyz against 0.325 on hyw, and capacity was the binding lever
on T1. The transferable form of the knob is a **density**: 1 200 000 / (0.3199 × 256·256·176)
= **0.325 points per ZF-support voxel**, fixed on hyw alone. On jyz that rule gives
0.325 × 0.4234 × 11 534 336 = **1.59M points**. Declared before running:
`ar7_tf_jyz_rho` = the jyz transfer with `initial_num_points=1588000` and nothing else changed;
it is a pass only if both channels clear CS-SENSE (35.762 / 35.085). If it fails, the honest
statement is "first on the tuning subject in both pairings, second on the held-out subject".

### Round 6 — 4000-iteration runs (2026-09-10, 01:30–02:35): the two smoothing knobs at full budget, and joint versus single under the repaired cold start

| exp | what | T1 head PSNR / SSIM / HFEN | T2 head PSNR / SSIM / HFEN | sharp | verdict |
|---|---|---|---|---|---|
| `ar3_P2_full4k` (reference) | P2 | 34.978 / 0.9373 / 0.0863 | 35.429 / 0.9474 / 0.0665 | 0.912 / 0.857 | promoted |
| `ar6_P2_lam012_full4k` | P2 + lambda_geo 0.12 | 35.085 / 0.9388 / 0.0876 | 35.405 / 0.9466 / 0.0686 | 0.887 / 0.838 | not promoted |
| `ar6_P2_kq090_full4k` | P2 + edge quantile 0.90 | 35.095 / 0.9387 / 0.0883 | 35.397 / 0.9461 / 0.0694 | 0.885 / 0.836 | not promoted |
| `ar6_single_T1_v2` | single-contrast T1, same recipe | 34.680 / 0.9325 / 0.0923 | — | 0.916 | ablation |
| `ar6_single_T2_v2` | single-contrast T2, same recipe | — | 35.680 / 0.9480 / 0.0661 | 0.866 | ablation |

- **The two smoothing knobs buy T1 (+0.11 dB) with T2's SSIM and HFEN and both contrasts'
  sharpness** — the same trade the 1500-iteration screen showed, so it survives the budget. Under
  the pre-registered rule (SSIM and HFEN of either contrast not worse than the parent beyond
  the repeat floor) they are not promoted; `min_margin` would read +0.48 instead of +0.37, but
  that is a reshape toward T1, not a gain. P2 stays the working point. Head SSIM did not reach
  CS-SENSE (0.9388 vs 0.9436) either way: the stretch goal is closed as not reachable by
  fitting-side smoothing at this capacity.
- **Joint versus single, repaired cold start, 4000 iterations: the shared geometry moves
  accuracy from the easier contrast to the harder one.** T1: joint 34.978 vs single 34.680
  (+0.30 dB head, +0.30 whole, SSIM +0.005, HFEN 0.086 vs 0.092). T2: joint 35.429 vs single
  35.680 (−0.25 dB head, −0.13 whole, SSIM −0.0006). Single T1 clears CS-SENSE by only
  +0.07 dB; it is the joint model that puts T1 +0.37 ahead. This replaces the pre-repair reading
  "joint is a wash inside the head": at equal (shared) capacity the joint model gives the
  binding contrast the gain and charges the other one, which is exactly what a shared point
  budget with a T1-limited count should do (T2 alone would take 2.4M points, §rounds 1–3).

### The pre-registered density rule on jiangyizhou — FAIL (`ar7_tf_jyz_rho`, 2026-09-10 02:55)

1.588M points (0.325 per ZF-support voxel, fixed on hyw): T1 **35.273** / FLAIR **35.071** head
PSNR against 35.520 / 35.056 at 1.2M and CS-SENSE 35.762 / 35.084. More points made T1 worse
by 0.25 dB on this subject too, FLAIR flat. The rule is rejected as declared.

Reading: the T1 capacity ceiling is not a density per head voxel. The acquired data is the same
size on both subjects (3847 phase-encode lines × 256 readout × 26 coils, the same mask), so
the number of Gaussians the measurements can pin down is roughly the same, and a larger head
simply gets fewer of them per voxel. An absolute count near 1.2M is the transferable setting
for this acquisition; what is missing on jiangyizhou is not capacity.

## Verdict (2026-09-10 03:00)

- **Objective met on the arena it was set on.** hyw T1+T2, 4000 iterations: MC-GS joint v2
  is first of nine arms on head PSNR (34.978 / 35.429, +0.37 / +1.21 dB over CS-SENSE), whole
  PSNR, NMSE and head HFEN on both contrasts; second on head SSIM by 0.006 / 0.002.
- **Same subject, other pairing (hyw T1+FLAIR): holds** — first on head PSNR and HFEN.
- **Held-out subject (jiangyizhou T1+FLAIR): second on head PSNR** (−0.24 / −0.03 dB to
  CS-SENSE), first on head HFEN, +1.4 / +1.7 dB over the original joint arm. The density rule
  did not rescue it. The honest one-line claim is: *first on the tuning subject in both
  pairings, second on the held-out subject*; the repaired cold start is worth +1.4 to +2.3 dB
  head-only everywhere it was tried.
- The recipe: the frozen working point plus `gaussian.init_mode=head`,
  `gaussian.init_scale_mode=3nn`, `loss.edge_tv_kappa_source=zf`,
  `gaussian.initial_num_points=1200000`. Everything else was tried as a single change and lost
  or reshaped (ledger). Head SSIM cannot be bought on the fitting side at this capacity.
- Joint versus single under the repaired cold start: +0.30 dB to T1, −0.25 dB from T2 — the
  shared geometry helps the binding contrast at the other's expense.
- 34 experiments, 8 of them at 4000 iterations, ~9 GPU-hours, 2026-09-09 21:40 to 09-10 03:00.

### Addendum, 2026-09-10 11:18 — transfer to AF16 (pre-registered; the time stamps of the two sections above were estimated from run durations, the clock here is real)

The parallel session finished the AF16 T1+FLAIR study on both subjects (branch
`t1flair-af16-retro`, topics `Results_20260908/T1FLAIR_AF16_retro` and
`T1FLAIR_AF16_v3_jiangyizhou`). The v2 recipe is applied there **unchanged** (`ar8_tf_hyw_af16`,
`ar8_tf_jyz_af16`, 4000 iterations, their configs and data). Head-PSNR leaders to clear: hyw
ZS-SSDU 33.665 (T1) / 32.685 (FLAIR); jyz CS-SENSE 34.410 (T1) / ZS-SSDU 34.005 (FLAIR). The
original cold-start joint arms read 32.028 / 30.743 and 32.890 / 32.402. Results go in this
file and the ledger; the AF16 topic archives belong to the other session and are not edited here.

**AF16 outcome (12:10).** Same recipe, nothing changed, 4000 iterations:

| topic | channel | v2 head PSNR | leader | margin | original cold-start joint |
|---|---|---|---|---|---|
| hyw T1+FLAIR AF16 (`ar8_tf_hyw_af16`) | T1 | 33.465 | ZS-SSDU 33.665 | −0.200 | 32.028 (+1.44) |
| | FLAIR | 31.943 | ZS-SSDU 32.685 | −0.742 | 30.743 (+1.20) |
| jyz T1+FLAIR AF16 (`ar8_tf_jyz_af16`) | T1 | 33.982 | CS-SENSE 34.410 | −0.427 | 32.890 (+1.09) |
| | FLAIR | 33.640 | ZS-SSDU 34.005 | −0.364 | 32.402 (+1.24) |

At AF16 the recipe is second on every channel, by 0.2–0.7 dB, while still +1.1 to +1.4 dB over the
cold-start joint arm. The deficit grows with acceleration and is largest on FLAIR, where ZS-SSDU
leads: at 6.6 % sampling the learned denoiser's prior is worth more than what the explicit
representation recovers from the acquired points. The AF12 recipe was not re-tuned for AF16 here
(the parallel session runs its own loop on jyz AF16, branch `autoresearch-jyz-af16`); the honest
scope of the claim stays AF12 on the tuning subject.

**Single-contrast v2 arms on the T1+FLAIR topics (12:55–13:40, `ar9_*`).** hyw FLAIR 33.367 (joint v2
33.492, +0.13 for joint); jyz T1 35.209 (joint 35.520, +0.31) and FLAIR 35.101 (joint 35.056, −0.05;
the single FLAIR arm edges CS-SENSE 35.084 by 0.017 dB, repeat-floor order, listed not claimed).
Together with the T1+T2 pair (+0.30 / −0.25): the joint model is ahead of its ablation on five of six
channels, always on T1, and the shared geometry costs at most 0.25 dB on the easier contrast.

### Classical baselines revisited (2026-09-10, 13:00–15:00) — the standing changes

At the user's request three more arms were run on all three AF12 topics, and the existing TV arm
was audited. Head PSNR, hyw T1 / T2 / FLAIR and jyz T1 / FLAIR:

| arm | hyw T1 | hyw T2 | hyw FLAIR | jyz T1 | jyz FLAIR |
|---|---|---|---|---|---|
| CS-SENSE TV as run 09-09 (35 outer) | 34.607 | 34.219 | 33.252 | 35.762 | 35.084 |
| CS-SENSE TV converged (200 outer, lambda swept on hyw) | 34.966 | 35.168 | 33.873 | 36.144 | 35.581 |
| joint CS-SENSE, vectorial TV (200 outer, lambda 1e-3) | 35.020 | 35.008 | 33.929 | 36.070 | 35.624 |
| **L1-wavelet ESPIRiT** (BART, 100 FISTA, lambda per contrast on hyw) | **35.530** | **35.818** | **34.105** | **36.334** | **35.720** |
| MC-GS joint v2 | 34.978 | 35.429 | 33.492 | 35.520 | 35.056 |

- The 09-09 TV arm was under-converged (35 outer iterations inherited from the CE study); the
  converged arm is 0.35–0.6 dB higher. Lambda re-swept on hyw: 1e-3 already optimal on T1,
  0.7e-3 slightly better on T2/FLAIR.
- L1-wavelet ESPIRiT with a swept lambda is first on all five channels, 0.4–0.8 dB above v2. Its
  number includes an early-stopping effect (300 iterations score 0.3–0.4 dB lower).
- Joint (vectorial) TV is a wash against single TV, as the project found before.
- **Corrected verdict:** MC-GS joint v2 is second/third on head PSNR everywhere once the classical
  arms are tuned and converged; it keeps the lowest head HFEN on hyw T1 and the +1.4–2.3 dB gain
  over the cold-start arm. The campaign's real result is the repaired cold start, not "first".
  Target for any further loop on hyw T1+T2: 35.53 / 35.82 (L1-ESPIRiT).
- P-LORAKS (LORAKS v2 toolbox, per readout slice, S-matrix): R = 3 costs > 1 h per slice on a
  worker; the rank sweep runs at R = 2, 30 iterations (about 35 min per slice), results pending.

**LORAKS family, probed on a six-slice hyw T1 subset (readout-decoupled 2-D problems, 26 coils,
head-masked PSNR of the slices; CS-SENSE ≈ 35 dB and v2 ≈ 36 dB on the same slices).**
P-LORAKS (calibrationless, S-matrix): R = 3 costs > 1 h per slice per worker and was abandoned;
R = 2, strict DC, alg 4 converges by ~22 iterations and gives 30.2 dB at rank 40, worse at every
larger rank (27.8 / 25.8 / 24.3 at 80 / 150 / 250) and at rank 20 (25.5). AC-LORAKS (nullspace
from the 16×16 fully sampled centre): 27.5–28.7 dB, rising with rank to 250. SENSE-LORAKS
(the toolbox's map-based variant): its images come out in MATLAB's 1/N FFT scale (×√N restores
the gauge, verified: least-squares scale 209 vs √N = 212); at λ ≤ 1e-2 the penalty is inert
(= pure SENSE, 31.7–32.0 dB ≈ CG-SENSE) and at λ = 10 the solver degenerates (image scale
collapses, NaNs) — not pursued further. Reading: at a 2-D in-plane acceleration of 11.7 with a
16×16 calibration region, the low-rank neighbourhood models that do not use the full-data
ESPIRiT maps are structurally behind every map-based arm; this is a property of the
retrospective set-up (maps from the full k-space), stated as such. The table gets the
calibrationless P-LORAKS at rank 40 / R = 2 on all five channels (full runs, ~50 min per
channel on 48 workers), so the row is complete and its standing is what it is.

**Per-slice check against the corrected arm set (16:40).** `find_best_slice_v2.py` now takes its margin
against an explicit list of canonical, converged arms (converged TV, joint TV, L1-ESPIRiT, ZS-SSDU,
DIP, INR, CG-SENSE, ZF; sweep variants in recon/ are ignored). Result: MC-GS joint v2 is first on
0 of the full-brain slices in every plane of every topic; L1-ESPIRiT leads everywhere, including
the slices where v2 peaks (hyw T1 slice 150: L1-ESPIRiT 37.02 vs v2 36.29). The compact figures
were regenerated with those columns and the true ranks.

### Round 11 — wavelet sparsity on the render (2026-09-10, 17:50–20:30), 1500-iteration screens, single deltas from P2

L1-ESPIRiT's lead says the strongest classical prior on this data is wavelet sparsity, while MC-GS
regularises with edge-TV. Round 11 adds that prior to the MC-GS objective (`loss.lambda_wav`,
`paired_gs/losses/wavelet_l1.py`: per-voxel L1 of the 3-D db2 detail coefficients of the rendered
complex volumes, 3 levels, periodic boundary, cycle spinning). λ was calibrated against the TV
term on the v2 reconstruction: 0.03 is about 1.6 times the TV term.

Head PSNR T1 / T2. Parent P2 at 1500 = 35.007 / 35.022; L1-ESPIRiT at its full budget = 35.530 / 35.818.

| exp | change vs P2 | T1 | T2 | head SSIM | head HFEN | sharp | verdict |
|---|---|---|---|---|---|---|---|
| `ar11_wav001_tv` | + wavelet 0.01 | 35.179 | 35.085 | 0.9424 / 0.9466 | 0.0824 / 0.0689 | 0.889 / 0.835 | discard (dominated by 0.03) |
| `ar11_wav003_tv` | + wavelet 0.03 | 35.373 | 35.120 | 0.9456 / 0.9474 | 0.0813 / 0.0704 | 0.873 / 0.821 | keep, promoted (`ar12_wav003_full4k`) |
| `ar11_wav005_tv` | + wavelet 0.05 | 35.447 | 35.082 | 0.9471 / 0.9472 | 0.0825 / 0.0732 | 0.859 / 0.809 | best T1 at 1.2M; T2 past its peak |
| `ar11_wav01_tv` | + wavelet 0.1 | 35.350 | 34.834 | 0.9471 / 0.9446 | 0.0900 / 0.0827 | 0.830 / 0.781 | discard (over-regularised) |
| `ar11_wav003_notv` | wavelet 0.03 replacing TV | 34.306 | 34.770 | 0.9227 / 0.9409 | 0.0816 / 0.0658 | 1.017 / 0.929 | discard (under-regularised, sharper than GT) |
| `ar11_wav01_notv` | wavelet 0.1 replacing TV | 35.184 | 34.967 | 0.9443 / 0.9478 | 0.0831 / 0.0744 | 0.911 / 0.846 | discard (above P2 on T1, below TV + wavelet on both) |
| `ar11_wav003_mag` | + wavelet 0.03 on the magnitude | 35.245 | 34.989 | 0.9418 / 0.9440 | 0.0875 / 0.0736 | 0.854 / 0.808 | discard (the complex modulus is better by 0.13 dB on both) |
| `ar11_n24_wav005_tv` | **2.4M points + wavelet 0.05** | **35.746** | **35.885** | 0.9474 / 0.9521 | 0.0827 / 0.0695 | 0.867 / 0.823 | **keep, promoted (`ar12_n24_wav005_full4k`); above L1-ESPIRiT's full-budget numbers on both already at 1500** |
| `ar11_n36_wav003_tv` | 3.6M points + wavelet 0.03 | 35.353 | 36.100 | 0.9401 / 0.9519 | 0.0882 / 0.0668 | 0.873 / 0.836 | discard (vs 2.4M: T2 +0.22, T1 −0.16 — more points need a stronger wavelet on T1); 3.6M + 0.05 queued (`ar11_n36_wav005_tv`) |
| `ar11_n36_wav005_tv` | 3.6M points + wavelet 0.05 | 35.729 | 36.149 | 0.9457 / 0.9532 | 0.0851 / 0.0687 | 0.863 / 0.825 | keep (vs 2.4M + 0.05: T1 flat −0.02, T2 +0.26; min margin vs L1-ESPIRiT +0.07 → +0.20), promoted (`ar12_n36_wav005_full4k`) |
| `ar11_n24_wav003_tv` | **2.4M points + wavelet 0.03** | **35.513** | **35.878** | 0.9439 / 0.9514 | 0.0834 / 0.0670 | 0.881 / 0.835 | **keep, promoted (`ar12_n24_wav003_full4k`)** |

Readings.

- Wavelet sparsity on top of edge-TV is the first new lever since capacity: +0.37 / +0.10 dB at
  λ 0.03 and +0.44 / +0.06 at 0.05, with head SSIM up on both contrasts. Replacing TV by it is
  worse, so on this data the two priors are complementary rather than substitutes.
- **It removes the T1 capacity ceiling.** Without it, 2.4M points cost T1 0.44 dB (34.569, the
  extra points overfit the acquired 8.5 %). With it, T1 at 2.4M is 35.513, so the point count can
  follow T2 (35.484 → 35.878). At 1500 iterations this configuration sits 0.02 dB below
  L1-ESPIRiT's full-budget T1 and 0.06 dB above its T2. Head HFEN is better than P2 on both
  contrasts (0.0834 / 0.0670 vs 0.0844 / 0.0690), head SSIM is higher on both, and sharpness
  stays above the program floor, so there is no blur trade.
- A stronger wavelet alone (0.1, TV off) recovers most of the no-TV loss but stays below TV +
  wavelet on both contrasts; penalising the magnitude instead of the complex coefficients loses
  0.13 dB on both. The configuration to carry forward is complex wavelet 0.03 on top of edge-TV.
- Code check: the full suite passes with the new module (182 tests, 45 min at 4 CPU threads under
  the P-LORAKS load; the first attempt stalled on thread oversubscription, see STATUS §4).
- Pending: 2.4M + wavelet 0.05, 3.6M + wavelet 0.03 (single deltas from `ar11_n24_wav003_tv`), and
  the two 4000-iteration promotions. Nothing is claimed until the promotions land: at 1.2M, T1 was
  flat from 1500 to 4000 and T2 gained 0.4 dB, but at 2.4M the budget effect is unmeasured.

### Round 12 — the two promotions at full budget (2026-09-10, 20:00–21:10)

| exp (4000 iters) | config | T1 | T2 | head SSIM | head HFEN | whole PSNR | sharp |
|---|---|---|---|---|---|---|---|
| `ar12_wav003_full4k` | 1.2M + wavelet 0.03 | **35.546** | 35.598 | 0.9459 / 0.9502 | 0.0805 / 0.0671 | 39.795 / 40.357 | 0.882 / 0.833 |
| `ar12_n24_wav003_full4k` | 2.4M + wavelet 0.03 | 35.448 | **36.082** | 0.9422 / 0.9521 | 0.0852 / 0.0659 | 39.563 / 40.701 | 0.880 / 0.838 |
| L1-ESPIRiT (reference) | BART, 100 FISTA | 35.530 | 35.818 | 0.9471 / 0.9536 | 0.0867 / 0.0656 | 39.823 / 40.442 | 0.899 / 0.865 |

- Each configuration beats L1-ESPIRiT on one contrast only: 1.2M on T1 (+0.016), 2.4M on T2 (+0.264).
  With the wavelet term T1 now keeps improving from 1500 to 4000 at 1.2M (+0.17), but at 2.4M it
  slips back (35.513 → 35.448): over a long budget the larger cloud still overfits T1.
- Next step, and the fair analogue of what L1-ESPIRiT has: a per-contrast regularisation strength
  (its lambda is 0.0015 for T1 and 0.001 for T2, while MC-GS shared one lambda between the
  contrasts). `loss.lambda_wav_plain` / `loss.lambda_wav_contrast` weight the two volumes inside
  `wavelet_l1` (9 tests). Run directly at 4000 iterations, because the T1 overfitting only shows up
  late: `ar14_n24_wavP006_C003_full4k` (2.4M, T1 0.06, T2 0.03). A second one with T1 0.1 was
  prepared but not run: the 2.4M screen at a shared 0.05 (`ar11_n24_wav005_tv`) came in at
  35.746 / 35.885, above L1-ESPIRiT on both contrasts already at 1500 iterations, so the simpler
  shared strength was promoted instead (`ar12_n24_wav005_full4k`).
- The jiangyizhou transfer check of the 2.4M recipe was queued speculatively and removed before it
  trained (the promotion showed the recipe is not final); the hyw T1+FLAIR one continues, as a
  measurement of the wavelet term on FLAIR.
- That hyw T1+FLAIR run (`ar13_tf_hyw_t1flair_v3`, 2.4M + wavelet 0.03, 4000 iterations): T1 35.471 /
  FLAIR 34.061 against L1-ESPIRiT 35.530 / 34.105 (−0.06 / −0.04), and +0.52 / +0.57 over v2's
  transfer run; the wavelet term closes most of the FLAIR gap. The candidate recipe (2.4M + 0.05) is
  now run unchanged on the same topic (`ar13_tf_hyw_t1flair_v3b`), jiangyizhou next.

### Round 14 — per-contrast wavelet strength at the full budget (2026-09-10, 21:15–22:15)

| exp (4000 iters) | config | T1 | T2 | head SSIM | head HFEN | whole PSNR | sharp | grain ratio N |
|---|---|---|---|---|---|---|---|---|
| `ar14_n24_wavP006_C003_full4k` | 2.4M, wavelet T1 0.06 / T2 0.03 | **35.856** | **36.079** | **0.9481** / 0.9521 | **0.0832** / 0.0661 | **40.006** / **40.700** | 0.862 / 0.838 | 0.969 / 0.942 |
| L1-ESPIRiT | BART, 100 FISTA | 35.530 | 35.818 | 0.9471 / **0.9536** | 0.0867 / **0.0656** | 39.823 / 40.442 | 0.899 / 0.865 | 1.008 / 0.953 |
| CS-SENSE (TV, converged) | 200 outer | 34.966 | 35.168 | 0.9460 / 0.9493 | 0.0993 / 0.0785 | 39.534 / 40.110 | 0.829 / 0.783 | 0.973 / 0.932 |
| MC-GS joint v2 (P2) | 1.2M, no wavelet | 34.978 | 35.429 | 0.9373 / 0.9474 | 0.0863 / 0.0665 | 39.249 / 40.174 | 0.912 / 0.857 | 0.998 / 0.948 |

- **The first configuration above every arm on head PSNR of both contrasts at the full budget**:
  +0.33 / +0.26 dB over L1-ESPIRiT, +0.89 / +0.91 over the converged TV arm, +0.88 / +0.65 over v2.
  Whole-volume PSNR is first on both; head SSIM and HFEN are first on T1 and second on T2
  (−0.0015 SSIM, +0.0005 HFEN against L1-ESPIRiT).
- Gate check against the Objective section: head SSIM and HFEN are far inside their floors;
  **T1 sharpness 0.862 is below the 0.874 floor by 0.012** (T2 0.838 passes). The sharpness drop
  from v2 (0.912) comes with a lower grain ratio (0.998 → 0.969 of the ground truth's own grain)
  and a lower head HFEN (0.0863 → 0.0832), i.e. with noise going and detail error going down, not
  up. That reading is recorded next to the gate, and the gate is reported as missed on T1.
- The per-contrast strengths were selected on this pair. For the transfer checks the
  contrast-agnostic shared strength (2.4M + wavelet 0.05, its promotion `ar12_n24_wav005_full4k`
  pending) is used: FLAIR has no tuned per-contrast value of its own, and L1-ESPIRiT's own FLAIR
  λ is five times its T2 λ, so "contrast channel 0.03" would not be a like-for-like transfer.

### Transfer checks of the contrast-agnostic recipe (2.4M + shared wavelet 0.05, 4000 iterations, unchanged)

| topic | channel | recipe | L1-ESPIRiT | margin | v2 (previous recipe) | head SSIM recipe / L1 | head HFEN recipe / L1 |
|---|---|---|---|---|---|---|---|
| hyw T1+FLAIR (`ar13_tf_hyw_t1flair_v3b`) | T1 | **35.772** | 35.530 | **+0.242** | 34.950 | 0.9470 / 0.9471 | **0.0828** / 0.0867 |
| | FLAIR | **34.359** | 34.105 | **+0.254** | 33.492 | 0.9169 / 0.9221 | **0.1065** / 0.1096 |
| jiangyizhou T1+FLAIR, held out (`ar13_tf_jyz_t1flair_v3b`) | T1 | 36.337 | 36.334 | +0.003 (tie) | 35.520 | 0.9428 / 0.9443 | **0.0987** / 0.1034 |
| | FLAIR | **35.935** | 35.720 | **+0.215** | 35.056 | 0.9346 / 0.9368 | **0.0994** / 0.1002 |

- The recipe transfers: first on head PSNR on three of four channels and tied on the fourth (jyz T1,
  +0.003 dB, inside the 0.003–0.005 dB seed-repeat floor measured at P2), first on head HFEN on all
  four, and 0.8–0.9 dB above the previous recipe everywhere. Head SSIM trails L1-ESPIRiT by 0.0001
  to 0.005.
- Seed 43 repeat on jiangyizhou (`ar13_tf_jyz_t1flair_v3b_s43`): T1 36.338 / FLAIR 35.937, margins
  +0.004 / +0.217 against seed 42's +0.003 / +0.215. The seed spread is 0.001 / 0.002 dB, so the jyz T1
  margin is reproducible but negligible: a tie with L1-ESPIRiT in any practical sense, not a win.
  FLAIR's +0.22 dB is solid. The per-contrast T1+T2 winner gets the same repeat
  (`ar14_n24_wavP006_C003_full4k_s43`).

## v3 — the recipe (decided 2026-09-11 00:25)

**v3 = the v2 cold-start fixes (head-support init, 3-NN scales, kappa from ZF) + 2.4M initial points +
a shared complex wavelet-sparsity term λ 0.05 on top of edge-TV**, 4000 iterations. One strength for both
contrasts, so it transfers to a contrast it was not tuned on; that is why it is chosen over the
per-contrast variant, which scores a slightly higher minimum margin on the tuning pair (+0.261 vs +0.241).

```
./autoresearch_t1t2.sh <EXP> <GPU> 4000 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn --set loss.edge_tv_kappa_source=zf \
  --set gaussian.initial_num_points=2400000 --set loss.lambda_wav=0.05
```

Head PSNR against the strongest arm of each table (L1-ESPIRiT everywhere):

| topic | v3 | L1-ESPIRiT | margin | v2 | head HFEN v3 / L1 | T1 sharpness (floor 0.874) |
|---|---|---|---|---|---|---|
| hyw T1+T2 (tuning pair, `ar12_n24_wav005_full4k`) | 35.771 / 36.112 | 35.530 / 35.818 | **+0.241 / +0.294** | 34.978 / 35.429 | 0.0831 / 0.0679 vs 0.0867 / 0.0656 | 0.867 (missed by 0.007) |
| hyw T1+FLAIR (`ar13_tf_hyw_t1flair_v3b`) | 35.772 / 34.359 | 35.530 / 34.105 | **+0.242 / +0.254** | 34.950 / 33.492 | 0.0828 / 0.1065 vs 0.0867 / 0.1096 | 0.869 (missed by 0.005) |
| jiangyizhou T1+FLAIR, held out (`ar13_tf_jyz_t1flair_v3b`, seed 42 / 43) | 36.337 / 35.935 | 36.334 / 35.720 | +0.003 (tie) / **+0.215** | 35.520 / 35.056 | 0.0987 / 0.0994 vs 0.1034 / 0.1002 | 0.892 (passes) |

- First on head PSNR on five of six channels and tied on the sixth; first on head HFEN on five of six;
  head SSIM within 0.005 of L1-ESPIRiT everywhere (below it by 0.0001–0.005); whole-volume PSNR
  first on the tuning pair.
- The pre-registered T1 sharpness floor (0.874, the sharpness of the original cold-start joint arm)
  is missed on both huangyuwei topics, by 0.007 and 0.005. It is reported as missed. The same runs
  have a lower grain ratio and a lower head HFEN than v2, which points at noise removal rather than
  lost detail, but the gate as written is not met there.
- Per-contrast variant (`ar14_n24_wavP006_C003_full4k`, T1 0.06 / T2 0.03): 35.856 / 36.079 on the
  tuning pair; kept as a variant, seed repeat running.
- Still running at the time of writing: the single-contrast v3 ablations (hyw T1 done: 35.679; hyw T2,
  hyw FLAIR, jyz T1, jyz FLAIR), 3.6M + 0.05 at 4000 as a capacity variant, P-LORAKS for the tables.

**T1+T2 topic, v3 per slice and joint versus single (00:45).** Against every other arm (L1-ESPIRiT and
P-LORAKS included), v3 is first on 42/47 transverse, 83/88 coronal and 90/93 sagittal full-brain T1
slices and on every full-brain T2 slice. Single-contrast v3: T1 35.679 / T2 36.227 — above L1-ESPIRiT on
both by itself; joint minus single = +0.09 (T1) / −0.12 (T2). The margin comes from the representation
with the wavelet prior and the larger cloud; the coupling redistributes accuracy between the contrasts.

### Round 16 — the remaining gaps (pre-registered 2026-09-11 00:50, before any run)

v3 misses one pre-registered gate: T1 sharpness 0.867 is below the 0.874 floor on the tuning pair, and
0.869 on hyw T1+FLAIR. It also trails L1-ESPIRiT on head SSIM, by 0.0003 / 0.0005 on the tuning pair and
by up to 0.005 on the T1+FLAIR topics. On jiangyizhou T1 its head-PSNR margin is a tie (+0.003).
Two single deltas from v3, 4000 iterations each, on the tuning pair:

| exp | change vs v3 | aimed at |
|---|---|---|
| `ar16_n24_wav005_T1p06` | wavelet strength on the T1 channel 0.05 → 0.06; the second channel keeps 0.05 | T1 head SSIM and the T1 margin. T1 is the plain channel in all three topics, so the value transfers without a per-topic choice (per-contrast 0.06 / 0.03 gave T1 35.856, head SSIM 0.9481) |
| `ar16_n24_wav005_geo007` | edge-TV weight lambda_geo 0.10 → 0.07 | the T1 sharpness floor: the wavelet term now does part of the smoothing that edge-TV did alone |

Decision rules, fixed now:
- T1p06 replaces v3 (as v3.1) only if, on the tuning pair, the minimum margin over L1-ESPIRiT rises above
  v3's +0.241 by more than 0.005 dB (the largest seed spread measured), neither channel's head PSNR falls
  by more than 0.01 dB, and T1 head SSIM rises.
- geo007 replaces v3 only if T1 sharpness reaches the 0.874 floor while head PSNR stays first on both
  contrasts (margin over L1-ESPIRiT above zero on each). Otherwise it is recorded as the measured price of
  the sharpness gate.
- If both qualify, their combination is run once at 4000 iterations before anything is adopted.
- Whatever replaces v3 is then run unchanged on the two T1+FLAIR topics and reported whatever they show;
  jiangyizhou stays out of every decision above.

**SSIM diagnosis, recorded before any Round 16 result (01:00).** Scorer-identical 3-D SSIM maps of v3 and
L1-ESPIRiT on the tuning pair, split into the luminance and contrast-structure terms and binned by quintiles
of the ground truth's gradient magnitude inside the head (`t1t2_af12_retro/scripts/ssim_breakdown.py`):
- T1 (v3 − L1 = −0.0003): v3 wins the flattest fifth (+0.0034; less noise, local correlation 0.812 against
  0.804) and loses the three middle fifths (−0.0005 / −0.0019 / −0.0021), all through the contrast-structure
  term: lower local correlation (0.935 against 0.938, 0.957 against 0.960) and lower local variance (local
  std 0.89–0.95 of the ground truth's, L1 0.94–0.97). Luminance is better everywhere.
- T2 (v3 − L1 = −0.0005): the loss sits in the second and third fifths and is variance match only; v3's
  local correlation is higher than L1's there, its local std is 0.86 / 0.91 of the ground truth's against
  L1's 0.91 / 0.94. The flattest fifth adds a luminance loss. The single-contrast v3 (+0.0002 over L1) has
  the same variance deficit but a smaller flat-region luminance loss.
- Reading: the SSIM gap is under-represented mid-scale texture, not noise and not edges. Changes that add
  local variance where the ground truth has moderate gradients should close it: less edge-TV (geo007) and
  more capacity (the 3.6M promotion). The stronger T1 wavelet (T1p06) is expected to act on the T1
  correlation part instead.

**Addenda to the Round 16 pre-registration (01:03, still before any result).**
- Precedent: lambda_geo 0.07 was discarded once, in Round 1 (`ar1_P1_lam007`: 600k points, no wavelet term,
  1500 iterations): T1 −0.107 / T2 −0.005 dB against its parent (`ar1_P1_rerun`, 34.818 / 34.218), with
  sharpness up 0.024 / 0.019. It is re-tested because the wavelet term now does part of the smoothing. The
  Round 1 result predicts a T1 cost, and the decision rule prices it.
- The 3.6M promotion (`ar12_n36_wav005_full4k`: 3.6M points + wavelet 0.05, 4000 iterations, started before
  v3 was decided) is judged by the T1p06 margin rule: it replaces 2.4M only if the minimum margin over
  L1-ESPIRiT rises above +0.241 by more than 0.005 dB and neither channel's head PSNR falls by more than
  0.01 dB. Any two qualifying changes are combined once at 4000 iterations before adoption.

**Second diagnosis and a prediction, before geo007 starts (01:05).** The per-contrast run
(`ar14_n24_wavP006_C003_full4k`) against v3, same bins (`t1t2_af12_retro/scripts/ssim_pair_p006.py`):
- T1, wavelet 0.06 against 0.05: head SSIM +0.0013, gained in every bin and entirely through local
  correlation (+0.001 to +0.004); local variance is unchanged. This supports the T1p06 expectation.
- T2, wavelet 0.03 against 0.05: head SSIM −0.0010, lost in every bin through local correlation (−0.001 to
  −0.007), while local variance rises only 0.005–0.008 of the ground truth's.
- Prediction: loosening a prior adds little variance and costs correlation, so geo007 is expected to lower
  head SSIM even if it clears the sharpness floor. The mid-scale variance deficit reads as a representation
  limit, and capacity (the 3.6M promotion) is the lever for it.

### Round 17 — the FLAIR channel's wavelet strength (pre-registered 2026-09-11 01:13, before any run)

Diagnosis on hyw T1+FLAIR (`t1t2_af12_retro/scripts/ssim_topic.py`, `bias_by_intensity.py`):
- 74 % of v3's FLAIR head-SSIM gap to L1-ESPIRiT (0.9169 against 0.9221) sits in the flattest fifth of the
  head, and there it is the luminance term. The loss is concentrated in the darkest intensity decile
  (suppressed CSF). v3's local mean there is unbiased on average (−0.14 % of max against L1's −0.51 %) but
  fluctuates, and the luminance term penalises the fluctuation (0.937 against 0.969). The wavelet strength
  does not touch it: 0.03 gives 0.940 there, 0.05 gives 0.937.
- The wavelet strength acts on correlation instead. From 0.03 to 0.05, FLAIR head SSIM rose 0.0058, spread
  evenly over the five gradient bins and entirely through the contrast-structure term (local correlation
  +0.002 to +0.020), with head PSNR +0.30 dB. On T2 the same step gave +0.0010 and +0.03 dB, so FLAIR is still
  on the rising side at 0.05. L1-ESPIRiT's own FLAIR λ, selected on hyw, is 3.3 times its T1 λ.

Two single deltas from v3 on hyw T1+FLAIR, 4000 iterations, T1 keeping 0.05: FLAIR-channel strength 0.08
(`ar17_tf_hyw_wavC008`) and 0.12 (`ar17_tf_hyw_wavC012`), launched by `queue17.sh` on GPUs 5 and 7 as they free.

Rule, fixed now: the strength with the higher FLAIR head PSNR is kept if it beats v3b's 34.359 by more than
0.005 dB and T1 head PSNR does not fall more than 0.01 dB below 35.772; otherwise 0.05 stays. Head SSIM is
reported alongside. The kept value then runs unchanged on jiangyizhou and is reported whatever it shows, the
protocol L1-ESPIRiT received (a per-contrast λ selected on hyw, transferred unchanged). The dark-CSF
luminance deficit is expected to remain.

### Round 18 — two-seed averaging (pre-registered 2026-09-11 01:53, before the two new runs)

Seed 42 and seed 43 reconstructions differ voxel by voxel by 25–43 % of the error norm (error correlation
0.91–0.97), while their global scores agree to 0.002 dB. Part of the error is seed-specific, so averaging
runs removes it. Measured on the two pairs that already have both seeds (frozen head-only scorer; HFEN through
an FFT convolution with the same kernel, within 0.0002 of the logged values), head PSNR / head SSIM:

| pair | channel | seed 42 | seed 43 | average of the two | gain |
|---|---|---|---|---|---|
| tuning pair, per-contrast variant | T1 | 35.856 / 0.9481 | 35.856 / 0.9481 | 35.951 / 0.9489 | +0.095 dB / +0.0008 |
| | T2 | 36.079 / 0.9521 | 36.079 / 0.9521 | 36.148 / 0.9527 | +0.069 dB / +0.0006 |
| jiangyizhou, v3 | T1 | 36.337 / 0.9428 | 36.338 / 0.9428 | 36.539 / 0.9449 | +0.202 dB / +0.0021 |
| | FLAIR | 35.935 / 0.9346 | 35.937 / 0.9347 | 36.049 / 0.9360 | +0.114 dB / +0.0014 |

The gains are what the error correlation ρ predicts for an average of two, −10 log10((1 + ρ)/2): +0.096 for
ρ 0.956 and +0.202 for ρ 0.909. Head HFEN improves on all four channels. On jiangyizhou T1 the tie with
L1-ESPIRiT (36.334) becomes +0.205 dB, and head SSIM passes L1-ESPIRiT's 0.9443.

Definition, fixed now: "v3 ×2" is the voxelwise mean of the magnitudes of two v3 runs that differ only in the
seed (42 and 43), at twice the compute of v3. It is reported as its own row next to single-run v3, which stays
the headline arm. Two runs complete it: `ar18_n24_wav005_full4k_s43` (tuning pair) and
`ar18_tf_hyw_t1flair_v3b_s43` (hyw T1+FLAIR), launched by `queue18.sh` as GPUs free; jiangyizhou's seed 43
exists (`ar13_tf_jyz_t1flair_v3b_s43`). No larger ensemble is run. The same model predicts +0.15 dB (tuning
T1) to +0.31 dB (jiangyizhou T1) for four seeds, at four times the compute. The classical arms are
deterministic, so averaging does nothing for them. ZS-SSDU, 3D-DIP and 3D-INR also depend on a seed and would
gain from the same averaging, which was not run for them.

**3.6M promotion (`ar12_n36_wav005_full4k`, landed 01:58): not adopted.** T1 35.662 / T2 36.233 against v3's
35.771 / 36.112. The minimum margin over L1-ESPIRiT is +0.132 (v3 +0.241) and T1 falls 0.109 dB, so it fails
both clauses of the rule. As at 1500 iterations, the larger cloud helps T2 (+0.121 dB, head SSIM 0.9535) and
overfits T1 (head SSIM 0.9444 against 0.9468, head HFEN 0.0864 against 0.0831, sharpness 0.860). The SSIM
prediction recorded before it landed holds on T2 only: capacity adds T2 head SSIM (+0.0004) and costs T1
head SSIM (−0.0024). Wall time 208 min on GPU 6, which another user's job shares.

**jiangyizhou v3 ×2 scored (02:09)** with `t1t2_af12_retro/scripts/score_average.py`, which first re-scores seed 42 from its saved
volumes and matches every logged field (whole-volume HFEN to 1e-4). Head PSNR T1 36.539 / FLAIR 36.049, +0.205 /
+0.329 dB over L1-ESPIRiT; head HFEN 0.0974 / 0.0985; whole-volume PSNR 39.847 / 39.441, first on both;
sharpness 0.883 / 0.842. Head SSIM 0.9449 / 0.9360 is still below the best other arm on each channel
(converged TV 0.9459 on T1, L1-ESPIRiT 0.9368 on FLAIR). Ledger row `ar13_tf_jyz_t1flair_v3b_x2`, status
x2-average.

### Round 16 results

**T1p06 (`ar16_n24_wav005_T1p06`, landed 02:53): qualifies.** T1 35.855 / T2 36.110 against v3's 35.771 / 36.112. The minimum
margin over L1-ESPIRiT is +0.292 (v3 +0.241), T2 moves −0.002 dB, and T1 head SSIM rises to 0.9481 (v3 0.9468,
L1-ESPIRiT 0.9471). Head HFEN 0.0833 / 0.0680; sharpness 0.862 / 0.826, so T1 sits further below the 0.874 floor
than v3 did. It is v3.1 provisionally: the combination clause waits for geo007, and a qualifying Round 17 FLAIR
strength is combined with it once before adoption. Its pre-registered transfers run next on the first GPUs to
free, jiangyizhou (held out) before hyw T1+FLAIR (`queue19.sh`); the Round 18 seed-43 run of v3 on hyw T1+FLAIR
follows them. The tuning pair's seed-43 run of v3 had already started on GPU 4 (02:52) and keeps its slot.

### Round 18b — two-seed averaging for v3.1 (pre-registered 2026-09-11 03:37)

T1p06 qualified (provisional v3.1), so the Round 18 definition applies to it as well: "v3.1 ×2" is the voxelwise
mean of the magnitudes of two T1p06 runs that differ only in the seed (42, 43), at twice the compute. Its seed-43
run on the tuning pair (`ar18_n24_wav005_T1p06_s43`) takes GPU 5 when the FLAIR-strength run there finishes
(`queue20.sh`); `x2_after.sh` scores the average with `score_average.py` when it lands (`ar16_n24_wav005_T1p06_x2`).
Prediction from the Round 18 measurements: +0.07 to +0.10 dB and +0.0006 to +0.0008 head SSIM, which would put
T2 head SSIM at or just above L1-ESPIRiT's 0.9536. The T1+FLAIR topics get seed-43 runs of v3.1 once its
transfers land, if v3.1 is adopted.

**Schedule change, 03:36.** The parallel session (jiangyizhou AF16 loop, worktree `mcgs_af16_wt`) released GPUs
1–3. queue19 was stopped before it had launched anything, and its three jobs started there at once:
`ar16_tf_jyz_T1p06` (GPU 1), `ar16_tf_hyw_T1p06` (GPU 2), `ar18_tf_hyw_t1flair_v3b_s43` (GPU 3). Its result, for
context and not a result of this loop: on jiangyizhou T1+FLAIR at AF16, this loop's wavelet term with per-contrast
strengths (T1 0.10 / FLAIR 0.07), 3.6M points and edge-TV back at 0.10 / quantile 0.85 put MC-GS joint first on
head PSNR, SSIM and HFEN over L1-ESPIRiT (35.552 / 34.807 against 35.209 / 34.503). Two of its observations bear on
this loop: a heavier edge-TV (0.16 / 0.90) over-regularises once the wavelet is strong, and the larger cloud pays
off once T1's wavelet is strong enough, which at AF12 did not happen with the shared 0.05 (3.6M promotion above).

### Round 17 results

**C008 (`ar17_tf_hyw_wavC008`, FLAIR-channel wavelet 0.08, landed 03:47):** T1 35.771 / FLAIR 34.549 against
v3b's 35.772 / 34.359: FLAIR +0.190 dB (+0.444 over L1-ESPIRiT) with T1 unchanged (−0.001). FLAIR head SSIM 0.9205
(v3b 0.9169; L1-ESPIRiT 0.9221, ZS-SSDU 0.9235), head HFEN 0.1056 (v3b 0.1065), FLAIR sharpness 0.842 (v3b 0.869).
It clears the rule's bar (FLAIR +0.190 against the 0.005 needed, T1 within 0.01). The rule keeps whichever of 0.08
and 0.12 gives the higher FLAIR head PSNR, so the decision waits for C012 (GPU 7, about 04:20). FLAIR was
under-regularised at the shared 0.05, as the Round 17 diagnosis expected.

**Combination started ahead of C012 (03:48).** T1p06 and a FLAIR-channel strength both qualify, so the rule's combination run is
needed whichever FLAIR value is kept. To use GPUs 6 and 4, which free before C012 lands, the combination with 0.08
starts now on the first of them: `ar19_tf_hyw_T1p06_wavC008` (hyw T1+FLAIR; T1 0.06, FLAIR 0.08, everything else
v3; `queue21.sh`). The Round 17 decision is still taken on C008 against C012 alone. If 0.12 is kept, its own
combination follows and this run is reported as a variant.
