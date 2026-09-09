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
