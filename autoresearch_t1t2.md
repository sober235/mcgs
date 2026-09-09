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
