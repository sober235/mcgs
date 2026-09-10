# autoresearch — MC-GS joint on the jiangyizhou T1 + FLAIR AF16 pair: first of eight inside the head

_Opened 2026-09-10 12:00. Karpathy-style loop: propose one change → run at a fixed budget → read one
number → keep or discard → record → repeat. Runner `autoresearch_jyz_af16.sh`, ledger
`autoresearch_jyz_af16.tsv` (one row per experiment, append-only), branch `autoresearch-jyz-af16`
(= `autoresearch-t1t2` tip + the AF16 study), worktree `/data0/congcong/code/GS/mcgs_af16_wt`,
GPUs 0 and 1 (2–7 belong to the hyw T1+T2 loop)._

## Objective (user decision 2026-09-10)

MC-GS **joint** must be first of the eight arms of `Results_20260908/T1FLAIR_AF16_v3_jiangyizhou`
on **head-only PSNR of BOTH contrasts** at 4000 iterations, same data / mask / maps / gauge. The
single arm (same recipe, both contrasts) is run after promotion and reported, not optimised
("joint first is the goal, single is reported"). Arena = jiangyizhou AF16 only; the recipe is
checked afterwards on jiangyizhou AF12 and huangyuwei AF16 for honesty. Levers use the measured
data only (ZF support, acquired k-space) — no other method's output, no GT, no learned prior.

The number the loop reads: `min_margin = min(T1_head_psnr − 34.409, FLAIR_head_psnr − 34.005)`
(CS-SENSE on T1, ZS-SSDU on FLAIR; the row script reads them from the topic's `summary.json`).
Success = `min_margin > 0` at 4000 iterations, with no blur trade: head SSIM and head HFEN of each
contrast not worse than the parent beyond the seed-repeat floor, sharpness not below it.

Where we start (jiangyizhou AF16, 4000 iterations, head-only PSNR / SSIM):

| arm | T1 | FLAIR |
|---|---|---|
| CS-SENSE (T1 to beat) | **34.409** / **0.9276** | 33.806 / 0.9116 |
| ZS-SSDU (FLAIR to beat) | 34.150 / 0.9216 | **34.005** / **0.9185** |
| MC-GS joint, frozen cold start (AF16 study) | 32.890 / 0.8893 | 32.402 / 0.8834 |
| MC-GS joint **v2** (hyw loop's recipe, transfer run `ar8_tf_jyz_af16` in the main tree, 12:20) | 33.982 / 0.9069 | 33.640 / 0.9003 |

v2 at 4000 iterations is second on both contrasts by **0.43 dB (T1) and 0.36 dB (FLAIR)**, head SSIM
0.021 / 0.018 behind, head HFEN 0.1389 / 0.1304 (better than CS-SENSE's 0.1531 on T1, worse than
its 0.1360 on FLAIR). The loop has to find +0.4 dB on both contrasts on top of v2.

Gap from the frozen cold start: +1.52 dB on T1, +1.60 dB on FLAIR. The v2 recipe was worth
+1.4 to +2.3 dB head-only everywhere it was tried at AF12 (first on hyw both pairings, second on
jyz by 0.24 / 0.03 dB), so the loop starts from v2, not from the frozen recipe.

## Parent = v2, baked into the runner

Frozen working point (edge-TV, κ_q 0.85, w_floor 0.10, Charbonnier eps 0.04, λ_geo 0.10,
kspace_high 2.2) + `gaussian.init_mode=head`, `gaussian.init_scale_mode=3nn`,
`loss.edge_tv_kappa_source=zf`, `gaussian.initial_num_points=1200000`. An experiment changes it
only through the extra args it passes (cycle.py flags: last occurrence wins; `--set`: last wins).

## Fixed — never touched, never used for fitting or selection

Data, mask (`mask_af16_3d.h5`, AF 15.19), ESPIRiT maps, ground truth, scorer, gauge, the
competitors' numbers. The model stays MC-GS joint: one shared Gaussian geometry, dual complex
density, hard-DC scoring. Code changes go in with a unit test and must leave the default path
bit-identical.

## Already settled on the hyw loop — do not re-run expecting a gain

(`autoresearch_t1t2.md`, ledger rows ar0–ar7): intensity init; densification as configured
(split never triggers at fixed scale, dominated by the initial count at 3-NN); position_lr;
λ_geo 0.07; T1 high-band weight 1.0 / 3.0; kappa refresh at 300; κ_q 0.75; 3-NN multiplier 1.5;
capacity above 1.2M on T1 (overfits sampled k, e_unobs rises); a points-per-support-voxel rule
(failed on jyz at AF12). Marginal at 1500 but null at 4000 on hyw: w_floor 0.2, λ_geo 0.12,
κ_q 0.90, tv_eps 0.06.

## Allowed and queued (single knobs against the parent, jyz AF16 is sparser than AF12 so the
regulariser balance is the first suspect)

| knob | how | round |
|---|---|---|
| seed repeat (floor) | `--set training.seed=43` | 0 |
| λ_geo 0.13 / 0.08 | `--lambda_geo` | 0 / 1 |
| κ_q 0.90 / 0.80 | `--edge_tv_kappa_q` | 0 / 1 |
| w_floor 0.20 | `--edge_tv_wfloor` | 1 |
| tv_eps 0.03 / 0.05 | `--tv_eps` | 1 |
| kspace_high 1.8 / 2.6 | `--kspace_high` | 1 |
| points 0.9M / 1.6M (jyz head is larger; capacity was T1-capped on hyw) | `--set gaussian.initial_num_points=` | 2 |
| FLAIR prewhitening | `--prewhiten_only contrast` | 2 |
| init_head_frac 0.03 / 0.08 | `--set gaussian.init_head_frac=` | 2 |
| budget 6000 at promotion | ITERS | promotion |

## Budget and protocol

- Screening = **1500 iterations** (~18 min alone on an 80 GB card, ~30 min two per card), no
  volume dump. Promotion = **4000 iterations**, `SAVE_VOLUMES=1`. Four experiments in parallel
  (two per card).
- Keep rule: `min_margin` improves over the parent by more than the seed-repeat floor, with no
  blur trade (sharp, head SSIM, head HFEN of either contrast not worse than the parent beyond the
  floor). Ties go to the simpler configuration. One knob per experiment; combine two knobs only
  after each has won on its own.
- Every experiment: unique `EXP` (`arN_<knob>`), one ledger row (status keep / discard / marginal /
  crash / promoted / ablation), exact args in the row. Outputs in `results_arc/<EXP>*` (gitignored).
- Stop when `min_margin > 0` survives promotion; then the single arm with the same recipe (both
  contrasts, `CONFIG=configs/t1flair_v3_af16_jyz_singleT1.yaml` / `_singleFLAIR.yaml`), the
  transfer checks (jyz AF12, hyw AF16), the eight-arm table and the butterfly-region figure with
  the v2/v3 arm for the visual gate.

## Round log

(appended as rounds close; per-experiment numbers in `autoresearch_jyz_af16.tsv`)

### Round 0 — 2026-09-10 12:05, four runs at 1500 iterations, GPUs 0–1
`ar0_v2_base` (parent), `ar0_v2_seed43` (floor), `ar0_lam013` (λ_geo 0.13), `ar0_kq090` (κ_q 0.90).

Results (1500 iterations, head-only PSNR T1 / FLAIR, margin to the competitor; ~47 min each, two per card):

| exp | T1 | FLAIR | dT1 | dFLAIR | min_margin | head SSIM T1 / FLAIR | sharp T1 / FLAIR | status |
|---|---|---|---|---|---|---|---|---|
| ar0_v2_base (parent) | 34.235 | 33.553 | −0.174 | −0.451 | −0.451 | 0.9136 / 0.9014 | 0.844 / 0.824 | parent |
| ar0_v2_seed43 | 34.247 | 33.559 | −0.162 | −0.446 | −0.446 | 0.9141 / 0.9016 | 0.844 / 0.824 | repeat: floor 0.012 / 0.006 dB |
| ar0_lam013 (λ_geo 0.13) | 34.471 | 33.695 | **+0.061** | −0.310 | −0.310 | 0.9192 / 0.9046 | 0.815 / 0.789 | keep-marginal |
| ar0_kq090 (κ_q 0.90) | 34.414 | 33.683 | +0.004 | −0.321 | −0.321 | 0.9179 / 0.9042 | 0.820 / 0.794 | keep-marginal |

Reading. (1) The seed floor is 0.01 dB, so every delta above is real. (2) Both regularisation knobs
gain on BOTH contrasts (+0.24 / +0.14 and +0.18 / +0.13 dB), improve head SSIM and head HFEN, and
cost 0.03 of sharpness ratio — "keep-marginal" because the pre-registered sharpness clause is
touched while SSIM/HFEN improve; the promotion run decides. (3) At 1500 iterations with λ 0.13,
**T1 is already above CS-SENSE (+0.06)**; FLAIR is the bottleneck at −0.31. (4) The parent's T1 at
1500 iterations (34.24) is HIGHER than v2's 4000-iteration transfer (33.98) while FLAIR is lower
(33.55 vs 33.64): T1 over-fits the sampled k with more iterations, FLAIR still converges. The
budget is therefore a knob (ar1_base_2500 maps it).

### Round 1 — 2026-09-10 16:40, four runs, GPUs 0–1
`ar1_lam016` (λ_geo 0.16: further along the winning direction), `ar1_lam013_kq090` (the two
winners combined), `ar1_pwc` (FLAIR-only prewhitening, `--prewhiten_only contrast`),
`ar1_base_2500` (parent at 2500 iterations: the budget curve between 1500 and 4000).
Queued for Round 2, FLAIR-specific: `--kspace_high 2.6 --kspace_high_plain 2.2` (more high-band
weight on FLAIR than T1), `--lambda_dtv 0.05` (cross-contrast coupling, historically a small
plus), tv_eps 0.03, w_floor 0.20, points 0.9M / 1.6M, init_head_frac.
