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

Results (1500 iterations; ~58 min each with two per card):

| exp | T1 | FLAIR | dT1 | dFLAIR | min_margin | head SSIM T1 / FLAIR | head HFEN T1 / FLAIR | sharp T1 / FLAIR | status |
|---|---|---|---|---|---|---|---|---|---|
| ar1_lam016 | 34.611 | 33.771 | +0.202 | −0.233 | −0.233 | 0.9221 / 0.9058 | 0.1320 / 0.1308 | 0.837 / 0.760 | marginal |
| ar1_lam013_kq090 | 34.601 | 33.778 | +0.192 | −0.226 | −0.226 | 0.9218 / 0.9056 | 0.1323 / 0.1316 | 0.841 / 0.758 | marginal |
| ar1_pwc (FLAIR prewhitening) | 34.218 | 33.518 | −0.191 | −0.486 | −0.486 | 0.9131 / 0.9000 | 0.1336 / 0.1315 | 0.905 / 0.824 | discard |

Reading. Stronger regularisation keeps paying on both contrasts with diminishing returns
(λ 0.10 → 0.13 → 0.16: T1 +0.24 then +0.14, FLAIR +0.14 then +0.08); λ 0.13 + κ_q 0.90 lands at the
same point as λ 0.16 — they are the same lever (more smoothing in flat regions). The price is
sharpness (T1 0.90 → 0.84, FLAIR 0.82 → 0.76; still sharper than the TV competitor's 0.77 / 0.66)
and a +0.001 FLAIR HFEN, so by the pre-registered clause these are "marginal", not "keep"; the
promotion run and the visual gate decide. FLAIR prewhitening costs 0.03 dB on both — discarded.
`ar1_base_2500` (budget curve) pending.

### The bar moves (2026-09-10 17:40) — classical arms re-run converged and λ-swept on this subject

The hyw loop's audit (`autoresearch_t1t2.md`, "Classical baselines revisited", commit db69313)
found the CS-SENSE arm of every topic under-converged (35 ADMM iterations inherited from the CE
study): converged TV is 0.35–0.6 dB higher and L1-wavelet ESPIRiT (BART) is first on all five AF12
channels, 0.4–0.8 dB above v2. The AF16 topics used the same under-converged arm, so the bar above
(34.409 / 34.005) is too low. The three revisited arms are being run on jiangyizhou AF16
(`Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/scripts/run_classical_revisited.sh`, GPUs 2–3):
converged TV at λ {0.7, 1, 1.4, 2}e-3 (T1) / {0.5, 0.7, 1, 1.4}e-3 (FLAIR), L1-ESPIRiT at
λ {1, 1.5, 2.5, 4}e-3 (T1) / {3, 5, 7, 10}e-3 (FLAIR), joint vectorial TV at λ {1, 1.4}e-3. **λ is
swept on this subject against the ground truth — exactly the privilege this loop has — and the
best λ per contrast is the competitor.** When they land, the topic's `summary.json` is rebuilt, the
row script picks the new bar up automatically (every row now records its bar in the note), and
`scripts/autoresearch_jyz_af16_view.py` re-reads all rows against it.

GPU split agreed with the hyw loop (17:50): this loop on GPUs 0–3, the hyw loop on 4–7.

### Round 2 — 2026-09-10 17:43, six runs at 1500 iterations
`ar2_P1_lam016_kq090` (the two Round-0/1 winners combined: candidate new parent),
`ar2_lam020` (λ 0.20: where the λ curve turns), `ar2_dtv005` (T1-guided directional TV on FLAIR,
λ_dtv 0.05: a FLAIR-targeted use of the shared anatomy), `ar2_khiFL26` (FLAIR high-band weight 2.6,
T1 kept at 2.2), `ar2_n900k` (0.9M points: capacity under sparser data), `ar2_eps003` (Charbonnier
ε 0.03, after `ar1_base_2500` frees its slot). GPUs 0, 1, 4, 5 (4 and 5 once, before the split).

**Converged TV on this subject (200 outer ADMM, λ swept, 18:15)** — both optima interior:

| λ (T1) | 0.0005 | **0.0007** | 0.001 | 0.0014 |
|---|---|---|---|---|
| T1 head PSNR | 34.834 | **34.945** | 34.867 | 34.631 |
| sharp | 0.856 | 0.823 | 0.789 | 0.755 |

| λ (FLAIR) | 0.00035 | **0.0005** | 0.0007 | 0.001 |
|---|---|---|---|---|
| FLAIR head PSNR | 34.158 | **34.302** | 34.278 | 34.110 |
| sharp | 0.789 | 0.751 | 0.717 | 0.681 |

The 09-09 arm (35 outer, λ 1e-3) read 34.409 / 33.806: convergence plus the λ choice is worth
+0.54 / +0.50 dB here. **Interim bar: T1 34.945 (converged TV), FLAIR 34.302 (converged TV)**;
L1-ESPIRiT (CPU-bound, the host is saturated by the hyw loop's 96 P-LORAKS workers) and joint TV
are still running. Against it the best screened MC-GS is −0.33 (T1) / −0.52 dB (FLAIR). At matched
smoothness (TV's optimum has sharpness 0.82 / 0.75; λ 0.16 MC-GS 0.84 / 0.76) MC-GS is 0.3–0.5 dB
behind, so the smoothing family alone cannot close the gap. Two structural suspects for Round 3:
(1) the Charbonnier ε = 0.04 is larger than typical flat-region gradients of the normalised image,
so the penalty is quadratic (Tikhonov-like) exactly where MC-GS shows its grain, while the
competitor is L1-TV; `ar2_eps003` tests the direction, ε 0.02 / 0.01 follow if it pays;
(2) a wavelet-sparsity term (what makes L1-ESPIRiT strong), which the hyw loop is building
(uncommitted as of 18:10; it will be picked up once it has a screening result there).
