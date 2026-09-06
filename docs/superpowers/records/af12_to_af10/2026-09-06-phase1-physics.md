# Phase 1 — known-zero region + prewhitening at AF12 (physics levers, no learnable prior)

Spec: `docs/superpowers/specs/2026-09-05-af12-to-af10-quality-design.md` §4.5 (runs and the `base1` rule) and appendix A (R1/R2/R1', which replaced §4.1/§4.2's ring-cliff estimator with the zero-line-shape estimator actually used below).
Task brief: `.superpowers/sdd/2026-09-05-af12-to-af10-phase0-phase1/task-8-brief.md`.
Branch `af12-to-af10` @ `9a7fbdc` (parent of the commit this record is part of).

## Commands (exact, as executed)

Both launched with `nohup ... &` at `2026-09-06T09:58:28Z` (17:58:28 CST) on two idle GPUs, chosen at launch time (`nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits` showed all 8 cards at 14 MiB / 0% util; GPU 0 and 1 used, matching the brief's default):

```bash
nohup env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID /home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py \
  --exp af12_kz --gpu 0 --iters 4000 --lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 --base_cfg configs/winner_v2_edgeTV.yaml \
  --save_volumes --known_zero > results_arc/af12_kz.log 2>&1 &

nohup env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID /home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py \
  --exp af12_kz_pw --gpu 1 --iters 4000 --lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 --base_cfg configs/winner_v2_edgeTV.yaml \
  --save_volumes --known_zero --prewhiten > results_arc/af12_kz_pw.log 2>&1 &
```

(Same flag set as the brief's `$COMMON`; written out literally here because the interactive shell is zsh, where an unquoted multi-word variable in command position does not word-split the way the brief's bash-style snippet assumes — confirmed by a failed first attempt (`nohup: failed to run command 'env ... python -u'`) before switching to literal commands. No flag or value differs from the brief.)

| run | GPU | launch (UTC) | done (UTC, log/npz mtime) | elapsed (wall) | script `elapsed_s` | peak VRAM |
|---|---|---|---|---|---|---|
| `af12_kz` | 0 | 09:58:28 | 10:53:24 | 54m57s | 3279 (train+score; +volume save) | 20.5 GB |
| `af12_kz_pw` | 1 | 09:58:28 | 10:49:32 | 51m04s | 3046 | 26.0 GB |

No traceback in either log; both printed `DONE -> .../results_grain.tsv`.

## Z / noise physics diagnostics, vs the oracle

The known-zero region `Z` is estimated once per contrast from the sampled k-space alone (rule R1/R1', appendix A); it does not depend on `--prewhiten`, and both runs recovered *bit-identical* cuts/ellipse/frac_z/outliers — confirmed by diffing the `[physics:...]` log lines and the score-JSON `physics` blocks of `af12_kz` and `af12_kz_pw`.

| contrast | cuts | ellipse (a, b, expand) | frac_z_used | n_zero_lines/explained (of 4760 sampled) | n_nonzero_in_z | n_outliers / n_significant_outliers | kz_energy (both runs) |
|---|---|---|---|---|---|---|---|
| T1 (plain) | axis1_lo=44, others none | none | 0.200000 | 383 / 383 | 0 | 0 / 0 | 0.0 |
| FLAIR (contrast) | none | (123.0, 88.0, 1.0) | 0.395898 | 772 / 735 | 0 | 36 / 0 | 0.0 |
| **oracle** (T1, ground truth, full k-space, `known_zero_diagnostics.md`) | — | none | true_zero_fraction 0.200 (frac_z 0.200) | — | — | precision 1.000, recall 1.000 | 0 false-positive lines |
| **oracle** (FLAIR, ground truth) | — | (123.0, 88.0, 1.0) | true_zero_fraction 0.3917 (frac_z 0.3959) | — | — | precision 0.9486, recall 0.9588 | 1146 false-positive lines, all below the noise floor (max 0.964×, excess-over-noise energy 0) |

Both hard invariants hold on the real runs: `n_nonzero_in_z = 0` for both contrasts/both runs, and `kz_energy = 0.0` for both contrasts/both runs (Z never overwrites sampled non-zero energy).

Noise / prewhitening (§4.2 rule R2 — Σ from the readout-end 1/8-index samples of all non-zero sampled lines; W = Σ^(−1/2)·sqrt(mean(diag Σ))):

| run | contrast | sigma_source | n_noise_samples | coil_corr_max_offdiag | noise_level | (eig_ratio) |
|---|---|---|---|---|---|---|
| `af12_kz` (prewhiten off) | T1 | not computed | not computed | not computed | not computed (`noise_level_used`=3.154294e-4 for the R1' outlier-significance gate only) | — |
| `af12_kz` (prewhiten off) | FLAIR | not computed | not computed | not computed | not computed (`noise_level_used`=4.831003e-4) | — |
| `af12_kz_pw` (prewhiten on) | T1 | readout_ends | 280128 | 0.560603 | 3.154294e-4 | 12.172 |
| `af12_kz_pw` (prewhiten on) | FLAIR | readout_ends | 255232 | 0.431198 | 4.831003e-4 | 19.678 |

(`af12_kz`'s `noise_level_used` and `af12_kz_pw`'s `noise_level` agree to 4 significant figures per contrast — both are the same readout-end noise floor, computed via two slightly different code paths [R1' significance gate vs R2 Σ], hence differing from the 9th significant digit; not a discrepancy.)

## Metrics: run × contrast vs the AF12 baseline and the AF10 reference

| run | contrast | head_psnr | head_ssim | head_hfen | N (grain_ratio) | e_unobs | whole PSNR |
|---|---|---|---|---|---|---|---|
| `tvs4k_e040` (AF12 baseline) | T1 | 33.922017 | 0.904705 | 0.169392 | 0.959902 | 0.648788 | 38.379 |
| `tvs4k_e040` (AF12 baseline) | FLAIR | 33.624357 | 0.925759 | 0.136522 | 0.948961 | 0.644410 | 38.221 |
| `af10_joint_edgetv_e040_rerun20260904` (AF10 reference) | T1 | 34.604925 | 0.915227 | 0.153519 | 0.952 | 0.635300 | 38.945 |
| `af10_joint_edgetv_e040_rerun20260904` (AF10 reference) | FLAIR | 34.400857 | 0.935228 | 0.122247 | 0.949 | 0.629800 | 38.867 |
| **`af12_kz`** | **T1** | 34.276970 | 0.909499 | 0.167053 | 0.951875 | 0.632776 | 38.707201 |
| **`af12_kz`** | **FLAIR** | 33.976881 | 0.929381 | 0.135225 | 0.942868 | 0.627476 | 38.563734 |
| **`af12_kz_pw`** | **T1** | 34.199419 | 0.912518 | 0.171761 | 0.919313 | 0.630854 | 38.654698 |
| **`af12_kz_pw`** | **FLAIR** | 34.341124 | 0.935353 | 0.129842 | 0.903360 | 0.617459 | 38.860607 |

## Per-gate verdicts (spec §2, `gates.json`, `scripts/eval_gates.py`)

```
af12_kz    T1     2/5 head_psnr=FAIL(34.2770,margin -0.3230) head_ssim=FAIL(0.9095,margin -0.0052) head_hfen=FAIL(0.1671,margin -0.0125) e_unobs=PASS(0.6328,margin +0.0160) grain_ratio=PASS(0.9519,margin +0.0481)
af12_kz    FLAIR  2/5 head_psnr=FAIL(33.9769,margin -0.4190) head_ssim=FAIL(0.9294,margin -0.0053) head_hfen=FAIL(0.1352,margin -0.0120) e_unobs=PASS(0.6275,margin +0.0169) grain_ratio=PASS(0.9429,margin +0.0571)
af12_kz_pw T1     2/5 head_psnr=FAIL(34.1994,margin -0.4005) head_ssim=FAIL(0.9125,margin -0.0022) head_hfen=FAIL(0.1718,margin -0.0172) e_unobs=PASS(0.6309,margin +0.0179) grain_ratio=PASS(0.9193,margin +0.0807)
af12_kz_pw FLAIR  3/5 head_psnr=FAIL(34.3411,margin -0.0547) head_ssim=PASS(0.9354,margin +0.0006) head_hfen=FAIL(0.1298,margin -0.0066) e_unobs=PASS(0.6175,margin +0.0270) grain_ratio=PASS(0.9034,margin +0.0966)
```

Neither run passes the FLAIR (or T1) gate outright. `af12_kz_pw` FLAIR comes closest: 3/5, missing only `head_psnr` (margin −0.0547 dB) and `head_hfen` (margin −0.0066); its `head_ssim` newly clears the AF10-reference gate (margin +0.0006).

**Caveat on `e_unobs` (final-review addition, 2026-09-06):** every `e_unobs` value in this record improves on `tvs4k_e040`'s AF12 gate (`e_unobs=PASS` on every run/contrast row above and in the addendum below), but part of that improvement is mechanical, not physical. `Z` is a subset of the unobserved set `(1−Ω)`, and `Z\Ω` — the part of `Z` outside the acquired mask — is ≈36% of k-space for FLAIR and ≈20% for T1. Inside `Z`, `mask_fit`/`kspace_dc` force the fit-side loss and hard-DC to drive the render toward `y_full ≈ 0` (the vendor shutter really is numerically zero there), so `e_unobs` — computed over the full `(1−Ω)`, which includes `Z\Ω` — gets an easy, guaranteed-zero-residual region baked in; it does not by itself demonstrate improved prediction over the genuinely-unknown k-space. The gate itself is one-sided (`e_unobs ≤` threshold) and is unaffected by this caveat: every `PASS` above still holds under its original definition. Phase 2 screens new runs against `base1_1500`, which carries the same `Z`, so the mechanical component is present in the comparison baseline too. Going forward, `arc/cycle.py::score_contrast` also reports `e_unobs_excl_z` (the same relative error restricted to `unobserved & ~Z`), which strips out this mechanical component; it was not computed for the runs in this record.

Verdict JSONs: `docs/superpowers/records/af12_to_af10/scores/af12_kz_verdict.json`, `.../af12_kz_pw_verdict.json`. Appended to `decisions.jsonl` (phase 1, two lines, one per run).

## Visual gate

**Pending user.** Figures for judging (butterfly z=160 vs the AF10 reference, plus 3-view GT/ZF/recon/error panels; error maps use jet, fixed [0, 0.1] of GT peak, per convention):

- `http://localhost:8765/af12_to_af10/af12_kz_vs_af10_T1.png`
- `http://localhost:8765/af12_to_af10/af12_kz_vs_af10_FLAIR.png`
- `http://localhost:8765/af12_to_af10/af12_kz_pw_vs_af10_T1.png`
- `http://localhost:8765/af12_to_af10/af12_kz_pw_vs_af10_FLAIR.png`
- `http://localhost:8765/af12_to_af10/af12_kz_T1_3views.png`
- `http://localhost:8765/af12_to_af10/af12_kz_FLAIR_3views.png`
- `http://localhost:8765/af12_to_af10/af12_kz_pw_T1_3views.png`
- `http://localhost:8765/af12_to_af10/af12_kz_pw_FLAIR_3views.png`

No new artefacts, patchiness, or coherent banding have been screened by the user yet; this record does not assert a visual pass or fail.

## `base1` decision (spec §4.5 rule)

Rule, in order: (1) more FLAIR gates passed; (2) tie → lower FLAIR `e_unobs`; (3) tie → `af12_kz_pw`. First disqualify any switch that regresses any of `{head_psnr, head_ssim, head_hfen}` (either contrast) beyond its registered tolerance (`gates.json` `tolerance_floor`: 0.005 dB / 0.0005 / 0.001) relative to `tvs4k_e040`.

**Disqualification check** (value vs. `tvs4k_e040 ± tol`):

- `af12_kz`: T1 head_psnr +0.354953, head_ssim +0.004794, head_hfen −0.002339 (better); FLAIR head_psnr +0.352524, head_ssim +0.003623, head_hfen −0.001297 (better). **No regression on any of the six values — eligible.**
- `af12_kz_pw`: T1 head_psnr +0.277402 (better), head_ssim +0.007812 (better), **head_hfen +0.002369 (worse: 0.171761 vs. baseline 0.169392 + tol 0.001 = 0.170392 ceiling — exceeds it by 0.001369)**; FLAIR head_psnr +0.716767, head_ssim +0.009594, head_hfen −0.006680 (all better). **T1 head_hfen regresses beyond tolerance → `af12_kz_pw` is disqualified from `base1`.**

Because `af12_kz_pw` is disqualified, the FLAIR-gates-passed tie-break is moot: `af12_kz_pw` would otherwise have won it outright (3/5 vs. `af12_kz`'s 2/5), but the disqualification clause is applied first per the rule as written.

**`base1` = `af12_kz`** (known-zero region only, no prewhitening) — selected by elimination, the only eligible switch.

This is a real trade-off, not a rounding artefact: prewhitening (`af12_kz_pw`) buys FLAIR most of its remaining head-PSNR/SSIM gap (closes 92% of the 0.777 dB baseline gap, and newly clears the FLAIR head_ssim gate) at the cost of a small but tolerance-exceeding T1 head_hfen regression (T1 grain_ratio also drops further, 0.9193 vs. 0.9519 for `af12_kz`, i.e. T1 gets measurably smoother/less textured under prewhitening than without it). Per the pre-registered rule this disqualifies it from the Phase 2 base regardless of its FLAIR advantage.

## What T1 gained (recorded, not judged)

Relative to the AF12 baseline `tvs4k_e040` (T1 head_psnr 33.922017 / head_ssim 0.904705 / head_hfen 0.169392, gap to the AF10 reference 34.604925 was 0.682908 dB):

- `af12_kz`: head_psnr +0.354953 dB (34.276970; 52.0% of the baseline gap closed, 0.327955 dB remaining), head_ssim +0.004794 (0.909499), head_hfen −0.002339 (0.167053, i.e. improved/lower), e_unobs −0.016012 (0.632776), grain_ratio −0.008028 (0.951875).
- `af12_kz_pw`: head_psnr +0.277402 dB (34.199419; 40.6% of the gap closed, 0.405506 dB remaining), head_ssim +0.007812 (0.912518), head_hfen +0.002369 (0.171761, i.e. regressed/higher, see disqualification above), e_unobs −0.017935 (0.630854), grain_ratio −0.040590 (0.919313).

Both switches move T1 head_psnr and head_ssim in the same direction (up) relative to the AF12 baseline; neither closes the full AF10 gap on any contrast. Under `af12_kz` (no prewhitening) T1 and FLAIR gain almost the same head_psnr (+0.354953 vs. +0.352524 dB — T1 fractionally more). Prewhitening is what splits the two contrasts apart: `af12_kz_pw` gives FLAIR far more than T1 (head_psnr +0.716767 dB vs. +0.277402 dB) — consistent with the spec's own expectation for this phase ("FLAIR +0.3~0.6 dB, T1 ≤ +0.4 dB") for T1, except `af12_kz_pw`'s FLAIR gain (+0.717 dB) exceeds the +0.3~0.6 dB expectation window on the high side.

## Files

- Score JSONs: `docs/superpowers/records/af12_to_af10/scores/{af12_kz_score.json, af12_kz_pw_score.json}` (copied from `results_arc/`, which is git-ignored).
- Verdict JSONs: `docs/superpowers/records/af12_to_af10/scores/{af12_kz_verdict.json, af12_kz_pw_verdict.json}`.
- `docs/superpowers/records/af12_to_af10/decisions.jsonl`: two phase-1 lines appended (one per run), each carrying the `base1` outcome and `"visual gate: pending user"` in its note.
- `docs/superpowers/records/af12_to_af10/reference_table.md`: appended a line under the table stating the `tvs4k_e040` e_unobs replay values (T1 0.6488, FLAIR 0.6444; checkpoint `latest.pth` @ iteration 4000, `best.pth` confirmed same iteration).
- Figures: `docs/figures/af12_to_af10/{af12_kz,af12_kz_pw}_{T1,FLAIR}_3views.png`, `docs/figures/af12_to_af10/{af12_kz,af12_kz_pw}_vs_af10_{T1,FLAIR}.png` (8 new files), copied to `/home/congcongliu/figs/af12_to_af10/`; figure server verified 200 on the index and on all 8 files.
- Full volumes (git-ignored, not committed): `results_arc/af12_kz_volumes.npz`, `results_arc/af12_kz_pw_volumes.npz`.

## Addendum: `af12_kz_pwc` (Task 12, per-contrast prewhitening)

Task brief: `.superpowers/sdd/2026-09-05-af12-to-af10-phase0-phase1/task-12-brief.md`. Why: FLAIR gained most of its remaining head-PSNR/SSIM gap under full prewhitening (`af12_kz_pw`), but that run's T1 `head_hfen` regressed beyond tolerance vs `tvs4k_e040`, disqualifying it from `base1` above. Since Z/W are now per-contrast switches (`data.<contrast>.prewhiten`, CLI `--prewhiten_only {plain,contrast}`; see the Step 1 commit for the trainer/CLI diff), this run applies Z to both contrasts but W to FLAIR only, leaving T1 unprewhitened.

### Command (exact, as executed)

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID /home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py \
  --exp af12_kz_pwc --gpu 0 --iters 4000 --lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 --base_cfg configs/winner_v2_edgeTV.yaml \
  --save_volumes --known_zero --prewhiten_only contrast > results_arc/af12_kz_pwc.log 2>&1
```

| run | GPU | launch (UTC) | done (UTC) | elapsed (wall) | script `elapsed_s` | peak VRAM |
|---|---|---|---|---|---|---|
| `af12_kz_pwc` | 0 (idle before launch, memory.used=14 MiB) | 2026-09-06T11:21:33Z | 2026-09-06T12:13:34Z | 52m01s | 3104 | 23.3 GB |

No traceback in the log; printed `[arc af12_kz_pwc] DONE -> .../results_grain.tsv`.

### Z / noise physics diagnostics

Z (rule R1/R1') is bit-identical to `af12_kz`/`af12_kz_pw` for both contrasts — confirmed by diffing the score-JSON `physics` blocks (cuts, ellipse, `frac_z_used`, outliers all match the table in the base record above). `n_nonzero_in_z = 0` and `kz_energy = 0.0` hold for both contrasts (the two hard invariants). Prewhitening ran on FLAIR only, per the new per-contrast switch:

| contrast | known_zero | prewhiten | sigma_source | n_noise_samples | coil_corr_max_offdiag | noise_level (eig_ratio) |
|---|---|---|---|---|---|---|
| T1 (plain) | True | **False** | — (not computed) | — | — | — (`noise_level_used` for the R1' outlier gate only: 3.154294e-4) |
| FLAIR (contrast) | True | **True** | readout_ends | 255232 | 0.431198 | 4.831003e-4 (eig_ratio 19.678) |

FLAIR's W diagnostics agree with `af12_kz_pw`'s FLAIR values to 6 significant figures (n_noise_samples identical, `coil_corr_max_offdiag` 0.431198 vs 0.431198, `eig_ratio` 19.678 vs 19.678) — confirms the per-contrast switch reproduces the same per-contrast W computation as the global switch did, just gated to one contrast.

### Metrics: three runs × two contrasts, vs the AF12 baseline and the AF10 reference

| run | contrast | head_psnr | head_ssim | head_hfen | N (grain_ratio) | e_unobs | whole PSNR |
|---|---|---|---|---|---|---|---|
| `tvs4k_e040` (AF12 baseline) | T1 | 33.922017 | 0.904705 | 0.169392 | 0.959902 | 0.648788 | 38.379 |
| `tvs4k_e040` (AF12 baseline) | FLAIR | 33.624357 | 0.925759 | 0.136522 | 0.948961 | 0.644410 | 38.221 |
| `af10_joint_edgetv_e040_rerun20260904` (AF10 reference) | T1 | 34.604925 | 0.915227 | 0.153519 | 0.952 | 0.635300 | 38.945 |
| `af10_joint_edgetv_e040_rerun20260904` (AF10 reference) | FLAIR | 34.400857 | 0.935228 | 0.122247 | 0.949 | 0.629800 | 38.867 |
| `af12_kz` | T1 | 34.276970 | 0.909499 | 0.167053 | 0.951875 | 0.632776 | 38.707201 |
| `af12_kz` | FLAIR | 33.976881 | 0.929381 | 0.135225 | 0.942868 | 0.627476 | 38.563734 |
| `af12_kz_pw` | T1 | 34.199419 | 0.912518 | 0.171761 | 0.919313 | 0.630854 | 38.654698 |
| `af12_kz_pw` | FLAIR | 34.341124 | 0.935353 | 0.129842 | 0.903360 | 0.617459 | 38.860607 |
| **`af12_kz_pwc`** | **T1** | 34.262346 | 0.909224 | 0.168038 | 0.957169 | 0.634223 | 38.700043 |
| **`af12_kz_pwc`** | **FLAIR** | 34.294122 | 0.934782 | 0.130339 | 0.901697 | 0.617442 | 38.827842 |

Relative to `af12_kz` (Z only, no W anywhere): `af12_kz_pwc`'s T1 is essentially unchanged (head_psnr −0.014624 dB, head_hfen +0.000985 — both within joint-training noise from the shared Gaussian geometry seeing a differently-scaled FLAIR gradient, not a deliberate T1 lever) while FLAIR gains head_psnr +0.317241 dB and head_hfen improves −0.004886. Relative to `af12_kz_pw` (Z + W on both): `af12_kz_pwc` gives up some of the FLAIR gain (head_psnr −0.047002 dB) in exchange for T1 losing none of its `af12_kz`-level head_hfen margin (0.168038 vs. `af12_kz_pw`'s 0.171761) — this is exactly the trade the per-contrast switch was built to make available.

**Cross-talk vs. the Task 12 brief's Step 2 prediction.** The brief predicted `af12_kz_pwc`'s T1 would stay within tolerance of `af12_kz` (head_psnr ≥ 34.276970 − 0.005 = 34.271970) and FLAIR would stay within tolerance of `af12_kz_pw` (head_psnr ≥ 34.341124 − 0.005 = 34.336124). Neither held: T1 head_psnr came in at 34.262346, 0.009624 dB below its predicted floor; FLAIR head_psnr came in at 34.294122, 0.042002 dB below its predicted floor. The two contrasts are not separable at the model level: `--prewhiten_only` changes FLAIR's data-consistency loss term only, but T1 and FLAIR share one Gaussian geometry (shared positions/scales/rotations; only the per-Gaussian density is per-contrast), so re-weighting FLAIR's gradient moves the shared geometry and T1 rides along — a ~0.01–0.05 dB cross-talk that a per-contrast *loss* switch cannot avoid by construction, since it does not decouple the *model*. Phase 2's coupled non-local prior should expect the same: any per-contrast lever will leak into the other contrast through the shared geometry it is deliberately coupling. This does not change the `base1` decision below: `af12_kz_pwc` still wins the FLAIR-gates-passed tie-break (3/5 vs. `af12_kz`'s 2/5) and would also win the e_unobs tie-break had that been needed (FLAIR e_unobs 0.617442 vs. `af12_kz`'s 0.627476).

### Per-gate verdicts (`scripts/eval_gates.py`, verbatim)

```
af12_kz_pwc T1     2/5 head_psnr=FAIL(34.2623,margin -0.3376) head_ssim=FAIL(0.9092,margin -0.0055) head_hfen=FAIL(0.1680,margin -0.0135) e_unobs=PASS(0.6342,margin +0.0146) grain_ratio=PASS(0.9572,margin +0.0428)
af12_kz_pwc FLAIR  3/5 head_psnr=FAIL(34.2941,margin -0.1017) head_ssim=PASS(0.9348,margin +0.0001) head_hfen=FAIL(0.1303,margin -0.0071) e_unobs=PASS(0.6174,margin +0.0270) grain_ratio=PASS(0.9017,margin +0.0983)
```

FLAIR clears 3/5 (newly passes `head_ssim`, margin +0.0001 — the same gate `af12_kz_pw`'s FLAIR cleared). T1 stays at 2/5, same two gates passing (`e_unobs`, `grain_ratio`) as every Phase-1 run so far. Neither contrast fully passes.

Verdict JSON: `docs/superpowers/records/af12_to_af10/scores/af12_kz_pwc_verdict.json`. Appended to `decisions.jsonl` (phase 1, one line).

### Visual gate

**Pending user.** New/regenerated figures (error maps: jet, fixed [0, 0.1] of GT peak; butterfly figures now carry a `<af12_label>` argument identifying the actual candidate run instead of a hard-coded baseline caption):

- `http://localhost:8765/af12_to_af10/af12_kz_pwc_T1_3views.png`
- `http://localhost:8765/af12_to_af10/af12_kz_pwc_FLAIR_3views.png`
- `http://localhost:8765/af12_to_af10/af12_kz_pwc_vs_af10_T1.png`
- `http://localhost:8765/af12_to_af10/af12_kz_pwc_vs_af10_FLAIR.png`
- `http://localhost:8765/af12_to_af10/af12_kz_vs_af10_T1.png` (regenerated: caption now reads "AF12 af12_kz", was mislabelled "AF12 tvs4k_e040")
- `http://localhost:8765/af12_to_af10/af12_kz_vs_af10_FLAIR.png` (regenerated, same fix)
- `http://localhost:8765/af12_to_af10/af12_kz_pw_vs_af10_T1.png` (regenerated, same fix)
- `http://localhost:8765/af12_to_af10/af12_kz_pw_vs_af10_FLAIR.png` (regenerated, same fix)

No new artefacts, patchiness, or coherent banding have been screened by the user yet; this record does not assert a visual pass or fail.

### `base1` decision, re-applied over the three runs (spec §4.5, generalized per the Task 12 brief)

Rule, in order: (1) disqualify any run that regresses any of `{head_psnr, head_ssim, head_hfen}` (either contrast) beyond its registered tolerance (`gates.json` `tolerance_floor`: 0.005 dB / 0.0005 / 0.001) relative to `tvs4k_e040`; (2) among the survivors, more FLAIR gates passed wins; (3) tie → lower FLAIR `e_unobs`; (4) tie → prefer the more-physics run.

**Disqualification check** (all three runs, value vs. `tvs4k_e040 ± tol`):

- `af12_kz`: no regression on any of the six values (see the base-record disqualification check above) — **eligible**.
- `af12_kz_pw`: T1 head_hfen 0.171761 > ceiling 0.169392+0.001=0.170392 (exceeds by 0.001369) — **disqualified**, unchanged from the base record (this run's own numbers did not change).
- `af12_kz_pwc`: T1 head_psnr +0.340329, head_ssim +0.004519, head_hfen −0.001355 (better, ceiling 0.170392 vs. value 0.168038); FLAIR head_psnr +0.669765, head_ssim +0.009023, head_hfen −0.006183 (better, ceiling 0.137522 vs. value 0.130339). No regression on any of the six values — **eligible**.

Two eligible runs remain: `af12_kz` and `af12_kz_pwc`.

**Tie-break 1 (more FLAIR gates passed):** `af12_kz` FLAIR = 2/5; `af12_kz_pwc` FLAIR = 3/5 (the extra pass is `head_ssim`, margin +0.0001). **`af12_kz_pwc` wins outright — not a tie**, so tie-breaks 2/3 (FLAIR `e_unobs`, "more-physics") are not reached by the rule as written. Recorded for completeness since the numbers were already computed: FLAIR `e_unobs` is *also* lower for `af12_kz_pwc` (0.617442 vs. `af12_kz`'s 0.627476) and `af12_kz_pwc` is *also* the more-physics run (W engaged on FLAIR vs. none for `af12_kz`) — both would have pointed the same way had a tie-break been needed.

**`base1` = `af12_kz_pwc`**, superseding the 2-run decision's `base1 = af12_kz`. This is a direct, intended consequence of the per-contrast switch: it recovers the FLAIR head_ssim gate pass that made `af12_kz_pw` attractive, without `af12_kz_pw`'s disqualifying T1 head_hfen regression, precisely because T1 is no longer forced through prewhitening it did not need.

### Files (this addendum)

- Score JSON: `docs/superpowers/records/af12_to_af10/scores/af12_kz_pwc_score.json` (copied from `results_arc/`, git-ignored).
- Verdict JSON: `docs/superpowers/records/af12_to_af10/scores/af12_kz_pwc_verdict.json`.
- `docs/superpowers/records/af12_to_af10/decisions.jsonl`: one phase-1 line appended, carrying the 3-run `base1` outcome and `"visual gate: pending user"`.
- Figures: `docs/figures/af12_to_af10/af12_kz_pwc_{T1,FLAIR}_3views.png` (new), `docs/figures/af12_to_af10/af12_kz_pwc_vs_af10_{T1,FLAIR}.png` (new), `docs/figures/af12_to_af10/{af12_kz,af12_kz_pw}_vs_af10_{T1,FLAIR}.png` (regenerated in place, caption fix only — pixel content from the same volumes, unaffected by the label change); all copied to `/home/congcongliu/figs/af12_to_af10/`; figure server verified 200 on the index and on all 8 touched files.
- Full volumes (git-ignored, not committed): `results_arc/af12_kz_pwc_volumes.npz`.
