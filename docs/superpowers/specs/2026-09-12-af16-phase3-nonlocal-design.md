# jiangyizhou AF16, phase 3 — widen the MC-GS lead: non-local prior + acquisition probe (design, 2026-09-12)

Branch `autoresearch-jyz-af16-phase3` (worktree `/data0/congcong/code/GS/mcgs_af16_wt`), started from the phase-2 tip
5348815. User decisions taken in chat on 2026-09-12 before this document: arena = jyz AF16 from P6 (the line is
reopened); levers allowed = model-structure changes, the AF12 line's findings, acquisition-side design; learned priors
stay excluded; success = head PSNR margin >= 1.0 dB over the strongest comparison arm on both contrasts, all six head
cells (PSNR / SSIM / HFEN x T1 / FLAIR) still first, sharpness not below L1-ESPIRiT's; route = model-side main line
plus a cheap acquisition-side probe; bm4d may be installed into the GSTMI env.

## 1. Starting point and bar

| | T1 | FLAIR |
|---|---|---|
| P6 (`ar14_n36_v2tv_full4k`, topic `gs_joint_ar`), head PSNR | 35.552 | 34.807 |
| L1-ESPIRiT (best other arm, lambda swept on this subject), head PSNR | 35.209 | 34.503 |
| margin today | +0.343 | +0.304 |
| P6 head SSIM / HFEN | 0.9351 / 0.1218 | 0.9203 / 0.1188 |
| L1-ESPIRiT head SSIM / HFEN | 0.9334 / 0.1286 | 0.9201 / 0.1210 |
| sharpness P6 / L1-ESPIRiT | 0.830 / 0.849 | 0.788 / 0.789 |

Bar file: `Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/logs/bar_complex_gauge.json` (magnitude and complex gauges).
Target: +1.0 dB on both contrasts, i.e. T1 >= 36.21 and FLAIR >= 35.50 head PSNR, with the six-cell and sharpness
conditions above. The sharpness condition is new to this line (P6 itself does not meet it).

Facts that shaped the design:
- Every single-knob lever of the current recipe family is settled (capacity, lambda_geo, kappa_q, eps, w_floor, wavelet
  strength / levels / basis, warm-up, k-space band weights, learning rates, directional TV, prewhitening). The AF12
  line's best structural change (db4, one level) lifted AF12 by +0.19 / +0.10 and, stacked on p6ptv, did not lift the
  AF16 magnitude gauge.
- The joint arm shares positions / scales / rotations and keeps a free complex amplitude per contrast. With one mask
  for both contrasts the coupling is worth +0.04 / -0.003 dB (joint minus single at P6). In the AF12 "design B" run
  (FLAIR on a fresh AF11 mask) T1 did not move (36.945 vs 36.944): the current coupling does not carry k-space
  information across contrasts.
- All priors in the recipe (edge-TV, wavelet L1) and in every comparison arm are local. The CS-MRI literature's
  next step above wavelet + TV is non-local self-similarity (BM3D/BM4D-class, NLR-CS), worth 0.5-1 dB at high
  acceleration; no arm in the table has it.
- The AF16 mask puts 233 of its 2966 phase-encode samples outside FLAIR's elliptical shutter (7.9 % of FLAIR's budget
  is wasted); an independent second draw would raise the union coverage outside the calibration block from 5.6 % to
  10.9 %.

## 2. Isolation and protocol (same shape as phase 2)

- Runner `autoresearch_af16_p3.sh EXP GPU ITERS [extra args]`: P6 baked in (edge-TV lambda_geo 0.10 / kappa_q 0.85 /
  w_floor 0.10 / eps 0.04, kspace_high 2.2, warm-up 500, max_radius 20, init head / 3-NN / kappa from ZF, wavelet
  T1 0.10 / FLAIR 0.07 (db2, 3 levels), 3.6M points). EXP must start with `p3_`; no CONFIG path override;
  `SINGLE=T1|FLAIR` picks the single-arm configs by name; `MASK_VARIANT=flair_c` picks the probe config by name.
- Ledger `autoresearch_af16_p3.tsv` via `scripts/autoresearch_af16_p3_row.py` (both gauges, SSIM, HFEN, sharpness,
  e_unobs; flock-appended). Program and round log `autoresearch_af16_p3.md`.
- Screen at 1500 iterations, promote at 4000 with `SAVE_VOLUMES=1`. One trainer per GPU (two stalls on 09-10 when a
  card got a second). Launch with `setsid nohup`; progress from `results_arc/<exp>/<exp>/joint_train.log`, never the
  launcher log. Seed floor of the P6 family: 0.000 / 0.002 dB at 4000, <= 0.003 at 1500.
- Promotion rule (pre-registered): a candidate is `keep` if none of the six head cells is worse than the parent by more
  than the seed floor, at least one is better by more than it, and sharpness is not lower than the parent's by more
  than 0.005. `keep-marginal` when PSNR gains but a structural cell or sharpness slips; it is only promoted if the
  4000-iteration run repairs the slip. Winners combine in the next round; the round's parent is the best `keep`.
- GPUs: 2, 3, 5 for the main line; 4 for the probe (and short tests); cards 0, 1, 6, 7 belong to other jobs. BART
  (L1-ESPIRiT) runs on the CPU under `nice 10`. The host runs at load ~120 on 112 cores today, so CPU timings are
  pessimistic.
- AF12 line: nothing is merged. Its findings enter only as parameter values (db4 / one level / T1 0.12 / FLAIR 0.10).

## 3. Round 0 — terms that already exist (no code)

Seven screens at 1500 iterations on GPUs 2 / 3 / 5, all off P6:

| exp | extra args | why |
|---|---|---|
| `p3_r0_control` | none | P6 on this runner at 1500: the reference for every Round-0 delta and the runner/seed floor |
| `p3_r0_vtv002` / `p3_r0_vtv005` | `--lambda_vtv 0.02` / `0.05` | vectorial TV couples edge locations symmetrically; never run on this subject |
| `p3_r0_edge` | `--lambda_edge W` (W calibrated so lambda*R_edge is 10-30 % of lambda_geo*R_geo in the smoke log) | common-edge direction prior; never run |
| `p3_r0_priv03` | `--set gaussian.num_private_points=300000` | contrast-private residual Gaussians for FLAIR, never run |
| `p3_r0_db4l1` | `--set loss.wav_name=db4 --set loss.wav_levels=1 --set loss.lambda_wav_plain=0.12 --set loss.lambda_wav=0.10` | the AF12 line's wavelet shape, on P6 directly (only ever tested on p6ptv at AF16) |
| `p3_r0_ptv002` | `--lambda_ptv 0.02` | phase TV at a fifth of phase 2's weight: buys sharpness (p6ptv reached 0.876 / 0.837) for a smaller magnitude cost |

Reference at 1500 iterations from phase 1: P6 = 35.515 / 34.724.

## 4. The non-local prior (Round 1 onward) — `paired_gs/losses/nonlocal_red.py`

Regularisation-by-denoising with a lagged target, on the rendered magnitude of each contrast:

```
R_nl(x)  = mean_B( (|x| - x_tilde)^2 ) / mean_B( x_tilde^2 )        B = head bounding box, x_tilde detached
x_tilde  = D( |x_k| )   refreshed every K iterations from iteration K0 on; between refreshes it is a constant
D        = BM4D (package bm4d 4.2.5, CPU), sigma = s * P99(|x_k| inside the head), profile 'np'
loss    += lambda_nl_plain * R_nl(x_plain) + lambda_nl_contrast * R_nl(x_contrast)
```

- Config keys under `loss` (all `--set`-able, defaults keep the trainer bit-identical): `lambda_nl` 0.0
  (`lambda_nl_plain` / `lambda_nl_contrast` default to it), `nl_refresh` 500 (K), `nl_start` 500 (K0), `nl_sigma_rel`
  0.04 (s), `nl_profile` 'np', `nl_denoiser` 'bm4d', `nl_threads` 32.
- Head bounding box: from the ZF-derived head support the trainer already computes for `init_mode=head`
  (`|zf| > 0.05 * P99.9`), dilated by 8 voxels, fixed for the run. BM4D runs on the crop only (the crop is ~7M of the
  11.5M voxels; measured BM4D cost under today's load: 64^3 in 9 s, 128^3 in 55 s, i.e. ~27 us per voxel, so a crop
  takes 3-4 min). The first implementation refreshes synchronously (training blocks during the call); with K = 500 a
  4000-iteration run pays 7 refreshes (~25 min). An asynchronous refresh (background thread, swap the target when
  ready) is the follow-up only if the synchronous cost proves limiting.
- Why magnitude, not complex: the gauge that decides the line is the magnitude gauge, edge-TV and the wavelet term on
  magnitude are what P6 tuned, and phase is handled by the phase-TV term if Round 0 keeps it. `bm4d` also ships a
  complex profile; it is a later knob, not part of this design.
- Wiring follows the phase-TV template: read the lambdas in `JointTrainer.train`, lazy-import the module, add the
  term after the wavelet term, register `R_nl` in the `components` dict and the log line, so `joint_train.log` shows
  when the target refreshes (log the refresh iteration, sigma and wall time).
- Tests (`tests/test_nonlocal_red.py`, GPU): lambda 0 leaves the loss and the components untouched; a fake denoiser
  (identity, then a Gaussian blur) drives the refresh cadence (K0, K) and the target is detached (no gradient into
  the target, gradient into |x| matches finite differences); crop / un-crop round trip is exact; the normaliser
  makes R_nl dimensionless (scaling x and x_tilde by c leaves R_nl unchanged); a 30-iteration trainer smoke on the
  AF16 config with `lambda_nl=0.5, nl_start=10, nl_refresh=10` and the fake denoiser runs end to end. One real BM4D
  timing on the true crop size is recorded in the program doc.
- Knobs for the rounds: lambda_nl in {0.5, 1, 2} (then per-contrast), s in {0.02, 0.04, 0.08}, K in {250, 500},
  K0 in {500, 1000}. Expected failure mode is over-smoothing: the sharpness gate and head HFEN decide.

## 5. The acquisition probe (decides whether route 2 is worth a full re-run)

- New FLAIR mask `Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/data/mask_af16c_flair_3d.h5` (+ the 2-D `.mat` and a
  json): vdPoisMex draws at an AF12-like density (accelx around 3.0, accely 4, calib 24, ellipse 0, pp 1; the
  generator reseeds from the clock, so every call is a fresh pattern); keep only the samples inside FLAIR's acquired
  support (the elliptical shutter, 34139 of 45056 points); repeat the draw and take the pattern whose in-support count
  is closest to 2966 from above, then drop random samples from the outermost radius until exactly 2966 remain. Record
  the overlap with the current mask, the in-support AF (target 34139 / 2966 = 11.51x against FLAIR's shutter, versus
  12.49x today) and the calibration-block count. T1 keeps `mask_af16_3d.h5`.
- Comparison-arm loader: `t1t2_common.Pair` reads `data/mask_af16_3d.h5` for both contrasts; add an environment
  override `T1T2_MASK_<CONTRAST>=<path>` (default unchanged) so L1-ESPIRiT can be re-run on the probe mask without
  touching the topic's registered results. Probe outputs go under `recon/` and `logs/` with the suffix `_probe_c`.
- Runs: (i) L1-ESPIRiT FLAIR on the probe mask, lambda in {0.002, 0.003, 0.005} (CPU, sequential); (ii) MC-GS P6 on
  the probe config, 1500 iterations (`MASK_VARIANT=flair_c`); (iii) MC-GS P6 + `--lambda_vtv` (the Round-0 vtv weight
  that kept, else 0.05), 1500 iterations. GPU 4.
- Read-out, pre-registered: FLAIR margin on the probe mask = MC-GS(ii or iii) - best L1-ESPIRiT(i), compared with the
  same-budget 1500-iteration margin on the current mask (P6 at 1500 = 34.724 vs L1-ESPIRiT 34.503, +0.221). Route 2
  (new topic directory, every arm re-run and lambda re-swept on the two masks) is worth it only if that margin grows
  by >= 0.3 dB, or T1 rises by >= 0.15 dB through the coupling. Otherwise the probe is closed and recorded.

## 6. Rounds after Round 0 and the finish

- Round 1: the non-local prior alone on P6 (lambda_nl x s grid at K = 500, K0 = 500), plus the Round-0 keeps combined.
- Round 2+: non-local prior stacked on the Round-0 keeps; refine K, K0, per-contrast lambda_nl.
- Promotion: 4000 iterations with volumes, seed repeat, GS-single T1 / FLAIR at the same recipe, then
  `scripts/add_gs_arm.py` registers the winner in the topic (a new key, `gs_joint_p3`; P6 stays `gs_joint_ar`),
  `scripts/refresh_readme_tables.py`, and the compact three-plane figures 174 / 110 / 47 are regenerated with
  `fig_three_planes_compact.py --mcgs_key gs_joint_p3`.
- Close-out: record `docs/superpowers/records/<date>-autoresearch-jyz-af16-phase3.md`, handoff
  `docs/handoff/<date>-autoresearch-jyz-af16-phase3.md`, STATUS.md, CLAUDE.md phase-3 section, topic README
  phase-3 section; commits under the repository identity with no AI trailer; `main` is not merged.
- Fairness note to carry into the record: the non-local prior is a handcrafted prior selected on this subject's ground
  truth like every other knob and like the classical lambdas; a reviewer may ask for L1-ESPIRiT + the same prior.

## 7. Implementation plan (two parallel tracks, one integrator)

Track A (module, in its own worktree `mcgs_af16_nl_wt` on branch `autoresearch-jyz-af16-phase3-nl`, TDD, tests on
GPU 4): 1 module + config keys, 2 tests, 3 trainer wiring + log, 4 the 30-iteration trainer smoke, 5 one real BM4D
timing on the crop, 6 a self-review of the diff against this section; merged fast-forward into phase 3 by the
integrator once `pytest tests -q --ignore=tests/smoke_test.py` passes with the new tests included.

Track B (probe, topic scripts + one config in the phase-3 worktree, no framework code): 1 mask generation and export
with the counts above, 2 loader override + a check that the default path is byte-identical, 3 config
`configs/t1flair_v3_af16_jyz_flair_c.yaml` (only the FLAIR `mask_path` and the informational acceleration differ),
4 the three L1-ESPIRiT runs on the CPU, 5 the two MC-GS probe runs on GPU 4 (`MASK_VARIANT=flair_c`), 6 the read-out
table in `autoresearch_af16_p3.md`.

Integrator (this session): Round 0 launch and ledger, program doc, merges, Rounds 1+, promotion, close-out.
