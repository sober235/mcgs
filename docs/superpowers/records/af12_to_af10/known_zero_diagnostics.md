# Known-zero region on the real data — R1' diagnostics and oracle precision/recall/energy

Spec: `docs/superpowers/specs/2026-09-05-af12-to-af10-quality-design.md` appendix A, R1' addendum (2026-09-06).
Integration run: `kz_pw_smoke10c` (10 iters, warmup 2, `--known_zero --prewhiten`, `configs/winner_v2_edgeTV.yaml`, GPU 1).
Oracle diagnostic: `scripts/kz_oracle_check.py` (uses the full k-space; never part of the fit).

## Physics lines (`[physics:...]`, verbatim from `results_arc/kz_pw_smoke10c.log`)

```
[physics:plain] known_zero=True cuts={'axis0_lo': None, 'axis0_hi': None, 'axis1_lo': 44, 'axis1_hi': None} ellipse=None frac_z_used=0.2000 zero_lines=383/4760 explained=383 nonzero_in_z=0 noise_level_used=0.0003154293808620423 outliers=0/0 prewhiten=True sigma_source=readout_ends n_noise=280128 corr_max=0.5606025868200968 noise_level=0.0003154293662385707
[physics:contrast] known_zero=True cuts={'axis0_lo': None, 'axis0_hi': None, 'axis1_lo': None, 'axis1_hi': None} ellipse=(123.0, 88.0, 1.0) frac_z_used=0.3959 zero_lines=772/4760 explained=735 nonzero_in_z=0 noise_level_used=0.00048310033162124455 outliers=36/0 prewhiten=True sigma_source=readout_ends n_noise=255232 corr_max=0.43119821746876413 noise_level=0.00048310033037098044
```

`plain` = T1, `contrast` = FLAIR. Both `sigma_source=readout_ends`, both `nonzero_in_z=0`, both `kz_energy=0.0` (from the run's score JSON) — the four hard invariants hold on the real data.

## Oracle metrics (ORACLE — ground truth from the full k-space, `scripts/kz_oracle_check.py`)

| contrast | precision | recall | false_positive_lines | fp_energy_fraction | fp_excess_over_noise_fraction | max_fp_line_energy_over_noise | true_zero_fraction | frac_z | ellipse | n_outliers | n_significant_outliers |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | 1.000 | 1.000 | 0 | 0.0 | 0.0 | 0.0 | 0.200 | 0.200 | None | 0 | 0 |
| FLAIR | 0.9486 | 0.9588 | 1146 | 2.574e-4 | 0.0 | 0.964 | 0.3917 | 0.3959 | (123.0, 88.0, 1.0) | 36 | 0 |

Acceptance (task-11 brief, Step 4): both `n_nonzero_in_z == 0` and `fp_excess_over_noise_fraction ≤ 1e-5` — **PASS** (both exactly 0.0); T1 recall ≥ 0.99 and precision ≥ 0.999 — **PASS** (1.000 / 1.000); FLAIR recall ≥ 0.90 and `max_fp_line_energy_over_noise ≤ 1.5` — **PASS** (0.9588 / 0.964).

FLAIR's 1146 false-positive lines (Z but oracle-nonzero) are the vendor's soft apodization roll-off just inside the (123, 88) ellipse (r ∈ [0.85, 1.0], amplitude 1e-6~1e-4 of max) — real but noise-level signal. Every one of them sits at or below the readout-end noise floor (max 0.964×, so the model can never be expected to reconstruct them better than by predicting zero: E|v−n|² ≥ E|n|²).

## History

| stage | T1 | FLAIR |
|---|---|---|
| §4.1 ring/cliff rule (superseded) | ρ_c = 1.15 (misaligned annulus fit) | ∅ (no cliff found) |
| R1, zero-line shape estimator, unexpanded gate (Task 9/10, every outlier drives expansion) | cuts axis1_lo=44 (unaffected — T1 has no outliers) | ellipse (123, 88, expand **1.185**), Z 22.6% of the plane (true 39.2%), oracle recall **0.575**, precision 0.9941, 75 truly-nonzero lines still inside Z (their energy: 9.4e-9 of the total); the 36 outlier lines forcing the expansion carry a median 2.6e-5 × the noise floor |
| R1' (this task), significance-gated expansion, `SIG_NOISE=3.0` | unchanged: precision/recall 1.000, 0 outliers | ellipse (123, 88, expand **1.0**), Z 39.6% of the plane, oracle recall **0.959**, precision 0.9486, n_nonzero_in_z 0, all 1146 false-positive lines below the noise floor (max 0.964×, excess-over-noise energy exactly 0) |

R1' changes nothing about the ellipse's (a, b) fit or the padding cuts — both stages find the same (123, 88) / axis1_lo=44 — only the *safety-expansion* step now ignores outlier lines whose mean per-(coil,sample) energy does not exceed `SIG_NOISE × noise_level` (the R2 readout-end noise floor), instead of reacting to every sampled nonzero line outside the unexpanded fit regardless of its energy.

## Figures

- `docs/figures/af12_to_af10/kz_tiny_geometry_T1.png` — evidence for T1's rectangular hard shutter (PE-axis-1 index < 44 numerically zero; 20.0% of lines).
- `docs/figures/af12_to_af10/kz_tiny_geometry_FLAIR.png` — evidence for FLAIR's DC-centred (123, 88) elliptical hard shutter (39.2% of lines), the same shape R1/R1' both recover from the sampled lines alone.

## Run record

```bash
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<100{print $1; exit}')
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID /home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py \
  --exp kz_pw_smoke10c --gpu $G --iters 10 --warmup 2 --lambda_geo 0.10 --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 --base_cfg configs/winner_v2_edgeTV.yaml \
  --known_zero --prewhiten 2>&1 | tee results_arc/kz_pw_smoke10c.log
```
GPU 1 (idle, memory.used=14 MiB before the run; GPU 0 was busy with another job at the time). Elapsed: script-reported `elapsed_s=277`; wall clock 13:40:16 → 13:44:57 CST (2026-09-06), ≈281 s. Exit clean, `both_beat_zf=False` expected (10-iteration, warmup=2 smoke test of the switches, not a scored training run). Score JSON copied to `docs/superpowers/records/af12_to_af10/scores/kz_pw_smoke10c_score.json`.
