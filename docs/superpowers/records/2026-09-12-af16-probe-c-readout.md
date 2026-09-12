# jiangyizhou AF16, phase 3 — acquisition probe "flair_c": read-out (2026-09-12)

Track B of the phase-3 design (`docs/superpowers/specs/2026-09-12-af16-phase3-nonlocal-design.md`, section 5): does an
in-shutter, complementary FLAIR mask widen the MC-GS lead over L1-ESPIRiT, or lift T1 through the coupling, enough to
justify route 2 (a new topic with every arm re-run and lambda re-swept on two masks)? Same subject, same k-space, same
maps and gauge as `Results_20260908/T1FLAIR_AF16_v3_jiangyizhou` (the "topic" below); T1 keeps `data/mask_af16_3d.h5`,
FLAIR gets a fresh draw with the same 2966-point budget kept entirely inside its elliptical vendor shutter. Nothing
registered in the topic was touched: probe outputs live under `logs/probe_c/` and `recon/probe_c/` with a `_probe_c`
suffix (`summarize.py`'s globs `logs/l1espirit_FLAIR_lam*.json` cannot match a `_probe_c_` name; checked by running the
globs), MC-GS rows carry `p3_probe_c_` names in `autoresearch_af16_p3.tsv`. Executed by the probe sub-agent on GPU 4
(MC-GS) and the CPU (BART, nice 10); the known-zero repair and the `_kz` reruns are the integrator's.

## 1. The probe mask `af16c` (`logs/prep_04c_export_af16c.json`)

Generator and convention unchanged (`vdPoisMex(256,176,256,176, accelx, 4, 24, 0, 1)`, `scripts/prep_02c_mask_af16c.m`):
50 draws, accelx swept 3.0-3.9 in steps of 0.1 (5 draws each), every draw saved to `data/probe_c_draws/draw_NN.mat`
with its accelx. Two facts that shaped the loop:

- `vdPoisMex` seeds from `time(0)` at one-second resolution: two calls in the same second return the same pattern
  (verified: `isequal` true back to back, false after a 1.2 s pause). The loop pauses 1.1 s per draw; the export
  de-duplicates as a second guard (50 of 50 unique).
- At the spec's "AF12-like" accelx 3.0 a draw puts ~3490 points inside the shutter (the AF12 mask itself: 3505), 520
  above the budget, so the "closest to 2966 from above, then trim the outermost radius" rule would have cut the whole
  outer ring. The sweep puts draws right at the budget: in-shutter counts per accelx (min-max of 5) — 3.0: 3467-3498,
  3.3: 3197-3236, 3.5: 3031-3061, **3.6: 2955-2979**, 3.7: 2865-2930, 3.9: 2773-2808.

`scripts/prep_04c_export_af16c.py` computes FLAIR's acquired support exactly as `prep_04_export.py` (readout-line
energy, every 8th readout line, max over coils: **34139 of 45056 = 0.7577**; identical to coil 0 over every readout
line), picks `draw_33` (accelx 3.6, 3231 points on the plane, **2968** inside the shutter), drops the 2 outermost
in-shutter samples (Euclidean distance from (128, 88): 126.5 and 127.0; seed 0; kept max radius 126.4, the current
mask reaches 153.1 because 233 of its points sit in the unacquired corners) and zeroes everything outside the support.

| quantity | probe mask `af16c` | current mask `af16` |
|---|---|---|
| samples on the plane / inside the FLAIR shutter | 2966 / 2966 | 2966 / 2733 |
| overlap of the two masks | 733 | — |
| 24x24 calibration block around (128, 88) | 475 / 576 | 472 / 576 |
| sampling fraction of the plane | 0.06583 | 0.06583 |
| AF vs the full grid | 15.19 | 15.19 |
| AF vs FLAIR's own support | **11.51** (target 34139 / 2966 = 11.51) | 12.49 |
| union coverage outside the calibration block | 4676 of 44480 plane points = **10.5 %** (13.2 % of the shutter's 33563) | one mask: 2494 = 5.6 % |

Files: `data/mask_vdpois_256x176_af16c_flair.mat` (key `mask`, md5 `48d6acbd…`), `data/mask_af16c_flair_3d.h5`
(dataset `mask`, uint8 (256,256,176), gzip 1, the layout of `mask_af16_3d.h5`; md5 `fc844500…`), figure
`figures/probe_c_mask_T1_vs_FLAIRc.png` (T1 mask with the shutter edge / new FLAIR mask / union), served at
`http://localhost:8765/t1flair_v3_af16/probe_c_mask_T1_vs_FLAIRc.png`.

## 2. Loader override and config

- `scripts/t1t2_common.py` (`Pair.__init__`): `T1T2_MASK_<CONTRAST>=<h5>` swaps the mask of that contrast only;
  unset = the registered `mask_af16_3d.h5`. `check_gauge.py --contrast T1 --method cgsense --gpu 4` printed `GAUGE OK`
  before and after the edit with byte-identical numbers (`logs/probe_c/check_gauge_{before,after}.log`). Override
  sanity (`logs/probe_c/override_sanity.log`): default FLAIR = 2966 samples of which 233 sampled lines have zero
  energy; override = 2966 samples, 0 zero-energy lines; the same P99.9 scale (9.85118e+01) either way.
- `configs/t1flair_v3_af16_jyz_flair_c.yaml` = `t1flair_v3_af16_jyz.yaml` + header, FLAIR `mask_path` → the new h5,
  FLAIR `acceleration_factor` 16 → 12 (informational); T1 unchanged, `known_zero: true` kept on FLAIR. Reached only
  through `MASK_VARIANT=flair_c` of `autoresearch_af16_p3.sh`. The integrator later added `known_zero_support`
  (explicit shutter file `data/flair_support_2d.h5`) for the `_kz` reruns.
- The launcher log of every probe run shows FLAIR loading `mask_af16c_flair_3d.h5` and `[physics:contrast] …
  zero_lines=0/2966` (233/2966 on the current mask).

## 3. Runs (head-only, 1500 iterations; both gauges from the ledger, L1-ESPIRiT from its json)

| arm | FLAIR mask | FLAIR known zeros | T1 mag | FLAIR mag | head SSIM T1 / FLAIR | head HFEN T1 / FLAIR | sharp T1 / FLAIR | cplx T1 / FLAIR | wall |
|---|---|---|---|---|---|---|---|---|---|
| `p3_r0_control` (P6) | current | estimated from the 233 sampled zero lines, frac_z 0.2423 | 35.515 | 34.725 | 0.9353 / 0.9198 | 0.1218 / 0.1192 | 0.829 / 0.785 | 31.981 / 31.187 | 26.0 min |
| `p3_r0_nokz` (P6, `known_zero=false`) | current | none | 35.525 | 34.738 | 0.9353 / 0.9184 | 0.1218 / 0.1222 | 0.829 / 0.782 | 31.986 / 31.132 | 26.3 min |
| L1-ESPIRiT FLAIR λ 0.003 (registered) | current | n/a | — | 34.503 | — / 0.9201 | — / 0.1210 | — / 0.789 | — | — |
| L1-ESPIRiT T1 λ 0.0015 (registered) | — | n/a | 35.209 | — | 0.9334 / — | 0.1286 / — | 0.849 / — | — | — |
| L1-ESPIRiT FLAIR λ 0.002 `_probe_c` | probe | n/a | — | 34.677 | — / 0.9189 | — / 0.1180 | — / 0.868 | — | 454 s BART |
| **L1-ESPIRiT FLAIR λ 0.003 `_probe_c`** (best) | probe | n/a | — | **34.871** | — / 0.9248 | — / 0.1148 | — / 0.815 | — | 564 s BART |
| L1-ESPIRiT FLAIR λ 0.005 `_probe_c` | probe | n/a | — | 34.778 | — / 0.9249 | — / 0.1151 | — / 0.763 | — | 438 s BART |
| `p3_probe_c_P6` (P6, no zero line sampled → Z empty, frac_z 0.0000) | probe | none (estimator finds nothing) | 35.526 | 34.847 | 0.9353 / 0.9208 | 0.1217 / 0.1234 | 0.829 / 0.810 | 31.987 / 31.176 | 28.8 min |
| **`p3_probe_c_P6_kz`** (P6, shutter given via `known_zero_support`, frac_z 0.2423) | probe | explicit | 35.513 | **35.072** | 0.9352 / 0.9246 | 0.1217 / 0.1128 | 0.830 / 0.813 | 31.983 / 31.494 | 26.1 min |
| `p3_probe_c_P6_kz_vtv0005` (coupling check, `--lambda_vtv 0.0005`) | probe | explicit | pending | pending | | | | | started 19:58, card 5 |

L1-ESPIRiT on the probe mask is bracketed (0.002 < 0.003 > 0.005), as on the current mask. BART reported
`Samples: 759295, Acc: 15.19` for the probe mask (2966 x 256 = 759296; one sampled point is exactly zero in the data)
against `699648, Acc 16.49` on the current mask. The wall times are at host load ~90 on 112 cores with 32 OpenMP threads.

## 4. Pre-registered read-out (spec section 5) and decision

- **FLAIR margin on the probe mask** = MC-GS − best L1-ESPIRiT = 35.072 − 34.871 = **+0.201 dB** (`p3_probe_c_P6_kz`),
  against **+0.222** on the current mask (34.725 − 34.503; the phase-1/2 value 34.724 gives +0.221). Change −0.021 dB;
  the **+0.3 dB threshold is not met**. Without the known zeros (`p3_probe_c_P6`) the margin is −0.024 dB.
- **T1 through the coupling**: 35.513 (`_kz`) / 35.526 (no kz) vs 35.515 — −0.002 / +0.011 dB; the **+0.15 dB threshold is
  not met**. The doubled union coverage outside the calibration block (10.5 % vs 5.6 %) does not reach T1, consistent
  with the AF12 "design B" observation that the shared geometry does not carry k-space information across contrasts.
- **Decision: route 2 is not worth a full re-run; the probe is closed and recorded.**

What the probe did show:
1. **An in-shutter FLAIR mask lifts every arm by about 0.35 dB**: L1-ESPIRiT +0.368 (34.503 → 34.871), MC-GS +0.347
   (34.725 → 35.072). It is an acquisition effect shared by all arms (7.9 % of FLAIR's budget stops being wasted on
   the unacquired corners; 12.49x → 11.51x against its own support), not a change in the ordering.
2. **The known-zero region is worth ~0 on the current mask and +0.225 dB on the probe mask** (`p3_r0_nokz` 34.738 vs
   control 34.725: −0.013; `p3_probe_c_P6` 34.847 vs `_kz` 35.072). On the current mask the 233 sampled zero lines tell
   the data term the corners are zero anyway; on an in-shutter mask no sampled line is zero, so the trainer's estimator
   finds nothing (`ellipse=None`, `frac_z_used=0.0000`) and the shutter has to be supplied explicitly. Any future
   in-shutter design needs `known_zero_support`; the first probe run is the no-known-zero data point.
3. Head SSIM and HFEN of MC-GS on the probe mask (`_kz`: 0.9246 / 0.1128) stay ahead of L1-ESPIRiT's (0.9249 is L1's
   λ 0.005 point, whose PSNR is lower; at L1's best λ: 0.9248 / 0.1148); sharpness 0.813 vs 0.815 — the same pattern
   as on the current mask, at the same margins.

Caveats: one subject, one draw of the probe mask (the pool of 50 is on disk), L1-ESPIRiT re-swept on three lambdas only,
MC-GS recipe (P6) not re-tuned on the probe mask, the comparison is at equal sample count (2966) rather than equal AF
against FLAIR's own support, and L1-ESPIRiT's complex-gauge number on the probe mask was not computed.

## 5. Files added or changed (for the commit)

Topic `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/` (mirrored to
`/data0/congcong/code/GS/mcgs_af16_wt/t1flair_af16_v3_jiangyizhou/` at the same relative paths; the mirror's `logs/`
is gitignored by `.gitignore:13 logs*/`, so the json logs need `git add -f` if they are to be tracked):
- added `scripts/prep_02c_mask_af16c.m`, `scripts/prep_04c_export_af16c.py` (`--fig_only` re-renders the figure),
  `scripts/run_l1_probe_c.sh`; changed `scripts/t1t2_common.py` (the override, 7 lines);
- added `logs/prep_04c_export_af16c.json`, `logs/probe_c/{prep_02c_mask_af16c,prep_04c_export_af16c,check_gauge_before,
  check_gauge_after,override_sanity}.log`, `logs/probe_c/l1espirit_FLAIR_probe_c_lam{0.002,0.003,0.005}.{json,log}`,
  `logs/probe_c/l1espirit_probe_c_chain.log`; `figures/probe_c_mask_T1_vs_FLAIRc.png` (also in
  `/home/congcongliu/figs/t1flair_v3_af16/`);
- data (not mirrored): `data/mask_vdpois_256x176_af16c_flair.mat`, `data/mask_af16c_flair_3d.h5`,
  `data/probe_c_draws/draw_01..50.mat`, `recon/probe_c/l1espirit_probe_c_lam*_FLAIR_complex.npy`.

Worktree: `configs/t1flair_v3_af16_jyz_flair_c.yaml` (this file; later edited by the integrator for
`known_zero_support`), this record; ledger rows `p3_probe_c_P6` (abb5f5b), `p3_probe_c_P6_kz` (44f3c3c) and the
launcher logs / score jsons under `results_arc/p3_probe_c_*`.
