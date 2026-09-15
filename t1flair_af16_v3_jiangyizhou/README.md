# Retrospective AF 16 on the 3D T1 + FLAIR pair — second subject (vol3_jiangyizhou)

Fifth topic in the series. **Read `../T1FLAIR_AF16_retro/README.md` first**: the AF 16 mask, the
worktree, the scheduler, the gauge proof and the gotchas are documented there and not repeated.
The data, geometry, maps, gauge and eight arms are those of `../T1FLAIR_AF12_v3_jiangyizhou/`
(this subject at AF 12), which in turn defers to `../T1T2_AF12_retro/README.md`.

## What this topic adds

The **same AF 16 study on a different person**. Its own T1 and FLAIR k-space and its own two
ESPIRiT maps (symlinks into `../T1FLAIR_AF12_v3_jiangyizhou/data/`, never written), the **same
mask** as the first subject (`data/mask_vdpois_256x176_af16.mat`, byte-identical; the AF 12 topics
shared their mask the same way), sixteen new reconstructions. Created from the first-subject topic
with `../T1FLAIR_AF16_retro/scripts/setup_af16_jyz.sh`; configs
`configs/t1flair_v3_af16_jyz{,_singleT1,_singleFLAIR}.yaml` in the worktree; run by the same
scheduler queue, after the first subject's core arms.

Mask against this subject's data: T1 support 1.0000 → AF 15.19; FLAIR support 0.7577 → AF 16.49
on the full grid, 12.49 against its own shutter (`logs/prep_04_export.json`). Identical to the
first subject to four digits, as at AF 12.

## Results

AF 16 is the nominal setting of the mask generator. The mask samples 2966 of 45056 phase-encode points: 15.19x for
T1, and 12.49x of FLAIR's acquired k-space (its vendor shutter keeps 75.8% of the plane; 233 mask points land on the
unacquired corners, `figures/t1flair_v3_af16_mask_T1_vs_FLAIR.png`). The user chose on 2026-09-11 to keep these
results under the nominal label rather than re-run at an exact 16x. FLAIR known zeros are used by MC-GS and GS-single
only; any re-run gives every arm the same region.

`logs/summary.tsv`; gauge proof `logs/post_gauge.log` (`GAUGE OK`). Head-only PSNR / SSIM are the
numbers to quote. AF 12 values per arm from `../T1FLAIR_AF12_v3_jiangyizhou/logs/summary.json`.

<!-- TABLES:BEGIN -->
**T1** (sorted by head-only PSNR; AF12 column = the same arm in `T1FLAIR_AF12_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @AF12 | AF16 − AF12 |
|---|---|---|---|---|---|---|---|---|
| MC-GS (phase 3) | 39.268 | 0.9207 | **35.706** | **0.9396** | 0.1139 | 0.855 | -- | -- |
| GS-single (phase 3) | 39.144 | 0.9273 | 35.558 | 0.9371 | 0.1186 | 0.859 | -- | -- |
| MC-GS joint (autoresearch) | 39.016 | 0.9124 | 35.552 | 0.9351 | 0.1218 | 0.830 | 36.944 | -1.39 |
| MC-GS single (autoresearch) | 38.988 | 0.9182 | 35.515 | 0.9343 | 0.1240 | 0.824 | 36.897 | -1.38 |
| MC-GS (wavelet up + phase TV) | 38.968 | 0.9084 | 35.364 | 0.9362 | 0.1265 | 0.844 | -- | -- |
| MC-GS (P6 + phase TV) | 38.927 | 0.9110 | 35.311 | 0.9353 | 0.1204 | 0.876 | 36.969 | -1.66 |
| CS-SENSE (TV, converged) (λ 0.0007) | 38.646 | 0.9506 | 34.945 | 0.9317 | 0.1380 | 0.823 | 36.162 | -1.22 |
| joint CS-SENSE (JTV) (λ 0.001) | 38.595 | 0.9514 | 34.831 | 0.9311 | 0.1429 | 0.811 | 36.070 | -1.24 |
| CS-SENSE | 38.187 | 0.9393 | 34.409 | 0.9276 | 0.1531 | 0.768 | 35.762 | -1.35 |
| ZS-SSDU | 37.823 | 0.9399 | 34.150 | 0.9216 | 0.1646 | 1.016 | 35.724 | -1.57 |
| L1-SPIRiT (λ 0.001) | 37.362 | 0.9224 | 33.788 | 0.9083 | 0.1448 | 0.934 | 35.017 | -1.23 |
| MC-GS joint | 34.628 | 0.8093 | 32.890 | 0.8893 | 0.1882 | 0.921 | 34.160 | -1.27 |
| CG-SENSE | 34.559 | 0.7621 | 32.354 | 0.8825 | 0.2239 | 0.859 | 33.514 | -1.16 |
| MC-GS single | 33.750 | 0.7608 | 32.331 | 0.8679 | 0.1875 | 1.018 | 33.799 | -1.47 |
| MC-INR (Niessen 2025) | 30.285 | 0.5575 | 31.590 | 0.8355 | 0.2325 | 1.010 | 32.192 | -0.60 |
| 3D-DIP | 34.938 | 0.8507 | 30.994 | 0.8549 | 0.3024 | 0.634 | 31.363 | -0.37 |
| 3D-INR | 32.903 | 0.7131 | 30.375 | 0.7951 | 0.2125 | 1.310 | 31.638 | -1.26 |
| ZF | 29.949 | 0.8242 | 25.841 | 0.6795 | 0.6950 | 0.276 | 26.130 | -0.29 |

**FLAIR** (sorted by head-only PSNR; AF12 column = the same arm in `T1FLAIR_AF12_v3_jiangyizhou`)

| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @AF12 | AF16 − AF12 |
|---|---|---|---|---|---|---|---|---|
| MC-GS (phase 3) | 38.585 | 0.9102 | **34.928** | **0.9261** | 0.1121 | 0.804 | -- | -- |
| GS-single (phase 3) | 38.498 | 0.8959 | 34.886 | 0.9249 | 0.1137 | 0.806 | -- | -- |
| MC-GS single (autoresearch) | 38.288 | 0.8922 | 34.810 | 0.9196 | 0.1182 | 0.795 | 36.323 | -1.51 |
| MC-GS joint (autoresearch) | 38.342 | 0.9045 | 34.807 | 0.9203 | 0.1188 | 0.788 | 36.268 | -1.46 |
| MC-GS (wavelet up + phase TV) | 38.432 | 0.8999 | 34.768 | 0.9234 | 0.1187 | 0.806 | -- | -- |
| MC-GS (P6 + phase TV) | 38.415 | 0.9029 | 34.740 | 0.9231 | 0.1147 | 0.837 | 36.310 | -1.57 |
| joint CS-SENSE (JTV) (λ 0.0007) | 38.048 | 0.9408 | 34.355 | 0.9177 | 0.1302 | 0.775 | 35.624 | -1.27 |
| CS-SENSE (TV, converged) (λ 0.0005) | 37.904 | 0.9365 | 34.302 | 0.9180 | 0.1309 | 0.751 | 35.581 | -1.28 |
| ZS-SSDU | 37.882 | 0.9461 | 34.005 | 0.9185 | 0.1529 | 1.036 | 35.034 | -1.03 |
| CS-SENSE | 37.613 | 0.9248 | 33.806 | 0.9116 | 0.1360 | 0.663 | 35.084 | -1.28 |
| L1-SPIRiT (λ 0.001) | 37.413 | 0.9211 | 33.720 | 0.9022 | 0.1207 | 0.890 | 35.103 | -1.38 |
| CG-SENSE | 35.220 | 0.7873 | 32.826 | 0.8860 | 0.1608 | 0.783 | 33.957 | -1.13 |
| MC-GS joint | 35.743 | 0.8923 | 32.402 | 0.8834 | 0.1724 | 0.807 | 33.391 | -0.99 |
| MC-INR (Niessen 2025) | 32.435 | 0.6365 | 32.243 | 0.8590 | 0.1711 | 0.889 | 33.042 | -0.80 |
| MC-GS single | 34.989 | 0.8420 | 32.137 | 0.8701 | 0.1719 | 0.922 | 33.409 | -1.27 |
| 3D-DIP | 34.581 | 0.8188 | 30.650 | 0.8290 | 0.2769 | 0.510 | 30.810 | -0.16 |
| 3D-INR | 33.316 | 0.7454 | 30.257 | 0.7889 | 0.1998 | 1.139 | 31.190 | -0.93 |
| ZF | 31.792 | 0.8410 | 27.879 | 0.7126 | 0.6595 | 0.262 | 28.193 | -0.31 |

**MC-GS joint minus MC-GS single**

| channel | whole PSNR | head PSNR | head SSIM | (AF12: whole / head) |
|---|---|---|---|---|
| T1 | +0.878 | +0.559 | +0.0214 | +0.820 / +0.361 |
| FLAIR | +0.754 | +0.266 | +0.0133 | +0.442 / -0.018 |
<!-- TABLES:END -->

### Reading

1. **The classical arms were re-run converged and λ-swept on this subject (2026-09-10)**, after the
   AF12 topics' audit found the 09-09 CS-SENSE under-converged (35 ADMM iterations inherited from the CE
   study). **L1-wavelet ESPIRiT leads both contrasts (35.209 / 34.503)**, then converged TV
   (34.945 / 34.302) and joint vectorial TV (34.831 / 34.355); every λ curve is bracketed
   (`logs/classical_*_o200_lam*.json`, `logs/l1espirit_*_lam*.json`, `logs/jcssense_o200_lam*.json`,
   `scripts/run_classical_revisited*.sh`). λ was chosen on this subject against the ground truth — the
   same privilege the MC-GS autoresearch loop has. The 09-09 CS-SENSE row is kept for continuity; it is
   0.8 / 0.7 dB below L1-ESPIRiT.
2. **The cold-start MC-GS joint is sixth on T1 and seventh on FLAIR of eleven arms**, 2.3 / 2.1 dB behind
   L1-ESPIRiT. The shared-geometry effect still holds between the two cold-start MC-GS arms: joint − single
   = +0.56 dB (T1) and +0.27 dB (FLAIR) head-only, +0.021 / +0.013 head SSIM.
3. **The autoresearch loop on this subject makes MC-GS joint first on head PSNR, SSIM and HFEN** (worktree
   `mcgs_af16_wt`, branch `autoresearch-jyz-af16`, program `autoresearch_jyz_af16.md`, ledger
   `autoresearch_jyz_af16.tsv`; row "MC-GS joint (autoresearch)" above). Final recipe: the repaired cold start
   (head-support initialisation, 3-NN scales, ZF-sourced edge-TV threshold), 3.6M points, edge-TV λ_geo 0.10 /
   κ_q 0.85 with Charbonnier ε 0.04, wavelet sparsity (db2, 3 levels, cycle-spun) at 0.10 on T1 and 0.07 on FLAIR,
   4000 iterations. Head PSNR 35.552 / 34.807 (+0.343 / +0.304 dB over L1-ESPIRiT), head SSIM 0.9351 / 0.9203,
   head HFEN 0.1218 / 0.1188 — all three better than L1-ESPIRiT on both contrasts; sharpness 0.830 / 0.788 stays
   just below it (0.849 / 0.789). The seed-to-seed spread at this recipe family is ≤ 0.001 dB. The lambdas of the
   classical arms and the MC-GS knobs were both chosen on this subject's ground truth, so the comparison is fair
   but it is not a held-out claim. The single-contrast arms at the same recipe and the seed repeat at 4000
   iterations are being added.
4. Per-slice sweep (`logs/per_slice.json`) covers the original eight arms only: there, MC-GS joint ranked
   first on 5 of 600 T1 slices and 0 of 606 FLAIR slices.
5. 3D-INR (30.4 / 30.3) and 3D-DIP (31.0 / 30.7) stay 3 to 4 dB behind the leaders.

### L1-ESPIRiT dropped from the comparison (user, 2026-09-15)

L1-ESPIRiT is no longer carried as a comparison arm: it is gone from the tables above and from the
figures. **Nothing was deleted** — its reconstructions, its lambda sweep logs and its row in
`logs/summary.json` are untouched, and `DROP` in the table generator is the single line that decides
what is listed. Sections of this file written before 2026-09-15 still discuss it; they are the record
of what was measured at the time, and are left as they are.

**It changes no headline number.** GS-single counts as a comparison arm (user, 2026-09-13), and
GS-single — not L1-ESPIRiT — is the strongest comparison arm on both contrasts here, as in every
topic of this series. The margin that MC-GS is quoted on is therefore MC-GS − GS-single, **+0.148 dB
on T1 and +0.042 dB on FLAIR**, and dropping L1-ESPIRiT leaves it untouched.

What does change is the strongest arm *below* the GS pair:

| head-only | T1 | FLAIR |
|---|---|---|
| strongest comparison arm (unchanged) | GS-single 35.558 | GS-single 34.886 |
| strongest non-GS arm, was | L1-ESPIRiT 35.209 | L1-ESPIRiT 34.503 |
| strongest non-GS arm, now | CS-SENSE (TV, converged) 34.945 | joint CS-SENSE (JTV) 34.355 |

So a sentence of the form "MC-GS leads the best classical arm by X" gets 0.264 dB bigger on T1 and
0.148 dB bigger on FLAIR with L1-ESPIRiT gone, purely from the arm no longer being shown. The
headline margin over GS-single does not move.

## Figures

`figures/`, mirrored to `/home/congcongliu/figs/t1flair_v3_af16/`, served at
`http://localhost:8765/t1flair_v3_af16/<file>`: `t1flair_v3_af16_{T1,FLAIR}_plane{0,1,2}.png`
(three mid-planes, all arms, jet error maps fixed [0, 0.1]) and
`t1flair_v3_af16_{T1,FLAIR}_butterfly_axis0_168.png` (the transverse slice the AF 12 topic of this
subject uses).

## Regenerating

See `../T1FLAIR_AF16_retro/README.md`; the scheduler runs both subjects, and
`scripts/postprocess.sh jyz <gpu>` does everything after the arms.

## Phase 2 — the complex gauge, and the line frozen (2026-09-11/12)

Phase 2 of the autoresearch loop (framework branch `autoresearch-jyz-af16-phase2`, `autoresearch_af16_p2.md`) asked
for MC-GS first on the strict complex gauge |rec − GT| as well, with the magnitude margin kept non-negative. The lever is
a phase-only TV term (`--lambda_ptv 0.05`); registered here as `gs_joint_p2` ("MC-GS (P6 + phase TV)") and
`gs_joint_p2b` (wavelet T1 0.14 / FLAIR 0.10 + phase TV). Head-only, 4000 iterations, T1 / FLAIR:

| arm | magnitude PSNR | complex PSNR | head SSIM | head HFEN | sharp |
|---|---|---|---|---|---|
| MC-GS (P6, the magnitude headline, `gs_joint_ar`) | **35.552 / 34.807** | 31.950 / 31.201 | 0.9351 / 0.9203 | 0.1218 / 0.1188 | 0.830 / 0.788 |
| MC-GS (P6 + phase TV, `gs_joint_p2`) | 35.311 / 34.740 | 32.322 / 31.851 | 0.9353 / 0.9231 | 0.1204 / 0.1147 | **0.876 / 0.837** |
| MC-GS (wavelet up + phase TV, `gs_joint_p2b`) | 35.364 / 34.768 | **32.413 / 31.922** | 0.9362 / 0.9234 | 0.1265 / 0.1187 | 0.844 / 0.806 |
| L1-ESPIRiT | 35.209 / 34.503 | 32.201 / 31.718 | 0.9334 / 0.9201 | 0.1286 / 0.1210 | 0.849 / 0.789 |

Both phase-2 recipes are first against the seven comparison arms on both gauges and on head SSIM / HFEN; neither beats
P6 on the magnitude gauge (phase TV costs 0.24 / 0.07 dB there). A transfer test of the AF12 line's wavelet shape
(db4, one level, T1 0.12 / FLAIR 0.10) stacked on P6 + phase TV read 35.528 / 34.805 magnitude, 32.485 / 31.918 complex,
SSIM 0.9382 / 0.9250, HFEN 0.1167 / 0.1140 — it is not part of any AF16 recipe. **The line is frozen (user, 2026-09-12):
no further runs, code and parameters kept separate from the AF12 line.** Figures:
`figures/t1flair_v3_af16_{T1,FLAIR}_row_p2_axis{0,1,2}_*.png`, `figures/t1flair_v3_af16_{T1,FLAIR}_row3planes_p2_174_110_102.png`.

## Phase 3 — widening the lead (2026-09-12/13)

The user reopened the line to increase the gap to the comparison methods (framework branch
`autoresearch-jyz-af16-phase3`, programme `autoresearch_af16_p3.md`). The winner is registered here as
`gs_joint_p3` ("MC-GS (phase 3)") with `gs_single_p3`; the phase-1 keys `gs_joint_ar` / `gs_single_ar` still hold P6.

**Recipe** = P6 + the wavelet prior changed to db4 on its finest level only (T1 0.12 / FLAIR 0.10) + the symmetric
common-edge cross-gradient prior at λ 0.001 + phase-only TV at λ 0.03. 4000 iterations, 3.6M points, everything else P6.

| head-only, 4000 it | T1 | FLAIR |
|---|---|---|
| **MC-GS (phase 3)** | **35.706** | **34.928** |
| MC-GS (P6) | 35.552 | 34.807 |
| L1-ESPIRiT (best other arm) | 35.209 | 34.503 |
| margin, phase 3 / P6 | **+0.497** / +0.343 | **+0.425** / +0.304 |
| head SSIM, phase 3 / P6 / L1 | 0.9396 / 0.9351 / 0.9334 | 0.9261 / 0.9203 / 0.9201 |
| head HFEN, phase 3 / P6 / L1 | 0.1139 / 0.1218 / 0.1286 | 0.1121 / 0.1188 / 0.1210 |
| sharpness, phase 3 / P6 / L1 | **0.855** / 0.830 / 0.849 | **0.804** / 0.788 / 0.789 |

Every head cell is first and, for the first time in this series, the sharpness is at or above L1-ESPIRiT's on both
contrasts. Seed floor 0.001 / 0.000 dB; MC-GS − GS-single = +0.148 / +0.042 (phase 1: +0.037 / −0.003). Figures
`figures/t1flair_v3_af16_{T1,FLAIR}_row3planes_compact_174_110_47_headerr_p3.png` — MC-GS is first on PSNR, SSIM and
NMSE in every plane of both contrasts. Closed with evidence in this phase: a non-local (BM4D / RED) prior, adaptive
density control, vectorial TV, contrast-private residual Gaussians, and a two-mask acquisition design (see
`scripts/prep_02c_mask_af16c.m` and the probe read-out). The known-zero region is worth −0.013 dB on this mask and
+0.225 dB on an in-shutter mask. Caveats are those of phases 1 and 2 (one subject, knobs and λ tuned on it, magnitude
gauge, nominal AF 16).

## MC-INR (Niessen et al. 2025) registered as a comparison arm (2026-09-13)

Same registration as the AF 12 topic of this subject (`../T1FLAIR_AF12_v3_jiangyizhou/README.md`, section "MC-INR
… registered as a comparison arm"), on this topic's mask. The multi-contrast INR of Niessen et al., *INR meets
Multi-Contrast MRI Reconstruction*, MICCAI 2025 RIME workshop (arXiv 2509.04888), reproduced from the authors' open
code at commit 08d37c5 and its config values: a 1-D inverse FFT along the readout, then one hash-grid + small MLP
trained per transverse slice, a single network emitting both contrasts, weighted k-space MSE, Adam lr 0.01, 200 steps
per slice, last iterate, no regulariser. Reproduction repo `/data0/congcong/code/GS/mcinr_niessen2025`; results topic
`../T1FLAIR_MCINR_Niessen2025_v3_jiangyizhou`.

It ran on **this topic's own `data/mask_af16_3d.h5`** (2966 points) with this subject's k-space and ESPIRiT maps, so
the row is like for like. Volumes symlinked in as `recon/mcinr_{T1,FLAIR}_complex.npy`, then **re-scored here through
`scripts/t1t2_common.py::Pair.score`** — the same code path every other arm used — agreeing with the reproduction's
own numbers to 1.4e-7 relative on all nine metrics (`scripts/score_mcinr_arm.py`, `logs/mcinr_{T1,FLAIR}.json`).
Registered in `summarize.py` / `readme_tables.py` as `mcinr` and added as a column to `fig_three_planes_compact.py`
and `fig_row_final.py`, next to 3D-INR. It is in the tables above, with its AF 12 counterpart alongside.

| arm | T1 head PSNR / SSIM / HFEN | FLAIR head PSNR / SSIM / HFEN | whole PSNR T1 / FLAIR | wall |
|---|---|---|---|---|
| **MC-INR (Niessen 2025)** | 31.590 / 0.8355 / 0.2325 | 32.243 / 0.8590 / 0.1711 | 30.285 / 32.435 | 11.9 min, both contrasts |
| 3D-INR (single-contrast) | 30.375 / 0.7951 / 0.2125 | 30.257 / 0.7889 / 0.1998 | 32.903 / 33.316 | |
| CG-SENSE | 32.354 / 0.8825 / 0.2239 | 32.826 / 0.8860 / 0.1608 | 34.559 / 35.220 | |
| L1-ESPIRiT | 35.209 / 0.9334 / 0.1286 | 34.503 / 0.9201 / 0.1210 | 38.758 / 38.027 | |
| MC-GS (phase 3) | 35.706 / 0.9396 / 0.1139 | 34.928 / 0.9261 / 0.1121 | 39.268 / 38.585 | |

- **Eighth of eleven on head PSNR on both contrasts**, the same position it takes at AF 12: above 3D-INR
  (+1.215 / +1.986 dB), 3D-DIP and ZF; below CG-SENSE (−0.764 / −0.584), L1-ESPIRiT (−3.620 / −2.260) and MC-GS
  phase 3 (−4.117 / −2.685).
- **Head-only is the only meaningful gauge for this arm.** The network fills air with noise: its whole-volume PSNR is
  second-lowest of every arm on both contrasts, above zero-filled only (30.285 vs 29.949 on T1, 32.435 vs 31.792 on
  FLAIR), and its whole-volume SSIM (0.5575 / 0.6365) is the lowest of every arm, ZF included. At AF 12 the T1 case
  is worse still — there it falls *below* zero-filled.
- **It changes no margin in this topic.** The binding arm in every one of the phase-3 recipe's six head cells is
  GS-single before and after registration; MC-INR is 4.1 / 2.7 dB away from being the bar.
- **It falls off least from AF 12 to AF 16** — −0.60 dB on T1 and −0.80 on FLAIR, against −0.93 to −1.57 for every
  other arm (the AF12 column of the tables above). That is the one direction in which the paper's claim is visible
  here; it is still below CG-SENSE at both accelerations, so it does not turn into a win.
- **Two asymmetries, stated rather than hidden.** Every classical λ was swept on this subject's ground truth and every
  MC-GS knob was tuned on it; ZS-SSDU / 3D-DIP / 3D-INR early-stop on held-out k-space; MC-INR ran at the paper's
  config values with nothing selected here. That does not explain its position — CG-SENSE has no λ to tune and is
  still above it. Separately, the reproduction's own fidelity review
  (`/data0/congcong/code/GS/mcinr_niessen2025/docs/2026-09-13-fidelity-review.md`) finds the code aligned line by
  line but the **protocol** not: the paper's complementary per-contrast sampling is absent (both contrasts share this
  topic's one mask), its ten contrasts come from a single acquisition with a shared coil map and correlated phase
  while ours are two scans, and the evaluation mask and normalisation differ. So this row reads as "below CG-SENSE
  under this topic's shared-mask, two-scan setting", not "the paper does not reproduce". Note that this topic does
  hold a same-budget complementary FLAIR mask (`data/mask_af16c_flair_3d.h5`, `scripts/prep_02c_mask_af16c.m`) — the
  remedy experiment the review proposes could be run on it, and has not been.
- **Figure.** The phase-3 figure of record re-rendered with MC-INR as a tenth method column, same slices, style and
  gauge (`_p3_mcinr` suffix; the nine-method `_p3` pair is untouched):
  `figures/t1flair_v3_af16_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr.png` plus vector PDFs,
  served from `http://localhost:8765/t1flair_v3_af16/`.

## L1-SPIRiT registered as a comparison arm (2026-09-15)

The table had no k-space arm: every comparison method solves for the image (CG-SENSE, the three
CS-SENSE variants and L1-ESPIRiT are SENSE-type; ZS-SSDU, 3D-DIP, 3D-INR and MC-INR are networks
whose unknown is also the image). SPIRiT is the k-space method that is *defined* for this mask --
GRAPPA proper is not, because a GRAPPA kernel is fitted for one fixed neighbourhood geometry and a
variable-density Poisson disc gives every hole a different one. Registered as `spirit`,
"L1-SPIRiT", next to L1-ESPIRiT. **L1-ESPIRiT stays in the table; nothing was replaced.**

**Implementation** (`scripts/recon_spirit.py`). A GPU port of Lustig's toolbox in
`/data0/congcong/code/ESPIRiT`, function by function: `dat2AtA` + `calibrate` (normal equations per
target coil, its own centre tap dropped, Tikhonov `||AtA||_F / rows * tyk`), `SPIRiT('fft')` +
`mtimes` for the operator, `pocsSPIRiT` for the iteration. Per readout slice, in the same hybrid
(x, ky, kz) space `recon_zsssdu.py` uses. The completed multi-coil k-space is combined with this
topic's ESPIRiT map, the operator the ground truth itself is built from, so the arm lands in the
shared gauge; RSS would not.

**Fidelity** (`scripts/spirit_align_check.py`, `scripts/spirit_align.m`, `logs/spirit_align_*.json`).
Against the original MATLAB on the same slice, in double precision: calibration kernel 1.6e-12
(ksize 3) / 4.1e-12 (ksize 5) relative L2, operator 1.2e-12 / 3.8e-12, POCS output after 10
iterations 5.8e-12 / 6.0e-12. In the single precision the arm actually runs in, the same three are
5.1e-4 / 3.3e-4 / 1.1e-4 -- the calibration solves ill-conditioned normal equations, and that is
the whole of the difference. The wavelet is the one piece not shared: Wavelab's `FWT2_PO` there, a
periodised orthogonal DWT written here, same Daubechies 4-tap filter and same 4 levels. They differ
by a signed permutation of the coefficients, which the joint soft threshold commutes with, and the
1e-12 agreement on the POCS output is the check of exactly that.

**Two things this arm turned up.**

1. **The ACS is 24 x 16, not 24 x 24.** `acs_box()` grows the largest fully sampled block around
   the k-space origin and finds 24 (ky) x 16 (kz) at (116, 80) -- the generator's `calib = 24` is
   not the same number of lines in both axes. With 26 coils that decides the kernel size: 5 x 5 is
   240 equations for 650 unknowns (with `tyk = 0` the self-consistency residual on full k-space
   blows up to 152), 3 x 3 is 308 for 234. Both are in the sweep; Tikhonov carries 5 x 5 and the
   two end up within 0.02 dB of each other.
2. **`pocsSPIRiT` diverges on this data, and the original diverges with it.** Its update is
   `x <- G x` on the holes, Richardson with step 1 on an operator that is not a contraction here.
   MATLAB's own output peaks at 5.97 after 10 iterations and 277.7 after 60; the port tracks that
   divergence to 7e-11, so it is the algorithm, not the port. The arm therefore runs a damped
   proximal-gradient step with FISTA momentum on the same objective,
   `min ||(G-I)x||^2 + lam ||W F^-1 x||_1` subject to `x = y` on the mask, with step
   `0.9 / ||(G-I)^H (G-I)||` from a power iteration. It converges (relative change 5.8e-2 -> 6.8e-4
   over 150 iterations). This is the only deliberate departure from the reference.

**Sweep** (`scripts/run_spirit_sweep.sh`, `logs/spirit/`). Kernel size 3 and 5 x lambda in
{0, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3}, 150 iterations, each contrast scored on this subject's ground
truth -- the same privilege every other classical arm here has. Both contrasts and both kernel
sizes peak at **lambda 1e-3, interior to the grid**; the kept row is ksize 5 (T1 33.788 / FLAIR
33.720 head PSNR, against 33.766 / 33.580 at ksize 3). Wall time per volume is 3.7 min on an
otherwise free card and the ~36 min the kept rows report when three of these share three cards that
other users are also on -- this machine is shared, and the arm is bound by memory traffic, not flops.

| head-only, AF 16 | T1 | FLAIR |
|---|---|---|
| L1-SPIRiT (ksize 5, lambda 1e-3) | 33.788 / 0.9083 / 0.1448 | 33.720 / 0.9022 / 0.1207 |
| plain SPIRiT (lambda 0, same solver) | 29.291 (ks3) / 28.444 (ks5) | 29.337 / 28.789 |
| GS-single, the strongest comparison arm | 35.558 | 34.886 |
| CS-SENSE (TV, converged) / joint CS-SENSE, the strongest non-GS arm | 34.945 | 34.355 |
| CG-SENSE, the arm below it | 32.354 | 32.826 |

- **Sixth of the eleven comparison arms on head PSNR on both contrasts.** Eleven counts GS-single,
  which is a comparison arm here (user, 2026-09-13), and excludes L1-ESPIRiT, dropped 2026-09-15.
  Below ZS-SSDU (-0.362 / -0.285) and every CS-SENSE variant, above CG-SENSE (+1.434 / +0.894),
  MC-INR, 3D-DIP, 3D-INR and ZF. It is 1.770 / 1.166 dB behind GS-single, the strongest comparison
  arm, and 1.157 / 0.635 dB behind the strongest non-GS arm.
- **The self-consistency prior alone does not carry this acceleration.** Dropping the wavelet term
  costs 4.5 dB and puts the arm below everything except ZF. What is competitive here is the hybrid
  -- k-space self-consistency plus an image-domain sparsity prior -- not "k-space" as such.
- **It is the sharpest comparison arm on FLAIR.** Head HFEN 0.1207 is the lowest of every
  comparison arm (L1-ESPIRiT 0.1210, ZS-SSDU 0.1529) and sharpness 0.890 the highest; only the four
  MC-GS / GS-single rows have lower HFEN. On T1 its HFEN 0.1448 also beats CS-SENSE (0.1531) and
  ZS-SSDU (0.1646). It loses on PSNR and SSIM, not on high-frequency fidelity: it is sharper and
  noisier than the SENSE-type arms.
- **Two checks that the position is not an artefact of what it was given.** The topic's ESPIRiT
  maps come from `bart ecalib -r 24` on the FULL k-space, so the centre 24 x 24 block is something
  every other arm -- and the ground truth -- already rests on, while the arm above calibrates only
  from the 24 x 16 the mask actually samples. Giving it that same block (`--calib full24`) is worth
  **+0.009 dB** (33.775 vs 33.766 at ksize 3, `logs/spirit_T1_cal24_ks3_lam0.001.json`). Running
  400 iterations instead of 150 is worth **+0.017 dB** (33.783, `logs/spirit_T1_it400_*.json`).
  Together under 0.03 dB against a 1.42 dB gap.
- **It changes no margin in this topic.** The binding arm in every one of the phase-3 recipe's six
  head cells is GS-single before and after; the strongest non-GS comparison arm is now CS-SENSE
  (TV, converged) on T1 and joint CS-SENSE on FLAIR.

**Figures with the k-space arm in the L1 column** (user request 2026-09-15). The figure of record
re-rendered with `fig_three_planes_compact.py --l1_arm spirit`, same slices, style, gauge and
recipe keys. **The existing `_p3_mcinr` pair is untouched** — the flag appends its own suffix:

```
figures/t1flair_v3_af16_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_p3_mcinr_spirit.{png,pdf}
```

served from `http://localhost:8765/t1flair_v3_af16/`, mirrored in `/home/congcongliu/figs/t1flair_v3_af16/`.

Two checks, because a figure of record should not be re-rendered on trust. Re-running the command
above with `--l1_arm espirit` reproduces the existing `_p3_mcinr` figure **pixel for pixel** (0
pixels differing), which is what establishes that this is the command that made it. And the
`_spirit` render differs from it only inside one column slot (a 337-pixel span, 0.96 of the 1/11
width), so nothing but that column moved.

**The `_bothl1` variant was made and then dropped.** An eleven-column render keeping both L1 arms
was produced first, as a hedge: removing the strongest comparison arm makes MC-GS's visual and
numerical lead look larger than it did. The user dropped L1-ESPIRiT from the comparison entirely on
2026-09-15, so that variant was deleted and only `_spirit` is kept. `--l1_arm both` still exists and
regenerates it in about six minutes per contrast if it is ever wanted. The effect on the numbers is
quantified in "L1-ESPIRiT dropped from the comparison" above: the margin grows by 0.26 dB on T1 and
0.15 dB on FLAIR purely from the arm no longer being shown.

**AF label corrected, 2026-09-15.** The title's acceleration now reads `AF_vs_full_rect` — whole
phase-encode plane over the points that contrast actually has data on, the user's 2026-09-14
definition, the same number BART's `pics` prints as `Acc`. It used to read `AF_vs_own_support`,
against the contrast's own vendor shutter, which is the superseded gauge; only FLAIR moves
(12.49x -> 16.49x), T1 was already right at 15.19x. Both numbers were always in
`logs/prep_04_export.json`; the figure took the wrong field. Consequence for the check above: the
pixel-identity re-render was done BEFORE this fix, so repeating it now reproduces the old figure
exactly on T1 and everywhere but the title on FLAIR.

**The figures of record themselves, re-archived with the corrected label (user, 2026-09-15: keep the
old, re-archive the new).** The `_p3` and `_p3_mcinr` pairs at 180/124/76 inset48 -- the historical figures, L1-ESPIRiT
column and all -- were re-rendered under their canonical names by `scripts/archive_and_relabel_figs.sh`
(in the AF16 topic) with the corrected FLAIR title; the originals are in `figures/archive_aflabel_pre20260915/` and in the
served mirror `/home/congcongliu/figs/t1flair_v3_af16/archive_aflabel_pre20260915/`, PNG and PDF, 8 files. Verified
against those originals: every T1 re-render is **pixel-identical**, every FLAIR re-render differs
**only in the title band** (image rows 13-35 of 2829). So the canonical names now carry the right
label and nothing else changed.

One thing went wrong on the way and is recorded here rather than hidden: the first pass of that
script built the archive file name with a doubled underscore, matched nothing, and the re-render
overwrote the originals before they had been moved. They were recovered from the committed
snapshot of this topic in the framework worktree (`mcgs_af16_p3_wt/t1flair_af16_v3_jiangyizhou/figures`, PNGs, 2026-09-13/14) and from this topic's
`export/` bundle (the PDFs the worktree does not carry), then the verification above was run against
the recovered files. The script is fixed. Older renders of this topic (the 174/110/47 pairs)
still carry the old FLAIR label and were not touched.
