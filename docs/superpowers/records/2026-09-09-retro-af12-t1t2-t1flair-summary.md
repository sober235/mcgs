# Retrospective AF 12 on true multi-contrast 3D pairs — session record, 2026-09-08 / 09

Three topics, forty-eight reconstruction arms, one gauge. Archives live under
`Results_20260908/{T1T2_AF12_retro, T1FLAIR_AF12_retro, T1FLAIR_AF12_v3_jiangyizhou}` on amax and
are mirrored (without the GB-scale volumes) as `t1t2_af12_retro/`, `t1flair_af12_retro/`,
`t1flair_af12_v3_jiangyizhou/` in this repo. Each has a README with the full numbers and the
regeneration commands; this record is the cross-cutting summary.

## 1. The data question came first, and its answer corrects the project's own bookkeeping

The frozen MC-GS work used one pair, T1 + FLAIR, from `/data1/.../3D/{T1,FLAIR,T2}_csm_ksp`
(12 coils, volume 256x220x256). Two facts about that tree, verified by md5 on both the `/data1`
originals and the `/data0/congcong/data/T1_T2_FLAIR_3D` copies:

- **`T2_csm_ksp/` is mostly a duplicate of `FLAIR_csm_ksp/`.** Of the 15 subjects present in both,
  12 are byte-identical, including `huangyuwei`, the subject every frozen number rests on. Only
  `790_jiangdian`, `790_tangbin` and `tangbing` genuinely differ. An earlier audit note that
  described `T2_csm_ksp/huangyuwei` as a separate T2 acquisition is wrong; it is the FLAIR file.
- The true 3D T1 + T2 (+ FLAIR) raw data is elsewhere: `/data1/data1_congcong/data/MC_Jia/
  Train_SpiritDiffusion_CFL/vol{1..7}_*`, BART CFL, **seven subjects x three contrasts**,
  `(256, 256, 176, 26)` complex64, 26 coils. Subject 7 has a second copy plus the vendor `.raw`
  under `/data0/congcong/data/MultiContrast3D/`.

## 2. Geometry, verified rather than assumed

| axis | direction | plane by fixing it | role |
|---|---|---|---|
| axis0 (256) | superior-inferior | **transverse** | readout, fully sampled |
| axis1 (256) | anterior-posterior | coronal | phase encode ky |
| axis2 (176) | left-right | sagittal | phase encode kz |

Readout by the all-or-nothing line test (partial-line fraction 0.0000 on axis0 vs 1.0000 / 0.75
on the others); left-right by mirror symmetry (corr with own flip 0.681 vs 0.394 / 0.487); the
planes confirmed by inspection (`figures/axis_orientation_check.png`). **The vd-Poisson mask on
(ky, kz) therefore lies in the transverse plane with the readout head-foot, the same physical
arrangement as the frozen house convention** — only the array order differs (S-I is axis0 here,
axis2 in the old bundle). k-space DC sits exactly on the array centre of both PE axes, so this
data does not carry the ACS/DC misregistration that cost the old bundle about 1 dB. T1 has no
shutter; T2 and FLAIR carry an elliptical vendor shutter (support 0.7771 / 0.7577), which the
`known_zero` estimator recovers from the acquired lines alone (21.5 % declared vs 22.3 %
measured, zero nonzero lines swallowed). 26 coils are kept: 20 are needed for 99 % of the energy.

## 3. Undersampling and gauge

House `vdPoisMex` mask on 256x176, `accelx 3 / accely 4 / calib 24 / ellipse 0 / pp 1`, one mask
for both contrasts, 3847 / 45056 points = 8.538 %, **AF 11.71** on the full PE plane. Against
each contrast's own acquired support: T1 11.71, T2 9.89, FLAIR 9.74 — both numbers are reported
everywhere because the variable-density mask is densest exactly where the shutter keeps data.

Ground truth = conjugate-SENSE combine of the full k-space with the frozen ESPIRiT recipe
(`ecalib -t 0.001 -c 0.0 -r 24 -m 1`, ~40 min per contrast, CPU only); metrics = whole@max
PSNR/SSIM/NMSE/HFEN plus the head-only set (`|GT| > 0.05 P99.9`). `scripts/check_gauge.py`
proves the shared scoring module reproduces the classical script to the last digit, so the eight
arms are comparable. **Whole-volume SSIM is misleading on this data** (CG-SENSE 0.727 below ZF's
0.831 while 5.4 dB better in PSNR): `ecalib -c 0.0` leaves full-amplitude noise in the GT's air.
Head-only numbers are the ones to quote.

## 4. Eight methods, three studies

Same eight arms and hyper-parameters as the CE pre/post study: ZF, CG-SENSE, CS-SENSE (ADMM-TV),
ZS-SSDU, 3D-DIP, 3D-INR, MC-GS single (same contrast in both channels), MC-GS joint (shared
geometry). No arm sees the ground truth during fitting or model selection. MC-GS runs at the
frozen edge-TV + Charbonnier working point with **one deliberate deviation: no warm start** —
the 576902-Gaussian dual-warm cloud belongs to the old FOV and cannot be registered here, so
every GS run is a cold start from 600k points.

Head-only PSNR, dB:

| method | hyw T1 (T1+T2) | hyw T2 | hyw T1 (T1+FLAIR) | hyw FLAIR | jyz T1 | jyz FLAIR |
|---|---|---|---|---|---|---|
| CS-SENSE | **34.61** | **34.22** | **34.61** | 33.25 | **35.76** | **35.08** |
| ZS-SSDU | 34.13 | 33.82 | 34.13 | **33.27** | 35.72 | 35.03 |
| MC-GS joint | 33.29 | 33.12 | 33.18 | 31.81 | 34.16 | 33.39 |
| MC-GS single | 33.30 | 33.50 | 33.30 | 31.76 | 33.80 | 33.41 |
| CG-SENSE | 31.94 | 32.51 | 31.94 | 32.19 | 33.51 | 33.96 |
| 3D-INR | 30.84 | 31.02 | 30.84 | 29.94 | 31.64 | 31.19 |
| 3D-DIP | 30.11 | 29.71 | 30.11 | 29.40 | 31.36 | 30.81 |
| ZF | 23.70 | 25.19 | 23.70 | 26.74 | 26.13 | 28.19 |

(hyw = vol1_huangyuwei, jyz = vol3_jiangyizhou. The hyw T1 columns coincide because the T1 single
arms were reused across the two pairings — identical inputs, seed and gauge.)

**The ranking is stable across subjects and pairings.** CS-SENSE and ZS-SSDU are first and second
within 0.1 dB on every column, MC-GS 1.3 to 1.7 dB behind, then CG-SENSE, then DIP / INR. This is
not the CE ranking (CG-SENSE first, GS last) and neither should be carried across: the CE study
had no ground truth and ranked by the k-space residual on acquired points, which rewards free
parameters rather than image quality.

DIP never converged inside its budget (best iterate at or near the last one, every time); INR
peaked early and over-fitted (highest HFEN, sharpness 1.2 to 1.3 against a GT of 1.0).

## 5. The shared-geometry claim, six channel-pairings

| subject | pairing | channel | joint − single, whole | joint − single, head | head SSIM |
|---|---|---|---|---|---|
| huangyuwei | T1+T2 | T1 | +0.343 | −0.001 | +0.003 |
| huangyuwei | T1+T2 | T2 | +0.090 | −0.376 | −0.002 |
| huangyuwei | T1+FLAIR | T1 | +0.891 | −0.114 | +0.002 |
| huangyuwei | T1+FLAIR | FLAIR | +0.290 | +0.051 | +0.006 |
| jiangyizhou | T1+FLAIR | T1 | +0.820 | **+0.361** | **+0.014** |
| jiangyizhou | T1+FLAIR | FLAIR | +0.442 | −0.018 | +0.006 |

Whole-volume PSNR gains in six of six. Head-only PSNR: mean −0.016 dB, one gain above 0.1 dB,
two losses below −0.1 dB. The joint model **can** gain inside the head (jiangyizhou's T1, visible
in the error map as less parenchymal speckle than the single arm), but which channel benefits is
not predictable from the pairing, and averaged over what has been run the head-only effect is
zero. The whole-volume gain is robust and is background suppression by the explicit
representation. Co-registration is not the explanation: T1/FLAIR is registered to (0,0,0) voxels
against (−1,0,0) for T1/T2 and the head-only result does not improve with it.

A per-slice sweep (`scripts/find_best_slice.py`) confirms the joint model does not win on
representative anatomy: on huangyuwei T1 it ranks first on 18 of 556 slices, all peripheral
(head area 2044–9481 voxels, median 17372); on the 228 slices above 20000 voxels CS-SENSE wins
all 228. The honest figure for the method is the joint-versus-single ablation on a full axial
slice (`t1t2_af12_T1_claim_axis0_154.png`: joint second of eight, 0.13 dB behind CS-SENSE,
0.33 dB ahead of its own single arm).

## 6. What is left on the table

- **The warm start.** Every MC-GS run here is cold. The frozen headline (38.4 dB on the old
  bundle) came from a warm-started 576902-Gaussian cloud; building one on this grid is the one
  untested lever for the 1.3 to 1.7 dB gap to CS-SENSE and the obvious next experiment.
- Four more subjects (vol2, 4, 5, 6, 7 minus the two done) are one command away:
  `t1flair_af12_retro/scripts/setup_subject.sh <vol_dir> <subject> <tag>`.
- `scripts/figure_viewer.html`: a zoom/pan viewer for the served figures
  (`http://localhost:8765/view.html?img=<dir>/<file>.png`), added because Ctrl+wheel on a bare
  PNG zooms the page, not the image.

## Gotchas recorded along the way

- Dropping `csm_path` on this config falls back to an ACS estimate that returns an all-zero map
  (PSNR ~20, NMSE exactly 1). Always pass `csm_path`.
- `ecalib` here has no GPU path; budget ~40 min per 26-coil contrast.
- A cloned topic must write its figures to its own directory: the first clone overwrote the
  parent's panels. `setup_subject.sh` carries the fix.
- The trainer's own prints are block-buffered; read `results_arc/<exp>/<exp>/joint_train.log`,
  not the launcher log, for progress.
