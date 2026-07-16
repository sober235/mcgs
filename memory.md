# Project memory — house conventions (mask + CSM 口径)

> The reproducible recipes for the two acquisition-side artifacts of the AF=12 T1+FLAIR
> dual-contrast recon: the undersampling **mask** and the **CSM** (coil sensitivity maps).
> Every fact below was verified against the real files on 2026-07-15 (not recited).
> Winner method / hyperparameters live in `.claude/memory.md`. Source configs: `configs/winner_v2.yaml`.

---

## 1. Undersampling mask — vd-Poisson (separable, prospectively-Cartesian-realizable)

**What it is.** A 2D **variable-density Poisson-disc** pattern on the two phase-encode (PE) axes,
**broadcast identically along the fully-sampled readout axis** (S-I) → separable → acquirable on a
real Cartesian scanner. Same mask for BOTH contrasts.

**Generator.** MATLAB **`vdPoisMex`** (Lustig ESPIRiT toolbox), on amax at
`/data0/congcong/code/ESPIRiT/utils/` (`vdPoisMex.mexa64` + `.c` source + `.m` help + `gen_vdpois_masks.m`).

**Exact generation command** (from `gen_vdpois_masks.m`, AF=12):
```matlab
mask = vdPoisMex(220, 256, 220, 256, 3, 4, 24, 0, 1);  % AF = accelx*accely = 3*4 = 12
mask = single(mask ~= 0);  save('mask_vdpois_220x256_af12.mat','mask','-v7');
```
Signature `vdPoisMex(sx, sy, fovx, fovy, accelx, accely, calib, ellipse, pp)`:
| arg | value | meaning |
|---|---|---|
| sx,sy / fovx,fovy | 220,256 | PE-plane size |
| accelx, accely | **3, 4** | per-axis accel → nominal AF 3×4 = **12** |
| calib | **24** | central fully-sampled ACS (calibration) lines |
| ellipse | 0 | no corner cut (rectangular FOV) |
| pp | **1** | polynomial order of variable density (0=uniform, 1=linear) |

**Algorithm (vdPoisMex.c).** (1) variable-density radius field `R=(y²+z²)^(pp/2)` on normalized
y,z∈[−1,1] → larger min-distance at the edges (sparser); (2) central calib region forced sampled;
(3) numerical bisection to hit target accel; (4) Bridson Poisson-disc rejection sampling
(`est_accl = 1.24²·masksum/dsum`, 1.24 = blue-noise packing constant) → blue-noise pattern.

**To change AF:** set `accelx = AF/4` (AF12→3, AF10→2.5, AF6→1.5); accely=4, calib=24, ellipse=0, pp=1 fixed.
Prebuilt masks exist: `/data0/congcong/code/ESPIRiT/utils/mask_vdpois_220x256_af{08,10,12,14,16}.mat`.
Helper: `python scripts/make_mask.py --af 12` prints the MATLAB command; `--verify <mat>` checks samp/AF/ACS.

**Shipped file** (winner): `/data0/.../3dgsVC/mask/mask3d_vdpois_af12_transverse.mat` (keys `mask`,`net_af`).
Config uses `mask_type: external`; `paired_gs/data/dataset.py::_conform_mask_shape` stacks the 2D pattern
along the readout axis and transposes to the (256,220,256) volume layout.

**Verified on the real file:**
| property | measured | meaning |
|---|---|---|
| shape / samp / net AF | (256,220,256) / **8.45%** / **11.83** | matches `net_af` key |
| separable | axis2 = **100% identical slabs** | 2D PE pattern (256×220) broadcast along readout |
| ACS | central 16×16 = **1.000**, 24×24 = 0.918 | calib=24 fully-sampled center (disc) |
| variable density | radial samp 0.98→0.44→0.20→0.10→0.06→**0.04** | dense center, sparse edge |
| blue-noise | NN dist mean **2.54** vs white-random 1.85 | Poisson-disc min-distance property |

**Why net AF 11.83 (not 12):** the calib=24 ACS is added back fully sampled, raising total samples above the
nominal 1/12. **Why reusable/realizable:** undersampling is entirely in the 2D PE plane; readout is free.

---

## 2. CSM — ESPIRiT (BART `ecalib`), per contrast, SoS≡1 full-FOV

**What it is.** Coil sensitivity maps estimated by **BART ESPIRiT `ecalib`** from a **fully-sampled ACS**
(central low-freq k-space). Estimated **separately per contrast** (T1 CSM ≠ FLAIR CSM). The SAME CSM
defines BOTH the forward operator and the ground truth, so a clean smooth CSM is essential.

**Exact recipe** (from `scripts/estimate_csm.py`; BART on amax `TOOLBOX_PATH=/home/congcongliu/src/bart`):
```bash
# ACS input = readout axis fully sampled + central 32×32 in the transverse PE plane
bart ecalib -t 0.001 -c 0.0 -r 24 -m 1  <acs_kspace>  <csm_out>
# save .mat key 'map', complex (256,220,256,12) = (axis0,axis1,axis2,coil)
```
| flag | value | meaning / why |
|---|---|---|
| -t | **0.001** | eigenvalue threshold — empirically smoothest (U-shaped vs -t) |
| -c | **0.0** | no soft crop → unit-modulus maps span whole FOV → **SoS ≡ 1 everywhere (incl. air)** |
| -r | **24** | 24×24 calibration region |
| -m | **1** | one set of ESPIRiT maps (single map, not multi-map soft-SENSE) |

**"Fully-sampled ACS" = the central, no-undersampling low-freq block of k-space.** CSM are smooth →
low-spatial-frequency → a small fully-sampled center (32×32) suffices for full-resolution maps. Because the
ACS is the fully-sampled center, it is **independent of the peripheral undersampling mask → CSM reusable
across AF** (mask changes, CSM does not). In this project the raw `ksp` is fully sampled (nonzero_frac=1.0,
`presampled:false`) so undersampling is retrospective; the ACS is just the central crop of that full data.

**Usage in pipeline** (`paired_gs/data/dataset.py`): forward `A·x = M·F3c(S·x)`; GT
`x_gt = Σ_c conj(S_c)·ifft3c(y_full_c)` (:313–315, conjugate-SENSE / matched-filter combine of fully-sampled
multicoil). SoS≡1 makes this the minimum-variance estimate. Coil axis auto-inferred (`_infer_coil_axis`)→dim0.

**Shipped files** (winner): `/data0/.../3dgsVC/csm_espirit/csm_espirit_{T1,FLAIR}.mat`, key `map`.
NOTE: the raw data `.mat` ALSO bundles a `map` CSM, but the winner points `csm_path` at these **external
ESPIRiT** files, not the bundled one.

**Verified on the real files:**
| property | T1 | FLAIR |
|---|---|---|
| shape / dtype / coils | (256,220,256,12) complex64 / 12 | same |
| SoS (∑\|S_c\|²)^½ | **1.000** everywhere, in-FOV std **0.0000** | same (the `-c 0.0` signature) |
| per-coil \|S\| range | [0, 0.991], complex (phase used) | [0, 0.989] |
| smoothness mean\|∇\| | 0.0030 | 0.0048 |
| per-contrast | **T1 ≠ FLAIR**, max\|diff\|=1.59 (separately estimated) | — |

**Two "24"s that are unrelated:** ecalib `-r 24` (CSM calibration kernel) vs vdPoisMex `calib=24`
(undersampling ACS). Same number, different roles. Reproduce/verify: `scripts/estimate_csm.py --recipe` / `--verify <csm.mat>`.

---

## 3. Why T1(plain) is harder than FLAIR(contrast) — spectrum–sampling mismatch (2026-07-16)

> In the winner recon T1's relative error is ~2.4× FLAIR's (NMSE). Root cause is a **spectrum–sampling
> mismatch**, NOT a bug / CSM / geometry issue — and it CANNOT be recovered by reallocating samples; the
> only lever is a learned prior for unsampled high-freq. Every number verified against real files
> (`reg_aligned_volumes.npz` + raw multicoil `ksp` + pre-gen masks); multicoil-gauge self-check reproduces
> §8's af12 73.8% / 89.8% at rel-err 0.00%.

**H2 (main cause) — k-space energy distribution.** T1 energy sits at higher freq. PE radius to reach:
| energy | T1 | FLAIR |
|---|---|---|
| 90% | 18.2 px | 11.4 px |
| 95% | 34.2 px | 23.1 px |

Real multicoil retained energy (‖zf‖²/‖gt‖², brain): **T1 73.8% / FLAIR 89.8%** (loses 26% vs 10%, 2.5×).
vd-Poisson is sparse at high freq ⇒ T1 loses more. FLAIR is water-suppressed ⇒ energy concentrated low-freq.

**H1 (image-domain root, mild).** FLAIR near-zero brain voxels 16.3% vs T1 10.7%, sparser (L1/L2 0.746 vs
0.805); but TV nearly equal and entropy higher, so the image-domain gap is moderate — H2 (frequency) is the
strong evidence.

**H3 (metric-gauge artifact, important).** Pipeline/TSV PSNR uses `data_range = max(gt)`. T1's peak is
higher (1.887 vs FLAIR 1.568) ⇒ inflates T1 PSNR by ~1.6 dB ⇒ makes T1 (38.51) *appear* to beat FLAIR
(38.26). Under a unified data_range **FLAIR 34.35 > T1 32.99 (leads ~1.36 dB)**, consistent with NMSE
(2.4×), SSIM, HFEN. ⚠️ Never compare contrasts with @max-PSNR; use NMSE or a fixed data_range. This
**corrects `.claude/memory.md` §8**'s "T1 crosses 38 / surpasses FLAIR" reading — T1's 38.51 value is fine,
but it does NOT actually beat FLAIR.

**Contrast-adaptive sampling = dead end (ruled out).** At fixed total budget (2×8.45%), reallocating
samples between T1/FLAIR barely moves retained energy — doubling sampling lifts T1 only +1.4 pts;
budget-fixed reallocation min 73.8→74.3 (+0.5). T1's missing energy is a far high-freq tail that even
af08 (12.65%) can't reach ⇒ no free lunch on the acquisition side ⇒ locks in the learned-prior direction.

**Figures** (`results_arc/`, jet error maps [0,0.1]): `tf_diff_H123.png` (H1/H2/H3 diagnosis),
`tf_adaptive_mask_multicoil.png` (adaptive-sampling null result, faithful multicoil gauge),
`final_recon_reg_aligned_T1_FLAIR.png` (winner, 3 views × [ZF|GT|Recon|err]). Analysis scripts live in the
session scratchpad (not committed).
