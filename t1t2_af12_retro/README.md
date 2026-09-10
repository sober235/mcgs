# Retrospective AF 12 on a 3D T1 + true T2 pair (new data)

Canonical results directory for this line of work. **All current and subsequent results go here.**

## Why this data

Every MC-GS / Paired-ARC-GS result up to now used ONE pair: T1 + **FLAIR**, 12 coils, volume
(256, 220, 256), from `/data1/.../3D/{T1,T2,FLAIR}_csm_ksp`. Two facts about that tree, both
verified here on 2026-09-08 and both worth writing down:

- `T2_csm_ksp/` is mostly a **duplicate of `FLAIR_csm_ksp/`**. Of the 15 subjects present in both,
  **12 are byte-identical** (head and tail 100 MB md5, on both the `/data1` originals and the
  `/data0/congcong/data/T1_T2_FLAIR_3D` copies). Only `790_jiangdian`, `790_tangbin` (different
  file size) and `tangbing` genuinely differ; a DC-centred reconstruction of `tangbing` shows the
  T2-directory file is a third, distinct acquisition. There is also `Train_3dbrain_xxxx.h5` with
  no FLAIR counterpart. **The subject the frozen work uses, `huangyuwei`, is one of the 12** — so
  the frozen "T1+T2" pairing would have been T1 + FLAIR twice.
- So "3D T1 and T2 with k-space" needed a different source. (This corrects an earlier audit note
  that described `T2_csm_ksp/huangyuwei` as a separate T2 acquisition; it is the FLAIR file.)

It is `/data1/data1_congcong/data/MC_Jia/Train_SpiritDiffusion_CFL/` — BART CFL, **7 subjects**,
each with **three** contrasts (`t1_fsp`, `t2`, `flair`), `(256, 256, 176, 26)` complex64, 26 coils.
A second copy of subject 7 sits in `/data0/congcong/data/MultiContrast3D/`. This work uses
`vol1_huangyuwei` — the same subject and exam as all the frozen results, so the numbers are as
comparable as a different grid and coil count allow.

| | file | PE-plane support |
|---|---|---|
| T1 | `t1_fsp_train_huangyuwei_cc` | **1.0000** (no shutter) |
| T2 | `t2_train_huangyuwei_cc` | 0.7771 (elliptical vendor shutter) |
| (FLAIR, unused here) | `flair_train_huangyuwei_cc` | 0.7577 |

The other six subjects are identical in shape and in support fraction
(`scripts/prep_00_scan_all_cfl.py`), so this study extends to them without new plumbing.

## Geometry (verified, not assumed)

`scripts/prep_01_inspect.py` → `logs/prep_01_inspect.json`.

- Volume `(256, 256, 176)`; **readout_axis = 0**, **phase_axes = [1, 2]** = (ky, kz) = (256, 176).
  The readout axis is decided by an all-or-nothing test: fixing axis 0, every PE line is either
  fully nonzero or fully zero (partial-line fraction 0.0000). Fixing axis 1 gives 1.0000 partial
  lines and axis 2 gives 0.7500 — neither can be the readout.
- **k-space DC sits exactly on the array centre of both PE axes, (128, 88).** This data does NOT
  carry the ACS/DC misregistration that cost the old bundle about 1 dB, so no roll is applied.
- Coils are **not** compressible for free: 20 of 26 virtual coils are needed for 99 % of the
  central-block energy, 25 for 99.9 %. All 26 are kept.
- **Anatomical axis labels, adjudicated 2026-09-09** (`figures/axis_orientation_check.png`):

  | axis | anatomical direction | plane you get by fixing it | role |
  |---|---|---|---|
  | axis0 (256) | superior-inferior | **transverse (axial)** | readout, fully sampled |
  | axis1 (256) | anterior-posterior | coronal | phase encode ky |
  | axis2 (176) | left-right | sagittal | phase encode kz |

  Three independent checks agree. Mirror symmetry holds only about axis2 (corr(v, flip) = 0.681
  against 0.394 and 0.487), so axis2 is the left-right axis. Fixing axis0 walks a textbook
  inferior-to-superior axial series: skull base with orbits, midbrain and cerebellum, lateral
  ventricles, vertex. Fixing axis2 gives the mid-sagittal plane with corpus callosum, brainstem,
  cerebellum and spine. The acquisition is a sagittal 1 mm slab, and 176 sagittal slices along
  left-right is exactly axis2.

  **So the undersampling mask, which lives on (ky, kz) = (axis1, axis2) = anterior-posterior by
  left-right, lies in the TRANSVERSE plane, and the fully sampled readout runs head-foot.** That
  is the same physical arrangement as the frozen house convention (Poisson pattern on the
  transverse plane, readout along S-I); only the array ordering differs, since S-I is axis0 here
  and axis2 in the (256,220,256) bundle.

### T1 / T2 co-registration

`scripts/prep_06_align_check.py` → `logs/prep_06_align.json`. The joint model shares one Gaussian
geometry between the contrasts, so the pair must be registered.

| quantity | value |
|---|---|
| peak cross-correlation shift of the two brain masks | **(-1, 0, 0) voxels** |
| brain-mask Dice | 0.860 |
| intensity-centroid offset | (10.19, 0.38, 0.12) voxels |

The 10-voxel centroid offset along axis 0 is a **contrast** effect, not a misregistration: T1 and
T2 weight fat, marrow and CSF differently, so a fixed threshold selects different amounts of neck
and scalp. The cross-correlation peak — the quantity that actually measures rigid displacement —
is one voxel. Figure `figures/rss_T1_T2_three_planes.png` shows the two fully sampled RSS volumes
on the same slices.

## Retrospective undersampling

House-convention variable-density Poisson-disc mask on the (ky, kz) plane, generated by the
ESPIRiT `vdPoisMex` MEX with the same settings as the frozen AF12 recipe
(`accelx = 12/4`, `accely = 4`, `calib = 24`, `ellipse = 0`, `pp = 1`); readout fully sampled;
**one mask for both contrasts**. `scripts/prep_02_mask.m` → `data/mask_vdpois_256x176_af12.mat`,
broadcast along the readout by `scripts/prep_04_export.py` → `data/mask_af12_3d.h5`.

```
PE grid 256 x 176 = 45056 points;  sampled 3847  =  8.538 %  ->  AF 11.712 over the full grid
central 16x16 fully sampled, 20x20 at 92.5 %, 24x24 at 83.7 %
   (the frozen 220x256 AF12 house mask behaves the same way: full to 20x20, 91.8 % at 24x24)
```

Effective acceleration is not one number, because T2 is shuttered and T1 is not:

| contrast | sampled, % of PE plane | AF vs the full grid | AF vs its own acquired support |
|---|---|---|---|
| T1 | 8.538 | **11.71** | **11.71** |
| T2 | 7.859 | 12.72 | **9.89** |

T2's two numbers differ because a variable-density mask is densest at the centre, which is exactly
where the elliptical shutter keeps data. Reported honestly: the *design* is AF 12 on the full PE
plane, T1 sees 11.71x, T2 sees 9.89x relative to what its scan actually acquired.

`presampled: false`, so the loader builds the ground truth from the **full** k-space with the same
ESPIRiT maps that define the forward operator, then applies the mask. That is what makes this
retrospective rather than a GT-free comparison.

## Coil sensitivities

Frozen house recipe, unchanged: `bart ecalib -t 0.001 -c 0.0 -r 24 -m 1`, run directly on the CFL
(already in BART dim order). 38 min for T1, 45 min for T2 on a loaded 112-core host.
`scripts/prep_03_csm.sh`, converted to h5 by `scripts/prep_05_csm_to_h5.py`.

SoS is **1.0000 everywhere** (min = mean = max) for both maps. With `-c 0.0` the single
eigenvector map has unit modulus over the whole FOV including air; that is the documented,
expected behaviour of this recipe, not a defect.

## The known-zero region on T2

The elliptical shutter is a **measured zero**, not missing data, so `known_zero: true` is set on
T2 (and only T2 — T1 has no shutter, the estimator would be a no-op). The estimator recovers it
from the acquired lines alone:

```
[physics:contrast] known_zero=True  ellipse=(128.0, 88.0, expand 1.0)  frac_z_used=0.2151
                   zero_lines=306/3847 explained=294  nonzero_in_z=0
```

21.51 % of the PE plane declared known-zero against a measured shutter of 22.29 %, with **zero**
sampled nonzero lines wrongly swallowed. `prewhiten` is deliberately left off: its per-contrast
choice in the `af12_kz_pwc` base1 was tuned on the old bundle and is not validated here.

## Methods compared

All eight reconstructions use the same data, the same mask, the same maps and the same gauge.
The set is the same eight the CE pre/post study compared, with the same hyper-parameters.

| method | what it is | script |
|---|---|---|
| ZF | zero-filled SENSE combine | `recon_classical.py` |
| CG-SENSE | CG on the Tikhonov normal equations, tik 0.01, 60 iterations | `recon_classical.py` |
| CS-SENSE | ADMM isotropic 3-D TV SENSE, lam 1e-3, mu 0.1, 35 outer x 8 CG | `recon_classical.py` |
| ZS-SSDU | zero-shot self-supervised DL (Yaman et al.): 15-layer 64-channel ResNet denoiser unrolled 10x around a CG data-consistency block, trained on this subject only. The mask is split Gamma 10 % (early stopping) and a pool re-split Theta/Lambda 80/20 every step. 100 epochs max, patience 10 | `recon_zsssdu.py` |
| 3D-DIP | untrained 3D ConvDecoder (Darestani & Heckel), deliberately under-parameterised, 3.87 M params against 23.07 M output values. 5000 iterations, lr 3e-3 | `recon_dip_inr.py` |
| 3D-INR | 3D coordinate network, hash-grid encoding into a small ReLU MLP (SUMMIT backbone, complex weighted-image head), 16.79 M params. 1500 iterations, lr 5e-3 | `recon_dip_inr.py` |
| MC-GS single | the frozen model with the SAME contrast in both channels (2x the single-scan objective; Adam is invariant to a global loss scale, so this is that contrast reconstructed alone) | `arc/cycle.py` |
| MC-GS joint | the frozen model with T1 in the plain channel and T2 in the contrast channel, one shared Gaussian geometry, per-primitive dual complex density | `arc/cycle.py` |
| MC-GS joint v2 | the same joint model with the cold-start path repaired (see the section below): points drawn inside the zero-filled head support, 3-NN spacing scales, edge-TV kappa from the ZF image, 1.2M points | `autoresearch_t1t2.sh` |

**No method sees the ground truth during fitting or model selection.** ZS-SSDU stops on a held-out
10 % of the acquired mask; DIP and INR stop on a held-out 5 % of the acquired PE lines, the same
95/5 protocol the CE study used; the rest have no free stopping decision. The ground truth enters
only at scoring.

MC-GS runs at the frozen "OUR METHOD" working point: edge-weighted TV with Charbonnier
`eps = 0.04`, `lambda_geo 0.10`, `kappa_q 0.85`, `w_floor 0.10`, `kspace_high 2.2`, warmup 500,
`max_radius 20`, 4000 iterations, seed 42. **No warm start**: the 576902-Gaussian dual-warm cloud
was fit on the (256,220,256) FOV and its normalised geometry does not register to this volume, so
training starts from a fresh 600 000-point cloud. That is the one deliberate deviation from the
frozen recipe and it costs the joint run whatever the warm start was worth.

## Results

All eight methods, both contrasts, one gauge. `scripts/check_gauge.py` proves the shared module
in `scripts/t1t2_common.py` reproduces `scripts/recon_classical.py` to the last digit on all nine
metrics, so the learned arms and the classical arms are directly comparable.

**T1** — sorted by head-only PSNR, which is the number that reflects image quality here:

| method | PSNR | SSIM | NMSE | HFEN | head PSNR | head SSIM | head HFEN | sharp | grain | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| **MC-GS joint v3** | **39.912** | 0.9019 | **0.01229** | 0.5631 | **35.771** | 0.9468 | **0.0831** | 0.867 | 0.0108 | 189 min |
| MC-GS single v3 | 39.840 | 0.9073 | 0.01250 | 0.5685 | 35.679 | 0.9455 | 0.0851 | 0.864 | 0.0107 | 55 min |
| L1-ESPIRiT | 39.823 | **0.9510** | 0.01254 | 0.5802 | 35.530 | **0.9471** | 0.0867 | 0.899 | 0.0118 | 5 min |
| joint CS-SENSE (JTV) | 39.669 | 0.9468 | 0.01300 | 0.5573 | 35.020 | 0.9461 | 0.1001 | 0.854 | 0.0107 | 5 min |
| MC-GS joint v2 | 39.249 | 0.9047 | 0.01432 | 0.6052 | 34.978 | 0.9373 | 0.0863 | 0.912 | 0.0122 | 40 min |
| CS-SENSE (TV, converged) | 39.534 | 0.9271 | 0.01341 | **0.5549** | 34.966 | 0.9460 | 0.0993 | 0.829 | 0.0101 | 4 min |
| MC-GS single v2 | 38.947 | 0.9084 | 0.01535 | 0.6195 | 34.680 | 0.9325 | 0.0923 | 0.916 | 0.0124 | 40 min |
| CS-SENSE (TV, 35 outer, as run 09-09) | 39.243 | 0.9294 | 0.01434 | 0.5682 | 34.607 | 0.9436 | 0.1071 | 0.817 | 0.0102 | 15 min |
| ZS-SSDU | 38.635 | 0.9412 | 0.01649 | 0.6465 | 34.134 | 0.9365 | 0.1213 | 1.012 | 0.0139 | 22 min |
| MC-GS single (old cold start) | 36.010 | 0.8479 | 0.03018 | 0.7521 | 33.296 | 0.9164 | 0.1162 | 0.935 | 0.0147 |  |
| MC-GS joint (old cold start) | 36.353 | 0.8682 | 0.02789 | 0.7319 | 33.294 | 0.9190 | 0.1220 | 0.874 | 0.0124 |  |
| CG-SENSE | 34.514 | 0.7269 | 0.04259 | 0.8528 | 31.941 | 0.8981 | 0.1705 | 0.903 | 0.0162 | 11 min |
| 3D-INR | 34.761 | 0.8182 | 0.04024 | 1.0278 | 30.838 | 0.8446 | 0.1435 | 1.218 | 0.0247 | 71 min |
| P-LORAKS | 35.512 | 0.8809 | 0.03385 | 0.9173 | 30.623 | 0.8413 | 0.1615 | **1.242** | 0.0284 |  |
| 3D-DIP | 35.127 | 0.8409 | 0.03699 | 0.7922 | 30.106 | 0.8542 | 0.2518 | 0.686 | 0.0078 | 57 min |
| ZF | 29.067 | 0.8309 | 0.14930 | 0.9326 | 23.703 | 0.6699 | 0.6348 | 0.310 | **0.0044** | 9 min |

**T2**:

| method | PSNR | SSIM | NMSE | HFEN | head PSNR | head SSIM | head HFEN | sharp | grain | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| MC-GS single v3 | **40.846** | 0.9286 | **0.00976** | **0.4745** | **36.227** | **0.9538** | 0.0675 | 0.827 | 0.0107 | 51 min |
| **MC-GS joint v3** | 40.766 | 0.9290 | 0.00994 | 0.4812 | 36.112 | 0.9531 | 0.0679 | 0.826 | 0.0108 | 189 min |
| L1-ESPIRiT | 40.442 | 0.9567 | 0.01072 | 0.5146 | 35.818 | 0.9536 | **0.0656** | 0.865 | 0.0119 | 5 min |
| MC-GS single v2 | 40.302 | 0.9290 | 0.01106 | 0.5082 | 35.680 | 0.9480 | 0.0661 | 0.866 | 0.0117 | 38 min |
| MC-GS joint v2 | 40.174 | 0.9314 | 0.01140 | 0.5239 | 35.429 | 0.9474 | 0.0665 | 0.857 | 0.0117 | 40 min |
| CS-SENSE (TV, converged) | 40.110 | 0.9509 | 0.01156 | 0.5064 | 35.168 | 0.9493 | 0.0785 | 0.783 | 0.0101 | 4 min |
| joint CS-SENSE (JTV) | 40.024 | 0.9552 | 0.01180 | 0.5109 | 35.008 | 0.9487 | 0.0818 | 0.798 | 0.0108 | 7 min |
| CS-SENSE (TV, 35 outer, as run 09-09) | 39.229 | 0.9307 | 0.01417 | 0.5440 | 34.219 | 0.9427 | 0.0916 | 0.741 | 0.0097 | 5 min |
| ZS-SSDU | 38.991 | **0.9677** | 0.01497 | 0.5626 | 33.822 | 0.9493 | 0.1168 | 1.128 | 0.0159 | 42 min |
| MC-GS single (old cold start) | 37.743 | 0.9162 | 0.01995 | 0.6361 | 33.497 | 0.9319 | 0.0932 | 0.866 | 0.0138 |  |
| MC-GS joint (old cold start) | 37.833 | 0.9365 | 0.01954 | 0.6478 | 33.121 | 0.9295 | 0.1005 | 0.802 | 0.0116 |  |
| CG-SENSE | 35.919 | 0.7829 | 0.03036 | 0.7270 | 32.509 | 0.9210 | 0.1181 | 0.812 | 0.0145 | 7 min |
| 3D-INR | 35.622 | 0.8584 | 0.03250 | 0.9478 | 31.021 | 0.8606 | 0.1053 | **1.233** | 0.0264 | 121 min |
| 3D-DIP | 35.071 | 0.8763 | 0.03690 | 0.8131 | 29.708 | 0.8635 | 0.1888 | 0.641 | 0.0090 | 57 min |
| ZF | 30.548 | 0.8659 | 0.10456 | 0.9198 | 25.193 | 0.7013 | 0.6406 | 0.277 | **0.0042** | 6 min |

Figures: `figures/t1t2_af12_{T1,T2}_plane{0,1,2}.png`, mirrored to
`/home/congcongliu/figs/t1t2_af12/`. Ten columns: ground truth (with the AF12 sampling pattern
below it) then the nine arms, magnitudes on top and `|recon - GT| / max(GT)` below in jet with
a fixed [0, 0.1] range.

### What the numbers say

**The ranking is not the CE study's ranking, and that is the point.** The CE pre/post work had no
ground truth and could only rank by the k-space residual on acquired points, which rewards
whichever model has the most freedom to fit them; CG-SENSE came first there and GS last. With a
ground truth the ordering is by image quality instead, and CS-SENSE and ZS-SSDU move to the top.
Do not carry either ranking across to the other study.

**Whole-volume SSIM is actively misleading on this data.** CG-SENSE scores 0.727 against ZF's
0.831 while being 5.4 dB better in PSNR. The cause is the CSM recipe: `ecalib -c 0.0` leaves
SoS == 1 in air, so the ground truth's background carries full-amplitude noise and any method that
does not suppress it is punished. Head-only reverses the verdict, 0.898 against 0.670. Quote
head-only numbers.

**The original MC-GS arms sit 1.3 dB behind CS-SENSE inside the head, and the reason turned out
to be the cold-start path, not a missing warm start as such.** The frozen headline came from a
576902-Gaussian dual-warm cloud that cannot be registered onto this grid, so these runs start
from a fresh 600 000-point cloud, and that path had never been exercised at quality: it scatters
the points uniformly over the render box (72 % of them in air, where they never move) and
calibrates the edge-TV threshold on the speckle of the first render. Repairing it, and giving
the cloud 1.2M points, is the "MC-GS joint v2" row: **+1.68 / +2.31 dB head-only over the
original joint arm, first on head PSNR, whole PSNR, NMSE and head HFEN on both contrasts**, head
SSIM second by 0.006 / 0.002. Details in the v2 section below.

**Joint versus single is a wash inside the head.** Joint wins on whole-volume PSNR (+0.343 dB on
T1, +0.090 on T2) but head-only is flat on T1 (33.294 vs 33.296) and *worse* on T2 (−0.376). So on
this pair the shared-geometry gain lives in the background, where the explicit Gaussian
representation simply places no density outside the head, and not in the anatomy. The frozen
work's joint benefit was measured on T1+FLAIR with a warm start; it does not transfer unexamined
to T1+T2 from a cold start.

**3D-DIP and 3D-INR come last, both for diagnosable reasons.** DIP's held-out residual was still
falling at the iteration budget on both contrasts (best iterate = 5000 of 5000 on T1, 4600 of 5000
on T2), exactly as in the CE study, so its numbers are a lower bound rather than a converged
result. INR peaked early (700 of 1500 on T1, 1100 on T2) and then over-fitted: it carries the
highest HFEN of any method and a sharpness ratio of 1.22 and 1.23 against a ground truth of 1.0,
i.e. it is sharper than the truth, which is speckle, not detail.

**ZS-SSDU is the strongest learned arm and the cheapest of the three.** It early-stopped at epoch
23 on T1 and ran 60 on T2, and it beats every MC-GS arm on both contrasts while using no training
data outside this subject.

Gauge notes:

- `psnr / ssim / nmse / hfen` are **whole-volume @max**, the project's headline gauge.
- `head_*` restrict to `|GT| > 0.05 * P99.9(|GT|)`.
- `grain_ratio` (N) uses the frozen "WM" band = strict [P55, P90] of in-brain intensity. On T1
  that band is white matter. **On T2 it is not** — T2 inverts the tissue ordering — so N is not
  comparable between the two contrasts. It remains comparable *across methods within* a contrast.

## MC-GS joint v2 — the cold-start path repaired (autoresearch, 2026-09-09/10)

Found and fixed with a Karpathy-style loop (one change per experiment, 1500-iteration
screening, 4000-iteration promotion) run on this pair: program `autoresearch_t1t2.md`, ledger
`autoresearch_t1t2.tsv`, runner `autoresearch_t1t2.sh`, branch `autoresearch-t1t2` of the
framework repo. Three defects of the fresh-cloud path in `JointTrainer._setup_model`, each
measured on its own at 1500 iterations (head PSNR T1 / T2, shipped cold start 32.55 / 32.35):

| change | what it fixes | head PSNR after | delta |
|---|---|---|---|
| positions drawn inside the zero-filled head support (`gaussian.init_mode=head`) | 72 % of a uniform cloud lands in air with density ~0 and position gradient ~0; those points never move | 34.17 / 34.27 | **+1.62 / +1.92** |
| + 3-NN spacing scales instead of a fixed 1-voxel scale (`init_scale_mode=3nn`) | coverage holes between points | 34.77 / 34.22 | +0.61 / −0.05 (SSIM and HFEN up on both) |
| + edge-TV kappa resolved on the ZF image (`loss.edge_tv_kappa_source=zf`) | the threshold was the 0.85-quantile of the *initial render's* gradients, i.e. of speckle | 34.82 / 34.22 | +0.05 / 0.00 once 3-NN is on (+0.30 / +0.08 on the fixed-scale cloud) |
| + 1.2M initial points (`gaussian.initial_num_points=1200000`) | capacity; the warm cloud had been densified, this one never was | 35.01 / 35.02 | +0.19 / +0.80 |

Promotion at 4000 iterations = the v2 rows above (`results_arc/ar3_P2_full4k*`, 40 min, 32 GB).
Repeat floor at this working point (seed 42 vs 43): 0.003 / 0.005 dB. Things that did **not**
help, each as a single change: intensity-weighted sampling, densification (inert at the fixed
scale; dominated by the initial count otherwise), position_lr x3.3, lambda_geo 0.07, plain
high-band weight 1.0 or 3.0. Capacity is contrast-asymmetric: T1 peaks at 1.2M and falls
beyond it (34.91 / 34.57 / 32.24 at 1.6 / 2.4 / 4.8M) because the extra points fit the acquired
8.5 % better and extrapolate worse (oracle unsampled-k error 0.590 → 0.617 → 0.726), while T2,
protected by its shutter, keeps rising to 2.4M (35.48). The shared geometry follows the binding
contrast.

**Joint versus single under the repaired cold start** (`ar6_single_{T1,T2}_v2`, same recipe with
one contrast in both channels, 4000 iterations): T1 joint 34.978 vs single 34.680 (+0.30 dB
head, +0.30 whole, head SSIM +0.005, head HFEN 0.086 vs 0.092); T2 joint 35.429 vs single
35.680 (−0.25 dB head, −0.13 whole). The shared geometry moves accuracy from the easier
contrast to the harder one: single T1 clears CS-SENSE by only +0.07 dB, the joint model by
+0.37. This supersedes the pre-repair reading above ("joint versus single is a wash inside the
head"), which was measured with 72 % of the cloud frozen in air. With a shared point budget
capped by T1 (T2 alone would take 2.4M points), T2 pays for T1's gain; single T2 v2 is
therefore the best T2 number in the table and joint v2 the best T1.

Read the v2 rows with two caveats. (1) The four changes were **selected on this subject with the
ground truth in the scorer**, unlike the other arms' fixed hyper-parameters; they are structural
(where the cloud starts, how big it is, where the edge threshold comes from), not per-image
tuning, and the same recipe is being checked unchanged on the two T1+FLAIR topics
(`T1FLAIR_AF12_retro`, `T1FLAIR_AF12_v3_jiangyizhou`) — see the framework repo's
`autoresearch_t1t2.md` round log for the outcome. (2) Whole-volume SSIM still favours ZS-SSDU
and CS-SENSE; as explained above that gauge is dominated by the noisy air in the ground truth
and is not the number to quote.

```bash
# v2, exactly (frozen working point baked into the runner; the four --set are the change)
cd $F && SAVE_VOLUMES=1 ./autoresearch_t1t2.sh ar3_P2_full4k 4 4000 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn \
  --set loss.edge_tv_kappa_source=zf --set gaussian.initial_num_points=1200000
env $E $PY $R/scripts/extract_gs_volumes.py && env $E $PY $R/scripts/summarize.py
```

## Layout

```
data/      k-space h5 (coil-first), ESPIRiT maps, the 2-D and 3-D masks
recon/     per-method complex volumes + the GT and the fully sampled RSS references
figures/   published figures (mirrors of /home/congcongliu/figs/t1t2_af12/)
logs/      every probe's JSON, every run's log, summary.tsv
scripts/   everything needed to regenerate, absolute paths pointing into this folder
```

## Classical arms revisited (2026-09-10): convergence, lambda sweeps, two new baselines

Three classical baselines were added or corrected on this topic's data and gauge, after the
"MC-GS joint v2" rows above had been established. Two findings change the reading of the tables.

**1. The CS-SENSE (TV) arm of 09-09 was not converged.** Its 35 outer ADMM iterations were
inherited from the CE study, where no ground truth could check convergence. At 100 and 200 outer
iterations (same lambda) the head PSNR rises by 0.35 to 0.61 dB and then stops moving
(hyw T1 34.607 → 34.955 → 34.966; T2 34.219 → 35.096 → 35.168 at its best lambda; FLAIR
33.252 → 33.873 at 200). The original row is kept, labelled, for traceability; "CS-SENSE (TV,
converged)" is the arm to compare against. Lambda was swept on huangyuwei with the ground truth
in the scorer — the same treatment the v2 recipe received — over {0.5, 0.7, 1.0, 1.4, 2.0}e-3:
the 09-09 value 1e-3 was already optimal on T1, and 0.7e-3 gains +0.07 (T2) / +0.01 (FLAIR).
Those lambdas were transferred unchanged to jiangyizhou.

**2. L1-wavelet ESPIRiT (BART `pics -l1`, 100 FISTA iterations, the BART default) is the
strongest arm on every channel.** Lambda swept on huangyuwei over 0.0005 to 0.02 (per-contrast
optima 0.0015 for T1, 0.001 for T2, 0.005 for FLAIR; both neighbours lower), transferred
unchanged to jiangyizhou. Head PSNR: hyw T1 35.530 / T2 35.818 / FLAIR 34.105, jyz T1 36.334 /
FLAIR 35.720 — above MC-GS joint v2 by 0.4 to 0.8 dB, and above the converged TV arm by
0.15 to 0.65 dB. One caveat belongs next to it: at 300 instead of 100 FISTA iterations the
hyw numbers drop (T1 35.526 → 35.218, T2 35.498 → 35.119 at lambda 0.002), so the arm as run
includes an early-stopping regularisation on top of the wavelet prior; "100 iterations" is part
of its definition, as "4000 iterations" is part of MC-GS's.

**3. Joint CS-SENSE (vectorial TV over both contrasts, 200 outer iterations, lambda 1e-3
selected on hyw over {0.5, 0.7, 1.0, 1.4, 2.0}e-3)** — the classical counterpart of "shared
geometry" — is a wash against the converged single-contrast TV: within ±0.16 dB on every
channel of every topic (hyw T1+T2 35.020 / 35.008; hyw T1+FLAIR 34.851 / 33.929; jyz 36.070 /
35.624). Consistent with the project's earlier finding that a vectorial TV at the same lambda is
just a weaker per-contrast penalty.

**Where MC-GS joint v2 stands after this.** Second or third on head PSNR everywhere: behind
L1-ESPIRiT on all five channels, tied with the converged TV on hyw T1 (34.978 vs 34.966), ahead
of it on hyw T2 (+0.26), behind on the FLAIR channels and on jiangyizhou. It keeps the lowest
head HFEN on hyw T1 (0.0863) and stays +1.4 to +2.3 dB above the cold-start joint arm it
replaced. The claim that survives is the diagnosis and repair of the cold-start path, not
"first among nine".
A per-slice sweep against the converged and new arms (`scripts/find_best_slice_v2.py`, output in
`logs/best_slice_v2.log`) finds v2 first on **none** of the full-brain slices (>= 20000 head
voxels) in any plane of any topic; L1-ESPIRiT leads on the slices where v2 itself peaks as well.
The compact figures (`scripts/fig_slice_v2.py`, `scripts/fig_planes_v2.py`) now carry the
converged TV, joint TV and L1-ESPIRiT columns and show those ranks as they are.

P-LORAKS (LORAKS v2 toolbox, per-readout-slice, S-matrix, R = 2) is the fourth arm of this
revisit; its rank is being selected on a six-slice subset of hyw T1 at the time of writing and
its rows enter `logs/summary.tsv` through `scripts/summarize.py` once the full runs land.
Scripts: `recon_classical.py --suffix` (sweeps), `recon_joint_cssense.py`, `recon_l1espirit.py`,
`ploraks_export.py` / `run_ploraks.m` / `run_ploraks_ranks.m` / `ploraks_import.py`; sweep logs
in `logs/sweeps/`, per-run json in `logs/`.

## MC-GS joint v3 — wavelet sparsity and capacity (autoresearch, 2026-09-10/11)

The classical revisit above left L1-wavelet ESPIRiT ahead of v2 on every channel, i.e. the strongest
classical prior on this data is wavelet sparsity, while MC-GS was regularised with edge-TV only. v3 adds
that prior to the MC-GS objective and, because the prior stops the extra Gaussians from overfitting
the acquired samples, doubles the cloud:

**v3 = v2 (head-support init, 3-NN scales, kappa from ZF) + 2.4M initial points + a complex
wavelet-sparsity term λ 0.05 on top of edge-TV** (`paired_gs/losses/wavelet_l1.py`: per-voxel L1 of the
3-D db2 detail coefficients of the rendered volume, 3 levels, periodic boundary, cycle spinning; one
strength for both contrasts). 4000 iterations, about 32.6 GB; the other eight 2.4M wavelet runs took 51–69 min
on an A800, the v3 run itself logged 189 min (the wavelet term costs +66 % time per iteration).

```bash
cd $F && SAVE_VOLUMES=1 ./autoresearch_t1t2.sh ar12_n24_wav005_full4k 7 4000 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn --set loss.edge_tv_kappa_source=zf \
  --set gaussian.initial_num_points=2400000 --set loss.lambda_wav=0.05
```

| arm (4000 iters) | head PSNR T1 / T2 | head SSIM | head HFEN | whole PSNR | sharp |
|---|---|---|---|---|---|
| **MC-GS joint v3** | 35.771 / 36.112 | 0.9468 / 0.9531 | 0.0831 / 0.0679 | 39.912 / 40.766 | 0.867 / 0.826 |
| MC-GS single v3 (same recipe, one contrast) | 35.679 / 36.227 | 0.9455 / 0.9538 | 0.0851 / 0.0675 | 39.840 / 40.846 | 0.864 / 0.827 |
| MC-GS joint v3, per-contrast wavelet (T1 0.06 / T2 0.03) | 35.856 / 36.079 | 0.9481 / 0.9521 | 0.0832 / 0.0661 | 40.006 / 40.700 | 0.862 / 0.838 |
| L1-ESPIRiT | 35.530 / 35.818 | 0.9471 / 0.9536 | 0.0867 / 0.0656 | 39.823 / 40.442 | 0.899 / 0.865 |
| CS-SENSE (TV, converged) | 34.966 / 35.168 | 0.9460 / 0.9493 | 0.0993 / 0.0785 | 39.534 / 40.110 | 0.829 / 0.783 |
| MC-GS joint v2 | 34.978 / 35.429 | 0.9373 / 0.9474 | 0.0863 / 0.0665 | 39.249 / 40.174 | 0.912 / 0.857 |

- **v3 is first on head PSNR on both contrasts**, +0.240 / +0.293 dB over L1-ESPIRiT and
  +0.79 / +0.68 dB over v2; whole-volume PSNR and NMSE above every competing method on both (on T2 its own single-contrast
  ablation is higher still); head HFEN first on T1 but not on T2 (0.0679 against L1-ESPIRiT 0.0656); head
  SSIM 0.0003 / 0.0005 below L1-ESPIRiT.
- **Per slice** (`scripts/find_best_slice_v2.py`, full-brain slices, against every other arm including
  L1-ESPIRiT and P-LORAKS; the single-contrast ablation is not counted as a competitor): v3 is first on
  42 of 47 transverse, 83 of 88 coronal and 90 of 93 sagittal T1 slices, and on every full-brain T2
  slice in all three planes.
- **Joint versus single under v3**: T1 +0.09 dB for the joint model, T2 -0.12 dB. The single-contrast
  v3 arm already beats L1-ESPIRiT on both contrasts on its own (35.679 / 36.227); the joint
  coupling moves accuracy from T2 to T1, as it did under v2 (+0.30 / −0.25), with a smaller swing.
  **The margin over L1-ESPIRiT comes from the Gaussian representation with the wavelet prior and the
  larger cloud, not from the coupling.**
- **Gate**: the pre-registered no-blur floor on T1 sharpness (0.874, the sharpness of the original
  cold-start joint arm) is **missed by 0.007** (T1 sharpness 0.867); T2 passes. The grain ratio falls
  from v2's 0.998 to 0.972 and the head HFEN from 0.0863 to 0.0831, which reads as noise removal
  rather than lost detail, but the gate as written is not met.
- **Selection**: the recipe (the wavelet strength and the point count) was selected on this subject with
  the ground truth in the scorer — the same treatment the L1-ESPIRiT and TV lambdas received here. It was
  then run unchanged on the two T1+FLAIR topics; see their READMEs (hyw T1+FLAIR +0.24 / +0.25 dB over
  L1-ESPIRiT, held-out jiangyizhou tied on T1 and +0.215 dB on FLAIR).
- The per-contrast variant scores a slightly higher minimum margin on this pair; v3 keeps one shared
  strength because FLAIR, the other topics' second contrast, has no tuned value of its own.
- Figures (compact panels, v3 as the proposed method): `figures/t1t2_af12_T1_best_axis0_150.png`,
  `figures/t1t2_af12_T2_best_axis0_144.png` (v3's best transverse slices), `figures/t1t2_af12_{T1,T2}_best3planes.png`.

## Regenerating

```bash
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
R=/data0/congcong/code/GS/Results_20260908/T1T2_AF12_retro
F=/data0/congcong/code/GS/Paired_ARC_GS_general_framework
E="PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID"

env $E $PY $R/scripts/prep_01_inspect.py
matlab -nodisplay -nosplash -r "run('$R/scripts/prep_02_mask.m')"
bash $R/scripts/prep_03_csm.sh                       # ~85 min for the pair
env $E $PY $R/scripts/prep_04_export.py
env $E $PY $R/scripts/prep_05_csm_to_h5.py T1 T2
env $E $PY $R/scripts/prep_06_align_check.py

env $E $PY $R/scripts/recon_classical.py --contrast T1 --gpu 3
env $E $PY $R/scripts/recon_classical.py --contrast T2 --gpu 5
env $E $PY $R/scripts/check_gauge.py --contrast T1 --method cgsense --gpu 3   # gauge proof
bash $R/scripts/run_learned.sh                       # ZS-SSDU + 3D-DIP + 3D-INR, GPUs 3-7

cd $F && env $E $PY arc/cycle.py --exp t1t2_af12_joint --gpu 0 --iters 4000 \
  --lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
  --base_cfg configs/t1t2_af12_hyw.yaml --save_volumes
#   ... likewise --exp t1t2_af12_single_T1 / _T2 with configs/t1t2_af12_hyw_single{T1,T2}.yaml

env $E $PY $R/scripts/summarize.py
env $E $PY $R/scripts/make_figures.py --contrast T1
env $E $PY $R/scripts/make_figures.py --contrast T2
```

## Gotchas that cost time (do not re-discover)

- **`T2_csm_ksp/` is FLAIR.** Byte-identical to `FLAIR_csm_ksp/`. Do not build a "T1+T2" pair from
  that tree; it is T1+FLAIR twice.
- **The trainer's own prints are block-buffered.** `arc/cycle.py` flushes its own `log()` lines but
  the trainer's `print()` calls are not flushed, so a running job shows nothing after the
  `tr.train()` warnings until it finishes. The job is fine; check GPU memory and utilisation
  instead of the log.
- **Dropping `csm_path` falls back to an ACS estimate that returns "center 0 lines" on this
  config and yields an all-zero CSM**, hence an all-zero zero-filled image, a loss that never
  moves and PSNR ~20 with NMSE exactly 1.0. Seen during a smoke run. Always pass `csm_path`.
- **`ecalib` here has no GPU flag** (neither the v1.0.00 build in `~/src/bart` nor
  `~/bart-0.9.00`). 26 coils x 11.5 M voxels is ~40 min per contrast on CPU. Budget for it.
- The BART CFL memmap is **Fortran-ordered**; `reshape(dims, order="F")` and drop the trailing
  singleton before touching it.
