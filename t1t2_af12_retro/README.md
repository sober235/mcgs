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
| CS-SENSE | **39.243** | 0.9294 | 0.01434 | 0.5682 | **34.607** | **0.9436** | 0.1071 | 0.817 | 0.01016 | 15 min |
| ZS-SSDU | 38.635 | 0.9412 | 0.01649 | 0.6465 | 34.134 | 0.9365 | 0.1213 | 1.012 | 0.01387 | 22 min |
| MC-GS single | 36.010 | 0.8479 | 0.03018 | 0.7521 | 33.296 | 0.9164 | **0.1162** | 0.935 | 0.01473 | 127 min |
| MC-GS joint | 36.353 | 0.8682 | 0.02789 | 0.7319 | 33.294 | 0.9190 | 0.1220 | 0.874 | 0.01243 | 99 min |
| CG-SENSE | 34.514 | 0.7269 | 0.04259 | 0.8528 | 31.941 | 0.8981 | 0.1705 | 0.903 | 0.01623 | 11 min |
| 3D-INR | 34.761 | 0.8182 | 0.04024 | 1.0278 | 30.838 | 0.8446 | 0.1435 | 1.218 | 0.02465 | 71 min |
| 3D-DIP | 35.127 | 0.8409 | 0.03699 | 0.7922 | 30.106 | 0.8542 | 0.2518 | 0.686 | 0.00777 | 57 min |
| ZF | 29.067 | 0.8309 | 0.14930 | 0.9326 | 23.703 | 0.6699 | 0.6348 | 0.310 | 0.00443 | 9 min |

**T2**:

| method | PSNR | SSIM | NMSE | HFEN | head PSNR | head SSIM | head HFEN | sharp | grain | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| CS-SENSE | **39.229** | 0.9307 | 0.01417 | 0.5440 | **34.219** | 0.9427 | 0.0916 | 0.741 | 0.00967 | 5 min |
| ZS-SSDU | 38.991 | **0.9677** | 0.01497 | 0.5626 | 33.822 | **0.9493** | 0.1168 | 1.128 | 0.01589 | 42 min |
| MC-GS single | 37.743 | 0.9162 | 0.01995 | 0.6361 | 33.497 | 0.9319 | 0.0932 | 0.866 | 0.01382 | 100 min |
| MC-GS joint | 37.833 | 0.9365 | 0.01954 | 0.6478 | 33.121 | 0.9295 | 0.1005 | 0.802 | 0.01161 | 99 min |
| CG-SENSE | 35.919 | 0.7829 | 0.03036 | 0.7270 | 32.509 | 0.9210 | 0.1181 | 0.812 | 0.01449 | 7 min |
| 3D-INR | 35.622 | 0.8584 | 0.03250 | 0.9478 | 31.021 | 0.8606 | 0.1053 | 1.233 | 0.02639 | 121 min |
| 3D-DIP | 35.071 | 0.8763 | 0.03690 | 0.8131 | 29.708 | 0.8635 | 0.1888 | 0.641 | 0.00896 | 57 min |
| ZF | 30.548 | 0.8659 | 0.10456 | 0.9198 | 25.193 | 0.7013 | 0.6406 | 0.277 | 0.00421 | 6 min |

Figures: `figures/t1t2_af12_{T1,T2}_plane{0,1,2}.png`, mirrored to
`/home/congcongliu/figs/t1t2_af12/`. Nine columns: ground truth (with the AF12 sampling pattern
below it) then the eight methods, magnitudes on top and `|recon - GT| / max(GT)` below in jet with
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

**MC-GS sits about 3 dB behind CS-SENSE here, and the likely reason is the missing warm start.**
The frozen headline (T1 38.4 dB on the old bundle) came from a 576902-Gaussian dual-warm cloud;
that cloud belongs to a different FOV and cannot be registered onto this grid, so these runs start
from a fresh 600 000-point cloud. That is the single deliberate deviation from the frozen recipe
and it is not a small one. Building a warm start on this grid is the obvious next experiment.

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

## Layout

```
data/      k-space h5 (coil-first), ESPIRiT maps, the 2-D and 3-D masks
recon/     per-method complex volumes + the GT and the fully sampled RSS references
figures/   published figures (mirrors of /home/congcongliu/figs/t1t2_af12/)
logs/      every probe's JSON, every run's log, summary.tsv
scripts/   everything needed to regenerate, absolute paths pointing into this folder
```

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
