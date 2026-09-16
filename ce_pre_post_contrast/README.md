# CE pair reconstruction results — UIH 3D T1 GRE pre/post contrast

> New session: read `STATUS.md` first (what is done and verified, what is waiting for a decision, what not to redo),
> then this file for the full protocol and history.

Canonical results directory. **All current and subsequent results for this line of work go here.**
(Superseded: `/data0/congcong/data/3D_data/ce_compare_20260907`, moved here 2026-09-08.)

## Data

One patient, one exam, two scans of the same 3D T1 GRE protocol:

| | scan | acquired |
|---|---|---|
| pre  (平扫) | `t1_3D_gre_fsp_sag_iso_ACS_16887049370010` | 2023-07-07 12:43 |
| post (增强) | `t1_3D_gre_fsp_sag_iso_ACS_16887049370060` | 12:50, `<IsInjection>1` |

42-channel head/neck coil, 256 x 232 x 176, 1 mm isotropic sagittal.
**Prospectively undersampled at AF 5.33** (identical mask for both scans, readout axis fully
sampled). **There is no fully sampled ground truth** — no vendor DICOM or image export exists,
so every metric here is GT-free.

Raw inputs live outside this tree and are unchanged:
`/data0/congcong/code/NV-Generate-CTMR/recon_real/work_ce_t1_168870493700{10,60}/{ksp_full,csm,mask2d}.npy`

## Evaluation protocol

Every method exports the **raw, pre-hard-data-consistency** complex volume. With hard DC the
on-mask residual is identically zero and every error map would be black.

All volumes are put on the measured-data scale by a best-fit complex scale, then scored by the
relative k-space residual:

```
alpha = argmin_a || M .( a . A(x) - y ) ||_2        A(x) = M . F3D . (S . x)
kres  = || M .( alpha . A(x) - y ) || / || M . y ||
```

`kres` is computed on **acquired** points, so it rewards whichever model has the most freedom
to fit them (CG-SENSE has 10.4M free voxels; GS has 600k Gaussians). **It ranks data-fit
tightness, not image quality.** Only 3D-DIP and 3D-INR additionally carry a held-out number
(they were fit on 95% of the acquired readout lines, scored on the withheld 5%).

## Results

```
method                        kres pre   kres post   held-out pre   held-out post
CG-SENSE                       0.18166     0.14640         -              -
CS-SENSE (ADMM-TV)             0.18295     0.14772         -              -
ZS-SSDU                        0.18312     0.14797         -              -
3D-INR (hash coord-MLP)        0.18573     0.15176      0.32226        0.27585
GS single (each scan alone)    0.19243     0.15856         -              -
GS joint (shared geometry)     0.19761     0.16307         -              -
3D-DIP (ConvDecoder)           0.22686     0.20722      0.35891        0.33845
ZF (reference anchor)          0.28449     0.27193         -              -
```

Visual findings (see `figures/`):
- **3D-DIP** — brain plausible but heavy artefact outside the head; the convolutional generator
  is spatially uniform and the background is unconstrained. Had NOT converged at 5000 iters
  (best checkpoint = last iter), so its number is a lower bound.
- **3D-INR** — sharp brain, but salt-and-pepper speckle in the background from hash collisions.
- **GS single / joint** — cleanest background of all: Gaussians place no density outside the head.
  This is a structural property of the explicit representation, not tuning.
- GS joint fits slightly looser than GS single on both scans (+0.005), which is what a shared-
  geometry constraint should do. That single/joint pair is the only apples-to-apples comparison
  in the table (same model class, same 600k point budget, same loss).

## 2026-09-14: Joint SENSE, L1-ESPIRiT, MC-INR added; CS-SENSE re-run converged

Three comparison arms were added and CS-SENSE was replaced. The user decided the protocol on 2026-09-14:
lambda is chosen on held-out data for every tuned arm, CS-SENSE included.

- **Held-out split.** `heldout_split/split_95_5_seed0.npz` freezes the 95/5 split that ce_dip_inr.py
  generates: 7663 PE lines, of which 7280 are fitted and 383 held out. Recomputing the archived 3D-DIP and
  3D-INR held-out kres with this file matches to within 1.1e-8 (`heldout_split/split_check.json`).
  This is not the June NV-Generate-CTMR split, which held out 5% of individual samples. The
  "matching the June contract" line in ce_dip_inr.py's docstring is wrong.
- **Lambda rule, fixed before any run.** Fit on the 95% lines over a factor-2 grid and take the lambda
  with the lowest held-out kres. Held-out kres is defined as in ce_dip_inr.py: alpha is refit on the
  held-out lines. A minimum at either end of the grid is extended by a factor of 2, at most 3 times per
  side. After selection the arm is refit on 100% of the lines, and that refit is the column shown.
  Scripts: `ce_select_lambda.py` and `ce_orchestrate.py`. Grids: `lambda_grids.json`. Curves:
  `lambda_selection.json`. Every selected lambda ended up inside its range.
- **CS-SENSE.** Same ADMM as ce_classical.py (tik 0.01, mu 0.1, 8 warm-started CG steps), now with 200
  outer iterations. The 35 used before were shown to be under-converged on the jyz/hyw topics.
  `ce_tv_sense.py` reproduces the archived 35-iteration run bit for bit: kres 0.18295208 vs 0.18295207,
  volume relative difference 7.7e-7 (`tv_sense/validation/`). Selected lambda: 2e-3 for pre and for post.
- **Joint SENSE.** The same ADMM, but the shrinkage acts on the 6-vector of both scans' gradients
  (vectorial TV). Each scan keeps its own maps, data term and P99.9 |ZF| scale. One lambda serves the
  pair, chosen on the mean held-out kres: 2.83e-3. That is sqrt(2) x the CS-SENSE lambda, as expected
  when two nearly identical images share one vectorial penalty.
- **L1-ESPIRiT.** `bart pics -l1 -r LAM -i 100 -S` on CPU, the jyz/hyw call, run by `ce_l1espirit.py`.
  It uses this topic's coil maps, which are the same maps every column uses (SoS == 1 inside the support).
  They were derived in MATLAB from csm*.mat (algorithm undocumented) and were not recomputed with
  `bart ecalib`. The pattern is passed with -p, and BART's data scaling is frozen per scan with -w, so a
  lambda means the same in the 95% and 100% runs. Selected lambda: 4e-3 for pre (after extension to 8e-3),
  2e-3 for post.
- **MC-INR** (Niessen et al. 2025). The faithful reproduction package `/data0/congcong/code/GS/mcinr_niessen2025`
  is used unchanged: per-slice 2D hash-grid INR along the readout axis emitting both scans jointly,
  weighted k-space MSE, Adam 1e-2, 200 steps, last step, no regulariser (`ce_mcinr.py`, GR_pt1.10 env).
  It has no lambda. The shown column is the 100% fit; a separate 95% fit supplies the held-out number.
  The two scans share one sampling mask, so the paper's complementary-sampling premise is absent here,
  as in the jyz reproduction. The per-slice fits leave faint slice-to-slice stripes in coronal and sagittal views.

```
method        kres pre  kres post   held-out pre  held-out post   mean held-out   lambda
ZF             0.28449    0.27193        -              -              -
CG-SENSE       0.18166    0.14640      0.32946        0.28164        0.30555
CS-SENSE       0.18487    0.14969      0.31344        0.26974        0.29159        2e-3 / 2e-3
Joint SENSE    0.18480    0.14999      0.31367        0.27052        0.29210        2.83e-3 (pair)
L1-ESPIRiT     0.18412    0.14728      0.30797        0.26127        0.28462        4e-3 / 2e-3   (not shown since 09-15)
L1-SPIRiT      0.18479    0.15138      0.31695        0.26954        0.29325        k3 2e-3 / k5 1e-3
ZS-SSDU        0.18312    0.14797      0.30698        0.26254        0.28476
3D-DIP         0.22686    0.20722      0.35891        0.33845        0.34868
3D-INR         0.18573    0.15176      0.32226        0.27585        0.29906
MC-INR         0.18168    0.14620      0.33333        0.28555        0.30944
GS-single      0.19243    0.15856      0.32376        0.28309        0.30343
MC-GS          0.19761    0.16307      0.32558        0.28462        0.30510
(superseded) CS-SENSE, 35 outer, lambda 1e-3:  0.18295 / 0.14772
```
Source: `kres_summary_20260914.json`. Held-out numbers for CG-SENSE, ZS-SSDU, GS-single and MC-GS come from
`heldout4_scores.json` (2026-09-15, see below).

Reading the held-out columns:
- **Ranking by mean held-out kres** (shown arms; L1-ESPIRiT, dropped on 09-15, would sit first at 0.28462):
  ZS-SSDU 0.28476 < CS-SENSE 0.29159 ≈ Joint SENSE 0.29210
  < L1-SPIRiT 0.29325 < 3D-INR 0.29906 < GS-single 0.30343 < MC-GS 0.30510 < CG-SENSE 0.30555 < MC-INR 0.30944 < 3D-DIP 0.34868.
  ZS-SSDU is lowest on pre, L1-ESPIRiT on post.
- **Coupling gives no held-out gain on this pair, in either family.** Joint SENSE ≈ CS-SENSE (+0.0005 mean), and
  MC-GS is slightly above GS-single on both scans (+0.0018 pre, +0.0015 post). The GS columns run the frozen
  09-07 recipe (600k Gaussians), not the later jyz autoresearch recipes.
- **Some held-out numbers are mildly optimistic.** The number was used to choose lambda (CS-SENSE, Joint SENSE,
  L1-ESPIRiT) or the checkpoint (3D-DIP, 3D-INR), always on the same 383 lines. CG-SENSE, ZS-SSDU, GS, MC-GS
  and MC-INR run fixed protocols; ZS-SSDU's early stopping uses its own internal split inside the 95% lines.
- **Which run the numbers come from.** Except for 3D-DIP and 3D-INR, the held-out number comes from a 95%-line
  run, separate from the 100% volume in the figure. The kres columns always belong to the shown volume.
- **Do not quote** "held-out" values inside the 100%-fit metas (`*_full_*_meta.json`, `heldout_is_unseen: false`):
  those lines were in the fit.

### 2026-09-15: held-out runs of CG-SENSE, ZS-SSDU, GS-single and MC-GS

Same frozen split, same scorer (`ce_score_heldout.py` → `heldout4_scores.json`), scheduler plan
`ce_orchestrate.py heldout4`:
- **CG-SENSE and ZS-SSDU:** `ce_classical.py --fit` restricts the acquired mask to the 7280 fit lines before
  anything else, so ZS-SSDU's internal Θ/Λ/Γ splits are drawn inside the fit lines. ZS-SSDU runs with
  `--epochs 100`, as on 09-07; early stop came at epoch 69 (pre) and 60 (post). Outputs in `classical_fit/`.
- **GS-single and MC-GS:** `ce_gs.py --fit` reads `heldout_split/ksp_fit_{pre,post}.h5`, which is ksp*.mat with
  the held-out lines zeroed (`ce_make_fit_kspace.py`; read-back check 0.0, 7280 non-zero lines per readout
  plane). The presampled loader infers its mask from non-zero samples; the log shows acceleration 5.61 =
  5.33 x 7663/7280. Held-out lines therefore enter neither the loss, the ZF normalisation nor the
  initialisation.
- **Pinned trainer.** GS runs on `scripts/_trainer_3dd24aa/`, a `git archive` of the trainer commit the
  09-07 runs imported (see SOURCE.txt), so the 95% and 100% GS runs share the code. Outputs in `gs_fit/`.
  MC-GS took 1 h 53 min on a shared card.

### 2026-09-15: L1-SPIRiT added

The user asked for L1-SPIRiT (Lustig & Pauly 2010), the k-space method, as well as L1-ESPIRiT; the 09-14 request's
"l1-spirit" had been read as L1-ESPIRiT. `ce_spirit.py` reuses the jyz topics' registered arm: every numerical
function between its "verbatim" markers is byte-identical to `T1FLAIR_AF16_v3_jiangyizhou/scripts/recon_spirit.py`
(md5 6c25087d20de; checked with diff). That port was aligned with Lustig's MATLAB toolbox to 1e-12, and it replaces
pocsSPIRiT, which diverges on that data, with a damped proximal-gradient/FISTA step on the same objective. Only the
data wiring is new:
- **Data.** Hybrid space along readout axis 0. Data are normalised by the classical P99.9 |ZF| scale, the completed
  coil k-space is combined with this topic's maps, and the result is scored like every column.
- **Calibration.** The kernel is calibrated from the full-data 24 x 24 ACS in every run. The 95% mask shatters that
  block (its largest centred fully sampled block is 18 x 4), and the coil maps every arm uses were also estimated
  from the full data. 26 of the 576 ACS lines (4.5%) are held-out lines; they inform the kernel but never the
  data-consistency term.
- **Grid.** Kernel 3 and 5 x lambda {0, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3}, 150 iterations, the jyz grid. Kernel 3 gives a
  484 x 378 calibration matrix; kernel 5 gives 400 x 1050, under-determined and carried by Tikhonov 0.01.
- **Selection.** Lambda per kernel on held-out kres, then the kernel on the same number (plan `ce_orchestrate.py spirit`).
  - Pre: kernel 3, lambda 2e-3, 0.31695. Kernel 5 at 2e-3 gives 0.31696, a tie.
  - Post: kernel 5, lambda 1e-3, 0.26954 (kernel 3 best 0.27029).
  - Every minimum is interior. Plain SPIRiT (lambda 0) gives 0.370 / 0.311: the self-consistency prior alone does
    not carry this data, as on jyz.
- **Result.** Mean held-out 0.29325, just behind CS-SENSE / Joint SENSE and ahead of 3D-INR, GS and CG-SENSE.
  kres 0.18479 / 0.15138 (100% refits). FISTA relative change after 150 iterations is 4e-4 to 2.5e-3.
- **Gotcha (CPU only).** In GSTMI (torch 2.7.1), batched single-precision complex `torch.linalg.solve` on CPU dies
  with "Intel oneMKL ERROR: Parameter 6 was incorrect on entry to CLASWP", and no Python traceback is printed. The
  calibration therefore needs a GPU; the CPU `--smoke` mode of ce_spirit.py cannot get past it.
- **Scheduling.** A job on a card another user was computing on ran ~9x slower (304 s vs 34 s per chunk). The
  scheduler now treats a card as free only below 30% utilisation. Its `add()` adopts a still-running job from a
  previous scheduler instance instead of launching it twice.

Figures of record (user, 2026-09-15: "使用 l1-spirit"): `figures/ce_pair_recon_3planes.{png,pdf}` and
`ce_pair_methods_errormap{,_zoom}.png` have 11 columns: ZF, CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU,
3D-DIP, 3D-INR, MC-INR, GS-single, MC-GS. **L1-ESPIRiT is dropped from the comparison**, as in the jyz topics.
Nothing was deleted: its reconstructions, sweep and table row stay. Both figure scripts drop it by default;
`--drop` with no labels renders all 12 columns. The earlier render with L1-ESPIRiT and without L1-SPIRiT is kept
as `*_11col_20260915`; a 12-column render and `*_spirit` duplicates were made and deleted after the decision.
Earlier: `figures/ce_pair_recon_3planes.{png,pdf}` had 11 columns in the order of the table above; the
single-scan and joint GS columns are relabelled GS-single and MC-GS. The 09-08 8-column version is kept
as `figures/ce_pair_recon_3planes_8col_20260908.{png,pdf}`. Since 2026-09-15 the axial panels of both figures are
shown anterior (eyes) up: the array slice v[z] without the former rot90; A-P index 0 is anterior. Sagittal panels
still show anterior on the right. Both figure scripts cache their per-volume magnitudes in `figures/cache/`, so
a re-render takes seconds while the input volumes are unchanged; pass `--recompute` to rebuild.

Slices (user, 2026-09-15: every method, pre and post, on the lesion slices in all three planes): both figures now
cut through the lesion core at axial z=201, coronal y=131, sagittal x=48 (`--slices`, default; see "Slices through
the lesion" below). The error-map figure is drawn once per plane: `ce_pair_methods_errormap{,_coronal,_sagittal}{,_zoom}.png`,
with `layout="compressed"` because tight_layout let the kres labels run into the next row on the wide sagittal panels.
The previous render is kept as `*_l1spirit_z170_20260915`: axial 170, with coronal 162 and sagittal 86 taken from the
strongest smoothed CG-SENSE post-minus-pre voxel, which is not the lesion. In the new coronal and sagittal error maps
MC-INR shows horizontal stripes: it fits one network per readout (S-I) slice, so neighbouring slices are not tied.

Enhancement version (2026-09-16, `ce_figure_paper.py --diff` -> `ce_pair_recon_3planes_enhancement.{png,pdf}`): the same
11 methods and the same three planes, but 9 rows, pre / post / post - pre per plane, the lesion core outlined in orange
and, under each column, the method's post minus pre in the reference regions of `enhance_rois_cssense.npz`. The displayed
difference is smoothed (sigma `--dsmooth` 1.0, the numbers are not) and shares the common intensity scale;
`--dvmax` 0.20 is 0.6 white-matter units. Per method (post minus pre, each scan in its own white-matter units):

```
              vessels  plexus  cortex     WM  lesion core
ZF              +1.81   +0.52   +0.099  +0.007   +0.019
CG-SENSE        +2.80   +1.10   +0.103  +0.002   +0.022
CS-SENSE        +2.84   +1.08   +0.105  +0.003   +0.021
Joint SENSE     +2.81   +1.04   +0.103  +0.003   +0.021
L1-SPIRiT       +2.86   +1.11   +0.101  -0.001   +0.021
ZS-SSDU         +3.02   +1.21   +0.110  +0.003   +0.023
3D-DIP          +2.60   +0.77   +0.109  +0.002   +0.057
3D-INR          +2.88   +1.17   +0.106  +0.004   +0.029
MC-INR          +2.86   +1.13   +0.102  +0.001   +0.022
GS-single       +2.92   +1.16   +0.108  +0.004   +0.021
MC-GS           +2.92   +1.15   +0.106  +0.003   +0.024
```

There is no ground truth, so these compare the methods with each other, not with truth. ZF and 3D-DIP keep much less of
the enhancement of small structures (vessels, choroid plexus), and 3D-DIP also puts +0.057 into a lesion that does not
enhance. MC-GS matches GS-single, so the joint arm does not suppress the difference between the two scans here.

Lesion boundary (asked 2026-09-16: does MC-GS hold the boundary better?). `scripts/ce_edge_profile.py` fixes one
boundary - the outer margin of the T1-dark lesion, thresholded once on CS-SENSE inside a box and anchored on the core
mask (37.7 mL at `--lesion_thr` 0.65) - and measures every arm against it: the intensity profile against signed
distance, the equivalent-rectangle width (step / max slope), the 20-80% width, the shift of the 50% crossing between
pre and post, the mean gradient in the 1.5 mm rim, and the high-frequency std in normal white matter as a noise
stand-in. Result (pre): equivalent width 2.57-2.61 mm for every arm except ZF 3.13 and 3D-DIP 2.71; 20-80% width
1.55-1.69 except ZF 2.72; the 50% crossing moves 0.12-0.19 mm between the two scans for all of them. MC-GS is 2.59 /
1.61 / 0.18, i.e. the same as CS-SENSE (2.60 / 1.62 / 0.18) and GS-single (2.61 / 1.62 / 0.17). **No arm preserves
this boundary better than another**, and the arms that look sharper are the noisy ones: the rim gradient divided by
the white-matter high frequency is 1.02-1.15 for all eleven, so what the gradient measures at this edge is mostly
noise. The lesion margin itself is a 2.6 mm ramp at 1 mm voxels, far coarser than the differences between
reconstructions; answering the question needs ground truth (retrospective undersampling of BMSR / PDGM) and a
boundary metric against it. Figure `ce_pair_lesion_edge_profile.png`, numbers in `lesion_edge_profile.json`.
Three traps in building this measure, all fixed: the core's own boundary sits inside a gradual ramp and gives every
arm the same 8 mm; thresholding without a brain mask takes the whole head (8 M voxels); and `distance_transform_edt`
puts no voxel within 1 mm of the surface, so the central bins are empty unless the interface is offset by 0.5.

Zoomed version for the proposed arm (`ce_figure_paper.py --zoom` -> `ce_pair_zoom_mcgs.png`): one axial slice, three
rows of 72 x 72 mm crops with a locator column - the lesion and a vessel region in post minus pre, and the background
outside the head in the post image. The boxes are picked automatically (lesion from the core mask, vessel box = the
window with the most voxels above 0.3 white-matter units in CS-SENSE and clear of the lesion, background = the window
free of the dilated head that carries the most signal). The numbers under each crop are whole-volume statistics, not
crop statistics: lesion core and vessels are ROI means of post minus pre, background is the mean of the post image
outside the head mask dilated by 6 mm, all in white-matter units. Background: 3D-DIP 0.878, 3D-INR 0.427, MC-INR 0.122,
ZS-SSDU 0.024, CG-SENSE 0.013, ZF 0.010, L1-SPIRiT 0.009, GS-single 0.009, MC-GS 0.008, CS-SENSE / Joint SENSE 0.005.
So among the arms that fit one scan with an untrained representation, MC-GS is the only one that at the same time keeps
the vessel enhancement (+2.92, second only to ZS-SSDU's +3.02), adds no enhancement to the non-enhancing lesion
(+0.024) and leaves a background as clean as the classical arms (0.008 against 0.878 for 3D-DIP).

Scheduling notes: GPU jobs take a card only when no process of this user is on it, with at most two cards
in use. A rule requiring the card to be free on two consecutive 60 s polls never fired, because other
sessions refill a freed card within a minute; a single poll every 15 s works. Kill jobs by PID only:
`pkill -f` also matches the invoking shell and any monitor script.

## Data bundle of the figure of record (2026-09-15)

`export/ce_l1spirit_20260915/` (15.05 GiB, 87 files, real copies, `md5sum -c checksums.md5` 87/87 OK):
- the 100%-line volume of each of the 11 shown arms plus L1-ESPIRiT, per scan;
- the 42-coil k-space, coil maps and mask; the held-out split; tables, selection records and run metas;
- the figures.
Its README gives layout, units (GS volumes need a transpose and the P99.9 scale) and a loader. It is written by
`scripts/export_ce_bundle.py` (`--no_raw` skips the 14 GiB of k-space and maps). Not de-identified: server only.

## Derived-results bundle (2026-09-16)

`export/ce_enhancement_20260916/` (64 files, 329 MiB, `md5sum -c checksums.md5` 64/64 OK, written by
`scripts/export_ce_derived.py`): everything built on top of the reconstructions in this round -
`lesion/` (SynthSeg labels and segmentation, the white-matter-normalised pre and post volumes, the enhancement map,
the candidate tables, the reference-region masks, and the superseded threshold segmentations kept for the record),
`tables/`, `figures/` including `figures/lesion/`, the scripts, and the topic README at export time. The
reconstruction volumes and raw k-space are not copied again; they are in `export/ce_l1spirit_20260915/`.
**Not de-identified, and the NIfTI volumes include the face: server only.** The figures and the numbers (not the
NIfTI, not the masks) are also on GitHub, branch `ce-pre-post-contrast`.

## Slices through the lesion (2026-09-15)

There is no annotation or clinical report for this exam. The automatic search for focal enhancement
(`ce_lesion_candidates.py analyze`, `blobs`) returned only vessels, dural sinuses and choroid plexus. A lesion was
found by eye on the axial montage instead: image-left upper hemisphere, T1-dark and lobulated with a darker core,
seen on both scans. `ce_lesion_candidates.py slices` (CS-SENSE, white-matter units) keeps the largest component of
smoothed pre < 0.32 inside a box set by eye (z 180:224, y 75:155, x 22:82):
- core 6039 mm3, centroid (z 202.1, y 124.9, x 48.4), extent z 190-217, y 105-140, x 33-60; mean pre 0.255,
  post 0.276, so the core does not clearly enhance; SynthSeg label 42.
- slices where the core is largest: axial z=201 (446 mm2; within 10% for z 199-203), coronal y=131 (279 mm2;
  y 130-135; a second lobe peaks at y=122, 216 mm2), sagittal x=48 (384 mm2; x 47-54).
- figure `/home/congcongliu/figs/ce_pair/lesion_candidates/lesion_slices_z201_y131_x48.png`, numbers in
  `lesion_candidates/lesion_cores_cssense.json`. Chain: `prep` (SynthSeg, CPU) -> `analyze` (labels) -> `slices`.
- Both comparison figures use these slices since 2026-09-15; their earlier planes (z 170, y 162, x 86) missed the
  core. The lesion figure shows sagittal anterior left; the comparison figures show it anterior right.
- **The lesion does not enhance** (asked 2026-09-16: a slice subtle on pre but obvious on post). post minus pre in
  WM units: core 0.021, rim 0-3 mm 0.026, rim 3-6 mm 0.009 (the rims here are also cut by the eroded brain mask; the
  `enhance` stage dilates without that cut and reports 0.028 for the same 0-3 mm rim), against 0.003 for white matter and 0.046 / 0.061 /
  0.035 for the mirrored contralateral regions, i.e. the lesion side enhances *less* than the normal side. Its
  contrast against its own 3-6 mm surroundings falls from 0.303 (pre) to 0.291 (post). A whole-brain search for
  foci that look like normal tissue on pre and bright on post (difference of Gaussians on each scan, inside
  parenchyma, away from ventricles) returns only vessels, sinuses and choroid plexus: of 218 blob peaks the
  blob-shaped deep survivors are 1, a vessel at (148, 133, 114). The densest such slice is z=129 (761 mm2), the
  skull base. Evidence figures: `lesion_lower_slices_scan.png` (z 180-198), `post_only_candidate_z148.png`,
  `post_only_is_vessels_z129.png`.
- **What the static post-contrast scan does show** (`ce_lesion_candidates.py enhance` -> `enhance_rois_cssense.json`,
  `enhancement_explained_z201_y131_x48.png`): barrier leakage and blood pool, not flow. Mean post minus pre in WM
  units: normal WM +0.004 (barrier intact, gadolinium stays intravascular and brain blood volume is only 3-5%),
  vessels and dural sinuses +2.858 (blood pool), choroid plexus +1.087 (no barrier there), normal cortex +0.105
  (cortical and pial vessels, partial volume, not leakage), lesion core +0.021 and its 0-3 mm rim +0.028. The figure
  draws the three lesion planes as pre / post / post - pre with the enhancing set (cyan) and the core (orange)
  outlined: the cyan set traces vessels, sulci and scalp and never enters the lesion.
- Not a diagnosis. Patient left/right is not recorded; label 42 reads "right cortex" only under the assumed
  radiological convention.
- Do not use `lesion_mask_cssense.npy`, `lesion_mask_thr0.55/0.6.npy` or `lesion_measure_cssense.json`: those
  threshold segmentations leaked into CSF and across the midline (82-199 mL).

## Layout

> **In git, the `*.npy` reconstruction volumes are not tracked** (1.3 GB, 83.6 MB each; excluded
> by `.gitignore`). Only the README, figures, per-method `*_meta.json`, run logs and configs are
> committed. The volumes live on amax at
> `/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/` and are regenerated by the
> commands under "Regenerating".

```
classical/       ZF, CG-SENSE, CS-SENSE, ZS-SSDU   <method>_<scan>_complex.npy + _meta.json
dipinr/          3D-DIP, 3D-INR                    (+ kres_heldout, best_iter in meta)
gs_joint/        joint GS          gs_joint_{pre,post}_raw_complex.npy
gs_single_pre/   single GS, pre    gs_single_pre_raw_complex.npy
gs_single_post/  single GS, post   gs_single_post_raw_complex.npy
figures/         published figures (mirrors of /home/congcongliu/figs/ce_pair/)
logs/            training logs (logs/ext_20260914/: the 09-14 arms, scheduler log, selection_*.json)
scripts/         everything needed to regenerate
kres_summary.json            the 09-08 eight columns (rewritten by ce_figure.py)
heldout_split/   split_95_5_seed0.npz + split_check.json                        (09-14)
tv_sense/        cssense_o200_* / jsense_o200_* grids (_fit_) and refits (_full_); validation/   (09-14)
l1espirit/       l1espirit_i100_* grids and refits                              (09-14)
mcinr/           mcinr_{full,fit}_{pre,post} + losses_*.npy                     (09-14)
lambda_grids.json, lambda_selection.json, kres_summary_20260914.json            (09-14)
classical_fit/   cgsense_*_fit / zsssdu_*_fit (95% lines)                        (09-15)
gs_fit/          single_pre/ single_post/ joint/ (95% lines, trainer scripts/_trainer_3dd24aa)   (09-15)
heldout_split/ksp_fit_{pre,post}.h5 (7 GB each, deletable: regenerate with ce_make_fit_kspace.py)
scripts/_trainer_3dd24aa/  not in git; recreate with
                  git -C <mcgs clone> archive 3dd24aa paired_gs | tar -x -C scripts/_trainer_3dd24aa
                  (the CUDA voxelizer compiles on first use; on amax cuda_ext/build was copied from the live worktree)
heldout4_scores.json, kres_summary_errormap_11col.json                            (09-15)
```

All volumes are `complex64`, shape `(256, 232, 176)` = (S-I, A-P, L-R), **except** the GS ones
which are `(256, 176, 232)`; map with `np.transpose(v, (0, 2, 1))` — verified exact (100% match
on the sampling mask, no flips).

## Regenerating

```bash
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
R=/data0/congcong/code/GS/Results_20260908
E="PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID"

# GS (~40 min each, run the three in parallel on separate GPUs)
env $E $PY $R/scripts/ce_gs.py --mode joint --gpu 0 --iters 4000 --warmup 500 \
    --points 600000 --out $R/gs_joint
#   --mode single_pre / single_post likewise

bash $R/scripts/run_classical.sh     # ZF/CG/CS + ZS-SSDU, GPUs 3/5/6
bash $R/scripts/run_dip_inr.sh       # 3D-DIP + 3D-INR, GPUs 0-3

env $E $PY $R/scripts/ce_figure.py       --gpu 0               # image + error-map figures (09-08: axial only, --slice 170)
env $E $PY $R/scripts/ce_figure_paper.py --gpu 0               # paper figure, 600 dpi + PDF (11 cols; --gpu -1 = CPU)

# 09-14 arms (Joint SENSE, L1-ESPIRiT, MC-INR, converged CS-SENSE): split, then the scheduler runs everything
env $E $PY -u $R/scripts/ce_split.py                          # freezes + checks the 95/5 split (CPU, ~1 min)
cd $R && env $E setsid nohup $PY -u scripts/ce_orchestrate.py >> logs/ext_20260914/orchestrate.log 2>&1 &
#   restartable; writes logs/ext_20260914/ALL_DONE when finished. GPU ~2 h on two A800/A100, BART ~1 h CPU.
cd $R && env $E setsid nohup $PY -u scripts/ce_orchestrate.py heldout4 >> logs/ext_20260914/orchestrate_heldout4.log 2>&1 &
#   09-15 held-out runs of CG-SENSE, ZS-SSDU, GS-single, MC-GS + ce_score_heldout.py (GPU ~4.5 h, ALL_DONE_heldout4)
env $E $PY $R/scripts/ce_figure.py --gpu -1                       # 11-col image + error-map figures, three planes (CPU ok)
env $E $PY $R/scripts/ce_figure_paper.py --gpu -1                 # both default to --slices 201,131,48 (lesion core)
```

## Gotchas that cost time (do not re-discover)

- **ConvDecoder needs BatchNorm.** Without it lr 1e-3 spikes and lr 1e-2 diverges to 3e10 with
  `|alpha| = 0`. With BN + zero-init output layer the loss starts at exactly 1.0 and descends
  monotonically.
- **Complex autograd:** when splitting the graph to bound memory, `x.backward(gradient=g)` must
  NOT conjugate `g`. Verified numerically against a single-graph backward (conjugating gives a
  max gradient error of 36.5; not conjugating is bit-exact).
- **ADMM-TV:** `div3` as written is already the adjoint `D^H`, so the CG operator is
  `AHA + tik.I + mu . div3(grad3(.))` — negating that term makes it indefinite and CG returns NaN.
- **Do not put long-running scripts in the session scratchpad.** A server reboot on 2026-09-07
  wiped `/tmp` and killed three GS runs mid-training.
- **Patient data is NOT de-identified** — `*_scanparam.xml` in the source acquisition folders
  carries a real name, date of birth and hospital patient ID. Handle before any external sharing.

## Open

Closed 2026-09-15: all ten fitted columns carry a held-out kres on one frozen split. ZF has no model.
Still open:
- Held-out kres measures prediction of unseen acquired samples. It is a GT-free proxy, not image quality,
  and cannot replace a retrospective study with ground truth.
- The GS columns use the 09-07 recipe. Porting the jyz autoresearch recipe (larger point budget, db4 wavelet,
  common edge, phase TV) to this pair has not been tried.
