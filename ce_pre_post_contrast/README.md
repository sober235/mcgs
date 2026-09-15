# CE pair reconstruction results — UIH 3D T1 GRE pre/post contrast

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
L1-ESPIRiT     0.18412    0.14728      0.30797        0.26127        0.28462        4e-3 / 2e-3
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
- **Ranking by mean held-out kres:** L1-ESPIRiT 0.28462 ≈ ZS-SSDU 0.28476 < CS-SENSE 0.29159 ≈ Joint SENSE 0.29210
  < 3D-INR 0.29906 < GS-single 0.30343 < MC-GS 0.30510 < CG-SENSE 0.30555 < MC-INR 0.30944 < 3D-DIP 0.34868.
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

Figure: `figures/ce_pair_recon_3planes.{png,pdf}` now has 11 columns in the order of the table above; the
single-scan and joint GS columns are relabelled GS-single and MC-GS. The 09-08 8-column version is kept
as `figures/ce_pair_recon_3planes_8col_20260908.{png,pdf}`. The coronal and sagittal planes pass through
the voxel with the strongest smoothed positive CG-SENSE post-minus-pre difference. This is an intensity
landmark, not an annotated lesion.

Scheduling notes: GPU jobs take a card only when no process of this user is on it, with at most two cards
in use. A rule requiring the card to be free on two consecutive 60 s polls never fired, because other
sessions refill a freed card within a minute; a single poll every 15 s works. Kill jobs by PID only:
`pkill -f` also matches the invoking shell and any monitor script.

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

env $E $PY $R/scripts/ce_figure.py       --gpu 0 --slice 170   # 8-col image + error-map figure
env $E $PY $R/scripts/ce_figure_paper.py --gpu 0               # paper figure, 600 dpi + PDF (11 cols; --gpu -1 = CPU)

# 09-14 arms (Joint SENSE, L1-ESPIRiT, MC-INR, converged CS-SENSE): split, then the scheduler runs everything
env $E $PY -u $R/scripts/ce_split.py                          # freezes + checks the 95/5 split (CPU, ~1 min)
cd $R && env $E setsid nohup $PY -u scripts/ce_orchestrate.py >> logs/ext_20260914/orchestrate.log 2>&1 &
#   restartable; writes logs/ext_20260914/ALL_DONE when finished. GPU ~2 h on two A800/A100, BART ~1 h CPU.
cd $R && env $E setsid nohup $PY -u scripts/ce_orchestrate.py heldout4 >> logs/ext_20260914/orchestrate_heldout4.log 2>&1 &
#   09-15 held-out runs of CG-SENSE, ZS-SSDU, GS-single, MC-GS + ce_score_heldout.py (GPU ~4.5 h, ALL_DONE_heldout4)
env $E $PY $R/scripts/ce_figure.py --gpu -1 --slice 170           # 11-col image + error-map figure (CPU ok)
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
