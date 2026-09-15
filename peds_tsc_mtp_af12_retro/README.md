# 16-year-old TSC exam, 3D MTP Cartesian — stage 2: retrospective AF 12, ten-arm comparison

Opened 2026-09-13. Stage 1 (the reference k-space this study undersamples) is
`../Peds_TSC_MTP_3D/`. Code: worktree `/data0/congcong/code/GS/mcgs_peds_wt`, branch `16yo-mtp`
(design `docs/superpowers/specs/2026-09-12-16yo-mtp-stage2-af12-design.md`, plan
`docs/superpowers/plans/2026-09-12-16yo-mtp-stage2-af12.md`, package `uih_mtp/`; mirrored here as
`peds_tsc_mtp_af12_retro/`). Figures served at `http://localhost:8765/peds_tsc_af12/`.
**The DICOM and raw headers carry the patient's name: it appears nowhere in this folder.**

## Question

Does MC-GS (joint reconstruction of two contrasts under a shared geometry) lead the method table on
a pediatric 0.7 mm multi-echo GRE exam, as it does on the adult 3D T1 + FLAIR pairs? The recipe is
transferred unchanged; nothing was tuned on this subject.

**Answer: no, and joint is not better than single here.** MC-GS is third on T1w, 1.0 dB behind
L1-ESPIRiT, and second on T2s, 0.3 dB ahead of it. Joint minus single is negative or level on every
metric of both contrasts. The classical joint arm (Joint SENSE, vectorial TV) behaves the same way,
which points at the data rather than at either implementation. See **Findings**. Since 2026-09-15 the L1 arm
counted in the tables is L1-SPIRiT (user decision; L1-ESPIRiT's rows are kept and printed as not counted) — see
**The L1-SPIRiT arm**; the stage-2 statements above stand as made, against L1-ESPIRiT.

## Data

One exam (UIH uMR 790, 3 T, 2022-01-24, age 16, F, head), 3D multi-echo spoiled GRE "MTP",
0.7 x 0.7 x 1.35 mm, 32 channels. Two contrasts, the pair that differs most of the ten acquired
(brain-mask Pearson r = 0.57):

| topic name | acquisition | weighting |
|---|---|---|
| `T1w` | FA 16°, TE 3.25 ms | T1-weighted |
| `T2s` | FA 4°, TE 27.41 ms | PD / T2*-weighted |

Inputs are symlinks into the reference dataset `/data1/data1_congcong/data/3D/Peds_TSC_age16_MTP_GT/`:
`ksp_<c>.h5` (dataset `kspace`, (32, 320, 304, 104) complex64, axes coil / RO / PE / SPE, DC at
(160, 152, 52)), `csm_espirit_<c>.h5` (dataset `map`, unit SoS), `gt_sense_<c>.npy`, `masks_<c>.npz`.

**The reference is not a fully sampled acquisition.** It is the GRAPPA-completed prospective 2x
k-space of stage 1 (ACS hold-out error 0.096 / 0.188 against a noise floor of 0.067 / 0.144), with
no partial-Fourier synthesis. Every number here is relative to that reference, and absolute values
are not comparable with the adult topics.

**Anatomy of the axes** (adjudicated 2026-09-13 against the vendor DICOM, not assumed): the volume is
(RO 320, PE 304, SPE 104); SPE is the partition axis the vendor exports as its slices, so **axis 2 is
transverse**, a fixed-RO slice (axis 0) is coronal and a fixed-PE slice (axis 1) is sagittal.
`gt[:, 2:-2, 52]` equals vendor DICOM slice 39 with r = 0.899. The scripts copied from the AF16 topic
had the opposite convention and were corrected; figures drawn before that correction are wrong.

## Undersampling

One vd-Poisson mask for both contrasts, drawn with the house generator
(`vdPoisMex(304, 104, 304, 104, a, a, 24, 0, 1)`, isotropic `a` swept 2.6 .. 3.6, 105 draws) and then
restricted to the acquired support: the draw whose in-support count first reaches the budget is taken
and trimmed from the outside in to exactly the budget (`scripts/prep_mask_draws.m`,
`scripts/prep_mask.py`, `uih_mtp/retro_mask.py`).

| item | value |
|---|---|
| AF convention (user, 2026-09-12) | sampled points / whole PE matrix, ACS counted |
| budget | 2635 of 31616 (304 x 104) = **AF 12.00** |
| against the acquired support | 19914 positions → **7.56x** |
| chosen draw / trim | `draw_052.mat` (a = 3.1), 2664 in support, 29 trimmed (1.1 %) |
| calibration block | 366 of the 576 positions of the 24 x 24 square |
| known zeros Z | 11702 positions: partial Fourier (PE 227..303) and the elliptical corners |
| forward-operator support Ω ∪ Z | 14337 positions, 45.3 % of the plane |

The mask may sample GRAPPA-interpolated positions: the completed k-space is the reference as a whole
(user decision). `calib` in this generator biases the density near DC but does **not** fully sample a
24 x 24 block — the adult topics' own masks sample 472 and 482 of their 576, full only to 16 x 16 — and
nothing here self-calibrates from the mask, so the count is recorded, not enforced.

## Known zeros for every arm (user decision 2026-09-12)

Z is handed to every method as measured zeros; the forward operator is Ω ∪ Z and the data on Z are
zero. Self-supervised splits are drawn from Ω alone, so no method trains or validates on a trivially
zero target.

| arm family | forward / data consistency | split |
|---|---|---|
| ZF, CG-SENSE, CS-SENSE, Joint SENSE | `data/mask_af12_union_3d.h5` | – |
| L1-ESPIRiT | the same k-space **plus** `bart pics -p data/pattern_union` | – |
| ZS-SSDU | union in the DC block | Θ / Λ / Γ from Ω |
| 3D-DIP, 3D-INR | union in the forward model | 95 / 5 from Ω |
| MC-INR | union as the per-slice sampling plane, so Z enters as measured zeros | – |
| GS-single, MC-GS | Ω as the mask, `known_zero_support: data/support_2d.h5` | – |

Without the explicit `-p` pattern BART infers the pattern from nonzero entries and reads Z as
unsampled; `tests/test_uih_mtp_bart_pattern.py` proves the flag is honoured. The GS arms needed a
framework port: `known_zero_support` was added after the commit this branch was cut from, and without
it the trainer's estimator found nothing (`frac_z_used = 0.0000`) because the mask never lands outside
the support. After the port, both channels report `frac_z_used = 0.3701` = 11702 / 31616 with
`nonzero_in_z = 0` and `kz_energy = 0`.

## The MC-INR arm (added 2026-09-14)

The multi-contrast INR of Niessen et al. 2025 (MICCAI; reproduction repo
`/data0/congcong/code/GS/mcinr_peds_wt`, branch `peds-mtp`), the same reproduction that ran on the
adult T1+FLAIR topics, pointed at this line's data: one hash-grid network per readout position
jointly outputs both contrasts, 200 epochs a slice, the paper's hyper-parameters unchanged, each
contrast keeping its own ESPIRiT maps. It sees this topic's union mask, so the measured zeros reach
it the way they reach every other arm. Nothing was tuned for this exam.

Checks before the run: the repo's single-precision hybrid path (readout inverse FFT, then per-slice
SENSE synthesis) reproduces this topic's ground truth to a maximum relative error of 2.0e-7 (T1w)
and 2.5e-7 (T2s), and the peak of the acquired samples equals the peak of the fully sampled data, so
the paper's normalisation and this topic's coincide (`mcinr/logs/check_real_data.json`). The run is
320 slices in 945 s on a free A800; it is scored by the reproduction through **this topic's own**
`t1t2_common.Pair`, after that gauge is re-checked against the recorded CG-SENSE row (`GAUGE OK`,
`mcinr/logs/score_af12.json`), and registered by `scripts/register_mcinr_arm.py`.

It lands eighth of ten on both contrasts — above 3D-DIP and 3D-INR, below CG-SENSE — which is where
it landed on the adult topics too (eighth of eleven there).

## The L1-SPIRiT arm (added 2026-09-15; the L1 arm counted since then)

User decision 2026-09-15 16:05, the same day the jyz T1/FLAIR topics made the same swap: L1-SPIRiT replaces
L1-ESPIRiT as the L1 arm of the shown tables and figures, and GS-single stays a comparison method. L1-ESPIRiT's
rows stay in `logs/rescore{,_full}.json` and `logs/summary.json` and are printed under each table as "not
counted", so the stage-2 and stage-3 text that quotes them stays checkable; the tables as they were before the
swap are in `logs/archive_pre_spirit_20260915/`.

The arm (`scripts/recon_spirit.py`, built and swept by the stage-4 session) is the jyz AF16 topic's script —
Lustig's SPIRiT toolbox transcribed function by function and aligned there to the MATLAB original to 1e-12 in
double precision — changed only in the contrast names and in where the ACS block is measured (on the sampled
set Ω, never on Ω ∪ Z). Solver: a damped proximal-gradient step with FISTA momentum on the L1-SPIRiT objective
(pocsSPIRiT diverges on this data, as the original does), 150 iterations, db2 wavelet with 4 levels, Tikhonov
0.01 in the calibration; the completed multi-coil k-space is combined with the topic's ESPIRiT maps, so the
result lands in the shared gauge. Data consistency uses Ω ∪ Z like every other arm.

Calibration is the one thing this mask changes. Ω's ACS block is 24 x 8 in (ky, kz) here (jyz: 24 x 16), which
under-determines a 5 x 5 x 32-coil kernel (80 patches for 800 unknowns), so the sweep ran both calibrations:
`acquired` (that block) and `full24` (the centre 24 x 24 of the full k-space — the block `bart ecalib -r 24`
took the topic's ESPIRiT maps from, on which every other arm and the ground truth itself already rest). Grids:
ksize {3, 5} x λ {0, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3} for `acquired` (T1w also 2.5e-4), ksize {3, 5} x λ {0,
2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3} for `full24`, both contrasts, 50 points (`scripts/spirit_sweep*.sh`,
`logs/spirit_sweep.log`, `logs/spirit_<c>_*.json`). `summarize.py` keeps the best head PSNR per contrast over
all of them as one family; on both contrasts that is `full24`, with the optimum interior to its λ grid:

| contrast | kept point | full volume, own scorer: head PSNR / SSIM / HFEN | best `acquired` point |
|---|---|---|---|
| T1w | full24, ks 5, λ 2.5e-4 | 38.384 / 0.9629 / 0.0689 | ks 3, λ 5e-4: 38.114 / 0.9642 / 0.0780 |
| T2s | full24, ks 3, λ 5e-4 | 32.144 / 0.8642 / 0.2302 | ks 3, λ 1e-3: 32.095 / 0.8682 / 0.2384 |

Registered with `bash scripts/rescore_spirit.sh` (`rescore.py --only spirit` on both gauges, merged into the
existing jsons; the full-volume rows reproduce the sweep's own JSON to every printed digit). Primary region:
**T1w 38.310 / 0.9607 / 0.0679, T2s 32.017 / 0.8630 / 0.2264**.

Where it lands. On T1w it is the strongest arm that is not a GS one on head PSNR (fifth of the fourteen counted
rows; 0.52 dB below L1-ESPIRiT, which is no longer counted) — and **it takes the T1w head HFEN column from
GS-single: 0.0679 against GS-single's 0.0764 and MC-GS's 0.0768**, at a sharpness of 0.982: the lightly
regularised k-space solution keeps the grain both GS arms smooth away, noise included. On T2s it is eighth of
the twelve counted rows (below CS-SENSE, Joint SENSE and ZS-SSDU, 1.84 dB below L1-ESPIRiT) and changes no bar.
For the six-cell target the swap therefore moves one bar, against MC-GS: T1w head HFEN is now 0.0089 short
instead of 0.0004.

Figures of record with this arm in the L1 column: `figures/*_headerr_p3f_spirit.png` and
`*_af16style_spirit.{png,pdf}` (`scripts/figs_spirit.sh`; the figure scripts take `--l1 espirit` to redraw the
older versions). Not yet swapped: the per-slice analysis scripts `slice_quality.py`, `find_mcgs_first_slice.py`
and `best_slice_mcgs.py` still list L1-ESPIRiT — swap it when they are next used. The stage-4 lesion tools were
switched by their own session (commit 43bfe56).

## Stage 3 — tuned on this subject (2026-09-14/15, closed)

Stage 2 transferred the GS working point unchanged while L1-ESPIRiT's λ was swept here over 14 values
and CS-SENSE's over 11. Stage 3 removed that asymmetry: the GS arms were tuned on this subject, the
single arm alongside the joint one at every setting, and the table keeps the transferred rows beside
the tuned ones. Design `docs/superpowers/specs/2026-09-14-peds-af12-mcgs-autoresearch-design.md`,
round log `autoresearch_peds.md`, record `docs/superpowers/records/2026-09-14-16yo-mtp-stage3.md`
(all in the worktree). Target: MC-GS first in all six head cells against every arm, GS-single
included. **It was not met, and what the campaign found instead is worth more.**

Final working point: wavelet T1w 0.06 / T2s 0.10, edge-TV geometry **T1w 0.03 / T2s 0.10**, 9.6M
points, 4000 iterations, `edge_tv_union_guide` on with `edge_tv_union_kappa_scale` 0.5. Two of those
are new code: the per-contrast geometry weight (`lambda_geo_plain` / `lambda_geo_contrast`) and the
union guide, which bounds each T2s edge-TV weight by the plain render's.

1. **The joint arm is the best T2s reconstruction in this table.** MC-GS (tuned, 9.6M + union guide)
   leads all three T2s head cells: 34.227 / 0.9056 / 0.1836 against GS-single's 34.175 / 0.9052 /
   0.1855. The union guide is what does it, and it is the one mechanism found here that a
   single-contrast arm cannot have — it bounds one contrast's smoothing by the other contrast's
   edges, and there is no other contrast to borrow from when reconstructing alone. It costs T1w
   nothing.
2. **On T1w the single arm still wins, by 0.057 dB.** GS-single (tuned, 9.6M) reaches 38.993 / 0.9687
   / 0.0764 against the joint arm's 38.936 / 0.9685 / 0.0768. Both GS arms pass L1-ESPIRiT on head
   PSNR and SSIM; L1-ESPIRiT keeps head HFEN (0.0756).
3. **Most of stage 2's joint deficit was capacity starvation, not the prior.** At matched settings the
   joint-minus-single difference on T1w head PSNR is −0.332 at 3.6M points, −0.198 at 6.0M and
   −0.057 at 9.6M: the single saturates after 6.0M (38.848 → 38.989 → 38.993) while the joint keeps
   climbing (38.516 → 38.791 → 38.936). One cloud carrying two contrasts needs more points than one
   carrying one, and every comparison on this line before 2026-09-15 ran at 3.6M. **Stage 2's "joint
   is not better than single here" therefore holds at 3.6M points; at saturation the residual is
   0.06 dB, not 0.33.**
4. **The two contrasts want opposite things from every knob, and capacity is the only one that cannot
   be split**, because the joint arm has a single cloud. T1w is representation-limited (it gains
   +0.275 dB from 3.6M → 6.0M); T2s is SNR-limited (+0.032, and by 19.2M its SSIM and HFEN fall below
   their bars as the extra Gaussians fit noise). The union guide is what holds T2s up once capacity
   is high.

Caveats: one subject, tuned on itself, no held-out exam exists on this line; the tuned GS rows are
comparable with the classical arms, which were swept on the same subject, but not with ZS-SSDU,
3D-DIP, 3D-INR or MC-INR, which were not tuned. `edge_tv_union_guide` and
`edge_tv_union_kappa_scale` come from the parallel stage-4 session on this branch (commit a22073f);
its `p4_union` and `p4_u_k05` runs were re-scored through this topic's own `score_candidate` before
being built on.

## Evaluation

The frozen gauge (`diag_artefact/headonly.py`, unmodified): head mask = |GT| > 0.05 · P99.9(|GT|),
head PSNR at the volume peak of |GT|, head SSIM as the mean of the 7³ uniform-window SSIM map over the
head, head HFEN as the LoG(σ 1.5) residual ratio, whole-volume PSNR / NMSE, sharpness.

- **Primary table**: the same maps averaged over the head **inside the vendor's central 80 partitions**
  (SPE 12..92). The outer 12 partitions at each end are the slab-oversampling margin, where the
  excitation profile has decayed to 6–10 % and the vendor exports nothing.
- **Secondary table**: the full 104 partitions, taken from each arm's own JSON. `scripts/rescore.py --full`
  reproduces it — 100 values across 20 arm/contrast pairs agree to 7e-6 — which is what licenses the
  primary table's numbers.
- `scripts/check_gauge.py` prints `GAUGE OK`: the shared module reproduces `recon_classical.py`'s nine
  metrics to the digit (differences 0.00e+00). `recon_classical.py` deliberately keeps its own
  transcription of the gauge so that check is not circular.
- **grain_ratio on T2s is not usable**: the frozen white-matter mask (strict P55–P90 band, σ 2 erosion)
  keeps 220039 voxels on T1w but only 664 on T2s, too few to read. The gauge was not changed.

Every λ was swept on this subject against the reference, the best kept per contrast, and every scan
ends with its optimum strictly inside its grid (the edge rule of the plan; three scans had to be
extended): CS-SENSE 11 values, L1-ESPIRiT 14, Joint SENSE 5.

<!-- TABLES:BEGIN -->
### Primary table — head inside partitions 12..92

Retrospective AF 12.00 of the 31616-point phase-encode matrix (7.56x of the 19914 acquired positions), one mask for both contrasts; the partial-Fourier block and the elliptical corners (11702 positions) are measured zeros for every method alike. Best of each column in bold.

**T1w** — T1-weighted (FA 16, TE 3.25 ms). head metrics inside the vendor's central 80 partitions.

| method | head PSNR | head SSIM | head HFEN | PSNR | NMSE | sharp |
|---|---|---|---|---|---|---|
| GS-single (tuned, 9.6M) | **38.993** | **0.9687** | 0.0764 | **41.803** | **0.00327** | 0.880 |
| MC-GS (tuned, 9.6M + union guide) | 38.936 | 0.9685 | 0.0768 | 41.739 | 0.00332 | 0.875 |
| GS-single (tuned) | 38.848 | 0.9681 | 0.0761 | 41.721 | 0.00334 | 0.885 |
| MC-GS (tuned) | 38.516 | 0.9668 | 0.0774 | 41.404 | 0.00359 | 0.876 |
| L1-SPIRiT (full-centre calibration, ks 5, λ 0.00025) | 38.310 | 0.9607 | **0.0679** | 41.225 | 0.00374 | 0.982 |
| GS-single | 38.086 | 0.9629 | 0.0910 | 41.089 | 0.00386 | 0.818 |
| MC-GS | 37.855 | 0.9618 | 0.0922 | 40.873 | 0.00406 | 0.813 |
| CS-SENSE (TV, converged) (λ 0.00035) | 37.801 | 0.9611 | 0.1057 | 40.711 | 0.00421 | 0.891 |
| Joint SENSE (JTV) (λ 0.0007) | 36.898 | 0.9546 | 0.1139 | 39.888 | 0.00509 | 0.903 |
| ZS-SSDU | 36.721 | 0.9609 | 0.1240 | 39.782 | 0.00521 | 0.994 |
| CG-SENSE | 34.307 | 0.9339 | 0.1505 | 36.771 | 0.01043 | 0.871 |
| MC-INR (Niessen 2025) | 33.948 | 0.9027 | 0.1576 | 35.905 | 0.01273 | 1.034 |
| 3D-INR | 30.793 | 0.8453 | 0.1237 | 33.850 | 0.02043 | **1.393** |
| 3D-DIP | 30.301 | 0.8678 | 0.2999 | 33.574 | 0.02178 | 0.667 |

Not counted — L1-ESPIRiT (λ 0.001), replaced by L1-SPIRiT (user 2026-09-15); its row stays in the json: head PSNR 38.830, head SSIM 0.9674, head HFEN 0.0756, PSNR 41.503, NMSE 0.00351, sharp 0.924.

**T2s** — T2*-weighted (FA 4, TE 27.41 ms). head metrics inside the vendor's central 80 partitions.

| method | head PSNR | head SSIM | head HFEN | PSNR | NMSE | sharp |
|---|---|---|---|---|---|---|
| MC-GS (tuned, 9.6M + union guide) | **34.227** | **0.9056** | **0.1836** | 36.893 | 0.01044 | 0.782 |
| GS-single | 34.175 | 0.9052 | 0.1855 | 36.991 | 0.01021 | 0.774 |
| MC-GS (tuned) | 34.163 | 0.9047 | 0.1857 | **36.999** | **0.01019** | 0.760 |
| MC-GS | 34.160 | 0.9047 | 0.1858 | 36.995 | 0.01020 | 0.759 |
| ZS-SSDU | 32.846 | 0.8960 | 0.2376 | 36.056 | 0.01266 | 0.896 |
| Joint SENSE (JTV) (λ 0.0014) | 32.821 | 0.8911 | 0.2424 | 35.949 | 0.01298 | 0.794 |
| CS-SENSE (TV, converged) (λ 0.001) | 32.763 | 0.8898 | 0.2438 | 35.904 | 0.01311 | 0.830 |
| L1-SPIRiT (full-centre calibration, ks 3, λ 0.0005) | 32.017 | 0.8630 | 0.2264 | 34.971 | 0.01625 | 0.951 |
| CG-SENSE | 30.598 | 0.8424 | 0.3179 | 32.877 | 0.02632 | 0.861 |
| MC-INR (Niessen 2025) | 29.710 | 0.7981 | 0.3479 | 31.604 | 0.03529 | 1.017 |
| 3D-DIP | 28.250 | 0.6997 | 0.4890 | 31.618 | 0.03518 | 0.465 |
| 3D-INR | 28.012 | 0.6958 | 0.3080 | 29.855 | 0.05279 | **1.417** |

Not counted — L1-ESPIRiT (λ 0.003), replaced by L1-SPIRiT (user 2026-09-15); its row stays in the json: head PSNR 33.860, head SSIM 0.9050, head HFEN 0.1882, PSNR 36.745, NMSE 0.01080, sharp 0.857.

### Secondary table — full volume, each arm's own scorer

The same gauge over all 104 partitions, taken from each arm's own JSON; the cross-check that the rescorer did not move anything (`scripts/rescore.py --full`).

**T1w** — T1-weighted (FA 16, TE 3.25 ms). full volume.

| method | head PSNR | head SSIM | head HFEN | PSNR | NMSE |
|---|---|---|---|---|---|
| GS-single (tuned, 9.6M) | **39.087** | **0.9704** | 0.0767 | **42.217** | **0.00358** |
| MC-GS (tuned, 9.6M + union guide) | 39.032 | 0.9703 | 0.0770 | 42.155 | 0.00363 |
| GS-single (tuned) | 38.928 | 0.9699 | 0.0764 | 42.132 | 0.00365 |
| MC-GS (tuned) | 38.602 | 0.9686 | 0.0777 | 41.825 | 0.00392 |
| L1-SPIRiT (full-centre calibration, ks 5, λ 0.00025) | 38.384 | 0.9629 | **0.0689** | 41.629 | 0.00410 |
| GS-single | 38.141 | 0.9647 | 0.0912 | 41.500 | 0.00422 |
| MC-GS | 37.915 | 0.9636 | 0.0924 | 41.293 | 0.00443 |
| CS-SENSE (TV, converged) (λ 0.00035) | 37.881 | 0.9631 | 0.1052 | 41.093 | 0.00464 |
| Joint SENSE (JTV) (λ 0.0007) | 36.982 | 0.9569 | 0.1138 | 40.292 | 0.00558 |
| ZS-SSDU | 36.734 | 0.9623 | 0.1264 | 40.160 | 0.00575 |
| CG-SENSE | 34.464 | 0.9376 | 0.1504 | 37.130 | 0.01156 |
| MC-INR (Niessen 2025) | 34.174 | 0.9102 | 0.1555 | 36.249 | 0.01416 |
| 3D-INR | 30.859 | 0.8521 | 0.1247 | 34.264 | 0.02236 |
| 3D-DIP | 30.294 | 0.8677 | 0.3047 | 33.983 | 0.02385 |

Not counted — L1-ESPIRiT (λ 0.001), replaced by L1-SPIRiT (user 2026-09-15); its row stays in the json: head PSNR 38.924, head SSIM 0.9692, head HFEN 0.0760, PSNR 41.851, NMSE 0.00390.

**T2s** — T2*-weighted (FA 4, TE 27.41 ms). full volume.

| method | head PSNR | head SSIM | head HFEN | PSNR | NMSE |
|---|---|---|---|---|---|
| MC-GS (tuned, 9.6M + union guide) | **34.375** | 0.9063 | **0.1874** | 37.431 | 0.01173 |
| GS-single | 34.320 | **0.9065** | 0.1891 | 37.575 | 0.01135 |
| MC-GS (tuned) | 34.315 | 0.9061 | 0.1893 | **37.588** | **0.01131** |
| MC-GS | 34.312 | 0.9060 | 0.1894 | 37.583 | 0.01133 |
| Joint SENSE (JTV) (λ 0.0014) | 32.974 | 0.8920 | 0.2461 | 36.578 | 0.01428 |
| ZS-SSDU | 32.941 | 0.8960 | 0.2430 | 36.689 | 0.01391 |
| CS-SENSE (TV, converged) (λ 0.001) | 32.903 | 0.8908 | 0.2474 | 36.524 | 0.01445 |
| L1-SPIRiT (full-centre calibration, ks 3, λ 0.0005) | 32.144 | 0.8642 | 0.2302 | 35.656 | 0.01765 |
| CG-SENSE | 30.814 | 0.8421 | 0.3210 | 33.238 | 0.03080 |
| MC-INR (Niessen 2025) | 29.930 | 0.7999 | 0.3519 | 31.984 | 0.04112 |
| 3D-DIP | 28.424 | 0.7053 | 0.4950 | 32.321 | 0.03805 |
| 3D-INR | 28.126 | 0.6930 | 0.3146 | 29.993 | 0.06502 |

Not counted — L1-ESPIRiT (λ 0.003), replaced by L1-SPIRiT (user 2026-09-15); its row stays in the json: head PSNR 33.983, head SSIM 0.9054, head HFEN 0.1923, PSNR 37.268, NMSE 0.01218.

### MC-GS minus GS-single

| channel | head PSNR | head SSIM | head HFEN | PSNR |
|---|---|---|---|---|
| T1w | -0.231 | -0.0012 | +0.0012 | -0.216 |
| T2s | -0.014 | -0.0006 | +0.0003 | +0.004 |
<!-- TABLES:END -->

## Findings

1. **MC-GS does not lead this exam.** T1w: L1-ESPIRiT 38.830 > GS-single 38.086 > MC-GS 37.855
   (head PSNR, central 80). T2s: GS-single 34.175 ≈ MC-GS 34.160 > L1-ESPIRiT 33.860. L1-ESPIRiT is
   first on every T1w column; on T2s the two GS arms are first on everything except sharpness.
2. **Joint buys nothing here, and costs on T1w.** MC-GS minus GS-single is −0.23 dB on T1w and
   −0.01 dB on T2s, with SSIM, HFEN and sharpness all level or worse. On the adult AF16 pairs the same
   difference was positive in all four channels, and the record of that study called the growth of the
   joint advantage with sparser data its one qualitative finding. It does not reproduce here.
3. **The classical joint arm agrees.** Joint SENSE (vectorial TV) minus the converged independent
   CS-SENSE is −0.90 dB on T1w and +0.06 dB on T2s — the same pattern from a completely different
   implementation. Two independent joint methods failing the same way points at the pair, not at the
   code: these are the two least similar contrasts of the ten (r = 0.57), a short-TE T1-weighted image
   against a long-TE T2*-weighted one whose susceptibility foci (the basis of the calcified-nodule
   appearance) are by construction not edges of the T1w image. A shared-geometry prior is the wrong
   prior for this pair.
4. **The learned arms are unremarkable here.** ZS-SSDU early-stops at epoch 43 (29 min, against 5 h on
   the adult AF16 data) and lands mid-table. 3D-DIP and 3D-INR are 6–8 dB behind, as in every earlier
   topic, and MC-INR sits between them and CG-SENSE; they are completeness, not competition. MC-INR
   is the only other arm here with a joint prior, and it is 3.9 dB (T1w) and 4.5 dB (T2s) behind
   MC-GS, so nothing in this table argues that joint reconstruction as such fails on this pair —
   only that a shared-geometry prior buys nothing over the same method run per contrast.
5. The recipe was transferred, not tuned. Whether a loop tuned on this subject closes the T1w gap is
   open and is a separate decision.
6. **L1 arm swap (2026-09-15).** With L1-SPIRiT counted instead of L1-ESPIRiT, the T1w head PSNR and SSIM
   bars stay GS-single's (38.993 / 0.9687) and the T2s bars stay MC-GS's own, but the T1w head HFEN bar
   moves from GS-single's 0.0764 to L1-SPIRiT's 0.0679 — MC-GS (0.0768) is 0.0089 short of it. See
   **The L1-SPIRiT arm**.

## Figures

Slices are chosen on anatomy, not on which arm wins: transverse SPE 53 is vendor slice 38, the level
at which the stage-1 lesion survey saw the subependymal nodules; coronal RO 160 and sagittal PE 152 are
the head-area centroids. Error maps are `jet` on a fixed [0, 0.1] of the GT peak; panels are drawn at
the true 0.7 x 0.7 x 1.35 mm proportions, and every plane is displayed vertex-up (user 2026-09-13:
SPE increases inferiorly, so the coronal and sagittal panels take the same `origin` as the transverse
one, which was verified slice-for-slice against the vendor DICOM).

| file | what |
|---|---|
| `mask_af12.png` | Ω, the acquired support, Z and Ω ∪ Z on the PE x SPE plane |
| `peds_af12_<c>_row_axis2_53.png` | the ten arms in one row on the transverse slice, error maps below |
| `peds_af12_<c>_row3planes_compact_53_160_152_headerr.png` | three planes, per-panel head PSNR / SSIM / NMSE, head-only error maps |
| `..._headerr_p3f.png`, `..._inset48_headerr_af16style.{png,pdf}` | the same for the stage-3 final configuration (MC-GS = `gs_joint_p3f`; GS-single = `gs_single_p3f` on T1w, `gs_single_ar` on T2s); L1 column = L1-ESPIRiT, as drawn on 2026-09-15 morning |
| `..._headerr_p3f_spirit.png`, `..._inset48_headerr_af16style_spirit.{png,pdf}` | **the figures of record since 2026-09-15**: the same two figures with L1-SPIRiT in the L1 column (`scripts/figs_spirit.sh`); the `_af16style` one adds the 48-voxel zoom, a grey colorbar on the transverse row and a vector PDF |

## Files

```
mcinr/         the MC-INR reproduction's own outputs: recon/af12/ (the network images and the
               gauge-unit volumes), logs/ (the real-data check, the score and its gauge re-check)
data/          symlinks to the reference dataset; mask_af12_2d.mat (Ω), mask_af12_3d.h5,
               mask_af12_union_3d.h5 (Ω ∪ Z), support_2d.h5, pattern_union.{cfl,hdr}, mask_draws/
scripts/       prep_mask_draws.m, prep_mask.py, fig_mask.py, bootstrap.sh, peds_queue.txt,
               peds_sched.sh, run_l1espirit{,_ext}.sh, t1t2_common.py and the arm scripts,
               rescore.py, summarize.py, readme_tables.py, check_gauge.py, the figure scripts;
               recon_spirit.py, spirit_sweep{,_full24}{,_ext*}.sh, rescore_spirit.sh, figs_spirit.sh (L1-SPIRiT, 2026-09-15)
recon/         <arm>_<c>_complex.npy and the GS magnitude volumes; gt_<c>_complex.npy
logs/          prep_mask.json, the per-arm JSONs, rescore{,_full}.json, summary.{tsv,json},
               sched.log, sched_status.tsv, readme_tables.md; spirit_<c>_*.json (the 50 sweep points),
               archive_pre_spirit_20260915/ (summary / rescore / readme_tables as they were before the L1 swap)
figures/
```

Rerun: `bash scripts/peds_sched.sh scripts/peds_queue.txt` (GPU queue, skips finished jobs) and
`bash scripts/run_l1espirit.sh` (CPU); then `scripts/rescore.py --full`, `scripts/rescore.py`,
`scripts/summarize.py`, `scripts/readme_tables.py .`, the figure scripts. L1-SPIRiT: `bash scripts/spirit_sweep.sh`
and `bash scripts/spirit_sweep_full24.sh` (GPU), then `scripts/summarize.py`, `bash scripts/rescore_spirit.sh` (rescores
that arm alone on both gauges and merges it), `scripts/readme_tables.py . > logs/readme_tables.md`,
`scripts/refresh_readme_tables.py .`, `bash scripts/figs_spirit.sh`.

## Caveats

- The reference carries 2x parallel-imaging noise on its interpolated lines and no data in the
  partial-Fourier region; it is not a fully sampled truth.
- One subject, one exam. Nothing here is a claim about pediatric imaging in general.
- The GS recipe (P10) was selected on a different subject and a different contrast pair.
- `grain_ratio` on T2s rests on 664 white-matter voxels and should not be quoted.
- The lesion appearances noted in stage 1 are visual, not a radiological reading.

## Isolation

GPU 2 only and only when empty (one training per card); BART on the CPU with `nice 10` and 8 threads.
Nothing was written outside this folder, the worktree, `~/figs/peds_tsc_af12/` and the session
scratchpad. 26 queued jobs, all returned 0.
