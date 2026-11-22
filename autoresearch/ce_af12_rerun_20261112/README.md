# Plain-scan / enhanced pair at AF12: all reconstruction methods again, mask replaced

User, 2026-11-12:
- about 04:44 PST: "First, pause the sequential experiments of plain scan and enhancement with af = 16, and then
  change af to 12 and re-execute all the experiments."
- about 05:00 PST: wait for a free GPU (no queue is asked to yield); "The entire experiment refers to merely
  conducting the re-run of plain scan and enhancement experiments, simply by replacing the mask with af = 12";
  "re-perform all the reconstruction experiments for all methods using the new mask with af = 12 based on the plain
  scan and enhanced data."

So: the September study `Results_20260908/CE_retro_cssense_af16` (eleven arms) is repeated with one change, the mask.
No coupling tuning and no framework-ablation rows at AF12 (an earlier plan of this folder, dropped).

## What was paused (AF16)

The five open finalisation steps of the AF16 coupling tuning (`s7_nojoint_post`, `s7_full_keeper`, `fin_iso`,
`fin_noprior`, `fin_full_vol`; `../ablation_coupling_tune_20261111/program.md` section 9). None was running; the
session that owns that queue confirmed at 04:47 PST that it will not start them. Finished AF16 results are kept.

## What changes for AF12

| | AF16 (September) | AF12 (this rerun) |
|---|---|---|
| sampled points of 232 x 176 | 2552 (AF 16.000) | 3403 (AF 11.999; 40832 / 12 is not an integer) |
| generator | BART `poisson`, seed 0, ACS 24 x 24, no shutter | the same functions, imported from the topic; only the target count differs |
| density parameter | 3.883, 3 points trimmed | 3.000, 13 points trimmed |
| points shared with the other mask | | 691 (the 576 ACS points included) |

Unchanged: truth (CS-SENSE volumes), 42-coil maps, the noisy k-space of the whole grid (same noise draw), gauge and
scorer, every solver setting, lambda grid, network setting, and the frozen MC-GS / GS-single recipe (P3, 3.6 M
Gaussians, 4000 iterations, seed 42, trainer e778bf2). No arm is tuned for AF12; the swept arms (CS-SENSE, Joint
SENSE, L1-SPIRiT) pick their lambda on this data's truth as in September.

Arms: ZF, CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR, GS-single, MC-GS.
Queue: 55 jobs (2 ZF + CG-SENSE, 3 Gaussian trainings, 2 ZS-SSDU, 14 CS-SENSE, 5 Joint SENSE, 24 L1-SPIRiT,
1 MC-INR, 2 INR, 2 DIP), plus grid-edge extensions by the September rule.

## Where

- New topic on the server: `/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af12/`
  (`sim/mask/mask_af12_*`, `sim/gs` = links to the September k-space and maps, `configs/`, `scripts/` = the
  September scripts with the textual port recorded in `logs/setup.json`, `logs/`, `recon/`, `figures/`).
- MC-GS and GS-single train in the September trainer snapshot, in place, because its compiled voxelizer is then
  used without a rebuild (checked: ninja has no work to do). Their raw outputs are three new experiments in
  `CE_retro_cssense_af16/trainer/_trainer_e778bf2/results_arc/ce_retro_af12_{joint,single_pre,single_post}*`.
  The trainer also appended one line per run to that directory's shared `results_grain.tsv`, as it does for
  every run. Nothing else of the September topic is written.
- `codex_artifacts/mcgs_ce_af12_20261112/` holds the mask as first generated (the topic's files are byte copies)
  and an unused copy of the joint config.

## Files here

| File | Content |
|---|---|
| `make_af12.py` | makes the mask (`gpu.py run --cpu`); done, `MAKE_AF12 OK` |
| `mask_af12_vs_af16.png`, `mask_af12.json`, `ce_retro_af12.yaml` | its outputs (copies) |
| `run_af12.py` | `--setup` (topic), `--check_ext`, `--worker` (queue on one card), `--table`, `--finalize`, `--script` |

## State (local clock, PST)

- 04:53 mask written and checked. 05:12 topic created (`SETUP OK`). 05:16 `--check_ext`: a trainer start compiles nothing.
- 05:20 worker launched on card 7, job `j-20261112T132034-0c6dd569`, in waiting mode: the first job starts only
  when no `run_uniform.py --worker` process of this user is left (the uniform-mask queue of another session, its
  estimate 12:00 to 12:30) and the card has then been empty for 10 minutes. Card 2 is left to the T1+T2 queue.
- Status without clock values: `CE_retro_cssense_af12/logs/run_af12_status.json`, log `logs/run_af12.log`.
- 12:11 first job started on card 7 (local poll), after the uniform-mask queue had ended.
- The three Gaussian trainings (MC-GS, GS-single plain scan, GS-single enhanced) finished, but the step after
  training (convert to the topic's gauge + score) failed three times: `extract_gs_volumes.py` was given the
  physical card index 7 while only one card was visible ("invalid device ordinal"). Fixed in `run_af12.py`
  (`extract()`, index 0); the step alone was repeated on card 2 at 17:51 (`--extract`, all three rc 0, the
  zero-filled norm check of the conversion passes). No training was repeated. The first worker's log therefore
  shows `FAILED gs_*`; the results exist (`logs/gs_joint_*.json`, `logs/gs_single_*.json`).
- 18:01 second worker on card 2 (job `j-20261113T020209-a89ce30e`, `--only spirit_,mcinr,inr_,dip_`), after the
  T1+T2 queue's session released that card; card 7 continues with ZS-SSDU, CS-SENSE and Joint SENSE.
- Interim, 18:10 (head PSNR, plain scan / enhanced): ZF 24.541 / 26.125, CG-SENSE 31.663 / 33.916,
  ZS-SSDU 34.783 / 36.315, GS-single 35.743 / 37.981, MC-GS 35.913 / 38.256. Six arms still open per scan.
- 22:00 both workers ended: all 55 jobs have a result (the three `FAILED gs_*` lines are the conversion step that
  was repeated, see above). No lambda sat on a grid edge, so no extension job was added.
- 2026-11-13 00:21 `run_af12.py --finalize` (summary of the swept arms, gauge cross-check of the converted arms).

## Results (head-masked, whole volume; `CE_retro_cssense_af12/logs/summary.json`)

| method | pre PSNR | pre SSIM | post PSNR | post SSIM | AF16 pre / post PSNR |
|---|---|---|---|---|---|
| ZF | 24.541 | 0.6643 | 26.125 | 0.6691 | 24.250 / 25.829 |
| CG-SENSE | 31.663 | 0.8527 | 33.916 | 0.8963 | |
| CS-SENSE (lam 1e-3 / 7e-4) | 34.402 | 0.9246 | 36.621 | 0.9430 | |
| Joint SENSE (lam 1e-3) | 34.366 | 0.9211 | 36.466 | 0.9408 | |
| L1-SPIRiT (kernel 5, lam 1e-3) | 34.030 | 0.9094 | 36.598 | 0.9388 | |
| ZS-SSDU | 34.783 | 0.9356 | 36.315 | 0.9495 | 32.096 / 33.504 |
| 3D-DIP | 30.096 | 0.8453 | 31.264 | 0.8355 | |
| 3D-INR | 30.717 | 0.8038 | 33.477 | 0.8674 | |
| MC-INR | 30.291 | 0.7872 | 33.010 | 0.8505 | |
| GS-single | 35.743 | 0.9381 | 37.981 | 0.9545 | 33.429 / 35.302 |
| MC-GS | **35.913** | **0.9405** | **38.256** | **0.9573** | 33.623 / 35.657 |

MC-GS is first on head PSNR, SSIM, NMSE and HFEN on both scans; runner-up GS-single (+0.170 / +0.275 dB).

## Figure (record format)

`figures/ce_af12_prepost_201_131_48.png` (PNG sha256 `23c818fe…d10c27`, 200 dpi, 17 MB; preview JPEG and the
record JSON beside it; server: `CE_retro_cssense_af12/figures/prepost_three_201_131_48/`). Drawn by
`fig_af12_prepost.py`, which runs the record script `../ce_af16_lesion_figs_20261107/fig_lesion_plane_prepost.py`
with only the mask file, the AF label and the file stem replaced (job j-20261113T083209-2485acea, CPU):
`--topic CE_retro_cssense_af12 --three 201 131 48 --err_layout zoom --cbar single` -- the slices of the record.

MC-GS is strictly first on PSNR, SSIM and NMSE in all six panels against all ten arms. PSNR margin over the
runner-up (GS-single in every panel), pre / post: transverse 201 +0.25 / +0.36 dB, coronal 131 +0.19 / +0.28 dB,
sagittal 48 +0.19 / +0.29 dB (AF16 record: +0.25 / +0.43, +0.19 / +0.30, +0.21 / +0.38).

Difference from the AF16 figure of record: there, 3D-DIP and 3D-INR are the reruns with output support
(`../ce_af16_lesion_figs_20261107/sept_cmp_support.py`, user's choice 2026-11-09); here they are the September
settings, as the rerun was asked to change only the mask, so 3D-DIP shows signal outside the head. The head-masked
numbers are not affected by that signal; the support versions were stronger at AF16 (DIP pre 29.32 -> 30.76 dB).

## 3D-DIP / 3D-INR with output support at AF12 (user, 2026-11-13: "That does indeed require a re-run.")

Done so that the AF12 figure uses the same 3D-DIP / 3D-INR versions as the AF16 figure of record. Wrapper
`af12_cmp_support.py` = `../ce_af16_lesion_figs_20261107/sept_cmp_support.py` with one change, the mask:

- audited `recon_dip_inr.py` (mcgs_ce_mask_wt @ b01fadf) as a byte copy; its `t1t2_common.py` with only the mask file
  name replaced (`mask_af16_3d` -> `mask_af12_3d`, three places); `diag_artefact/` byte copy; all in the new topic
  `codex_artifacts/mcgs_ce_af12_cmp_20261113/scripts/`, whose `sim/` links to the AF12 topic's `sim/`
  (`logs/setup.json` has the hashes);
- arguments of the AF16 support run, unchanged (DIP `--support 6 --dip_base 96 --tv 0.03`, 12000 iterations;
  INR `--support 6 --inr_nmax 512 --tv 0.03`, 1500 iterations; checkpoint chosen by held-out k-space, no GT);
- CPU check (`--check`, job j-20261113T123027-c3066ce8): the loader reads the AF12 mask (3403 points, AF 11.999),
  truth and scales identical to September; each run also stops unless the script reports AF_rect ~ 12.
- Raw logs stay in the topic; printed lines and fetched JSON copies have the program's elapsed seconds removed.

Launched 2026-11-13 04:51 PST, one card each after two empty checks 75 s apart (any free GPU, user rule of that day):
DIP pre GPU 2 (j-20261113T125204-bae5f3f3), DIP post GPU 3 (j-20261113T125251-1b5b7b1b), INR pre GPU 4
(j-20261113T125344-67cb90ea), INR post GPU 5 (j-20261113T125425-d0f02f5d). Support covers 0.628 (pre) / 0.643
(post) of the grid.

### Figure with the support versions (2026-11-14, user: "Draw the supporting version with af equal to 12")

`figures/ce_af12_prepost_201_131_48_supfix.png` (PNG sha256 `69aaade8…3efd9e`, 16 MB; preview JPEG and record JSON
beside it; server: `CE_retro_cssense_af12/figures/prepost_three_201_131_48_supfix/`). Same script, slices and
layout as the figure above (CPU job j-20261114T145926-1fba1f4b); the only change is the source of two arms:
`--arm "3D-DIP=<O>/recon/dip_sup_b96_tv0.03_{s}_complex.npy" --arm "3D-INR=<O>/recon/inr_i10_n512_sup6_tv0.03_{s}_complex.npy"`
with `<O>` = `codex_artifacts/mcgs_ce_af12_cmp_20261113`. The other nine arms and the truth are the same files
(hashes in the record JSON).

MC-GS stays strictly first on PSNR, SSIM and NMSE in all six panels. The runner-up is GS-single in every panel, so
the PSNR margins are those of the figure above: transverse 201 +0.25 / +0.36 dB, coronal 131 +0.19 / +0.28 dB,
sagittal 48 +0.19 / +0.29 dB (pre / post). 3D-DIP no longer shows signal outside the head. This figure now matches
the AF16 figure of record in which versions of 3D-DIP / 3D-INR it shows.

## Optimal result for the plain-scan and enhanced reconstruction (user, 2026-11-22)

User, 2026-11-22: change the error colour bar to [0, 0.08] and use the "nature" style; then, of the two figures drawn
that way, "consider the second image as the optimal result for both plain scan and enhanced reconstruction", save it
as a 600 dpi PDF, keep the PDF on the GPU server too, and push a summary to a GitHub branch named for it
(`optimal-for-plain-scan-and-enhancement-phases`).

**The figure.** The support-version figure above (3D-DIP / 3D-INR with output support, the other nine arms and the
truth unchanged), redrawn in the article look:

| item | value |
|---|---|
| data | UIH plain-scan (pre-contrast) + enhanced (post-contrast) pair, CS-SENSE truth, BART Poisson-disc mask, nominal AF 12, actual 12.00x (40832 plane points / 3403 sampled) |
| planes | transverse 201 / coronal 131 / sagittal 48, all through the lesion; 48x48 box centred on the lesion core, magnified in the row below |
| methods | ZF, CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR, GS-single, MC-GS (September recipe, untuned) |
| error map | abs(recon - GT) / max(GT), raw per-pixel values, inferno, range [0, 0.08], inside the head only |
| arrow | red, on every slice image only (none in the error maps), same place along a row: the point where the error of the other methods (median of the nine reconstructions other than ZF and MC-GS, 7x7 local mean) exceeds that of MC-GS the most, searched only on the boundary of the lesion core in that plane (ROI dilated 2 px minus eroded 2 px) and at least 10 px from the edge of the head |

"Nature" style was read as the look of the T1 + FLAIR article figures (`../t1flair_af_record_fig_20261120/RECORD.md`,
server folder `article_nature_edge`): inferno error map and red arrows on tissue boundaries. The project has no other
style of that name. The lesion box was kept (it is not re-selected as in the T1 + FLAIR figures).

**Result.** The panel numbers are those of the support-version figure; only the drawing changed. MC-GS is strictly
first on PSNR, SSIM and NMSE in all six panels (18 of 18) against all ten arms. Runner-up: GS-single in every panel.

| plane | MC-GS pre / post PSNR (dB) | GS-single pre / post PSNR (dB) | margin pre / post (dB) | SSIM margin pre / post |
|---|---|---|---|---|
| transverse 201 | 36.84 / 39.95 | 36.59 / 39.59 | +0.252 / +0.364 | +0.0015 / +0.0031 |
| coronal 131 | 36.95 / 38.81 | 36.76 / 38.52 | +0.188 / +0.282 | +0.0031 / +0.0030 |
| sagittal 48 | 36.41 / 38.93 | 36.21 / 38.64 | +0.195 / +0.292 | +0.0033 / +0.0031 |

Whole head (table above): MC-GS 35.913 / 38.256 dB, +0.170 / +0.275 dB over GS-single.

**Arrow rule, second version (user, 2026-11-22: "it should point to the boundary area of the lesion. And not the area
of the skull. And the arrows should point to the areas on the error map that are beneficial for the MCGS").** The first
version searched all tissue boundaries of the box, and in the coronal rows the arrow landed next to the skull. Now the
point is taken on the lesion-core boundary only, away from the head edge, the script stops if no candidate has a lower
MC-GS error. A draft also repeated the arrow in the magnified error map; the user removed that ("Do not add arrows to
the error map"), so the error maps carry no arrow. Local error (7x7 mean) at the six points:

| row | point in the box (x, y) | others' median minus MC-GS | reconstructions with a higher error than MC-GS (of 9, ZF not counted) |
|---|---|---|---|
| transverse 201 pre | (34, 14) | 0.0033 | 9 |
| transverse 201 post | (30, 40) | 0.0029 | 9 |
| coronal 131 pre | (36, 22) | 0.0030 | 9 |
| coronal 131 post | (13, 21) | 0.0027 | 9 |
| sagittal 48 pre | (29, 35) | 0.0022 | 8 (GS-single is not higher) |
| sagittal 48 post | (35, 29) | 0.0021 | 7 (3D-DIP and 3D-INR are not higher) |

**Final version = the clinical verification scenario of the paper (user, 2026-11-22).** "Record the results here as
the optimal ones for both plain scan and enhanced scan. Then, the corresponding 600 dpi high-definition images were
organized into PDF vector graphics to serve as the clinical verification scenarios in the paper. The original data and
PDFs are saved on the GPU server." The title of the figure starts with "Clinical verification:".

**Files.** Server folder `codex_artifacts/mcgs_ce_af12_prepost_optimal_20261122/v3_clinical_verification/`:

| file | content | sha256 |
|---|---|---|
| `ce_af12_prepost_clinical_verification_600dpi.pdf` | the figure as a one-page PDF, 7.7 MB: text, boxes, arrows and colour bars are vector objects (TrueType text), every image panel is embedded at 600 dpi | `222b871f...ce8b18eb` |
| `ce_af12_prepost_clinical_verification.png` | the same figure, 200 dpi | `7c8cb5f0...72e3c437` |
| `ce_af12_prepost_clinical_verification_data.npz` | the drawn data, 10.9 MB: for each of the six rows the GT slice, the eleven reconstructed slices, head mask, error region and lesion core in volume orientation; the sampling-mask plane of each orientation; `meta_json` with box and arrow positions, panel numbers, display settings and the path + sha256 of every source volume | `90aafc16...c6b4261a` |
| `record_Sept_recipe_untuned_BART_Poisson_disc_AF12_three.json` | record: hashes of the 22 drawn volumes and of the truth, margins, arrow points | `a60ce196...67d55181` |
| `fig_lesion_plane_prepost_as_run_af12.py`, `fig_af12_prepost.py` | the code as it ran | |
| `ce_af12_lesion_three_planes_..._arrowlesion*` | the same PDF / PNG / npz / preview under the script's long name | |

- The full volumes stay where they were: the nine September-recipe arms and the truth in `CE_retro_cssense_af12/recon/`,
  3D-DIP / 3D-INR with output support in `codex_artifacts/mcgs_ce_af12_cmp_20261113/recon/` (paths in the record JSON).
- Mac: `figures/results_optimal/ce_af12_prepost_optimal_600dpi.pdf` and `ce_af12_prepost_optimal.png` (the same two
  files, same hashes; git-ignored: a large file in the tree makes the gpu.py snapshot exceed its 100 MiB limit);
  `figures/ce_af12_prepost_optimal_preview.jpg` and `figures/ce_af12_prepost_optimal_record.json` are committed.
- Job (CPU): j-20261122T092140-3ebf8da4. Superseded drafts on the server: the files directly in
  `mcgs_ce_af12_prepost_optimal_20261122/` (arrows on all tissue boundaries), `v2_lesion_arrow/` (arrow also in the
  error map), and `CE_retro_cssense_af12/figures/prepost_three_201_131_48*_nature*` (first drawings, both figures).

**Command.** `fig_af12_prepost.py` (the wrapper of the record script) with
`--three 201 131 48 --err_layout zoom --cbar single --err_vmax 0.08 --err_cmap inferno --arrow --arrow_mode lesion
--arrow_color red --role "Clinical verification" --save_data --pdf --pdf_dpi 600` and the two `--arm` overrides of the support version. The options
`--err_cmap`, `--arrow`, `--arrow_mode`, `--arrow_on`, `--arrow_skull_px`, `--arrow_color`, `--role`, `--save_data` were added to
`../ce_af16_lesion_figs_20261107/fig_lesion_plane_prepost.py` on 2026-11-22; their defaults leave every earlier
figure unchanged (jet, no arrow).

**To state with it.** The arrow is placed with the GT, at the lesion-boundary point most favourable to MC-GS, and is for
display only; at the two sagittal points one or two other reconstructions are locally as good or better. The margins over GS-single are 0.19 to
0.36 dB on one subject, one mask draw and one seed; the limits below apply.

## Limits to state with any AF12 number

- Same subject, same truth (a CS-SENSE reconstruction, re-simulated with the same coil maps), same noise draw.
- One mask draw, one seed.
- The lambda of the three swept arms is chosen on the truth; MC-GS is not tuned (as in September).
- GS-single here is the September arm (one scan fed to both channels of the joint trainer), not the independent
  single-contrast model of the framework ablation.
