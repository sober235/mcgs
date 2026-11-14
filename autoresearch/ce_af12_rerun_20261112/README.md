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
(post) of the grid. Next: redraw the figure with `--arm 3D-DIP=... --arm 3D-INR=...`.

## Limits to state with any AF12 number

- Same subject, same truth (a CS-SENSE reconstruction, re-simulated with the same coil maps), same noise draw.
- One mask draw, one seed.
- The lambda of the three swept arms is chosen on the truth; MC-GS is not tuned (as in September).
- GS-single here is the September arm (one scan fed to both channels of the joint trainer), not the independent
  single-contrast model of the framework ablation.
