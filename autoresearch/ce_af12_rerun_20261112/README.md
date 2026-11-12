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
  Nothing else of the September topic is written.
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
- After the queue: `run_af12.py --finalize` (summary, gauge cross-check, table against AF16), then the figures.

## Limits to state with any AF12 number

- Same subject, same truth (a CS-SENSE reconstruction, re-simulated with the same coil maps), same noise draw.
- One mask draw, one seed.
- The lambda of the three swept arms is chosen on the truth; MC-GS is not tuned (as in September).
- GS-single here is the September arm (one scan fed to both channels of the joint trainer), not the independent
  single-contrast model of the framework ablation.
