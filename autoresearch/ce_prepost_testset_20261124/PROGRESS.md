# Plain-scan / enhanced test set — progress

State of 2026-11-27 11:35 PST (clock of the local Mac; no server time is recorded anywhere).
Session `plain-enhan`. Plan and pipeline: [README.md](README.md).

## 1. Task

User, 2026-11-24: use the plain-scan / enhanced subjects of `/data2/congcong/data/3D/3D_PE_20261119` (on amax) as
test sets for the AF12 study. Order of work confirmed by the user: first a traditional reconstruction gives the
complete image (truth), then the 12-fold downsampling is simulated from it.

Choices made by the user:

| question | choice |
|---|---|
| which subjects | start with the subjects whose multi-coil k-space is readable for both scans |
| MC-GS setting | the tuned final config of `../ce_af12_tune_20261122`, seed 42 |

## 2. Status at a glance

| step | state |
|---|---|
| 0. probe of the data (CPU) | done |
| 1. k-space, prospective mask, coil maps (CPU) | done for all three subjects |
| 2. truth = CS-SENSE of all acquired lines, lambda on held-out lines (GPU) | done for all three subjects |
| 3. AF12 mask + simulated k-space per subject | 9003136585 and 9002944080 done (mask identical to the record's AF12 mask, truth residual reproduced); 9000103445 not started |
| 4. eleven methods | 9003136585: all 11 done, every stage exit 0; 9002944080: running (same job, card 3); 9000103445 waiting |
| 5. scoring, tables, figures | not started |

## 3. Subjects

| subject | plain scan | enhanced | grid (x, ky, kz) | coils | acquired points | prospective AF | ACS |
|---|---|---|---|---|---|---|---|
| 9003136585 | 16886147050000 | 16886147050050 | 256 x 232 x 176 | 18 | 7663 | 5.328 | 24 |
| 9002944080 | 16886244540000 | 16886244540050 | 256 x 232 x 176 | 50 | 7663 | 5.328 | 24 |
| 9000103445 | 16884476230000 | 16884476230050 | 256 x 232 x 190 | 42 | 8234 | 5.353 | 24 |
| 9003135093 (record) | 16887049370010 | 16887049370060 | 256 x 232 x 176 | 42 | 7663 | 5.328 | 24 |

- The enhanced series carry `<IsInjection>1</IsInjection>` in `scanparam.xml`; plain scan and enhanced share one
  prospective mask in every subject.
- The record subject is the one MC-GS was tuned on. It is reported apart from the three test subjects.
- Not usable yet (UIH `.raw` only, or only one scan converted to cfl; no UIH reader was found on the server):
  9003065128, 9003122771, 9003128786, 9003141620, 9003145528, 9003137160, 9003137288.

## 4. Checks before the reconstruction

| check | result | consequence |
|---|---|---|
| zero-filled images (`probe/probe_<subject>.png`) | normal anatomy and orientation in all three | data usable |
| k-space peak along the readout | index 126 of 256 in all three, and also 126 in the record subject | same convention as the record, nothing to correct |
| 9000103445: `Echo_1` vs `rawdata_T1_turmor` | identical up to a factor sqrt(2) (relative difference after scaling below 1e-5) | `rawdata_T1_turmor` is used |
| house ESPIRiT (`bart ecalib -t 0.001 -c 0.0 -r 24 -m 1`) on the record's plain-scan k-space vs the record's maps | per-voxel agreement in the head: median 0.99999, 5th percentile 0.9993 | the house recipe stands in for the record's maps |
| support of the record's maps | zero outside 61.2 % of the volume; an eigenvalue threshold of 0.8 (BART's default crop) gives Dice 0.989, not an exact match | new subjects' maps are cropped at eigenvalue 0.8 (decision of this session, not of the user) |

## 5. Preparation (step 1)

`prep_cases.py` writes, for each subject and scan,
`codex_artifacts/mcgs_ce_prepost_testset_20261124/subjects/<subject>/work_<pre|post>/`:

| file | content |
|---|---|
| `ksp_full.npy` | (coil, x, ky, kz) complex64, the source cfl, exact zeros off the acquired points |
| `mask2d.npy` | (x, ky, kz) float32, the prospective mask repeated along the readout |
| `csm.npy` | ESPIRiT maps, uncropped |
| `ev.npy` | ESPIRiT eigenvalue map (used for the crop) |
| `prep.json` | source, shapes, points, AF, ACS, readout peak, sha256 of the files |

A complete directory is skipped, an incomplete one is an error, nothing is overwritten.

## 6. Truth (step 2)

`truth_cssense.py` runs the record subject's own code (`Results_20260908/CE_pre_post_contrast/scripts`:
`ce_tv_sense.py`, `ce_common.py`) from a per-subject copy in which one line (the topic path) is changed.

| item | rule |
|---|---|
| solver | CS-SENSE, ADMM, 200 outer iterations, 8 CG steps, mu 0.1, Tikhonov 0.01 |
| normalisation | P99.9 of the zero-filled SENSE combine of this subject's scan |
| lambda | per scan: lowest residual on 5 % held-out acquired lines (95 / 5 split, seed 0) over 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3; a minimum on an end of the range extends it by a factor 2 (at most 3 times); then a refit on all lines |
| coil maps | house ESPIRiT, cropped at eigenvalue 0.8 |

No truth or other reference enters the lambda choice.

Result for 9003136585 (job `j-20261126T092259-9993259a`, card 4):

| scan | 2.5e-4 | 5e-4 | 1e-3 | 2e-3 | 4e-3 | chosen lambda | k-space residual after the refit on all lines |
|---|---|---|---|---|---|---|---|
| plain scan | 0.23968 | 0.23492 | 0.23058 | **0.22856** | 0.23170 | 0.002 | 0.12767 |
| enhanced | 0.23222 | 0.22849 | **0.22566** | 0.22585 | 0.23316 | 0.001 | 0.11670 |

(columns 2.5e-4 ... 4e-3: residual on the held-out lines). Both minima lie inside the range, so no extension was
needed. Maps kept after the crop: 61.0 % (plain scan) and 58.3 % (enhanced) of the volume.

All three subjects (card 4, one job each):

| subject | job | lambda plain scan / enhanced | k-space residual after the refit, plain scan / enhanced | note |
|---|---|---|---|---|
| 9003136585 | `j-20261126T092259-9993259a` | 0.002 / 0.001 | 0.12767 / 0.11670 | minima inside the range |
| 9002944080 | `j-20261126T100150-3456372d` | 0.004 / 0.004 | 0.20783 / 0.20562 | minimum on the upper end, range extended to 0.008, 0.004 stays the minimum |
| 9000103445 | `j-20261126T113155-ab2a2ad5` | 0.002 / 0.002 | 0.17402 / 0.13998 | minima inside the range |

Truths viewed by eye (`view_truth.py`, three central planes of plain scan, enhanced and difference):

- 9003136585: clean; an enhancing lesion behind the cerebellum.
- 9000103445: clean, fine cortical and cerebellar detail; sinus and meningeal enhancement; no obvious enhancing lesion
  in the central planes; a small wrapped region at the top left of the sagittal plane.
- 9002944080: normal anatomy and clear enhancement, but visibly smoother than the other two (larger lambda, higher
  residual: noisier acquisition). A bright wrapped region outside the head in the upper-left corner of the sagittal
  plane, in both scans. Caveat for the scores: a smooth truth favours smooth methods on this subject; whether the
  wrapped region falls inside the head mask is checked when the scores are made.

## 7. Next steps

1. `subject_topic.py` per subject: AF12 Poisson-disc mask by the September generator (seed 0, ACS 24 x 24; on the
   232 x 176 plane it must reproduce the record's AF12 mask bit for bit, 3403 points; 232 x 190 gets 3673 points),
   simulated k-space by the September simulator (noise level from the residual on the prospective mask, seeds
   100 / 200, gate: scale 1 and the truth run's residual reproduced), byte copies of the AF12 topic's method
   scripts, trainer configs with only paths and names replaced.
2. Eleven methods with the frozen settings of the AF12 topic (`run_subject.py`: the AF12 study's own `run_af12.py` and
   `af12_cmp_support.py` with only the topic path, grid, names and output folder replaced; 3D-DIP / 3D-INR = the
   output-support versions of the optimal figure); MC-GS = tuned final config, seed 42, stored as `gs_joint_tuned`;
   MC-GS with the registered recipe is kept beside it.
3. Head-masked PSNR / SSIM / NMSE / HFEN against the truth; table and figure per subject.

## 8. Results: 9003136585 at AF12 (all eleven methods)

Head-masked scores against the subject's truth, PSNR (dB) / SSIM / NMSE. Classical methods: best lambda of the frozen
grid by head PSNR, as in the AF12 study (all optima lie inside the grids). One seed (42) for the Gaussian models.

| method | plain scan | enhanced |
|---|---|---|
| ZF | 23.922 / 0.6673 | 25.811 / 0.6532 |
| CG-SENSE | 30.685 / 0.8389 | 32.948 / 0.8726 |
| CS-SENSE | 33.150 / 0.9149 (lambda 0.001) | 35.356 / 0.9267 (lambda 0.0007) |
| Joint SENSE | 33.094 / 0.9136 (lambda 0.0014) | 35.189 / 0.9238 (lambda 0.001) |
| L1-SPIRiT (kernel 5) | 32.839 / 0.9040 (lambda 0.002) | 35.243 / 0.9176 (lambda 0.001) |
| ZS-SSDU | 33.370 / 0.9248 / 0.01063 | 34.446 / 0.9276 / 0.01576 |
| MC-INR | 29.037 / 0.7591 / 0.02882 | 31.634 / 0.8048 / 0.03010 |
| GS-single | 34.528 / 0.9303 / 0.00814 | 36.677 / 0.9415 / 0.00942 |
| MC-GS, registered recipe | 34.701 / 0.9331 / 0.00782 | 36.864 / 0.9438 / 0.00903 |
| **MC-GS, tuned final config** | **34.824 / 0.9356 / 0.00760** | **37.038 / 0.9459 / 0.00867** |
| 3D-DIP (support version) | 30.815 / 0.8880 / 0.01914 | 32.475 / 0.8734 / 0.02480 |
| 3D-INR (support version) | 33.520 / 0.9180 / 0.01027 | 35.556 / 0.9267 / 0.01220 |

- MC-GS (tuned) is first on PSNR, SSIM and NMSE in both scans among all eleven methods; over GS-single
  +0.30 dB (plain scan) and +0.36 dB (enhanced); over the registered recipe +0.12 / +0.17 dB.
- MC-INR is below CG-SENSE on this subject; not yet compared with its level on the record subject.
- `run_subject.py --table` (registered MC-GS, without the tuned row and the support versions) also puts MC-GS first
  on head PSNR, SSIM, NMSE and HFEN in both scans. One combined table and the figure are still to be made.

## 9. Server power-off announced on 2026-11-27, and how to resume

Update 2026-11-27 11:35 PST: the server was still on; the job finished 9003136585 completely, built the topic of
9002944080 and started its 51-job worker. The text below was written at 06:20 PST; its resume steps hold with
"9003136585 complete" (its support results exist, so that stage is skipped).

The user powers the GPU server off. Job `j-20261127T034454-326e0b09` (card 3, `chain.py`, started 2026-11-26 19:44 PST,
local clock) was inside `--support pre` of 9003136585 (3D-DIP, about iteration 9200 of 12000) when this was written;
the power-off ends it there.

On disk on the server (`codex_artifacts/mcgs_ce_prepost_testset_20261124/subjects/`):

| subject | work, truth | AF12 topic | worker (51 jobs) | tuned MC-GS | support DIP / INR |
|---|---|---|---|---|---|
| 9003136585 | done | done | done, all exit 0 | done (`gs_joint_tuned`) | pre: 3D-DIP interrupted; rest not started |
| 9002944080 | done | not started | - | - | - |
| 9000103445 | done | not started | - | - | - |

Resume after power-on:

1. Check that the server paths and the GSTMI Python are back; count the project's cards (cap 3), two empty checks.
2. Look at `subjects/9003136585/cmp_support/` for partial files of the interrupted 3D-DIP run before relaunching
   (`run_subject.py --support` skips a scan only when both `logs/record_dip_<scan>.json` and
   `logs/record_inr_<scan>.json` exist; how the ported `af12_cmp_support.py` treats a partial run has not been tested).
3. Relaunch the same command:
   `gpu.py run --gpu ID -- @python autoresearch/ce_prepost_testset_20261124/chain.py --subjects 9003136585,9002944080,9000103445 --csm_crop 0.8`.
   An existing topic is only verified; the tuned run is skipped when its score and volumes exist. Whether the worker
   skips its 51 finished jobs on a rerun was not tested on these topics: watch the first lines of the log.
4. Then tables and figures per subject; check on 9002944080 whether the wrapped region lies inside the head mask.

## 10. GPU use

Rule of 2026-11-25 20:06 PST (user, workspace `CLAUDE.md`): all mcgs sessions together hold at most 3 amax cards, and
a card that already carries a task gets no further job. The sessions queued by waiting time; this study holds one
card and runs its jobs there one after another (truth: card 4; steps 3 and 4: card 3).

## 11. Files of this folder

| file | purpose |
|---|---|
| `README.md` | plan: user's request, subjects, pipeline |
| `PROGRESS.md` | this report |
| `probe_cases.py`, `probe/` | read-only probe of the data; its json and zero-filled images |
| `prep_cases.py` | k-space, mask, coil maps; the record and echo checks |
| `truth_cssense.py` | truth reconstruction with held-out lambda |
| `subject_topic.py` | AF12 mask, simulation, scripts and configs per subject |
| `run_subject.py` | the eleven methods on one subject's topic |
| `chain.py` | one job: `subject_topic.py` then `run_subject.py --all` for each subject |
| `view_truth.py` | three-plane picture of a subject's truth |
