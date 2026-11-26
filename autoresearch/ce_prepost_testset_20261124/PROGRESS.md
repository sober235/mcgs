# Plain-scan / enhanced test set — progress

State of 2026-11-26 03:14 PST (clock of the local Mac; no server time is recorded anywhere).
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
| 2. truth = CS-SENSE of all acquired lines, lambda on held-out lines (GPU) | 9003136585 done; 9002944080 running on card 4; 9000103445 waiting |
| 3. AF12 mask + simulated k-space per subject | script written (`subject_topic.py`), not run yet |
| 4. eleven methods | not started |
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
needed. Maps kept after the crop: 61.0 % (plain scan) and 58.3 % (enhanced) of the volume. The truth volumes have not
been looked at by eye yet (`view_truth.py` is written for this).

9002944080 (50 coils): job `j-20261126T100150-3456372d`, running on card 4 since about 02:01 PST 2026-11-26.

## 7. Next steps

1. Truth of 9002944080 (running) and 9000103445; view the three truths.
2. `subject_topic.py` per subject: AF12 Poisson-disc mask by the September generator (seed 0, ACS 24 x 24; on the
   232 x 176 plane it must reproduce the record's AF12 mask bit for bit, 3403 points; 232 x 190 gets 3673 points),
   simulated k-space by the September simulator (noise level from the residual on the prospective mask, seeds
   100 / 200, gate: scale 1 and the truth run's residual reproduced), byte copies of the AF12 topic's method
   scripts, trainer configs with only paths and names replaced.
3. Eleven methods with the frozen settings of the AF12 topic; MC-GS = tuned final config, seed 42.
4. Head-masked PSNR / SSIM / NMSE / HFEN against the truth; table and figure per subject.

## 8. GPU use

Rule of 2026-11-25 20:06 PST (user, workspace `CLAUDE.md`): all mcgs sessions together hold at most 3 amax cards, and
a card that already carries a task gets no further job. The sessions queued by waiting time; this study holds one
card (card 4) and runs its jobs there one after another.

## 9. Files of this folder

| file | purpose |
|---|---|
| `README.md` | plan: user's request, subjects, pipeline |
| `PROGRESS.md` | this report |
| `probe_cases.py`, `probe/` | read-only probe of the data; its json and zero-filled images |
| `prep_cases.py` | k-space, mask, coil maps; the record and echo checks |
| `truth_cssense.py` | truth reconstruction with held-out lambda |
| `subject_topic.py` | AF12 mask, simulation, scripts and configs per subject (not run yet) |
| `view_truth.py` | three-plane picture of a subject's truth |
