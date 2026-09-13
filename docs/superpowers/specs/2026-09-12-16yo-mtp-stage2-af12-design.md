# 16yo TSC MTP, stage 2: retrospective AF 12 on the two-contrast reference, nine-arm comparison — design

Date 2026-09-12 (evening). Branch `16yo-mtp`, worktree `/data0/congcong/code/GS/mcgs_peds_wt`. Stage-1 design
and reference: `docs/superpowers/specs/2026-09-12-peds-tsc-mtp-grappa-design.md` (the file keeps its original
name; the pointers in CLAUDE.md, STATUS.md and the stage-1 README that say `...-16yo-mtp-grappa-design.md` are
wrong and are corrected as part of this stage). Reference dataset `/data1/data1_congcong/data/3D/Peds_TSC_age16_MTP_GT/`.
New topic folder `/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro/`, mirrored in the repo as
`peds_tsc_mtp_af12_retro/`. Figures served at `http://localhost:8765/peds_tsc_af12/`. **DICOM and raw headers carry
the patient's name; nothing in the topic, the repo or the figures carries it.**

Approved in chat by the user on 2026-09-12: AF 12 as the overall acceleration of the complete mask (ACS counted,
sampled points over the whole matrix); the known-zero region is given to every method; the 23 GB of unwanted
contrast outputs stay; all nine arms; "continue" on the five-section design with the three defaults of section 3.

## 1. Goal

Undersample the GRAPPA-completed reference k-space of the two contrasts retrospectively at AF 12 with one
vd-Poisson mask, reconstruct with the nine-arm method table of the AF12 / AF16 topics, and report the head
metrics on the frozen gauge. The scientific question of the line: does MC-GS (joint T1w + T2*w) lead the table on
a pediatric 0.7 mm multi-echo GRE exam with a partial-Fourier, elliptically cut acquisition, as it does on the
adult T1 + FLAIR pairs.

## 2. Inputs (verified 2026-09-12)

| item | value |
|---|---|
| contrasts | `T1w` = FA16 / TE 3.25 ms (raw echo 0, u1 = 1); `T2s` = FA4 / TE 27.41 ms (raw echo 4, u1 = 0) |
| k-space | `ksp_<c>.h5`, dataset `kspace`, (32, 320, 304, 104) complex64, axes (coil, RO, PE, SPE), DC (160, 152, 52) |
| coil maps | `csm_espirit_<c>.h5`, dataset `map`, same layout, unit SoS (`ecalib -t 0.001 -c 0.0 -r 24 -m 1`) |
| reference image | `gt_sense_<c>.npy`, (320, 304, 104) complex64 = sum_c conj(S_c) ifft3c(ksp_c); the framework loader recomputes exactly this from k-space and maps (`presampled: false`) |
| support S | `masks_<c>.npz` key `envelope` = `m_acq | m_grappa`: 19914 of the 304 x 104 = 31616 (PE, SPE) positions, PE 0..226 inside the elliptical envelope; identical for both contrasts (same md5) |
| known zeros Z | `m_zero` = not S: 11702 positions (partial-Fourier block PE 227..303 and the elliptical corners); the reference k-space is exactly zero there |
| ACS | PE [140, 164) fully acquired; the 24 x 24 calibration square around DC lies entirely on acquired lines |
| partitions | 104 acquired with 30.75 % slab oversampling; the vendor exports the central 80 (SPE 12..91); the outer 12 on each side are the slab-oversampling margin where the excitation profile decays (mean signal per partition falls to 6–10 % of the centre at the two ends; the decay is asymmetric, partitions 92–96 still hold about 70 % on T1w) |

The reference carries 2x parallel-imaging noise on the GRAPPA-filled lines (ACS hold-out 0.096 / 0.188 against a
noise floor of 0.067 / 0.144) and no partial-Fourier synthesis. Every stage-2 number is relative to this reference,
not to a fully sampled truth, and absolute values are not comparable with the adult topics.

## 3. Decisions

| # | question | decision | alternative not taken |
|---|---|---|---|
| 1 | AF denominator | **user definition**: AF = whole matrix / sampled points, ACS counted. Matrix = 304 x 104 = 31616 → **budget N = 2635** points, all placed inside S; effective 19914 / 2635 = **7.56x against the acquired support**, quoted in every table and figure label as "AF 12 (7.56x of the acquired support)" | matrix = support → N = 1660 (12.00x of S, 19.05x of the grid). Default chosen because the user wrote "sampled points over the whole matrix"; switching is one number in `prep_mask.py` |
| 2 | may the mask sample GRAPPA-interpolated lines | **yes**: the completed k-space is the reference as a whole (user: "GT = the parallel-imaging reconstruction, then undersample") | restrict to `m_acq` (would impose the 2x comb on the pattern) |
| 3 | known zeros | **every arm** gets Z as measured zeros (user) | GS arms only, as in the AF12 / AF16 topics |
| 4 | mask sharing | **one mask for both contrasts** (same support, same prospective pattern in the acquisition) | per-contrast masks |
| 5 | primary evaluation region | **head mask restricted to the vendor's central 80 partitions**, computed by a topic-level rescore of every saved volume with the frozen functions (default of this design) | frozen gauge over all 104 partitions with no new code; kept as the secondary, cross-check table |
| 6 | method table | **all nine arms** (user) | classical + GS first |
| 7 | GS recipe | **P10 transferred as is, no tuning on this subject** | a tuning loop (a separate decision if MC-GS does not lead) |
| 8 | GPU | **GPU 2 only, and only when empty** (the line's standing rule; default of this design) | any empty card |
| 9 | the 23 GB of unwanted stage-1 outputs | **untouched** (user) | delete |

## 4. Mask

Grid (PE, SPE) = (304, 104), DC (152, 52). Support S and known zeros Z from `masks_T1w.npz` (asserted identical to
`masks_T2s.npz`). Budget N = round(31616 / 12) = 2635.

1. **Draw pool** (`scripts/prep_mask_draws.m`, MATLAB, house generator `/data0/congcong/code/ESPIRiT/utils/vdPoisMex.mexa64`
   as in every earlier topic): `vdPoisMex(304, 104, 304, 104, a, a, 24, 0, 1)` with an **isotropic** split
   `accelx = accely = a` swept over 2.6 .. 3.6 in steps of 0.05, five draws per value, `pause(1.1)` between draws
   because the generator seeds from `time(0)` at one-second resolution. Every draw is saved
   (`data/mask_draws/draw_a<a>_<k>.mat`). Isotropic rather than the old `accely = 4`: the 304 x 104 plane is far
   from the 256 x 176 of the adult topics and a fixed 4x along the 104-partition axis would starve it. If no draw
   reaches N inside S the sweep is extended downwards in `a`.
2. **Selection and trim** (`uih_mtp/retro_mask.py`, called by `scripts/prep_mask.py`): for every draw count the
   points inside S; take the draw with the smallest count that is >= N; remove points of S \ calib in order of
   decreasing Euclidean distance from DC (ties broken by `numpy.random.default_rng(0)`) until exactly N remain;
   never remove a calibration point. The trim must remove at most 10 % of N, otherwise the sweep is refined
   around the chosen `a` in steps of 0.01. The chosen draw file and the trim count are recorded.
3. **Outputs** (topic `data/`):
   - `mask_af12_2d.mat` (key `mask`, float32 (304, 104)) and `mask_af12_3d.h5` (key `mask`, uint8 (320, 304, 104),
     readout-constant) — the sampling mask Ω, N points. GS configs read the 2-D file; figures read the 3-D file.
   - `mask_af12_union_3d.h5` (key `mask`, uint8 (320, 304, 104)) — Ω ∪ Z, the forward-operator mask of every
     non-GS arm.
   - `support_2d.h5` (key `support`, uint8 (304, 104)) — S, the `known_zero_support` file of the GS arms.
   - `logs/prep_mask.json` — N, `AF_vs_full_grid` (31616 / N), `AF_vs_support` (19914 / N), chosen `a`, draw file,
     in-support count before the trim, trim count, calib check, Ω ⊂ S, Ω ∩ Z = ∅, union fraction
     (2635 + 11702) / 31616 = 0.4535, Ω fraction 0.0833. Also the two-contrast "effective AF" in the format of
     `prep_04_export.json` so the figure scripts can read the AF label.
   - `figures/mask_af12.png` — Ω, Z, S and the union in the PE x SPE plane, with the counts in the title.
4. **Acceptance**: exactly N points; every point inside S; every calibration point the chosen draw sampled
   survives the trim; the 3-D files reduce to the 2-D mask along the readout; the union equals Ω | Z bit for
   bit; the JSON numbers reproduce from the files.
   *Corrected 2026-09-13 during execution:* `vdPoisMex`'s `calib` argument biases the sampling density near DC
   but does **not** fully sample a `calib` x `calib` block — the adult topics' own masks sample 472 (AF16) and
   482 (AF12) of their 576 calibration positions, full only out to 16 x 16. "Fully sampled calibration square"
   was therefore never a property of the house generator and is not required here; this plane's mask samples
   366 of 576. Harmless for this study: every arm reconstructs with the ESPIRiT maps computed from the complete
   reference k-space, so no arm self-calibrates from the retrospective mask.

## 5. Known zeros for every arm

Z is handed to every method as measured zeros: the forward operator uses Ω ∪ Z and the data on Z are zero.
Self-supervised splits are drawn from Ω only, so no method trains or validates on trivially zero targets.

| arm | forward / data-consistency mask | split (if any) | change against the AF16 topic scripts |
|---|---|---|---|
| ZF, CG-SENSE, CS-SENSE, Joint SENSE | union 3-D h5 (`Pair.m3`); `y = k_full * m3` is zero on Z by construction | – | mask path becomes a parameter (`--mask` / `T1T2_MASK_<c>`), default the union file; `recon_classical.py` no longer hard-codes the mask name |
| L1-ESPIRiT (BART `pics`) | k-space zero outside Ω ∪ Z **and** an explicit pattern: `pics -p data/pattern_union.cfl -l1 -r λ -i 100 -S`; without `-p` BART infers the pattern from nonzero entries and would treat Z as unsampled | – | write the pattern CFL from the union mask, add `-p`; assert the pattern sum equals the union count |
| ZS-SSDU | DC block on the union; Θ / Λ / Γ drawn from Ω | Ω | split mask separated from the DC mask (`Pair.m3_omega`), counts logged |
| 3D-DIP, 3D-INR | forward on the union; the 95 / 5 held-out lines drawn from Ω | Ω | same separation |
| GS-single, MC-GS | `mask_path: data/mask_af12_2d.mat` (Ω), `known_zero: true`, `known_zero_support: data/support_2d.h5`; the framework builds `mask_fit = Ω ∪ Z`, zeroes the data on Z, keeps the DC normaliser at |Ω| (`paired_gs/data/known_zero.py`, `known_zero_from_support`) | – | none (config only); the score JSON must show `n_nonzero_in_z = 0`, `kz_energy = 0`, `frac_z = 0.370` |

`Pair` (topic copy of `t1t2_common.py`) loads both masks and exposes `m3` (union) and `m3_omega` (Ω); `mask_frac`
in every arm's JSON is the union fraction and `mask_frac_omega` the Ω fraction, so a wrong file is visible in the
logs. Ω ⊂ S guarantees no conflict between a sampled line and Z.

## 6. Arms and recipes

Scripts are copied from `Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/scripts/` by a bootstrap (section 8) and
kept byte-identical except for names, paths, the mask parameter and the split separation of section 5.

| arm | script | settings | λ scan (per contrast, chosen by the primary head PSNR; Joint SENSE has one λ for the pair, chosen by the sum of the two head PSNRs; if the best λ sits at an edge of its grid, one more value is added beyond that edge — the edge value times or divided by 1.4, the grids' own ratio — and rerun until the best is interior) |
|---|---|---|---|
| ZF (anchor, not tabled) | `recon_classical.py` | – | – |
| CG-SENSE | `recon_classical.py --methods cgsense` | Tikhonov 0.01, 60 CG iterations | – |
| CS-SENSE | `recon_classical.py --methods cssense --admm_outer 200` | ADMM isotropic 3-D TV, μ 0.1, 8 inner CG | T1w {0.0007, 0.001, 0.0014, 0.002}; T2s {0.0005, 0.0007, 0.001, 0.0014} |
| Joint SENSE | `recon_joint_cssense.py --contrasts T1w,T2s --admm_outer 200` | vectorial TV | {0.001, 0.0014} |
| L1-ESPIRiT | `recon_l1espirit.py --iters 100 --threads 8` | BART on the CPU, `nice -n 10`, `-S`, `-p` | T1w {0.001, 0.0015, 0.0025, 0.004}; T2s {0.003, 0.005, 0.007, 0.01} |
| ZS-SSDU | `recon_zsssdu.py --epochs 100` | as AF16 (10 unrolls, 15-layer ResNet, Γ 10 %, Θ/Λ 80/20 re-split) | – |
| 3D-DIP | `recon_dip_inr.py --method dip --iters 5000 --lr 3e-3 --eval_every 100` | checkpoint by held-out 5 % residual | – |
| 3D-INR | `recon_dip_inr.py --method inr --iters 1500 --eval_every 50` | same | – |
| MC-GS | `CONFIG=configs/peds_af12_T1w_T2s.yaml SAVE_VOLUMES=1 ./autoresearch_peds.sh peds_af12_P10_joint <gpu> 4000 <P10 flags>` | P10 | – |
| GS-single | same with `configs/peds_af12_single_T1w.yaml` (both channels T1w, both wavelet λ 0.12) and `configs/peds_af12_single_T2s.yaml` (both T2s, both λ 0.10) | P10 | – |

P10 (the jyz AF12 line, `autoresearch_jyz_af12.md` "Promotion 8") = db4 wavelet on the finest level only,
wavelet λ T1 0.12 / FLAIR 0.10, edge-TV 0.10 / κ_q 0.85, repaired cold start, 3.6M points, 4000 iterations.
Command-line flags shared by the three GS runs (T1 → T1w = `plain`, FLAIR → T2s = `contrast`):

```
--iters 4000 --lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 --edge_tv --edge_tv_kappa_q 0.85
--edge_tv_wfloor 0.10 --tv_eps 0.04 --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn
--set loss.edge_tv_kappa_source=zf --set gaussian.initial_num_points=3600000 --set loss.wav_name=db4
--set loss.wav_levels=1
```

The two wavelet weights are **not** on the command line (a `--set` would override every config alike); they live
in the configs: joint `loss.lambda_wav_plain: 0.12`, `loss.lambda_wav: 0.10`; single T1w 0.12 / 0.12; single T2s
0.10 / 0.10. The runner records the resolved config in `results_arc/<exp>/<exp>/config.yaml`, and the plan checks
the four values there before the runs are read.

Config template: `configs/t1flair_v3_af16_jyz_flair_c.yaml` (the variant with an explicit support file), with
`data.plain` = T1w and `data.contrast` = T2s, `data_path` / `csm_path` pointing at the dataset files, `mask_type:
external`, `readout_axis: 0`, `phase_axes: [1, 2]`, `presampled: false`, `normalize_kspace: true`,
`normalization_percentile: 99.9`, `acceleration_factor: 12`, `known_zero: true`, `known_zero_support` as above.

## 7. Evaluation

Frozen definitions (`diag_artefact/headonly.py`, not modified): GT and reconstruction divided by P99.9 of |GT|;
head mask = |GT| > 0.05 · P99.9(|GT|) per contrast; head PSNR with the volume peak of |GT|; head SSIM = mean of the
7^3 uniform-window SSIM map over the head; head HFEN = LoG(σ 1.5) residual norm ratio over the head; whole-volume
PSNR / NMSE; sharpness ratio; grain. Magnitude gauge is primary; the complex-gauge head PSNR is reported for every
arm that saves a complex volume (all nine: the GS arms save complex volumes with `--save_volumes`).

**Primary table** — `scripts/rescore.py` (uses `uih_mtp/rescore.py`): for every arm and contrast load
`recon/gt_<c>_complex.npy` and the arm's volume (complex where available, else magnitude), compute the metric maps
on the **full** volume with the frozen functions (SSIM map, LoG residual, squared error; the head mask and the
PSNR peak from the full-volume |GT| exactly as in the frozen gauge), then average / norm them over
`head ∧ (12 <= SPE < 92)`; whole-volume PSNR / NMSE over all voxels with `12 <= SPE < 92`. Writes
`logs/rescore.json` and `logs/summary_central80.tsv`; `readme_tables.py` builds the README tables from it, with
the arm's λ choice made on the same numbers.

**Secondary table** — each arm's own JSON (full 104 partitions, the frozen gauge exactly as in the other topics)
collected by `summarize.py` into `logs/summary.json` as before. `check_gauge.py --contrast T1w --method cgsense`
must print `GAUGE OK` (the copied gauge equals the framework's to 2e-4). The rescore with the range `0 <= SPE < 104`
must reproduce the secondary numbers to the same tolerance (unit test).

**Figures** (scripts copied from the AF16 topic, shape read from the GT, error maps in `jet` on [0, 0.1] of the
GT peak, head-only errors in the compact figure): the nine-arm row figure per contrast with error maps below and
the mask bottom-left; the compact three-plane figure per contrast with per-panel head PSNR / SSIM / NMSE; slices
chosen by `best_slice_mcgs.py` inside the central 80 partitions plus one typical level; every title carries
"AF 12 (7.56x of the acquired support)". Figures are mirrored to `~/figs/peds_tsc_af12/` and probed over HTTP.

## 8. Topic folder, naming, bootstrap

```
Results_20260908/Peds_TSC_MTP_AF12_retro/
  README.md                       data, protocol, mask, tables (between TABLES markers), figures, gotchas
  data/ksp_T1w.h5 -> /data1/.../ksp_FA16_TE3.25.h5        (symlinks; likewise csm_espirit_T1w.h5, gt_sense_T1w.npy,
  data/ksp_T2s.h5 -> /data1/.../ksp_FA4_TE27.41.h5         masks_T1w.npz; and the T2s four)
  data/mask_draws/ mask_af12_2d.mat mask_af12_3d.h5 mask_af12_union_3d.h5 support_2d.h5 pattern_union.{cfl,hdr}
  scripts/  py (GSTMI wrapper, PYTHONPATH -> mcgs_peds_wt), bootstrap.sh, prep_mask_draws.m, prep_mask.py,
            t1t2_common.py, recon_classical.py, recon_joint_cssense.py, recon_l1espirit.py, recon_zsssdu.py,
            recon_dip_inr.py, extract_gs_volumes.py, add_gs_arm.py, summarize.py, rescore.py, check_gauge.py,
            readme_tables.py, refresh_readme_tables.py, make_figures.py, fig_row_final.py,
            fig_three_planes_compact.py, best_slice_mcgs.py, postprocess.sh, peds_queue.txt, peds_sched.sh
  recon/    <arm>[_suffix]_<c>_complex.npy, gs_*_<c>_{mag,complex}.npy, gt_<c>_complex.npy
  logs/     prep_mask.json, classical_*.json, l1espirit_*.json, jcssense_*.json, zsssdu_*.json, dip_*.json,
            inr_*.json, rescore.json, summary*.tsv/json, sched_status.json, run logs
  figures/
```

- Contrast names inside the topic are `T1w` and `T2s` (no dots; the λ-file regexes of the figure scripts break on
  `FA16_TE3.25`). The mapping to the acquisition is stated in the README and in `data/contrasts.json`.
- `scripts/bootstrap.sh` copies the AF16 jyz scripts, rewrites topic paths, the subject / topic names, `T1` → `T1w`,
  `FLAIR` → `T2s` (argparse choices, key maps, `--map T1w:plain,T2s:contrast` for `add_gs_arm.py`), the volume
  shape (read from the GT instead of the literal 256 x 256 x 176), and the mask parameter of section 5; then runs a
  path check that every referenced input resolves. Gauge functions are diffed against the source to prove they are
  unchanged.
- Repo side: `configs/peds_af12_T1w_T2s.yaml`, `configs/peds_af12_single_T1w.yaml`, `configs/peds_af12_single_T2s.yaml`;
  runner `autoresearch_peds.sh` (copy of the AF16 runner with `CONFIG=`, ledger `autoresearch_peds.tsv`,
  `scripts/autoresearch_peds_row.py`); package code `uih_mtp/retro_mask.py`, `uih_mtp/rescore.py`; mirror
  `peds_tsc_mtp_af12_retro/` (README, scripts, logs, figures; no volumes; symlinks are never written through).
- The stage-1 topic `Peds_TSC_MTP_3D/` is not modified except its README gaining a pointer to this topic.

## 9. Compute, isolation, scheduling

- GPU 2 only, and only when `nvidia-smi --query-compute-apps` shows it empty (decision 8); one training per card.
  Every GPU job is a queue entry in `scripts/peds_queue.txt`; `scripts/peds_sched.sh` launches the next entry only
  on an empty allowed card, one job at a time, restart-safe (done files, adoption of a running job by its process
  signature), status in `logs/sched_status.json`. Queue order: ZF + CG-SENSE + CS-SENSE scans (short) →
  MC-GS → GS-single T1w → GS-single T2s → Joint SENSE scan → ZS-SSDU T1w, T2s → DIP T1w, T2s → INR T1w, T2s.
- L1-ESPIRiT runs on the CPU (`nice -n 10`, 8 threads, one BART job at a time because the cardiac line also uses
  BART on the CPU) in parallel with the GPU queue, from the start.
- Budget estimate from the AF16 timings (the peds volume is 12 % smaller than 256 x 256 x 176): classical scans
  about 1 h, GS 3 x ~2 h at ~33 GB VRAM, ZS-SSDU 2 x ~5 h, DIP 2 x ~3 h, INR 2 x ~2 h — about 30 h of serial GPU
  time plus about 8 h of CPU BART. Nothing starts until GPU 2 is free (it carries the AF12 phase-2 chain at the
  time of writing).
- Long steps detached from bash script files with `setsid nohup … > log 2>&1 < /dev/null &` and a PID file;
  waiting with `kill -0`. Nothing is written outside the worktree, the topic folder, `~/figs/peds_tsc_af12/` and
  the session scratchpad; the other worktrees, their ledgers and topic folders are not touched; the archive root
  README gets one added row (and its stale branch name `peds-tsc-mtp` corrected to `16yo-mtp`).

## 10. Tests and acceptance

- `tests/test_uih_mtp_retro_mask.py`: on a synthetic support and synthetic draws — selection takes the smallest
  in-support count >= target; the trim yields exactly N, keeps every calibration point, removes the outermost
  points first and is deterministic; exported `.mat` / `.h5` keys, shapes and dtypes; the union equals Ω | Z and
  Ω ∩ Z is empty; the AF numbers reproduce from the files.
- `tests/test_uih_mtp_rescore.py`: with the full partition range the rescore equals `score_headonly` and the
  whole-volume metrics of the frozen gauge to 1e-6 relative on random volumes; with a restricted range it equals a
  direct computation on the restricted mask.
- `tests/test_uih_mtp_bart_pattern.py` (skipped when `bart` is absent): on a small synthetic problem `pics -p`
  with the nonzero pattern equals `pics` without `-p`; with a pattern that also covers zero-valued entries the
  result differs and its k-space on those entries moves towards zero — proof that `-p` is honoured.
- Split separation: `recon_zsssdu.py` / `recon_dip_inr.py` gain a `--dry_run` that prints the DC-mask count
  (union) and the split-source count (Ω); the plan checks both against `prep_mask.json`.
- Gauge: `check_gauge.py` prints `GAUGE OK` on CG-SENSE T1w; the full suite (194 at the start of this stage) passes
  before every push.
- The topic is complete when all nine arms x two contrasts have primary and secondary rows, every λ scan satisfies
  the edge rule, the GS volumes are saved, the README tables and figures are regenerated by `postprocess.sh`, the
  record and STATUS are written, and the branch is committed, pushed and tagged `handoff/<date>-16yo-mtp-af12`.

## 11. Out of scope

Tuning the GS recipe on this subject; noise prewhitening; the other eight contrasts; the 2D FSE and wave-CAIPI
scans; acceleration levels other than 12; deleting the 23 GB of stage-1 leftovers; changes to framework code
outside `uih_mtp/` (the GS arms run on configuration alone).

## 12. Deliverables

Topic README with the two tables (primary central-80, secondary full-volume) and the λ scans; the row and compact
figures per contrast; `docs/superpowers/records/<date>-16yo-mtp-af12.md`; STATUS.md and CLAUDE.md updated
(including the corrected stage-1 spec pointers); the archive root README row; commit, push, tag.
