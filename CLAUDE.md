# Paired-ARC-GS / MC-GS framework — session entry

Read in this order: this file → `STATUS.md` (current state, what to re-run first) →
`autoresearch_t1t2.md` (the live campaign: program, verdicts, round log) →
`docs/superpowers/records/` (dated session records, newest first) → `.claude/memory.md`
(the long project memory, §0–19; do not re-derive anything it already closed).

## Environment
- Python: `/home/congcongliu/anaconda3/envs/GSTMI/bin/python` (torch 2.7.1+cu118), always with
  `PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- Tests: `PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests -q --ignore=tests/smoke_test.py` (174, ~2 min, needs a GPU).
- Data roots: `/data0/congcong/code/GS/Results_20260908/<TOPIC>/data` (retrospective AF12 topics:
  k-space h5, ESPIRiT maps, masks); the old T1+FLAIR bundle paths are in `configs/winner_v2*.yaml`.
- Results archive: `/data0/congcong/code/GS/Results_20260908/` — one topic per folder, README +
  scripts + logs + figures; the repo mirrors each topic (without volumes) as `<topic>/`.
- Figures are served at `http://localhost:8765/<dir>/<file>.png` from `/home/congcongliu/figs/`.

## Running things
- One MC-GS experiment: `./autoresearch_t1t2.sh EXP GPU ITERS [cycle.py args | --set key=value]`;
  `CONFIG=` picks the topic config, `SAVE_VOLUMES=1` dumps volumes, `WARMUP=20` for short smokes.
- Launch anything longer than minutes detached, from a **bash** script file:
  `setsid nohup ./autoresearch_t1t2.sh … > /dev/null 2>&1 < /dev/null &` — the session's memory
  watchdog kills background tasks on MemFree, and the tool shell (zsh) does not word-split `$VAR`.
- GPUs are shared with other users and sessions: check `nvidia-smi --query-compute-apps` first.

## Commits
- Author = the repo identity (`Congcong Liu <liuc23539@gmail.com>`), plain human messages, no
  AI trailers. Stage named files only: the working tree carries old untracked material.
- Branch topology: this framework's lineage is `master` (d0a47c0) → `af10-data-audit` →
  `af12-to-af10` → `t1t2-af12-retro` → `autoresearch-t1t2` (current). `main` / `origin/main`
  are an unrelated earlier snapshot plus the CE work; never merge across the two lineages.

## jiangyizhou AF16 autoresearch line (branch `autoresearch-jyz-af16`, this worktree)
Read `docs/handoff/2026-09-11-autoresearch-jyz-af16.md` first, then `autoresearch_jyz_af16.md`. Results summary:
`docs/superpowers/records/2026-09-11-autoresearch-jyz-af16.md`. Tests on this branch: 183 pass; `tests/test_score_edge.py`
needs the gitignored `results_arc/reg_aligned_4k_volumes.npz`, which only the main checkout has.

## Best 12x reconstruction so far (user decision 2026-09-11, provisional)
The user marked this line's result as the current best AF 12 reconstruction on jiangyizhou, to be used until superseded.
- Figure the user identified: `t1flair_af16_v3_jiangyizhou/figures/t1flair_v3_af16_{T1,FLAIR}_row3planes_compact_174_110_47_headerr.png`
  (served at `http://localhost:8765/t1flair_v3_af16/`), orange in-panel metrics = head-masked slice PSNR / SSIM / NMSE, error
  maps inside the head only; script `t1flair_af16_v3_jiangyizhou/scripts/fig_three_planes_compact.py`. That figure is at
  nominal AF 16 (actual T1 15.19x, FLAIR 12.49x of its acquired k-space), not AF 12.
- Recipe (P6, `autoresearch_jyz_af16.md`): repaired cold start (init_mode head, 3-NN scales, kappa from ZF), 3.6M points,
  wavelet T1 0.10 / FLAIR 0.07, edge-TV lambda_geo 0.10 / kappa_q 0.85, 4000 iterations. AF16 run `ar14_n36_v2tv_full4k`:
  head PSNR 35.552 / 34.807 (T1 / FLAIR), first on head PSNR, SSIM and HFEN over L1-ESPIRiT.
- The same recipe at AF 12 (the jyz AF12 topic's mask: one vd-Poisson mask, AF 11.71 on T1 / 9.74 on FLAIR against their
  own support): `results_arc/ar17_P6_tf_jyz_af12_full4k` — head PSNR 36.753 / 36.165, head SSIM 0.9480 / 0.9374, head HFEN
  0.1009 / 0.1002, whole-volume PSNR 40.071 / 39.539; first on every head metric of the jyz AF12 table except FLAIR head
  HFEN (level with L1-ESPIRiT). Volumes (magnitude only): `results_arc/ar17_P6_tf_jyz_af12_full4k_volumes.npz`, score json
  beside it. Not registered in `Results_20260908/T1FLAIR_AF12_v3_jiangyizhou`, whose proposed arm is the huangyuwei-selected
  v3.2 (36.299 / 36.039). No compact orange-metric figure exists at AF 12 yet.
- Caveats carried over: recipe and classical lambda were selected on this subject (fair, not held out); all numbers are on
  the magnitude gauge — on the strict complex gauge L1-ESPIRiT ranks first (record, section 4).

## Phase 2 and the freeze (branch `autoresearch-jyz-af16-phase2`, this worktree, 2026-09-12)
Read `STATUS.md`, then `docs/handoff/2026-09-12-autoresearch-jyz-af16-phase2.md` and the phase-2 record. Phase 2 added
the complex-gauge scorer and the phase-only TV term (`paired_gs/losses/phase_tv.py`); p6ptv = P6 + phase TV 0.05 is first
on both gauges. **User decision 2026-09-12: this line is frozen — no further runs; no code or parameter from the AF12
line (`autoresearch-jyz-af12`, recipe P10) is merged or adopted here.** The AF12 numbers quoted in the section above
are superseded by that line's own record; they stay here as history.
**User decision 2026-09-12 (later the same day): the recipe quoted externally is P6 (`gs_joint_ar`, magnitude gauge);
p6ptv (`gs_joint_p2`) is reported only as the both-gauges note.**

## Phase 3 (branch `autoresearch-jyz-af16-phase3`, this worktree, 2026-09-12/13)
The user reopened this line to widen the gap to the comparison methods. Read `STATUS.md`, then
`docs/handoff/2026-09-13-autoresearch-jyz-af16-phase3.md` and the phase-3 record. **Phase-3 recipe** = P6 + the wavelet
prior changed to db4 on its finest level only (T1 0.12 / FLAIR 0.10) + common-edge `--lambda_edge 0.001` + phase TV
`--lambda_ptv 0.03`, 4000 iterations: head PSNR **35.706 / 34.928**, margins over L1-ESPIRiT **+0.497 / +0.425**
(P6: +0.343 / +0.304), head SSIM 0.9396 / 0.9261, HFEN 0.1139 / 0.1121, sharpness **0.855 / 0.804 — at or above
L1-ESPIRiT's 0.849 / 0.789 on both contrasts, which P6 never was**. Registered as `gs_joint_p3` / `gs_single_p3`;
figures of record (user decision 2026-09-13) `figures/t1flair_v3_af16_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_p3.png`: whole transverse 180 / coronal 124 / sagittal 76 with the 48-voxel box outlined and magnified in the corner, in-panel metrics kept, colorbars on the transverse row only. The 174 / 110 / 47 pair without a zoom is also there. The 1.0 dB target the user
set was NOT reached (the margin roughly doubled). **Which recipe is quoted externally is still the user's call — the
section above (P6) stands until they change it.**

## Nominal AF20 study (branch `autoresearch-jyz-af20`, this worktree, 2026-09-13/14) — complete
The worktree directory is still called `mcgs_af16_wt`; the branch checked out here is the AF20 study, forked from the
phase-3 tip f0fe009. Read `STATUS.md`, then `docs/handoff/2026-09-14-af20-jyz.md` and the record
`docs/superpowers/records/2026-09-14-af20-jyz.md`; topic `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou/`
(mirror `t1flair_af20_v3_jiangyizhou/`). **The only difference from the AF16 topic is the mask** (`vdPoisMex` accelx 5:
2427 PE points, T1 18.56x, FLAIR 15.13x against its shutter; time-seeded, cannot be regenerated). MC-GS = the AF16
phase-3 recipe frozen in `run_af20_mcgs.sh`; classical λ re-swept at AF20. Result, nine arms: head PSNR
**34.832 / 34.179**, margins over L1-ESPIRiT **+0.461 / +0.491** (AF16 +0.497 / +0.425), first on all six head cells;
coupling +0.165 / +0.052; T1 sharpness 0.833 below L1-ESPIRiT's 0.845. The margin does not grow with acceleration.
Figures `t1flair_af20_v3_jiangyizhou/figures/t1flair_v3_af20_*_p3.*` (AF16 figure-of-record layout, same 48 box; the box
leads shrink at AF20 — selection effect, quote the whole-volume margin). Mask-draw variance (four draws, record §8): margins
+0.459 ± 0.002 / +0.488 ± 0.004 dB, so the AF16→AF20 shifts are real; T1 sharpness gate fails on 4/4 draws. MC-INR
(Niessen 2025, parallel session, same mask) registered: 30.969 / 31.666. User 2026-09-14: no huangyuwei, masks not in git,
MC-INR not rerun. Every result is at the NOMINAL AF20 (actual T1 18.56x, FLAIR 15.13x; record §10); open items: the T1
sharpness gate, and nominal label vs an exact-20x rerun. Chinese summary: `docs/2026-09-14-af20-jyz-summary.md`.

## Branch `jyz-af20-results` (2026-09-14, this worktree)
Cut from the AF20 tip 1cc612a at the user's request and now checked out here; it carries everything above plus one commit of
the afternoon's review. Read `STATUS.md` first.
- **AF definition (user decision 2026-09-14):** AF = PE-plane lines / sampled lines that carry that contrast's data (what BART
  `pics` prints as `Acc`). For the jiangyizhou masks: AF12 topic T1 11.71 / FLAIR 12.86, AF16 topic 15.19 / 16.49, AF20 topic
  18.56 / 19.96. The "nominal AF 20 (actual T1 18.56x, FLAIR 15.13x)" labels in the AF20 docs and figure titles are the older
  convention; relabelling them (and the AF16 figures) waits for the user's go.
- Cross-AF review topic `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_16_20_trend_v3_jiangyizhou/` (mirror
  `t1flair_af12_16_20_trend_v3_jiangyizhou/`, README in Chinese): all 72 saved volumes of the three topics re-scored with one
  scorer; every arm's head PSNR / SSIM / HFEN worsens at both acceleration steps except 3D-DIP's flat AF16->AF20; the same holds
  on the figure slices and on (almost) every slice.
- AF20 topic additions: `scripts/fig_mask_reinspect.py` and `figures/t1flair_v3_af20_mask_T1_FLAIR_reinspect.png` (T1 and FLAIR
  read the same mask file; 170 of its lines fall in FLAIR's unacquired corners).

## AF24 study (branch `autoresearch-jyz-af24`, this worktree, 2026-09-14) — complete (2026-09-15 05:30)
Cut from `jyz-af20-results` 4d13197; read `STATUS.md` first. Topic `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou/`
(mirror `t1flair_af24_v3_jiangyizhou/`, README). One variable again, the mask: T1 at exactly 24x under the user's AF definition
(1877 points), the house generator with accelx 6.8656, draw_025 of a 134-draw pool, untrimmed (md5 c3688d61...). Figures and tables
label both contrasts **AF 24.00x** = plane points / sampled points (user 2026-09-16); FLAIR's scan carries data on 1766 of those
points (25.51x on the per-contrast gauge of 09-14, 19.33x against its own acquisition).
MC-GS = the AF16 phase-3 recipe frozen in `run_af24_mcgs.sh`; every arm is queued by `scripts/run_all_af24.py` (priority queue,
one GPU job of this user per card, at most four at once, then finalize). The user approved the mask (`figures/t1flair_v3_af24_mask_*.png`) on 2026-09-14 before any GPU job. Result, nine arms: head PSNR
**33.767 / 33.182**, first on all six head cells over every comparison arm including GS-single. The margin that is quoted is over
the strongest comparison arm, GS-single (user 2026-09-13: GS-single counts as a comparison arm): **+0.179 / +0.071** (AF20 +0.165 / +0.052).
L1-ESPIRiT was dropped from the comparison on 2026-09-15 (user; data kept, not listed): the strongest non-GS arm is CS-SENSE (TV,
converged) 32.820 on T1 and joint CS-SENSE 32.605 on FLAIR. The earlier "+0.366 / +0.400 over L1-ESPIRiT" is history, not a margin to
quote. L1-SPIRiT (T1 31.783 / FLAIR 31.803) and MC-INR (30.076 / 30.947) were added at AF24 on 2026-09-15 to match AF12/16/20; the figure
set of record is `t1flair_af24_v3_jiangyizhou/figures/..._180_124_76_inset48_headerr_p3_mcinr_spirit.{png,pdf}`. Mask-draw variance (record §8):
margin over L1-ESPIRiT +0.359 ± 0.006 / +0.397 ± 0.006 over four draws (history framing; GS-single not run on the draws). Record `docs/superpowers/records/2026-09-15-af24-jyz.md`,
handoff `docs/handoff/2026-09-15-af24-jyz.md`, figures `t1flair_af24_v3_jiangyizhou/figures/t1flair_v3_af24_*_p3.*`.

## L1-SPIRiT arm across AF 12 / 16 / 20, the 2026-09-15 conventions, AF-label fix (claude-ad, 2026-09-14/16; on this branch since 13b7bce)
Read `STATUS.md` (section "L1-SPIRiT 臂…") then `docs/handoff/2026-09-16-spirit-arm-jyz-af12-16-20.md`. Per topic:
`scripts/recon_spirit.py` (k-space arm; `--solver fista`, the reference POCS diverges), `spirit_align_check.py` + `spirit_align.m`
(1e-12 agreement with the MATLAB original in double), `run_spirit_sweep.sh`; the topic READMEs end with the decision notes
(L1-ESPIRiT dropped from display, GS-single the strongest comparison arm, AF label = `AF_vs_full_rect`) and the archive note
(`figures/archive_aflabel_pre20260915/`). The delivered figures are the `_spirit` set. AF 24 data bundle:
`T1FLAIR_AF24_v3_jiangyizhou/export/af24_p3_mcinr_spirit/` (72 files, md5 72/72); Chinese summary `docs/2026-09-15-af24-jyz-summary.md`.


## Hyper-parameter sensitivity study (branch `hparam-sensitivity`, worktree `mcgs_hparam_wt`, 2026-09-16)
Cut from `origin/autoresearch-jyz-af24` 5ea6189. Purpose: a two-group sensitivity design (shared Gaussian count N_s;
private-to-shared ratio N_p/N_s) read against this code; nothing has run. Read `STATUS.md` (top section) first: the
knobs (`gaussian.initial_num_points`, `gaussian.num_private_points`), the AF16 capacity and private-residual points that
already exist, the four mismatches waiting on the user's decision (N_s scale, the private branch's status, the AF, the
lesion-CNR data), and the do-not-disturb rules (other worktrees' GPUs, `TORCH_EXTENSIONS_DIR`, outputs only in this
worktree's `results_arc/`).
User decisions 2026-09-16 (all four recommendations accepted): N_s 0.9M / 1.8M / 3.6M / 7.2M; the published MC-GS keeps
N_p = 0 and group 2 adds 5 % / 10 % FLAIR-private Gaussians on top; AF16 (the existing house mask); no lesion CNR.
Spec `docs/superpowers/specs/2026-09-16-hparam-sensitivity-design.md`, plan `docs/superpowers/plans/2026-09-16-hparam-sensitivity.md`.
Launcher `run_hp_mcgs.sh` (recipe block identical to `run_af24_mcgs.sh`, AF16 config), queue `scripts/hp_queue.sh` +
`scripts/hp_queue.txt`, summary `scripts/hp_summarize.py` (test `tests/test_hp_summarize.py`). Six runs hp_n09/n18/n36/n72/p05/p10;
hp_n36 is the regression anchor (gs_joint_p3 35.706 / 34.928, tolerance 0.02 dB). **Complete 2026-09-16 21:00**: anchor delta
0.000 / 0.000 (PASS); capacity 0.9M -> 3.6M monotone (T1 +0.349 / FLAIR +0.793 dB), 3.6M -> 7.2M a trade (T1 -0.036 / FLAIR +0.052);
private 5 % / 10 % null (at most +0.010 dB). Topic `Results_20260908/T1FLAIR_AF16_hparam_v3_jiangyizhou/` (mirror
`t1flair_af16_hparam_v3_jiangyizhou/`), record `docs/superpowers/records/2026-09-16-hparam-sensitivity.md`, hand-off in `STATUS.md`.
