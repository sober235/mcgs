# This branch: autoresearch-gauss-af16 (tune MC-GS on the Gaussian-random mask, from jyz-af16-diffmask 889448a)

Read `autoresearch_gauss.md` first (objective, bar, parent, rules, round log), then `STATUS.md`. The loop's arena is the
gauss topic `Results_20260908/T1FLAIR_AF16_gauss_v3_jiangyizhou/` (read-only from here; its result tables are the
comparison bar, `logs/bar_gauss.json`). Runner `autoresearch_gauss.sh` (parent = the frozen AF16 phase-3 recipe on this
mask), ledger `autoresearch_gauss.tsv`, queue `scripts/gauss_queue.txt`, scheduler `scripts/gauss_sched.sh` (tmux
`gauss_sched`, empty cards only). Target (user 2026-09-16): MC-GS >= every comparison arm + 0.5 dB head PSNR on both
contrasts, GS-single at the frozen recipe counting as a comparison arm. Everything below this line is the diff-mask
branch's entry, kept as inherited context.

---

# This branch: jyz-af16-diffmask (mask-pattern study on the jiangyizhou T1+FLAIR pair, from autoresearch-jyz-af24 13b7bce)

Three alternative mask patterns at the AF16 topic's actual T1 acceleration (15.20x, 2964 points), labelled AF 16
(nominal): Gaussian variable-density random, a 2-D CAIPIRINHA lattice, and Cartesian pseudo-radial spokes. Topics
`Results_20260908/T1FLAIR_AF16_{gauss,caipi,radial}_v3_jiangyizhou/` (mirrored here as
`t1flair_af16_{gauss,caipi,radial}_v3_jiangyizhou/`, without volumes or masks). A literature check on 2026-09-15 put
**variable-density random (Lustig 2007) and the 2-D CAIPIRINHA lattice (Breuer 2006) as the other two standard 3-D head
patterns beside vd-Poisson disc**; radial phase encoding on the ky-kz plane is a real family but its home is cardiac /
dynamic imaging, so the radial topic is the extra arm, not one of the three. Generator
`scripts/make_masks_af16_diffmask.py` (seeded, regenerable; refuses to overwrite), figures `scripts/fig_masks_af16_diffmask.py`,
served at `http://localhost:8765/t1flair_v3_af16_diffmask/`. **Status: masks built 2026-09-15 17:21 for the user's
inspection; no reconstruction has run -- the user decides on the masks first.** The rest of this file is the AF24 line's
entry and still describes the framework, the recipe and the launcher this study will reuse.

---

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
(1877 points; FLAIR 25.51x), the house generator with accelx 6.8656, draw_025 of a 134-draw pool, untrimmed (md5 c3688d61...).
MC-GS = the AF16 phase-3 recipe frozen in `run_af24_mcgs.sh`; every arm is queued by `scripts/run_all_af24.py` (priority queue,
one GPU job of this user per card, at most four at once, then finalize). The user approved the mask (`figures/t1flair_v3_af24_mask_*.png`) on 2026-09-14 before any GPU job. Result, nine arms: head PSNR
**33.767 / 33.182**, first on all six head cells over every comparison arm including GS-single. The margin that is quoted is over
the strongest comparison arm, GS-single (user 2026-09-13: GS-single counts as a comparison arm): **+0.179 / +0.071** (AF20 +0.165 / +0.052).
L1-ESPIRiT was dropped from the comparison on 2026-09-15 (user; data kept, not listed): the strongest non-GS arm is CS-SENSE (TV,
converged) 32.820 on T1 and joint CS-SENSE 32.605 on FLAIR. The earlier "+0.366 / +0.400 over L1-ESPIRiT" is history, not a margin to
quote. L1-SPIRiT and MC-INR (added at AF12/16/20 on 2026-09-14/15) are being run at AF24; the figure set of record there is `_mcinr_spirit`. Record `docs/superpowers/records/2026-09-15-af24-jyz.md`,
handoff `docs/handoff/2026-09-15-af24-jyz.md`, figures `t1flair_af24_v3_jiangyizhou/figures/t1flair_v3_af24_*_p3.*`.
