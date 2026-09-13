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
