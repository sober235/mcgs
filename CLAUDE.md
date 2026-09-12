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

## Current AF 12 result (user decision 2026-09-12, recipe updated by the phase-2 loop 2026-09-13)
The user fixed the current AF 12 result of this line as two compact three-plane figures — T1 with every arm at AF 12,
FLAIR in **design B** (MC-GS on the AF 11 mask `data/mask_af11_3d.h5`, every comparison arm at AF 12) — and made
design B the setting to tune in. The phase-2 loop then raised the recipe from P10 to **P13 = P10 +
`gaussian.rotation_lr = 0.008`** (runner `autoresearch_jyz_af12_P13.sh`; program `autoresearch_jyz_af12_p2.md`,
record `docs/superpowers/records/2026-09-13-autoresearch-jyz-af12-phase2.md`, handoff
`docs/handoff/2026-09-13-autoresearch-jyz-af12-phase2.md`). The current figures are P13's:
- T1, every arm at AF 12: `t1flair_af12_v3_jiangyizhou/figures/t1flair_v3_af12_T1_row3planes_compact_p2_174_110_47_headerr.png`
  (`http://localhost:8765/t1flair_v3_af12/t1flair_v3_af12_T1_row3planes_compact_p2_174_110_47_headerr.png`).
- FLAIR, design B: `..._FLAIR_row3planes_compact_p2_flair_af11_174_110_47_headerr.png` (same URL prefix).
Numbers (head PSNR / SSIM / HFEN): T1 **36.969 / 0.9504 / 0.0948** (all AF 12), FLAIR **36.683 / 0.9444 / 0.0904**
(AF 11 mask; 36.310 / 0.9401 / 0.0956 at AF 12). Margins over the best non-MC-GS arm at every swept λ: +0.543 /
+0.926 dB. First on every full-brain slice in every direction. The P10 pair (36.944 / 36.637) is the previous
reference and its figures are still in the topic. Everything is on the magnitude gauge (the complex-gauge caveat of
the P10 section stands).

## jiangyizhou AF12 autoresearch line (branch `autoresearch-jyz-af12`, this worktree)
Read `STATUS.md` first, then `autoresearch_jyz_af12.md` (program, pre-registered rules, round log) and the record
`docs/superpowers/records/2026-09-12-autoresearch-jyz-af12.md`. Update to the section above (2026-09-12): the loop
promoted **P10 = P6 + db4 wavelet on 1 level, T1 0.12 / FLAIR 0.10** — head PSNR 36.944 / 36.268 at 4000
iterations (P6 36.753 / 36.165), every head cell ahead of the re-swept L1-ESPIRiT; the runner bakes P10;
registered in the AF12 topic table as `gs_joint_ar`. The user's provisional-best note above stands as history.

## jiangyizhou AF12 autoresearch lines (branches `autoresearch-jyz-af12` and `-phase2`, this worktree)
Phase 1 (`autoresearch_jyz_af12.md`, handoff `docs/handoff/2026-09-12-autoresearch-jyz-af12.md`) promoted **P10** =
P6 + db4 wavelet on 1 level, T1 0.12 / FLAIR 0.10. Phase 2 (`autoresearch_jyz_af12_p2.md`, handoff
`docs/handoff/2026-09-13-autoresearch-jyz-af12-phase2.md`) tuned in design B and promoted **P13** = P10 +
`gaussian.rotation_lr=0.008`, the current recipe; runner `autoresearch_jyz_af12_P13.sh`. Ledger
`autoresearch_jyz_af12.tsv` carries both phases (`p2_` prefix for phase 2); reader `scripts/p2_read.py`.
