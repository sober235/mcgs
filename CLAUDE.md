# This branch: cardiac-freerunning (cardiac line, phase 1)

Read in this order: this section -> `STATUS.md` -> the spec
`docs/superpowers/specs/2026-09-10-cardiac-freerunning-phase1-design.md` -> the plan
`docs/superpowers/plans/2026-09-10-cardiac-freerunning-phase1.md` -> the topic README
`/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/README.md`.

- Code: package `cardiac_fr/` with CPU tests `tests/test_cardiac_fr_*.py`; drivers live in the topic's
  `scripts/` and always run through `scripts/py` (GSTMI python, CPU only, nice 10, 32 BART threads).
  The topic is mirrored without volumes in `cardiac_freerunning_megre/` (`git add -f` for logs and figures).
- Isolation: never edit the main checkout (`Paired_ARC_GS_general_framework`) or `mcgs_af16_wt`; phase 1
  uses no GPU. Raw data is read-only; derived data lives in `/data2/congcong/data/cardiac_3d_cine_public/derived/megre/`.
- Tests: `cd /data0/congcong/code/GS/mcgs_cardiac_wt && /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py -m pytest tests/test_cardiac_fr_*.py -q` (34 tests).
- Seven corrections to the spec were approved during the run (polarity boxes and their 1-voxel floor, REF lambda relative
  to lambda_max, L1 lambda absolute, early-stopped FISTA, the 2x FOV grid, the self-check gate, per-subject orientation:
  V04 is a coronal slab); the topic README lists them with their evidence. Do not revert them.

- Phase 2 (branch `cardiac-phase2`, from the tag above, with `autoresearch-t1t2` merged): spec
  `docs/superpowers/specs/2026-09-12-cardiac-phase2-mcgs-design.md` (MC-GS and every comparison arm with a
  radial NUFFT forward). The trainer's default Cartesian path must stay byte-identical (spec §1.4).

The framework notes below come from `autoresearch-t1t2` and still describe the MC-GS code.

---

# Paired-ARC-GS / MC-GS framework — session entry

Read in this order: this file → `STATUS.md` (current state, what to re-run first) →
`autoresearch_t1t2.md` (the live campaign: program, verdicts, round log) →
`docs/superpowers/records/` (dated session records, newest first) → `.claude/memory.md`
(the long project memory, §0–19; do not re-derive anything it already closed).

## Environment
- Python: `/home/congcongliu/anaconda3/envs/GSTMI/bin/python` (torch 2.7.1+cu118), always with
  `PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- Tests: `PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests -q --ignore=tests/smoke_test.py` (183, ~2 min on an idle
  host, needs a GPU; add `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` when MATLAB loads the CPU, then ~45 min).
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
