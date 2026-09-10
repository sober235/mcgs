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
Read `docs/handoff/2026-09-11-autoresearch-jyz-af16.md` first, then `autoresearch_jyz_af16.md`.
