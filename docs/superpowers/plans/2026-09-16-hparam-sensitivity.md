# MC-GS hyper-parameter sensitivity (N_s, N_p/N_s) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the six MC-GS sensitivity runs (shared count 0.9M/1.8M/3.6M/7.2M; private 0/5/10 % at 3.6M) on jiangyizhou at AF16 with the frozen phase-3 recipe, on free GPUs only, and produce one table plus a two-panel figure.

**Architecture:** A launcher (`run_hp_mcgs.sh`) whose recipe block is byte-identical to `run_af24_mcgs.sh` but points at the AF16 config and its own build cache; a lock-holding wrapper and a queue runner copied from the proven gauss-loop scheduler (shared `.gpulock.<card>` convention, one trainer per card); a summariser that reads the trainer's score jsons and the saved volumes into a table with an anchor check; a figure script for the two panels.

**Tech Stack:** bash (launch/queue), Python 3 in `/home/congcongliu/anaconda3/envs/GSTMI/bin/python` (torch 2.7.1+cu118, numpy, matplotlib, pytest), `arc/cycle.py` trainer, `diag_artefact.headonly` head mask.

**Spec:** `docs/superpowers/specs/2026-09-16-hparam-sensitivity-design.md`

## Global Constraints

- Every Python command: `env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID /home/congcongliu/anaconda3/envs/GSTMI/bin/python …` (the repo's CLAUDE.md rule).
- Run every multi-command shell step through `bash` (`bash <<'EOF' … EOF` or a script file); the tool shell is zsh and does not word-split `$VAR`.
- Experiment names start with `hp_`; outputs only under `/data0/congcong/code/GS/mcgs_hparam_wt/results_arc/` (gitignored); nothing is written into any other worktree or any existing `Results_20260908/` topic.
- GPUs: only a card that carries no compute process, uses < 1500 MiB and has no live `.gpulock.<card>` owner under `/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MCGS/work/`; default allow-list `1 2`; one trainer per card; launches at least 90 s apart; anything longer than minutes is launched with `setsid nohup … < /dev/null &` from a bash script.
- `TORCH_EXTENSIONS_DIR=/data0/congcong/code/GS/mcgs_hparam_wt.torch_ext` (never the AF24 worktree's cache).
- Commits: `git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit …`, plain human messages, no AI trailers, stage named files only.
- Recipe (must stay byte-identical to `run_af24_mcgs.sh` lines 51-60): `--lambda_geo 0.10 --warmup "$WARMUP" --max_radius 20 --kspace_high 2.2 --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn --set loss.edge_tv_kappa_source=zf --set gaussian.initial_num_points=3600000 --set loss.wav_name=db4 --set loss.wav_levels=1 --set loss.lambda_wav_plain=0.12 --set loss.lambda_wav=0.10 --lambda_edge 0.001 --lambda_ptv 0.03`, WARMUP default 500, ITERS 4000, base config `configs/t1flair_v3_af16_jyz.yaml`.
- Anchor: hp_n36 head PSNR within 0.02 dB of 35.706 (T1) / 34.928 (FLAIR).

---

## File structure

| file | responsibility |
|---|---|
| `run_hp_mcgs.sh` (create) | one MC-GS run: prefix guard, overwrite guard, recipe block, own build cache, one line into `results_arc/hp_runs.log` |
| `scripts/hp_launch_one.sh` (create) | holds the shared card lock for the run's lifetime, then calls `run_hp_mcgs.sh` |
| `scripts/hp_queue.sh` (create) | queue runner: free-card rule, skip items with outputs, pause file, exits when the queue is drained; `HP_DRY=1` / `HP_ONCE=1` for tests |
| `scripts/hp_queue.txt` (create) | the six runs, anchor first |
| `scripts/hp_summarize.py` (create) | score jsons + volumes -> `results_arc/hp_summary.{md,csv}`, anchor check |
| `tests/test_hp_summarize.py` (create) | unit test on synthetic jsons / npz (no GPU) |
| `scripts/hp_fig_curves.py` (create, Task 5) | the two-panel figure from `hp_summary.csv` |
| `STATUS.md`, `CLAUDE.md` (modify) | queue state and controls; entry pointer |

---

### Task 1: Launcher and lock wrapper, verified by a smoke run

**Files:**
- Create: `run_hp_mcgs.sh`
- Create: `scripts/hp_launch_one.sh`
- Reference: `run_af24_mcgs.sh` (recipe block), `/data0/congcong/code/GS/mcgs_gauss_wt/scripts/gauss_launch_one.sh` (lock pattern)

**Interfaces:**
- Produces: `run_hp_mcgs.sh EXP GPU ITERS [extra cycle.py args]` (env `WARMUP`, `SAVE_VOLUMES=1`), outputs `results_arc/EXP.log`, `results_arc/EXP_score.json`, `results_arc/EXP/` (checkpoints), `results_arc/EXP_volumes.npz` when `SAVE_VOLUMES=1`, one line in `results_arc/hp_runs.log`.
- Produces: `scripts/hp_launch_one.sh EXP CARD ITERS [KEY=VAL …] [-- extra args]` — same outputs, lock held at `…/work/.gpulock.CARD` with `pid` = the wrapper.

- [ ] **Step 1: Write `run_hp_mcgs.sh`**

```bash
#!/usr/bin/env bash
# One MC-GS run for the hyper-parameter sensitivity study (spec docs/superpowers/specs/2026-09-16-hparam-sensitivity-design.md):
# jiangyizhou T1+FLAIR at nominal AF 16 (the house mask of T1FLAIR_AF16_v3_jiangyizhou, T1 15.19x / FLAIR 16.49x).
#   run_hp_mcgs.sh EXP GPU ITERS [extra arc/cycle.py args...]
#   SAVE_VOLUMES=1 dumps the recon npz.  WARMUP (default 500) for short smokes.
#
# The recipe block below is byte-identical to run_af24_mcgs.sh (the AF16 phase-3 recipe, frozen): P6 cold start, 3.6M points,
# edge-TV lambda_geo 0.10 / kappa_q 0.85 / wfloor 0.10 / eps 0.04, kspace_high 2.2, warmup 500, wavelet db4 finest level
# (T1 0.12 / FLAIR 0.10), common-edge 0.001, phase TV 0.03. The study varies ONLY what the caller appends:
#   --set gaussian.initial_num_points=N     (shared count; --set applies in argv order, the appended value wins)
#   --set gaussian.num_private_points=N     (FLAIR-private residual set; 0 = the published recipe)
set -uo pipefail
if [[ $# -lt 3 ]]; then echo "usage: $0 EXP GPU ITERS [extra cycle.py args...]" >&2; exit 2; fi
EXP=$1; GPU=$2; ITERS=$3; shift 3
if [[ "$EXP" != hp_* ]]; then
  echo "EXP must start with hp_ (keeps this study apart from the AF12/16/20/24 and diff-mask lines): $EXP" >&2; exit 2
fi
WARMUP=${WARMUP:-500}
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
CONFIG=configs/t1flair_v3_af16_jyz.yaml
LOG="$ROOT/results_arc/${EXP}.log"
for path in "$LOG" "$ROOT/results_arc/${EXP}_score.json" "$ROOT/results_arc/${EXP}"; do
  if [[ -e "$path" ]]; then echo "refusing to overwrite existing output: $path" >&2; exit 1; fi
done
VOL=()
if [[ "${SAVE_VOLUMES:-0}" == "1" ]]; then VOL=(--save_volumes); fi
mkdir -p "$ROOT/results_arc"
cd "$ROOT"
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_EXTENSIONS_DIR=/data0/congcong/code/GS/mcgs_hparam_wt.torch_ext \
  "$PY" -u arc/cycle.py --exp "$EXP" --gpu "$GPU" --iters "$ITERS" \
  --lambda_geo 0.10 --warmup "$WARMUP" --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn \
  --set loss.edge_tv_kappa_source=zf \
  --set gaussian.initial_num_points=3600000 \
  --set loss.wav_name=db4 --set loss.wav_levels=1 \
  --set loss.lambda_wav_plain=0.12 --set loss.lambda_wav=0.10 \
  --lambda_edge 0.001 --lambda_ptv 0.03 \
  --base_cfg "$CONFIG" "${VOL[@]}" "$@" > "$LOG" 2>&1
RC=$?
echo "[$(date '+%F %H:%M')] $EXP rc=$RC  $(grep -o 'head_psnr[^,]*' "$ROOT/results_arc/${EXP}_score.json" 2>/dev/null | head -2 | tr '\n' ' ')" >> "$ROOT/results_arc/hp_runs.log"
exit $RC
```

- [ ] **Step 2: Guards and the recipe-block identity check**

Run (bash):
```bash
cd /data0/congcong/code/GS/mcgs_hparam_wt && chmod +x run_hp_mcgs.sh && bash -n run_hp_mcgs.sh && echo SYNTAX_OK
./run_hp_mcgs.sh bad_x 1 60; echo "prefix guard rc=$? (expect 2)"
diff <(sed -n '/"\$PY" -u arc\/cycle.py/,/--base_cfg/p' run_af24_mcgs.sh) <(sed -n '/"\$PY" -u arc\/cycle.py/,/--base_cfg/p' run_hp_mcgs.sh) && echo RECIPE_BLOCK_IDENTICAL
```
Expected: `SYNTAX_OK`, `prefix guard rc=2`, `RECIPE_BLOCK_IDENTICAL` (no diff lines).

- [ ] **Step 3: Write `scripts/hp_launch_one.sh`**

```bash
#!/usr/bin/env bash
# hp_launch_one.sh EXP CARD ITERS [KEY=VAL ...] [-- extra args...]
# Runs one run_hp_mcgs.sh experiment on CARD while holding the shared card lock
# <cardiac topic>/work/.gpulock.<CARD> (mkdir-atomic, pid file = this wrapper), which the gauss-loop, diff-mask and
# cardiac schedulers also honour; the lock is removed when the run ends, whatever the exit code.
set -uo pipefail
cd "$(dirname "$0")/.."
EXP=$1; CARD=$2; IT=$3; shift 3
ENVS=()
while [[ $# -gt 0 && "$1" == *=* && "$1" != -* ]]; do ENVS+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift
LOCK=/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MCGS/work/.gpulock.$CARD
if ! mkdir "$LOCK" 2>/dev/null; then
  owner=$(cat "$LOCK/pid" 2>/dev/null)
  if [[ -n "$owner" ]] && kill -0 "$owner" 2>/dev/null; then echo "card $CARD locked by live pid $owner" >&2; exit 3; fi
  rm -rf "$LOCK"; mkdir "$LOCK" || exit 3
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT
env "${ENVS[@]}" ./run_hp_mcgs.sh "$EXP" "$CARD" "$IT" "$@"
```

- [ ] **Step 4: Smoke run 1 (0.9M shared, 60 iterations) on a free card, detached**

Pick the card: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` must show < 1500 MiB and `nvidia-smi --query-compute-apps=pid -i CARD` nothing; the lock dir must have no live owner. Then (bash):
```bash
cd /data0/congcong/code/GS/mcgs_hparam_wt && chmod +x scripts/hp_launch_one.sh
setsid nohup bash scripts/hp_launch_one.sh hp_smoke_n09 1 60 WARMUP=20 -- --set gaussian.initial_num_points=900000 > results_arc/hp_smoke_n09.launcher.log 2>&1 < /dev/null &
```
Poll `results_arc/hp_smoke_n09.log` until `SCORE` appears (the first run also builds the CUDA voxelizer into the new `TORCH_EXTENSIONS_DIR`, a few minutes). Expected lines: `[arc hp_smoke_n09] gpu=1 iters=60`, `SCORE … N=900000`, `DONE`; `results_arc/hp_smoke_n09_score.json` exists; `hp_runs.log` has one line with `rc=0`; the lock dir `.gpulock.1` is gone after the run.

- [ ] **Step 5: Smoke run 2 (private set attached)**

```bash
cd /data0/congcong/code/GS/mcgs_hparam_wt
setsid nohup bash scripts/hp_launch_one.sh hp_smoke_p05 1 60 WARMUP=20 -- --set gaussian.initial_num_points=900000 --set gaussian.num_private_points=45000 > results_arc/hp_smoke_p05.launcher.log 2>&1 < /dev/null &
```
Expected in `results_arc/hp_smoke_p05.log`: `[FLAIR-private] attached 45000 private residual Gaussians (init shared stride-subset, factor 0.1)` and `SCORE … N=900000` (N counts the shared cloud only). Smoke numbers are feasibility checks, not results — never quote them.

- [ ] **Step 6: Commit**

```bash
cd /data0/congcong/code/GS/mcgs_hparam_wt && git add run_hp_mcgs.sh scripts/hp_launch_one.sh
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "hparam-sensitivity: launcher (recipe block identical to run_af24_mcgs.sh, AF16 config, own build cache, hp_ prefix) and the lock-holding wrapper; both smoke-tested at 60 iterations"
```

---

### Task 2: Queue runner and queue file

**Files:**
- Create: `scripts/hp_queue.sh`
- Create: `scripts/hp_queue.txt`
- Reference: `/data0/congcong/code/GS/mcgs_gauss_wt/scripts/gauss_sched.sh`

**Interfaces:**
- Consumes: `scripts/hp_launch_one.sh EXP CARD ITERS [KEY=VAL …] -- extra` (Task 1).
- Produces: `bash scripts/hp_queue.sh scripts/hp_queue.txt` (env `ALLOWED` default `1 2`, `HP_DRY=1` prints instead of launching, `HP_ONCE=1` single pass), log lines on stdout, per-item `results_arc/EXP.launcher.log`; exits 0 when every queue item has outputs.

- [ ] **Step 1: Write `scripts/hp_queue.txt`**

```
# hyper-parameter sensitivity queue (spec docs/superpowers/specs/2026-09-16-hparam-sensitivity-design.md)
# format: [KEY=VAL ...] EXP ITERS [extra run_hp_mcgs.sh args]; consumed by scripts/hp_queue.sh, top first.
# anchor first (repeats the registered gs_joint_p3: head PSNR 35.706 / 34.928), then the capacity axis, then the private axis.
SAVE_VOLUMES=1 hp_n36 4000 --set gaussian.initial_num_points=3600000
SAVE_VOLUMES=1 hp_n18 4000 --set gaussian.initial_num_points=1800000
SAVE_VOLUMES=1 hp_n09 4000 --set gaussian.initial_num_points=900000
SAVE_VOLUMES=1 hp_n72 4000 --set gaussian.initial_num_points=7200000
SAVE_VOLUMES=1 hp_p05 4000 --set gaussian.initial_num_points=3600000 --set gaussian.num_private_points=180000
SAVE_VOLUMES=1 hp_p10 4000 --set gaussian.initial_num_points=3600000 --set gaussian.num_private_points=360000
```

- [ ] **Step 2: Write `scripts/hp_queue.sh`**

```bash
#!/usr/bin/env bash
# hp_queue.sh QUEUE_FILE -- queue runner for the hyper-parameter sensitivity study; only ever uses an EMPTY card.
# Queue lines: "[KEY=VAL ...] EXP ITERS [extra run_hp_mcgs.sh args...]" (blank lines and # comments ignored; re-read on
# every pass, top first). A card is usable when it is in ALLOWED (default "1 2"), carries no compute process, uses
# < 1500 MiB, has no live card lock of another scheduler (<cardiac topic>/work/.gpulock.<card> with a live owner) and
# no trainer of this study. One trainer per card, launches at least 90 s apart. Items whose score json, log or
# launcher log already exists are skipped (a crashed item is not retried automatically). Touch
# results_arc/.hp_queue_pause to hold launches. Exits 0 once every item has outputs.
# Tests: HP_DRY=1 prints the launches instead of starting them (a dry-launched card is taken out of ALLOWED for the
# pass); HP_ONCE=1 makes a single pass over the queue and exits.
cd "$(dirname "$0")/.."
Q=${1:?queue file}
ALLOWED=${ALLOWED:-"1 2"}
PYPAT='^/home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py --exp hp_'
LOCKD=/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MCGS/work
log() { echo "$(date '+%F %T') $*"; }
free_card() {
  for g in $ALLOWED; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [[ -z "$used" ]] && continue
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null | grep -c .)
    mine=$(pgrep -f "$PYPAT.* --gpu $g " | grep -c .)
    locked=0
    if [[ -d "$LOCKD/.gpulock.$g" ]]; then
      owner=$(cat "$LOCKD/.gpulock.$g/pid" 2>/dev/null)
      if [[ -n "$owner" ]] && kill -0 "$owner" 2>/dev/null; then locked=1; fi
    fi
    if [[ "$used" -lt 1500 && "$procs" -eq 0 && "$mine" -eq 0 && "$locked" -eq 0 ]]; then echo "$g"; return 0; fi
  done
  return 1
}
log "queue runner up, queue=$Q allowed=[$ALLOWED] dry=${HP_DRY:-0}"
while true; do
  pending=0
  while read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    set -- $line
    ENVS=""
    while [[ "$1" == *=* && "$1" != -* ]]; do ENVS="$ENVS $1"; shift; done
    EXP=$1; IT=$2; shift 2; EXTRA="$*"
    [[ "$EXP" != hp_* ]] && { log "bad queue line (not hp_): $line"; continue; }
    if [[ -e "results_arc/${EXP}_score.json" || -e "results_arc/${EXP}.log" || -e "results_arc/${EXP}.launcher.log" ]]; then continue; fi
    pending=$((pending + 1))
    [[ -e results_arc/.hp_queue_pause ]] && { log "paused (results_arc/.hp_queue_pause exists)"; break; }
    card=$(free_card) || { log "no free card in [$ALLOWED]"; break; }
    if [[ "${HP_DRY:-0}" == "1" ]]; then
      log "DRY launch $EXP on card $card ($IT it)${ENVS:+ env:$ENVS} $EXTRA"
      ALLOWED=$(echo " $ALLOWED " | sed "s/ $card / /" | xargs)
      pending=$((pending - 1))
      continue
    fi
    log "launch $EXP on card $card ($IT it)${ENVS:+ env:$ENVS} $EXTRA"
    # detached (setsid + nohup): the run must not depend on this runner's process tree
    setsid nohup bash scripts/hp_launch_one.sh $EXP $card $IT $ENVS -- $EXTRA > results_arc/${EXP}.launcher.log 2>&1 < /dev/null &
    pending=$((pending - 1))
    sleep 90
  done < "$Q"
  if [[ "${HP_ONCE:-0}" == "1" ]]; then log "single pass done, pending=$pending"; exit 0; fi
  # drained = every item has outputs AND no trainer of this study is still running
  running=$(pgrep -f "$PYPAT" | grep -c .)
  if [[ "$pending" -eq 0 && "$running" -eq 0 ]]; then log "queue drained"; exit 0; fi
  sleep 60
done
```

- [ ] **Step 3: Dry-run test (no GPU job started)**

```bash
cd /data0/congcong/code/GS/mcgs_hparam_wt && chmod +x scripts/hp_queue.sh && bash -n scripts/hp_queue.sh && echo SYNTAX_OK
HP_DRY=1 HP_ONCE=1 ALLOWED="1 2" bash scripts/hp_queue.sh scripts/hp_queue.txt
```
Expected (when cards 1 and 2 are free): `DRY launch hp_n36 on card 1 (4000 it) env: SAVE_VOLUMES=1 --set gaussian.initial_num_points=3600000`, then `DRY launch hp_n18 on card 2 …`, then `no free card in []`, then `single pass done, pending=1` (the count stops at the first item that cannot be launched; it only has to be non-zero for the drained check). Then the skip rule:
```bash
cd /data0/congcong/code/GS/mcgs_hparam_wt && touch results_arc/hp_n36.launcher.log
HP_DRY=1 HP_ONCE=1 ALLOWED="1 2" bash scripts/hp_queue.sh scripts/hp_queue.txt | grep -c "DRY launch hp_n18\|DRY launch hp_n09"
rm results_arc/hp_n36.launcher.log
```
Expected: `2` (hp_n36 skipped, hp_n18 and hp_n09 dry-launched). The `rm` is mandatory — a stale launcher log would make the real queue skip the anchor.

- [ ] **Step 4: Commit**

```bash
cd /data0/congcong/code/GS/mcgs_hparam_wt && git add scripts/hp_queue.sh scripts/hp_queue.txt
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "hparam-sensitivity: queue runner (gauss-loop scheduler pattern: empty cards only, shared card lock, pause file, exits when drained; dry-run and single-pass switches) and the six-run queue, anchor first"
```

---

### Task 3: Summariser with a unit test

**Files:**
- Create: `scripts/hp_summarize.py`
- Test: `tests/test_hp_summarize.py`
- Reference: `diag_artefact/headonly.py` (`head_mask`), `arc/cycle.py:295-317` (score json layout: top-level `npts`, `elapsed_s`, `vram_gb`, per-contrast dicts `plain` (T1) and `contrast` (FLAIR) with `head_psnr`, `head_ssim`, `head_hfen`, `psnr`, `ssim`, `nmse`, `head_psnr_cplx`)

**Interfaces:**
- Produces: `RUNS: list[tuple[str, int, int]]` (exp, n_shared, n_private); `head_nmse(rec, gt) -> float`; `load_run(arc: Path, exp: str) -> dict | None`; `build_rows(arc: Path, runs=RUNS) -> list[dict]`; `anchor_check(rows, anchor: str, ref: tuple[float, float], tol: float) -> dict`; `render_md(rows, check) -> str`; `write_csv(rows, path: Path) -> None`. Row keys: `exp, n_shared, n_private, ratio, status, npts, elapsed_h, vram_gb, T1_head_psnr, T1_head_ssim, T1_head_hfen, T1_head_nmse, T1_psnr, T1_ssim, T1_nmse, T1_head_psnr_cplx, FLAIR_* (same eight)`.
- Consumed by Task 5 (`hp_fig_curves.py` reads `results_arc/hp_summary.csv`).

- [ ] **Step 1: Write the failing test `tests/test_hp_summarize.py`**

```python
#!/usr/bin/env python3
"""Unit test for scripts/hp_summarize.py on synthetic score jsons and a tiny volumes npz (no GPU).
Run: PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_hp_summarize.py -q"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hp_summarize as hs  # noqa: E402


def _score(t1_psnr, fl_psnr, npts=3600000):
    def c(p):
        return {"head_psnr": p, "head_ssim": 0.93, "head_hfen": 0.11, "psnr": p + 3.5, "ssim": 0.92,
                "nmse": 0.004, "head_psnr_cplx": p - 5.0}
    return {"npts": npts, "elapsed_s": 25200.0, "vram_gb": 33.0, "plain": c(t1_psnr), "contrast": c(fl_psnr)}


def _volumes(path):
    rng = np.random.default_rng(0)
    gt = np.zeros((12, 12, 12), np.float32)
    gt[3:9, 3:9, 3:9] = 1.0                      # a bright cube = the "head"
    rec = gt + rng.normal(0, 0.01, gt.shape).astype(np.float32)
    np.savez(path, recT1=rec, gtT1=gt, recFL=rec, gtFL=gt)
    return rec, gt


def test_head_nmse_matches_direct_formula(tmp_path):
    rec, gt = _volumes(tmp_path / "v.npz")
    head = gt > 0.05 * np.quantile(gt, 0.999)
    direct = float(((rec - gt)[head] ** 2).sum() / (gt[head] ** 2).sum())
    assert abs(hs.head_nmse(rec, gt) - direct) < 1e-9
    assert 0.0 < direct < 1e-2


def test_rows_order_ratio_and_missing(tmp_path):
    arc = tmp_path
    (arc / "hp_n36_score.json").write_text(json.dumps(_score(35.70, 34.93)))
    (arc / "hp_n18_score.json").write_text(json.dumps(_score(35.40, 34.60, npts=1800000)))
    (arc / "hp_p05_score.json").write_text(json.dumps(_score(35.71, 34.94)))
    _volumes(arc / "hp_n36_volumes.npz")
    rows = hs.build_rows(arc)
    assert [r["exp"] for r in rows] == ["hp_n09", "hp_n18", "hp_n36", "hp_n72", "hp_p05", "hp_p10"]
    by = {r["exp"]: r for r in rows}
    assert by["hp_n09"]["status"] == "missing" and by["hp_n36"]["status"] == "done"
    assert by["hp_p05"]["ratio"] == 0.05 and by["hp_p10"]["ratio"] == 0.10 and by["hp_n36"]["ratio"] == 0.0
    assert by["hp_n36"]["T1_head_psnr"] == 35.70 and by["hp_n36"]["FLAIR_head_psnr"] == 34.93
    assert np.isfinite(by["hp_n36"]["T1_head_nmse"]) and np.isnan(by["hp_n18"]["T1_head_nmse"])
    assert abs(by["hp_n36"]["elapsed_h"] - 7.0) < 1e-9


def test_anchor_check_and_rendering(tmp_path):
    arc = tmp_path
    (arc / "hp_n36_score.json").write_text(json.dumps(_score(35.70, 34.93)))
    rows = hs.build_rows(arc)
    ok = hs.anchor_check(rows, "hp_n36", (35.706, 34.928), 0.02)
    assert ok["passed"] is True and abs(ok["delta"][0] - (-0.006)) < 1e-9
    bad = hs.anchor_check(rows, "hp_n36", (35.80, 34.928), 0.02)
    assert bad["passed"] is False
    md = hs.render_md(rows, ok)
    assert "hp_n36" in md and "anchor" in md.lower() and "PASS" in md
    hs.write_csv(rows, arc / "hp_summary.csv")
    head = (arc / "hp_summary.csv").read_text().splitlines()[0]
    assert head.startswith("exp,n_shared,n_private,ratio,status")
```

- [ ] **Step 2: Run the test to verify it fails**

Run (bash): `cd /data0/congcong/code/GS/mcgs_hparam_wt && env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_hp_summarize.py -q`
Expected: FAIL / error at import: `ModuleNotFoundError: No module named 'hp_summarize'`.

- [ ] **Step 3: Write `scripts/hp_summarize.py`**

```python
#!/usr/bin/env python3
"""hp_summarize.py -- one table for the hyper-parameter sensitivity runs
(spec docs/superpowers/specs/2026-09-16-hparam-sensitivity-design.md).

Input : results_arc/<exp>_score.json (arc/cycle.py) for the six runs; <exp>_volumes.npz when present (head NMSE on the
        scorer's head mask |GT| > 0.05 P99.9|GT|, diag_artefact.headonly.head_mask -- the json has no head_nmse).
Output: results_arc/hp_summary.md and hp_summary.csv; the anchor check of hp_n36 against the registered gs_joint_p3.
usage: hp_summarize.py [--arc results_arc] [--anchor hp_n36] [--anchor_ref 35.706 34.928] [--tol 0.02]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diag_artefact.headonly import head_mask  # noqa: E402

# exp, shared count, private count (spec section 4)
RUNS = [
    ("hp_n09", 900_000, 0),
    ("hp_n18", 1_800_000, 0),
    ("hp_n36", 3_600_000, 0),
    ("hp_n72", 7_200_000, 0),
    ("hp_p05", 3_600_000, 180_000),
    ("hp_p10", 3_600_000, 360_000),
]
CONTRASTS = (("T1", "plain", "recT1", "gtT1"), ("FLAIR", "contrast", "recFL", "gtFL"))
JSON_KEYS = ("head_psnr", "head_ssim", "head_hfen", "psnr", "ssim", "nmse", "head_psnr_cplx")


def head_nmse(rec: np.ndarray, gt: np.ndarray) -> float:
    rec = np.asarray(rec, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    head = head_mask(gt.astype(np.float32))
    den = float((gt[head] ** 2).sum())
    if den == 0.0:
        return float("inf")
    return float(((rec - gt)[head] ** 2).sum() / den)


def load_run(arc: Path, exp: str) -> dict | None:
    p = arc / f"{exp}_score.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    out = {"npts": int(d.get("npts", 0)), "elapsed_h": float(d.get("elapsed_s", float("nan"))) / 3600.0,
           "vram_gb": float(d.get("vram_gb", float("nan")))}
    vol = arc / f"{exp}_volumes.npz"
    z = np.load(vol) if vol.exists() else None
    for name, key, rk, gk in CONTRASTS:
        c = d.get(key, {})
        for k in JSON_KEYS:
            out[f"{name}_{k}"] = float(c[k]) if k in c else float("nan")
        out[f"{name}_head_nmse"] = head_nmse(z[rk], z[gk]) if z is not None and rk in z.files else float("nan")
    return out


def build_rows(arc: Path, runs=RUNS) -> list[dict]:
    rows = []
    for exp, n_shared, n_private in runs:
        row = {"exp": exp, "n_shared": n_shared, "n_private": n_private, "ratio": n_private / n_shared}
        got = load_run(Path(arc), exp)
        row["status"] = "done" if got else "missing"
        if got is None:
            got = {"npts": 0, "elapsed_h": float("nan"), "vram_gb": float("nan")}
            for name, *_ in CONTRASTS:
                for k in JSON_KEYS + ("head_nmse",):
                    got[f"{name}_{k}"] = float("nan")
        row.update(got)
        rows.append(row)
    return rows


def anchor_check(rows: list[dict], anchor: str, ref: tuple[float, float], tol: float) -> dict:
    row = next((r for r in rows if r["exp"] == anchor), None)
    if row is None or row["status"] != "done":
        return {"anchor": anchor, "ref": ref, "tol": tol, "delta": (float("nan"), float("nan")), "passed": None}
    delta = (row["T1_head_psnr"] - ref[0], row["FLAIR_head_psnr"] - ref[1])
    return {"anchor": anchor, "ref": ref, "tol": tol, "delta": delta,
            "passed": bool(all(abs(x) <= tol for x in delta))}


def _f(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render_md(rows: list[dict], check: dict) -> str:
    lines = ["| exp | N_s | N_p | N_p/N_s | T1 head PSNR / SSIM / HFEN / NMSE | FLAIR head PSNR / SSIM / HFEN / NMSE | "
             "T1 whole PSNR / NMSE | FLAIR whole PSNR / NMSE | cplx T1 / FLAIR | h | GB |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] != "done":
            lines.append(f"| {r['exp']} | {r['n_shared']/1e6:.1f}M | {r['n_private']/1e3:.0f}k | {r['ratio']:.0%} | (missing) | | | | | | |")
            continue
        cell = lambda c: f"{_f(r[c+'_head_psnr'])} / {_f(r[c+'_head_ssim'], 4)} / {_f(r[c+'_head_hfen'], 4)} / {_f(r[c+'_head_nmse'], 5)}"
        whole = lambda c: f"{_f(r[c+'_psnr'])} / {_f(r[c+'_nmse'], 5)}"
        lines.append(f"| {r['exp']} | {r['n_shared']/1e6:.1f}M | {r['n_private']/1e3:.0f}k | {r['ratio']:.0%} | {cell('T1')} | "
                     f"{cell('FLAIR')} | {whole('T1')} | {whole('FLAIR')} | {_f(r['T1_head_psnr_cplx'])} / "
                     f"{_f(r['FLAIR_head_psnr_cplx'])} | {_f(r['elapsed_h'], 1)} | {_f(r['vram_gb'], 1)} |")
    base = next((r for r in rows if r["exp"] == check["anchor"] and r["status"] == "done"), None)
    if base is not None:
        lines.append("")
        lines.append(f"Deltas against {check['anchor']} (head PSNR T1 / FLAIR, dB):")
        for r in rows:
            if r["status"] == "done" and r["exp"] != check["anchor"]:
                lines.append(f"- {r['exp']}: {r['T1_head_psnr'] - base['T1_head_psnr']:+.3f} / "
                             f"{r['FLAIR_head_psnr'] - base['FLAIR_head_psnr']:+.3f}")
    verdict = {True: "PASS", False: "FAIL", None: "not run yet"}[check["passed"]]
    lines.append("")
    lines.append(f"Anchor {check['anchor']} vs registered gs_joint_p3 {check['ref'][0]:.3f} / {check['ref'][1]:.3f}: "
                 f"delta {_f(check['delta'][0])} / {_f(check['delta'][1])} dB, tolerance {check['tol']:.2f} -> {verdict}")
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict], path: Path) -> None:
    cols = ["exp", "n_shared", "n_private", "ratio", "status", "npts", "elapsed_h", "vram_gb"]
    for name, *_ in CONTRASTS:
        cols += [f"{name}_{k}" for k in JSON_KEYS + ("head_nmse",)]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc", default=str(ROOT / "results_arc"))
    ap.add_argument("--anchor", default="hp_n36")
    ap.add_argument("--anchor_ref", nargs=2, type=float, default=[35.706, 34.928])
    ap.add_argument("--tol", type=float, default=0.02)
    a = ap.parse_args(argv)
    arc = Path(a.arc)
    rows = build_rows(arc)
    check = anchor_check(rows, a.anchor, tuple(a.anchor_ref), a.tol)
    md = render_md(rows, check)
    (arc / "hp_summary.md").write_text(md)
    write_csv(rows, arc / "hp_summary.csv")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run (bash): `cd /data0/congcong/code/GS/mcgs_hparam_wt && env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_hp_summarize.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Run the summariser on the empty results dir (all six "missing", anchor "not run yet")**

Run (bash): `cd /data0/congcong/code/GS/mcgs_hparam_wt && env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/hp_summarize.py | tail -3`
Expected: the anchor line ends with `-> not run yet`; `results_arc/hp_summary.md` and `.csv` exist.

- [ ] **Step 6: Commit**

```bash
cd /data0/congcong/code/GS/mcgs_hparam_wt && git add scripts/hp_summarize.py tests/test_hp_summarize.py
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "hparam-sensitivity: summariser (score jsons + saved volumes -> table with head NMSE on the scorer's head mask, deltas against hp_n36, anchor gate against gs_joint_p3) with a unit test on synthetic inputs"
```

---

### Task 4: Launch the queue, record the state, hand off

**Files:**
- Modify: `STATUS.md` (top section, below the "### 下一步" heading of the branch section)
- Modify: `CLAUDE.md` (the branch's section)

**Interfaces:**
- Consumes: `scripts/hp_queue.sh`, `scripts/hp_queue.txt` (Task 2).

- [ ] **Step 1: Confirm the cards are free right now**

Run (bash):
```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '2,3p'
for g in 1 2; do echo "card $g procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g | grep -c .)"; done
ls -d /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MCGS/work/.gpulock.[12] 2>/dev/null || echo "no lock on 1/2"
```
Expected: cards 1 and 2 < 1500 MiB, 0 procs, no lock (or a lock whose `pid` is dead). If either card is busy, launch anyway — the runner waits for a free card.

- [ ] **Step 2: Launch the runner detached**

```bash
cd /data0/congcong/code/GS/mcgs_hparam_wt
setsid nohup bash scripts/hp_queue.sh scripts/hp_queue.txt > results_arc/hp_queue.log 2>&1 < /dev/null &
sleep 100; cat results_arc/hp_queue.log; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 1
```
Expected: `launch hp_n36 on card 1 (4000 it) env: SAVE_VOLUMES=1 …` then (90 s later) `launch hp_n18 on card 2 …`; card 1 shows one python process; `results_arc/hp_n36.log` grows past `[arc hp_n36] gpu=1 iters=4000`.

- [ ] **Step 3: Record the queue state in `STATUS.md`**

Replace the branch section's "### 下一步" paragraph with:
```markdown
### 队列已起跑（2026-09-16，Task 4）
- 运行器 `scripts/hp_queue.sh scripts/hp_queue.txt`（`setsid nohup`，日志 `results_arc/hp_queue.log`）；六个运行按 hp_n36 → hp_n18 → hp_n09 → hp_n72 → hp_p05 → hp_p10 排队，
  只用空卡（默认卡 1、2），一卡一训练，持有 `Cardiac_FreeRunning_MCGS/work/.gpulock.<卡>`。每个运行 6–7.5 h（7.2M 更久），全部约一天。
- 暂停：`touch results_arc/.hp_queue_pause`；停运行器：`pkill -f "scripts/hp_queue.sh"`（训练进程独立，不会跟着死）；停一个训练：
  `pkill -f "cycle.py --exp hp_n72"`（之后删掉它的 `results_arc/hp_n72*` 才会重排）。
- 进度：`tail -2 results_arc/hp_runs.log`；汇总：`env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/hp_summarize.py`（`results_arc/hp_summary.md`），
  先看 hp_n36 的 anchor 行是否 PASS（对 35.706 / 34.928 容差 0.02 dB）。
- 冒烟 `hp_smoke_n09` / `hp_smoke_p05`（60 步）只是可行性检查，不是结果，别引用。

### 跑完以后（下一会话，plan Task 5）
汇总表 → anchor 门 → 双面板图（先读 dataviz skill）→ 课题目录 `Results_20260908/T1FLAIR_AF16_hparam_v3_jiangyizhou/` → 记录
`docs/superpowers/records/2026-09-16-hparam-sensitivity.md` → 提交、推送、打 tag `handoff/<日期>-hparam`。
```

- [ ] **Step 4: Update the branch section of `CLAUDE.md`**

Append to that section:
```markdown
Spec `docs/superpowers/specs/2026-09-16-hparam-sensitivity-design.md`, plan `docs/superpowers/plans/2026-09-16-hparam-sensitivity.md`.
Launcher `run_hp_mcgs.sh` (recipe block identical to `run_af24_mcgs.sh`, AF16 config), queue `scripts/hp_queue.sh` +
`scripts/hp_queue.txt`, summary `scripts/hp_summarize.py` (test `tests/test_hp_summarize.py`). Six runs hp_n09/n18/n36/n72/p05/p10;
hp_n36 is the regression anchor (gs_joint_p3 35.706 / 34.928, tolerance 0.02 dB). Queue state and controls: `STATUS.md`.
```

- [ ] **Step 5: Commit and push**

```bash
cd /data0/congcong/code/GS/mcgs_hparam_wt && git add STATUS.md CLAUDE.md docs/superpowers/plans/2026-09-16-hparam-sensitivity.md
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "hparam-sensitivity: implementation plan; queue launched (hp_n36 anchor and hp_n18 first on cards 1 and 2), controls and hand-off recorded"
git push origin hparam-sensitivity
```

---

### Task 5 (next session, after the six runs land): summary, anchor gate, figure, topic folder, record

**Files:**
- Create: `scripts/hp_fig_curves.py`
- Create: `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_hparam_v3_jiangyizhou/{README.md,scripts/,logs/,figures/,data/}` and the repo mirror `t1flair_af16_hparam_v3_jiangyizhou/`
- Create: `docs/superpowers/records/2026-09-16-hparam-sensitivity.md`
- Modify: `STATUS.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: `results_arc/hp_summary.csv` columns `exp, n_shared, n_private, ratio, status, T1_head_psnr, T1_head_ssim, T1_head_nmse, FLAIR_head_psnr, FLAIR_head_ssim, FLAIR_head_nmse` (Task 3).

- [ ] **Step 1: Wait for the queue to drain, then summarise**

Run (bash): `cd /data0/congcong/code/GS/mcgs_hparam_wt && tail -3 results_arc/hp_queue.log && cat results_arc/hp_runs.log && env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/hp_summarize.py`
Expected: `queue drained`; six `rc=0` lines; a table with six `done` rows; the anchor line. If the anchor reads FAIL: stop, diff `paired_gs/` and `arc/` between `f0fe009` (phase-3 tip) and this branch, record the drift in STATUS.md, and do not compare to registered numbers until it is explained. If any run has `rc!=0`: read its log's last 30 lines, fix nothing silently, record the cause; re-queue only after the cause is known (delete that run's `results_arc/hp_<exp>*` first).

- [ ] **Step 2: Load the dataviz skill, then write `scripts/hp_fig_curves.py`**

Invoke `Skill: dataviz` before writing (its palette validator and mark rules apply). The script:
```python
#!/usr/bin/env python3
"""hp_fig_curves.py -- the two sensitivity panels (spec section 1): left N_s -> head PSNR, right N_p/N_s -> head PSNR,
second row the same for head SSIM. T1 = filled marker, solid; FLAIR = open marker, dashed; one hue per contrast
(blue T1, orange FLAIR, the trend topic's palette). Reads results_arc/hp_summary.csv (scripts/hp_summarize.py).
Output: figures/hp_sensitivity_{psnr_ssim}.{png,pdf} in the topic folder, mirrored to /home/congcongliu/figs/t1flair_v3_af16_hparam/.
"""
import csv, os, shutil, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results_arc" / "hp_summary.csv"
TOPIC = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_hparam_v3_jiangyizhou")
SERVED = Path("/home/congcongliu/figs/t1flair_v3_af16_hparam")
BLUE, ORANGE, INK, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#e1e0d9"

rows = [r for r in csv.DictReader(open(CSV)) if r["status"] == "done"]
cap = sorted([r for r in rows if int(r["n_private"]) == 0], key=lambda r: int(r["n_shared"]))
priv = sorted([r for r in rows if int(r["n_shared"]) == 3_600_000], key=lambda r: float(r["ratio"]))
if len(cap) < 2 or len(priv) < 2:
    sys.exit(f"not enough finished runs: capacity {len(cap)}, private {len(priv)}")

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), dpi=200)
for row_i, metric, label in ((0, "head_psnr", "head PSNR (dB)"), (1, "head_ssim", "head SSIM")):
    for col_j, series, xkey, xlabel in ((0, cap, "n_shared", "shared Gaussians N_s (M)"),
                                        (1, priv, "ratio", "private ratio N_p / N_s")):
        ax = axes[row_i, col_j]
        xs = [int(r[xkey]) / 1e6 if xkey == "n_shared" else float(r[xkey]) for r in series]
        for name, color, mfc, ls in (("T1", BLUE, BLUE, "-"), ("FLAIR", ORANGE, "white", "--")):
            ys = [float(r[f"{name}_{metric}"]) for r in series]
            ax.plot(xs, ys, ls, color=color, marker="o", mfc=mfc, mec=color, lw=1.4, ms=5, label=name)
        if xkey == "n_shared":
            ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}" for x in xs])
        else:
            ax.set_xticks(xs); ax.set_xticklabels([f"{x:.0%}" for x in xs])
        ax.set_xlabel(xlabel); ax.set_ylabel(label)
        ax.grid(True, color=GRID, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
        if row_i == 0 and col_j == 0:
            ax.legend(frameon=False, loc="lower right")
fig.suptitle("MC-GS sensitivity, jiangyizhou T1+FLAIR, nominal AF 16 (T1 15.19x / FLAIR 16.49x), frozen phase-3 recipe, 4000 it",
             fontsize=8.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96))
(TOPIC / "figures").mkdir(parents=True, exist_ok=True); SERVED.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    out = TOPIC / "figures" / f"hp_sensitivity_psnr_ssim.{ext}"
    fig.savefig(out, bbox_inches="tight")
    if ext == "png":
        shutil.copy(out, SERVED / out.name)
print("wrote", TOPIC / "figures", "and", SERVED)
```
Run (bash): `cd /data0/congcong/code/GS/mcgs_hparam_wt && env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/hp_fig_curves.py && curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8765/t1flair_v3_af16_hparam/hp_sensitivity_psnr_ssim.png`
Expected: `wrote …` and `200`. Look at the PNG (Read tool) before reporting; give the user the served URL line and the absolute path line.

- [ ] **Step 3: Build the topic folder and its README**

```bash
T=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_hparam_v3_jiangyizhou
W=/data0/congcong/code/GS/mcgs_hparam_wt
mkdir -p $T/{scripts,logs,figures,data,mcgs}
ln -sfn /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/data/ksp_T1.h5 $T/data/ksp_T1.h5
ln -sfn /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/data/ksp_FLAIR.h5 $T/data/ksp_FLAIR.h5
ln -sfn /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/data/csm_espirit_T1.h5 $T/data/csm_espirit_T1.h5
ln -sfn /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/data/csm_espirit_FLAIR.h5 $T/data/csm_espirit_FLAIR.h5
ln -sfn /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/data/mask_af16_3d.h5 $T/data/mask_af16_3d.h5
cp $W/run_hp_mcgs.sh $W/scripts/hp_launch_one.sh $W/scripts/hp_queue.sh $W/scripts/hp_queue.txt $W/scripts/hp_summarize.py $W/scripts/hp_fig_curves.py $T/scripts/
cp $W/results_arc/hp_n??.log $W/results_arc/hp_p??.log $W/results_arc/hp_runs.log $W/results_arc/hp_queue.log $W/results_arc/hp_summary.md $W/results_arc/hp_summary.csv $T/logs/
cp $W/results_arc/hp_n??_score.json $W/results_arc/hp_p??_score.json $T/mcgs/
mkdir -p $W/t1flair_af16_hparam_v3_jiangyizhou && rsync -a --exclude data --exclude mcgs $T/ $W/t1flair_af16_hparam_v3_jiangyizhou/
```
README sections (write in Chinese like the sibling topics): 什么数据（jyz AF16，符号链接到 AF16 课题）、协议（六个运行、冻结配方、只变一个旋钮、anchor 门）、结果表（粘贴 `hp_summary.md`）、两条读法（第 1 组饱和判据、第 2 组 null 判据）、复现命令、坑（`TORCH_EXTENSIONS_DIR`、锁、time 记录）。Volumes stay in `results_arc/` (about 1.4 GB each) unless the user asks for the bundle.

- [ ] **Step 4: Write the record `docs/superpowers/records/2026-09-16-hparam-sensitivity.md`**

Five sections: what ran (six exps, cards, wall time from `hp_runs.log`), the table (`hp_summary.md` verbatim), the anchor verdict, the pre-registered readings from spec section 8 applied to the measured numbers (no other interpretation), caveats (single subject, single mask, magnitude gauge, private set FLAIR-only, no lesion metric). Add the topic row to `Results_20260908/README.md`'s table.

- [ ] **Step 5: Update `STATUS.md` (five hand-off sections), `CLAUDE.md`, commit, push, tag**

```bash
cd /data0/congcong/code/GS/mcgs_hparam_wt
git add STATUS.md CLAUDE.md scripts/hp_fig_curves.py docs/superpowers/records/2026-09-16-hparam-sensitivity.md t1flair_af16_hparam_v3_jiangyizhou
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "hparam-sensitivity complete: six runs, summary table, anchor verdict, the two-panel figure, topic folder and record"
git push origin hparam-sensitivity && git tag handoff/$(date +%F)-hparam && git push origin handoff/$(date +%F)-hparam
```
