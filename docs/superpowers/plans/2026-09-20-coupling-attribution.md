# MC-GS coupling attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how much of the published MC-GS − GS-single margin (+0.148 dB T1 / +0.042 dB FLAIR at AF 16) survives when the only tunable cross-contrast term, the common-edge prior R_edge, is switched off — and report the split against a rule written before the runs.

**Architecture:** A launcher `run_ca_mcgs.sh` whose recipe block is byte-identical to `run_hp_mcgs.sh` except that the base config and λ_edge become positionals; the proven lock wrapper and queue runner copied from the hparam study (shared `.gpulock.<card>`, one trainer per card); a summariser that turns the six score jsons into the 2×2 margin table; then a dose-response stage and a placebo stage, each gated on the user reading the previous one.

**Tech Stack:** bash (launch/queue), Python 3 at `/home/congcongliu/anaconda3/envs/GSTMI/bin/python` (torch 2.7.1+cu118, numpy, matplotlib, pytest), `arc/cycle.py` trainer, `diag_artefact.headonly` head mask.

**Spec:** `docs/superpowers/specs/2026-09-20-coupling-attribution-design.md`

## Global Constraints

- Every Python command: `env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID /home/congcongliu/anaconda3/envs/GSTMI/bin/python …` (repo CLAUDE.md rule).
- Run every multi-command shell step through `bash` (`bash <<'EOF' … EOF` or a script file); the tool shell is zsh and does not word-split `$VAR`.
- Experiment names start with `ca_`; outputs only under `/data0/congcong/code/GS/mcgs_hparam_wt/results_arc/` (gitignored). Nothing is written into `mcgs_af16_wt` or into any `Results_20260908/` topic before Task 7.
- GPUs: only a card with no compute process, < 1500 MiB used and no live `.gpulock.<card>` owner under `/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MCGS/work/`; one trainer per card; launches ≥ 90 s apart; anything longer than minutes goes through `setsid nohup … < /dev/null &` from a bash script. All eight cards were idle at 2026-09-20 14:5x; re-check before launching, other sessions run on this machine.
- `TORCH_EXTENSIONS_DIR=/data0/congcong/code/GS/mcgs_hparam_wt.torch_ext` (never another worktree's cache).
- Commits: `git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit …`, plain human messages, no AI trailers, stage named files only.
- Recipe, byte-identical to `run_hp_mcgs.sh` lines 27-36 except `--base_cfg` and `--lambda_edge`: `--lambda_geo 0.10 --warmup "$WARMUP" --max_radius 20 --kspace_high 2.2 --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn --set loss.edge_tv_kappa_source=zf --set gaussian.initial_num_points=3600000 --set loss.wav_name=db4 --set loss.wav_levels=1 --set loss.lambda_wav_plain=0.12 --set loss.lambda_wav=0.10 --lambda_ptv 0.03`; WARMUP default 500, ITERS 4000, `SAVE_VOLUMES=1`.
- Configs are already on this branch and byte-identical to `mcgs_af16_wt`: `t1flair_v3_af16_jyz.yaml` (md5 `8c1b07fefcb49f83f1e39b56c07344dc`), `…_singleT1.yaml` (`7f13941af2f1ea40ae7e3d7eae203f34`), `…_singleFLAIR.yaml` (`c1f172cd606b3c440ecd490bcede443b`). Do not edit them.
- Anchors: `hp_n36` = J1 = 35.706 / 34.928 (already in `results_arc/`). `ca_s_t1_p3` must land within 0.02 dB of 35.558 (T1), `ca_s_fl_p3` within 0.02 dB of 34.886 (FLAIR).

---

## File structure

| file | responsibility |
|---|---|
| `run_ca_mcgs.sh` (create) | one run: `ca_` prefix guard, overwrite guard, recipe block, config + λ_edge positionals, own build cache, one line into `results_arc/ca_runs.log` |
| `scripts/ca_launch_one.sh` (create) | holds the shared card lock for the run's lifetime, then calls `run_ca_mcgs.sh` |
| `scripts/ca_queue.sh` (create) | queue runner: free-card rule, skip items that already have outputs, pause file, exits when drained; `CA_DRY=1` / `CA_ONCE=1` for tests |
| `scripts/ca_queue_stage1.txt` (create) | the five Stage-1 runs |
| `scripts/ca_queue_stage2.txt` (create, Task 5) | the five Stage-2 runs |
| `scripts/ca_summarize.py` (create) | score jsons + saved volumes → `results_arc/ca_summary.{md,csv}`, the 2×2 margin table, anchor gate |
| `tests/test_ca_summarize.py` (create) | unit test on synthetic jsons / npz, no GPU |
| `scripts/ca_flip_flair.py` (create, Task 6) | builds the placebo flipped-FLAIR k-space and CSM |
| `tests/test_ca_flip_flair.py` (create, Task 6) | asserts the flip is self-consistent and preserves the elliptical support |
| `scripts/ca_fig_dose.py` (create, Task 5) | λ_edge dose-response figure |
| `STATUS.md`, `CLAUDE.md` (modify) | queue state and controls; entry pointer |

---

### Task 1: Launcher and lock wrapper, verified by a smoke run

**Files:**
- Create: `run_ca_mcgs.sh`
- Create: `scripts/ca_launch_one.sh`
- Reference: `run_hp_mcgs.sh` (recipe block), `scripts/hp_launch_one.sh` (lock pattern)

**Interfaces:**
- Produces: `run_ca_mcgs.sh EXP GPU ITERS CONFIG LAMBDA_EDGE [extra cycle.py args]` (env `WARMUP`, `SAVE_VOLUMES=1`) → `results_arc/EXP.log`, `results_arc/EXP_score.json`, `results_arc/EXP/`, `results_arc/EXP_volumes.npz`, one line in `results_arc/ca_runs.log`.
- Produces: `scripts/ca_launch_one.sh EXP CARD ITERS CONFIG LAMBDA_EDGE [KEY=VAL …] [-- extra args]` — same outputs, lock held at `…/work/.gpulock.CARD`.

- [ ] **Step 1: Write `run_ca_mcgs.sh`**

```bash
#!/usr/bin/env bash
# One MC-GS / GS-single run for the coupling-attribution study
# (spec docs/superpowers/specs/2026-09-20-coupling-attribution-design.md):
# jiangyizhou T1+FLAIR at nominal AF 16, house mask (T1 15.19x / FLAIR 16.49x).
#   run_ca_mcgs.sh EXP GPU ITERS CONFIG LAMBDA_EDGE [extra arc/cycle.py args...]
#   SAVE_VOLUMES=1 dumps the recon npz.  WARMUP (default 500) for short smokes.
#
# The recipe block below is byte-identical to run_hp_mcgs.sh except that the base config and
# --lambda_edge are positionals: CONFIG selects the arm (joint / singleT1 / singleFLAIR) and
# LAMBDA_EDGE selects whether the common-edge cross-gradient prior is on (0.001) or off (0).
# Everything else -- capacity, edge-TV, wavelet, phase TV, warm-up, iterations, seed -- is frozen.
set -uo pipefail
if [[ $# -lt 5 ]]; then echo "usage: $0 EXP GPU ITERS CONFIG LAMBDA_EDGE [extra cycle.py args...]" >&2; exit 2; fi
EXP=$1; GPU=$2; ITERS=$3; CONFIG=$4; LEDGE=$5; shift 5
if [[ "$EXP" != ca_* ]]; then
  echo "EXP must start with ca_ (keeps this study apart from the hp_ and AF12/16/20/24 lines): $EXP" >&2; exit 2
fi
if [[ ! -f "$(dirname "$0")/$CONFIG" ]]; then echo "no such config: $CONFIG" >&2; exit 2; fi
WARMUP=${WARMUP:-500}
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
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
  --lambda_edge "$LEDGE" --lambda_ptv 0.03 \
  --base_cfg "$CONFIG" "${VOL[@]}" "$@" > "$LOG" 2>&1
RC=$?
echo "[$(date '+%F %H:%M')] $EXP cfg=$CONFIG ledge=$LEDGE rc=$RC  $(grep -o 'head_psnr[^,]*' "$ROOT/results_arc/${EXP}_score.json" 2>/dev/null | head -2 | tr '\n' ' ')" >> "$ROOT/results_arc/ca_runs.log"
exit $RC
```

- [ ] **Step 2: Write `scripts/ca_launch_one.sh`**

```bash
#!/usr/bin/env bash
# ca_launch_one.sh EXP CARD ITERS CONFIG LAMBDA_EDGE [KEY=VAL ...] [-- extra args...]
# Runs one run_ca_mcgs.sh experiment on CARD while holding the shared card lock
# <cardiac topic>/work/.gpulock.<CARD> (mkdir-atomic, pid file = this wrapper), which the gauss-loop,
# diff-mask, hparam and cardiac schedulers also honour; released when the run ends, whatever the exit code.
set -uo pipefail
cd "$(dirname "$0")/.."
EXP=$1; CARD=$2; IT=$3; CFG=$4; LEDGE=$5; shift 5
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
env "${ENVS[@]}" ./run_ca_mcgs.sh "$EXP" "$CARD" "$IT" "$CFG" "$LEDGE" "$@"
```

- [ ] **Step 3: Make both executable and smoke the joint arm for 30 iterations**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
chmod +x run_ca_mcgs.sh scripts/ca_launch_one.sh
WARMUP=10 ./run_ca_mcgs.sh ca_smoke_joint 0 30 configs/t1flair_v3_af16_jyz.yaml 0.001
echo "rc=$?"
grep -E "^\[arc|lambda_dtv|R_edge" results_arc/ca_smoke_joint.log | head -4
EOF
```

Expected: `rc=0`; the header line prints `lambda_dtv=0.0`; the first `[iter ...]` line prints a non-zero `R_edge`.

- [ ] **Step 4: Smoke the single-T1 arm and the λ_edge = 0 path**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
WARMUP=10 ./run_ca_mcgs.sh ca_smoke_single 0 30 configs/t1flair_v3_af16_jyz_singleT1.yaml 0.001
WARMUP=10 ./run_ca_mcgs.sh ca_smoke_edge0 0 30 configs/t1flair_v3_af16_jyz.yaml 0
grep -o "R_edge=[0-9.e+-]*" results_arc/ca_smoke_single.log | head -2
grep -o "R_edge=[0-9.e+-]*" results_arc/ca_smoke_edge0.log | head -2
EOF
```

Expected: `ca_smoke_single` prints a non-zero `R_edge` (the single arm pays the prior too — this is the finding the design rests on); `ca_smoke_edge0` prints `R_edge=0.0000e+00` on every line.

- [ ] **Step 5: Verify the prefix and overwrite guards**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
./run_ca_mcgs.sh hp_bad 0 30 configs/t1flair_v3_af16_jyz.yaml 0.001; echo "prefix guard rc=$?"
./run_ca_mcgs.sh ca_smoke_joint 0 30 configs/t1flair_v3_af16_jyz.yaml 0.001; echo "overwrite guard rc=$?"
./run_ca_mcgs.sh ca_x 0 30 configs/nope.yaml 0.001; echo "config guard rc=$?"
EOF
```

Expected: `prefix guard rc=2`, `overwrite guard rc=1`, `config guard rc=2`.

- [ ] **Step 6: Delete the smoke outputs and commit**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
rm -rf results_arc/ca_smoke_* 
git add run_ca_mcgs.sh scripts/ca_launch_one.sh
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "coupling attribution: launcher and lock wrapper; smoke runs confirm R_edge is non-zero in the single arm and exactly zero at lambda_edge 0"
EOF
```

---

### Task 2: Queue runner and the five Stage-1 runs

**Files:**
- Create: `scripts/ca_queue.sh`, `scripts/ca_queue_stage1.txt`
- Reference: `scripts/hp_queue.sh`

**Interfaces:**
- Consumes: `scripts/ca_launch_one.sh` from Task 1.
- Produces: `results_arc/{ca_s_t1_p3,ca_s_fl_p3,ca_j_edge0,ca_s_t1_edge0,ca_s_fl_edge0}_score.json` and `_volumes.npz`; `results_arc/ca_queue.log`.
- Queue line format: `[KEY=VAL …] EXP ITERS CONFIG LAMBDA_EDGE [extra args]`.

- [ ] **Step 1: Write `scripts/ca_queue_stage1.txt`**

```
# coupling-attribution Stage 1 (spec docs/superpowers/specs/2026-09-20-coupling-attribution-design.md)
# format: [KEY=VAL ...] EXP ITERS CONFIG LAMBDA_EDGE [extra run_ca_mcgs.sh args]; consumed by scripts/ca_queue.sh, top first.
# The joint anchor J1 is hp_n36 (already in results_arc/, 35.706 / 34.928) and is NOT re-run.
# S1 anchors first -- they gate the whole study: within 0.02 dB of 35.558 (T1) / 34.886 (FLAIR).
SAVE_VOLUMES=1 ca_s_t1_p3 4000 configs/t1flair_v3_af16_jyz_singleT1.yaml 0.001
SAVE_VOLUMES=1 ca_s_fl_p3 4000 configs/t1flair_v3_af16_jyz_singleFLAIR.yaml 0.001
SAVE_VOLUMES=1 ca_j_edge0 4000 configs/t1flair_v3_af16_jyz.yaml 0
SAVE_VOLUMES=1 ca_s_t1_edge0 4000 configs/t1flair_v3_af16_jyz_singleT1.yaml 0
SAVE_VOLUMES=1 ca_s_fl_edge0 4000 configs/t1flair_v3_af16_jyz_singleFLAIR.yaml 0
```

- [ ] **Step 2: Copy the queue runner and adapt it**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
sed -e 's/hp_/ca_/g' -e 's/HP_/CA_/g' scripts/hp_queue.sh > scripts/ca_queue.sh
chmod +x scripts/ca_queue.sh
diff <(sed -e 's/ca_/hp_/g' -e 's/CA_/HP_/g' scripts/ca_queue.sh) scripts/hp_queue.sh && echo "IDENTICAL MODULO PREFIX"
EOF
```

Expected: `IDENTICAL MODULO PREFIX`. Then read `scripts/ca_queue.sh` and fix by hand the one thing the rename cannot do: the queue line now carries two extra positionals (CONFIG, LAMBDA_EDGE) before the optional extra args, so the call into `ca_launch_one.sh` must pass `EXP CARD ITERS CONFIG LAMBDA_EDGE` and the default queue file must be `scripts/ca_queue_stage1.txt`.

- [ ] **Step 3: Dry-run the queue**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
CA_DRY=1 bash scripts/ca_queue.sh scripts/ca_queue_stage1.txt
EOF
```

Expected: five planned launches printed in file order, each naming its config and λ_edge, no GPU touched, exit 0.

- [ ] **Step 4: Check the cards are free, then launch**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
ls -d /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MCGS/work/.gpulock.* 2>/dev/null || echo "no locks"
pgrep -af "arc/cycle.py" | grep -v grep || echo "no trainers running"
EOF
```

Expected: the cards the queue will use are under 1500 MiB with no live lock owner. If another session holds cards, narrow the allow-list rather than killing anything (this machine runs 8+ concurrent sessions).

- [ ] **Step 5: Start the queue detached**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
setsid nohup bash scripts/ca_queue.sh scripts/ca_queue_stage1.txt > results_arc/ca_queue.log 2>&1 < /dev/null &
sleep 5; tail -3 results_arc/ca_queue.log
EOF
```

Expected: the first launches are reported. Five runs at ~66 min each; on five free cards the stage finishes in roughly 70 minutes.

- [ ] **Step 6: Wait for the queue to drain, then confirm all five scored**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
for e in ca_s_t1_p3 ca_s_fl_p3 ca_j_edge0 ca_s_t1_edge0 ca_s_fl_edge0; do
  f=results_arc/${e}_score.json
  if [ -f "$f" ]; then echo "$e OK"; else echo "$e MISSING"; fi
done
cat results_arc/ca_runs.log
EOF
```

Expected: five `OK` lines and five `rc=0` entries in `ca_runs.log`.

- [ ] **Step 7: Commit the scheduler**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
git add scripts/ca_queue.sh scripts/ca_queue_stage1.txt
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "coupling attribution: Stage 1 queue, five runs scored"
EOF
```

---

### Task 3: Summariser and the 2x2 margin table

**Files:**
- Create: `scripts/ca_summarize.py`
- Create: `tests/test_ca_summarize.py`
- Reference: `scripts/hp_summarize.py` (head-NMSE recomputation, score-json field names)

**Interfaces:**
- Consumes: `results_arc/<exp>_score.json`, `results_arc/<exp>_volumes.npz` for the six experiments (`hp_n36` is J1).
- Produces: `results_arc/ca_summary.md`, `results_arc/ca_summary.csv`; `margins(rows) -> {"T1": {"M1": float, "M0": float, "delta": float}, "FLAIR": {…}}`; exit code 1 if an anchor gate fails.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ca_summarize.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ca_summarize import margins


def test_margins_split_the_published_gain():
    rows = {
        ("joint", 0.001): {"T1": 35.706, "FLAIR": 34.928},
        ("single", 0.001): {"T1": 35.558, "FLAIR": 34.886},
        ("joint", 0.0): {"T1": 35.600, "FLAIR": 34.900},
        ("single", 0.0): {"T1": 35.550, "FLAIR": 34.890},
    }
    m = margins(rows)
    assert abs(m["T1"]["M1"] - 0.148) < 1e-9
    assert abs(m["T1"]["M0"] - 0.050) < 1e-9
    assert abs(m["T1"]["delta"] - 0.098) < 1e-9
    assert abs(m["FLAIR"]["M1"] - 0.042) < 1e-9
    assert abs(m["FLAIR"]["M0"] - 0.010) < 1e-9


def test_margins_reports_a_negative_shared_geometry_margin():
    rows = {
        ("joint", 0.001): {"T1": 35.0, "FLAIR": 34.0},
        ("single", 0.001): {"T1": 34.9, "FLAIR": 34.0},
        ("joint", 0.0): {"T1": 34.8, "FLAIR": 33.9},
        ("single", 0.0): {"T1": 34.9, "FLAIR": 34.0},
    }
    m = margins(rows)
    assert m["T1"]["M0"] < 0
    assert abs(m["T1"]["M0"] + 0.1) < 1e-9
```

- [ ] **Step 2: Run it to verify it fails**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_ca_summarize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ca_summarize'`.

- [ ] **Step 3: Write `scripts/ca_summarize.py`**

The pure function first, so the test can import it without touching disk:

```python
def margins(rows):
    """rows[(arm, lambda_edge)][contrast] -> head PSNR.  Returns per-contrast M1 / M0 / delta.

    M1 = joint - single at the published lambda_edge 0.001 (the margin the paper quotes)
    M0 = joint - single at lambda_edge 0 (what shared geometry alone buys)
    delta = M1 - M0 (what the common-edge prior buys)
    """
    out = {}
    for c in ("T1", "FLAIR"):
        m1 = rows[("joint", 0.001)][c] - rows[("single", 0.001)][c]
        m0 = rows[("joint", 0.0)][c] - rows[("single", 0.0)][c]
        out[c] = {"M1": m1, "M0": m0, "delta": m1 - m0}
    return out
```

Then the disk side: read the six score jsons (`hp_n36` for `("joint", 0.001)`; `ca_s_t1_p3` supplies
T1 and `ca_s_fl_p3` supplies FLAIR for `("single", 0.001)`; `ca_j_edge0`, `ca_s_t1_edge0`,
`ca_s_fl_edge0` for the λ_edge 0 row), recompute head NMSE from the saved volumes exactly as
`scripts/hp_summarize.py` does, write `ca_summary.md` and `ca_summary.csv`, and exit 1 if
`ca_s_t1_p3` is further than 0.02 dB from 35.558 or `ca_s_fl_p3` further than 0.02 dB from 34.886.
Each single-arm run reports its contrast under the `plain` channel, so read `plain_*` fields for the
single arms and `plain_*` / `contrast_*` for the joint arms.

- [ ] **Step 4: Run the test to verify it passes**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_ca_summarize.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the summariser on the real runs**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/ca_summarize.py
echo "exit=$?"
cat results_arc/ca_summary.md
EOF
```

Expected: `exit=0` (both single anchors reproduce) and a table carrying M1, M0 and delta per contrast. If the anchor gate fails, stop and report — a single arm that does not reproduce across worktrees invalidates the comparison and must be understood before any reading is written down.

- [ ] **Step 6: Commit**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
git add scripts/ca_summarize.py tests/test_ca_summarize.py
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "coupling attribution: summariser with the 2x2 margin split and the single-arm anchor gate, two unit tests"
EOF
```

---

### Task 4: Read Stage 1 against the pre-registered rule, and stop

**Files:**
- Create: `docs/superpowers/records/2026-09-20-coupling-attribution.md`
- Modify: `STATUS.md`

- [ ] **Step 1: Write the record**

Copy the `ca_summary.md` table in verbatim, then state which row of the spec's §4 table the result
lands in, per contrast, using the measured numbers. Do not soften a negative or near-zero M0: the
spec pre-registered every outcome as publishable, and the AF 12 FLAIR history (−0.038 → +0.036) means
a small or negative M0 on FLAIR is the expected case, not a failure.

Include, as context and clearly marked as read from the topic rather than measured here:
`T1FLAIR_AF16_v3_jiangyizhou/README.md:195` (M1 = +0.148 / +0.042, phase-1 +0.037 / −0.003, seed
floor 0.001 / 0.000) and `T1FLAIR_AF12_v3_jiangyizhou/README.md:643` (the AF 12 sign flip).

- [ ] **Step 2: Update `STATUS.md`** with: Stage 1 complete, the three numbers per contrast, what the
user has to decide (run Stage 2 dose-response? Stage 3 placebo? Stage 4 equal-budget search?), and
the exact commands to resume.

- [ ] **Step 3: Commit and stop for the user**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
git add docs/superpowers/records/2026-09-20-coupling-attribution.md STATUS.md
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "coupling attribution Stage 1: shared-geometry margin measured against the published margin, read against the pre-registered rule"
EOF
```

**Gate.** Stages 2 and 3 are only worth running for certain Stage-1 outcomes, and the user decides.
Do not start Task 5 without them.

---

### Task 5: Stage 2, dose-response on λ_edge

**Files:**
- Create: `scripts/ca_queue_stage2.txt`, `scripts/ca_fig_dose.py`
- Modify: `scripts/ca_summarize.py` (accept the extra λ values)

**Interfaces:**
- Consumes: `scripts/ca_queue.sh` and `run_ca_mcgs.sh` unchanged.
- Produces: `results_arc/ca_dose.{md,csv}`, `figures/ca_dose_lambda_edge.{png,pdf}`.

- [ ] **Step 1: Write `scripts/ca_queue_stage2.txt`**

```
# coupling-attribution Stage 2: dose-response on lambda_edge.
# The joint curve gets five points with Stage 1 (0, 0.0003, 0.001=hp_n36, 0.003, 0.01);
# the margin gets three (0 and 0.001 from Stage 1, 0.003 here).
SAVE_VOLUMES=1 ca_j_e0003 4000 configs/t1flair_v3_af16_jyz.yaml 0.0003
SAVE_VOLUMES=1 ca_j_e003 4000 configs/t1flair_v3_af16_jyz.yaml 0.003
SAVE_VOLUMES=1 ca_j_e01 4000 configs/t1flair_v3_af16_jyz.yaml 0.01
SAVE_VOLUMES=1 ca_s_t1_e003 4000 configs/t1flair_v3_af16_jyz_singleT1.yaml 0.003
SAVE_VOLUMES=1 ca_s_fl_e003 4000 configs/t1flair_v3_af16_jyz_singleFLAIR.yaml 0.003
```

- [ ] **Step 2: Dry-run, check the cards, launch detached** — same three commands as Task 2 Steps 3-5 with `scripts/ca_queue_stage2.txt` and `results_arc/ca_queue_stage2.log`.

- [ ] **Step 3: Extend the summariser** to emit `ca_dose.csv` with one row per λ_edge (`lambda_edge, arm, contrast, head_psnr, head_ssim, head_hfen, head_nmse`) and the margin where both arms exist.

- [ ] **Step 4: Write `scripts/ca_fig_dose.py`** — two panels (T1, FLAIR), x = λ_edge on a log axis with 0 drawn at the left edge as a separate tick, y = head PSNR, two lines (joint, single) plus a margin line on a twin axis; vertical grey line at the published 0.001. Palette validated with the `dataviz` skill's checker, as in `scripts/hp_fig_curves.py`.

- [ ] **Step 5: Run the figure script and check both files exist**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/ca_fig_dose.py
ls -la figures/ca_dose_lambda_edge.png figures/ca_dose_lambda_edge.pdf
EOF
```

- [ ] **Step 6: Commit**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
git add scripts/ca_queue_stage2.txt scripts/ca_fig_dose.py scripts/ca_summarize.py
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "coupling attribution Stage 2: lambda_edge dose-response, five runs and the two-panel curve"
EOF
```

---

### Task 6: Stage 3, the placebo second contrast

**Files:**
- Create: `scripts/ca_flip_flair.py`, `tests/test_ca_flip_flair.py`
- Create: `configs/t1flair_v3_af16_jyz_placebo.yaml`

**Interfaces:**
- Produces: `results_arc/placebo/ksp_FLAIR_flip.h5`, `results_arc/placebo/csm_espirit_FLAIR_flip.h5`; `flip_pair(ksp, csm, axis) -> (ksp_flip, csm_flip)`.
- Consumes: `data/ksp_FLAIR.h5`, `data/csm_espirit_FLAIR.h5` (read-only symlinks into the AF16 topic — never written to).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ca_flip_flair.py
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ca_flip_flair import flip_pair, fft3c, ifft3c


def _sense(ksp, csm):
    return (np.conj(csm) * ifft3c(ksp)).sum(axis=-1)


def test_flip_pair_flips_the_sense_image_and_keeps_the_support():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(8, 8, 6)) + 1j * rng.normal(size=(8, 8, 6))
    csm = rng.normal(size=(8, 8, 6, 3)) + 1j * rng.normal(size=(8, 8, 6, 3))
    ksp = fft3c(img[..., None] * csm)
    support = np.zeros((8, 8, 6), bool)
    support[:, 1:7, :] = True                      # a centred, symmetric shutter
    ksp = ksp * support[..., None]

    ksp_f, csm_f = flip_pair(ksp, csm, axis=1)

    got = _sense(ksp_f, csm_f)
    want = np.flip(_sense(ksp, csm), axis=1)
    assert np.allclose(got, want, atol=1e-8)
    got_support = np.abs(ksp_f).sum(axis=-1) > 0
    assert np.array_equal(got_support, support)    # known_zero stays valid
```

- [ ] **Step 2: Run it to verify it fails**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_ca_flip_flair.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ca_flip_flair'`.

- [ ] **Step 3: Write `scripts/ca_flip_flair.py`**

```python
import numpy as np


def fft3c(x):
    ax = (0, 1, 2)
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)


def ifft3c(x):
    ax = (0, 1, 2)
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)


def flip_pair(ksp, csm, axis):
    """Flip the object and the coils together along `axis`, in image space.

    Flipping the coil images and the sensitivity maps by the same amount keeps the model
    self-consistent: the SENSE combination of the result is exactly the flipped original.
    The roll by 1 puts DC back on the array centre after the reversal, so the measured
    elliptical support is preserved sample-for-sample and `known_zero` stays valid.
    """
    coil_img = ifft3c(ksp)
    coil_img = np.roll(np.flip(coil_img, axis=axis), 1, axis=axis)
    csm_f = np.roll(np.flip(csm, axis=axis), 1, axis=axis)
    return fft3c(coil_img), csm_f
```

Add a `__main__` block that reads the two h5 files with `h5py`, applies `flip_pair(..., axis=1)`,
writes `results_arc/placebo/` with the same dataset names and dtypes as the inputs, and prints the
md5 of each output.

- [ ] **Step 4: Run the test to verify it passes**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_ca_flip_flair.py -v`
Expected: 1 passed.

- [ ] **Step 5: Build the placebo data and the config**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/ca_flip_flair.py
ls -la results_arc/placebo/
EOF
```

Then copy `configs/t1flair_v3_af16_jyz.yaml` to `configs/t1flair_v3_af16_jyz_placebo.yaml` and change
only the contrast channel's `data_path` and `csm_path` to the two files in `results_arc/placebo/`.
Keep `known_zero: true` (the test proved the support survives) and keep the channel `name: FLAIR`.

- [ ] **Step 6: Run the placebo and compare the T1 channel**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
setsid nohup bash scripts/ca_launch_one.sh ca_j_placebo 0 4000 configs/t1flair_v3_af16_jyz_placebo.yaml 0.001 SAVE_VOLUMES=1 \
  > results_arc/ca_j_placebo.launcher.log 2>&1 < /dev/null &
EOF
```

When it finishes, compare its `plain` (T1) head PSNR with `ca_s_t1_p3` and with `hp_n36`. Read only
the T1 channel: the contrast channel is reconstructing a flipped brain and its score is meaningless.

- [ ] **Step 7: Commit**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
git add scripts/ca_flip_flair.py tests/test_ca_flip_flair.py configs/t1flair_v3_af16_jyz_placebo.yaml
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "coupling attribution Stage 3: placebo second contrast, flip verified self-consistent and support-preserving, one joint run"
EOF
```

---

### Task 7: Close the line

**Files:**
- Create: `/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_coupling_v3_jiangyizhou/` (README, logs, figures, scripts, mcgs mirrors — same layout as `T1FLAIR_AF16_hparam_v3_jiangyizhou`)
- Modify: `docs/superpowers/records/2026-09-20-coupling-attribution.md`, `STATUS.md`, `CLAUDE.md`

- [ ] **Step 1: Build the topic folder** with the layout of the hparam topic: `README.md` (what data, the protocol, the results table, the reading, reproduce commands, pitfalls), `data/` as symlinks into `T1FLAIR_AF16_v3_jiangyizhou/data`, `logs/` with the training logs and summaries, `mcgs/` with the score jsons, `figures/`, `scripts/` snapshots. Saved volumes stay in the gitignored `results_arc/`.

- [ ] **Step 2: Write the paragraph the paper needs** into the record: one paragraph that a reviewer asking "is this joint modelling or tuning?" can be answered with, quoting M1, M0, delta, the dose-response shape and the placebo result, and naming what is still not covered (one subject, one AF, Stage 4 not run).

- [ ] **Step 3: Commit, tag and hand off**

```bash
bash <<'EOF'
cd /data0/congcong/code/GS/mcgs_hparam_wt
git add -A docs/ STATUS.md CLAUDE.md
git -c user.name="Congcong Liu" -c user.email="liuc23539@gmail.com" commit -m "coupling attribution complete: margin split, dose-response and placebo, topic folder and hand-off"
git tag handoff/2026-09-20-coupling-attribution
EOF
```

---

## Self-review

**Spec coverage.** §2 coupling inventory → Task 1 Steps 3-4 assert it in the logs (`lambda_dtv=0.0`,
`R_edge` non-zero in the single arm, exactly zero at λ_edge 0). §3 Stage 1 → Tasks 2-3. §3 Stage 2 →
Task 5. §3 Stage 3 → Task 6. §3 Stage 4 → deliberately not planned; the gate in Task 4 hands it to
the user. §4 decision rule → Task 4 Step 1. §5 what is not covered → Task 7 Step 2. §6 fixed factors →
the Global Constraints block.

**Placeholders.** Task 3 Step 3 and Task 5 Steps 3-4 describe the disk-side and figure code in prose
rather than giving it in full; both follow `scripts/hp_summarize.py` and `scripts/hp_fig_curves.py`
line for line in this same worktree, which is named as the reference. Every other step carries its
command or its code. Task 2 Step 2 deliberately leaves one hand edit after the `sed` rename and says
exactly what it is.

**Type consistency.** `margins(rows)` is keyed `(arm, lambda_edge)` in the test and in the
implementation; `flip_pair(ksp, csm, axis)` returns `(ksp_flip, csm_flip)` in the test and in the
implementation; experiment names match between `ca_queue_stage1.txt`, the summariser's expected set
and Task 2 Step 6. Every interpreter invocation is the single path named in the Global Constraints.
