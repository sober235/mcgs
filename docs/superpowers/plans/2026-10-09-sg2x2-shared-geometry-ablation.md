# MC-GS shared-geometry 2x2 ablation (sg2x2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the four cells A / B / C / D of the shared-geometry 2x2 ablation runnable on the jiangyizhou AF16 pair from one launcher, with a true single-contrast GS-single, identical initial geometry in every cell, every regulariser switch audited from the resolved config and the training log, and nothing else in the repository changing behaviour.

**Architecture:** A new `SingleContrastTrainer` (subclass of `JointTrainer`, own file) reconstructs one contrast per process: B and D are two such runs (T1, FLAIR), A and C are the existing `JointTrainer`. A new init mode `gaussian.init_mode=support_file` draws the initial Gaussian positions uniformly inside a frozen boolean support (an `.npz` derived once from the development subject) with the run seed, so every cell of a subject starts from the same positions and 3-NN scales; densities still come from each contrast's own zero-filled image. A pure-python cell table (`scripts/sg2x2_cells.py`) is the single source of the four cells' `arc/cycle.py` arguments; the launcher, the manifest (provenance + regulariser audit) and the result table are thin scripts around it. All edits to existing files are additive and default-off (`arc/cycle.py --trainer joint` default, the `support_file` branch is only entered when asked).

**Tech Stack:** Python 3 / torch 2.7.1+cu118 in `/home/congcongliu/anaconda3/envs/GSTMI`, pytest, h5py, numpy, scipy (`cKDTree`, `binary_dilation`), bash launchers; GPU required for the trainer tests.

**Spec:** `docs/experiments/2026-10-09-mcgs-shared-geometry-2x2-ablation.md` (the four cells, §2–§7). Code facts the plan relies on were read from `paired_gs/trainers/joint_trainer.py`, `paired_gs/gaussian/init_cloud.py`, `arc/cycle.py`, `run_wd_mcgs.sh` at commit 9df60e7.

## Global Constraints

- Environment for every command: `PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python`, `env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID`. Tests: `CUDA_VISIBLE_DEVICES=<idle card> $PY -m pytest tests -q --ignore=tests/smoke_test.py --ignore=tests/test_score_edge.py -p no:cacheprovider` (baseline on this branch: 223 passed).
- Worktree `/data0/congcong/code/GS/mcgs_sg2x2_wt`, branch `exp/shared-geometry-2x2-ablation-20261009`. Never run commands in other worktrees; never write under `Results_20260908/` in this plan.
- Existing files that may be modified, and only additively: `paired_gs/trainers/joint_trainer.py` (one `elif init_mode == "support_file"` branch in `_setup_model`, and — ruling R9, found in Task 3 — two lines in `train()` that initialise `self._edge_kappa_p = None` / `self._edge_kappa_c = None` just before the `if self._edge_tv:` block, because the guide tuples `gd_p` / `gd_c` read those attributes every iteration and a run with `loss.edge_tv=false` has died at iteration 1 since f413c26; cell C needs that path), `paired_gs/gaussian/init_cloud.py` (new functions appended), `arc/cycle.py` (new `--trainer` flag; every new line is behind `args.trainer == "single"` or guards a missing `data.contrast`). No other existing file changes. The regression gate (Task 7) proves the default path reproduces the pre-change smoke run.
- The spec's recipe for A/B (spec §3): `lambda_geo 0.10`, edge-TV on with `kappa_q 0.85 / w_floor 0.10 / tv_eps 0.04`, wavelet db4 one level, T1 `0.12`, FLAIR `0.10`, `lambda_ptv 0.03`; every cross-contrast weight 0 (`lambda_edge`, `lambda_dtv`, `lambda_vtv`, `lambda_dtv_proj`, `edge_tv_guide_alpha`, `ptv_guide_alpha`, `guide_path_*`, `lambda_nl*`). C/D: `lambda_geo 0`, `edge_tv false`, `lambda_wav 0`, `lambda_wav_plain 0`, `lambda_ptv 0`. Shared in all cells: `kspace_freq_adaptive` with `kspace_high 2.2`, `warmup 500`, `max_radius 20`, `init_scale_mode 3nn`, 4000 iterations, `adaptive_control` off, `num_private_points 0`, seed 42 (43, 44 later).
- Capacity (spec §4): A/C 3.6M shared; B/D 3.6M **per contrast** (main), 1.8M per contrast (supplementary, `NPTS=1800000`).
- A single-contrast run gets that contrast's own wavelet weight (`lambda_wav` and `lambda_wav_plain` both = own λ): T1 0.12, FLAIR 0.10. The spec's historical confound (FLAIR scored with T1's 0.12) must not recur.
- Commits: author is the repo identity (`Congcong Liu <liuc23539@gmail.com>`, already set in this worktree), plain human messages, no AI trailers, stage named files only. Do not push.
- GPU runs (Task 0 reference smoke, Task 7 regression smoke, Task 8 pilot) only on cards with no compute process (`nvidia-smi --query-compute-apps=pid --format=csv,noheader -i N` empty and `< 1500 MiB` used). Cards 5/6/7 belong to other sessions today. The six pilot runs (Task 8) are launched only after the user says go.
- Main-table decision carried from the spec review: A and B use the identical frozen per-channel recipe above (spec §3). The equal-budget per-arm λ search of spec §4 is **not** in this plan; it is a later sensitivity analysis.

## Review Focus

1. A single-contrast config whose `data.plain` still carries `known_zero: true` (the FLAIR shutter) must build the fit tensors exactly as the joint trainer builds them for the FLAIR channel; a run where the shutter is silently dropped would hand FLAIR 22 % more unknowns than cell A. Pinned in Task 3 (`test_single_known_zero_matches_joint_contrast_channel`).
2. A C/D config whose YAML base still says `edge_tv: true` (the topic YAML does) must come out with `edge_tv false` and `lambda_ptv 0`; otherwise the trainer resolves an edge kappa and the audit reads a "regulariser on" start-up line. Pinned in Task 1 (`test_reg_off_cells_force_edge_tv_false`) and Task 5 (`test_audit_rejects_regulariser_startup_lines_in_reg_off_cell`).
3. A support file whose shape differs from the subject's volume (another grid) must stop the run before any GPU time is spent, not index out of range later. Pinned in Task 2 (`test_support_file_shape_mismatch_raises`).
4. A subject whose head lies partly outside the frozen support must still get densities only from its own ZF, and the log must say what fraction of the drawn points fell inside that subject's own ZF support. Pinned in Task 2 (`test_support_file_logs_own_support_fraction`).
5. Re-running a cell name must refuse to overwrite (one name, one run): the launcher refuses when the log, the score json or the output directory exists. Pinned in Task 4 (`test_launcher_refuses_existing_outputs`).

---

### Task 0: Baseline and the pre-change reference smoke

**Files:**
- Create: `results_arc/sg2x2_regress_before.log` (gitignored, produced by the run)
- Create: `results_arc/baseline_tests_20261009.log` (gitignored; already produced on 2026-10-09, 223 passed expected)

**Interfaces:**
- Produces: the reference log consumed by Task 7's `scripts/sg2x2_regress_compare.py`.

- [ ] **Step 1: Confirm the tree is clean and the test baseline passed**

Run:
```bash
cd /data0/congcong/code/GS/mcgs_sg2x2_wt && git status --porcelain | head && git log -1 --oneline
tail -2 results_arc/baseline_tests_20261009.log
```
Expected: no modified tracked files; the last line of the baseline log says `223 passed`.

- [ ] **Step 2: Pick an idle card**

Run:
```bash
for g in 0 1 2 3 4; do echo -n "card $g: "; nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $g; nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g | grep -c . ; done
```
Expected: at least one card with `< 1500 MiB` and `0` processes. Use it as `G` below.

- [ ] **Step 3: Run the 50-iteration reference smoke with the frozen A recipe on the unmodified tree**

Run (from the worktree root, detached, bash):
```bash
cat > results_arc/sg2x2_regress_before.sh <<'EOF'
#!/usr/bin/env bash
cd /data0/congcong/code/GS/mcgs_sg2x2_wt
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID "$PY" -u arc/cycle.py --exp sg2x2_regress_before --gpu "$1" --iters 50 \
  --lambda_geo 0.10 --warmup 20 --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn --set loss.edge_tv_kappa_source=zf \
  --set gaussian.initial_num_points=3600000 --set loss.wav_name=db4 --set loss.wav_levels=1 \
  --set loss.lambda_wav_plain=0.12 --set loss.lambda_wav=0.10 --lambda_edge 0 --lambda_ptv 0.03 \
  --base_cfg configs/t1flair_v3_af16_jyz.yaml
EOF
chmod +x results_arc/sg2x2_regress_before.sh
setsid nohup bash results_arc/sg2x2_regress_before.sh G > results_arc/sg2x2_regress_before.log 2>&1 < /dev/null &
```
Expected: about 5 minutes. The log ends with `[arc sg2x2_regress_before] DONE`, contains `[iter     1/50]` and `[iter    50/50]` lines and the two score lines `T1    psnr=...` / `FLAIR psnr=...`.

- [ ] **Step 4: Record the reference numbers in the ledger**

Run:
```bash
grep -E '^\[iter +(1|50)/50\]|^(T1|FLAIR) +psnr=' results_arc/sg2x2_regress_before.log
```
Expected: four lines. Copy them into `docs/superpowers/records/2026-10-09-sg2x2-implementation.md` (created in Task 8) under "Regression reference".

---

### Task 1: The cell table — one source of truth for A / B / C / D

**Files:**
- Create: `scripts/sg2x2_cells.py`
- Test: `tests/test_sg2x2_cells.py`

**Interfaces:**
- Produces: `CELLS: dict[str, Cell]` with keys `A`, `B_T1`, `B_FLAIR`, `C`, `D_T1`, `D_FLAIR`; `Cell` has fields `trainer` (`"joint"` | `"single"`), `contrast` (`None` | `"T1"` | `"FLAIR"`), `reg` (bool), `config` (relative path under `configs/sg2x2/`).
- Produces: `cell_args(cell: str, npts: int, seed: int, support: str, init: str = "support_file", iters: int = 4000, warmup: int = 500) -> list[str]` — the complete `arc/cycle.py` argument list after `--exp/--gpu` (both added by the launcher).
- Produces: `resolved_switches(cell: str) -> dict[str, float | bool]` — the regulariser values the audit (Task 5) expects in the resolved config: `lambda_geo, edge_tv, lambda_wav, lambda_wav_plain, lambda_ptv, lambda_edge, lambda_dtv, lambda_vtv, lambda_dtv_proj, edge_tv_guide_alpha, ptv_guide_alpha`.
- CLI: `python scripts/sg2x2_cells.py CELL --npts N --seed S --support PATH [--init head] [--iters I] [--warmup W]` prints the argument list one per line (consumed by the bash launcher through `mapfile`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sg2x2_cells.py
"""The four cells of the shared-geometry 2x2 ablation, as arc/cycle.py arguments (scripts/sg2x2_cells.py).

A/B carry the frozen AF16 phase-3 per-channel recipe and no cross-contrast weight; C/D carry no explicit image
prior at all; a single-contrast cell gets its own wavelet weight on both wavelet keys; every cell shares the
init mode, the support file, the point count and the seed.
Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_cells.py -q
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sg2x2_cells import CELLS, cell_args, resolved_switches  # noqa: E402

SUP = "/tmp/sup.npz"


def _flag_value(args, flag):
    i = args.index(flag)
    return args[i + 1]


def _sets(args):
    return {a.split("=", 1)[0]: a.split("=", 1)[1] for a in args if "=" in a and args[args.index(a) - 1] == "--set"}


def test_six_cells_and_their_structure():
    assert set(CELLS) == {"A", "B_T1", "B_FLAIR", "C", "D_T1", "D_FLAIR"}
    assert CELLS["A"].trainer == "joint" and CELLS["A"].reg and CELLS["A"].contrast is None
    assert CELLS["C"].trainer == "joint" and not CELLS["C"].reg
    assert CELLS["B_FLAIR"].trainer == "single" and CELLS["B_FLAIR"].contrast == "FLAIR" and CELLS["B_FLAIR"].reg
    assert CELLS["D_T1"].trainer == "single" and CELLS["D_T1"].contrast == "T1" and not CELLS["D_T1"].reg


def test_reg_on_cells_carry_the_frozen_recipe_and_no_cross_contrast_weight():
    for cell in ("A", "B_T1", "B_FLAIR"):
        a = cell_args(cell, npts=3_600_000, seed=42, support=SUP)
        assert _flag_value(a, "--lambda_geo") == "0.10" and "--edge_tv" in a
        assert _flag_value(a, "--edge_tv_kappa_q") == "0.85" and _flag_value(a, "--edge_tv_wfloor") == "0.10"
        assert _flag_value(a, "--tv_eps") == "0.04" and _flag_value(a, "--lambda_ptv") == "0.03"
        assert _flag_value(a, "--lambda_edge") == "0" and _flag_value(a, "--lambda_dtv") == "0"
        assert _flag_value(a, "--lambda_vtv") == "0"
        s = _sets(a)
        assert s["loss.wav_name"] == "db4" and s["loss.wav_levels"] == "1"
        assert s["loss.lambda_dtv_proj"] == "0" and s["loss.edge_tv_guide_alpha"] == "0" and s["loss.ptv_guide_alpha"] == "0"


def test_wavelet_weights_per_cell():
    assert _sets(cell_args("A", 1, 42, SUP))["loss.lambda_wav_plain"] == "0.12"
    assert _sets(cell_args("A", 1, 42, SUP))["loss.lambda_wav"] == "0.10"
    s_t1 = _sets(cell_args("B_T1", 1, 42, SUP))
    assert s_t1["loss.lambda_wav_plain"] == "0.12" and s_t1["loss.lambda_wav"] == "0.12"
    s_fl = _sets(cell_args("B_FLAIR", 1, 42, SUP))
    assert s_fl["loss.lambda_wav_plain"] == "0.10" and s_fl["loss.lambda_wav"] == "0.10"


def test_reg_off_cells_force_edge_tv_false():
    for cell in ("C", "D_T1", "D_FLAIR"):
        a = cell_args(cell, npts=3_600_000, seed=42, support=SUP)
        assert _flag_value(a, "--lambda_geo") == "0" and "--edge_tv" not in a
        assert _flag_value(a, "--lambda_ptv") == "0"
        s = _sets(a)
        assert s["loss.edge_tv"] == "false" and s["loss.lambda_wav"] == "0" and s["loss.lambda_wav_plain"] == "0"
        assert _flag_value(a, "--lambda_edge") == "0"


def test_shared_knobs_identical_across_cells():
    seen = set()
    for cell in CELLS:
        a = cell_args(cell, npts=1_800_000, seed=43, support=SUP, iters=4000, warmup=500)
        s = _sets(a)
        assert s["gaussian.init_mode"] == "support_file" and s["gaussian.init_support_path"] == SUP
        assert s["gaussian.initial_num_points"] == "1800000" and s["training.seed"] == "43"
        assert s["gaussian.init_scale_mode"] == "3nn" and s["loss.edge_tv_kappa_source"] == "zf"
        assert _flag_value(a, "--kspace_high") == "2.2" and _flag_value(a, "--warmup") == "500"
        assert _flag_value(a, "--max_radius") == "20" and _flag_value(a, "--iters") == "4000"
        seen.add((s["gaussian.init_mode"], s["gaussian.initial_num_points"], s["training.seed"]))
    assert len(seen) == 1


def test_trainer_flag_and_config_per_cell():
    a = cell_args("A", 1, 42, SUP)
    assert _flag_value(a, "--trainer") == "joint" and _flag_value(a, "--base_cfg") == "configs/sg2x2/af16_jyz_joint.yaml"
    b = cell_args("D_FLAIR", 1, 42, SUP)
    assert _flag_value(b, "--trainer") == "single" and _flag_value(b, "--base_cfg") == "configs/sg2x2/af16_jyz_single_FLAIR.yaml"


def test_head_init_variant_drops_the_support_file():
    s = _sets(cell_args("A", 1, 42, SUP, init="head"))
    assert s["gaussian.init_mode"] == "head" and "gaussian.init_support_path" not in s


def test_resolved_switches_table():
    on = resolved_switches("B_FLAIR")
    assert on["lambda_geo"] == 0.10 and on["edge_tv"] is True and on["lambda_wav_plain"] == 0.10 and on["lambda_ptv"] == 0.03
    off = resolved_switches("D_T1")
    assert off["lambda_geo"] == 0 and off["edge_tv"] is False and off["lambda_wav"] == 0 and off["lambda_ptv"] == 0
    for k in ("lambda_edge", "lambda_dtv", "lambda_vtv", "lambda_dtv_proj", "edge_tv_guide_alpha", "ptv_guide_alpha"):
        assert on[k] == 0 and off[k] == 0


def test_cli_prints_one_argument_per_line():
    out = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/sg2x2_cells.py"), "C",
                          "--npts", "100", "--seed", "42", "--support", SUP], capture_output=True, text=True, check=True).stdout
    lines = out.strip().split("\n")
    assert lines == cell_args("C", 100, 42, SUP)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_sg2x2_cells.py -q -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'sg2x2_cells'`.

- [ ] **Step 3: Write the cell table**

```python
# scripts/sg2x2_cells.py
#!/usr/bin/env python3
"""The four cells of the shared-geometry 2x2 ablation (spec docs/experiments/2026-10-09-mcgs-shared-geometry-2x2-ablation.md,
section 2-4) as arc/cycle.py arguments. This file is the only place the cells are defined: run_sg2x2.sh asks it for
the argument list, scripts/sg2x2_manifest.py asks it what the resolved config must say.

    A      MC-GS + Reg        joint   shared geometry, frozen per-channel recipe, no cross-contrast weight
    B_T1   GS-single + Reg    single  T1 alone, the same per-channel recipe, its own wavelet weight
    B_FLAIR                   single  FLAIR alone
    C      MC-GS - Reg        joint   data consistency only
    D_T1   GS-single - Reg    single  T1 alone, data consistency only
    D_FLAIR                   single  FLAIR alone, data consistency only

Everything the spec freezes across cells (k-space weighting, warm-up, max_radius, 3-NN scales, point count, seed, the
init support file) is in COMMON; the recipe block equals run_wd_mcgs.sh with lambda_edge 0.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

WAV = {"T1": "0.12", "FLAIR": "0.10"}          # the frozen AF16 phase-3 per-contrast wavelet weights


@dataclass(frozen=True)
class Cell:
    trainer: str          # "joint" | "single"
    contrast: str | None  # None for joint, "T1" | "FLAIR" for single
    reg: bool             # per-channel explicit priors on (A/B) or off (C/D)
    config: str           # base config under configs/sg2x2/


CELLS = {
    "A": Cell("joint", None, True, "configs/sg2x2/af16_jyz_joint.yaml"),
    "B_T1": Cell("single", "T1", True, "configs/sg2x2/af16_jyz_single_T1.yaml"),
    "B_FLAIR": Cell("single", "FLAIR", True, "configs/sg2x2/af16_jyz_single_FLAIR.yaml"),
    "C": Cell("joint", None, False, "configs/sg2x2/af16_jyz_joint.yaml"),
    "D_T1": Cell("single", "T1", False, "configs/sg2x2/af16_jyz_single_T1.yaml"),
    "D_FLAIR": Cell("single", "FLAIR", False, "configs/sg2x2/af16_jyz_single_FLAIR.yaml"),
}

CROSS_OFF = ["--lambda_edge", "0", "--lambda_dtv", "0", "--lambda_vtv", "0",
             "--set", "loss.lambda_dtv_proj=0", "--set", "loss.edge_tv_guide_alpha=0", "--set", "loss.ptv_guide_alpha=0",
             "--set", "loss.lambda_nl=0"]


def _wavelet(cell: Cell) -> tuple[str, str]:
    """(lambda_wav_plain, lambda_wav) for the cell: the joint model keeps T1 on the plain slot and FLAIR on the
    contrast slot; a single-contrast run puts its one contrast's weight on both keys."""
    if cell.contrast is None:
        return WAV["T1"], WAV["FLAIR"]
    return WAV[cell.contrast], WAV[cell.contrast]


def cell_args(cell: str, npts: int, seed: int, support: str, init: str = "support_file",
              iters: int = 4000, warmup: int = 500) -> list[str]:
    c = CELLS[cell]
    args = ["--iters", str(iters), "--warmup", str(warmup), "--max_radius", "20", "--kspace_high", "2.2",
            "--trainer", c.trainer, "--base_cfg", c.config]
    if c.reg:
        wp, wc = _wavelet(c)
        args += ["--lambda_geo", "0.10", "--edge_tv", "--edge_tv_kappa_q", "0.85", "--edge_tv_wfloor", "0.10",
                 "--tv_eps", "0.04", "--lambda_ptv", "0.03",
                 "--set", "loss.wav_name=db4", "--set", "loss.wav_levels=1",
                 "--set", f"loss.lambda_wav_plain={wp}", "--set", f"loss.lambda_wav={wc}"]
    else:
        args += ["--lambda_geo", "0", "--tv_eps", "0.04", "--lambda_ptv", "0",
                 "--set", "loss.edge_tv=false", "--set", "loss.lambda_wav=0", "--set", "loss.lambda_wav_plain=0"]
    args += CROSS_OFF
    args += ["--set", "gaussian.init_scale_mode=3nn", "--set", "loss.edge_tv_kappa_source=zf",
             "--set", f"gaussian.initial_num_points={int(npts)}", "--set", f"training.seed={int(seed)}",
             "--set", "gaussian.num_private_points=0", "--set", "adaptive_control.enable=false"]
    if init == "support_file":
        args += ["--set", "gaussian.init_mode=support_file", "--set", f"gaussian.init_support_path={support}"]
    elif init == "head":
        args += ["--set", "gaussian.init_mode=head"]
    else:
        raise ValueError(f"init must be 'support_file' or 'head', got {init!r}")
    return args


def resolved_switches(cell: str) -> dict:
    """What the trainer's resolved config (results_arc/<exp>/<exp>/config.yaml) must say for this cell."""
    c = CELLS[cell]
    out = {"lambda_edge": 0, "lambda_dtv": 0, "lambda_vtv": 0, "lambda_dtv_proj": 0,
           "edge_tv_guide_alpha": 0, "ptv_guide_alpha": 0}
    if c.reg:
        wp, wc = _wavelet(c)
        out.update({"lambda_geo": 0.10, "edge_tv": True, "lambda_wav_plain": float(wp), "lambda_wav": float(wc),
                    "lambda_ptv": 0.03})
    else:
        out.update({"lambda_geo": 0, "edge_tv": False, "lambda_wav_plain": 0, "lambda_wav": 0, "lambda_ptv": 0})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell", choices=sorted(CELLS))
    ap.add_argument("--npts", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--support", required=True)
    ap.add_argument("--init", default="support_file", choices=["support_file", "head"])
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--warmup", type=int, default=500)
    a = ap.parse_args()
    print("\n".join(cell_args(a.cell, a.npts, a.seed, a.support, a.init, a.iters, a.warmup)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_sg2x2_cells.py -q -p no:cacheprovider`
Expected: `9 passed`.

- [ ] **Step 5: Ignore the plan workspace, then commit**

`.superpowers/` (this plan's ledger and briefs) is not ignored on this branch. Append one line to `.gitignore`:
```bash
printf '\n# subagent-driven-development workspaces (ledger, briefs, reports)\n.superpowers/\n' >> .gitignore
git check-ignore -q .superpowers && echo ignored
```
Expected: `ignored`.

```bash
git add .gitignore scripts/sg2x2_cells.py tests/test_sg2x2_cells.py
git commit -m "sg2x2: the four ablation cells as one argument table, with tests"
```

---

### Task 2: Frozen init support — `init_mode=support_file`

**Files:**
- Modify: `paired_gs/gaussian/init_cloud.py` (append two functions; nothing existing changes)
- Modify: `paired_gs/trainers/joint_trainer.py:251-273` (insert one `elif init_mode == "support_file":` branch between the `uniform` branch and the existing `else`)
- Create: `scripts/sg2x2_make_init_support.py`
- Create: `tests/sg2x2_fixture.py` (synthetic pair builder shared by Tasks 2 and 3)
- Test: `tests/test_sg2x2_init_support.py`

**Interfaces:**
- Produces (`init_cloud.py`): `load_init_support(path: str, volume_shape: tuple[int,int,int], device) -> torch.BoolTensor` (reads `.npz` key `support`, raises `ValueError` on a shape mismatch); `sample_positions_in_support(support: torch.BoolTensor, num_points: int, generator: torch.Generator | None = None, jitter: bool = True) -> (vox (N,3) long, pos (N,3) float in [-1,1])`.
- Produces (trainer): config keys `gaussian.init_mode: support_file`, `gaussian.init_support_path: <.npz>`; start-up line `[init_cloud] mode=support_file path=... n=... support_frac=... in_own_zf_support=...`.
- Produces (`tests/sg2x2_fixture.py`): `make_pair(tmp: Path, shape=(32, 32, 24), seed=0) -> dict` with keys `T1`, `FLAIR` (each `{"data_path","csm_path","mask_path","name"}`), `support` (path of the `.npz`), `shape`; `base_cfg(pair, tmp, exp, reg=True, single=None) -> dict` returning a complete trainer config (joint when `single is None`, else `data.plain` only for that contrast), 1 iteration, tensorboard off.
- Produces (script): `scripts/sg2x2_make_init_support.py --config configs/sg2x2/af16_jyz_joint.yaml --out configs/sg2x2/init_support_af16_jyz.npz [--dilate 3]` writes the union-ZF head support (same rule as `init_mode=head`: `union_magnitude` > 0.05 · P99.9) of the config's pair, dilated, as a compressed `.npz` (key `support`, bool, (D,H,W)), and prints `support_frac`, `dilate`, md5.

- [ ] **Step 1: Write the synthetic fixture**

```python
# tests/sg2x2_fixture.py
"""A tiny synthetic T1/FLAIR pair on disk for the sg2x2 trainer tests: one phantom geometry rendered at two
contrast scalings, a smooth two-coil CSM with unit sum-of-squares, a 40 % random phase-encode mask with a fully
sampled 6x6 centre (constant along the readout axis 0), written as the h5 files MRIDataset reads (keys 'kspace',
'map', 'mask'), plus the frozen init support of the phantom (.npz key 'support')."""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
from data.transforms import fft3c  # noqa: E402

RECIPE_REG = {"tv_weight": 0.10, "tv_eps": 0.04, "edge_tv": True, "edge_tv_kappa_q": 0.85, "edge_tv_wfloor": 0.1,
              "edge_tv_kappa_source": "zf", "lambda_wav": 0.10, "lambda_wav_plain": 0.10, "wav_name": "db4",
              "wav_levels": 1, "lambda_ptv": 0.03}
RECIPE_OFF = {"tv_weight": 0.0, "tv_eps": 0.04, "edge_tv": False, "lambda_wav": 0.0, "lambda_wav_plain": 0.0,
              "lambda_ptv": 0.0}
DC = {"loss_type": "l2", "kspace_freq_adaptive": True, "kspace_center_fraction": 0.05, "kspace_center_weight": 0.3,
      "kspace_mid_weight": 1.0, "kspace_high_weight": 2.2}


def make_pair(tmp: Path, shape=(32, 32, 24), seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d, h, w = shape
    zz, yy, xx = np.meshgrid(np.linspace(-1, 1, d), np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    r2 = zz ** 2 + yy ** 2 + xx ** 2
    base = np.exp(-r2 / 0.18) * (1.0 + 0.3 * np.cos(4.0 * xx))
    base[r2 > 0.8] = 0.0
    csm = np.stack([np.exp(-((xx - 0.5) ** 2) / 0.8), np.exp(-((xx + 0.5) ** 2) / 0.8)]).astype(np.complex64)
    csm = csm / np.sqrt((np.abs(csm) ** 2).sum(0, keepdims=True))
    mask2d = rng.random((h, w)) < 0.4
    mask2d[h // 2 - 3:h // 2 + 3, w // 2 - 3:w // 2 + 3] = True
    mask = np.broadcast_to(mask2d[None], shape).astype(np.float32).copy()
    out = {"shape": shape}
    for name, scale, ph in (("T1", 1.0, 0.3), ("FLAIR", 0.6, -0.2)):
        img = (scale * base * np.exp(1j * ph * xx)).astype(np.complex64)
        ksp = fft3c(torch.from_numpy(csm * img[None])).numpy().astype(np.complex64)
        kp, cp, mp = tmp / f"ksp_{name}.h5", tmp / f"csm_{name}.h5", tmp / "mask.h5"
        with h5py.File(kp, "w") as f:
            f.create_dataset("kspace", data=ksp)
        with h5py.File(cp, "w") as f:
            f.create_dataset("map", data=csm)
        if not mp.exists():
            with h5py.File(mp, "w") as f:
                f.create_dataset("mask", data=mask)
        out[name] = {"name": name, "data_path": str(kp), "csm_path": str(cp), "mask_path": str(mp)}
    support = base > 0.05 * base.max()
    sp = tmp / "support.npz"
    np.savez_compressed(sp, support=support)
    out["support"] = str(sp)
    return out


def _data_block(spec: dict) -> dict:
    return {"name": spec["name"], "data_path": spec["data_path"], "csm_path": spec["csm_path"],
            "mask_path": spec["mask_path"], "mask_type": "external", "readout_axis": 0, "phase_axes": [1, 2],
            "presampled": False, "skip_readout_check": True, "normalize_kspace": True, "normalization_percentile": 99.9,
            "acceleration_factor": 2}


def base_cfg(pair: dict, tmp: Path, exp: str, reg: bool = True, single: str | None = None,
             init_mode: str = "support_file", npts: int = 3000, seed: int = 42, iters: int = 1) -> dict:
    data = {"plain": _data_block(pair[single])} if single else {"plain": _data_block(pair["T1"]),
                                                                  "contrast": _data_block(pair["FLAIR"])}
    loss = dict(DC)
    loss.update(RECIPE_REG if reg else RECIPE_OFF)
    return {
        "data": data,
        "gaussian": {"initial_num_points": npts, "init_mode": init_mode, "init_support_path": pair["support"],
                     "init_scale_mode": "3nn", "init_scale": 0.06, "position_lr": 3e-4, "rotation_lr": 3e-4,
                     "scale_lr": 2e-3, "density_lr": 1.5e-2},
        "loss": loss,
        "training": {"max_iterations": iters, "shared_warmup_iters": 0, "seed": seed, "log_every": 1,
                     "save_every": 100000, "eval_every": 100000, "max_grad_norm": 1.0,
                     "lr_scheduler": {"type": "exponential", "gamma": 0.9995}},
        "voxelizer": {"type": "tile_cuda", "tile_size": 8, "max_radius": 20},
        "output": {"output_dir": str(tmp / "out"), "exp_name": exp, "use_tensorboard": False},
    }
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_sg2x2_init_support.py
"""init_mode=support_file: positions drawn inside a frozen boolean support, seeded, independent of the subject's
data; densities from each contrast's own ZF; shape mismatch refused before any training.
Run: PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=<idle> /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_init_support.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paired_gs"))
sys.path.insert(0, str(ROOT / "tests"))
from gaussian.init_cloud import load_init_support, sample_positions_in_support  # noqa: E402
from sg2x2_fixture import base_cfg, make_pair  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="trainer needs the CUDA voxelizer")


def test_sample_positions_in_support_inside_and_seeded():
    sup = torch.zeros(16, 20, 12, dtype=torch.bool)
    sup[4:12, 5:15, 3:9] = True
    g = torch.Generator().manual_seed(3)
    vox, pos = sample_positions_in_support(sup, 500, generator=g)
    assert vox.shape == (500, 3) and pos.shape == (500, 3)
    assert bool(sup[vox[:, 0], vox[:, 1], vox[:, 2]].all())
    assert pos.min() >= -1.0 and pos.max() <= 1.0
    g2 = torch.Generator().manual_seed(3)
    vox2, pos2 = sample_positions_in_support(sup, 500, generator=g2)
    assert torch.equal(vox, vox2) and torch.equal(pos, pos2)


def test_sample_positions_in_support_empty_raises():
    with pytest.raises(RuntimeError):
        sample_positions_in_support(torch.zeros(4, 4, 4, dtype=torch.bool), 10)


def test_load_init_support_reads_npz_and_checks_shape(tmp_path):
    sup = np.zeros((6, 5, 4), dtype=bool); sup[1:4, 1:3, 1:2] = True
    p = tmp_path / "s.npz"; np.savez_compressed(p, support=sup)
    t = load_init_support(str(p), (6, 5, 4), "cpu")
    assert t.dtype == torch.bool and int(t.sum()) == 6
    with pytest.raises(ValueError, match="shape"):
        load_init_support(str(p), (6, 5, 5), "cpu")


@cuda
def test_support_file_init_is_identical_for_joint_and_single_and_densities_are_own(tmp_path):
    from trainers.joint_trainer import JointTrainer
    from trainers.single_contrast_trainer import SingleContrastTrainer
    pair = make_pair(tmp_path)
    dev = torch.device("cuda", 0)
    joint = JointTrainer(base_cfg(pair, tmp_path, "j"), device=dev)
    single = SingleContrastTrainer(base_cfg(pair, tmp_path, "s", single="FLAIR"), device=dev)
    assert torch.equal(joint.model.positions, single.model.positions)
    assert torch.equal(joint.model.scales, single.model.scales)
    # densities: joint plain slot = T1 ZF, joint contrast slot = FLAIR ZF, single plain slot = FLAIR ZF
    assert torch.equal(joint.model.density_real_contrast, single.model.density_real)
    assert not torch.equal(joint.model.density_real, single.model.density_real)
    # every drawn point sits inside the frozen support
    sup = load_init_support(pair["support"], pair["shape"], dev)
    d, h, w = pair["shape"]
    p = joint.model.positions.detach()
    ix = ((p[:, 0] + 1) * 0.5 * d).clamp(0, d - 1).long(); iy = ((p[:, 1] + 1) * 0.5 * h).clamp(0, h - 1).long()
    iz = ((p[:, 2] + 1) * 0.5 * w).clamp(0, w - 1).long()
    assert bool(sup[ix, iy, iz].all())


@cuda
def test_support_file_positions_do_not_depend_on_the_data(tmp_path):
    from trainers.joint_trainer import JointTrainer
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    a = make_pair(tmp_path / "a", seed=1); b = make_pair(tmp_path / "b", seed=2)
    cfg_b = base_cfg(b, tmp_path / "b", "b"); cfg_b["gaussian"]["init_support_path"] = a["support"]   # same support
    dev = torch.device("cuda", 0)
    ta = JointTrainer(base_cfg(a, tmp_path / "a", "a"), device=dev)
    tb = JointTrainer(cfg_b, device=dev)
    assert torch.equal(ta.model.positions, tb.model.positions)


@cuda
def test_support_file_shape_mismatch_raises(tmp_path):
    from trainers.joint_trainer import JointTrainer
    pair = make_pair(tmp_path)
    bad = tmp_path / "bad.npz"; np.savez_compressed(bad, support=np.ones((8, 8, 8), dtype=bool))
    cfg = base_cfg(pair, tmp_path, "bad"); cfg["gaussian"]["init_support_path"] = str(bad)
    with pytest.raises(ValueError, match="shape"):
        JointTrainer(cfg, device=torch.device("cuda", 0))


@cuda
def test_support_file_logs_own_support_fraction(tmp_path, capsys):
    from trainers.joint_trainer import JointTrainer
    pair = make_pair(tmp_path)
    JointTrainer(base_cfg(pair, tmp_path, "log"), device=torch.device("cuda", 0))
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("[init_cloud] mode=support_file"))
    assert "in_own_zf_support=" in line and "support_frac=" in line
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES=G $PY -m pytest tests/test_sg2x2_init_support.py -q -p no:cacheprovider`
Expected: FAIL with `ImportError: cannot import name 'load_init_support'`.

- [ ] **Step 4: Append the two functions to `paired_gs/gaussian/init_cloud.py`**

```python


# --- frozen init support (sg2x2 shared-geometry ablation, 2026-10-09) ------------------------------------------
def load_init_support(path: str, volume_shape: Tuple[int, int, int], device) -> torch.Tensor:
    """A frozen boolean (D, H, W) support from an .npz (key 'support'), e.g. the development subject's union-ZF
    head support written by scripts/sg2x2_make_init_support.py. Refuses a shape that is not this volume's."""
    import numpy as np
    with np.load(path) as z:
        sup = np.asarray(z["support"]).astype(bool)
    if tuple(sup.shape) != tuple(volume_shape):
        raise ValueError(f"init support {path} has shape {tuple(sup.shape)}, the volume is {tuple(volume_shape)}")
    return torch.from_numpy(sup).to(device)


def sample_positions_in_support(support: torch.Tensor, num_points: int,
                                generator: torch.Generator | None = None,
                                jitter: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniform draw of `num_points` voxels inside a boolean support (same conventions as sample_positions: with
    replacement only when the support is smaller than N, voxel centre plus a sub-voxel jitter, [-1, 1] coords).
    The draw depends only on (support, num_points, generator): nothing of the subject's data enters."""
    dev = support.device
    cand = torch.nonzero(support.flatten(), as_tuple=False).squeeze(-1)
    if cand.numel() == 0:
        raise RuntimeError("init support is empty")
    w = torch.ones(cand.numel(), device=dev)
    pick = torch.multinomial(w, num_points, replacement=cand.numel() < num_points, generator=generator)
    idx = cand[pick]
    d, h, wd = tuple(support.shape)
    z = idx // (h * wd)
    rem = idx % (h * wd)
    y = rem // wd
    x = rem % wd
    vox = torch.stack([z, y, x], dim=-1)
    off = torch.rand(num_points, 3, device=dev, generator=generator) - 0.5 if jitter else torch.zeros(num_points, 3, device=dev)
    pos = torch.stack([
        ((z.float() + 0.5 + off[:, 0]) / d) * 2 - 1,
        ((y.float() + 0.5 + off[:, 1]) / h) * 2 - 1,
        ((x.float() + 0.5 + off[:, 2]) / wd) * 2 - 1,
    ], dim=-1).clamp_(-1.0, 1.0)
    return vox, pos
```

- [ ] **Step 5: Insert the trainer branch**

In `paired_gs/trainers/joint_trainer.py`, between the `if init_mode == "uniform":` block (ends at the `iz = ...` line) and the existing `else:` (the head/intensity branch), insert:

```python
            elif init_mode == "support_file":
                # frozen support (sg2x2 ablation, 2026-10-09): positions depend only on the file, N and the seed,
                # so every cell of a subject starts from the same cloud; densities still come from each
                # contrast's own ZF below. The subject's own union-ZF support is read only for the log line.
                from gaussian.init_cloud import head_support, load_init_support, sample_positions_in_support, union_magnitude
                sup_path = str(gcfg.get("init_support_path", ""))
                if not sup_path:
                    raise ValueError("gaussian.init_mode=support_file needs gaussian.init_support_path")
                support = load_init_support(sup_path, self.volume_shape, self.device)
                with torch.no_grad():
                    gen = torch.Generator(device=self.device)
                    gen.manual_seed(int(self.config.get("training", {}).get("seed", 42)))
                    vox, positions = sample_positions_in_support(support, n_init, generator=gen)
                    ix, iy, iz = vox[:, 0], vox[:, 1], vox[:, 2]
                    own = head_support(union_magnitude(self.zf_p, self.zf_c), float(gcfg.get("init_head_frac", 0.05)))
                    in_own = float(own[ix, iy, iz].float().mean())
                print(f"[init_cloud] mode=support_file path={sup_path} n={n_init} "
                      f"support_frac={float(support.float().mean()):.4f} in_own_zf_support={in_own:.4f}")
```
The `with torch.no_grad(): d_p = self.zf_p[ix, iy, iz] ...` block and the scale block that follow the branches are unchanged and apply to this branch too.

- [ ] **Step 6: Run the init tests (the two that need `SingleContrastTrainer` still fail until Task 3)**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES=G $PY -m pytest tests/test_sg2x2_init_support.py -q -p no:cacheprovider -k "not joint_and_single"`
Expected: `6 passed, 1 deselected`.

- [ ] **Step 7: Write the support-file maker**

```python
# scripts/sg2x2_make_init_support.py
#!/usr/bin/env python3
"""Write the frozen init support for the sg2x2 ablation: the union-ZF head support of a config's pair (the rule of
gaussian.init_mode=head: union_magnitude > 0.05 * P99.9), binary-dilated, as a compressed .npz (key 'support').
Run it ONCE on the development subject; every cell and every later subject reads the same file.

    env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/sg2x2_make_init_support.py \
        --config configs/sg2x2/af16_jyz_joint.yaml --out configs/sg2x2/init_support_af16_jyz.npz --dilate 3
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paired_gs"))
from data.transforms import ifft3c                                   # noqa: E402
from gaussian.init_cloud import head_support, union_magnitude       # noqa: E402


def _zf(block: dict, mask: torch.Tensor) -> torch.Tensor:
    with h5py.File(block["data_path"], "r") as f:
        k = torch.from_numpy(f["kspace"][...]).to(torch.complex64)
    with h5py.File(block["csm_path"], "r") as f:
        s = torch.from_numpy(f["map"][...]).to(torch.complex64)
    return torch.sum(torch.conj(s) * ifft3c(k * mask.unsqueeze(0)), dim=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dilate", type=int, default=3, help="binary dilation iterations (voxels)")
    ap.add_argument("--head_frac", type=float, default=0.05)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    plain, contrast = cfg["data"]["plain"], cfg["data"]["contrast"]
    with h5py.File(plain["mask_path"], "r") as f:
        mask = torch.from_numpy(f["mask"][...]).float()
    mag = union_magnitude(_zf(plain, mask), _zf(contrast, mask))
    sup = head_support(mag, a.head_frac).numpy()
    if a.dilate > 0:
        from scipy.ndimage import binary_dilation
        sup = binary_dilation(sup, iterations=a.dilate)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    np.savez_compressed(out, support=sup.astype(bool))
    print(f"wrote {out} shape={sup.shape} support_frac={sup.mean():.4f} dilate={a.dilate} head_frac={a.head_frac} "
          f"md5={hashlib.md5(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Commit**

```bash
git add paired_gs/gaussian/init_cloud.py paired_gs/trainers/joint_trainer.py scripts/sg2x2_make_init_support.py tests/sg2x2_fixture.py tests/test_sg2x2_init_support.py
git commit -m "sg2x2: frozen init support (init_mode=support_file) so every cell starts from the same cloud"
```

---

### Task 3: `SingleContrastTrainer` — a true GS-single, one contrast per process

**Files:**
- Create: `paired_gs/trainers/single_contrast_trainer.py`
- Modify: `arc/cycle.py` (add `--trainer {joint,single}`; guard the three places that index `cfg["data"]["contrast"]`; skip the contrast score / volumes when single)
- Test: `tests/test_sg2x2_single_trainer.py`

**Interfaces:**
- Produces: `class SingleContrastTrainer(JointTrainer)` — same constructor `(config, device)`; `config["data"]` has only `plain`; refuses with `ValueError` any of `loss.lambda_dtv/lambda_edge/lambda_vtv/lambda_dtv_proj/edge_tv_guide_alpha/ptv_guide_alpha/lambda_nl/lambda_nl_plain/lambda_nl_contrast/lambda_private_sp > 0`, `loss.guide_path_plain/contrast`, `loss.kspace_high_weight_plain`, `adaptive_control.enable`, `gaussian.num_private_points > 0`, `gaussian.warm_start_checkpoint`, or a `data.contrast` block. After `_setup_data`: `ds_c is ds_p`, `kspace_c is kspace_p`, `csm_c is csm_p`, `mask_c is mask_p`, `zf_c is zf_p`, the four fit tensors and `_dc_norm_c` alias the plain ones. `_render("contrast")` raises `RuntimeError`. `train()` optimises `L_dc(plain) + lambda_geo R_geo(plain) + lambda_wav_plain R_wav(plain) + lambda_ptv R_ptv(plain)`, logs `[iter N/M] loss= dc_p= dc_c=0.0000e+00 R_geo= R_dtv=0.0000e+00 R_edge=0.0000e+00 R_vtv=0.0000e+00 R_dtvp=0.0000e+00 R_wav= R_ptv= R_nl=0.0000e+00 balance_violation=False` (the joint line's keys plus `R_ptv`), writes `joint_train.log`, `checkpoints/latest.pth` and `best.pth` like the joint trainer.
- Produces (`arc/cycle.py`): `--trainer joint|single` (default `joint`). Single: scores the plain channel only; the score json has `contrast: null`, `contrast_name: null`; `--save_volumes` writes `rec, gt, zf` (+ `rec_complex, gt_complex, zf_complex`) and `contrast_name`; no TSV row.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sg2x2_single_trainer.py
"""SingleContrastTrainer = one contrast per process. Pinned here: (1) its objective is the plain half of the
joint objective (a duplicate-slot joint run on the same contrast reads the same dc and twice each regulariser,
and the joint gradient of every shared tensor is one positive multiple of the single run's gradient, twice as
large for the geometry as for a density table), (2) nothing of a second channel exists (contrast
density parameters never receive a gradient, the contrast tensors are aliases), (3) every cross-contrast knob is
refused, (4) the FLAIR shutter (known_zero) builds the same fit tensors as the joint trainer's contrast channel,
(5) arc/cycle.py --trainer single produces a plain-only score json.
Run: PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=<idle> /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_single_trainer.py -q
"""
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paired_gs"))
sys.path.insert(0, str(ROOT / "tests"))
from sg2x2_fixture import base_cfg, make_pair  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="trainer needs the CUDA voxelizer")
PY = "/home/congcongliu/anaconda3/envs/GSTMI/bin/python"


def _dup_joint_cfg(pair, tmp, exp, contrast, reg=True):
    """the historical proxy: the same contrast in both slots of the joint trainer"""
    cfg = base_cfg(pair, tmp, exp, reg=reg)
    blk = cfg["data"]["plain"] if contrast == "T1" else cfg["data"]["contrast"]
    cfg["data"] = {"plain": copy.deepcopy(blk), "contrast": copy.deepcopy(blk)}
    return cfg


def _iter_line(out_dir: Path, exp: str, it: int) -> dict:
    txt = (out_dir / exp / "joint_train.log").read_text()
    line = next(l for l in txt.splitlines() if l.startswith(f"[iter {it:5d}/"))
    return {k: float(v) for k, v in re.findall(r"(\w+)=([0-9.e+-]+)", line)}


@cuda
@pytest.mark.parametrize("reg", [True, False])
def test_single_objective_is_the_plain_half_of_the_duplicate_slot_joint(tmp_path, reg):
    from trainers.joint_trainer import JointTrainer
    from trainers.single_contrast_trainer import SingleContrastTrainer
    pair = make_pair(tmp_path)
    dev = torch.device("cuda", 0)
    s = SingleContrastTrainer(base_cfg(pair, tmp_path, "s", reg=reg, single="FLAIR"), device=dev)
    j = JointTrainer(_dup_joint_cfg(pair, tmp_path, "j", "FLAIR", reg=reg), device=dev)
    s.train(); j.train()
    ls, lj = _iter_line(tmp_path / "out", "s", 1), _iter_line(tmp_path / "out", "j", 1)
    assert ls["dc_p"] == pytest.approx(lj["dc_p"], rel=1e-5) and lj["dc_c"] == pytest.approx(lj["dc_p"], rel=1e-5)
    assert ls["dc_c"] == 0.0
    for k in ("R_geo", "R_wav"):
        assert lj[k] == pytest.approx(2.0 * ls[k], rel=1e-4, abs=1e-12)
    if reg:
        assert ls["R_geo"] > 0 and ls["R_wav"] > 0 and ls["R_ptv"] > 0
    else:
        assert ls["R_geo"] == 0 and ls["R_wav"] == 0 and ls["R_ptv"] == 0
    # the single objective is the plain half of the joint objective, so after the one backward pass (clip_grad_norm_
    # rescales all gradients of a run by one positive constant) each shared tensor's joint gradient is one positive
    # multiple of the single run's gradient, and the geometry multiple is twice the density multiple (both DC halves
    # push the shared geometry, each density table sees one half). Rotations are left out: the 3-NN scales are
    # isotropic, so their gradient is identically zero at this step.
    def ratio(a, b):
        a, b = a.detach().flatten().double(), b.detach().flatten().double()
        r = float((a @ b) / (b @ b))
        return r, float(torch.linalg.norm(a - r * b) / torch.linalg.norm(a))
    r_pos, e_pos = ratio(j.model.positions.grad, s.model.positions.grad)
    r_scl, e_scl = ratio(j.model.scales.grad, s.model.scales.grad)
    r_dr, e_dr = ratio(j.model.density_real.grad, s.model.density_real.grad)
    r_di, e_di = ratio(j.model.density_imag.grad, s.model.density_imag.grad)
    assert max(e_pos, e_scl, e_dr, e_di) < 1e-4
    assert r_scl == pytest.approx(r_pos, rel=1e-4) and r_di == pytest.approx(r_dr, rel=1e-4)
    assert r_pos == pytest.approx(2.0 * r_dr, rel=1e-4)


@cuda
def test_single_has_no_second_channel(tmp_path):
    from trainers.single_contrast_trainer import SingleContrastTrainer
    pair = make_pair(tmp_path)
    s = SingleContrastTrainer(base_cfg(pair, tmp_path, "s", single="T1"), device=torch.device("cuda", 0))
    assert s.ds_c is s.ds_p and s.kspace_c is s.kspace_p and s.csm_c is s.csm_p and s.mask_c is s.mask_p
    assert s.zf_c is s.zf_p and s.kspace_fit_c is s.kspace_fit_p and s.mask_fit_c is s.mask_fit_p
    with pytest.raises(RuntimeError):
        s._render("contrast")
    s.train()
    assert s.model.density_real_contrast.grad is None and s.model.density_imag_contrast.grad is None
    assert s.model.positions.grad is not None


@cuda
@pytest.mark.parametrize("key,val", [
    (("loss", "lambda_edge"), 0.001), (("loss", "lambda_dtv"), 0.01), (("loss", "lambda_vtv"), 0.01),
    (("loss", "lambda_dtv_proj"), 0.01), (("loss", "edge_tv_guide_alpha"), 0.25), (("loss", "ptv_guide_alpha"), 0.25),
    (("loss", "guide_path_plain"), "/tmp/x.npy"), (("loss", "lambda_nl"), 0.1), (("loss", "kspace_high_weight_plain"), 3.0),
    (("adaptive_control", "enable"), True), (("gaussian", "num_private_points"), 10),
    (("gaussian", "warm_start_checkpoint"), "/tmp/x.pth"),
])
def test_single_refuses_every_cross_contrast_or_out_of_scope_knob(tmp_path, key, val):
    from trainers.single_contrast_trainer import SingleContrastTrainer
    pair = make_pair(tmp_path)
    cfg = base_cfg(pair, tmp_path, "r", single="T1")
    cfg.setdefault(key[0], {})[key[1]] = val
    with pytest.raises(ValueError, match=key[1]):
        SingleContrastTrainer(cfg, device=torch.device("cuda", 0))


@cuda
def test_single_refuses_a_contrast_block(tmp_path):
    from trainers.single_contrast_trainer import SingleContrastTrainer
    pair = make_pair(tmp_path)
    cfg = base_cfg(pair, tmp_path, "two")           # joint-style config with data.contrast
    with pytest.raises(ValueError, match="data.contrast"):
        SingleContrastTrainer(cfg, device=torch.device("cuda", 0))


@cuda
def test_single_known_zero_matches_joint_contrast_channel(tmp_path):
    """the FLAIR shutter: the single run's fit tensors equal the joint run's contrast-channel fit tensors"""
    from trainers.joint_trainer import JointTrainer
    from trainers.single_contrast_trainer import SingleContrastTrainer
    pair = make_pair(tmp_path)
    dev = torch.device("cuda", 0)
    cj = base_cfg(pair, tmp_path, "kj"); cj["data"]["contrast"]["known_zero"] = True
    cs = base_cfg(pair, tmp_path, "ks", single="FLAIR"); cs["data"]["plain"]["known_zero"] = True
    j = JointTrainer(cj, device=dev); s = SingleContrastTrainer(cs, device=dev)
    assert torch.equal(j.mask_fit_c, s.mask_fit_p) and torch.equal(j.kspace_fit_c, s.kspace_fit_p)
    assert torch.equal(j.csm_fit_c, s.csm_fit_p) and j._dc_norm_c == s._dc_norm_p
    assert s.physics_info.get("p", {}).get("known_zero") is True and "c" not in s.physics_info


@cuda
def test_cycle_single_writes_a_plain_only_score_json(tmp_path):
    pair = make_pair(tmp_path)
    cfg = base_cfg(pair, tmp_path, "cyc", single="T1", iters=2)
    cfg_path = tmp_path / "single_T1.yaml"; cfg_path.write_text(yaml.safe_dump(cfg))
    exp = f"sg2x2_test_cycle_single_{os.getpid()}"
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": ""}
    r = subprocess.run([PY, "-u", str(ROOT / "arc/cycle.py"), "--exp", exp, "--gpu", gpu, "--iters", "2", "--warmup", "0",
                        "--trainer", "single", "--base_cfg", str(cfg_path), "--lambda_geo", "0.10", "--edge_tv",
                        "--edge_tv_kappa_q", "0.85", "--edge_tv_wfloor", "0.10", "--tv_eps", "0.04", "--kspace_high", "2.2",
                        "--lambda_ptv", "0.03", "--save_volumes"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    row = json.load(open(ROOT / "results_arc" / f"{exp}_score.json"))
    assert row["contrast"] is None and row["contrast_name"] is None and row["plain_name"] == "T1"
    assert row["plain"]["head_psnr"] > 0
    import numpy as np
    z = np.load(ROOT / "results_arc" / f"{exp}_volumes.npz")
    assert {"rec", "gt", "zf", "rec_complex"} <= set(z.files) and str(z["contrast_name"]) == "T1"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES=G $PY -m pytest tests/test_sg2x2_single_trainer.py -q -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'trainers.single_contrast_trainer'`.

- [ ] **Step 3: Write the trainer**

```python
# paired_gs/trainers/single_contrast_trainer.py
"""SingleContrastTrainer: one contrast, its own Gaussian geometry, in its own process.

The GS-single arm of the shared-geometry 2x2 ablation (docs/experiments/2026-10-09-mcgs-shared-geometry-2x2-ablation.md).
It reuses JointTrainer's data loading, model construction (init modes, 3-NN scales), voxelizer, optimiser,
checkpointing and logging, and replaces two things:

  _setup_data  loads data.plain only; every `*_c` attribute JointTrainer exposes is an alias of the plain one so the
               inherited helpers (_setup_model's union_magnitude(zf_p, zf_c), the scorer's tr.kspace_c, ...) see one
               channel. Nothing of a second contrast is read.
  train        the plain half of JointTrainer.train's objective:
                   L = L_dc(plain) + lambda_geo * R_geo(plain) + lambda_wav_plain * R_wav(plain) + lambda_ptv * R_ptv(plain)
               with the same edge-TV kappa rule, the same wavelet cycle-spin generator and the same clip / scheduler /
               checkpoint schedule. The contrast density parameters that JointGaussianModel3D allocates are never
               rendered and never receive a gradient (asserted in tests/test_sg2x2_single_trainer.py).

Every cross-contrast weight, the private residual, densification, a file guide, a warm start and a data.contrast block
are refused at construction: this trainer exists to be a clean single-contrast control, not a general one.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import asdict
from typing import Any, Dict

import torch

from data.known_zero import build_fit_tensors, estimate_known_zero, estimate_noise_cov, known_zero_from_support, \
    load_support_map, whitening_matrix
from data.transforms import ifft3c
from losses.edge_tv import edge_preserving_tv, resolve_edge_kappa, tv_magnitude
from trainers.joint_trainer import JointTrainer, _load_dataset

_REFUSED_LOSS_WEIGHTS = ("lambda_dtv", "lambda_edge", "lambda_vtv", "lambda_dtv_proj", "edge_tv_guide_alpha",
                         "ptv_guide_alpha", "lambda_nl", "lambda_nl_plain", "lambda_nl_contrast", "lambda_private_sp")
_REFUSED_LOSS_KEYS = ("guide_path_plain", "guide_path_contrast", "kspace_high_weight_plain")


class SingleContrastTrainer(JointTrainer):
    """One contrast per process; see the module docstring."""

    def __init__(self, config: Dict[str, Any], device: torch.device | None = None):
        self._refuse_out_of_scope(config)
        super().__init__(config, device)

    # -- scope guard ---------------------------------------------------
    @staticmethod
    def _refuse_out_of_scope(config: Dict[str, Any]) -> None:
        data = config.get("data", {})
        if "contrast" in data:
            raise ValueError("SingleContrastTrainer takes data.plain only; remove the data.contrast block")
        if "plain" not in data:
            raise ValueError("SingleContrastTrainer needs data.plain")
        loss = config.get("loss", {})
        for k in _REFUSED_LOSS_WEIGHTS:
            if float(loss.get(k, 0.0) or 0.0) > 0:
                raise ValueError(f"loss.{k} is a cross-contrast / out-of-scope term; it must be 0 in a single-contrast run")
        for k in _REFUSED_LOSS_KEYS:
            if loss.get(k) not in (None, "", False):
                raise ValueError(f"loss.{k} is not allowed in a single-contrast run")
        if bool(config.get("adaptive_control", {}).get("enable", False)):
            raise ValueError("adaptive_control.enable must be false in a single-contrast run (fixed capacity)")
        g = config.get("gaussian", {})
        if int(g.get("num_private_points", 0)) > 0:
            raise ValueError("gaussian.num_private_points must be 0 in a single-contrast run")
        if g.get("warm_start_checkpoint"):
            raise ValueError("gaussian.warm_start_checkpoint is not allowed in a single-contrast run")

    # -- data: plain only, contrast attributes alias plain ---------------
    def _setup_data(self):
        data_cfg = self.config["data"]
        print(">>> loading PLAIN (single-contrast run)")
        self.ds_p = _load_dataset(data_cfg["plain"])
        self.ds_c = self.ds_p
        self.volume_shape = self.ds_p.volume_shape
        print(f"volume_shape = {self.volume_shape}")
        self.kspace_p = self.ds_p.kspace_undersampled_complex.to(self.device)
        self.csm_p = self.ds_p.sensitivity_maps.to(self.device)
        self.mask_p = self.ds_p.mask.to(self.device)
        self.kspace_c, self.csm_c, self.mask_c = self.kspace_p, self.csm_p, self.mask_p
        with torch.no_grad():
            self.zf_p = torch.sum(torch.conj(self.csm_p) * ifft3c(self.kspace_p), dim=0)
        self.zf_c = self.zf_p
        print(f"|zf_p| max {self.zf_p.abs().max().item():.4f}")

        # physics levers for the one channel, exactly as JointTrainer._setup_data does for tag "p"
        self.mask_fit_p, self.kspace_dc_p, self.kspace_fit_p, self.csm_fit_p = self.mask_p, self.kspace_p, self.kspace_p, self.csm_p
        self._dc_norm_p = max(self.mask_p.sum().item(), 1.0)
        self.physics_info: Dict[str, Any] = {}
        sub = data_cfg["plain"]
        use_kz = bool(sub.get("known_zero", False))
        use_pw = bool(sub.get("prewhiten", self.config.get("loss", {}).get("prewhiten", False)))
        if use_kz or use_pw:
            y, m, S = self.kspace_p, self.mask_p, self.csm_p
            sup_path = sub.get("known_zero_support")
            if sup_path:
                Z, info = known_zero_from_support(load_support_map(sup_path), y, m, int(self.ds_p.readout_axis))
            else:
                Z, info = estimate_known_zero(y, m, int(self.ds_p.readout_axis))
            W, winfo = None, {}
            if use_pw:
                Sigma, winfo = estimate_noise_cov(y, m, Z, info)
                W = whitening_matrix(Sigma) if Sigma is not None else None
            if not use_kz:
                Z = torch.zeros_like(m, dtype=torch.bool)
            self.mask_fit_p, self.kspace_dc_p, self.kspace_fit_p, self.csm_fit_p = build_fit_tensors(y, m, S, Z, W)
            rec = {"known_zero": use_kz, "prewhiten": bool(use_pw and W is not None), **asdict(info), **winfo,
                   "frac_z_used": float(Z.float().mean())}
            self.physics_info["p"] = rec
            print(f"[physics:plain] known_zero={use_kz} support={sup_path or 'estimated'} cuts={info.cuts} "
                  f"ellipse={info.ellipse} frac_z_used={rec['frac_z_used']:.4f} "
                  f"zero_lines={info.n_zero_lines}/{info.n_sampled_lines} prewhiten={rec['prewhiten']}")
        self.mask_fit_c, self.kspace_dc_c, self.kspace_fit_c, self.csm_fit_c = \
            self.mask_fit_p, self.kspace_dc_p, self.kspace_fit_p, self.csm_fit_p
        self._dc_norm_c = self._dc_norm_p

    # -- rendering: one channel -----------------------------------------
    def _render(self, channel: str) -> torch.Tensor:
        if channel != "plain":
            raise RuntimeError("SingleContrastTrainer has no contrast channel")
        return super()._render("plain")

    # -- train loop: the plain half of JointTrainer.train -----------------
    def train(self):
        train_cfg = self.config["training"]
        loss_cfg = self.config["loss"]
        max_iter = train_cfg.get("max_iterations", 8000)
        log_every = train_cfg.get("log_every", 50)
        save_every = train_cfg.get("save_every", 1000)
        eval_every = train_cfg.get("eval_every", 200)

        if "tv_weight" in loss_cfg and "lambda_geo" not in loss_cfg:
            warnings.warn("loss.tv_weight is deprecated; read as lambda_geo", DeprecationWarning, stacklevel=2)
            lambda_geo = float(loss_cfg["tv_weight"])
        else:
            lambda_geo = float(loss_cfg.get("lambda_geo", 5e-4))
        lambda_ptv = float(loss_cfg.get("lambda_ptv", 0.0))
        if lambda_ptv > 0:
            if not bool(loss_cfg.get("edge_tv", False)):
                raise ValueError("loss.lambda_ptv reuses the edge-TV kappa / w_floor; enable loss.edge_tv")
            from losses.phase_tv import phase_tv
            print(f"[phase-tv] lambda_ptv={lambda_ptv} eps={float(loss_cfg.get('tv_eps', 0.0))}")
        lambda_wav = float(loss_cfg.get("lambda_wav", 0.0))
        wav_lam_p = float(loss_cfg.get("lambda_wav_plain", lambda_wav))
        if lambda_wav > 0:
            from losses.wavelet_l1 import wavelet_l1
            wav_name = str(loss_cfg.get("wav_name", "db2"))
            wav_levels = int(loss_cfg.get("wav_levels", 3))
            wav_eps = float(loss_cfg.get("wav_eps", 0.0))
            wav_spin = bool(loss_cfg.get("wav_cycle_spin", True))
            wav_on = str(loss_cfg.get("wav_on", "complex"))
            wav_gen = torch.Generator().manual_seed(int(train_cfg.get("seed", 42)) + 7919)
            print(f"[wavelet] lambda={lambda_wav} (plain {wav_lam_p}) wavelet={wav_name} levels={wav_levels} "
                  f"eps={wav_eps} cycle_spin={wav_spin} on={wav_on}")

        self._static_freq_w = self._build_freq_weight()
        self._static_freq_w_plain = None
        self._edge_tv = bool(loss_cfg.get("edge_tv", False))
        self._edge_tv_wfloor = float(loss_cfg.get("edge_tv_wfloor", 0.1))
        self._tv_eps = float(loss_cfg.get("tv_eps", 0.0))
        self._tv_vector = bool(loss_cfg.get("tv_vector", False))
        if self._tv_eps > 0 or self._tv_vector:
            print(f"[tv_shape] eps={self._tv_eps} vector={self._tv_vector}")
        _kappa_refresh = 0
        if self._edge_tv:
            _kq = float(loss_cfg.get("edge_tv_kappa_q", 0.75))
            _ksrc = str(loss_cfg.get("edge_tv_kappa_source", "render"))
            _kappa_refresh = int(loss_cfg.get("edge_tv_kappa_refresh_iter", 0))
            with torch.no_grad():
                if _ksrc == "render":
                    self._edge_kappa_p = resolve_edge_kappa(self._render("plain"), _kq)
                elif _ksrc == "zf":
                    self._edge_kappa_p = resolve_edge_kappa(self.zf_p, _kq)
                else:
                    raise ValueError(f"loss.edge_tv_kappa_source must be 'render' or 'zf', got {_ksrc!r}")
            print(f"[edge_tv] kappa_q={_kq} source={_ksrc} refresh_iter={_kappa_refresh} -> "
                  f"kappa_plain={self._edge_kappa_p:.5f} w_floor={self._edge_tv_wfloor}")
        print(f"[single-contrast] lambda_geo={lambda_geo} edge_tv={self._edge_tv} lambda_wav_plain={wav_lam_p} "
              f"lambda_ptv={lambda_ptv} (every cross-contrast weight 0 by construction)")

        log_f = open(self.out_dir / "joint_train.log", "w")
        zero = torch.tensor(0.0, device=self.device)
        t0 = time.time()
        best_loss = float("inf")
        for it in range(1, max_iter + 1):
            self.optimizer.zero_grad()
            freq_w = self._freq_weight_curriculum(it, max_iter)
            vol_p = self._render("plain")
            if self._edge_tv and _kappa_refresh > 0 and it == _kappa_refresh:
                with torch.no_grad():
                    self._edge_kappa_p = resolve_edge_kappa(vol_p, float(loss_cfg.get("edge_tv_kappa_q", 0.75)))
                print(f"[edge_tv] kappa refreshed at iter {it}: kappa_plain={self._edge_kappa_p:.5f}")
            l_dc_p = self._kspace_loss(vol_p, self.csm_fit_p, self.kspace_fit_p, self.mask_fit_p, freq_w, norm=self._dc_norm_p)
            loss = l_dc_p
            if lambda_geo > 0:
                if self._edge_tv:
                    l_geo = edge_preserving_tv(vol_p, self._edge_kappa_p, self._edge_tv_wfloor, self._tv_eps, self._tv_vector)
                else:
                    l_geo = tv_magnitude(vol_p, self._tv_eps, self._tv_vector)
                loss = loss + lambda_geo * l_geo
            else:
                l_geo = zero
            if lambda_ptv > 0:
                l_ptv = phase_tv(vol_p, self._edge_kappa_p, self._edge_tv_wfloor, self._tv_eps)
                loss = loss + lambda_ptv * l_ptv
            else:
                l_ptv = zero
            if lambda_wav > 0:
                sh = tuple(torch.randint(0, 2 ** wav_levels, (3,), generator=wav_gen).tolist()) if wav_spin else None
                l_wav = wavelet_l1([vol_p], wav_name, wav_levels, wav_eps, sh, wav_on)
                loss = loss + wav_lam_p * l_wav
            else:
                l_wav = zero

            if self.writer is not None and (it % log_every == 0 or it == 1):
                self.writer.add_scalar("loss/total", float(loss.item()), it)
                self.writer.add_scalar("loss/dc_p", float(l_dc_p.item()), it)
                self.writer.add_scalar("loss/R_geo", float((lambda_geo * l_geo).item()), it)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg.get("max_grad_norm", 1.0))
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            if it % log_every == 0 or it == 1:
                msg = (f"[iter {it:5d}/{max_iter}] loss={loss.item():.4e} dc_p={l_dc_p.item():.4e} dc_c=0.0000e+00 "
                       f"R_geo={l_geo.item():.4e} R_dtv=0.0000e+00 R_edge=0.0000e+00 R_vtv=0.0000e+00 R_dtvp=0.0000e+00 "
                       f"R_wav={l_wav.item():.4e} R_ptv={l_ptv.item():.4e} R_nl=0.0000e+00 balance_violation=False")
                print(msg)
                log_f.write(msg + "\n"); log_f.flush()
            if it % eval_every == 0:
                with torch.no_grad():
                    snr_p = vol_p.abs().mean().item() / max(vol_p.abs().std().item(), 1e-8)
                    print(f"  eval @ iter {it}: |x_p|.mean/std={snr_p:.3f}")
            if it % save_every == 0 or it == max_iter:
                ckpt_dir = self.out_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                ckpt = {"iter": it, "model_state": self.model.state_dict(), "optim_state": self.optimizer.state_dict()}
                torch.save(ckpt, ckpt_dir / "latest.pth")
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(ckpt, ckpt_dir / "best.pth")
        print(f"single-contrast training done in {time.time() - t0:.1f}s")
        log_f.close()
        if self.writer is not None:
            self.writer.close()
```

- [ ] **Step 4: Add `--trainer` to `arc/cycle.py`**

Four edits, each guarded:

(a) After `ap.add_argument("--set", ...)`:
```python
ap.add_argument("--trainer", choices=["joint", "single"], default="joint")  # single = SingleContrastTrainer (sg2x2 GS-single: data.plain only)
```

(b) In `build_cfg`, replace the loop header `for s in ("plain", "contrast"):` with
```python
    for s in [k for k in ("plain", "contrast") if k in cfg["data"]]:
```
and the two `--known_zero` lines with
```python
    if args.known_zero:
        for s in [k for k in ("plain", "contrast") if k in cfg["data"]]:
            cfg["data"][s]["known_zero"] = True
```

(c) In `main()`, replace `name_c = cfg["data"]["contrast"].get("name", "contrast")` with
```python
    name_c = cfg["data"]["contrast"].get("name", "contrast") if "contrast" in cfg["data"] else None
```
and `tr = JointTrainer(cfg, device=dev)` with
```python
    if args.trainer == "single":
        from trainers.single_contrast_trainer import SingleContrastTrainer
        tr = SingleContrastTrainer(cfg, device=dev)
    else:
        tr = JointTrainer(cfg, device=dev)
```

(d) Replace the block from `sp = score_contrast(tr, "plain", dev)` through the end of `main()` with the version below (the joint path is byte-for-byte the old code; the single path is the new branch):
```python
    sp = score_contrast(tr, "plain", dev)
    vram = torch.cuda.max_memory_allocated() / 1e9
    npts = tr.model.num_points
    if args.trainer == "single":
        log(f"{name_p:5s} psnr={sp['psnr']:.3f} ssim={sp['ssim']:.4f} nmse={sp['nmse']:.4f} hfen={sp['hfen']:.4f} "
            f"e_unobs={sp['e_unobs']:.4f} head_psnr={sp['head_psnr']:.3f} head_ssim={sp['head_ssim']:.4f} "
            f"head_hfen={sp['head_hfen']:.4f} sharp={sp['sharp']:.3f} grain={sp['grain']:.4f} comp={sp['comp']:.3f} beat={sp['beat']}")
        log(f"SCORE single={sp['comp']:.3f} beat_zf={sp['beat']} N={npts} vram_gb={vram:.1f} elapsed_s={time.time()-t0:.0f}")
        row = dict(exp=args.exp, iters=args.iters, lambda_geo=args.lambda_geo, warmup=args.warmup,
                   lambda_dtv=args.lambda_dtv, max_radius=args.max_radius, kspace_high=args.kspace_high,
                   set=list(args.set), argv=sys.argv[1:], elapsed_s=time.time() - t0, trainer="single",
                   known_zero=bool(args.known_zero), prewhiten=bool(args.prewhiten), prewhiten_only=args.prewhiten_only,
                   plain=sp, contrast=None, plain_name=name_p, contrast_name=None,
                   combined=sp["comp"], both_beat=sp["beat"], npts=int(npts), vram_gb=vram)
        (RES / f"{args.exp}_score.json").write_text(json.dumps(row, indent=2))
        if args.save_volumes:
            with torch.no_grad():
                gt = tr.ds_p.get_data()["ground_truth"].to(dev)
                m = (tr.mask_p if tr.mask_p.ndim == 3 else tr.mask_p[0]).to(dev).float().unsqueeze(0)
                mf = (tr.mask_fit_p if tr.mask_fit_p.ndim == 3 else tr.mask_fit_p[0]).to(dev).float().unsqueeze(0)
                x = tr._render("plain"); k_model = fft3c(tr.csm_p * x.unsqueeze(0))
                x_dc = (ifft3c(mf * tr.kspace_dc_p + (1.0 - mf) * k_model) * tr.csm_p.conj()).sum(0)
                x_zf = (ifft3c(m * tr.kspace_p) * tr.csm_p.conj()).sum(0)
            f = lambda t: t.abs().cpu().numpy().astype("float32")
            fc = lambda t: t.cpu().numpy().astype("complex64")
            np.savez_compressed(RES / f"{args.exp}_volumes.npz", rec=f(x_dc), gt=f(gt), zf=f(x_zf),
                                rec_complex=fc(x_dc), gt_complex=fc(gt), zf_complex=fc(x_zf), contrast_name=name_p)
            log(f"  saved volumes -> {RES}/{args.exp}_volumes.npz")
        log(f"[arc {args.exp}] DONE (single-contrast)")
        return
    sc = score_contrast(tr, "contrast", dev)
    combined = 0.5 * (sp["comp"] + sc["comp"])
    both_beat = sp["beat"] and sc["beat"]
    ...  # the existing joint logging / json / tsv / volumes code, unchanged, follows here
```
Keep every existing joint line exactly as it is (the two `log(...)` lines, `row = dict(...)`, the TSV block, the `_vol` helper and the `np.savez_compressed(..., recT1=..., ...)` call). Only the three lines `vram = ...`, `npts = ...` move above the new branch (they do not depend on `sc`).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES=G $PY -m pytest tests/test_sg2x2_single_trainer.py tests/test_sg2x2_init_support.py -q -p no:cacheprovider`
Expected: `25 passed` (18 + 7; the parametrised refusal test counts 12). If `test_single_objective...` fails only because an `e_*` residual sits slightly above 1e-4 while the log components match, loosen that bound to `5e-4` and record the measured residuals in the commit message; a mismatch of the log components or of the 2:1 geometry-to-density ratio is a real bug.

- [ ] **Step 6: Commit**

```bash
git add paired_gs/trainers/single_contrast_trainer.py arc/cycle.py tests/test_sg2x2_single_trainer.py
git commit -m "sg2x2: SingleContrastTrainer (one contrast per process) and arc/cycle.py --trainer single"
```

---

### Task 4: Configs, the init support of the development subject, and the launcher

**Files:**
- Create: `configs/sg2x2/af16_jyz_joint.yaml`, `configs/sg2x2/af16_jyz_single_T1.yaml`, `configs/sg2x2/af16_jyz_single_FLAIR.yaml`
- Create: `configs/sg2x2/init_support_af16_jyz.npz` (made by the script, committed: ~100 KB)
- Create: `run_sg2x2.sh`
- Test: `tests/test_sg2x2_configs.py`

**Interfaces:**
- Produces: `run_sg2x2.sh EXP GPU CELL [ITERS] [SEED] [extra cycle.py args...]`; env `NPTS` (default 3600000), `INIT` (`support_file` default | `head`), `SUPPORT` (default `configs/sg2x2/init_support_af16_jyz.npz`), `SAVE_VOLUMES` (default 1), `WARMUP` (default 500). `EXP` must start with `sg2x2_`. Writes `results_arc/EXP.log`, appends to `results_arc/sg2x2_runs.log`, then calls `scripts/sg2x2_manifest.py EXP CELL` (Task 5).

- [ ] **Step 1: Write the three configs**

`configs/sg2x2/af16_jyz_joint.yaml` = the `data`, `gaussian`, `hardware`, `loss`, `mode`, `training`, `voxelizer` blocks of `configs/t1flair_v3_af16_jyz.yaml` copied verbatim (keep the comments), with the header replaced by:
```yaml
# sg2x2 cell A / C base: a frozen copy of configs/t1flair_v3_af16_jyz.yaml (jiangyizhou retrospective AF16, one
# vd-Poisson mask for both contrasts, FLAIR known_zero). The loss block is overridden by run_sg2x2.sh / scripts/sg2x2_cells.py;
# only data / gaussian / training / voxelizer matter here. Do not edit; copy if a new line needs a change.
```
and `output: {exp_name: sg2x2_joint, output_dir: results_arc/sg2x2_joint}`.

`configs/sg2x2/af16_jyz_single_T1.yaml`: the same file with the `data.contrast` block **deleted** and the header line `# sg2x2 cell B_T1 / D_T1 base: data.plain = T1 only (SingleContrastTrainer refuses a data.contrast block).`; `output.exp_name: sg2x2_single_T1`.

`configs/sg2x2/af16_jyz_single_FLAIR.yaml`: `data.plain` = the joint file's `data.contrast` block (name FLAIR, the FLAIR k-space / CSM / mask, `known_zero: true`), no `data.contrast`; header `# sg2x2 cell B_FLAIR / D_FLAIR base: data.plain = FLAIR only, with the shutter (known_zero) kept.`; `output.exp_name: sg2x2_single_FLAIR`.

- [ ] **Step 2: Write the config tests**

```python
# tests/test_sg2x2_configs.py
"""The three sg2x2 base configs: same data / gaussian / training / voxelizer as the AF16 topic config, single
configs carry one contrast only, FLAIR keeps its shutter, and the init support file exists with the volume's shape.
Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_configs.py -q
"""
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
TOPIC = yaml.safe_load(open(ROOT / "configs/t1flair_v3_af16_jyz.yaml"))
JOINT = yaml.safe_load(open(ROOT / "configs/sg2x2/af16_jyz_joint.yaml"))
T1 = yaml.safe_load(open(ROOT / "configs/sg2x2/af16_jyz_single_T1.yaml"))
FL = yaml.safe_load(open(ROOT / "configs/sg2x2/af16_jyz_single_FLAIR.yaml"))


def test_joint_base_equals_topic_config_outside_output():
    for k in ("data", "gaussian", "loss", "training", "voxelizer", "mode", "hardware"):
        assert JOINT[k] == TOPIC[k], k


def test_single_configs_carry_one_contrast_and_the_right_block():
    assert "contrast" not in T1["data"] and T1["data"]["plain"] == TOPIC["data"]["plain"]
    assert "contrast" not in FL["data"] and FL["data"]["plain"] == TOPIC["data"]["contrast"]
    assert FL["data"]["plain"]["name"] == "FLAIR" and FL["data"]["plain"].get("known_zero") is True
    assert T1["data"]["plain"]["name"] == "T1" and not T1["data"]["plain"].get("known_zero", False)
    for c in (T1, FL):
        for k in ("gaussian", "loss", "training", "voxelizer"):
            assert c[k] == TOPIC[k], k


def test_init_support_file_matches_the_volume():
    z = np.load(ROOT / "configs/sg2x2/init_support_af16_jyz.npz")
    sup = z["support"]
    assert sup.dtype == bool and sup.shape == (256, 256, 176)
    assert 0.3 < sup.mean() < 0.7       # the dev subject's union-ZF support covers ~43 % of the box; dilation adds a little


def test_launcher_refuses_existing_outputs(tmp_path):
    import subprocess
    exp = "sg2x2_refuse_test"
    (ROOT / "results_arc").mkdir(exist_ok=True)
    marker = ROOT / "results_arc" / f"{exp}_score.json"
    marker.write_text("{}")
    try:
        r = subprocess.run(["bash", str(ROOT / "run_sg2x2.sh"), exp, "0", "A", "1"], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 1 and "refusing to overwrite" in r.stderr
    finally:
        marker.unlink()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_sg2x2_configs.py -q -p no:cacheprovider`
Expected: FAIL (`FileNotFoundError` on the sg2x2 configs).

- [ ] **Step 4: Create the configs (Step 1) and make the support file**

Run (CPU, ~2 minutes, 3 GB RAM):
```bash
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= $PY scripts/sg2x2_make_init_support.py --config configs/sg2x2/af16_jyz_joint.yaml --out configs/sg2x2/init_support_af16_jyz.npz --dilate 0
```
Expected: `wrote configs/sg2x2/init_support_af16_jyz.npz shape=(256, 256, 176) support_frac=0.4273 dilate=0 head_frac=0.05 md5=...`. `--dilate 0` is deliberate (ruling R7 in the ledger): the undilated union-ZF support is exactly the support `init_mode=head` draws from on this subject, so cell A starts from the registered recipe's own cloud and every cell's `in_own_zf_support` reads 1.0000; Task 2 measured dilate 1 / 2 / 3 at 0.4811 / 0.5431 / 0.6035, i.e. a 3-voxel dilation would park 29 % of the cloud outside the subject's own support. The core test on the 16-pair set lives on another grid and needs its own support file anyway. Record the printed md5 in the Task 8 record.

- [ ] **Step 5: Write the launcher**

```bash
#!/usr/bin/env bash
# run_sg2x2.sh EXP GPU CELL [ITERS] [SEED] [extra arc/cycle.py args...]
#   One cell of the shared-geometry 2x2 ablation (docs/experiments/2026-10-09-mcgs-shared-geometry-2x2-ablation.md).
#   CELL = A | B_T1 | B_FLAIR | C | D_T1 | D_FLAIR (scripts/sg2x2_cells.py is the only place the cells are defined).
#   Env: NPTS (3600000), INIT (support_file | head), SUPPORT (configs/sg2x2/init_support_af16_jyz.npz),
#        SAVE_VOLUMES (1), WARMUP (500). Extra args come last and override (repeated flags: last wins).
#   Refuses to overwrite an existing run of the same name. Writes the manifest + regulariser audit afterwards.
set -uo pipefail
if [[ $# -lt 3 ]]; then echo "usage: $0 EXP GPU CELL [ITERS] [SEED] [extra cycle.py args...]" >&2; exit 2; fi
EXP=$1; GPU=$2; CELL=$3; ITERS=${4:-4000}; SEED=${5:-42}; shift 3; [[ $# -gt 0 ]] && shift; [[ $# -gt 0 ]] && shift
[[ "$EXP" == sg2x2_* ]] || { echo "EXP must start with sg2x2_: $EXP" >&2; exit 2; }
case "$CELL" in A|B_T1|B_FLAIR|C|D_T1|D_FLAIR) ;; *) echo "CELL must be A, B_T1, B_FLAIR, C, D_T1 or D_FLAIR: $CELL" >&2; exit 2 ;; esac
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
NPTS=${NPTS:-3600000}; INIT=${INIT:-support_file}; SUPPORT=${SUPPORT:-$ROOT/configs/sg2x2/init_support_af16_jyz.npz}
WARMUP=${WARMUP:-500}
cd "$ROOT"
for path in "results_arc/${EXP}.log" "results_arc/${EXP}_score.json" "results_arc/${EXP}"; do
  if [[ -e "$path" ]]; then echo "refusing to overwrite existing output: $path" >&2; exit 1; fi
done
mapfile -t CELLARGS < <(env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" scripts/sg2x2_cells.py "$CELL" --npts "$NPTS" --seed "$SEED" \
  --support "$SUPPORT" --init "$INIT" --iters "$ITERS" --warmup "$WARMUP")
[[ ${#CELLARGS[@]} -gt 0 ]] || { echo "scripts/sg2x2_cells.py produced no arguments for $CELL (a process substitution hides its exit code)" >&2; exit 2; }
VOL=(); if [[ "${SAVE_VOLUMES:-1}" == "1" ]]; then VOL=(--save_volumes); fi
mkdir -p results_arc
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID \
  "$PY" -u arc/cycle.py --exp "$EXP" --gpu "$GPU" "${CELLARGS[@]}" "${VOL[@]}" "$@" > "results_arc/${EXP}.log" 2>&1
RC=$?
echo "[$(date '+%F %H:%M')] $EXP cell=$CELL seed=$SEED npts=$NPTS init=$INIT iters=$ITERS extra=[$*] rc=$RC" >> results_arc/sg2x2_runs.log
if [[ $RC -eq 0 ]]; then
  env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" scripts/sg2x2_manifest.py "$EXP" "$CELL" --seed "$SEED" --npts "$NPTS" --init "$INIT" \
    --support "$SUPPORT" >> "results_arc/${EXP}.log" 2>&1 || echo "[$(date '+%F %H:%M')] $EXP manifest FAILED" >> results_arc/sg2x2_runs.log
fi
exit $RC
```
Note the positional handling: after `shift 3`, the two optional positionals (ITERS, SEED) are consumed only when present; extra args that start with `--` must therefore be given after both positionals (`run_sg2x2.sh EXP GPU CELL 4000 42 --set ...`).

- [ ] **Step 6: Run the config tests**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_sg2x2_configs.py -q -p no:cacheprovider`
Expected: `4 passed` (the launcher test needs `scripts/sg2x2_manifest.py` to exist only on success paths, so it passes now).

- [ ] **Step 7: A two-iteration end-to-end smoke of all four cell kinds on real data**

Run (idle card G, ~4 min each, sequential; ruling R10: C is included because the joint trainer's `edge_tv=false` path only runs there):
```bash
WARMUP=0 bash run_sg2x2.sh sg2x2_smoke_A G A 2 42
WARMUP=0 bash run_sg2x2.sh sg2x2_smoke_C G C 2 42
WARMUP=0 bash run_sg2x2.sh sg2x2_smoke_B_T1 G B_T1 2 42
WARMUP=0 bash run_sg2x2.sh sg2x2_smoke_D_FLAIR G D_FLAIR 2 42
grep -E '^\[init_cloud\]|^\[single-contrast\]|^\[edge_tv\]|^\[wavelet\]|^\[phase-tv\]|^\[iter +1/|DONE' results_arc/sg2x2_smoke_{A,C,B_T1,D_FLAIR}.log
```
Expected: all four logs reach `DONE`; every log shows `[init_cloud] mode=support_file ... n=3600000 support_frac=0.4273`; `in_own_zf_support` reads `1.0000` in the joint cells (the file is this subject's own union-ZF support) and a value below 1 in the single cells (ruling R12: a single run's "own" support is that one contrast's ZF support, and the union file also covers the voxels only the other contrast lights up — record the two values, they are diagnostics, not gates); `sg2x2_smoke_A.log` shows `[edge_tv]`, `[wavelet] lambda=0.1 (plain 0.12, contrast 0.1)`, `[phase-tv]`; `sg2x2_smoke_B_T1.log` shows `[single-contrast] lambda_geo=0.1 edge_tv=True lambda_wav_plain=0.12 lambda_ptv=0.03` and `[wavelet] lambda=0.12 (plain 0.12)`; `sg2x2_smoke_C.log` shows no `[edge_tv]` / `[wavelet]` / `[phase-tv]` line and its `[iter 1/2]` line has `R_geo=0.0000e+00 R_wav=0.0000e+00`; `sg2x2_smoke_D_FLAIR.log` shows `[single-contrast] lambda_geo=0.0 edge_tv=False lambda_wav_plain=0.0 lambda_ptv=0.0`, no `[edge_tv]` / `[wavelet]` / `[phase-tv]` line, and its `[iter 1/2]` line has `R_geo=0.0000e+00 R_wav=0.0000e+00 R_ptv=0.0000e+00`. (The manifest step prints a warning until Task 5 lands; that is expected here.)

- [ ] **Step 8: Commit**

```bash
git add configs/sg2x2/af16_jyz_joint.yaml configs/sg2x2/af16_jyz_single_T1.yaml configs/sg2x2/af16_jyz_single_FLAIR.yaml configs/sg2x2/init_support_af16_jyz.npz run_sg2x2.sh tests/test_sg2x2_configs.py
git commit -m "sg2x2: cell configs, the frozen init support of the development subject, and the launcher"
```

---

### Task 5: Manifest and regulariser audit

**Files:**
- Create: `scripts/sg2x2_manifest.py`
- Test: `tests/test_sg2x2_manifest.py`

**Interfaces:**
- Produces: `scripts/sg2x2_manifest.py EXP CELL --seed S --npts N --init I --support PATH [--arc results_arc]` writes `results_arc/EXP_manifest.json` with: `exp, cell, trainer, contrast, seed, npts, init, support_md5, git_head, git_dirty (list of paths from git status --porcelain), config_md5 (the cell's base config), data_md5 ({ksp,csm,mask} of every channel in the resolved config, read from results_arc/EXP/EXP/config.yaml), resolved (the 11 switches of resolved_switches read back from that config.yaml), startup_lines (the [edge_tv]/[wavelet]/[phase-tv]/[init_cloud]/[single-contrast] lines of the log), iter_first / iter_last (the parsed first and last [iter] lines), audit {resolved_ok, startup_ok, loss_ok, ok}`. Exit code 0 when `ok`, 3 otherwise (the launcher records both).
- Produces (importable): `parse_iter_line(line) -> dict`, `audit(cell, resolved_cfg, log_text) -> dict` (pure functions used by the tests).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sg2x2_manifest.py
"""The regulariser audit of scripts/sg2x2_manifest.py: a reg-off cell must show zero weighted priors in the resolved
config, no prior start-up line, and zero R_* on the first and last logged iteration; a reg-on cell must show the
frozen recipe. Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_manifest.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sg2x2_manifest import audit, parse_iter_line  # noqa: E402

LINE_ON = ("[iter     1/4000] loss=1.2345e-01 dc_p=1.0000e-01 dc_c=2.0000e-02 R_geo=3.0000e-03 R_dtv=0.0000e+00 "
           "R_edge=0.0000e+00 R_vtv=0.0000e+00 R_dtvp=0.0000e+00 R_wav=4.0000e-03 R_nl=0.0000e+00 balance_violation=False")
LINE_OFF = ("[iter  4000/4000] loss=1.0000e-01 dc_p=1.0000e-01 dc_c=0.0000e+00 R_geo=0.0000e+00 R_dtv=0.0000e+00 "
            "R_edge=0.0000e+00 R_vtv=0.0000e+00 R_dtvp=0.0000e+00 R_wav=0.0000e+00 R_ptv=0.0000e+00 R_nl=0.0000e+00 balance_violation=False")
RES_ON = {"lambda_geo": 0.10, "edge_tv": True, "lambda_wav": 0.10, "lambda_wav_plain": 0.12, "lambda_ptv": 0.03,
          "lambda_edge": 0.0, "lambda_dtv": 0.0, "lambda_vtv": 0.0, "lambda_dtv_proj": 0.0, "edge_tv_guide_alpha": 0.0,
          "ptv_guide_alpha": 0.0}
RES_OFF = {**{k: 0.0 for k in RES_ON}, "edge_tv": False}


def test_parse_iter_line():
    d = parse_iter_line(LINE_ON)
    assert d["it"] == 1 and d["R_geo"] == 3e-3 and d["R_wav"] == 4e-3 and d["dc_c"] == 2e-2


def test_audit_reg_on_cell_passes_with_recipe_lines():
    log = "[edge_tv] kappa_q=0.85 source=zf refresh_iter=0 -> kappa_plain=0.01 kappa_contrast=0.01 w_floor=0.1\n" \
          "[wavelet] lambda=0.1 (plain 0.12, contrast 0.1) wavelet=db4 levels=1 eps=0.0 cycle_spin=True on=complex\n" \
          "[phase-tv] lambda_ptv=0.03 eps=0.04\n" + LINE_ON + "\n" + LINE_ON.replace("    1/", " 4000/") + "\n"
    a = audit("A", {"loss": {"tv_weight": 0.10, **{k: v for k, v in RES_ON.items() if k != "lambda_geo"}}}, log)
    assert a["resolved_ok"] and a["startup_ok"] and a["loss_ok"] and a["ok"]


def test_audit_reg_off_cell_rejects_nonzero_logged_prior():
    log = LINE_ON + "\n" + LINE_OFF + "\n"
    a = audit("D_T1", {"loss": {"tv_weight": 0.0, **{k: v for k, v in RES_OFF.items() if k != "lambda_geo"}}}, log)
    assert a["resolved_ok"] and not a["loss_ok"] and not a["ok"]


def test_audit_rejects_regulariser_startup_lines_in_reg_off_cell():
    log = "[wavelet] lambda=0.1 (plain 0.1)\n" + LINE_OFF + "\n" + LINE_OFF + "\n"
    a = audit("D_FLAIR", {"loss": {"tv_weight": 0.0, **{k: v for k, v in RES_OFF.items() if k != "lambda_geo"}}}, log)
    assert not a["startup_ok"] and not a["ok"]


def test_audit_rejects_wrong_resolved_wavelet_weight():
    bad = dict(RES_ON); bad["lambda_wav_plain"] = 0.12         # FLAIR single scored with T1's weight: the old confound
    log = LINE_ON + "\n" + LINE_ON + "\n"
    a = audit("B_FLAIR", {"loss": {"tv_weight": 0.10, **{k: v for k, v in bad.items() if k != "lambda_geo"}}}, log)
    assert not a["resolved_ok"] and not a["ok"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_sg2x2_manifest.py -q -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'sg2x2_manifest'`.

- [ ] **Step 3: Write the manifest script**

```python
# scripts/sg2x2_manifest.py
#!/usr/bin/env python3
"""Provenance + regulariser audit of one sg2x2 run (called by run_sg2x2.sh after arc/cycle.py; re-runnable).

Reads results_arc/EXP.log and the trainer's resolved config results_arc/EXP/EXP/config.yaml, checks them against
scripts/sg2x2_cells.resolved_switches(CELL) in three ways (the spec asks for the weighted losses, not only the config):
  resolved_ok   every switch in the resolved config equals the cell's value
  startup_ok    a reg-on cell printed [edge_tv] / [wavelet] / [phase-tv]; a reg-off cell printed none of them, and no
                [dtv-proj] / [guide] / [nl] / [FLAIR-private] line appeared in any cell
  loss_ok       R_dtv / R_edge / R_vtv / R_dtvp / R_nl are 0 on the first and last logged iteration in every cell;
                in a reg-off cell R_geo / R_wav (and R_ptv when logged) are 0 there too
and writes results_arc/EXP_manifest.json. Exit 0 when ok, 3 when not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sg2x2_cells import CELLS, resolved_switches  # noqa: E402

_ITER = re.compile(r"^\[iter\s+(\d+)/(\d+)\]\s+(.*)$")
_KV = re.compile(r"(\w+)=([0-9.eE+-]+|True|False)")
STARTUP_PREFIXES = ("[edge_tv]", "[wavelet]", "[phase-tv]", "[init_cloud]", "[single-contrast]", "[dtv-proj]", "[guide]",
                    "[nl]", "[FLAIR-private]", "[tv_shape]")
CROSS_R = ("R_dtv", "R_edge", "R_vtv", "R_dtvp", "R_nl")
PRIOR_R = ("R_geo", "R_wav", "R_ptv")


def md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_iter_line(line: str) -> dict:
    m = _ITER.match(line.strip())
    if not m:
        raise ValueError(f"not an iter line: {line!r}")
    d = {"it": int(m.group(1)), "max_iter": int(m.group(2))}
    for k, v in _KV.findall(m.group(3)):
        d[k] = (v == "True") if v in ("True", "False") else float(v)
    return d


def _resolved_from_cfg(cfg: dict) -> dict:
    loss = cfg.get("loss", {})
    geo = loss["lambda_geo"] if "lambda_geo" in loss else loss.get("tv_weight", 0.0)
    out = {"lambda_geo": float(geo), "edge_tv": bool(loss.get("edge_tv", False))}
    for k in ("lambda_wav", "lambda_wav_plain", "lambda_ptv", "lambda_edge", "lambda_dtv", "lambda_vtv",
              "lambda_dtv_proj", "edge_tv_guide_alpha", "ptv_guide_alpha"):
        out[k] = float(loss.get(k, 0.0) or 0.0)
    if "lambda_wav_plain" not in loss:
        out["lambda_wav_plain"] = out["lambda_wav"]
    return out


def audit(cell: str, resolved_cfg: dict, log_text: str) -> dict:
    want = resolved_switches(cell)
    got = _resolved_from_cfg(resolved_cfg)
    resolved_ok = all(abs(float(got[k]) - float(want[k])) < 1e-12 for k in want)
    lines = log_text.splitlines()
    startup = [l for l in lines if l.startswith(STARTUP_PREFIXES)]
    reg = CELLS[cell].reg
    has = {p: any(l.startswith(p) for l in startup) for p in STARTUP_PREFIXES}
    startup_ok = (not any(has[p] for p in ("[dtv-proj]", "[guide]", "[nl]", "[FLAIR-private]"))
                  and (all(has[p] for p in ("[edge_tv]", "[wavelet]", "[phase-tv]")) if reg
                       else not any(has[p] for p in ("[edge_tv]", "[wavelet]", "[phase-tv]"))))
    iters = [parse_iter_line(l) for l in lines if _ITER.match(l.strip())]
    loss_ok = bool(iters)
    for d in ([iters[0], iters[-1]] if iters else []):
        if any(d.get(k, 0.0) != 0.0 for k in CROSS_R):
            loss_ok = False
        if not reg and any(d.get(k, 0.0) != 0.0 for k in PRIOR_R):
            loss_ok = False
    return {"resolved_ok": resolved_ok, "startup_ok": startup_ok, "loss_ok": loss_ok,
            "ok": resolved_ok and startup_ok and loss_ok, "want": want, "got": got,
            "startup_lines": startup, "iter_first": iters[0] if iters else None, "iter_last": iters[-1] if iters else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp"); ap.add_argument("cell", choices=sorted(CELLS))
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--npts", type=int, required=True)
    ap.add_argument("--init", required=True); ap.add_argument("--support", required=True)
    ap.add_argument("--arc", default="results_arc")
    a = ap.parse_args()
    arc = ROOT / a.arc
    log_text = (arc / f"{a.exp}.log").read_text()
    cfg = yaml.safe_load(open(arc / a.exp / a.exp / "config.yaml"))
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    git_dirty = [l[3:] for l in subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                               text=True).stdout.splitlines()]
    data_md5 = {}
    for ch, blk in cfg["data"].items():
        data_md5[ch] = {k: md5(blk[k]) for k in ("data_path", "csm_path", "mask_path") if blk.get(k)}
    res = audit(a.cell, cfg, log_text)
    cell = CELLS[a.cell]
    man = {"exp": a.exp, "cell": a.cell, "trainer": cell.trainer, "contrast": cell.contrast, "reg": cell.reg,
           "seed": a.seed, "npts": a.npts, "init": a.init, "support": a.support,
           "support_md5": md5(a.support) if a.init == "support_file" else None,
           "git_head": git_head, "git_dirty": git_dirty, "config": cell.config, "config_md5": md5(ROOT / cell.config),
           "data_md5": data_md5, "resolved": res["got"], "expected": res["want"], "startup_lines": res["startup_lines"],
           "iter_first": res["iter_first"], "iter_last": res["iter_last"],
           "audit": {k: res[k] for k in ("resolved_ok", "startup_ok", "loss_ok", "ok")}}
    (arc / f"{a.exp}_manifest.json").write_text(json.dumps(man, indent=2))
    print(f"[sg2x2-manifest] {a.exp} cell={a.cell} audit={man['audit']}")
    sys.exit(0 if man["audit"]["ok"] else 3)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass; re-audit the two Task 4 smokes**

Run:
```bash
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_sg2x2_manifest.py -q -p no:cacheprovider
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/sg2x2_manifest.py sg2x2_smoke_A A --seed 42 --npts 3600000 --init support_file --support configs/sg2x2/init_support_af16_jyz.npz
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/sg2x2_manifest.py sg2x2_smoke_D_FLAIR D_FLAIR --seed 42 --npts 3600000 --init support_file --support configs/sg2x2/init_support_af16_jyz.npz
```
Expected: `5 passed`; both manifest calls print `audit={'resolved_ok': True, 'startup_ok': True, 'loss_ok': True, 'ok': True}` and exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/sg2x2_manifest.py tests/test_sg2x2_manifest.py
git commit -m "sg2x2: run manifest with the regulariser audit (resolved config, start-up lines, logged losses)"
```

---

### Task 6: The 2x2 table

**Files:**
- Create: `scripts/sg2x2_table.py`
- Test: `tests/test_sg2x2_table.py`

**Interfaces:**
- Produces: `scripts/sg2x2_table.py --prefix sg2x2_pilot_af16 --seeds 42 [43 44] [--arc results_arc] [--out results_arc/<prefix>_table.md]`. Run names are `<prefix>_s<seed>_<CELL>`. Reads each `<exp>_score.json` (head PSNR / SSIM / HFEN from `plain` for single runs, `plain` = T1 and `contrast` = FLAIR for joint runs) and `<exp>_volumes.npz` for head NMSE (joint keys `recT1/gtT1/recFL/gtFL`, single keys `rec/gt`). Writes the spec §6 table: rows A, B, C, D (per contrast: PSNR, SSIM, NMSE, HFEN, mean ± SD over seeds), then `A-B`, `C-D`, `(A-B)-(C-D)` with per-seed paired differences (mean ± SD). Missing runs print `—`.
- Produces (importable): `collect(arc: Path, prefix: str, seeds: list[int]) -> dict[(cell, contrast, seed) -> dict metrics]`, `table(collected, seeds) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sg2x2_table.py
"""scripts/sg2x2_table.py: the four cells per contrast, the two shared-geometry differences and the interaction,
from fake score jsons. Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_table.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sg2x2_table import collect, table  # noqa: E402


def _score(h):
    return {"head_psnr": h, "head_ssim": 0.9, "head_hfen": 0.1, "psnr": h + 3}


def _write(arc, exp, joint, t1=None, fl=None):
    if joint:
        row = {"plain": _score(t1), "contrast": _score(fl), "plain_name": "T1", "contrast_name": "FLAIR"}
    else:
        row = {"plain": _score(t1 if t1 is not None else fl), "contrast": None,
               "plain_name": "T1" if t1 is not None else "FLAIR", "contrast_name": None}
    (arc / f"{exp}_score.json").write_text(json.dumps(row))


def test_collect_and_differences(tmp_path):
    arc = tmp_path
    for s, (a_t1, a_fl, b_t1, b_fl, c_t1, c_fl, d_t1, d_fl) in {42: (35.0, 34.0, 34.9, 33.95, 33.0, 32.0, 32.8, 31.9),
                                                                   43: (35.1, 34.1, 35.0, 34.05, 33.1, 32.1, 32.9, 32.0)}.items():
        _write(arc, f"p_s{s}_A", True, a_t1, a_fl); _write(arc, f"p_s{s}_C", True, c_t1, c_fl)
        _write(arc, f"p_s{s}_B_T1", False, t1=b_t1); _write(arc, f"p_s{s}_B_FLAIR", False, fl=b_fl)
        _write(arc, f"p_s{s}_D_T1", False, t1=d_t1); _write(arc, f"p_s{s}_D_FLAIR", False, fl=d_fl)
    got = collect(arc, "p", [42, 43])
    assert got[("A", "T1", 42)]["head_psnr"] == 35.0 and got[("B", "FLAIR", 43)]["head_psnr"] == 34.05
    assert np.isnan(got[("A", "T1", 42)]["head_nmse"])          # no volumes written in this test
    md = table(got, [42, 43])
    assert "| A |" in md and "| A−B |" in md and "| (A−B)−(C−D) |" in md
    assert "+0.100 ± 0.000" in md                                 # A-B on T1: 0.1 in both seeds
    assert "+0.050 ± 0.000" in md                                 # A-B on FLAIR
    assert "−0.100 ± 0.000" in md                                 # interaction on T1: (0.1) - (0.2)


def test_missing_runs_render_as_dash(tmp_path):
    md = table(collect(tmp_path, "none", [42]), [42])
    assert "—" in md
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_sg2x2_table.py -q -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'sg2x2_table'`.

- [ ] **Step 3: Write the table script**

```python
# scripts/sg2x2_table.py
#!/usr/bin/env python3
"""The 2x2 table of the shared-geometry ablation (spec section 5.3 / 6) from arc/cycle.py score jsons.

Runs are named <prefix>_s<seed>_<CELL>; a joint run scores T1 on its plain slot and FLAIR on its contrast slot, a
single run scores its one contrast on the plain slot. Head NMSE is recomputed from the saved volumes on the scorer's
head mask (the json has none), as scripts/hp_summarize.py does; NaN when no volumes were saved.
    env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/sg2x2_table.py --prefix sg2x2_pilot_af16 --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diag_artefact.headonly import head_mask  # noqa: E402

CELLS = ("A", "B", "C", "D")
CONTRASTS = ("T1", "FLAIR")
METRICS = (("head_psnr", "PSNR", "{:.3f}"), ("head_ssim", "SSIM", "{:.4f}"), ("head_nmse", "NMSE", "{:.5f}"),
           ("head_hfen", "HFEN", "{:.4f}"))


def head_nmse(rec: np.ndarray, gt: np.ndarray) -> float:
    rec = np.asarray(rec, dtype=np.float64); gt = np.asarray(gt, dtype=np.float64)
    head = head_mask(gt.astype(np.float32))
    den = float((gt[head] ** 2).sum())
    return float(((rec - gt)[head] ** 2).sum() / den) if den > 0 else float("nan")


def _metrics(score: dict, vol, rk: str, gk: str) -> dict:
    out = {k: float(score.get(k, float("nan"))) for k, _, _ in METRICS if k != "head_nmse"}
    out["head_nmse"] = head_nmse(vol[rk], vol[gk]) if vol is not None and rk in vol.files else float("nan")
    return out


def collect(arc: Path, prefix: str, seeds: list[int]) -> dict:
    got = {}
    for s in seeds:
        for cell in CELLS:
            if cell in ("A", "C"):
                exp = f"{prefix}_s{s}_{cell}"
                p = arc / f"{exp}_score.json"
                if not p.exists():
                    continue
                row = json.load(open(p))
                vp = arc / f"{exp}_volumes.npz"
                vol = np.load(vp) if vp.exists() else None
                got[(cell, "T1", s)] = _metrics(row["plain"], vol, "recT1", "gtT1")
                got[(cell, "FLAIR", s)] = _metrics(row["contrast"], vol, "recFL", "gtFL")
            else:
                for c in CONTRASTS:
                    exp = f"{prefix}_s{s}_{cell}_{c}"
                    p = arc / f"{exp}_score.json"
                    if not p.exists():
                        continue
                    row = json.load(open(p))
                    vp = arc / f"{exp}_volumes.npz"
                    vol = np.load(vp) if vp.exists() else None
                    got[(cell, c, s)] = _metrics(row["plain"], vol, "rec", "gt")
    return got


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    v = [x for x in vals if not math.isnan(x)]
    if not v:
        return float("nan"), float("nan")
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return m, sd


def _fmt(m: float, sd: float, fmt: str, signed: bool = False) -> str:
    if math.isnan(m):
        return "—"
    s = fmt.format(abs(m)) if signed else fmt.format(m)
    if signed:
        s = ("+" if m >= 0 else "−") + s
    return f"{s} ± {fmt.format(sd)}"


def table(got: dict, seeds: list[int]) -> str:
    lines = ["| cell | contrast | " + " | ".join(n for _, n, _ in METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for cell in CELLS:
        for c in CONTRASTS:
            vals = [got.get((cell, c, s)) for s in seeds]
            cells = []
            for k, _, fmt in METRICS:
                m, sd = _mean_sd([v[k] if v else float("nan") for v in vals])
                cells.append(_fmt(m, sd, fmt))
            lines.append(f"| {cell} | {c} | " + " | ".join(cells) + " |")
    diffs = (("A−B", "A", "B"), ("C−D", "C", "D"))
    for name, x, y in diffs:
        for c in CONTRASTS:
            cells = []
            for k, _, fmt in METRICS:
                d = [got[(x, c, s)][k] - got[(y, c, s)][k] if (x, c, s) in got and (y, c, s) in got else float("nan")
                     for s in seeds]
                m, sd = _mean_sd(d)
                cells.append(_fmt(m, sd, fmt, signed=True))
            lines.append(f"| {name} | {c} | " + " | ".join(cells) + " |")
    for c in CONTRASTS:
        cells = []
        for k, _, fmt in METRICS:
            d = []
            for s in seeds:
                keys = [(q, c, s) for q in CELLS]
                if all(kk in got for kk in keys):
                    d.append((got[keys[0]][k] - got[keys[1]][k]) - (got[keys[2]][k] - got[keys[3]][k]))
                else:
                    d.append(float("nan"))
            m, sd = _mean_sd(d)
            cells.append(_fmt(m, sd, fmt, signed=True))
        lines.append(f"| (A−B)−(C−D) | {c} | " + " | ".join(cells) + " |")
    n = len(seeds)
    return (f"Head-only metrics (|GT| > 0.05 P99.9), mean ± SD over {n} seed(s) {seeds}; differences are per-seed "
            f"paired differences. PILOT numbers on one subject are engineering checks, not evidence.\n\n" + "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--arc", default="results_arc"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arc = ROOT / a.arc
    md = table(collect(arc, a.prefix, a.seeds), a.seeds)
    out = Path(a.out) if a.out else arc / f"{a.prefix}_table.md"
    out.write_text(md)
    print(md); print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_sg2x2_table.py -q -p no:cacheprovider`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/sg2x2_table.py tests/test_sg2x2_table.py
git commit -m "sg2x2: the 2x2 result table with per-seed paired differences and the interaction"
```

---

### Task 7: Regression gate — the default path is unchanged

**Files:**
- Create: `scripts/sg2x2_regress_compare.py`
- Create: `results_arc/sg2x2_regress_after.log` (produced)

**Interfaces:**
- Produces: `scripts/sg2x2_regress_compare.py BEFORE_LOG AFTER_LOG` prints each compared number and `REGRESSION PASS` / `REGRESSION FAIL`; exit 0 / 4. Rules: `[iter 1]` components `dc_p, dc_c, R_geo, R_wav` within rel 1e-4; `[iter 50]` within rel 1e-2; the `T1`/`FLAIR` score lines' `head_psnr` within 0.02 dB and `psnr` within 0.02 dB.

- [ ] **Step 1: Re-run the reference smoke on the modified tree**

Run (same card class as Task 0; detached):
```bash
sed 's/sg2x2_regress_before/sg2x2_regress_after/' results_arc/sg2x2_regress_before.sh > results_arc/sg2x2_regress_after.sh
setsid nohup bash results_arc/sg2x2_regress_after.sh G > results_arc/sg2x2_regress_after.log 2>&1 < /dev/null &
```
Expected: ~5 minutes; the log ends with `DONE`. (This command line has no `--trainer` and no `support_file`, so it exercises only the pre-existing joint path.)

- [ ] **Step 2: Write the comparison script**

```python
# scripts/sg2x2_regress_compare.py
#!/usr/bin/env python3
"""Compare two arc/cycle.py logs of the same 50-iteration joint smoke (before / after the sg2x2 edits).
Usage: sg2x2_regress_compare.py results_arc/sg2x2_regress_before.log results_arc/sg2x2_regress_after.log"""
from __future__ import annotations

import re
import sys

ITER = re.compile(r"^\[iter\s+(\d+)/(\d+)\]\s+(.*)$")
KV = re.compile(r"(\w+)=([0-9.eE+-]+)")


def parse(path: str) -> dict:
    out = {}
    for line in open(path):
        m = ITER.match(line.strip())
        if m:
            out[f"iter{int(m.group(1))}"] = {k: float(v) for k, v in KV.findall(m.group(3))}
        elif re.match(r"^(T1|FLAIR)\s+psnr=", line):
            out[line.split()[0]] = {k: float(v) for k, v in KV.findall(line)}
    return out


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def main() -> None:
    before, after = parse(sys.argv[1]), parse(sys.argv[2])
    ok = True
    for it, tol in (("iter1", 1e-4), ("iter50", 1e-2)):
        for k in ("dc_p", "dc_c", "R_geo", "R_wav"):
            a, b = before[it][k], after[it][k]
            good = rel(a, b) <= tol or (a == 0 and b == 0)
            ok &= good
            print(f"{it:6s} {k:6s} before={a:.6e} after={b:.6e} rel={rel(a, b):.2e} tol={tol:g} {'ok' if good else 'FAIL'}")
    for ch in ("T1", "FLAIR"):
        for k in ("head_psnr", "psnr"):
            a, b = before[ch][k], after[ch][k]
            good = abs(a - b) <= 0.02
            ok &= good
            print(f"{ch:6s} {k:9s} before={a:.3f} after={b:.3f} diff={b - a:+.4f} tol=0.02 {'ok' if good else 'FAIL'}")
    print("REGRESSION PASS" if ok else "REGRESSION FAIL")
    sys.exit(0 if ok else 4)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Compare, run the whole suite, and list the diff of the three modified files**

Run:
```bash
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/sg2x2_regress_compare.py results_arc/sg2x2_regress_before.log results_arc/sg2x2_regress_after.log
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES=G $PY -m pytest tests -q --ignore=tests/smoke_test.py --ignore=tests/test_score_edge.py -p no:cacheprovider | tail -2
git diff --stat 9df60e7 -- paired_gs/trainers/joint_trainer.py paired_gs/gaussian/init_cloud.py arc/cycle.py
git diff 9df60e7 -- paired_gs/trainers/joint_trainer.py | grep -E '^[-+]' | grep -v -E '^(\+\+\+|---)' | grep -c '^-'
```
Expected: `REGRESSION PASS`; `223 + 45 = 268 passed` (9 + 7 + 18 + 4 + 5 + 2 new tests); the last command prints `0` (no line of `joint_trainer.py` was removed; the branch is purely inserted). If the regression fails on `iter50` only and `iter1` matches, re-run `after` once more: a second disagreement is a real change and must be found before Task 8.

- [ ] **Step 4: Commit**

```bash
git add scripts/sg2x2_regress_compare.py
git commit -m "sg2x2: regression comparison of the unchanged joint path (before / after the edits)"
```

---

### Task 8: Pilot — the six cells on jiangyizhou AF16, seed 42 (needs the user's go for the GPU time)

**Files:**
- Create: `scripts/sg2x2_queue.sh` (queue runner, same rules as `scripts/wd_queue.sh`: idle cards only, one trainer per card, 90 s between launches, skips names with any output)
- Create: `scripts/sg2x2_pilot_s42.txt` (the six queue lines)
- Create: `docs/superpowers/records/2026-10-09-sg2x2-implementation.md`
- Modify: `STATUS.md` (new top section), `CLAUDE.md` (new entry under the branch list)

**Interfaces:**
- Consumes: `run_sg2x2.sh`, `scripts/sg2x2_table.py`, the manifests.
- Produces: `results_arc/sg2x2_pilot_af16_s42_{A,B_T1,B_FLAIR,C,D_T1,D_FLAIR}.log / _score.json / _volumes.npz / _manifest.json` and `results_arc/sg2x2_pilot_af16_table.md`.

- [ ] **Step 1: Write the queue runner and the queue file**

`scripts/sg2x2_queue.sh` = `scripts/wd_queue.sh` with these substitutions: `PYPAT='^/home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py --exp sg2x2_'`, the `wd_` name check becomes `sg2x2_`, the launch line becomes `setsid nohup env "${ENVS[@]}" bash ./run_sg2x2.sh "$EXP" "$card" "$@" > "results_arc/${EXP}.launcher.log" 2>&1 < /dev/null &`, the pause file is `results_arc/.sg2x2_queue_pause`, `ALLOWED` defaults to `"0 1 2 3 4"`, `MAXRUN` to 5. Copy the file, apply exactly these edits, keep everything else.

`scripts/sg2x2_pilot_s42.txt`:
```
# EXP CELL ITERS SEED  (run_sg2x2.sh positional order after EXP GPU)
sg2x2_pilot_af16_s42_A A 4000 42
sg2x2_pilot_af16_s42_C C 4000 42
sg2x2_pilot_af16_s42_B_T1 B_T1 4000 42
sg2x2_pilot_af16_s42_B_FLAIR B_FLAIR 4000 42
sg2x2_pilot_af16_s42_D_T1 D_T1 4000 42
sg2x2_pilot_af16_s42_D_FLAIR D_FLAIR 4000 42
```

- [ ] **Step 2: Ask the user before launching**

Report: six runs, ~1 h each on an idle A100, cards 0–4 idle at the time of writing (5/6/7 belong to other sessions), queue command below. Launch only after the user says go:
```bash
setsid nohup bash scripts/sg2x2_queue.sh scripts/sg2x2_pilot_s42.txt > results_arc/sg2x2_queue_pilot_s42.log 2>&1 < /dev/null &
```
Wait with a loop anchored on the python path (`pgrep -f '^/home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py --exp sg2x2_pilot'`), never on a bare name.

- [ ] **Step 3: Acceptance checks when the queue drains**

Run:
```bash
for e in A C B_T1 B_FLAIR D_T1 D_FLAIR; do echo -n "$e: "; $PY -c "import json;m=json.load(open('results_arc/sg2x2_pilot_af16_s42_$e'+'_manifest.json'));print(m['audit'], 'dirty' if m['git_dirty'] else 'clean')"; done
grep -h -E '^\[init_cloud\] mode=support_file' results_arc/sg2x2_pilot_af16_s42_*.log | sort | uniq -c
grep -l -i -E 'nan|out of memory' results_arc/sg2x2_pilot_af16_s42_*.log || echo "no NaN / OOM"
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY scripts/sg2x2_table.py --prefix sg2x2_pilot_af16 --seeds 42
```
Expected: all six audits `ok: True` and `clean`; the six `[init_cloud]` lines agree on `n=3600000 support_frac=0.4273`, with `in_own_zf_support=1.0000` in A and C and the two single-cell values recorded (ruling R12; T1 cells share one value, FLAIR cells the other); no NaN / OOM; the table prints eight cell rows, A−B, C−D and the interaction, both contrasts.

- [ ] **Step 4: Write the record, STATUS and CLAUDE entries**

`docs/superpowers/records/2026-10-09-sg2x2-implementation.md`, sections: 1 what was built (files, the design choice: one contrast per process; frozen support instead of an ellipsoid, with the 2026-10-09 measurement: union-ZF support 0.4273 of the box, bbox = the whole volume, inscribed ellipsoid covers only 0.895 of it); 2 regression reference and result (the four lines from Task 0 and the `REGRESSION PASS` output); 3 test counts; 4 the pilot table verbatim with "PILOT, one subject, seed 42, NOT EVIDENCE"; 5 what is not done (seeds 43/44, the 1.8M capacity supplement `NPTS=1800000`, `INIT=head` end-to-end control, the 16-pair core test and its per-subject DC-centring check and masks, the equal-budget per-arm λ search).

`STATUS.md`: a new top section `# STATUS — 2026-10-09 (worktree mcgs_sg2x2_wt, branch exp/shared-geometry-2x2-ablation-20261009) — sg2x2 four-cell ablation` with the five fixed parts (done + verified with the commands above; waiting for the user: push, seeds 43/44, supplement runs; next; pitfalls: `--set` values are YAML, `run_sg2x2.sh` positional order, the audit's exit code 3; why: one contrast per process is the only reading of "真正独立" that needs no new model class).

`CLAUDE.md`: under the branch list, three lines: branch, entry (`STATUS.md` top → spec → plan → record), launcher and table commands.

- [ ] **Step 5: Commit and tag**

```bash
git add scripts/sg2x2_queue.sh scripts/sg2x2_pilot_s42.txt docs/superpowers/records/2026-10-09-sg2x2-implementation.md STATUS.md CLAUDE.md
git commit -m "sg2x2: pilot queue, record, STATUS and CLAUDE entries"
git tag handoff/2026-10-09-sg2x2-pilot
```
Do not push; the user decides.

---

## Self-review notes

- Spec coverage: §1 (true independent GS-single → Task 3; no cross-contrast weight in any cell → Task 1 `CROSS_OFF` + Task 3 refusals + Task 5 audit), §2 (cells → Task 1; `−Reg` keeps DC and the k-space weighting → Task 1 `COMMON`), §3 (recipe values → Task 1; FLAIR's own wavelet weight → Task 1 `_wavelet` + Task 5 test; resolved config audited → Task 5), §4 (per-contrast capacity 3.6M and the 1.8M supplement → launcher `NPTS`; matched init independent of the subject's ZF → Task 2; densities from own ZF → Task 2 test; 4000 steps, optimiser, clip, scheduler identical → Task 3 reuses the base; adaptive control and private points off → Task 1 + Task 3 refusals; seeds paired → launcher `SEED`), §5.1 pilot → Task 8; §5.2 metrics (head PSNR/SSIM/NMSE/HFEN → Task 6; `e_unobs` is in the score json; raw-render image metrics are not produced (descoped, see the record)), §5.3 (A−B, C−D, interaction → Task 6), §6 template → Task 6, §7 (the historical proxy is not used; Task 3's equivalence test documents its relation).
- Not covered by this plan, by design: multi-subject core test (data vetting, masks per grid, DC centring), seeds 43/44 beyond the queue file pattern, the equal-budget λ search, statistics across subjects (Task 6 aggregates over seeds only).
- Type consistency: `cell_args` / `resolved_switches` (Task 1) are consumed by `run_sg2x2.sh` (Task 4) and `sg2x2_manifest.py` (Task 5) with the signatures given; `load_init_support` / `sample_positions_in_support` (Task 2) by the trainer branch; the single run's volume keys `rec/gt/zf` (Task 3) by `sg2x2_table.py` (Task 6).
