# AF12→AF10 Quality — Plan 1: Phase 0 (instrumentation + gates) and Phase 1 (physics levers)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pin the pre-registered acceptance gates for "AF=12 reaches AF=10 quality", then add the two physics levers (known-zero k-space region Z, coil-noise prewhitening W) to MC-GS and run them at full budget against those gates.

**Architecture:** A new head-only metric module (`diag_artefact/headonly.py`) feeds both an offline npz scorer (used to compute the AF10 reference values) and the training-time scorer in `arc/cycle.py`. Z and W are estimated from the acquired samples only (`paired_gs/data/known_zero.py`, pure functions), and wired into `JointTrainer` as *fit-side* tensors (`mask_fit_*`, `kspace_dc_*`, `kspace_fit_*`, `csm_fit_*`) whose defaults are the original tensors, so every run without the switches stays bit-identical. Gates live in `docs/superpowers/records/af12_to_af10/gates.json`, verdicts in `decisions.jsonl`.

**Tech Stack:** Python 3.11 (`/home/congcongliu/anaconda3/envs/GSTMI/bin/python`, torch 2.7.1+cu118, numpy, scipy 1.15, scikit-image 0.26, pytest 9), the repo's CUDA tile voxelizer, 8 idle 80 GB GPUs.

**Spec:** `docs/superpowers/specs/2026-09-05-af12-to-af10-quality-design.md` (sections §2 gates, §3 Phase 0, §4 Phase 1). Read it first; this plan argues from it.

## Global Constraints

- Repo: `/data0/congcong/code/GS/Paired_ARC_GS_general_framework`, branch `af12-to-af10`. Every command below runs from the repo root.
- Interpreter and env for EVERY python invocation (tests, scripts, training):
  `PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID /home/congcongliu/anaconda3/envs/GSTMI/bin/python`
  (`PYTHONNOUSERSITE=1` hides a broken torch in `~/.local`; never drop it.)
- pytest: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest <path> -v`.
- **No learnable prior anywhere** (spec §1.1). Nothing in this plan trains anything.
- **Defaults are bit-identical**: with `known_zero` and `prewhiten` off, every tensor the trainer uses must be the *same object* as before and the loss denominator must equal `mask.sum()` (spec §1.2, §4.1).
- **Z and Σ are estimated from acquired samples only** — never from `ds.kspace_multicoil` (the oracle full k-space) and never from the GT (spec §4.1). The oracle may only be used by `e_unobs` and by *diagnostics that are labelled as such*.
- Frozen working point, always passed on the CLI because `arc/cycle.py` defaults clobber the yaml:
  `--lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 --base_cfg configs/winner_v2_edgeTV.yaml`
- `results_arc/` is git-ignored except `LOOK_*.png` and `diag_artefact/*.json`. Every score JSON that a decision cites must be COPIED into `docs/superpowers/records/af12_to_af10/scores/` and committed. Directory names under `docs/` must not be `results*` or `logs*` (both are ignored at any depth).
- Commits: local repo identity (Congcong Liu), conventional prefixes (`feat:`, `test:`, `docs:`, `records:`), **no AI co-author trailers or "generated with" lines**.
- Deviation from spec §4.3, recorded here: no new `configs/af12_kz*.yaml`. The switches are CLI flags on the same base config (`--known_zero`, `--prewhiten`); the exact command lines are versioned in the records instead. Equivalent, less duplication.

---

## File Structure

| File | Responsibility |
|---|---|
| `diag_artefact/headonly.py` (new) | Head mask + head-only PSNR/SSIM/HFEN + grain ratio N. Pure numpy. Imported by the offline scorer and by `arc/cycle.py`. |
| `tests/diag_artefact/test_headonly.py` (new) | Unit tests for the above on the existing `phantom` fixture. |
| `scripts/score_volumes_headonly.py` (new) | CLI: `*_volumes.npz` → head-only JSON (both contrasts, recon and ZF). Used for the AF10/AF12 reference values. |
| `scripts/make_gates.py` (new) | Build `gates.json` from reference / repeat / baseline head-only JSONs + the baseline `e_unobs` replay JSON. |
| `scripts/eval_gates.py` (new) | Evaluate one `arc/cycle.py` score JSON against `gates.json`; write a verdict JSON; optionally append to `decisions.jsonl`. |
| `tests/diag_artefact/test_gates_scripts.py` (new) | Tests for `make_gates.py` / `eval_gates.py` on synthetic JSONs. |
| `paired_gs/data/known_zero.py` (new) | `estimate_known_zero`, `estimate_noise_cov`, `whitening_matrix`, `build_fit_tensors`. Pure torch/numpy, no trainer state. |
| `tests/test_known_zero.py` (new) | Synthetic band-limited / shutter / unfiltered k-space; hard-DC zero property; whitening identities. |
| `paired_gs/trainers/joint_trainer.py` (modify) | `_setup_data` builds the fit-side tensors; `_kspace_loss` gains `norm=`; train loop and `export_results` use the fit-side tensors. |
| `arc/cycle.py` (modify) | `--known_zero`, `--prewhiten`; scorer reports head-only metrics, `kz_energy`, physics diagnostics; hard-DC uses fit-side tensors, `e_unobs`/ZF keep the originals. |
| `docs/superpowers/records/af12_to_af10/` (new) | `gates.json`, `decisions.jsonl`, `reference_table.md`, `scores/*.json`, `2026-09-XX-phase1-physics.md`. |
| `docs/figures/af12_to_af10/` (new) | 3-view and butterfly figures of the Phase 1 runs. |

---

### Task 1: Head-only metrics module

**Files:**
- Create: `diag_artefact/headonly.py`
- Test: `tests/diag_artefact/test_headonly.py`

**Interfaces:**
- Consumes: `diag_artefact.volumes.brain_mask(gt, rel=0.05)`, `diag_artefact.volumes.wm_mask(gt, brain)` (existing).
- Produces:
  - `head_mask(gt: np.ndarray) -> np.ndarray[bool]`
  - `log_kernel(sigma: float = 1.5, size: int = 15) -> np.ndarray`
  - `head_psnr(rec, gt, head, gmax) -> float`, `head_ssim(rec, gt, head, gmax) -> float`, `head_hfen(rec, gt, head, gmax) -> float`, `grain_ratio(rec, gt) -> float`
  - `score_headonly(rec: np.ndarray, gt: np.ndarray) -> dict` with keys `head_psnr, head_ssim, head_hfen, grain_ratio, head_frac, gmax` (all Python floats).

- [ ] **Step 1: Write the failing tests**

`tests/diag_artefact/test_headonly.py`:

```python
"""Head-only metrics (spec 2026-09-05 §2.1). The reference values in gates.json and every candidate
are scored by THIS code, so the tests pin the definitions, not just the plumbing."""
import numpy as np
import pytest
from scipy import ndimage

from diag_artefact.headonly import (grain_ratio, head_hfen, head_mask, head_psnr, head_ssim,
                                    log_kernel, score_headonly)


def test_head_mask_is_five_percent_of_p999(phantom):
    head = head_mask(phantom)
    p999 = np.quantile(phantom, 0.999)
    assert head.dtype == bool
    assert np.array_equal(head, phantom > 0.05 * p999)
    assert 0.2 < head.mean() < 0.6          # the ball, not the whole box, not empty


def test_identical_volumes_are_perfect(phantom):
    s = score_headonly(phantom, phantom)
    assert s["head_psnr"] == float("inf")
    assert s["head_ssim"] == pytest.approx(1.0, abs=1e-9)
    assert s["head_hfen"] == pytest.approx(0.0, abs=1e-12)
    assert s["grain_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert s["gmax"] == pytest.approx(float(phantom.max()))


def test_head_psnr_ignores_background_error(phantom):
    head = head_mask(phantom)
    rec = phantom.copy()
    rec[~head] += 0.3                           # wreck the background only
    assert head_psnr(rec, phantom, head, float(phantom.max())) == float("inf")


def test_head_hfen_ranks_high_frequency_error_above_smooth_error(phantom):
    rng = np.random.default_rng(3)
    head = head_mask(phantom)
    gmax = float(phantom.max())
    hf = (phantom + np.where(head, rng.normal(0, 0.02, phantom.shape), 0.0)).astype(np.float32)
    lf = (phantom + np.where(head, 0.02 * ndimage.gaussian_filter(rng.normal(0, 1, phantom.shape), 6.0), 0.0)).astype(np.float32)
    assert head_hfen(phantom, phantom, head, gmax) == 0.0
    assert head_hfen(hf, phantom, head, gmax) > head_hfen(lf, phantom, head, gmax)


def test_head_psnr_formula_and_no_rescaling(phantom):
    head = head_mask(phantom)
    rec = phantom * 1.1                          # a pure gain must COST psnr (no LS amplitude projection)
    gmax = float(phantom.max())
    mse = float(np.mean((rec[head] - phantom[head]) ** 2))
    expect = 20 * np.log10(gmax) - 10 * np.log10(mse)
    assert head_psnr(rec, phantom, head, gmax) == pytest.approx(expect, abs=1e-9)
    assert np.isfinite(expect)


def test_head_ssim_is_mean_of_map_inside_head(phantom):
    from skimage.metrics import structural_similarity
    rng = np.random.default_rng(1)
    rec = (phantom + rng.normal(0, 0.02, phantom.shape)).astype(np.float32)
    head = head_mask(phantom)
    gmax = float(phantom.max())
    _, smap = structural_similarity(rec, phantom, data_range=gmax, full=True)
    assert head_ssim(rec, phantom, head, gmax) == pytest.approx(float(smap[head].mean()), abs=1e-9)
    assert 0.3 < head_ssim(rec, phantom, head, gmax) < 1.0


def test_log_kernel_matches_paired_gs_definition():
    k = log_kernel(1.5, 15)
    assert k.shape == (15, 15, 15)
    assert abs(k.sum()) < 1e-12                 # DC-removed
    assert k[7, 7, 7] < 0                       # centre of a LoG is negative
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "paired_gs"))
        from metrics.metrics import _log3d_kernel
    except Exception:
        pytest.skip("paired_gs metrics not importable here")
    assert np.allclose(k, _log3d_kernel(1.5, 15), atol=1e-15)


def test_grain_ratio_orders_smooth_and_noisy(phantom):
    rng = np.random.default_rng(2)
    smooth = ndimage.gaussian_filter(phantom, 1.0).astype(np.float32)
    noisy = (phantom + rng.normal(0, 0.03, phantom.shape)).astype(np.float32)
    assert grain_ratio(smooth, phantom) < 1.0
    assert grain_ratio(noisy, phantom) > 1.0


def test_score_headonly_returns_python_floats(phantom):
    s = score_headonly(phantom * 0.98, phantom)
    for k in ("head_psnr", "head_ssim", "head_hfen", "grain_ratio", "head_frac", "gmax"):
        assert type(s[k]) is float, k
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/diag_artefact/test_headonly.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'diag_artefact.headonly'`.

- [ ] **Step 3: Implement `diag_artefact/headonly.py`**

```python
"""Head-only image metrics for the AF12->AF10 gates (spec 2026-09-05 §2.1).

Definitions are frozen here because the reference values in gates.json and every candidate are
scored by this code; a change here silently moves the goalposts.

    head      = |GT| > 0.05 * P99.9(|GT|)            (same as the 2026-09-04 audit probe1)
    head_psnr = 20 log10(max|GT|) - 10 log10( mean over head of (rec - gt)^2 )   -- NO amplitude rescaling
    head_ssim = mean over head of the skimage SSIM map (7^3 uniform window, K1 .01, K2 .03, data_range max|GT|)
    head_hfen = || (LoG rec - LoG gt)[head] ||_2 / || LoG gt [head] ||_2, LoG sigma 1.5, 15 taps, DC-removed
    N         = rms(hp(rec)[wm]) / rms(hp(gt)[wm]),  hp(v) = v - gaussian_filter(v, 4),  wm = memory §14c mask
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity

from .volumes import brain_mask, wm_mask

HEAD_REL = 0.05
LOG_SIGMA = 1.5
LOG_SIZE = 15
HP_SIGMA = 4.0


def head_mask(gt: np.ndarray) -> np.ndarray:
    p999 = float(np.quantile(gt, 0.999))
    return gt > HEAD_REL * p999


def log_kernel(sigma: float = LOG_SIGMA, size: int = LOG_SIZE) -> np.ndarray:
    """Transcribed from paired_gs/metrics/metrics.py::_log3d_kernel (DC-removed 3-D LoG). Kept in
    sync by test_log_kernel_matches_paired_gs_definition."""
    if size % 2 == 0 or size < 3:
        raise ValueError(f"size must be odd and >= 3, got {size}")
    radius = size // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    r2 = x * x + y * y + z * z
    s2 = sigma * sigma
    gauss = np.exp(-r2 / (2.0 * s2))
    gauss /= gauss.sum()
    log = ((r2 / s2) - 3.0) / s2 * gauss
    log -= log.mean()
    return log


def head_psnr(rec: np.ndarray, gt: np.ndarray, head: np.ndarray, gmax: float) -> float:
    mse = float(np.mean((rec[head].astype(np.float64) - gt[head].astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(gmax) - 10.0 * np.log10(mse))


def head_ssim(rec: np.ndarray, gt: np.ndarray, head: np.ndarray, gmax: float) -> float:
    _, smap = structural_similarity(rec, gt, data_range=gmax, full=True)
    return float(smap[head].mean())


def head_hfen(rec: np.ndarray, gt: np.ndarray, head: np.ndarray, gmax: float) -> float:
    k = log_kernel()
    lr = ndimage.convolve(rec.astype(np.float64) / gmax, k, mode="reflect")
    lg = ndimage.convolve(gt.astype(np.float64) / gmax, k, mode="reflect")
    den = float(np.linalg.norm(lg[head]))
    if den == 0.0:
        return float("inf")
    return float(np.linalg.norm((lr - lg)[head]) / den)


def grain_ratio(rec: np.ndarray, gt: np.ndarray) -> float:
    brain = brain_mask(gt)
    wm = wm_mask(gt, brain)
    if not wm.any():
        return float("nan")

    def hp(v):
        v = v.astype(np.float32)
        return v - ndimage.gaussian_filter(v, HP_SIGMA)

    r = float(np.sqrt(np.mean(hp(rec)[wm] ** 2)))
    g = float(np.sqrt(np.mean(hp(gt)[wm] ** 2)))
    if g == 0.0:
        return float("inf")
    return r / g


def score_headonly(rec: np.ndarray, gt: np.ndarray) -> dict:
    rec = np.asarray(rec, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    head = head_mask(gt)
    gmax = float(gt.max())
    return {
        "head_psnr": head_psnr(rec, gt, head, gmax),
        "head_ssim": head_ssim(rec, gt, head, gmax),
        "head_hfen": head_hfen(rec, gt, head, gmax),
        "grain_ratio": grain_ratio(rec, gt),
        "head_frac": float(head.mean()),
        "gmax": gmax,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/diag_artefact/test_headonly.py -v`
Expected: 9 passed. Do not touch the metric definitions to make a test pass; if a test is wrong, fix the test and say so in the commit message.

- [ ] **Step 5: Commit**

```bash
git add diag_artefact/headonly.py tests/diag_artefact/test_headonly.py
git commit -m "feat(diag): head-only PSNR/SSIM/HFEN and grain ratio for the AF12->AF10 gates"
```

---

### Task 2: Offline npz scorer

**Files:**
- Create: `scripts/score_volumes_headonly.py`
- Test: `tests/diag_artefact/test_score_volumes_headonly.py`

**Interfaces:**
- Consumes: `diag_artefact.headonly.score_headonly`.
- Produces: `score_npz(path: Path, run: str) -> dict` with keys `run, npz, plain, contrast`; each contrast dict = `score_headonly(rec, gt)` plus `"zf": score_headonly(zf, gt)`. CLI writes that dict as JSON.

- [ ] **Step 1: Write the failing test**

`tests/diag_artefact/test_score_volumes_headonly.py`:

```python
"""scripts/ has no __init__.py, so the module is loaded by path (same as test_check_nonvacuity.py)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "score_volumes_headonly.py"
spec = importlib.util.spec_from_file_location("score_volumes_headonly", SCRIPT)
svh = importlib.util.module_from_spec(spec)
sys.modules["score_volumes_headonly"] = svh
spec.loader.exec_module(svh)


def _npz(tmp_path, phantom):
    rng = np.random.default_rng(0)
    rec = (phantom + rng.normal(0, 0.01, phantom.shape)).astype(np.float32)
    zf = (phantom * 0.8).astype(np.float32)
    p = tmp_path / "toy_volumes.npz"
    np.savez_compressed(p, recT1=rec, gtT1=phantom, zfT1=zf, recFL=rec, gtFL=phantom, zfFL=zf)
    return p


def test_score_npz_has_both_contrasts_and_zf(tmp_path, phantom):
    out = svh.score_npz(_npz(tmp_path, phantom), run="toy")
    assert out["run"] == "toy"
    for key in ("plain", "contrast"):
        for m in ("head_psnr", "head_ssim", "head_hfen", "grain_ratio"):
            assert isinstance(out[key][m], float)
            assert isinstance(out[key]["zf"][m], float)
        assert out[key]["head_psnr"] > out[key]["zf"]["head_psnr"]


def test_cli_writes_json(tmp_path, phantom):
    p = _npz(tmp_path, phantom)
    out = tmp_path / "s.json"
    svh.main([str(p), "--run", "toy", "--out", str(out)])
    d = json.loads(out.read_text())
    assert d["run"] == "toy" and "plain" in d and "contrast" in d
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/diag_artefact/test_score_volumes_headonly.py -v`
Expected: FAIL at module load (`FileNotFoundError` / `AttributeError`).

- [ ] **Step 3: Implement `scripts/score_volumes_headonly.py`**

```python
#!/usr/bin/env python3
"""Head-only metrics (spec 2026-09-05 §2.1) for a saved `*_volumes.npz` (recT1/gtT1/zfT1/recFL/gtFL/zfFL,
magnitudes). Used to compute the AF10 reference values and to re-score historical runs. The npz has no
phase, so e_unobs cannot be computed here; that comes from arc/cycle.py or the checkpoint replay.

usage: score_volumes_headonly.py <volumes.npz> --run <name> --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diag_artefact.headonly import score_headonly  # noqa: E402

KEYS = (("plain", "recT1", "gtT1", "zfT1"), ("contrast", "recFL", "gtFL", "zfFL"))


def score_npz(path: Path, run: str) -> dict:
    z = np.load(path)
    out = {"run": run, "npz": str(Path(path).resolve())}
    for key, rk, gk, zk in KEYS:
        rec, gt, zf = (z[k].astype(np.float32) for k in (rk, gk, zk))
        d = score_headonly(rec, gt)
        d["zf"] = score_headonly(zf, gt)
        out[key] = d
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = score_npz(Path(a.npz), a.run)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d, indent=2))
    for key, *_ in KEYS:
        s = d[key]
        print(f"{a.run} {key:8s} head_psnr={s['head_psnr']:.3f} head_ssim={s['head_ssim']:.4f} "
              f"head_hfen={s['head_hfen']:.4f} N={s['grain_ratio']:.3f} (zf head_psnr {s['zf']['head_psnr']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/diag_artefact/test_score_volumes_headonly.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/score_volumes_headonly.py tests/diag_artefact/test_score_volumes_headonly.py
git commit -m "feat(scripts): offline head-only scorer for saved volume npz files"
```

---

### Task 3: Gate builder and gate evaluator

**Files:**
- Create: `scripts/make_gates.py`, `scripts/eval_gates.py`
- Test: `tests/diag_artefact/test_gates_scripts.py`

**Interfaces:**
- Consumes: head-only JSONs from Task 2 (`d[key][metric]`), the replay JSON of `scripts/autoresearch_kspace_score.py` (`d["modalities"][key]["e_unobs"]`), and `arc/cycle.py` score JSONs (`d[key][metric]`, `d["exp"]`, extended in Task 7 with `head_*` and `grain_ratio`).
- Produces:
  - `make_gates.build_gates(ref, rep, base, base_kspace, names=("T1","FLAIR"), floors=FLOORS) -> dict`
  - `eval_gates.evaluate(gates: dict, score: dict) -> dict` with `verdict["contrasts"][key][metric]["pass"]`, `n_pass`, `all_pass`.
  - gates.json schema (frozen):
    ```
    {"spec": ..., "reference_run", "repeat_run", "baseline_run", "tolerance_rule", "tolerance_floor": {...},
     "grain_ratio_max": 1.0,
     "contrasts": {"plain": {"name": "T1", "head_psnr": {"ref", "repeat", "tol", "direction": "max"},
                              "head_ssim": {...,"direction":"max"}, "head_hfen": {...,"direction":"min"},
                              "e_unobs_baseline": float, "baseline_head": {...}}, "contrast": {...}}}
    ```

- [ ] **Step 1: Write the failing tests**

`tests/diag_artefact/test_gates_scripts.py`:

```python
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


make_gates = _load("make_gates")
eval_gates = _load("eval_gates")


def _head(run, psnr, ssim, hfen, n=0.95):
    d = {"run": run}
    for key in ("plain", "contrast"):
        d[key] = {"head_psnr": psnr, "head_ssim": ssim, "head_hfen": hfen, "grain_ratio": n,
                  "zf": {"head_psnr": psnr - 8, "head_ssim": ssim - 0.1, "head_hfen": hfen + 0.2, "grain_ratio": 1.5}}
    return d


def test_tolerance_is_twice_delta_with_floor():
    ref = _head("ref", 34.605, 0.9000, 0.5000)
    rep = _head("rep", 34.604, 0.9000, 0.5000)          # delta psnr 0.001 -> 2x = 0.002 < floor 0.005
    base = _head("base", 33.9, 0.89, 0.55)
    base_k = {"modalities": {"plain": {"e_unobs": 0.64}, "contrast": {"e_unobs": 0.63}}}
    g = make_gates.build_gates(ref, rep, base, base_k)
    p = g["contrasts"]["plain"]
    assert p["name"] == "T1" and g["contrasts"]["contrast"]["name"] == "FLAIR"
    assert p["head_psnr"]["tol"] == pytest.approx(0.005)
    assert p["head_ssim"]["tol"] == pytest.approx(0.0005)
    assert p["head_hfen"]["tol"] == pytest.approx(0.001)
    assert p["head_psnr"]["direction"] == "max" and p["head_hfen"]["direction"] == "min"
    assert p["e_unobs_baseline"] == 0.64
    assert g["grain_ratio_max"] == 1.0


def test_tolerance_grows_with_a_large_repeat_delta():
    ref = _head("ref", 34.60, 0.90, 0.50)
    rep = _head("rep", 34.50, 0.90, 0.50)               # delta 0.1 -> tol 0.2
    g = make_gates.build_gates(ref, rep, _head("b", 33, 0.8, 0.6),
                               {"modalities": {"plain": {"e_unobs": 1}, "contrast": {"e_unobs": 1}}})
    assert g["contrasts"]["plain"]["head_psnr"]["tol"] == pytest.approx(0.2)


def _score(exp, psnr, ssim, hfen, e_unobs, n):
    d = {"exp": exp}
    for key in ("plain", "contrast"):
        d[key] = {"head_psnr": psnr, "head_ssim": ssim, "head_hfen": hfen, "e_unobs": e_unobs, "grain_ratio": n}
    return d


def test_evaluate_passes_and_fails_per_gate():
    g = make_gates.build_gates(_head("ref", 34.6, 0.90, 0.50), _head("rep", 34.6, 0.90, 0.50),
                               _head("b", 33, 0.8, 0.6),
                               {"modalities": {"plain": {"e_unobs": 0.64}, "contrast": {"e_unobs": 0.63}}})
    v = eval_gates.evaluate(g, _score("cand", 34.597, 0.9001, 0.5009, 0.60, 0.99))   # inside every tolerance
    c = v["contrasts"]["plain"]
    assert c["head_psnr"]["pass"] and c["head_ssim"]["pass"] and c["head_hfen"]["pass"]
    assert c["e_unobs"]["pass"] and c["grain_ratio"]["pass"]
    assert c["n_pass"] == 5 and c["all_pass"]
    v2 = eval_gates.evaluate(g, _score("bad", 34.59, 0.90, 0.5011, 0.65, 1.01))      # psnr, hfen, e_unobs, N fail
    c2 = v2["contrasts"]["contrast"]
    assert not c2["head_psnr"]["pass"] and c2["head_ssim"]["pass"] and not c2["head_hfen"]["pass"]
    assert not c2["e_unobs"]["pass"] and not c2["grain_ratio"]["pass"]
    assert c2["n_pass"] == 1 and not c2["all_pass"]
    assert v2["run"] == "bad"


def test_cli_roundtrip_and_append(tmp_path):
    ref, rep, base = (tmp_path / n for n in ("ref.json", "rep.json", "base.json"))
    ref.write_text(json.dumps(_head("ref", 34.6, 0.9, 0.5)))
    rep.write_text(json.dumps(_head("rep", 34.6, 0.9, 0.5)))
    base.write_text(json.dumps(_head("base", 33, 0.8, 0.6)))
    bk = tmp_path / "bk.json"
    bk.write_text(json.dumps({"modalities": {"plain": {"e_unobs": 0.64}, "contrast": {"e_unobs": 0.63}}}))
    gates = tmp_path / "gates.json"
    make_gates.main(["--ref", str(ref), "--repeat", str(rep), "--baseline", str(base),
                     "--baseline-kspace", str(bk), "--out", str(gates)])
    score = tmp_path / "cand_score.json"
    score.write_text(json.dumps(_score("cand", 34.6, 0.9, 0.5, 0.6, 0.9)))
    verdict = tmp_path / "v.json"
    dec = tmp_path / "decisions.jsonl"
    eval_gates.main(["--gates", str(gates), "--score", str(score), "--out", str(verdict),
                     "--append", str(dec), "--phase", "1", "--note", "toy"])
    v = json.loads(verdict.read_text())
    assert v["contrasts"]["plain"]["all_pass"]
    lines = dec.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["run"] == "cand" and rec["phase"] == "1" and rec["note"] == "toy" and "ts" in rec
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/diag_artefact/test_gates_scripts.py -v`
Expected: FAIL at module load.

- [ ] **Step 3: Implement `scripts/make_gates.py`**

```python
#!/usr/bin/env python3
"""Build the pre-registered gates.json (spec 2026-09-05 §2) from three head-only score JSONs
(scripts/score_volumes_headonly.py) and the baseline's e_unobs replay JSON
(scripts/autoresearch_kspace_score.py).

    tol = max(2 * |ref - repeat|, floor)     floors: 0.005 dB / 0.0005 SSIM / 0.001 HFEN

usage: make_gates.py --ref REF.json --repeat REP.json --baseline BASE.json --baseline-kspace BASE_kspace.json
                     --out gates.json [--names T1 FLAIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FLOORS = {"head_psnr": 0.005, "head_ssim": 0.0005, "head_hfen": 0.001}
DIRECTION = {"head_psnr": "max", "head_ssim": "max", "head_hfen": "min"}
SPEC = "docs/superpowers/specs/2026-09-05-af12-to-af10-quality-design.md §2"


def build_gates(ref: dict, rep: dict, base: dict, base_kspace: dict,
                names=("T1", "FLAIR"), floors: dict = FLOORS) -> dict:
    gates = {
        "spec": SPEC,
        "reference_run": ref["run"], "repeat_run": rep["run"], "baseline_run": base["run"],
        "tolerance_rule": "tol = max(2*|ref-repeat|, floor); pass iff value >= ref-tol (max) or value <= ref+tol (min)",
        "tolerance_floor": dict(floors),
        "e_unobs_rule": "pass iff e_unobs <= baseline e_unobs (checkpoint replay of the AF12 baseline)",
        "grain_ratio_max": 1.0,
        "visual_gate": "butterfly z=160 + three views, judged by the user, recorded in decisions.jsonl",
        "contrasts": {},
    }
    for key, name in zip(("plain", "contrast"), names):
        c = {"name": name}
        for m in ("head_psnr", "head_ssim", "head_hfen"):
            delta = abs(float(ref[key][m]) - float(rep[key][m]))
            c[m] = {"ref": float(ref[key][m]), "repeat": float(rep[key][m]),
                    "tol": max(2.0 * delta, float(floors[m])), "direction": DIRECTION[m]}
        c["e_unobs_baseline"] = float(base_kspace["modalities"][key]["e_unobs"])
        c["baseline_head"] = {m: float(base[key][m]) for m in ("head_psnr", "head_ssim", "head_hfen", "grain_ratio")}
        gates["contrasts"][key] = c
    return gates


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--repeat", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--baseline-kspace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", nargs=2, default=("T1", "FLAIR"))
    a = ap.parse_args(argv)
    load = lambda p: json.loads(Path(p).read_text())
    g = build_gates(load(a.ref), load(a.repeat), load(a.baseline), load(a.baseline_kspace), tuple(a.names))
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f"{out} exists; gates are pre-registered and must not be rewritten in place")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(g, indent=2))
    print(json.dumps(g, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Implement `scripts/eval_gates.py`**

```python
#!/usr/bin/env python3
"""Evaluate one arc/cycle.py score JSON against gates.json (spec 2026-09-05 §2), per contrast.

usage: eval_gates.py --gates gates.json --score <exp>_score.json [--out verdict.json]
                     [--append decisions.jsonl --phase 1 --note "..."]
The visual gate is not computed here; record it in --note.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

METRICS = ("head_psnr", "head_ssim", "head_hfen")


def evaluate(gates: dict, score: dict) -> dict:
    v = {"run": score["exp"], "gates_reference_run": gates["reference_run"], "contrasts": {}}
    for key, g in gates["contrasts"].items():
        s = score[key]
        r = {"name": g["name"]}
        for m in METRICS:
            val, ref, tol = float(s[m]), g[m]["ref"], g[m]["tol"]
            if g[m]["direction"] == "min":
                ok, margin = val <= ref + tol, (ref + tol) - val
            else:
                ok, margin = val >= ref - tol, val - (ref - tol)
            r[m] = {"value": val, "ref": ref, "tol": tol, "pass": bool(ok), "margin": float(margin)}
        e, eb = float(s["e_unobs"]), g["e_unobs_baseline"]
        r["e_unobs"] = {"value": e, "baseline": eb, "pass": bool(e <= eb), "margin": float(eb - e)}
        n, nmax = float(s["grain_ratio"]), gates["grain_ratio_max"]
        r["grain_ratio"] = {"value": n, "max": nmax, "pass": bool(n <= nmax), "margin": float(nmax - n)}
        r["n_pass"] = sum(int(r[k]["pass"]) for k in (*METRICS, "e_unobs", "grain_ratio"))
        r["all_pass"] = r["n_pass"] == 5
        v["contrasts"][key] = r
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", required=True)
    ap.add_argument("--score", required=True)
    ap.add_argument("--out")
    ap.add_argument("--append")
    ap.add_argument("--phase", default="")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    gates = json.loads(Path(a.gates).read_text())
    score = json.loads(Path(a.score).read_text())
    v = evaluate(gates, score)
    for key, r in v["contrasts"].items():
        cells = " ".join(f"{m}={'PASS' if r[m]['pass'] else 'FAIL'}({r[m]['value']:.4f},margin {r[m]['margin']:+.4f})"
                         for m in (*METRICS, "e_unobs", "grain_ratio"))
        print(f"{v['run']} {r['name']:6s} {r['n_pass']}/5 {cells}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(v, indent=2))
    if a.append:
        rec = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "run": v["run"],
               "phase": a.phase, "note": a.note,
               "verdict": {k: {"n_pass": r["n_pass"], "all_pass": r["all_pass"]} for k, r in v["contrasts"].items()},
               "score": str(Path(a.score))}
        with open(a.append, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/diag_artefact/test_gates_scripts.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/make_gates.py scripts/eval_gates.py tests/diag_artefact/test_gates_scripts.py
git commit -m "feat(scripts): pre-registered gate builder and per-contrast gate evaluator"
```

---

### Task 4: Phase 0 — reference table, e_unobs baseline replay, gates.json, decisions ledger

**Files:**
- Create: `docs/superpowers/records/af12_to_af10/{reference_table.md, gates.json, decisions.jsonl}` and `docs/superpowers/records/af12_to_af10/scores/{tvs4k_e040_headonly.json, af10_joint_edgetv_e040_headonly.json, af10_joint_edgetv_e040_rerun20260904_headonly.json, tvs4k_e040_kspace.json}`

**Interfaces:**
- Consumes: Tasks 2–3 CLIs; `scripts/autoresearch_kspace_score.py --experiment tvs4k_e040 --gpu N --output <path>` (existing; reads `results_arc/tvs4k_e040/tvs4k_e040/{config.yaml,checkpoints/latest.pth}`, refuses to overwrite).
- Produces: the committed gate file that every later run is judged against.

- [ ] **Step 1: Score the three reference volumes (CPU, ~3 min each)**

```bash
R=docs/superpowers/records/af12_to_af10/scores; mkdir -p $R
P="env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python"
$P scripts/score_volumes_headonly.py results_arc/tvs4k_e040_volumes.npz --run tvs4k_e040 --out $R/tvs4k_e040_headonly.json
$P scripts/score_volumes_headonly.py results_arc/af10_joint_edgetv_e040_volumes.npz --run af10_joint_edgetv_e040 --out $R/af10_joint_edgetv_e040_headonly.json
$P scripts/score_volumes_headonly.py results_arc/af10_joint_edgetv_e040_rerun20260904_volumes.npz --run af10_joint_edgetv_e040_rerun20260904 --out $R/af10_joint_edgetv_e040_rerun20260904_headonly.json
```
Expected: head_psnr ≈ 33.9 / 33.6 (AF12) and ≈ 34.6 / 34.4 (AF10) for T1 / FLAIR; the two AF10 runs within ~0.002 dB.

**Sanity check that must pass before continuing:** `grain_ratio` of `tvs4k_e040` T1 must be within 0.02 of 0.944 (memory §0's "grain 0.944 of GT", same definition). If it is not, STOP: the N definition differs from §14's; recover the producing script (`rc_figure.py`, session c6621154's scratchpad) and reconcile before pinning N in the gates. Record the outcome either way in `reference_table.md`.

- [ ] **Step 2: Replay the AF12 baseline checkpoint for e_unobs (GPU, ~4 min)**

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID /home/congcongliu/anaconda3/envs/GSTMI/bin/python \
  scripts/autoresearch_kspace_score.py --experiment tvs4k_e040 --gpu 2 \
  --output docs/superpowers/records/af12_to_af10/scores/tvs4k_e040_kspace.json
```
Expected: `KSPACE_SCORE_DONE`, `modalities.plain.e_unobs` ≈ 0.64–0.66 and `checkpoint.iteration == 4000`. If the iteration is not 4000, stop and find the right checkpoint; the gate must be against the full-budget baseline.

- [ ] **Step 3: Build gates.json**

```bash
S=docs/superpowers/records/af12_to_af10/scores
env PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/make_gates.py \
  --ref $S/af10_joint_edgetv_e040_rerun20260904_headonly.json --repeat $S/af10_joint_edgetv_e040_headonly.json \
  --baseline $S/tvs4k_e040_headonly.json --baseline-kspace $S/tvs4k_e040_kspace.json \
  --out docs/superpowers/records/af12_to_af10/gates.json
```

- [ ] **Step 4: Write `reference_table.md` (generated, then hand-annotated)**

```bash
env PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python - <<'EOF'
import json
from pathlib import Path
S = Path("docs/superpowers/records/af12_to_af10/scores")
runs = ["tvs4k_e040", "af10_joint_edgetv_e040", "af10_joint_edgetv_e040_rerun20260904"]
H = {r: json.loads((S / f"{r}_headonly.json").read_text()) for r in runs}
whole = {}
for r in runs:
    for p in (Path("results_arc") / f"{r}_score.json", Path("docs/superpowers/records") / f"{r}_score.json"):
        if p.exists():
            whole[r] = json.loads(p.read_text()); break
g = json.loads(Path("docs/superpowers/records/af12_to_af10/gates.json").read_text())
lines = ["# Reference table — AF12 baseline vs AF10 (Phase 0, spec §3)", "",
         "| run | contrast | head PSNR | head SSIM | head HFEN | N | whole PSNR | whole SSIM | whole HFEN | e_unobs |", "|---|---|---|---|---|---|---|---|---|---|"]
for r in runs:
    for key, name in (("plain", "T1"), ("contrast", "FLAIR")):
        h = H[r][key]; w = whole.get(r, {}).get(key, {})
        f = lambda v, n=3: (f"{v:.{n}f}" if isinstance(v, (int, float)) else "—")
        lines.append(f"| {r} | {name} | {f(h['head_psnr'])} | {f(h['head_ssim'],4)} | {f(h['head_hfen'],4)} | {f(h['grain_ratio'])} | "
                     f"{f(w.get('psnr'))} | {f(w.get('ssim'),4)} | {f(w.get('hfen'),4)} | {f(w.get('e_unobs'),4)} |")
lines += ["", "## Gates (from gates.json)", ""]
for key in ("plain", "contrast"):
    c = g["contrasts"][key]
    lines.append(f"- **{c['name']}**: head_psnr ≥ {c['head_psnr']['ref']:.3f} − {c['head_psnr']['tol']:.4f}; "
                 f"head_ssim ≥ {c['head_ssim']['ref']:.4f} − {c['head_ssim']['tol']:.4f}; "
                 f"head_hfen ≤ {c['head_hfen']['ref']:.4f} + {c['head_hfen']['tol']:.4f}; "
                 f"e_unobs ≤ {c['e_unobs_baseline']:.4f}; N ≤ {g['grain_ratio_max']:.2f}")
Path("docs/superpowers/records/af12_to_af10/reference_table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
```
Then append by hand: the N-sanity outcome from Step 1, and one line stating the gap to close per contrast (AF10 ref − AF12 baseline, head PSNR).

- [ ] **Step 5: Start the decisions ledger and commit Phase 0**

```bash
printf '%s\n' '{"ts":"'"$(date -u +%FT%TZ)"'","run":"-","phase":"0","note":"gates.json pre-registered from af10 rerun (ref) / af10 08-11 (repeat) / tvs4k_e040 (baseline, e_unobs by checkpoint replay). No Phase 1/2 run existed before this commit.","verdict":{}}' \
  > docs/superpowers/records/af12_to_af10/decisions.jsonl
git add docs/superpowers/records/af12_to_af10
git commit -m "records: Phase 0 reference table, pre-registered gates and decisions ledger (AF12->AF10)"
```

---

### Task 5: Known-zero region estimator

**Files:**
- Create: `paired_gs/data/known_zero.py`
- Test: `tests/test_known_zero.py`

**Interfaces:**
- Produces:
  - `KnownZeroInfo` dataclass: `dc: tuple[int,int,int]`, `ro_axis: int`, `rho_c: float|None`, `kro_c: float|None`, `n0_pe: float`, `n0_ro: float`, `frac_z: float`, `frac_sampled_in_z: float`, `pe_profile: list[tuple[float,float,int]]`, `ro_profile: list[...]`.
  - `estimate_known_zero(kspace: Tensor(C,D,H,W) complex, mask: Tensor(D,H,W), ro_axis: int, **rules) -> (Z: BoolTensor(D,H,W), KnownZeroInfo)`
  - `pe_radius_map(shape: tuple, dc: tuple, ro_axis: int, device) -> Tensor(D,H,W) float` (ρ broadcast along readout).
  - Rule constants: `RING_WIDTH=0.05, MARGIN=1.5, CLIFF_RATIO=5.0, CLIFF_SPAN=0.15, MIN_RINGS=2, MIN_RING_POINTS=30`.
- Task 6 adds `estimate_noise_cov`, `whitening_matrix`, `build_fit_tensors` to the same file.

- [ ] **Step 1: Write the failing tests (synthetic k-space, CPU)**

`tests/test_known_zero.py`:

```python
"""Known-zero region Z (spec 2026-09-05 §4.1): estimated from ACQUIRED samples only, by the cliff rule.

Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_known_zero.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
from data.known_zero import RING_WIDTH, KnownZeroInfo, estimate_known_zero, pe_radius_map  # noqa: E402

C, P0, P1, R = 4, 48, 40, 24


def _synth(rho_c=None, shutter=None, noise=2e-3, seed=0, ro_axis=2):
    """Multicoil k-space of a textured object (broad spectrum), optionally low-pass filtered in the PE plane
    (vendor filter, keeps noise) or hard-shuttered (exact zeros), vd-random mask broadcast along readout.
    Returns (kspace (C,P0,P1,R) undersampled, mask (P0,P1,R)) with ro_axis last, then moved to `ro_axis`."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.ogrid[:P0, :P1, :R]
    blob = np.exp(-(((zz - P0 / 2) / (P0 / 4)) ** 2 + ((yy - P1 / 2) / (P1 / 4)) ** 2 + ((xx - R / 2) / (R / 4)) ** 2))
    texture = ndimage.gaussian_filter(rng.normal(size=(P0, P1, R)), 0.7)
    obj = blob * (1.0 + 0.8 * texture / texture.std())          # texture -> nearly flat high-frequency spectrum
    ks = []
    for _ in range(C):
        cm = ndimage.gaussian_filter(rng.normal(size=(P0, P1, R)), 8.0)
        cm = cm / np.abs(cm).max() + 0.5
        k = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(cm * obj), norm="ortho"))
        ks.append(k)
    k = np.stack(ks).astype(np.complex64)
    ii, jj = np.meshgrid(np.arange(P0), np.arange(P1), indexing="ij")
    rho = np.sqrt(((ii - P0 // 2) / (P0 / 2)) ** 2 + ((jj - P1 // 2) / (P1 / 2)) ** 2)
    if rho_c is not None:
        k = k * (rho < rho_c)[None, :, :, None]
    if noise:
        k = k + noise * (rng.normal(size=k.shape) + 1j * rng.normal(size=k.shape)).astype(np.complex64) / np.sqrt(2)
    if shutter is not None:
        k = k * (rho < shutter)[None, :, :, None]                # exact zeros AFTER noise
    m2 = rng.random((P0, P1)) < 0.35
    m2[P0 // 2 - 4:P0 // 2 + 4, P1 // 2 - 4:P1 // 2 + 4] = True
    mask = np.repeat(m2[:, :, None], R, axis=2).astype(np.float32)
    k = k * mask[None]
    kt, mt = torch.from_numpy(k), torch.from_numpy(mask)
    return torch.movedim(kt, 3, ro_axis + 1).contiguous(), torch.movedim(mt, 2, ro_axis).contiguous()


def test_vendor_lowpass_is_found_within_one_ring():
    k, m = _synth(rho_c=0.8)
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    assert isinstance(info, KnownZeroInfo)
    assert info.rho_c is not None and abs(info.rho_c - 0.8) <= RING_WIDTH + 1e-9
    assert info.kro_c is None                                    # nothing along readout
    rho = pe_radius_map(tuple(m.shape), info.dc, 2, m.device)
    assert Z.dtype == torch.bool and Z.shape == m.shape
    assert bool(Z[rho >= 0.9].all()) and not bool(Z[rho <= 0.7].any())
    assert 0.0 < info.frac_z < 0.5


def test_unfiltered_data_has_no_known_zero():
    k, m = _synth(rho_c=None)
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    assert info.rho_c is None and info.kro_c is None
    assert not bool(Z.any()) and info.frac_z == 0.0


def test_exact_shutter_is_found():
    k, m = _synth(rho_c=None, shutter=1.0)
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    assert info.rho_c is not None and abs(info.rho_c - 1.0) <= RING_WIDTH + 1e-9
    assert info.n0_pe == 0.0


def test_uses_only_sampled_points():
    """Filling the UNSAMPLED points with garbage must not change the estimate."""
    k, m = _synth(rho_c=0.8, seed=3)
    _, info_a = estimate_known_zero(k, m, ro_axis=2)
    junk = torch.randn_like(k) * (1 - m).unsqueeze(0) * 10.0
    _, info_b = estimate_known_zero(k + junk, m, ro_axis=2)
    assert info_a.rho_c == info_b.rho_c and info_a.pe_profile == info_b.pe_profile


def test_readout_axis_position_does_not_matter():
    k0, m0 = _synth(rho_c=0.8, seed=5, ro_axis=2)
    k1, m1 = _synth(rho_c=0.8, seed=5, ro_axis=0)
    _, i0 = estimate_known_zero(k0, m0, ro_axis=2)
    _, i1 = estimate_known_zero(k1, m1, ro_axis=0)
    assert i0.rho_c == i1.rho_c and i0.dc[2] == i1.dc[0]
    assert np.allclose([p[1] for p in i0.pe_profile], [p[1] for p in i1.pe_profile], rtol=1e-5)   # reduction order may differ


def test_info_is_json_serializable():
    import dataclasses, json
    k, m = _synth(rho_c=0.8)
    _, info = estimate_known_zero(k, m, ro_axis=2)
    json.dumps(dataclasses.asdict(info))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_known_zero.py -v`
Expected: `ModuleNotFoundError: No module named 'data.known_zero'`.

- [ ] **Step 3: Implement the estimator (`paired_gs/data/known_zero.py`)**

```python
"""Known-zero k-space region Z and coil-noise prewhitening W — the Phase 1 physics levers
(docs/superpowers/specs/2026-09-05-af12-to-af10-quality-design.md §4).

Everything here is estimated from the ACQUIRED samples of one scan (kspace already multiplied by the
mask). Nothing reads the full k-space or the ground truth. Pure functions; the trainer only wires the
outputs in. All rule constants are fixed a priori (spec §4.1) and are NOT tuning knobs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

RING_WIDTH = 0.05        # ring / band width in normalised radius
MARGIN = 1.5             # a ring is "at the floor" if E_r <= MARGIN * n0
CLIFF_RATIO = 5.0        # the energy just inside Z must exceed the Z energy by this factor
CLIFF_SPAN = 0.15        # ... within this radial distance
MIN_RINGS = 2            # Z must span at least this many rings
MIN_RING_POINTS = 30     # PE rings with fewer sampled points merge into the inner neighbour
MIN_NOISE_SAMPLES = 2000 # per-coil samples needed for a noise covariance


@dataclass
class KnownZeroInfo:
    dc: tuple
    ro_axis: int
    rho_c: Optional[float]
    kro_c: Optional[float]
    n0_pe: float
    n0_ro: float
    frac_z: float
    frac_sampled_in_z: float
    pe_profile: list
    ro_profile: list


def _profile(e_mean: np.ndarray, coord: np.ndarray, sampled: np.ndarray, width: float, min_points: int) -> list:
    """[(coord_lo, median energy, n_points)] over sampled entries, ring by ring from the centre outward.
    A ring with fewer than `min_points` sampled entries is merged into the previous (inner) ring."""
    if not sampled.any():
        return []
    edges = np.arange(0.0, float(coord[sampled].max()) + width, width)
    out: list = []
    for lo in edges:
        sel = sampled & (coord >= lo) & (coord < lo + width)
        vals = e_mean[sel]
        if vals.size == 0:
            continue
        if vals.size < min_points and out:
            out[-1][1] = np.concatenate([out[-1][1], vals])
        else:
            out.append([float(lo), vals])
    return [(lo, float(np.median(v)), int(v.size)) for lo, v in out]


def _cutoff(profile: list, *, margin: float, cliff_ratio: float, cliff_span: float, min_rings: int):
    """The cliff rule (spec §4.1 step 4). Returns (cutoff or None, n0)."""
    if len(profile) < min_rings + 1:
        return None, (float(min(p[1] for p in profile)) if profile else 0.0)
    lo = np.array([p[0] for p in profile]); E = np.array([p[1] for p in profile])
    n0 = float(E.min())
    thr = margin * n0
    k = len(E)
    while k > 0 and E[k - 1] <= thr:          # maximal suffix at the floor, walking inward from the outermost ring
        k -= 1
    if len(E) - k < min_rings or k == 0:
        return None, n0
    rho_in = float(lo[k])
    before = E[(lo >= rho_in - cliff_span) & (lo < rho_in)]
    if before.size == 0:
        return None, n0
    e_z = float(E[k:].max())
    cliff = (float(before.max()) >= cliff_ratio * e_z) if e_z > 0 else (float(before.max()) > 0)
    if not cliff:
        return None, n0
    return rho_in, n0


def pe_radius_map(shape: tuple, dc: tuple, ro_axis: int, device) -> torch.Tensor:
    """Normalised PE-plane radius about `dc`, broadcast along the readout axis. (D,H,W) float32."""
    pe = [a for a in range(3) if a != ro_axis]
    n0, n1 = shape[pe[0]], shape[pe[1]]
    i = torch.arange(n0, device=device, dtype=torch.float32)
    j = torch.arange(n1, device=device, dtype=torch.float32)
    rho2 = torch.sqrt(((i - dc[pe[0]]) / (n0 / 2)).view(-1, 1) ** 2 + ((j - dc[pe[1]]) / (n1 / 2)).view(1, -1) ** 2)
    rho = rho2.unsqueeze(-1).expand(n0, n1, shape[ro_axis])
    return torch.movedim(rho, -1, ro_axis).contiguous()


def estimate_known_zero(kspace: torch.Tensor, mask: torch.Tensor, ro_axis: int, *,
                        ring_width: float = RING_WIDTH, margin: float = MARGIN, cliff_ratio: float = CLIFF_RATIO,
                        cliff_span: float = CLIFF_SPAN, min_rings: int = MIN_RINGS,
                        min_points: int = MIN_RING_POINTS):
    """Z from acquired samples only. kspace (C,D,H,W) complex, already masked; mask (D,H,W) in {0,1}."""
    assert kspace.ndim == 4 and mask.ndim == 3 and tuple(kspace.shape[1:]) == tuple(mask.shape)
    dev = kspace.device
    Cn = kspace.shape[0]
    m = mask > 0.5
    e = (kspace.real.square() + kspace.imag.square()).sum(0) * m            # sampled energy, (D,H,W)
    e_r = torch.movedim(e, ro_axis, -1); m_r = torch.movedim(m, ro_axis, -1)  # (P0,P1,R)
    P0, P1, Rn = e_r.shape
    # --- PE plane: DC = ACS energy peak of the sampled data
    e_pe = e_r.sum(-1)
    flat = int(torch.argmax(e_pe)); dc0, dc1 = flat // P1, flat % P1
    n_ro = m_r.sum(-1)                                                     # sampled readout entries per PE point
    sampled_pe = (n_ro > 0).cpu().numpy()
    e_mean_pe = (e_pe / (Cn * n_ro.clamp_min(1))).cpu().numpy()           # per (coil, sample)
    ii, jj = np.meshgrid(np.arange(P0), np.arange(P1), indexing="ij")
    rho = np.sqrt(((ii - dc0) / (P0 / 2)) ** 2 + ((jj - dc1) / (P1 / 2)) ** 2)
    pe_profile = _profile(e_mean_pe, rho, sampled_pe, ring_width, min_points)
    rho_c, n0_pe = _cutoff(pe_profile, margin=margin, cliff_ratio=cliff_ratio, cliff_span=cliff_span, min_rings=min_rings)
    # --- readout axis: each index already averages thousands of PE samples, so no merging (min_points=1)
    e_ro = e_r.sum((0, 1))
    n_pe = m_r.sum((0, 1))
    dc2 = int(torch.argmax(e_ro))
    e_mean_ro = (e_ro / (Cn * n_pe.clamp_min(1))).cpu().numpy()
    kro = np.abs(np.arange(Rn) - dc2) / (Rn / 2)
    ro_profile = _profile(e_mean_ro, kro, (n_pe > 0).cpu().numpy(), ring_width, 1)
    kro_c, n0_ro = _cutoff(ro_profile, margin=margin, cliff_ratio=cliff_ratio, cliff_span=cliff_span, min_rings=min_rings)
    # --- assemble Z
    Z_r = torch.zeros((P0, P1, Rn), dtype=torch.bool, device=dev)
    if rho_c is not None:
        Z_r |= torch.from_numpy(rho >= rho_c).to(dev)[:, :, None]
    if kro_c is not None:
        Z_r |= torch.from_numpy(kro >= kro_c).to(dev)[None, None, :]
    Z = torch.movedim(Z_r, -1, ro_axis).contiguous()
    dc = [0, 0, 0]
    pe = [a for a in range(3) if a != ro_axis]
    dc[pe[0]], dc[pe[1]], dc[ro_axis] = int(dc0), int(dc1), int(dc2)
    info = KnownZeroInfo(dc=tuple(dc), ro_axis=int(ro_axis), rho_c=rho_c, kro_c=kro_c,
                         n0_pe=float(n0_pe), n0_ro=float(n0_ro), frac_z=float(Z.float().mean()),
                         frac_sampled_in_z=float((Z & m).sum().item() / max(m.sum().item(), 1)),
                         pe_profile=pe_profile, ro_profile=ro_profile)
    return Z, info
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_known_zero.py -v`
Expected: 6 passed. If `test_unfiltered_data_has_no_known_zero` fails because the synthetic spectrum still decays ≥5× within 0.15 near the corners, raise the texture amplitude (0.8 → 1.2) in `_synth` — the fixture must have a flat high-frequency spectrum; do NOT change the rule constants.

- [ ] **Step 5: Commit**

```bash
git add paired_gs/data/known_zero.py tests/test_known_zero.py
git commit -m "feat(data): known-zero k-space region estimated from acquired samples by the cliff rule"
```

---

### Task 6: Noise covariance, whitening, fit-side tensors

**Files:**
- Modify: `paired_gs/data/known_zero.py` (append)
- Test: `tests/test_known_zero.py` (append)

**Interfaces:**
- Produces:
  - `estimate_noise_cov(kspace, mask, Z, info: KnownZeroInfo, *, min_samples=MIN_NOISE_SAMPLES) -> (Sigma: Tensor(C,C) complex | None, winfo: dict)` with `winfo["sigma_source"] in {"known_zero","outer_ring","identity"}`, `winfo["n_noise_samples"]`, `winfo["coil_corr_max_offdiag"]`, `winfo["eig_ratio"]`.
  - `whitening_matrix(Sigma: Tensor(C,C), eig_floor: float = 1e-6) -> Tensor(C,C)` = Σ^{-1/2} · sqrt(mean diag Σ) (so Σ = σ²I ⇒ W = I).
  - `build_fit_tensors(kspace, mask, csm, Z: BoolTensor, W: Tensor | None) -> (mask_fit, kspace_dc, kspace_fit, csm_fit)`.

- [ ] **Step 1: Write the failing tests (append to `tests/test_known_zero.py`)**

```python
from data.known_zero import MIN_NOISE_SAMPLES, build_fit_tensors, estimate_noise_cov, whitening_matrix  # noqa: E402


def _corr_noise(sigma_true, n, seed=0):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(sigma_true)
    w = (rng.normal(size=(sigma_true.shape[0], n)) + 1j * rng.normal(size=(sigma_true.shape[0], n))) / np.sqrt(2)
    return L @ w


def test_whitening_of_diagonal_equal_variance_is_identity():
    S = torch.eye(C, dtype=torch.complex64) * 0.3
    W = whitening_matrix(S)
    assert torch.allclose(W, torch.eye(C, dtype=torch.complex64), atol=1e-6)


def test_whitening_decorrelates_and_keeps_mean_variance():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(C, C)) + 1j * rng.normal(size=(C, C))
    S_true = A @ A.conj().T + np.eye(C)
    W = whitening_matrix(torch.from_numpy(S_true.astype(np.complex64))).numpy().astype(np.complex128)
    out = W @ S_true @ W.conj().T
    c = np.real(np.trace(S_true)) / C
    assert np.allclose(out, c * np.eye(C), atol=1e-4 * c)


def test_noise_cov_from_known_zero_samples():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(C, C)) + 1j * rng.normal(size=(C, C))
    S_true = (A @ A.conj().T + np.eye(C)) * 1e-4
    n = P0 * P1 * R
    noise = _corr_noise(S_true, n).reshape(C, P0, P1, R).astype(np.complex64)
    mask = (rng.random((P0, P1, R)) < 0.6).astype(np.float32)
    k = torch.from_numpy(noise) * torch.from_numpy(mask)
    Z = torch.ones((P0, P1, R), dtype=torch.bool)
    _, info = estimate_known_zero(k, torch.from_numpy(mask), ro_axis=2)
    S, winfo = estimate_noise_cov(k, torch.from_numpy(mask), Z, info)
    assert winfo["sigma_source"] == "known_zero" and winfo["n_noise_samples"] > MIN_NOISE_SAMPLES
    rel = np.linalg.norm(S.numpy().astype(np.complex128) - S_true) / np.linalg.norm(S_true)
    assert rel < 0.08


def test_noise_cov_falls_back_to_identity_when_z_is_exact_zero_and_ring_too_small():
    k, m = _synth(rho_c=None, shutter=1.0, seed=1)
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    S, winfo = estimate_noise_cov(k, m, Z, info, min_samples=10 ** 9)   # force both routes to fail
    assert S is None and winfo["sigma_source"] == "identity"


def test_noise_cov_outer_ring_fallback_when_z_is_exact_zero():
    k, m = _synth(rho_c=None, shutter=1.0, seed=1)
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    S, winfo = estimate_noise_cov(k, m, Z, info, min_samples=50)
    assert S is not None and winfo["sigma_source"] == "outer_ring"


def test_build_fit_tensors_defaults_and_hard_dc_zero_on_Z():
    k, m = _synth(rho_c=0.8, seed=2)
    csm = torch.randn(C, *m.shape, dtype=torch.complex64)
    Z, _ = estimate_known_zero(k, m, ro_axis=2)
    mask_fit, k_dc, k_fit, csm_fit = build_fit_tensors(k, m, csm, Z, None)
    assert k_fit is k_dc and csm_fit is csm                       # no W -> no whitened copies
    assert torch.equal(mask_fit, torch.maximum(m, Z.float()))
    assert float(k_dc[:, Z].abs().max()) == 0.0
    pred = torch.randn_like(k)                                     # any model k-space
    hard_dc = pred * (1 - mask_fit).unsqueeze(0) + k_dc
    assert float(hard_dc[:, Z].abs().max()) == 0.0                 # spec §4.1: exactly zero on Z
    assert torch.equal(hard_dc[:, (m > 0.5) & ~Z], k[:, (m > 0.5) & ~Z])   # measured samples untouched
    # denominator invariant: the caller keeps |Omega|; mask_fit grew
    assert float(mask_fit.sum()) > float(m.sum())


def test_build_fit_tensors_with_identity_W_is_exact():
    k, m = _synth(rho_c=0.8, seed=2)
    csm = torch.randn(C, *m.shape, dtype=torch.complex64)
    Z = torch.zeros_like(m, dtype=torch.bool)
    W = torch.eye(C, dtype=torch.complex64)
    mask_fit, k_dc, k_fit, csm_fit = build_fit_tensors(k, m, csm, Z, W)
    assert torch.equal(mask_fit, m) and torch.equal(k_dc, k)
    assert torch.allclose(k_fit, k) and torch.allclose(csm_fit, csm)
```

- [ ] **Step 2: Run the tests to verify the new ones fail**

Run: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_known_zero.py -v`
Expected: `ImportError: cannot import name 'build_fit_tensors'`.

- [ ] **Step 3: Append to `paired_gs/data/known_zero.py`**

```python
def estimate_noise_cov(kspace: torch.Tensor, mask: torch.Tensor, Z: torch.Tensor, info: KnownZeroInfo, *,
                       min_samples: int = MIN_NOISE_SAMPLES):
    """Coil noise covariance Σ (C×C) from the ACQUIRED samples inside Z (pure noise by construction).
    Fallback (spec §4.2): the 0.1-wide sampled ring just inside the PE cliff, flagged `outer_ring`;
    then `identity` (caller uses W = I)."""
    m = mask > 0.5
    Cn = kspace.shape[0]

    def _take(sel):
        v = kspace[:, sel]
        return v if (v.shape[1] >= min_samples and float(v.abs().max()) > 0.0) else None

    vals, source = _take(m & Z), "known_zero"
    if vals is None:
        rho = pe_radius_map(tuple(mask.shape), info.dc, info.ro_axis, kspace.device)
        edge = info.rho_c if info.rho_c is not None else (info.pe_profile[-1][0] + RING_WIDTH if info.pe_profile else 1.0)
        vals, source = _take(m & (rho >= edge - 0.1) & (rho < edge)), "outer_ring"
    if vals is None:
        return None, {"sigma_source": "identity", "n_noise_samples": 0}
    vals = vals.to(torch.complex128)
    Sigma = (vals @ vals.conj().T) / vals.shape[1]
    d = torch.sqrt(torch.diagonal(Sigma).real)
    corr = (Sigma / (d.view(-1, 1) * d.view(1, -1))).abs()
    corr.fill_diagonal_(0.0)
    ev = torch.linalg.eigvalsh(0.5 * (Sigma + Sigma.conj().T)).real
    winfo = {"sigma_source": source, "n_noise_samples": int(vals.shape[1]),
             "coil_corr_max_offdiag": float(corr.max()), "coil_corr_mean_offdiag": float(corr.sum() / (Cn * (Cn - 1))),
             "eig_ratio": float(ev.max() / ev.clamp_min(1e-30).min())}
    return Sigma.to(kspace.dtype), winfo


def whitening_matrix(Sigma: torch.Tensor, eig_floor: float = 1e-6) -> torch.Tensor:
    """W = Σ^{-1/2} · sqrt(mean diag Σ): whitens the coil dimension while keeping the mean variance,
    so Σ = σ²I gives W = I exactly and lambda_geo keeps its meaning."""
    S = 0.5 * (Sigma + Sigma.conj().T).to(torch.complex128)
    evals, evecs = torch.linalg.eigh(S)
    evals = evals.real.clamp_min(eig_floor * float(evals.real.max()))
    W = evecs @ torch.diag((evals ** -0.5).to(evecs.dtype)) @ evecs.conj().T
    scale = torch.sqrt(torch.diagonal(S).real.mean())
    return (W * scale).to(Sigma.dtype)


def build_fit_tensors(kspace: torch.Tensor, mask: torch.Tensor, csm: torch.Tensor, Z: torch.Tensor, W):
    """Fit-side tensors (spec §4.1/§4.2):
        mask_fit  = mask ∪ Z                 (loss and hard-DC)
        kspace_dc = kspace with Z zeroed     (hard-DC, original coil basis)
        kspace_fit = W · kspace_dc           (loss only; == kspace_dc when W is None)
        csm_fit   = W · csm                  (loss only; == csm when W is None)
    The DC-loss denominator is NOT produced here: the caller keeps |Omega| = mask.sum()."""
    mask_fit = torch.maximum(mask, Z.to(mask.dtype))
    kspace_dc = kspace * (~Z).unsqueeze(0).to(kspace.dtype)
    if W is None:
        return mask_fit, kspace_dc, kspace_dc, csm
    Wk = W.to(kspace.dtype)
    kspace_fit = torch.einsum("ij,jdhw->idhw", Wk, kspace_dc)
    csm_fit = torch.einsum("ij,jdhw->idhw", W.to(csm.dtype), csm)
    return mask_fit, kspace_dc, kspace_fit, csm_fit
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_known_zero.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add paired_gs/data/known_zero.py tests/test_known_zero.py
git commit -m "feat(data): coil-noise covariance from the known-zero samples, whitening, fit-side tensors"
```

---

### Task 7: Wire Z / W into `JointTrainer` and `arc/cycle.py`; regression checks

**Files:**
- Modify: `paired_gs/trainers/joint_trainer.py` — imports (line ~47), `_setup_data` (after line 169), `_kspace_loss` (lines 428–434), train-loop DC lines (785–786), `export_results` (lines 956–957)
- Modify: `arc/cycle.py` — argparse (after line 55), `build_cfg` (before the `output` block), imports (after line 75), `score_contrast` (185–222), `main` log/JSON (247–260), `_vol` (274–283)

**Interfaces:**
- Consumes: Task 5/6 functions; Task 1 `score_headonly`.
- Produces on the trainer: `mask_fit_p/c`, `kspace_dc_p/c`, `kspace_fit_p/c`, `csm_fit_p/c`, `_dc_norm_p/c`, `physics_info: dict` (keys `"p"`, `"c"`), `_kspace_loss(..., norm=None)`.
- Produces in the score JSON: per contrast `head_psnr, head_ssim, head_hfen, grain_ratio, head_frac, gmax, kz_energy, physics`; top level `known_zero: bool, prewhiten: bool`.

- [ ] **Step 1: Record the pre-change iteration-1 loss (regression anchor, GPU, ~3 min)**

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID /home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py \
  --exp reg_anchor_pre --gpu 3 --iters 10 --warmup 2 --lambda_geo 0.10 --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 --base_cfg configs/winner_v2_edgeTV.yaml \
  2>&1 | tee results_arc/reg_anchor_pre.log
grep -m1 "iter     1/" results_arc/reg_anchor_pre.log
```
Write the `loss=` and `dc_p=` / `dc_c=` values of iter 1 into `docs/superpowers/records/af12_to_af10/regression_anchor.txt` (created now, committed in Step 8).

- [ ] **Step 2: Modify `JointTrainer`**

Imports (next to the other `from data...` imports):

```python
from dataclasses import asdict
from data.known_zero import build_fit_tensors, estimate_known_zero, estimate_noise_cov, whitening_matrix
```

Append to the end of `_setup_data` (after the `print(f"|zf_p| max ...")` line):

```python
        # --- Phase 1 physics levers (spec 2026-09-05 §4): known-zero region Z, noise prewhitening W.
        # Defaults: every fit-side tensor IS the original tensor and the DC denominator is |Omega|,
        # so a run without the switches is bit-identical to before.
        self.mask_fit_p, self.kspace_dc_p, self.kspace_fit_p, self.csm_fit_p = self.mask_p, self.kspace_p, self.kspace_p, self.csm_p
        self.mask_fit_c, self.kspace_dc_c, self.kspace_fit_c, self.csm_fit_c = self.mask_c, self.kspace_c, self.kspace_c, self.csm_c
        self._dc_norm_p = max(self.mask_p.sum().item(), 1.0)   # |Omega|: never grows with Z (spec §4.1)
        self._dc_norm_c = max(self.mask_c.sum().item(), 1.0)
        self.physics_info: Dict[str, Any] = {}
        prewhiten = bool(self.config.get("loss", {}).get("prewhiten", False))
        for tag, sub in (("p", "plain"), ("c", "contrast")):
            use_kz = bool(data_cfg[sub].get("known_zero", False))
            if not (use_kz or prewhiten):
                continue
            ds = self.ds_p if tag == "p" else self.ds_c
            y, m, S = getattr(self, f"kspace_{tag}"), getattr(self, f"mask_{tag}"), getattr(self, f"csm_{tag}")
            Z, info = estimate_known_zero(y, m, int(ds.readout_axis))
            W, winfo = None, {}
            if prewhiten:
                Sigma, winfo = estimate_noise_cov(y, m, Z, info)
                W = whitening_matrix(Sigma) if Sigma is not None else None
            if not use_kz:
                Z = torch.zeros_like(m, dtype=torch.bool)
            mask_fit, k_dc, k_fit, S_fit = build_fit_tensors(y, m, S, Z, W)
            setattr(self, f"mask_fit_{tag}", mask_fit)
            setattr(self, f"kspace_dc_{tag}", k_dc)
            setattr(self, f"kspace_fit_{tag}", k_fit)
            setattr(self, f"csm_fit_{tag}", S_fit)
            rec = {"known_zero": use_kz, "prewhiten": bool(prewhiten and W is not None), **asdict(info), **winfo,
                   "frac_z_used": float(Z.float().mean())}
            self.physics_info[tag] = rec
            print(f"[physics:{sub}] known_zero={use_kz} rho_c={info.rho_c} kro_c={info.kro_c} "
                  f"frac_z_used={rec['frac_z_used']:.4f} n0_pe={info.n0_pe:.3e} "
                  f"prewhiten={rec['prewhiten']} sigma_source={winfo.get('sigma_source')} "
                  f"n_noise={winfo.get('n_noise_samples')} corr_max={winfo.get('coil_corr_max_offdiag')}")
```

`_kspace_loss`:

```python
    def _kspace_loss(self, vol: torch.Tensor, csm: torch.Tensor, ksp_meas: torch.Tensor,
                     mask: torch.Tensor, freq_weight: torch.Tensor | None,
                     norm: float | None = None) -> torch.Tensor:
        pred_ksp = fft3c(vol.unsqueeze(0) * csm)
        diff = (pred_ksp - ksp_meas) * mask.unsqueeze(0)
        if freq_weight is not None:
            diff = diff * freq_weight.unsqueeze(0)
        denom = norm if norm is not None else max(mask.sum().item(), 1.0)
        return (diff.real.pow(2) + diff.imag.pow(2)).sum() / denom
```

Train loop (the two DC lines):

```python
            l_dc_p = self._kspace_loss(vol_p, self.csm_fit_p, self.kspace_fit_p, self.mask_fit_p, freq_w_p, norm=self._dc_norm_p)
            l_dc_c = self._kspace_loss(vol_c, self.csm_fit_c, self.kspace_fit_c, self.mask_fit_c, freq_w, norm=self._dc_norm_c)
```

`export_results`:

```python
        vol_p_dc = _hard_dc(vol_p, self.csm_p, self.kspace_dc_p, self.mask_fit_p)
        vol_c_dc = _hard_dc(vol_c, self.csm_c, self.kspace_dc_c, self.mask_fit_c)
```

- [ ] **Step 3: Modify `arc/cycle.py`**

argparse (after `--save_volumes`):

```python
ap.add_argument("--known_zero", action="store_true")            # data.*.known_zero: vendor band-limit / shutter as measured zeros (spec 2026-09-05 §4.1)
ap.add_argument("--prewhiten", action="store_true")             # loss.prewhiten: coil-noise whitening on the fit side only (§4.2)
```

imports (after `from metrics import evaluate_reconstruction`):

```python
sys.path.insert(0, str(ROOT))
from diag_artefact.headonly import score_headonly
```

`build_cfg` (before `cfg.setdefault("output", {})`):

```python
    if args.known_zero:
        cfg["data"]["plain"]["known_zero"] = True
        cfg["data"]["contrast"]["known_zero"] = True
    if args.prewhiten:
        cfg["loss"]["prewhiten"] = True
```

`score_contrast` — replace the whole function:

```python
def score_contrast(tr, which, dev):
    """Score hard-DC image quality and raw-model unobserved multicoil k-space error.
    Hard-DC uses the FIT-side tensors (mask ∪ Z, y zeroed on Z); e_unobs and the ZF anchor use the
    ORIGINAL mask/y so they stay comparable with every earlier run (spec 2026-09-05 §4.1)."""
    if which == "plain":
        csm, y, mask, ds = tr.csm_p, tr.kspace_p, tr.mask_p, tr.ds_p
        mask_fit, y_dc, tag = tr.mask_fit_p, tr.kspace_dc_p, "p"
    else:
        csm, y, mask, ds = tr.csm_c, tr.kspace_c, tr.mask_c, tr.ds_c
        mask_fit, y_dc, tag = tr.mask_fit_c, tr.kspace_dc_c, "c"
    gt = ds.get_data()["ground_truth"].to(dev)
    m = (mask if mask.ndim == 3 else mask[0]).to(dev).float()
    mf = (mask_fit if mask_fit.ndim == 3 else mask_fit[0]).to(dev).float()
    with torch.no_grad():
        x = tr._render(which)
        k_model = fft3c(csm * x.unsqueeze(0))
        y_full = ds.kspace_multicoil.to(dev)
        unobserved = (1.0 - m).unsqueeze(0)
        unobserved_error = ((k_model - y_full) * unobserved).abs().square().sum()
        unobserved_energy = (y_full * unobserved).abs().square().sum().clamp_min(1e-12)
        nmse_unobs = float(unobserved_error / unobserved_energy)
        e_unobs = float(torch.sqrt(unobserved_error / unobserved_energy))
        k_dc = mf.unsqueeze(0) * y_dc + (1.0 - mf).unsqueeze(0) * k_model
        z = (mf - m).clamp_min(0.0).unsqueeze(0)                       # Z = fit mask minus acquired mask
        kz_energy = float((k_dc * z).abs().square().sum())              # must be 0.0 whenever Z is non-empty
        x_dc = (ifft3c(k_dc) * csm.conj()).sum(0)
        met = evaluate_reconstruction(x_dc, gt, compute_extended=False)
        hf = hfen(x_dc.abs(), gt.abs())
        sr = sharp_ratio(x_dc.abs(), gt.abs())
        gr = grain(x_dc.abs(), gt.abs())
        ho = score_headonly(x_dc.abs().cpu().numpy(), gt.abs().cpu().numpy())
        x_zf = (ifft3c(m.unsqueeze(0) * y) * csm.conj()).sum(0)
        mz = evaluate_reconstruction(x_zf, gt, compute_extended=False)
        zf_hf = hfen(x_zf.abs(), gt.abs())
    psnr, ssim, nmse = float(met["psnr"]), float(met["ssim"]), float(met["nmse"])
    comp = psnr + 25.0 * ssim - 60.0 * nmse - 25.0 * hf - 40.0 * max(0.0, 1.0 - sr) - 200.0 * gr
    beat = (ssim >= float(mz["ssim"])) and (nmse <= float(mz["nmse"])) and (hf <= zf_hf)
    return dict(psnr=psnr, ssim=ssim, nmse=nmse, hfen=hf, sharp=sr, grain=gr,
                e_unobs=e_unobs, nmse_unobs=nmse_unobs, comp=comp, beat=beat,
                zf=(float(mz["psnr"]), float(mz["ssim"]), float(mz["nmse"]), zf_hf),
                kz_energy=kz_energy, physics=tr.physics_info.get(tag, {}), **ho)
```

`main` — extend the two per-contrast log lines and the JSON row:

```python
    log(f"{name_p:5s} psnr={sp['psnr']:.3f} ssim={sp['ssim']:.4f} nmse={sp['nmse']:.4f} hfen={sp['hfen']:.4f} "
        f"e_unobs={sp['e_unobs']:.4f} head_psnr={sp['head_psnr']:.3f} head_ssim={sp['head_ssim']:.4f} "
        f"head_hfen={sp['head_hfen']:.4f} N={sp['grain_ratio']:.3f} kz_energy={sp['kz_energy']:.3e} "
        f"sharp={sp['sharp']:.3f} grain={sp['grain']:.4f} comp={sp['comp']:.3f} beat={sp['beat']}")
    log(f"{name_c:5s} psnr={sc['psnr']:.3f} ssim={sc['ssim']:.4f} nmse={sc['nmse']:.4f} hfen={sc['hfen']:.4f} "
        f"e_unobs={sc['e_unobs']:.4f} head_psnr={sc['head_psnr']:.3f} head_ssim={sc['head_ssim']:.4f} "
        f"head_hfen={sc['head_hfen']:.4f} N={sc['grain_ratio']:.3f} kz_energy={sc['kz_energy']:.3e} "
        f"sharp={sc['sharp']:.3f} grain={sc['grain']:.4f} comp={sc['comp']:.3f} beat={sc['beat']}")
```

```python
    row = dict(exp=args.exp, iters=args.iters, lambda_geo=args.lambda_geo, warmup=args.warmup,
               lambda_dtv=args.lambda_dtv, max_radius=args.max_radius, kspace_high=args.kspace_high,
               known_zero=bool(args.known_zero), prewhiten=bool(args.prewhiten),
               plain=sp, contrast=sc, plain_name=name_p, contrast_name=name_c,
               combined=combined, both_beat=both_beat, npts=int(npts), vram_gb=vram)
```

`_vol` (so the saved npz is the same hard-DC volume that was scored):

```python
        @torch.no_grad()
        def _vol(which):
            if which == "plain":
                csm, y, mask, ds, mask_fit, y_dc = tr.csm_p, tr.kspace_p, tr.mask_p, tr.ds_p, tr.mask_fit_p, tr.kspace_dc_p
            else:
                csm, y, mask, ds, mask_fit, y_dc = tr.csm_c, tr.kspace_c, tr.mask_c, tr.ds_c, tr.mask_fit_c, tr.kspace_dc_c
            gt = ds.get_data()["ground_truth"].to(dev)
            m = (mask if mask.ndim == 3 else mask[0]).to(dev).float().unsqueeze(0)
            mf = (mask_fit if mask_fit.ndim == 3 else mask_fit[0]).to(dev).float().unsqueeze(0)
            x = tr._render(which); k_model = fft3c(csm * x.unsqueeze(0))
            x_dc = (ifft3c(mf * y_dc + (1.0 - mf) * k_model) * csm.conj()).sum(0)
            x_zf = (ifft3c(m * y) * csm.conj()).sum(0)                # zero-filled from the ORIGINAL acquisition
            f = lambda t: t.abs().cpu().numpy().astype("float32")
            return f(x_dc), f(gt), f(x_zf)
```

The TSV header/row are left unchanged (the old file would mis-align new columns); head-only metrics live in the JSON.

- [ ] **Step 4: Regression check with the switches OFF (GPU, ~3 min)**

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID /home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py \
  --exp reg_anchor_post --gpu 3 --iters 10 --warmup 2 --lambda_geo 0.10 --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 --base_cfg configs/winner_v2_edgeTV.yaml \
  2>&1 | tee results_arc/reg_anchor_post.log
grep -m1 "iter     1/" results_arc/reg_anchor_pre.log; grep -m1 "iter     1/" results_arc/reg_anchor_post.log
```
Expected: iter-1 `loss`, `dc_p`, `dc_c` equal to the pre-change values to ≥ 5 significant digits (the render is near-deterministic; the code path is identical). Also expected in the JSON: `kz_energy == 0.0` on both contrasts only because Z is empty (`physics == {}`), and `head_psnr` present. Any larger difference = a wiring bug; fix before continuing.

- [ ] **Step 5: Integration check with the switches ON (GPU, ~4 min)**

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID /home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py \
  --exp kz_pw_smoke10 --gpu 3 --iters 10 --warmup 2 --lambda_geo 0.10 --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 --base_cfg configs/winner_v2_edgeTV.yaml \
  --known_zero --prewhiten 2>&1 | tee results_arc/kz_pw_smoke10.log
grep "\[physics:" results_arc/kz_pw_smoke10.log
env PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python - <<'EOF'
import json; d = json.load(open("results_arc/kz_pw_smoke10_score.json"))
for k in ("plain", "contrast"):
    p = d[k]["physics"]; print(k, "rho_c", p["rho_c"], "kro_c", p["kro_c"], "frac_z_used", round(p["frac_z_used"], 4),
                                "sigma_source", p.get("sigma_source"), "n_noise", p.get("n_noise_samples"),
                                "corr_max", p.get("coil_corr_max_offdiag"), "kz_energy", d[k]["kz_energy"])
    assert d[k]["kz_energy"] == 0.0
EOF
```
Expected (from the 2026-09-04 audit): FLAIR `rho_c` ≈ 1.00 (0.95–1.05), T1 `rho_c` ≈ 1.28–1.35, `kz_energy == 0.0` for both, FLAIR `sigma_source == "known_zero"` with ≥ 1e5 samples, T1 `sigma_source == "outer_ring"` (its Z is exact zeros), `coil_corr_max_offdiag` ≈ 0.5 for T1. **Diagnostic (oracle, labelled as such, not used for fitting):** compare `rho_c` with the audit's profile in `docs/superpowers/records/probe_output/diag_data_information_probe2_20260904.log`; write both into the Phase 1 record. If FLAIR yields `rho_c is None`, that is a finding to record, not a reason to change the rule constants (spec §4.1); stop and report before running Phase 1.

Also run the existing regression tests once: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_tv_charbonnier.py tests/test_known_zero.py tests/diag_artefact -q` → all pass.

- [ ] **Step 6: Commit**

```bash
git add paired_gs/trainers/joint_trainer.py arc/cycle.py docs/superpowers/records/af12_to_af10/regression_anchor.txt
git commit -m "feat(trainer): known-zero region and prewhitening on the fit side; head-only metrics in the scorer"
```

---

### Task 8: Phase 1 runs, gates, figures, record

**Files:**
- Create: `docs/superpowers/records/af12_to_af10/scores/{af12_kz_score.json, af12_kz_pw_score.json, af12_kz_verdict.json, af12_kz_pw_verdict.json}`, `docs/superpowers/records/af12_to_af10/2026-09-XX-phase1-physics.md`, `docs/figures/af12_to_af10/*.png`
- Modify: `docs/superpowers/records/af12_to_af10/decisions.jsonl` (append)

**Interfaces:**
- Consumes: Task 7 CLI; `scripts/eval_gates.py`; `scripts/fig_3views_jet.py <npz> <score.json> <out_prefix> "<title>"`; `scripts/fig_butterfly_z160.py <af12.npz> <af10.npz> <out_prefix>`.
- Produces: the `base1` decision for Plan 2 (spec §4.5 rule).

- [ ] **Step 1: Launch both full-budget runs in parallel (GPU 0 and 1, ~1 h)**

```bash
COMMON="--iters 4000 --lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 --base_cfg configs/winner_v2_edgeTV.yaml --save_volumes"
PY="env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID /home/congcongliu/anaconda3/envs/GSTMI/bin/python -u"
nohup $PY arc/cycle.py --exp af12_kz    --gpu 0 $COMMON --known_zero             > results_arc/af12_kz.log 2>&1 &
nohup $PY arc/cycle.py --exp af12_kz_pw --gpu 1 $COMMON --known_zero --prewhiten > results_arc/af12_kz_pw.log 2>&1 &
```
Monitor with `tail -n 3 results_arc/af12_kz.log results_arc/af12_kz_pw.log`; done when both logs contain `DONE ->`.

- [ ] **Step 2: Evaluate the gates and copy the score JSONs**

```bash
R=docs/superpowers/records/af12_to_af10
for e in af12_kz af12_kz_pw; do
  cp results_arc/${e}_score.json $R/scores/${e}_score.json
  env PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/eval_gates.py \
    --gates $R/gates.json --score $R/scores/${e}_score.json --out $R/scores/${e}_verdict.json
done
```

- [ ] **Step 3: Figures (3 views + butterfly vs the AF10 reference) and the viewing URLs**

```bash
F=docs/figures/af12_to_af10; mkdir -p $F /home/congcongliu/figs/af12_to_af10
P="env PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python"
AF10=results_arc/af10_joint_edgetv_e040_rerun20260904_volumes.npz
for e in af12_kz af12_kz_pw; do
  $P scripts/fig_3views_jet.py results_arc/${e}_volumes.npz results_arc/${e}_score.json $F/$e "AF12 $e"
  $P scripts/fig_butterfly_z160.py results_arc/${e}_volumes.npz $AF10 $F/${e}_vs_af10
done
cp $F/*.png /home/congcongliu/figs/af12_to_af10/
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8765/af12_to_af10/   # 200, else restart the server (see memory env-amax-remote)
```
Give the user one URL line per figure: `http://localhost:8765/af12_to_af10/<file>.png`, and the absolute path underneath. The visual gate is the user's call.

- [ ] **Step 4: Apply the base1 rule (spec §4.5) and append decisions**

Rule, in order: (1) more FLAIR gates passed; (2) tie → lower FLAIR `e_unobs`; (3) tie → `af12_kz_pw`. A switch whose run regresses ANY metric of either contrast beyond its tolerance relative to `tvs4k_e040` does not enter base1 (record why). Then:

```bash
R=docs/superpowers/records/af12_to_af10
for e in af12_kz af12_kz_pw; do
  env PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/eval_gates.py \
    --gates $R/gates.json --score $R/scores/${e}_score.json --append $R/decisions.jsonl --phase 1 \
    --note "visual gate: <user verdict>; base1 rule: <outcome>"
done
```

- [ ] **Step 5: Write the Phase 1 record and commit**

`docs/superpowers/records/af12_to_af10/2026-09-XX-phase1-physics.md` must contain: the two exact command lines; the Z diagnostics per contrast (`rho_c`, `kro_c`, `frac_z_used`, `n0_pe`, `sigma_source`, `n_noise_samples`, `coil_corr_max_offdiag`, `eig_ratio`) next to the audit's oracle profile values (labelled oracle); a table `run × contrast × {head_psnr, head_ssim, head_hfen, N, e_unobs, whole psnr}` with the AF12 baseline and AF10 reference rows; the per-gate verdicts; the user's visual verdict; the base1 decision; and what T1 gained (recorded, not judged). Then:

```bash
git add docs/superpowers/records/af12_to_af10 docs/figures/af12_to_af10
git commit -m "records: Phase 1 physics levers (known-zero region, prewhitening) at AF12 - gates and base1 decision"
```

Update the memory card `af12-to-af10-quality.md` (outside the repo) with the base1 decision and the numbers.

---

## Self-review (done while writing)

- **Spec coverage.** §2 gates → Tasks 1, 3, 4 (definitions, tolerance floors, e_unobs baseline, N ≤ 1, visual gate recorded in the ledger). §3 Phase 0 items 1–5 → Tasks 1–4 (item 1 = Task 7's scorer change; item 2 = Task 2; items 3–5 = Task 4). §4.1 Z (acquired-only estimate, cliff rule, constants, |Ω| denominator, hard-DC zero, e_unobs/ZF on the original mask, diagnostics vs the oracle profile) → Tasks 5, 7. §4.2 W (Σ from Z∩Ω, fallbacks, normalisation, fit-side only) → Tasks 6, 7. §4.3 files → as listed, with the documented deviation (no new yaml). §4.4 tests → Tasks 1, 2, 3, 5, 6 plus the two GPU checks in Task 7 (the smoke JSON regression in the spec is realised as the iter-1 loss anchor, because `tests/smoke_test.py` carries a 0.6 dB tolerance and cannot see a bit-level change). §4.5 runs and the base1 rule → Task 8. The 0.9–1.0 transition-band follow-up is NOT in this plan by design (spec §4.1: only if FLAIR misses by about one tolerance; it would be a separate, recorded step).
- **Placeholders.** None: every code step is complete; the only "XX" is the date in a record file name, filled on the day.
- **Type consistency.** `score_headonly` keys (`head_psnr, head_ssim, head_hfen, grain_ratio, head_frac, gmax`) are the ones `make_gates.py`, `eval_gates.py`, and the scorer JSON use. `estimate_known_zero` returns `(Z, KnownZeroInfo)` and `estimate_noise_cov(kspace, mask, Z, info, *, min_samples)` in both the module and the trainer. `build_fit_tensors` returns `(mask_fit, kspace_dc, kspace_fit, csm_fit)` in that order everywhere. Replay JSON is read at `["modalities"][key]["e_unobs"]` as the script writes it.

---

## Addendum 2026-09-06 — spec appendix A (the estimator rule was falsified on the real data)

Task 7's integration run showed that the shipped ring/cliff estimator cannot see the vendor support on this
data (see spec appendix A and `docs/superpowers/records/af12_to_af10/diagnostics/`). Tasks 9 and 10 below
replace Tasks 5–6's estimator and re-run Task 7's integration check. **Execution order: 9 → 10 → 8.** Task 8's
text is unchanged except that the two GPU cards are chosen from `nvidia-smi` idle cards at dispatch time.

Facts the new code must reproduce (measured, oracle-labelled where the full k-space was used):

| | T1 | FLAIR |
|---|---|---|
| zero lines (whole readout line, max\|y\| < 1e-10·max) | 20.0% of the PE plane | 39.2% |
| geometry | rectangle: PE axis-1 index < 44 is zero (padding); nothing else | ellipse centred on DC (128,132), semi-axes (123, 88) px, ∪ the same padding; soft roll-off inside r∈[0.85,1.0] |
| sampled-only shape fit | rect cut at 44, misclassified 0 / 4760 | ellipse (123.0, 88.0), misclassified 73 / 4760 (the roll-off/notch lines) |
| values between 1e-12 and 1e-4 of max | none | 8.8% of sampled voxels (roll-off, all inside the ellipse) |
| noise floor per (coil, sample) | 3.0e-4 (PE r ≥ 0.7) = readout ends 2.8–3.4e-4 | readout ends 4.6–4.8e-4 |

### Task 9: Zero-line shape estimator and readout-end noise covariance (replaces the Task 5/6 estimator)

**Files:**
- Rewrite: `paired_gs/data/known_zero.py` (keep `whitening_matrix` and `build_fit_tensors` verbatim; replace everything else)
- Rewrite: `tests/test_known_zero.py`

**Interfaces (unchanged names, so Task 7's wiring keeps working):**
- `estimate_known_zero(kspace (C,D,H,W) complex, mask (D,H,W), ro_axis: int) -> (Z: BoolTensor (D,H,W), KnownZeroInfo)`
- `KnownZeroInfo` dataclass fields: `dc: tuple, ro_axis: int, zero_threshold: float, cuts: dict, ellipse: tuple|None, n_sampled_lines: int, n_zero_lines: int, n_zero_explained: int, n_nonzero_in_z: int, frac_z: float, frac_sampled_in_z: float` (all JSON-serialisable via `dataclasses.asdict`).
- `estimate_noise_cov(kspace, mask, Z, info, *, min_samples=MIN_NOISE_SAMPLES, end_frac=READOUT_END_FRAC) -> (Sigma|None, winfo)`; `winfo` always has `sigma_source ∈ {"readout_ends","identity"}, n_noise_samples, coil_corr_max_offdiag, coil_corr_mean_offdiag, eig_ratio, noise_level`.
- `whitening_matrix(Sigma, eig_floor=1e-6)`, `build_fit_tensors(kspace, mask, csm, Z, W)` — unchanged from commit c8adb79.
- Constants: `ZERO_REL = 1e-10, COARSE_STEP_PX = 4.0, GRID_STEP_PX = 0.5, EXPAND_STEP = 0.005, EXPAND_MAX = 2.0, READOUT_END_FRAC = 0.125, MIN_NOISE_SAMPLES = 2000, CUT_SPAN_MIN = 0.5`.

- [ ] **Step 1: Replace `tests/test_known_zero.py` with the following and run it (expect ImportError / AttributeError failures)**

```python
"""Known-zero region Z (spec 2026-09-05 §4.1 as amended by appendix A, 2026-09-06): whole PE lines that are
numerically zero outside a hard shutter made of padding cuts and/or a DC-centred ellipse. Estimated from the
ACQUIRED lines only. Plus the readout-end noise covariance (R2), whitening and the fit-side tensors.

Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_known_zero.py -v
"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
from data.known_zero import (MIN_NOISE_SAMPLES, KnownZeroInfo, build_fit_tensors, estimate_known_zero,  # noqa: E402
                             estimate_noise_cov, whitening_matrix)

C, P0, P1, R = 4, 48, 40, 24


def _synth(cut_lo1=None, ellipse=None, noise=2e-3, seed=0, ro_axis=2, dc_shift=(0, 4), hard=True):
    """Textured-object multicoil k-space (C,P0,P1,R) with the DC shifted by `dc_shift` on the PE plane (the real
    data has a +22 px offset), whole PE lines outside the vendor support made numerically zero: array columns
    j < cut_lo1 (padding) and/or lines outside an axis-aligned ellipse (a, b) about the DC. hard=True zeroes
    exactly; hard=False leaves a 1e-30 residue like the real data. vd-random mask (35%) + ACS about the DC,
    broadcast along readout. Returns (kspace, mask, dc, keep) with the readout axis moved to `ro_axis`."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.ogrid[:P0, :P1, :R]
    blob = np.exp(-(((zz - P0 / 2) / (P0 / 4)) ** 2 + ((yy - P1 / 2) / (P1 / 4)) ** 2 + ((xx - R / 2) / (R / 4)) ** 2))
    tex = ndimage.gaussian_filter(rng.normal(size=(P0, P1, R)), 0.7)
    obj = blob * (1.0 + 0.8 * tex / tex.std())
    ks = []
    for _ in range(C):
        cm = ndimage.gaussian_filter(rng.normal(size=(P0, P1, R)), 8.0)
        cm = cm / np.abs(cm).max() + 0.5
        ks.append(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(cm * obj), norm="ortho")))
    k = np.stack(ks).astype(np.complex64)
    k = k + noise * (rng.normal(size=k.shape) + 1j * rng.normal(size=k.shape)).astype(np.complex64) / np.sqrt(2)
    k = np.roll(k, dc_shift, axis=(1, 2))
    dc = (P0 // 2 + dc_shift[0], P1 // 2 + dc_shift[1])
    ii, jj = np.meshgrid(np.arange(P0) - dc[0], np.arange(P1) - dc[1], indexing="ij")
    keep = np.ones((P0, P1), bool)
    if cut_lo1 is not None:
        keep[:, :cut_lo1] = False
    if ellipse is not None:
        a, b = ellipse
        keep &= (ii / a) ** 2 + (jj / b) ** 2 < 1.0
    fac = np.where(keep, 1.0, 0.0 if hard else 1e-30).astype(np.float32)
    k = k * fac[None, :, :, None]
    m2 = rng.random((P0, P1)) < 0.35
    m2[dc[0] - 4:dc[0] + 4, dc[1] - 4:dc[1] + 4] = True
    mask = np.repeat(m2[:, :, None], R, axis=2).astype(np.float32)
    k = k * mask[None]
    kt, mt = torch.from_numpy(k), torch.from_numpy(mask)
    return torch.movedim(kt, 3, ro_axis + 1).contiguous(), torch.movedim(mt, 2, ro_axis).contiguous(), dc, keep


def test_padding_cut_is_found_and_broadcast_along_readout():
    k, m, dc, keep = _synth(cut_lo1=12)
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    assert isinstance(info, KnownZeroInfo) and info.ellipse is None
    lo = info.cuts["axis1_lo"]
    assert lo is not None and 12 <= lo <= 15                      # first SAMPLED nonzero column is >= 12
    assert info.cuts["axis0_lo"] is None and info.cuts["axis0_hi"] is None and info.cuts["axis1_hi"] is None
    zn = Z.numpy()
    assert zn.dtype == bool and zn.shape == tuple(m.shape)
    assert zn[:, :12, :].all() and not zn[:, lo:, :].any()
    assert (zn == zn[:, :, :1]).all()                              # broadcast along readout
    assert info.n_nonzero_in_z == 0 and info.n_zero_explained == info.n_zero_lines
    assert info.frac_z == pytest.approx(lo / P1, abs=1e-9)


def test_ellipse_is_found_within_one_pixel():
    k, m, dc, keep = _synth(ellipse=(18.0, 14.0))
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    assert info.ellipse is not None
    a, b, s = info.ellipse
    assert abs(a - 18.0) <= 1.0 and abs(b - 14.0) <= 1.0 and 1.0 <= s <= 1.1
    assert info.dc[:2] == dc
    assert all(v is None for v in info.cuts.values())              # an ellipse tip is not a padding cut
    ii, jj = np.meshgrid(np.arange(P0) - dc[0], np.arange(P1) - dc[1], indexing="ij")
    r = np.sqrt((ii / 18.0) ** 2 + (jj / 14.0) ** 2)
    zn = Z.numpy()[:, :, 0]
    assert zn[r >= 1.12].all() and not zn[r <= 0.98].any()
    assert info.n_nonzero_in_z == 0 and info.n_zero_explained >= 0.9 * info.n_zero_lines


def test_ellipse_plus_padding_cut_with_residue():
    k, m, dc, keep = _synth(cut_lo1=8, ellipse=(20.0, 22.0), hard=False)     # 1e-30 residue like the real data
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    assert info.cuts["axis1_lo"] is not None and info.ellipse is not None
    zn = Z.numpy()[:, :, 0]
    assert zn[:, :8].all()
    sampled = m.numpy()[:, :, 0] > 0.5
    assert not zn[keep & sampled].any()                            # never a sampled nonzero line
    assert info.n_nonzero_in_z == 0 and info.frac_z > 0.3


def test_no_shutter_gives_empty_Z():
    k, m, dc, keep = _synth()
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    assert not bool(Z.any()) and info.ellipse is None and all(v is None for v in info.cuts.values())
    assert info.n_zero_lines == 0 and info.frac_z == 0.0


def test_uses_only_sampled_points():
    k, m, dc, keep = _synth(ellipse=(18.0, 14.0), seed=3)
    _, a = estimate_known_zero(k, m, ro_axis=2)
    junk = torch.randn_like(k) * (1 - m).unsqueeze(0) * 10.0
    _, b = estimate_known_zero(k + junk, m, ro_axis=2)
    assert a.ellipse == b.ellipse and a.cuts == b.cuts and a.frac_z == b.frac_z


def test_readout_axis_position_does_not_matter():
    k2, m2, dc, _ = _synth(cut_lo1=10, ellipse=(20.0, 22.0), seed=5, ro_axis=2)
    k0, m0, _, _ = _synth(cut_lo1=10, ellipse=(20.0, 22.0), seed=5, ro_axis=0)
    Z2, i2 = estimate_known_zero(k2, m2, ro_axis=2)
    Z0, i0 = estimate_known_zero(k0, m0, ro_axis=0)
    assert i2.cuts == i0.cuts and i2.ellipse == i0.ellipse and i2.frac_z == i0.frac_z
    assert torch.equal(torch.movedim(Z0, 0, 2), Z2)


def test_info_is_json_serializable():
    k, m, dc, keep = _synth(cut_lo1=8, ellipse=(20.0, 22.0))
    _, info = estimate_known_zero(k, m, ro_axis=2)
    json.dumps(dataclasses.asdict(info))


def _corr_noise(sigma_true, n, seed=0):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(sigma_true)
    w = (rng.normal(size=(sigma_true.shape[0], n)) + 1j * rng.normal(size=(sigma_true.shape[0], n))) / np.sqrt(2)
    return L @ w


def test_noise_cov_from_readout_ends_recovers_sigma():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(C, C)) + 1j * rng.normal(size=(C, C))
    S_true = (A @ A.conj().T + np.eye(C)) * 1e-4
    noise = _corr_noise(S_true, P0 * P1 * R).reshape(C, P0, P1, R).astype(np.complex64)
    mask = (rng.random((P0, P1)) < 0.6)[:, :, None].repeat(R, axis=2).astype(np.float32)
    k = torch.from_numpy(noise) * torch.from_numpy(mask)
    Z, info = estimate_known_zero(k, torch.from_numpy(mask), ro_axis=2)
    S, winfo = estimate_noise_cov(k, torch.from_numpy(mask), Z, info)
    assert winfo["sigma_source"] == "readout_ends" and winfo["n_noise_samples"] > MIN_NOISE_SAMPLES
    assert winfo["n_noise_samples"] == int(mask[:, :, 0].sum()) * 2 * 3      # 1/8 of R=24 is 3 indices per end
    rel = np.linalg.norm(S.numpy().astype(np.complex128) - S_true) / np.linalg.norm(S_true)
    assert rel < 0.1
    assert winfo["noise_level"] == pytest.approx(float(np.real(np.trace(S_true)) / C), rel=0.1)


def test_noise_cov_identity_fallback_has_uniform_keys():
    k, m, dc, keep = _synth(cut_lo1=8)
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    S, winfo = estimate_noise_cov(k, m, Z, info, min_samples=10 ** 9)
    assert S is None and winfo["sigma_source"] == "identity"
    assert set(winfo) == {"sigma_source", "n_noise_samples", "coil_corr_max_offdiag", "coil_corr_mean_offdiag", "eig_ratio", "noise_level"}


def test_whitening_of_diagonal_equal_variance_is_identity():
    S = torch.eye(C, dtype=torch.complex64) * 0.3
    assert torch.equal(whitening_matrix(S), torch.eye(C, dtype=torch.complex64))


def test_whitening_decorrelates_and_keeps_mean_variance():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(C, C)) + 1j * rng.normal(size=(C, C))
    S_true = A @ A.conj().T + np.eye(C)
    W = whitening_matrix(torch.from_numpy(S_true.astype(np.complex64))).numpy().astype(np.complex128)
    out = W @ S_true @ W.conj().T
    c = np.real(np.trace(S_true)) / C
    assert np.allclose(out, c * np.eye(C), atol=1e-4 * c)


def test_build_fit_tensors_defaults_and_hard_dc_zero_on_Z():
    k, m, dc, keep = _synth(cut_lo1=8, ellipse=(20.0, 22.0), seed=2)
    csm = torch.randn(C, *m.shape, dtype=torch.complex64)
    Z, _ = estimate_known_zero(k, m, ro_axis=2)
    mask_fit, k_dc, k_fit, csm_fit = build_fit_tensors(k, m, csm, Z, None)
    assert k_fit is k_dc and csm_fit is csm
    assert torch.equal(mask_fit, torch.maximum(m, Z.float()))
    assert float(k_dc[:, Z].abs().max()) == 0.0
    pred = torch.randn_like(k)
    hard_dc = pred * (1 - mask_fit).unsqueeze(0) + k_dc
    assert float(hard_dc[:, Z].abs().max()) == 0.0
    assert torch.equal(hard_dc[:, (m > 0.5) & ~Z], k[:, (m > 0.5) & ~Z])
    assert float(mask_fit.sum()) > float(m.sum())


def test_build_fit_tensors_with_identity_W_is_exact():
    k, m, dc, keep = _synth(seed=2)
    csm = torch.randn(C, *m.shape, dtype=torch.complex64)
    Z = torch.zeros_like(m, dtype=torch.bool)
    W = torch.eye(C, dtype=torch.complex64)
    mask_fit, k_dc, k_fit, csm_fit = build_fit_tensors(k, m, csm, Z, W)
    assert torch.equal(mask_fit, m) and torch.equal(k_dc, k)
    assert torch.allclose(k_fit, k) and torch.allclose(csm_fit, csm)
```

Run: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_known_zero.py -v`
Expected: import error on the new names / attribute errors — the old estimator has none of `zero_threshold`, `cuts`, `ellipse`.

- [ ] **Step 2: Rewrite `paired_gs/data/known_zero.py`** (keep `whitening_matrix` and `build_fit_tensors` exactly as committed in c8adb79; everything above them becomes:)

```python
"""Known-zero k-space region Z and coil-noise prewhitening W — the Phase 1 physics levers
(docs/superpowers/specs/2026-09-05-af12-to-af10-quality-design.md §4, as amended by appendix A, 2026-09-06).

On this data the vendor's k-space support is a HARD shutter in the phase-encoding (PE) plane: whole readout
lines are numerically zero (|y| ~ 1e-20 of the data scale; nothing between 1e-12 and 1e-4) outside an
axis-aligned region built from rectangular padding cuts and/or an ellipse centred on the DC. Z is estimated
from the ACQUIRED lines only; ZERO_REL, the grid steps and EXPAND_STEP are numerical resolutions, not knobs.

    Z_PE = padding cuts (a side is cut only where the nonzero lines end on a straight, wide edge and every
           sampled line beyond it is zero)
         ∪ exterior of the fitted DC-centred ellipse (expanded until no sampled nonzero line lies outside)
    Z_PE never contains a sampled nonzero line.      Z = Z_PE broadcast along the readout axis.

Σ (prewhitening, R2) comes from the readout-end samples of the sampled nonzero lines, where the per-sample
energy sits on the noise floor for both contrasts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

ZERO_REL = 1e-10            # a sampled line is numerically zero if max|y| over coils & readout < ZERO_REL * global max
COARSE_STEP_PX = 4.0        # ellipse semi-axis grid, coarse stage
GRID_STEP_PX = 0.5          # ellipse semi-axis grid, fine stage (±COARSE_STEP_PX around the coarse optimum)
EXPAND_STEP = 0.005         # safety expansion resolution of the ellipse
EXPAND_MAX = 2.0            # give the ellipse up beyond this expansion
CUT_SPAN_MIN = 0.5          # a padding edge must carry nonzero lines over >= this fraction of the other axis' extent
READOUT_END_FRAC = 0.125    # Σ source: outer 1/8 of the readout indices at both ends
MIN_NOISE_SAMPLES = 2000


@dataclass
class KnownZeroInfo:
    dc: tuple                    # (i, j, k) array index of the sampled-energy peak (PE axes + readout)
    ro_axis: int
    zero_threshold: float        # absolute max|y| below which a sampled line is numerically zero
    cuts: dict                   # axis{0,1}_{lo,hi}: index bound of the zero padding (lo: indices < bound are zero;
                                 # hi: indices > bound are zero), or None
    ellipse: Optional[tuple]     # (a_px, b_px, expand) about dc on the PE axes, or None
    n_sampled_lines: int
    n_zero_lines: int
    n_zero_explained: int        # sampled zero lines inside Z
    n_nonzero_in_z: int          # sampled nonzero lines inside Z — 0 by construction
    frac_z: float                # fraction of the PE plane in Z
    frac_sampled_in_z: float     # |Z ∩ Ω| / |Ω| over lines


def _pe_lines(kspace: torch.Tensor, mask: torch.Tensor, ro_axis: int):
    """Per-PE-line max|y| and energy over the SAMPLED readout entries only; (P0,P1) each, plus sampled (P0,P1)."""
    m_r = torch.movedim(mask > 0.5, ro_axis, -1)
    a_r = torch.movedim(kspace.abs().amax(0), ro_axis, -1) * m_r
    e_r = torch.movedim((kspace.real.square() + kspace.imag.square()).sum(0), ro_axis, -1) * m_r
    return a_r.amax(-1), e_r.sum(-1), m_r.any(-1)


def _rect_cuts(nonzero: np.ndarray, zero: np.ndarray, span_min: float = CUT_SPAN_MIN) -> dict:
    """Per-side padding cuts. A side is cut only if (1) at least one sampled zero line lies beyond the bounding
    box of the sampled nonzero lines on that side (every sampled line there is zero by construction) and (2) the
    edge row/column of the bounding box carries nonzero lines over >= span_min of the other axis' nonzero extent
    (a straight padding edge, not the tip of an ellipse)."""
    cuts = {}
    for ax in (0, 1):
        other = 1 - ax
        nz_idx = np.nonzero(nonzero.any(axis=other))[0]
        z_idx = np.nonzero(zero.any(axis=other))[0]
        if nz_idx.size == 0:
            cuts[f"axis{ax}_lo"] = cuts[f"axis{ax}_hi"] = None
            continue
        lo, hi = int(nz_idx.min()), int(nz_idx.max())
        ext_o = np.nonzero(nonzero.any(axis=ax))[0]
        extent = int(ext_o.max() - ext_o.min() + 1)

        def wide(idx):
            line = np.nonzero(np.take(nonzero, idx, axis=ax))[0]
            return line.size > 0 and (line.max() - line.min() + 1) >= span_min * extent

        cuts[f"axis{ax}_lo"] = lo if (np.any(z_idx < lo) and wide(lo)) else None
        cuts[f"axis{ax}_hi"] = hi if (np.any(z_idx > hi) and wide(hi)) else None
    return cuts


def _rect_zero(shape, cuts: dict) -> np.ndarray:
    z = np.zeros(shape, bool)
    if cuts["axis0_lo"] is not None:
        z[:cuts["axis0_lo"], :] = True
    if cuts["axis0_hi"] is not None:
        z[cuts["axis0_hi"] + 1:, :] = True
    if cuts["axis1_lo"] is not None:
        z[:, :cuts["axis1_lo"]] = True
    if cuts["axis1_hi"] is not None:
        z[:, cuts["axis1_hi"] + 1:] = True
    return z


def _fit_ellipse(di: np.ndarray, dj: np.ndarray, is_nonzero: np.ndarray, is_target: np.ndarray,
                 half0: float, half1: float):
    """Axis-aligned ellipse about the DC on the SAMPLED lines: (a, b, expand) or None.
    di/dj: PE offsets of the sampled lines; is_nonzero: sampled nonzero lines (must end up outside Z);
    is_target: sampled zero lines the padding cuts do not explain. Misclassification = nonzero outside the
    ellipse + target inside it; 'no ellipse' competes with misclassification = |target|. The winner is then
    expanded until no sampled nonzero line lies outside it, and dropped if it then explains no target line."""
    n_target = int(is_target.sum())
    if n_target == 0:
        return None
    di2, dj2 = di.astype(np.float64) ** 2, dj.astype(np.float64) ** 2

    def mis(a, b):
        outside = di2 / (a * a) + dj2 / (b * b) >= 1.0
        return int((outside & is_nonzero).sum() + (~outside & is_target).sum())

    best = (n_target, None)
    for a in np.arange(COARSE_STEP_PX, 1.5 * half0 + COARSE_STEP_PX, COARSE_STEP_PX):
        for b in np.arange(COARSE_STEP_PX, 1.5 * half1 + COARSE_STEP_PX, COARSE_STEP_PX):
            m_ = mis(a, b)
            if m_ < best[0]:
                best = (m_, (float(a), float(b)))
    if best[1] is None:
        return None
    a0, b0 = best[1]
    for a in np.arange(a0 - COARSE_STEP_PX, a0 + COARSE_STEP_PX + 1e-9, GRID_STEP_PX):
        for b in np.arange(b0 - COARSE_STEP_PX, b0 + COARSE_STEP_PX + 1e-9, GRID_STEP_PX):
            if a <= 0 or b <= 0:
                continue
            m_ = mis(a, b)
            if m_ < best[0]:
                best = (m_, (float(a), float(b)))
    a, b = best[1]
    expand = 1.0
    while expand <= EXPAND_MAX:
        outside = di2 / (a * expand) ** 2 + dj2 / (b * expand) ** 2 >= 1.0
        if not bool((outside & is_nonzero).any()):
            break
        expand += EXPAND_STEP
    else:
        return None
    if int((outside & is_target).sum()) == 0:
        return None
    return (a, b, float(round(expand, 6)))


def estimate_known_zero(kspace: torch.Tensor, mask: torch.Tensor, ro_axis: int):
    """Z from the acquired lines only (appendix A, R1). kspace (C,D,H,W) complex, already masked; mask (D,H,W)."""
    assert kspace.ndim == 4 and mask.ndim == 3 and tuple(kspace.shape[1:]) == tuple(mask.shape)
    dev = kspace.device
    amax, energy, sampled = _pe_lines(kspace, mask, ro_axis)
    P0, P1 = sampled.shape
    thr = ZERO_REL * float(amax.max())
    sampled_np = sampled.cpu().numpy()
    zero = (sampled & (amax < thr)).cpu().numpy()
    nonzero = sampled_np & ~zero
    flat = int(torch.argmax(energy))
    dc0, dc1 = flat // P1, flat % P1
    m_r = torch.movedim(mask > 0.5, ro_axis, -1)
    e_ro = (torch.movedim((kspace.real.square() + kspace.imag.square()).sum(0), ro_axis, -1) * m_r).sum((0, 1))
    dc2 = int(torch.argmax(e_ro))
    cuts = _rect_cuts(nonzero, zero)
    z_pe = _rect_zero((P0, P1), cuts)
    si, sj = np.nonzero(sampled_np)
    ell = _fit_ellipse(si - dc0, sj - dc1, nonzero[si, sj], (zero & ~z_pe)[si, sj], P0 / 2, P1 / 2)
    if ell is not None:
        a, b, s = ell
        ii, jj = np.meshgrid(np.arange(P0) - dc0, np.arange(P1) - dc1, indexing="ij")
        z_pe |= (ii / (a * s)) ** 2 + (jj / (b * s)) ** 2 >= 1.0
    z_pe &= ~nonzero                                                  # never override a measured nonzero line
    Rn = int(mask.shape[ro_axis])
    Z = torch.movedim(torch.from_numpy(z_pe).to(dev).unsqueeze(-1).expand(P0, P1, Rn), -1, ro_axis).contiguous()
    dc = [0, 0, 0]
    pe = [a_ for a_ in range(3) if a_ != ro_axis]
    dc[pe[0]], dc[pe[1]], dc[ro_axis] = int(dc0), int(dc1), dc2
    info = KnownZeroInfo(dc=tuple(dc), ro_axis=int(ro_axis), zero_threshold=float(thr), cuts=cuts, ellipse=ell,
                         n_sampled_lines=int(sampled_np.sum()), n_zero_lines=int(zero.sum()),
                         n_zero_explained=int((zero & z_pe).sum()), n_nonzero_in_z=int((nonzero & z_pe).sum()),
                         frac_z=float(z_pe.mean()),
                         frac_sampled_in_z=float((z_pe & sampled_np).sum() / max(int(sampled_np.sum()), 1)))
    return Z, info


def estimate_noise_cov(kspace: torch.Tensor, mask: torch.Tensor, Z: torch.Tensor, info: KnownZeroInfo, *,
                       min_samples: int = MIN_NOISE_SAMPLES, end_frac: float = READOUT_END_FRAC):
    """Coil noise covariance Σ (C×C) from the readout-end samples (outer `end_frac` of the readout indices at
    both ends) of the sampled NONZERO lines (appendix A, R2). `Z` is accepted for interface compatibility and
    not used. Returns (Σ | None, winfo) with a uniform winfo key set; None means W = I."""
    ro = info.ro_axis
    m_r = torch.movedim(mask > 0.5, ro, -1)                           # (P0,P1,R)
    y_r = torch.movedim(kspace, ro + 1, -1)                           # (C,P0,P1,R)
    Rn = m_r.shape[-1]
    n_end = max(int(round(end_frac * Rn)), 1)
    a_line = (y_r.abs().amax(0) * m_r).amax(-1)
    nonzero = m_r.any(-1) & (a_line >= info.zero_threshold)
    sel = torch.zeros_like(m_r)
    sel[..., :n_end] = True
    sel[..., Rn - n_end:] = True
    sel = sel & m_r & nonzero.unsqueeze(-1)
    vals = y_r[:, sel]                                                # (C, N)
    empty = {"coil_corr_max_offdiag": None, "coil_corr_mean_offdiag": None, "eig_ratio": None, "noise_level": None}
    if vals.shape[1] < min_samples or float(vals.abs().max()) < info.zero_threshold:
        return None, {"sigma_source": "identity", "n_noise_samples": int(vals.shape[1]), **empty}
    vals = vals.to(torch.complex128)
    Sigma = (vals @ vals.conj().T) / vals.shape[1]
    d = torch.sqrt(torch.diagonal(Sigma).real)
    corr = (Sigma / (d.view(-1, 1) * d.view(1, -1))).abs()
    corr.fill_diagonal_(0.0)
    ev = torch.linalg.eigvalsh(0.5 * (Sigma + Sigma.conj().T)).real
    Cn = Sigma.shape[0]
    winfo = {"sigma_source": "readout_ends", "n_noise_samples": int(vals.shape[1]),
             "coil_corr_max_offdiag": float(corr.max()), "coil_corr_mean_offdiag": float(corr.sum() / (Cn * (Cn - 1))),
             "eig_ratio": float(ev.max() / ev.clamp_min(1e-30).min()),
             "noise_level": float(torch.diagonal(Sigma).real.mean())}
    return Sigma.to(kspace.dtype), winfo
```

(then `whitening_matrix` and `build_fit_tensors` verbatim from c8adb79, including the `Z = Z.bool()` guard.)

- [ ] **Step 3: Run the tests**

Run: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_known_zero.py -v`
Expected: 13 passed, pristine. If `test_ellipse_is_found_within_one_pixel` fails on the `r <= 0.98` bound because the fitted ellipse's expansion pushes Z inward is impossible (expansion only grows the ellipse); if it fails on `r >= 1.12` the expansion exceeded 1.1 — print `info.ellipse` and report; do not loosen the bounds. Also run `tests/test_tv_charbonnier.py` and `tests/diag_artefact` once (unchanged code, must still pass).

- [ ] **Step 4: Commit**

```bash
git add paired_gs/data/known_zero.py tests/test_known_zero.py
git commit -m "feat(data): zero-line shape estimator for the known-zero region; noise covariance from readout ends

Replaces the ring/cliff rule (spec appendix A): sampled PE lines are classified
numerically zero or not (1e-10 of the data scale; the two populations are ten
orders of magnitude apart), Z is the union of straight padding cuts and the
exterior of a DC-centred ellipse fitted on the sampled lines and expanded until
no sampled nonzero line lies outside it; Z never contains a sampled nonzero line.
Sigma comes from the readout-end samples of the nonzero lines, which sit on the
noise floor for both contrasts."
```

### Task 10: Integration re-check of Z / W on the real data, and the known-zero record

**Files:**
- Create: `docs/superpowers/records/af12_to_af10/known_zero_diagnostics.md`, `docs/superpowers/records/af12_to_af10/scores/kz_pw_smoke10b_score.json`
- Create: `scripts/kz_oracle_check.py` (diagnostic, oracle-labelled)

**Interfaces:** consumes Task 7's CLI (`--known_zero --prewhiten`) and Task 9's `estimate_known_zero`.

- [ ] **Step 1: Switches-on integration run (10 iterations, idle GPU)**

```bash
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<100{print $1; exit}')
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID /home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py \
  --exp kz_pw_smoke10b --gpu $G --iters 10 --warmup 2 --lambda_geo 0.10 --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 --base_cfg configs/winner_v2_edgeTV.yaml \
  --known_zero --prewhiten 2>&1 | tee results_arc/kz_pw_smoke10b.log
grep "\[physics:" results_arc/kz_pw_smoke10b.log
```
Expected (hard stop if violated — report BLOCKED with the numbers): T1 `cuts.axis1_lo` in 44..47, `ellipse` None, `frac_z_used` in 0.19..0.21; FLAIR `ellipse` ≈ (123 ± 2, 88 ± 2, expand ≤ 1.03), `frac_z_used` in 0.36..0.40; `n_nonzero_in_z == 0` on both; `sigma_source == "readout_ends"` with `n_noise_samples ≥ 1e5` on both; `kz_energy == 0.0` on both.

- [ ] **Step 2: Oracle check (labelled diagnostic, never used for fitting): `scripts/kz_oracle_check.py`**

```python
#!/usr/bin/env python3
"""ORACLE diagnostic (uses the full k-space; never part of the fit): precision/recall of the estimated Z against
the true numerically-zero lines, per contrast, for the winner_v2_edgeTV data config."""
import sys, json
from pathlib import Path
import numpy as np, torch, yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paired_gs"))
from trainers.joint_trainer import _load_dataset
from data.private_paths import resolve_subject_paths
from data.known_zero import ZERO_REL, estimate_known_zero
cfg = yaml.safe_load(open(ROOT / "configs/winner_v2_edgeTV.yaml"))
out = {}
for sub in ("plain", "contrast"):
    d = dict(cfg["data"][sub])
    for k in ("data_path", "csm_path", "mask_path"):
        d[k] = resolve_subject_paths(d[k])
    ds = _load_dataset(d)
    Z, info = estimate_known_zero(ds.kspace_undersampled_complex, ds.mask, int(ds.readout_axis))
    ro = int(ds.readout_axis)
    a_full = torch.movedim(ds.kspace_multicoil.abs().amax(0), ro, -1)
    true_zero = (a_full.amax(-1) < ZERO_REL * float(a_full.max())).numpy()          # ORACLE whole-line zero map
    z_pe = torch.movedim(Z, ro, -1)[:, :, 0].numpy()
    tp = int((z_pe & true_zero).sum()); fp = int((z_pe & ~true_zero).sum()); fn = int((~z_pe & true_zero).sum())
    out[sub] = {"name": d.get("name", sub), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                "false_positive_lines": fp, "true_zero_fraction": float(true_zero.mean()), "frac_z": info.frac_z,
                "cuts": info.cuts, "ellipse": info.ellipse}
    print(sub, json.dumps(out[sub]))
Path(ROOT / "results_arc" / "kz_oracle_check.json").write_text(json.dumps(out, indent=2))
```
Run it (`env PYTHONNOUSERSITE=1 PYTHONPATH= .../python scripts/kz_oracle_check.py`, CPU, ~4 min). Expected: precision ≥ 0.999 on both (Z must not contain true nonzero lines: `false_positive_lines` ≤ a handful, all in FLAIR's roll-off band), recall ≥ 0.95 (T1) and ≥ 0.85 (FLAIR; the notch and roll-off lines are left out by design).

- [ ] **Step 3: Record and commit**

Copy `results_arc/kz_pw_smoke10b_score.json` to `docs/superpowers/records/af12_to_af10/scores/`. Write `known_zero_diagnostics.md`: the physics lines for both contrasts, the oracle precision/recall table (labelled ORACLE), the Task 7 finding that led here (old rule: T1 rho_c 1.15 / FLAIR None), and the figure paths `docs/figures/af12_to_af10/kz_tiny_geometry_{T1,FLAIR}.png`. Then:

```bash
git add scripts/kz_oracle_check.py docs/superpowers/records/af12_to_af10/known_zero_diagnostics.md docs/superpowers/records/af12_to_af10/scores/kz_pw_smoke10b_score.json
git commit -m "records: known-zero region on the real data - estimator diagnostics and oracle precision/recall"
```

### Task 11: Significance-gated expansion (R1'), oracle energy metrics, integration re-check and record (supersedes Task 10 steps 1–3)

**Why.** Task 10's run on FLAIR: the fit (123, 88) is right, but 36 sampled nonzero lines lie outside it (a cap over rows 1–19 centred ~35 lines left of the DC, and a tighter right boundary), so the safety expansion grew the ellipse to 1.185×, Z fell to 22.6% of the plane (true 39.2%) and 75 truly-nonzero lines were still inside Z. Those outlier lines carry a median energy of 2.6e-5 × the noise floor; the 75 false positives carried 9.4e-9 of the total energy. Gating the expansion by the noise floor measured in R2 (prototype, scored against the oracle) gives: expansion 1.0, Z 39.6%, recall 0.959, and every one of the 1146 false-positive lines below the noise floor (max 0.96×, excess-over-noise energy exactly 0). Forcing zeros on noise-only lines cannot raise the expected error (the model cannot predict noise: E|v−n|² ≥ E|n|²). T1 is unchanged (precision and recall 1.000).

**Files:**
- Modify: `paired_gs/data/known_zero.py`, `tests/test_known_zero.py`
- Replace: `scripts/kz_oracle_check.py` (Task 10 left an untracked draft; overwrite it)
- Create: `docs/superpowers/records/af12_to_af10/known_zero_diagnostics.md`, `docs/superpowers/records/af12_to_af10/scores/kz_pw_smoke10c_score.json`

**Interfaces:** `estimate_known_zero(kspace, mask, ro_axis)` (unchanged signature) now computes the readout-end noise floor internally; `KnownZeroInfo` gains `noise_level_used: float|None, n_outliers: int, n_significant_outliers: int` (appended after `frac_sampled_in_z`); new constant `SIG_NOISE = 3.0`; new helper `_readout_end_samples(kspace, mask, ro_axis, zero_threshold, end_frac) -> Tensor (C, N)` shared with `estimate_noise_cov`.

- [ ] **Step 1: Tests first — extend `_synth` and add two tests to `tests/test_known_zero.py`**

Extend the `_synth` signature with `extra_lines=(), extra_amp=0.0` and, right after `k = k * fac[None, :, :, None]`, add:

```python
    for (i, j) in extra_lines:                                    # sampled lines outside the support with given content
        k[:, i, j, :] = (extra_amp * (rng.normal(size=(C, R)) + 1j * rng.normal(size=(C, R))) / np.sqrt(2)).astype(np.complex64)
```
and, right after `m2[dc[0] - 4:dc[0] + 4, dc[1] - 4:dc[1] + 4] = True`, add:
```python
    for (i, j) in extra_lines:
        m2[i, j] = True
```

Then add (after `test_no_shutter_gives_empty_Z`):

```python
OUTSIDE = [(24 + 20, 24), (24 - 20, 24), (24, 24 + 16), (24, 24 - 16)]    # just outside the (18, 14) ellipse about dc=(24, 24)


def test_expansion_ignores_noise_level_outliers():
    k, m, dc, keep = _synth(ellipse=(18.0, 14.0), extra_lines=OUTSIDE, extra_amp=2e-3)   # pure noise at the fixture's noise level
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    assert info.noise_level_used is not None and info.n_outliers >= 4 and info.n_significant_outliers == 0
    assert info.ellipse is not None and info.ellipse[2] == 1.0
    zn = Z.numpy()[:, :, 0]
    for (i, j) in OUTSIDE:
        assert not zn[i, j]                                        # a sampled nonzero line is never zeroed
    assert zn[24 + 22, 24] and zn[24 - 22, 24] and zn[24, 24 + 18]  # lines further out are still in Z
    assert info.n_nonzero_in_z == 0


def test_expansion_respects_significant_outliers():
    k, m, dc, keep = _synth(ellipse=(18.0, 14.0), extra_lines=OUTSIDE, extra_amp=0.2)    # signal 100x the noise
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    assert info.n_significant_outliers == 4 and info.ellipse is not None
    a, b, s = info.ellipse
    assert s > 1.0
    for (i, j) in OUTSIDE:
        assert ((i - 24) / (a * s)) ** 2 + ((j - 24) / (b * s)) ** 2 < 1.0   # the expanded ellipse covers them
    assert info.n_nonzero_in_z == 0
```

Run: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_known_zero.py -v` → the two new tests fail (`AttributeError: noise_level_used` / unexpected keyword); the other 15 pass.

- [ ] **Step 2: Implement R1' in `paired_gs/data/known_zero.py`**

Constants (add after `READOUT_END_FRAC`):
```python
SIG_NOISE = 3.0             # an outlier line drives the safety expansion only if its mean per-(coil,sample) energy
                            # exceeds SIG_NOISE x the readout-end noise floor: a ~100-sigma detection on a ~3000-sample
                            # mean, not a quality knob. Without a noise floor every outlier counts.
```

`KnownZeroInfo`: append three fields at the end:
```python
    noise_level_used: Optional[float]   # readout-end noise floor (mean |y|^2 per coil sample) used for the gate, or None
    n_outliers: int                     # sampled nonzero lines outside the fitted (unexpanded) ellipse
    n_significant_outliers: int         # ... of which above SIG_NOISE x the noise floor
```

`_pe_lines` returns a fourth item, the sampled readout count per line:
```python
    return a_r.amax(-1), e_r.sum(-1), m_r.any(-1), m_r.sum(-1)
```

New helper (place before `estimate_known_zero`):
```python
def _readout_end_samples(kspace: torch.Tensor, mask: torch.Tensor, ro_axis: int, zero_threshold: float,
                         end_frac: float = READOUT_END_FRAC) -> torch.Tensor:
    """(C, N) samples from the outer `end_frac` of the readout indices at both ends of the sampled NONZERO lines —
    the noise-floor region on this data (appendix A). Shared by the Z gate (R1') and Σ (R2)."""
    m_r = torch.movedim(mask > 0.5, ro_axis, -1)
    y_r = torch.movedim(kspace, ro_axis + 1, -1)
    Rn = m_r.shape[-1]
    n_end = max(int(round(end_frac * Rn)), 1)
    a_line = (y_r.abs().amax(0) * m_r).amax(-1)
    nonzero = m_r.any(-1) & (a_line >= zero_threshold)
    sel = torch.zeros_like(m_r)
    sel[..., :n_end] = True
    sel[..., Rn - n_end:] = True
    sel = sel & m_r & nonzero.unsqueeze(-1)
    return y_r[:, sel]
```

`_fit_ellipse` gains an `is_significant` argument and uses it in the expansion loop (the misclassification fit is unchanged):
```python
def _fit_ellipse(di, dj, is_nonzero, is_target, is_significant, half0, half1):
    ...  # identical up to the expansion loop, then:
    expand = 1.0
    while expand <= EXPAND_MAX:
        outside = di2 / (a * expand) ** 2 + dj2 / (b * expand) ** 2 >= 1.0
        if not bool((outside & is_significant).any()):
            break
        expand += EXPAND_STEP
    else:
        return None
    if int((outside & is_target).sum()) == 0:
        return None
    return (a, b, float(round(expand, 6)))
```
(Docstring: "expanded until no SIGNIFICANT sampled nonzero line lies outside it".)

`estimate_known_zero`: after computing `thr`, `zero`, `nonzero`, and before the ellipse fit:
```python
    amax, energy, sampled, n_ro = _pe_lines(kspace, mask, ro_axis)
    ...
    vals = _readout_end_samples(kspace, mask, ro_axis, thr)
    noise_level = float((vals.abs() ** 2).mean()) if (vals.shape[1] >= MIN_NOISE_SAMPLES and float(vals.abs().max()) > 0.0) else None
    e_mean = (energy / (kspace.shape[0] * n_ro.clamp_min(1))).cpu().numpy()      # per (coil, sample), sampled entries only
    significant = nonzero & (e_mean > SIG_NOISE * noise_level) if noise_level is not None else nonzero.copy()
    ...
    si, sj = np.nonzero(sampled_np)
    is_nz, is_t, is_sig = nonzero[si, sj], (zero & ~z_pe)[si, sj], significant[si, sj]
    ell = _fit_ellipse(si - dc0, sj - dc1, is_nz, is_t, is_sig, P0 / 2, P1 / 2)
    n_out = n_sig = 0
    if ell is not None:
        a, b, s = ell
        out = (((si - dc0) / a) ** 2 + ((sj - dc1) / b) ** 2 >= 1.0) & is_nz          # relative to the UNEXPANDED fit
        n_out, n_sig = int(out.sum()), int((out & is_sig).sum())
        ii, jj = np.meshgrid(np.arange(P0) - dc0, np.arange(P1) - dc1, indexing="ij")
        z_pe |= (ii / (a * s)) ** 2 + (jj / (b * s)) ** 2 >= 1.0
```
and fill the three new info fields (`noise_level_used=noise_level, n_outliers=n_out, n_significant_outliers=n_sig`).

`estimate_noise_cov`: replace its selection block by `vals = _readout_end_samples(kspace, mask, info.ro_axis, info.zero_threshold, end_frac)`; everything after (the guards, Σ, winfo) unchanged.

Run the test file → 17 passed, pristine; also `tests/test_tv_charbonnier.py` and `tests/diag_artefact` once. Commit the two files:
`fix(data): gate the ellipse safety expansion by the readout-end noise floor (appendix A, R1')` with a body stating the FLAIR numbers above (expansion 1.185 → 1.0, recall 0.575 → 0.959, all false positives below the noise floor).

- [ ] **Step 3: Update the trainer print** in `paired_gs/trainers/joint_trainer.py` (`[physics:{sub}]` line): append `f"noise_level_used={info.noise_level_used} outliers={info.n_outliers}/{info.n_significant_outliers} "` before the `prewhiten=` field. Include this file in the same commit as Step 2 (it is a print only).

- [ ] **Step 4: Replace `scripts/kz_oracle_check.py`** with a version that also reports energy metrics (ORACLE, never part of the fit):

```python
#!/usr/bin/env python3
"""ORACLE diagnostic (uses the full k-space; never part of the fit): grades the estimated Z per contrast for the
winner_v2_edgeTV data config — line precision/recall against the true numerically-zero lines, and the ENERGY the
false positives carry: total, and the excess above the readout-end noise floor (a false positive at or below the
noise floor is harmless: the model cannot predict noise, so forcing a zero there never raises the expected error)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paired_gs"))
from trainers.joint_trainer import _load_dataset                       # noqa: E402
from data.private_paths import resolve_subject_paths                   # noqa: E402
from data.known_zero import ZERO_REL, estimate_known_zero, estimate_noise_cov   # noqa: E402

cfg = yaml.safe_load(open(ROOT / "configs/winner_v2_edgeTV.yaml"))
out = {}
for sub in ("plain", "contrast"):
    d = dict(cfg["data"][sub])
    for k in ("data_path", "csm_path", "mask_path"):
        d[k] = resolve_subject_paths(d[k])
    ds = _load_dataset(d)
    y, m, ro = ds.kspace_undersampled_complex, ds.mask, int(ds.readout_axis)
    Z, info = estimate_known_zero(y, m, ro)
    _, winfo = estimate_noise_cov(y, m, Z, info)
    nl = winfo["noise_level"]
    Cn, Rn = y.shape[0], m.shape[ro]
    a_full = torch.movedim(ds.kspace_multicoil.abs().amax(0), ro, -1)
    true_zero = (a_full.amax(-1) < ZERO_REL * float(a_full.max())).numpy()                 # ORACLE
    e_line = torch.movedim((ds.kspace_multicoil.abs() ** 2).sum(0), ro, -1).sum(-1).numpy()  # ORACLE line energies
    z_pe = torch.movedim(Z, ro, -1)[:, :, 0].numpy()
    tp = int((z_pe & true_zero).sum()); fp_mask = z_pe & ~true_zero; fp = int(fp_mask.sum()); fn = int((~z_pe & true_zero).sum())
    E = float(e_line.sum()); fp_e = e_line[fp_mask]
    excess = float(np.maximum(fp_e - (nl * Cn * Rn if nl else 0.0), 0.0).sum()) if fp else 0.0
    out[sub] = {"name": d.get("name", sub), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                "false_positive_lines": fp, "fp_energy_fraction": float(fp_e.sum() / E) if fp else 0.0,
                "fp_excess_over_noise_fraction": excess / E,
                "max_fp_line_energy_over_noise": float(fp_e.max() / (nl * Cn * Rn)) if (fp and nl) else 0.0,
                "true_zero_fraction": float(true_zero.mean()), "frac_z": info.frac_z, "cuts": info.cuts,
                "ellipse": info.ellipse, "noise_level": nl, "n_outliers": info.n_outliers,
                "n_significant_outliers": info.n_significant_outliers, "n_nonzero_in_z": info.n_nonzero_in_z}
    print(sub, json.dumps(out[sub]), flush=True)
Path(ROOT / "results_arc" / "kz_oracle_check.json").write_text(json.dumps(out, indent=2))
```
Run it (CPU, ~4 min). **Acceptance (stop with BLOCKED otherwise):** both contrasts `n_nonzero_in_z == 0` and `fp_excess_over_noise_fraction ≤ 1e-5`; T1 recall ≥ 0.99 and precision ≥ 0.999; FLAIR recall ≥ 0.90 and `max_fp_line_energy_over_noise ≤ 1.5`.

- [ ] **Step 5: Integration run (10 iterations, idle GPU)** exactly as Task 10 Step 1 but `--exp kz_pw_smoke10c`. Expected: T1 `cuts.axis1_lo` 44, ellipse None, frac_z_used 0.19–0.21; FLAIR ellipse ≈ (123 ± 2, 88 ± 2, 1.0), frac_z_used 0.37–0.41, `n_significant_outliers == 0`; both `n_nonzero_in_z == 0`, `sigma_source == "readout_ends"`, `kz_energy == 0.0`.

- [ ] **Step 6: Record and commit.** Copy `results_arc/kz_pw_smoke10c_score.json` to `docs/superpowers/records/af12_to_af10/scores/`. Write `docs/superpowers/records/af12_to_af10/known_zero_diagnostics.md`: the two `[physics:...]` lines, a table of the oracle metrics per contrast (labelled ORACLE), the history (old rule T1 rho_c 1.15 / FLAIR None; first shape estimator FLAIR expand 1.185 / recall 0.575; R1' result), and the figure paths. Commit:
```bash
git add scripts/kz_oracle_check.py docs/superpowers/records/af12_to_af10/known_zero_diagnostics.md docs/superpowers/records/af12_to_af10/scores/kz_pw_smoke10c_score.json
git commit -m "records: known-zero region on the real data - R1' diagnostics and oracle precision/recall/energy"
```
