# Edge-preserving TV (CSF/sharp-edge error) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config-gated edge-weighted TV regularizer to the MC-GS winner and test, against a strict pre-registered bar, whether it reduces sharp-edge (CSF/ventricle) reconstruction error without hurting SSIM.

**Architecture:** A pure edge-weighted-TV function in `paired_gs/losses/edge_tv.py` (detached Perona-Malik spatial weight on per-axis TV), wired into `JointTrainer.train()` behind a `loss.edge_tv` flag; a `scripts/score_edge.py` that computes edge-set / ventricle / gradient-quintile fidelity metrics from a run's `*_volumes.npz`; then a κ×λ sweep and a strict-bar evaluation vs the `reg_aligned_4k` baseline.

**Tech Stack:** Python, PyTorch 2.7.1 (GSTMI conda env), NumPy, SciPy (`ndimage`), scikit-image (SSIM); training via `arc/cycle.py` + `paired_gs` (JointTrainer, tile-CUDA voxelizer). No pytest infra — tests are standalone scripts run with the GSTMI interpreter, matching `tests/smoke_test.py`.

## Global Constraints

- Interpreter (ALWAYS): `PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID /home/congcongliu/anaconda3/envs/GSTMI/bin/python`
- Baseline to beat: `results_arc/reg_aligned_4k` — `configs/winner_v2_aligned.yaml`, 4000 iters, seed 42, warm-start 576902 Gaussians, aligned mask, isotropic TV `lambda_geo=0.06`, `kspace_high=2.2`, `warmup=500`, `max_radius=20`.
- `arc/cycle.py` CLI defaults CLOBBER the yaml: EVERY training command MUST pass `--lambda_geo 0.06 --kspace_high 2.2 --warmup 500 --max_radius 20 --base_cfg configs/winner_v2_edgeTV.yaml --save_volumes`.
- `loss.edge_tv` defaults **False** — never change the shipped winner default.
- Gauge discipline: unified `data_range = P99.9`, NMSE is gauge-free, NEVER @max-PSNR.
- Error-map figures: `jet`, fixed range `[0, 0.1]` (`|recon−GT|/GT_peak`), house convention.
- Run training on an idle GPU (GPU 5 was free this session); pin with `--gpu N`.
- Strict acceptance bar (pre-registered): PASS = `nmse_edge` (or `hfen_edge`) down > noise floor AND whole-brain `ssim` not down AND whole-brain `nmse` not up. "edge down but ssim down" = RESHAPE = FAIL.

---

### Task 1: Edge-weighted TV loss + κ resolver (pure functions, unit-tested)

**Files:**
- Create: `paired_gs/losses/edge_tv.py`
- Test: `tests/test_edge_tv.py`

**Interfaces:**
- Produces: `edge_preserving_tv(vol: torch.Tensor, kappa: float, w_floor: float) -> torch.Tensor` (scalar); `resolve_edge_kappa(vol: torch.Tensor, q: float, brain_frac: float = 0.05) -> float`; `_edge_weight(g: torch.Tensor, kappa: float, w_floor: float) -> torch.Tensor`.

- [ ] **Step 1: Write the failing test** — `tests/test_edge_tv.py`

```python
#!/usr/bin/env python3
"""Standalone unit test for edge-preserving TV (no GPU/data).
Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python tests/test_edge_tv.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
import torch
from losses.edge_tv import edge_preserving_tv, resolve_edge_kappa, _edge_weight

def approx(a, b, tol=1e-5): return abs(float(a) - float(b)) < tol

def test_weight_flat_and_edge():
    w_flat = _edge_weight(torch.zeros(4), kappa=1.0, w_floor=0.1)
    w_edge = _edge_weight(torch.full((4,), 100.0), kappa=1.0, w_floor=0.1)
    assert torch.all(w_flat > 0.99), f"flat weight ~1 expected, got {w_flat}"
    assert torch.all(w_edge <= 0.1 + 1e-6), f"edge weight should hit floor, got {w_edge}"
    print("PASS test_weight_flat_and_edge")

def test_wfloor_respected():
    w = _edge_weight(torch.linspace(0, 1000, 50), kappa=1.0, w_floor=0.2)
    assert torch.all(w >= 0.2 - 1e-6) and torch.all(w <= 1.0 + 1e-6)
    print("PASS test_wfloor_respected")

def test_flat_region_matches_isotropic_tv():
    torch.manual_seed(0)
    vol = 0.01 * torch.randn(8, 8, 8)                       # grain only, no edge
    m = vol.abs()
    iso = ((m[1:]-m[:-1]).abs().mean() + (m[:,1:]-m[:,:-1]).abs().mean() + (m[:,:,1:]-m[:,:,:-1]).abs().mean())
    ew = edge_preserving_tv(vol, kappa=1.0, w_floor=0.1)    # kappa>>grain -> w~1
    assert approx(ew, iso, tol=1e-3), f"flat edge-TV {float(ew)} should ~ iso {float(iso)}"
    print("PASS test_flat_region_matches_isotropic_tv")

def test_edge_penalized_less_than_isotropic():
    vol = torch.zeros(8, 8, 8); vol[4:] = 1.0              # sharp step across axis0
    m = vol.abs()
    iso = ((m[1:]-m[:-1]).abs().mean() + (m[:,1:]-m[:,:-1]).abs().mean() + (m[:,:,1:]-m[:,:,:-1]).abs().mean())
    ew = edge_preserving_tv(vol, kappa=0.1, w_floor=0.1)   # kappa<<step -> step ~w_floor
    assert float(ew) < float(iso), f"edge-TV {float(ew)} should be < iso {float(iso)}"
    print("PASS test_edge_penalized_less_than_isotropic")

def test_differentiable_and_finite():
    vol = torch.randn(8, 8, 8, requires_grad=True)
    loss = edge_preserving_tv(vol, kappa=0.5, w_floor=0.1); loss.backward()
    assert torch.isfinite(loss) and vol.grad is not None and torch.all(torch.isfinite(vol.grad))
    print("PASS test_differentiable_and_finite")

def test_resolve_kappa_positive_finite():
    vol = torch.zeros(10, 10, 10); vol[5:] = 1.0; vol += 0.05 * torch.randn(10, 10, 10)
    k = resolve_edge_kappa(vol, q=0.75)
    assert k > 0 and torch.isfinite(torch.tensor(k)), f"kappa positive finite expected, got {k}"
    print("PASS test_resolve_kappa_positive_finite")

if __name__ == "__main__":
    test_weight_flat_and_edge()
    test_wfloor_respected()
    test_flat_region_matches_isotropic_tv()
    test_edge_penalized_less_than_isotropic()
    test_differentiable_and_finite()
    test_resolve_kappa_positive_finite()
    print("ALL EDGE-TV UNIT TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python tests/test_edge_tv.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'losses.edge_tv'`

- [ ] **Step 3: Write minimal implementation** — `paired_gs/losses/edge_tv.py`

```python
"""Edge-preserving (edge-weighted) anisotropic TV — a self-contained edge-directed
regularizer that smooths flat-region grain while sparing sharp anatomical edges.
Detached Perona-Malik spatial weight => stable (magnitude/bounded-gradient convention).
Spec: docs/superpowers/specs/2026-07-23-edge-preserving-tv-csf-design.md"""
import numpy as np
import torch


def _edge_weight(g: torch.Tensor, kappa: float, w_floor: float) -> torch.Tensor:
    """~1 in flat regions, ->w_floor at sharp edges. |g| detached => spatial weight only."""
    gd = g.detach().abs()
    return torch.clamp(1.0 / (1.0 + (gd / kappa) ** 2), min=w_floor, max=1.0)


def edge_preserving_tv(vol: torch.Tensor, kappa: float, w_floor: float) -> torch.Tensor:
    """Edge-weighted anisotropic TV of |vol|. Per-axis finite differences (matches
    JointTrainer._tv_loss), each down-weighted at its own sharp gradients -> smoothing
    stays ON along an edge but ~OFF across it."""
    m = vol.abs()
    gx = m[1:] - m[:-1]
    gy = m[:, 1:] - m[:, :-1]
    gz = m[:, :, 1:] - m[:, :, :-1]
    return ((_edge_weight(gx, kappa, w_floor) * gx.abs()).mean()
            + (_edge_weight(gy, kappa, w_floor) * gy.abs()).mean()
            + (_edge_weight(gz, kappa, w_floor) * gz.abs()).mean())


def resolve_edge_kappa(vol: torch.Tensor, q: float, brain_frac: float = 0.05) -> float:
    """Absolute edge threshold = q-quantile of the in-brain per-axis gradient magnitude
    of the (initial) render. Fixed once so the edge bands do not drift during training."""
    m = vol.abs()
    thr = brain_frac * float(m.max())
    parts = [
        (m[1:] - m[:-1]).abs()[m[1:] > thr].reshape(-1),
        (m[:, 1:] - m[:, :-1]).abs()[m[:, 1:] > thr].reshape(-1),
        (m[:, :, 1:] - m[:, :, :-1]).abs()[m[:, :, 1:] > thr].reshape(-1),
    ]
    g = torch.cat(parts).detach().cpu().numpy()
    if g.size == 0:
        return max(thr, 1e-8)
    return max(float(np.quantile(g, q)), 1e-8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python tests/test_edge_tv.py`
Expected: PASS — ends with `ALL EDGE-TV UNIT TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add paired_gs/losses/edge_tv.py tests/test_edge_tv.py
git commit -m "feat(edge-tv): edge-weighted TV loss + kappa resolver (unit-tested)"
```

---

### Task 2: Wire edge-TV into JointTrainer + cycle.py CLI + config (integration smoke)

**Files:**
- Modify: `paired_gs/trainers/joint_trainer.py` (imports near line 48; `train()` loss loop lines 775-779; add κ resolution before the loop, before line 760)
- Modify: `arc/cycle.py` (argparse after line 39; `build_cfg` after line 142)
- Create: `configs/winner_v2_edgeTV.yaml`
- Test: `tests/smoke_edge_tv.sh` (2-iter dry-run through the real entry point)

**Interfaces:**
- Consumes: `edge_preserving_tv`, `resolve_edge_kappa` from Task 1.
- Produces: config keys `loss.edge_tv` (bool), `loss.edge_tv_kappa_q` (float), `loss.edge_tv_wfloor` (float); CLI `--edge_tv`, `--edge_tv_kappa_q`, `--edge_tv_wfloor`.

- [ ] **Step 1: Write the failing integration smoke** — `tests/smoke_edge_tv.sh`

```bash
#!/usr/bin/env bash
# 2-iter dry-run of the edge-TV path through arc/cycle.py. Verifies wiring end-to-end.
# Usage: bash tests/smoke_edge_tv.sh <gpu>
set -euo pipefail
GPU="${1:-5}"
cd "$(dirname "$0")/.."
LOG=$(mktemp)
CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONPATH= \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python arc/cycle.py \
  --exp smoke_edge_tv --gpu "$GPU" --iters 2 --warmup 0 \
  --lambda_geo 0.06 --kspace_high 2.2 --max_radius 20 \
  --base_cfg configs/winner_v2_edgeTV.yaml 2>&1 | tee "$LOG"
grep -q "\[edge_tv\] kappa_q" "$LOG" || { echo "FAIL: no [edge_tv] kappa log line"; exit 1; }
echo "PASS smoke_edge_tv"
```

- [ ] **Step 2: Run smoke to verify it fails**

Run: `bash tests/smoke_edge_tv.sh 5`
Expected: FAIL — `configs/winner_v2_edgeTV.yaml` missing / no `[edge_tv]` log line.

- [ ] **Step 3a: Create the config** — `configs/winner_v2_edgeTV.yaml`

Copy `configs/winner_v2_aligned.yaml` verbatim, then in its `loss:` block append these three lines (leave `tv_weight: 0.06` — it is the edge-TV term's weight `λ_edgeTV`, swept later):

```yaml
  edge_tv: true                  # replace isotropic _tv_loss with edge-weighted TV
  edge_tv_kappa_q: 0.75          # in-brain render-gradient quantile -> absolute kappa (fixed at init)
  edge_tv_wfloor: 0.1            # residual edge smoothing (damps Gibbs ringing)
```

- [ ] **Step 3b: Add the import** — `paired_gs/trainers/joint_trainer.py` after line 48 (`from data.transforms import fft3c, ifft3c`)

```python
from losses.edge_tv import edge_preserving_tv, resolve_edge_kappa
```

- [ ] **Step 3c: Read config + resolve κ before the loop** — in `train()`, immediately after line 746 (the `self._static_freq_w_plain = ...` block) and before line 759 (`t0 = time.time()`), insert:

```python
        # --- edge-preserving TV (self-contained edge-directed regularizer) ---
        _loss_cfg = self.config.get("loss", {})
        self._edge_tv = bool(_loss_cfg.get("edge_tv", False))
        self._edge_tv_wfloor = float(_loss_cfg.get("edge_tv_wfloor", 0.1))
        if self._edge_tv:
            _kq = float(_loss_cfg.get("edge_tv_kappa_q", 0.75))
            with torch.no_grad():
                self._edge_kappa_p = resolve_edge_kappa(self._render("plain"), _kq)
                self._edge_kappa_c = resolve_edge_kappa(self._render("contrast"), _kq)
            print(f"[edge_tv] kappa_q={_kq} -> kappa_plain={self._edge_kappa_p:.5f} "
                  f"kappa_contrast={self._edge_kappa_c:.5f} w_floor={self._edge_tv_wfloor}")
```

- [ ] **Step 3d: Swap the R_geo term** — `paired_gs/trainers/joint_trainer.py`, replace lines 775-777:

```python
            if lambda_geo > 0:
                l_geo = self._tv_loss(vol_p) + self._tv_loss(vol_c)
                loss = loss + lambda_geo * l_geo
```

with:

```python
            if lambda_geo > 0:
                if self._edge_tv:
                    l_geo = (edge_preserving_tv(vol_p, self._edge_kappa_p, self._edge_tv_wfloor)
                             + edge_preserving_tv(vol_c, self._edge_kappa_c, self._edge_tv_wfloor))
                else:
                    l_geo = self._tv_loss(vol_p) + self._tv_loss(vol_c)
                loss = loss + lambda_geo * l_geo
```

- [ ] **Step 3e: Add CLI knobs** — `arc/cycle.py`, after line 39 (`ap.add_argument("--kspace_high_plain", ...)`):

```python
ap.add_argument("--edge_tv", action="store_true")               # enable edge-weighted TV (replaces iso-TV)
ap.add_argument("--edge_tv_kappa_q", type=float, default=0.75)  # in-brain gradient quantile -> kappa
ap.add_argument("--edge_tv_wfloor", type=float, default=0.1)    # residual edge smoothing floor
```

- [ ] **Step 3f: Wire CLI into cfg** — `arc/cycle.py`, in `build_cfg`, after line 142 (`cfg["loss"]["kspace_high_weight"] = args.kspace_high`):

```python
    if args.edge_tv:
        cfg["loss"]["edge_tv"] = True
        cfg["loss"]["edge_tv_kappa_q"] = args.edge_tv_kappa_q
        cfg["loss"]["edge_tv_wfloor"] = args.edge_tv_wfloor
```

- [ ] **Step 4: Run smoke to verify it passes**

Run: `bash tests/smoke_edge_tv.sh 5`
Expected: PASS — log shows `[edge_tv] kappa_q=0.75 -> kappa_plain=... kappa_contrast=... w_floor=0.1`, run reaches `[iter 2/2]`, script prints `PASS smoke_edge_tv`.

- [ ] **Step 5: Commit**

```bash
git add paired_gs/trainers/joint_trainer.py arc/cycle.py configs/winner_v2_edgeTV.yaml tests/smoke_edge_tv.sh
git commit -m "feat(edge-tv): config-gated edge-TV in JointTrainer + cycle.py CLI + config"
```

---

### Task 3: `scripts/score_edge.py` — edge/ventricle/quintile metrics (tested on baseline npz)

**Files:**
- Create: `scripts/score_edge.py`
- Test: `tests/test_score_edge.py`

**Interfaces:**
- Consumes: any run's `results_arc/<exp>_volumes.npz` (keys `recT1,gtT1,recFL,gtFL`).
- Produces: `score_contrast(rc, gt, name) -> dict` with keys `nmse_whole, ssim, hfen, nmse_edge, hfen_edge, nmse_vent, quintiles`; a TSV row per contrast appended to `results_arc/edge_scores.tsv`.

- [ ] **Step 1: Write the failing test** — `tests/test_score_edge.py`

```python
#!/usr/bin/env python3
"""Validate scripts/score_edge.py on the existing baseline volumes.
Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python tests/test_score_edge.py"""
import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from score_edge import score_contrast

d = np.load(ROOT / "results_arc" / "reg_aligned_4k_volumes.npz")
r = score_contrast(np.abs(d["recT1"]), np.abs(d["gtT1"]), "T1")
for k in ["nmse_whole","ssim","hfen","nmse_edge","hfen_edge","nmse_vent","quintiles"]:
    assert k in r, f"missing key {k}"
assert all(np.isfinite(v) for v in [r["nmse_whole"], r["ssim"], r["hfen"], r["nmse_edge"], r["nmse_vent"]])
assert 0.0 < r["ssim"] < 1.0, f"ssim out of range: {r['ssim']}"
assert r["nmse_edge"] > r["nmse_whole"], f"edge NMSE {r['nmse_edge']} should exceed whole {r['nmse_whole']}"
assert r["quintiles"][4] > r["quintiles"][0], "top gradient quintile error should exceed the flattest"
print("PASS test_score_edge:", {k: round(r[k],4) for k in ["nmse_whole","ssim","nmse_edge","nmse_vent"]})
print("ALL SCORE-EDGE TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python tests/test_score_edge.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'score_edge'`

- [ ] **Step 3: Write the script** — `scripts/score_edge.py`

```python
#!/usr/bin/env python3
"""score_edge.py — edge-region / ventricle / gradient-quintile fidelity for a recon run,
from its *_volumes.npz. Implements the strict-bar metrics in
docs/superpowers/specs/2026-07-23-edge-preserving-tv-csf-design.md.

Usage:
  PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/score_edge.py \
    --npz results_arc/<exp>_volumes.npz --tag <exp> [--tsv results_arc/edge_scores.tsv]
"""
import argparse, os
import numpy as np
from scipy.ndimage import sobel, gaussian_laplace, binary_dilation, binary_opening
from skimage.metrics import structural_similarity as ssim


def grad_mag(v):
    return np.sqrt(sobel(v, 0) ** 2 + sobel(v, 1) ** 2 + sobel(v, 2) ** 2)


def nmse(a, g, m):
    return float(np.sum(((a - g) * m) ** 2) / np.sum((g * m) ** 2))


def hfen(a, g, m, sigma=1.5):
    la, lg = gaussian_laplace(a, sigma), gaussian_laplace(g, sigma)
    return float(np.sqrt(np.sum(((la - lg) * m) ** 2) / np.sum((lg * m) ** 2)))


def edge_set(gt, brain, q=0.80, dilate=2):
    gm = grad_mag(gt)
    thr = np.quantile(gm[brain], q)
    return binary_dilation((gm >= thr) & brain, iterations=dilate) & brain


def ventricle_mask(gt, zc=160, half=42, zrange=7, pct=24):
    peak = gt.max()
    head = gt[:, :, zc] > 0.02 * peak
    ys, xs = np.where(head); cy, cx = ys.mean(), xs.mean()
    y0, y1, x0, x1 = int(cy - half), int(cy + half), int(cx - half), int(cx + half)
    m = np.zeros_like(gt, dtype=bool)
    for k in range(zc - zrange, zc + zrange + 1):
        box = gt[y0:y1, x0:x1, k]
        m[y0:y1, x0:x1, k] = binary_opening(box < np.percentile(box, pct), iterations=1)
    return m


def quintile_curve(rc, gt, brain):
    ae = (np.abs(rc - gt) / gt.max())[brain]
    gm = grad_mag(gt)[brain]
    e = np.quantile(gm, [0, .2, .4, .6, .8, 1.0])
    return [float(ae[(gm >= e[i]) & (gm <= e[i + 1])].mean()) for i in range(5)]


def score_contrast(rc, gt, name):
    peak = gt.max(); brain = gt > 0.05 * peak
    es, vent = edge_set(gt, brain), ventricle_mask(gt)
    return dict(
        contrast=name,
        nmse_whole=nmse(rc, gt, brain),
        ssim=float(ssim(gt / peak, rc / peak, data_range=1.0)),
        hfen=hfen(rc, gt, brain),
        nmse_edge=nmse(rc, gt, es),
        hfen_edge=hfen(rc, gt, es),
        nmse_vent=nmse(rc, gt, vent),
        quintiles=quintile_curve(rc, gt, brain),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tsv", default="results_arc/edge_scores.tsv")
    a = ap.parse_args()
    d = np.load(a.npz)
    rows = [score_contrast(np.abs(d["recT1"]), np.abs(d["gtT1"]), "T1"),
            score_contrast(np.abs(d["recFL"]), np.abs(d["gtFL"]), "FLAIR")]
    for r in rows:
        print(f"[{a.tag}] {r['contrast']}: nmse_edge={r['nmse_edge']:.4f} hfen_edge={r['hfen_edge']:.4f} "
              f"ssim={r['ssim']:.4f} nmse_whole={r['nmse_whole']:.4f} nmse_vent={r['nmse_vent']:.4f} "
              f"quintiles={['%.4f' % q for q in r['quintiles']]}")
    new = not os.path.exists(a.tsv)
    with open(a.tsv, "a") as f:
        if new:
            f.write("tag\tcontrast\tnmse_edge\thfen_edge\tssim\tnmse_whole\thfen\tnmse_vent\tq1\tq2\tq3\tq4\tq5\n")
        for r in rows:
            f.write(f"{a.tag}\t{r['contrast']}\t{r['nmse_edge']:.5f}\t{r['hfen_edge']:.5f}\t{r['ssim']:.5f}\t"
                    f"{r['nmse_whole']:.5f}\t{r['hfen']:.5f}\t{r['nmse_vent']:.5f}\t"
                    + "\t".join('%.5f' % q for q in r['quintiles']) + "\n")
    print(f"appended -> {a.tsv}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python tests/test_score_edge.py`
Expected: PASS — ends with `ALL SCORE-EDGE TESTS PASSED` (baseline T1 `nmse_edge`≈0.019, `nmse_whole`≈0.012 from this session's numbers).

- [ ] **Step 5: Commit**

```bash
git add scripts/score_edge.py tests/test_score_edge.py
git commit -m "feat(edge-tv): score_edge.py edge/ventricle/quintile metrics + test"
```

---

### Task 4: P0 noise floor + P1 coarse sweep (execute, aggregate, pick best cell)

**Files:**
- Create: `scripts/run_edge_tv_sweep.sh`
- Produces: `results_arc/<cell>_volumes.npz` + `results_arc/<cell>_score.json` per cell; `results_arc/edge_scores.tsv` rows.

**Interfaces:**
- Consumes: Task 2 (training entry) + Task 3 (`score_edge.py`).

- [ ] **Step 1: Write the driver** — `scripts/run_edge_tv_sweep.sh`

```bash
#!/usr/bin/env bash
# P0 noise floor (baseline re-run) + P1 coarse kappa_q x lambda sweep @1500 iters.
# Usage: bash scripts/run_edge_tv_sweep.sh <gpu>
set -euo pipefail
GPU="${1:-5}"
cd "$(dirname "$0")/.."
PY="/home/congcongliu/anaconda3/envs/GSTMI/bin/python"
ENV="CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONPATH="
run () {  # $1=exp $2=extra-args
  eval "$ENV $PY arc/cycle.py --exp $1 --gpu $GPU --iters 1500 \
    --lambda_geo 0.06 --kspace_high 2.2 --warmup 500 --max_radius 20 \
    --base_cfg configs/winner_v2_aligned.yaml --save_volumes $2"
  $PY scripts/score_edge.py --npz results_arc/$1_volumes.npz --tag $1
}
# P0: baseline re-run (isotropic TV) for the noise floor — same base_cfg, no edge_tv
run p0_noise ""
# P1: 9-cell edge-TV sweep (kappa_q x lambda), w_floor=0.1
for KQ in 0.60 0.75 0.85; do
  for LAM in 0.06 0.10 0.15; do
    run "etv_kq${KQ}_l${LAM}" "--edge_tv --edge_tv_kappa_q ${KQ} --edge_tv_wfloor 0.1 --lambda_geo ${LAM} --base_cfg configs/winner_v2_edgeTV.yaml"
  done
done
echo "SWEEP DONE -> results_arc/edge_scores.tsv"
```

Note (`run`): the second `--base_cfg`/`--lambda_geo` in `$2` intentionally overrides the earlier ones on the same command line (argparse keeps the last), so P1 cells use `winner_v2_edgeTV.yaml` and the swept `λ`.

- [ ] **Step 2: Launch the sweep (background)**

Run: `bash scripts/run_edge_tv_sweep.sh 5 > results_arc/edge_sweep.log 2>&1 &`
Expected: ~10 runs × ~17 min ≈ 3 h; each appends 2 rows to `results_arc/edge_scores.tsv`. Monitor for `Traceback|Error|Killed|OOM` and `SWEEP DONE`.

- [ ] **Step 3: Pick the best cell**

Run:
```bash
PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python - <<'PY'
import csv, collections
rows=list(csv.DictReader(open("results_arc/edge_scores.tsv"), delimiter="\t"))
by=collections.defaultdict(dict)
for r in rows: by[r["tag"]][r["contrast"]]=r
base=by["p0_noise"]
b_edge=(float(base["T1"]["nmse_edge"])+float(base["FLAIR"]["nmse_edge"]))/2
b_ssim=(float(base["T1"]["ssim"])+float(base["FLAIR"]["ssim"]))/2
print(f"baseline mean nmse_edge={b_edge:.4f} ssim={b_ssim:.4f}")
best=None
for tag,d in by.items():
    if not tag.startswith("etv_"): continue
    e=(float(d["T1"]["nmse_edge"])+float(d["FLAIR"]["nmse_edge"]))/2
    s=(float(d["T1"]["ssim"])+float(d["FLAIR"]["ssim"]))/2
    ok = s >= b_ssim - 0.003          # SSIM guard (~noise band; refine after P0 gives the real floor)
    print(f"{tag}: nmse_edge={e:.4f} ssim={s:.4f} edge_gain={b_edge-e:+.4f} ssim_ok={ok}")
    if ok and (best is None or e<best[1]): best=(tag,e,s)
print("BEST CELL:", best)
PY
```
Expected: prints per-cell `nmse_edge`/`ssim`/gain; names the best edge-TV cell that lowers mean `nmse_edge` while holding SSIM. Record the winning `κ_q`/`λ`.

- [ ] **Step 4: Commit the sweep artifacts**

```bash
git add scripts/run_edge_tv_sweep.sh
git commit -m "exp(edge-tv): P0 noise floor + P1 coarse kappa_q x lambda sweep driver @1500 iters"
```
(Note: `results_arc/*` is gitignored — the TSV/npz/logs are bring-your-own artifacts, never committed; only the driver script is tracked.)

---

### Task 5: P2 full-budget best cell + strict-bar decision + figure + memory

**Files:**
- Create: `scripts/fig_edge_tv_verdict.py`
- Produces: `results_arc/edgeTV_best_volumes.npz`, `results_arc/edgeTV_verdict.png`, `memory.md` update.

**Interfaces:**
- Consumes: best `κ_q`/`λ` from Task 4; `score_edge.py`; baseline `reg_aligned_4k_volumes.npz`.

- [ ] **Step 1: Run the best cell at full budget (4000 iters)**

Run (substitute `<KQ>`/`<LAM>` from Task 4 Step 3):
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONPATH= \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python arc/cycle.py \
  --exp edgeTV_best --gpu 5 --iters 4000 --warmup 500 --max_radius 20 \
  --kspace_high 2.2 --edge_tv --edge_tv_kappa_q <KQ> --edge_tv_wfloor 0.1 \
  --lambda_geo <LAM> --base_cfg configs/winner_v2_edgeTV.yaml --save_volumes
PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python \
  scripts/score_edge.py --npz results_arc/edgeTV_best_volumes.npz --tag edgeTV_best
PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python \
  scripts/score_edge.py --npz results_arc/reg_aligned_4k_volumes.npz --tag baseline_4k
```
Expected: `edgeTV_best` completes (~44 min); two more TSV rows.

- [ ] **Step 2: Apply the strict bar + build the verdict figure** — `scripts/fig_edge_tv_verdict.py`

```python
#!/usr/bin/env python3
"""Strict-bar verdict + figure for edge-TV vs baseline. Uses the P0 re-run as the noise
floor. Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/fig_edge_tv_verdict.py"""
import csv, collections, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import sobel

R = "results_arc"
rows = list(csv.DictReader(open(f"{R}/edge_scores.tsv"), delimiter="\t"))
by = collections.defaultdict(dict)
for r in rows: by[r["tag"]][r["contrast"]] = r

def mean(tag, key): return (float(by[tag]["T1"][key]) + float(by[tag]["FLAIR"][key])) / 2
# noise floor = |baseline_4k - p0_noise| on each metric (two isotropic-TV runs, only seed-nondeterminism differs)
floor = {k: abs(mean("baseline_4k", k) - mean("p0_noise", k)) for k in ["nmse_edge", "ssim", "nmse_whole"]}
d_edge = mean("baseline_4k", "nmse_edge") - mean("edgeTV_best", "nmse_edge")   # + = improvement
d_ssim = mean("edgeTV_best", "ssim") - mean("baseline_4k", "ssim")            # + = improvement
d_nmse = mean("edgeTV_best", "nmse_whole") - mean("baseline_4k", "nmse_whole")  # + = worse
PASS = (d_edge > floor["nmse_edge"]) and (d_ssim >= -floor["ssim"]) and (d_nmse <= floor["nmse_whole"])
verdict = "PASS" if PASS else ("RESHAPE/FAIL" if d_edge > floor["nmse_edge"] else "NO-GAIN/FAIL")
print(f"noise floor: {floor}")
print(f"edge_gain={d_edge:+.4f} (floor {floor['nmse_edge']:.4f}) | dSSIM={d_ssim:+.4f} | dNMSE_whole={d_nmse:+.4f}")
print(f"VERDICT: {verdict}")

B = np.load(f"{R}/reg_aligned_4k_volumes.npz"); E = np.load(f"{R}/edgeTV_best_volumes.npz")
def gmag(v): return np.sqrt(sobel(v,0)**2+sobel(v,1)**2+sobel(v,2)**2)
AX = lambda v: v[:, :, 160]
fig, ax = plt.subplots(2, 4, figsize=(15, 7.4))
for r, c in enumerate(["T1", "FL"]):
    gt, rb, re = np.abs(B[f"gt{c}"]), np.abs(B[f"rec{c}"]), np.abs(E[f"rec{c}"]); peak = gt.max()
    eb, ee = np.abs(AX(rb)-AX(gt))/peak, np.abs(AX(re)-AX(gt))/peak
    for j,(im,ttl,cm,vm) in enumerate([(AX(gt),"GT","gray",peak),(eb,"baseline |err|","jet",0.1),
                                       (ee,"edge-TV |err|","jet",0.1),(AX(gmag(gt)),"|grad(GT)|","inferno",np.percentile(gmag(gt),99))]):
        h=ax[r,j].imshow(im.T,cmap=cm,vmin=0,vmax=vm,origin="lower"); ax[r,j].set_xticks([]); ax[r,j].set_yticks([])
        if r==0: ax[r,j].set_title(ttl,fontsize=11)
        if cm=="jet": plt.colorbar(h,ax=ax[r,j],fraction=0.046,pad=0.03,ticks=[0,.02,.04,.06,.08,.1])
    ax[r,0].set_ylabel(c,fontsize=11)
fig.suptitle(f"edge-TV vs baseline (butterfly z=160)  VERDICT={verdict}\n"
             f"mean edge-NMSE gain {d_edge:+.4f} (noise {floor['nmse_edge']:.4f}), dSSIM {d_ssim:+.4f}, dNMSE {d_nmse:+.4f}",
             fontsize=12)
plt.tight_layout(rect=[0,0,1,0.92]); plt.savefig(f"{R}/edgeTV_verdict.png", dpi=125, bbox_inches="tight")
print(f"saved {R}/edgeTV_verdict.png")
```

Run: `PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/fig_edge_tv_verdict.py`
Expected: prints noise floor + deltas + `VERDICT: PASS|RESHAPE/FAIL|NO-GAIN/FAIL`; saves `results_arc/edgeTV_verdict.png`.

- [ ] **Step 3: Record the outcome in memory** — append a dated entry to `memory.md` (both PASS and FAIL are results worth freezing): the best `κ_q`/`λ`/`w_floor`, the noise floor, the four numbers (edge-NMSE gain, dSSIM, dNMSE, dHFEN), the VERDICT, and the figure path. If FAIL, state whether it was RESHAPE or NO-GAIN and that Approach B (shared-geometry edge anisotropy) is the gated next step.

- [ ] **Step 4: Commit**

```bash
git add scripts/fig_edge_tv_verdict.py memory.md
git commit -m "exp(edge-tv): P2 full-budget verdict vs baseline (strict bar) + figure + memory"
```
(Note: `results_arc/*` is gitignored — `edgeTV_verdict.png` / `edge_scores.tsv` stay as local artifacts; only the figure script + memory are tracked.)

---

## Notes for the executor

- Never run bare `python` — always the GSTMI interpreter with the env prefix (Global Constraints). Bare/base `python` has no torch.
- Training is `deterministic=False` — that is WHY P0 (a second isotropic-TV run) defines the noise floor; a "win" must beat it.
- If Task 5 verdict is RESHAPE or NO-GAIN, STOP — do not tune further within Approach A; the spec gates Approach B (geometry-side) or stopping on this result. Report honestly.
- `edge_tv` defaults off everywhere; the shipped winner (`winner_v2_aligned.yaml`, `winner_v2.yaml`) is untouched.
