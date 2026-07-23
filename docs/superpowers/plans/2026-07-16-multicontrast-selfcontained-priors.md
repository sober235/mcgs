# Self-contained Multi-contrast Priors (P1 + P3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two self-contained multi-contrast priors — P1 (self-supervised frequency extrapolation) and P3 (cross-contrast local low-rank) — to the Paired-ARC-GS frontier trainer, then run the aligned-mask ablation (baseline / +P1 / +P3 / +P1+P3) against the MC-GS winner, judged on a unified (non-@max) gauge.

**Architecture:** Both priors attach at `PairedARCTrainer.compute_losses`, which already renders both contrasts (`x_p`, `x_c`) each step. P1 splits each contrast's sampled fit-set k-space into Θ (fit) / Λ (held-out mid/high-freq), runs data-consistency on Θ, and adds the *same* DC formula on Λ as a self-supervised extrapolation target (redrawn every K iters, λ warmed up). P3 adds one new vectorized loss term: a nuclear-norm penalty on per-block 2-column Casorati matrices of the two contrasts' magnitudes. No new loss math for P1 (reuses `data_consistency`); P3 is one new pure function.

**Tech Stack:** Python 3, PyTorch (CUDA), the existing `paired_arc_gs` package + `paired_gs` shared transforms/metrics. Tests are standalone scripts (existing project convention — NOT pytest).

## Global Constraints

- **Python interpreter (exact):** `/home/congcongliu/anaconda3/envs/GSTMI/bin/python`, always run with env `PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=` and (for training) `CUDA_VISIBLE_DEVICES` set by the `--gpu` flag. Free GPUs this session: **5, 7**.
- **Repo root (exact):** `/data0/congcong/code/GS/Paired_ARC_GS_general_framework` — all relative paths below are under it. Tests/scripts insert both the root and `root/paired_gs` on `sys.path` (mirror the pattern already in `paired_arc_gs/losses/test_arc_losses.py`).
- **Aligned mask (exact):** `/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/mask/mask3d_vdpois_af12_transverse_ALIGNED.mat` — the frontier harness currently points at the *misaligned* `mask3d_vdpois_af12_transverse.mat`; switching to ALIGNED (Task 4) is the first thing that must change before any experiment.
- **Metric gauge:** report PSNR with `data_range = P99.9(gt_magnitude)` (a fixed per-contrast value, robust to the outlier peak), held constant across all experiments; **always co-report NMSE (gauge-free)** and use NMSE as the arbiter for any cross-contrast statement. **Never** use `data_range=max(gt)` PSNR for cross-contrast comparison. The shared `paired_gs/metrics/metrics.py` `compute_psnr` normalizes by `max(target)` (the @max gauge) — do NOT edit that shared file; add a *local* `psnr_unified` helper in `arc/cycle_arc.py`.
- **Baseline invariance:** with both new weights at 0, training must be byte-identical to today. Enforce by (a) `mask_theta == mask_fit` when P1 off, and (b) computing each new term only when its λ > 0 AND it is in the stage's active set.
- **LLR uses magnitudes** (`|x_p|`, `|x_c|`) to avoid phase coupling.
- **Commits:** the per-task `git commit` steps create LOCAL commits only. Per the user's standing rule, **never `git push`** and never commit outside these listed steps without asking. If the user prefers to review before any commit, batch the commits — do not add unlisted ones.
- **DRY / YAGNI / TDD:** each code task writes its test first, watches it fail, implements minimally, watches it pass, commits. Experiment tasks (7–10) are empirical run-and-record protocols with explicit go/no-go rules, not asserted unit tests — do not fake asserts around GPU training.

## File Structure

- `paired_arc_gs/losses/arc_losses.py` (MODIFY) — add `cross_contrast_llr()` (P3); register `freqextrap`/`llr` weights in `total_loss`. One responsibility: pure loss-term math.
- `paired_arc_gs/losses/test_arc_losses.py` (MODIFY) — append P3 LLR checks in the existing `ok.append((name, bool))` style.
- `paired_arc_gs/train/kspace_split.py` (CREATE) — pure `frequency_holdout_split()` (P1's Θ/Λ split). Isolated so it tests without importing the heavy trainer.
- `paired_arc_gs/train/test_kspace_split.py` (CREATE) — standalone test for the split.
- `paired_arc_gs/train/stages.py` (MODIFY) — add `freqextrap`/`llr` to `STAGE_LOSS_TERMS` for the DC-bearing refinement stages.
- `paired_arc_gs/train/trainer.py` (MODIFY) — build P1 masks (`_build_prior_masks` + redraw), DC-on-Θ, the `freqextrap` and `llr` terms in `compute_losses`, warmup + K-redraw in `train()`.
- `arc/cycle_arc.py` (MODIFY) — ALIGNED mask; CLI knobs; thread weights/schedule into cfg; local `psnr_unified` helper; report `psnr_uni` + held-out test-set k-space NRMSE.
- `arc/ref_mcgs_unified.py` (CREATE) — one-shot: compute the MC-GS winner's unified-gauge PSNR + NMSE from `results_arc/reg_aligned_volumes.npz` (the numbers to beat).

**Task dependency order:** pure functions (1 LLR, 2 split, 3 metric) → harness enablement (4: aligned mask + all CLI knobs + reporting; harmless if the trainer does not yet read them) → consumers (5 wires P3, 6 wires P1; their smoke tests use the knobs Task 4 added) → experiments (7–10).

---

### Task 1: P3 — cross-contrast local low-rank loss (`cross_contrast_llr`)

**Files:**
- Modify: `paired_arc_gs/losses/arc_losses.py` (add one function, after `route_exclusivity`, ~line 138)
- Test: `paired_arc_gs/losses/test_arc_losses.py` (append checks before the final print block, ~line 95)

**Interfaces:**
- Produces: `cross_contrast_llr(x_p, x_c, block=8, stride=4, eps=1e-8) -> torch.Tensor` (real scalar). Consumed by `PairedARCTrainer.compute_losses` (Task 5) and registered under key `"llr"` in `total_loss` (Task 5).

- [ ] **Step 1: Write the failing test.** Append to `paired_arc_gs/losses/test_arc_losses.py` (after the aggregator block, before `print("="*60)`):

```python
# ---- P3 cross-contrast LLR: matches explicit per-block svdvals sum; rank-1 on identical ----
from paired_arc_gs.losses.arc_losses import cross_contrast_llr

def _ref_llr(a, b, blk, st):
    D, H, W = a.shape; tot = a.new_zeros(())
    for i in range(0, D - blk + 1, st):
        for j in range(0, H - blk + 1, st):
            for k in range(0, W - blk + 1, st):
                A = a[i:i+blk, j:j+blk, k:k+blk].reshape(-1)
                B = b[i:i+blk, j:j+blk, k:k+blk].reshape(-1)
                M = torch.stack([A, B], dim=1)            # (blk^3, 2)
                tot = tot + torch.linalg.svdvals(M).sum()
    return tot

ap = torch.rand(8, 8, 8, device=dev)                     # small vol, blk4/st4 -> 8 blocks
bp = torch.rand(8, 8, 8, device=dev)
fast = cross_contrast_llr(ap, bp, block=4, stride=4)
ref = _ref_llr(ap, bp, 4, 4)
ok.append(("P3 LLR matches per-block svd sum", abs(float(fast) - float(ref)) / float(ref) < 2e-3))
# identical contrasts -> every block rank-1 -> nuclear == Frobenius (== sqrt(2)*||A||)
same = cross_contrast_llr(ap, ap, block=4, stride=4)
fro = _ref_llr(ap, ap, 4, 4)
ok.append(("P3 LLR rank-1 on identical", abs(float(same) - float(fro)) / float(fro) < 2e-3))
# gradient flows and is finite
xg = torch.rand(8, 8, 8, device=dev, requires_grad=True)
yg = torch.rand(8, 8, 8, device=dev, requires_grad=True)
cross_contrast_llr(xg, yg, block=4, stride=4).backward()
ok.append(("P3 LLR grad finite", torch.isfinite(xg.grad).all().item() and torch.isfinite(yg.grad).all().item()))
# accepts complex input (uses magnitude)
xc_ = torch.randn(8, 8, 8, dtype=torch.complex64, device=dev)
yc_ = torch.randn(8, 8, 8, dtype=torch.complex64, device=dev)
ok.append(("P3 LLR accepts complex", torch.isfinite(cross_contrast_llr(xc_, yc_, block=4, stride=4)).item()))
```

- [ ] **Step 2: Run the test, verify it fails.**

Run: `cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework && PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python paired_arc_gs/losses/test_arc_losses.py`
Expected: `ImportError: cannot import name 'cross_contrast_llr'`.

- [ ] **Step 3: Implement `cross_contrast_llr`.** Add to `paired_arc_gs/losses/arc_losses.py` after `route_exclusivity`:

```python
# --------------------------------------------------------------------------------------
# P3  Cross-contrast local low-rank (nuclear-norm) prior  (self-contained, design §5)
# --------------------------------------------------------------------------------------
def cross_contrast_llr(x_p: torch.Tensor, x_c: torch.Tensor,
                       block: int = 8, stride: int = 4, eps: float = 1e-8) -> torch.Tensor:
    """L_LLR = Σ_b ||M_b||_*  where M_b ∈ R^{block^3 × 2} stacks the two contrasts'
    magnitudes over sliding block b (design §5).

    For an N×2 matrix the nuclear norm is σ₁+σ₂ = √λ₊ + √λ₋, the sqrt-eigenvalues of the
    2×2 Gram G_b = M_bᵀM_b = [[p, q],[q, r]] with p=Σa², r=Σb², q=Σab (a=|x_p|, b=|x_c| on
    the block). λ± = (p+r)/2 ± √(((p−r)/2)² + q²). The three block sums are STRIDED BOX
    FILTERS (avg_pool3d × block³), so the whole penalty is fully vectorized — no per-block
    loop — and differentiable. Magnitudes are used to avoid phase coupling."""
    import torch.nn.functional as F
    a = x_p.abs() if x_p.is_complex() else x_p
    b = x_c.abs() if x_c.is_complex() else x_c
    n = float(block ** 3)

    def boxsum(v: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(v[None, None], kernel_size=block, stride=stride)[0, 0] * n

    p = boxsum(a * a)
    r = boxsum(b * b)
    q = boxsum(a * b)
    half = 0.5 * (p + r)
    disc = torch.sqrt(torch.clamp((0.5 * (p - r)) ** 2 + q * q, min=0.0) + eps)
    lam_hi = torch.clamp(half + disc, min=0.0)
    lam_lo = torch.clamp(half - disc, min=0.0)
    return (torch.sqrt(lam_hi + eps) + torch.sqrt(lam_lo + eps)).sum()
```

- [ ] **Step 4: Run the test, verify it passes.**

Run: `cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework && PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python paired_arc_gs/losses/test_arc_losses.py`
Expected: last line `ALL_PASS`; four new `[PASS] P3 LLR ...` lines.

- [ ] **Step 5: Commit.**

```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework
git add paired_arc_gs/losses/arc_losses.py paired_arc_gs/losses/test_arc_losses.py
git commit -m "feat(arc-losses): add P3 cross-contrast local low-rank (LLR) prior"
```

---

### Task 2: P1 — frequency-selective Θ/Λ split (`frequency_holdout_split`)

**Files:**
- Create: `paired_arc_gs/train/kspace_split.py`
- Test: `paired_arc_gs/train/test_kspace_split.py`

**Interfaces:**
- Produces: `frequency_holdout_split(mask, phase_axes, r_acs, rho, generator) -> (mask_theta, mask_lambda)`. `mask`: `(D,H,W)` {0,1} float; `phase_axes`: 2-tuple of undersampled axes (readout excluded from radius); `r_acs`: float ACS radius (plane-radius `< r_acs` always in Θ); `rho`: fraction of sampled `r≥r_acs` held out; `generator`: `torch.Generator(device="cpu")`. Returns two tensors, disjoint, union = sampled set. Consumed by `PairedARCTrainer._build_prior_masks` (Task 6).

- [ ] **Step 1: Write the failing test.** Create `paired_arc_gs/train/test_kspace_split.py`:

```python
"""Standalone test for P1 frequency-selective Θ/Λ split."""
import sys
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
import torch
from paired_arc_gs.train.kspace_split import frequency_holdout_split

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
ok = []
D, H, W = 64, 64, 8                      # phase axes 0,1 ; readout axis 2 fully sampled
torch.manual_seed(0)
c0, c1 = D // 2, H // 2
ii = torch.arange(D).view(D, 1) - c0
jj = torch.arange(H).view(1, H) - c1
rad = torch.sqrt((ii.float() ** 2 + jj.float() ** 2))
prob = torch.clamp(1.5 / (1.0 + rad), 0, 1)                # denser near center
plane = (torch.rand(D, H) < prob).float()
plane[rad < 8] = 1.0                                       # full ACS disc
mask = plane[:, :, None].expand(D, H, W).contiguous().to(dev)

gen = torch.Generator(device="cpu").manual_seed(123)
theta, lam = frequency_holdout_split(mask, phase_axes=(0, 1), r_acs=8.0, rho=0.3, generator=gen)

sampled = (mask > 0).float()
r3 = rad[:, :, None].expand(D, H, W).to(dev)
ok.append(("disjoint", float((theta * lam).sum()) == 0.0))
ok.append(("union == sampled", torch.equal((theta + lam), sampled)))
ok.append(("ACS never held out", float(lam[r3 < 8.0].sum()) == 0.0))
ok.append(("lambda is high-freq only", float(((lam > 0) & (r3 < 8.0)).sum()) == 0.0))
hf = sampled * (r3 >= 8.0)
frac = float(lam.sum()) / max(1.0, float(hf.sum()))
ok.append(("held-out fraction ~= rho", 0.2 < frac < 0.4))
gen2 = torch.Generator(device="cpu").manual_seed(123)
_, lam2 = frequency_holdout_split(mask, (0, 1), 8.0, 0.3, gen2)
ok.append(("reproducible with same seed", torch.equal(lam, lam2)))

npass = sum(1 for _, v in ok if v)
for name, v in ok:
    print(f"  [{'PASS' if v else 'FAIL'}] {name}")
print(f"{npass}/{len(ok)} passed")
print("ALL_PASS" if npass == len(ok) else "SOME_FAIL")
```

- [ ] **Step 2: Run the test, verify it fails.**

Run: `cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework && PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python paired_arc_gs/train/test_kspace_split.py`
Expected: `ModuleNotFoundError: No module named 'paired_arc_gs.train.kspace_split'`.

- [ ] **Step 3: Implement the split.** Create `paired_arc_gs/train/kspace_split.py`:

```python
"""P1 self-supervised frequency extrapolation — Θ/Λ k-space split (design §4).

Hold out a ρ fraction of the SAMPLED mid/high-frequency phase-plane locations as Λ
(the extrapolation target); the ACS disc and the remaining sampled locations form Θ
(the fit set). The readout axis is fully sampled and excluded from the radius.
"""
from __future__ import annotations
from typing import Tuple
import torch


def frequency_holdout_split(mask: torch.Tensor, phase_axes: Tuple[int, int],
                            r_acs: float, rho: float,
                            generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    dev, dt = mask.device, mask.dtype
    ax0, ax1 = phase_axes
    n0, n1 = mask.shape[ax0], mask.shape[ax1]
    c0, c1 = n0 // 2, n1 // 2
    s0 = [1, 1, 1]; s0[ax0] = n0
    s1 = [1, 1, 1]; s1[ax1] = n1
    coord0 = (torch.arange(n0) - c0).float().view(s0)
    coord1 = (torch.arange(n1) - c1).float().view(s1)
    r = torch.sqrt(coord0 ** 2 + coord1 ** 2).expand(mask.shape)          # (D,H,W) on CPU
    sampled_cpu = (mask.detach().cpu() > 0)
    highfreq = sampled_cpu & (r >= float(r_acs))
    u = torch.rand(mask.shape, generator=generator)                      # CPU draw (reproducible)
    lam_cpu = highfreq & (u < float(rho))
    mask_lambda = lam_cpu.to(dtype=dt, device=dev)
    mask_theta = (sampled_cpu & (~lam_cpu)).to(dtype=dt, device=dev)
    return mask_theta, mask_lambda
```

- [ ] **Step 4: Run the test, verify it passes.**

Run: `cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework && PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python paired_arc_gs/train/test_kspace_split.py`
Expected: last line `ALL_PASS` (6/6).

- [ ] **Step 5: Commit.**

```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework
git add paired_arc_gs/train/kspace_split.py paired_arc_gs/train/test_kspace_split.py
git commit -m "feat(arc-train): add P1 frequency-selective theta/lambda k-space split"
```

---

### Task 3: Unified-gauge PSNR helper + MC-GS reference numbers

**Files:**
- Modify: `arc/cycle_arc.py` (add `psnr_unified` after `grain`, ~line 100)
- Create: `arc/ref_mcgs_unified.py`

**Interfaces:**
- Produces: `psnr_unified(rec, gt, p=99.9) -> float` (in `cycle_arc.py`), used by `score_head` (Task 4). `rec`, `gt`: real magnitude tensors.

- [ ] **Step 1: Add the helper.** In `arc/cycle_arc.py`, after `grain(...)` (~line 100), add:

```python
def psnr_unified(rec, gt, p=99.9):
    """Unified-gauge PSNR: data_range = P_p percentile of GT magnitude (robust to the
    single-voxel peak that inflates @max-PSNR). rec, gt: real magnitude tensors, same
    normalized scale. NMSE (gauge-free) remains the cross-contrast arbiter."""
    r = rec.detach().cpu().numpy().astype("float64")
    g = gt.detach().cpu().numpy().astype("float64")
    dr = float(np.percentile(g, p))
    mse = float(np.mean((r - g) ** 2))
    return float("inf") if mse == 0 else float(10.0 * np.log10(dr ** 2 / mse))
```

- [ ] **Step 2: Write the reference script.** Create `arc/ref_mcgs_unified.py`:

```python
#!/usr/bin/env python
"""Print the MC-GS winner reg_aligned unified-gauge PSNR + NMSE (the numbers to beat).
Source of truth = results_arc/reg_aligned_volumes.npz (recon/GT magnitudes, both contrasts)."""
import numpy as np
ROOT = "/data0/congcong/code/GS/Paired_ARC_GS_general_framework"
d = np.load(f"{ROOT}/results_arc/reg_aligned_volumes.npz")

def psnr_uni(rec, gt, p=99.9):
    dr = float(np.percentile(gt, p)); mse = float(np.mean((rec - gt) ** 2))
    return 10.0 * np.log10(dr ** 2 / mse)

def nmse(rec, gt):
    return float(((rec - gt) ** 2).sum() / (gt ** 2).sum())

for name, rk, gk in [("T1", "recT1", "gtT1"), ("FLAIR", "recFL", "gtFL")]:
    rec = d[rk].astype("float64"); gt = d[gk].astype("float64")
    print(f"MC-GS winner {name}: PSNR_uni(P99.9)={psnr_uni(rec, gt):.3f} dB  "
          f"NMSE={nmse(rec, gt):.4f}  (@max PSNR={10*np.log10(gt.max()**2/np.mean((rec-gt)**2)):.3f})")
```

- [ ] **Step 3: Run the reference script.**

Run: `cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework && PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python arc/ref_mcgs_unified.py`
Expected: two lines with finite `PSNR_uni` + `NMSE`. Sanity anchors: NMSE ≈ **T1 0.0202 / FLAIR 0.0085**; `@max PSNR` ≈ **T1 38.5 / FLAIR 38.26**. **Record the two `PSNR_uni` values — the Task 7–10 target.**

- [ ] **Step 4: Verify the helper is defined.**

Run: `cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework && grep -q "def psnr_unified" arc/cycle_arc.py && echo "psnr_unified defined"`
Expected: prints `psnr_unified defined`.

- [ ] **Step 5: Commit.**

```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework
git add arc/cycle_arc.py arc/ref_mcgs_unified.py
git commit -m "feat(arc-score): add unified-gauge PSNR helper + MC-GS reference numbers"
```

---

### Task 4: `cycle_arc.py` harness enablement — aligned mask, CLI knobs, unified/NRMSE reporting

**Files:**
- Modify: `arc/cycle_arc.py` — `MSK` → ALIGNED; argparse knobs; thread into `build_cfg`; smoke-warmup override; `psnr_uni` + held-out NRMSE reporting.

**Interfaces:**
- Consumes: `psnr_unified` (Task 3).
- Produces: config keys `loss.{lambda_llr,llr_block,llr_stride,lambda_freqextrap}` and `training.p1.{rho,K,r_acs,warmup_start,warmup_ramp}`; the runnable sweep entrypoint for Tasks 5–10. NOTE: after this task the trainer does **not yet consume** the prior weights (Tasks 5/6 wire them), so a `--lambda_llr`/`--lambda_freqextrap` run here behaves like the aligned baseline — that is expected.

- [ ] **Step 1: Switch to the aligned mask.** In `arc/cycle_arc.py`, change `MSK` (~line 68):

```python
MSK = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/mask/mask3d_vdpois_af12_transverse_ALIGNED.mat"
```

- [ ] **Step 2: Add CLI knobs.** After `--save_volumes` (~line 41), add:

```python
ap.add_argument("--lambda_llr", type=float, default=0.0)             # P3 cross-contrast LLR weight
ap.add_argument("--llr_block", type=int, default=8)
ap.add_argument("--llr_stride", type=int, default=4)
ap.add_argument("--lambda_freqextrap", type=float, default=0.0)      # P1 extrapolation weight
ap.add_argument("--p1_rho", type=float, default=0.3)                 # P1 held-out fraction
ap.add_argument("--p1_K", type=int, default=50)                      # P1 Λ redraw period
ap.add_argument("--p1_r_acs", type=float, default=12.0)              # P1 ACS radius (always in Θ)
```

- [ ] **Step 3: Thread the knobs into `build_cfg`.** Replace the `"loss": {...}` entry in the `merge_config({...})` call (~line 128) with:

```python
        "loss": {"lambda_geo": args.lambda_geo, "lambda_change": args.lambda_change,
                 "lambda_route": args.lambda_route,
                 "lambda_llr": args.lambda_llr, "llr_block": args.llr_block, "llr_stride": args.llr_stride,
                 "lambda_freqextrap": args.lambda_freqextrap},
```

and add a `"p1"` block inside the `"training": {...}` dict (next to `"audit"`, ~line 137) — tiny warmup under `--smoke` so P1 is exercised in 30 iters:

```python
            "p1": {"rho": args.p1_rho, "K": args.p1_K, "r_acs": args.p1_r_acs,
                   "warmup_start": (0 if args.smoke else 500),
                   "warmup_ramp": (5 if args.smoke else 500)},
```

- [ ] **Step 4: Compute unified PSNR in `score_head`.** In `score_head` (~line 158), after `psnr, ssim, nmse = ...`, add:

```python
    psnr_uni = psnr_unified(x_dc.abs(), gt.abs())
```
and add `psnr_uni=psnr_uni` to the returned `dict(...)` (~line 161).

- [ ] **Step 5: Report unified PSNR + held-out k-space NRMSE, and add tsv columns.** In `main()`, after the two `score_head` calls (~line 179), add:

```python
    try:
        ho = tr.holdout_kspace_nrmse()                       # fixed 10% test set, disjoint from Θ and Λ
        ho_t1, ho_fl = ho["plain"]["holdout_nrmse"], ho["contrast"]["holdout_nrmse"]
    except Exception:
        ho_t1 = ho_fl = float("nan")
    log(f"UNIFIED psnr_uni T1={sp['psnr_uni']:.3f} FLAIR={sc['psnr_uni']:.3f} | "
        f"holdout_kspace_nrmse T1={ho_t1:.4f} FLAIR={ho_fl:.4f}")
```

Then append two columns to the tsv (append at the END, after `vram_gb`, to avoid ambiguous mid-string insertion). Change the header string (~line 206) to end with `...\tnpts\tvram_gb\tT1_psnr_uni\tFLAIR_psnr_uni\n` and the line string (~line 211) to end with `...\t{npts}\t{vram:.1f}\t{sp['psnr_uni']:.3f}\t{sc['psnr_uni']:.3f}\n`.

- [ ] **Step 6: Smoke — aligned mask + reporting (baseline behaviour, priors unwired).**

Run:
```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework && \
PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH= \
/home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle_arc.py \
  --exp smoke_harness --gpu 7 --smoke --lambda_llr 1e-2 --lambda_freqextrap 0.5 2>&1 | tail -30
```
Expected: run completes (`DONE`); the mask-load log shows the `_ALIGNED` path; a final `UNIFIED psnr_uni T1=… FLAIR=…` line with finite values + `holdout_kspace_nrmse` in `(0, 1.5)`. **The flags are accepted but produce no `llr=`/`freqextrap=` lines yet** (trainer unwired — Tasks 5/6 add them). No NaN/Inf.

- [ ] **Step 7: Commit.**

```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework
git add arc/cycle_arc.py
git commit -m "feat(arc-cycle): aligned mask + P1/P3 CLI knobs + unified-gauge & holdout-NRMSE reporting"
```

---

### Task 5: Wire P3 (LLR) into the aggregator + stages + trainer

**Files:**
- Modify: `paired_arc_gs/losses/arc_losses.py` (`total_loss` lam dict, ~line 185)
- Modify: `paired_arc_gs/train/stages.py` (`STAGE_LOSS_TERMS`, ~line 65)
- Modify: `paired_arc_gs/train/trainer.py` (`compute_losses`, ~line 276)

**Interfaces:**
- Consumes: `cross_contrast_llr` (Task 1); CLI/config keys from Task 4.
- Produces: loss key `"llr"` (weight `lambda_llr`); also registers `"freqextrap"` weight + stage activation now so Task 6 needs no second edit here. Active in stages s2/s4/s5/s7/s9.

- [ ] **Step 1: Register both weights in `total_loss`.** In `paired_arc_gs/losses/arc_losses.py`, inside `total_loss`'s `lam` dict (after the `"def"` entry, ~line 191), add:

```python
        "llr": weights.get("lambda_llr", 0.0),
        "freqextrap": weights.get("lambda_freqextrap", 0.0),
```

- [ ] **Step 2: Activate both terms in the DC-bearing stages.** In `paired_arc_gs/train/stages.py`, replace the s2/s4/s5/s7/s9 entries of `STAGE_LOSS_TERMS` with:

```python
    "s2_response_fit":  ("dc", "geo", "phase", "llr", "freqextrap"),
    "s3_deformation":   ("dc", "def"),
    "s4_geometry_lowlr":("dc", "geo", "def", "phase", "llr", "freqextrap"),
    "s5_change_birth":  ("dc", "geo", "change_sparse", "route", "def", "phase", "llr", "freqextrap"),
    "s6_gabor_birth":   ("dc", "geo", "change_sparse", "route", "phase"),
    "s7_joint_prune":   ("dc", "geo", "change_sparse", "route", "def", "phase", "llr", "freqextrap"),
    "s8_stability":     ("dc", "change_sparse", "route"),
    "s9_full_refine":   ("dc", "geo", "change_sparse", "route", "phase", "llr", "freqextrap"),
```

- [ ] **Step 3: Add the LLR term + extend the weights dict in `compute_losses`.** In `paired_arc_gs/train/trainer.py`, in `compute_losses` after the `change_sparse` block and before `weights = {...}` (~line 310), add:

```python
        if "llr" in active_terms and float(lcfg.get("lambda_llr", 0.0)) > 0.0:
            # P3 cross-contrast local low-rank on the two rendered magnitudes (design §5)
            terms["llr"] = L.cross_contrast_llr(
                x_p, x_c,
                block=int(lcfg.get("llr_block", 8)), stride=int(lcfg.get("llr_stride", 4)))
```

and replace the `weights = {...}` line (~line 311) with the extended tuple:

```python
        weights = {k: lcfg.get(k) for k in ("lambda_geo", "lambda_change", "lambda_route",
                                            "lambda_phase", "lambda_def",
                                            "lambda_llr", "lambda_freqextrap")}
```

- [ ] **Step 4: Smoke — P3 fires.**

Run:
```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework && \
PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH= \
/home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle_arc.py --exp smoke_llr --gpu 7 --smoke --lambda_llr 1e-2 2>&1 | tail -30
```
Expected: completes; ≥1 `[PairedARC] it… llr=<finite>` line; no NaN/Inf.

- [ ] **Step 5: Baseline invariance — llr off ⇒ no `llr` term.**

Run: same command with `--lambda_llr 0 --exp smoke_llr0`.
Expected: completes; **no** `llr=` token in `it…` lines (zero weight ⇒ term skipped ⇒ not summed).

- [ ] **Step 6: Commit.**

```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework
git add paired_arc_gs/losses/arc_losses.py paired_arc_gs/train/stages.py paired_arc_gs/train/trainer.py
git commit -m "feat(arc-train): wire P3 LLR term into compute_losses + stage schedule"
```

---

### Task 6: Wire P1 (frequency extrapolation) into the trainer

**Files:**
- Modify: `paired_arc_gs/train/trainer.py` — import + `_build_prior_masks`/`_redraw_lambda`; call in `__init__`; DC-on-Θ + `freqextrap` term + warmup scale in `compute_losses`; warmup + K-redraw in `train()`.

**Interfaces:**
- Consumes: `frequency_holdout_split` (Task 2); `data_consistency` (existing); config keys from Task 4; the `freqextrap` weight/stage registered in Task 5.
- Produces: attrs `self.mask_theta_{p,c}`, `self.mask_lambda_{p,c}`, `self.p1_enabled`, `self.p1_lambda_scale`; `freqextrap` loss key populated when active & λ>0.

- [ ] **Step 1: Import + add the mask builder/redraw.** Add the import with the other `paired_arc_gs.train` imports (~line 31):

```python
from paired_arc_gs.train.kspace_split import frequency_holdout_split
```

Add these methods after `_audit_split` (~line 250):

```python
    def _build_prior_masks(self):
        """P1 Θ/Λ split of the FIT set (design §4). When P1 is off, Θ == fit and Λ is None,
        so DC is byte-identical to baseline. The fit pool (self.mask_fit_*) is preserved so
        Λ can be redrawn every K iters from the SAME pool."""
        pcfg = self.cfg["training"].get("p1", {})
        self.p1_enabled = float(self.cfg["loss"].get("lambda_freqextrap", 0.0)) > 0.0
        self.p1_rho = float(pcfg.get("rho", 0.3))
        self.p1_K = int(pcfg.get("K", 50))
        self.p1_r_acs = float(pcfg.get("r_acs", 12.0))
        self.p1_warmup_start = int(pcfg.get("warmup_start", 500))
        self.p1_warmup_ramp = int(pcfg.get("warmup_ramp", 500))
        self.p1_lambda_scale = 1.0
        self._p1_phase_axes = tuple(self.cfg["data"]["plain"].get("phase_axes", [1, 2]))
        self._p1_gen = torch.Generator(device="cpu").manual_seed(int(self.cfg.get("seed", 42)))
        if not self.p1_enabled:
            self.mask_theta_p, self.mask_lambda_p = self.mask_fit_p, None
            self.mask_theta_c, self.mask_lambda_c = self.mask_fit_c, None
            return
        self._redraw_lambda()

    def _redraw_lambda(self):
        """Draw a fresh Λ (mid/high-freq holdout) for each contrast from the fit pool."""
        self.mask_theta_p, self.mask_lambda_p = frequency_holdout_split(
            self.mask_fit_p, self._p1_phase_axes, self.p1_r_acs, self.p1_rho, self._p1_gen)
        self.mask_theta_c, self.mask_lambda_c = frequency_holdout_split(
            self.mask_fit_c, self._p1_phase_axes, self.p1_r_acs, self.p1_rho, self._p1_gen)
```

- [ ] **Step 2: Call it in `__init__`.** After `self._audit_split()` (~line 72), add:

```python
        self._build_prior_masks()
```

- [ ] **Step 3: DC-on-Θ + `freqextrap` term in `compute_losses`.** Replace the DC block (~lines 282-284):

```python
        dc_p = L.data_consistency(x_p, self.csm_p, self.ksp_p, self.mask_fit_p, fft3c, self.Linv_p)
        dc_c = L.data_consistency(x_c, self.csm_c, self.ksp_c, self.mask_fit_c, fft3c, self.Linv_c)
        terms["dc"] = dc_p + dc_c
```

with:

```python
        dc_p = L.data_consistency(x_p, self.csm_p, self.ksp_p, self.mask_theta_p, fft3c, self.Linv_p)
        dc_c = L.data_consistency(x_c, self.csm_c, self.ksp_c, self.mask_theta_c, fft3c, self.Linv_c)
        terms["dc"] = dc_p + dc_c
        if ("freqextrap" in active_terms and self.mask_lambda_p is not None
                and float(lcfg.get("lambda_freqextrap", 0.0)) > 0.0):
            # P1 self-supervised extrapolation: SAME DC formula on the held-out Λ set (design §4)
            fe_p = L.data_consistency(x_p, self.csm_p, self.ksp_p, self.mask_lambda_p, fft3c, self.Linv_p)
            fe_c = L.data_consistency(x_c, self.csm_c, self.ksp_c, self.mask_lambda_c, fft3c, self.Linv_c)
            terms["freqextrap"] = fe_p + fe_c
```

- [ ] **Step 4: Apply the warmup scale to the freqextrap weight.** Still in `compute_losses`, immediately after the `weights = {...}` dict (extended in Task 5 Step 3), add:

```python
        if weights.get("lambda_freqextrap"):
            weights["lambda_freqextrap"] = float(weights["lambda_freqextrap"]) * getattr(self, "p1_lambda_scale", 1.0)
```

- [ ] **Step 5: Warmup + K-redraw in `train()`.** In `train()`, inside `for it in range(total):` right after `stage = self.plan.at(it)` and its `if stage is None: break` (~line 487), add:

```python
            if getattr(self, "p1_enabled", False):
                self.p1_lambda_scale = min(1.0, max(0.0,
                    (it - self.p1_warmup_start) / max(1, self.p1_warmup_ramp)))
                if it > 0 and it % self.p1_K == 0:
                    self._redraw_lambda()
```

- [ ] **Step 6: Smoke — P1 fires, and P1+P3 together.**

Run:
```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework && \
PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH= \
/home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle_arc.py \
  --exp smoke_p1p3 --gpu 7 --smoke --lambda_llr 1e-2 --lambda_freqextrap 0.5 2>&1 | tail -40
```
Expected: completes; `it…` lines show **both** `llr=<finite>` and `freqextrap=<finite>`; final `UNIFIED psnr_uni …` + `holdout_kspace_nrmse …` line present; no NaN/Inf.

- [ ] **Step 7: Baseline invariance — P1 off ⇒ Θ == fit.**

Run:
```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework && \
grep -q "self.mask_theta_p, self.mask_lambda_p = self.mask_fit_p, None" paired_arc_gs/train/trainer.py && \
grep -q "self._build_prior_masks()" paired_arc_gs/train/trainer.py && \
echo "P1-off aliases theta->fit (baseline invariant); builder wired"
```
Expected: prints the confirmation line.

- [ ] **Step 8: Commit.**

```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework
git add paired_arc_gs/train/trainer.py
git commit -m "feat(arc-train): wire P1 self-supervised frequency extrapolation (theta/lambda DC + warmup + redraw)"
```

---

### Task 7: Phase 0 — aligned ARC baseline (empirical protocol)

**Goal:** Establish the frontier ARC baseline on the ALIGNED mask (no priors) — the ablation's reference row — and confirm it beats ZF on both contrasts.

- [ ] **Step 1: Run the aligned baseline at sweep budget.**

```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework && \
PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH= \
/home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle_arc.py \
  --exp p0_baseline_aligned --gpu 7 --budget 0.5 2>&1 | tee /tmp/p0.log | tail -20
```

- [ ] **Step 2: Record + check.** Read `results_arc_frontier/results_arc.tsv` (last row) + the `UNIFIED …` line in `/tmp/p0.log`.
Expected/record: `both_beat_zf=True`; note `T1_psnr_uni`, `FLAIR_psnr_uni`, `T1_nmse`, `FLAIR_nmse`, `holdout_kspace_nrmse`. Sanity: aligned baseline should be **≥** the misaligned frontier numbers in the `cycle_arc.py` docstring (T1 36.95 / FLAIR 37.46 @max). If `both_beat_zf=False`, STOP and debug mask/scoring before continuing.

- [ ] **Step 3: Compare to the MC-GS target.** Put the Task 3 `PSNR_uni` reference and this baseline side by side; this gap is what P1/P3 must close. (No commit — records numbers, not code.)

---

### Task 8: Phase 1 — P1 sweep (empirical protocol)

**Goal:** Does P1 alone improve unified PSNR / NMSE / held-out k-space NRMSE over Phase 0?

- [ ] **Step 1: Sweep `lambda_freqextrap` ∈ {0.1, 0.3, 1.0}** (ρ=0.3, K=50, warmup@500 defaults):

```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework
for LFE in 0.1 0.3 1.0; do
  PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH= \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle_arc.py \
    --exp p1_fe${LFE} --gpu 7 --budget 0.5 --lambda_freqextrap ${LFE} 2>&1 | tail -6
done
```

- [ ] **Step 2: Record + decide.** Compare each `p1_fe*` row to `p0_baseline_aligned`.
Decision rule: keep the `lambda_freqextrap` with the best **NMSE** (arbiter) that does not reduce SSIM below baseline and keeps `both_beat_zf=True`. Confirm P1's direct signal — `holdout_kspace_nrmse` **≤** baseline (extrapolation generalizing to the disjoint fixed test set). Record the winning `λ₁`. If nothing beats baseline NMSE, record P1 as null here and note the design §10 fallback before Task 10.

---

### Task 9: Phase 2 — P3 sweep (empirical protocol)

**Goal:** Does P3 alone improve SSIM/HFEN without washing out contrast-specific structure?

- [ ] **Step 1: Sweep `lambda_llr` ∈ {1e-3, 1e-2, 1e-1}** (block 8, stride 4):

```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework
for LLR in 0.001 0.01 0.1; do
  PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH= \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle_arc.py \
    --exp p3_llr${LLR} --gpu 7 --budget 0.5 --lambda_llr ${LLR} 2>&1 | tail -6
done
```

- [ ] **Step 2: Record + decide.** Compare each `p3_llr*` to baseline on SSIM + HFEN + NMSE.
Decision rule: keep the largest `lambda_llr` that improves SSIM∧HFEN while NMSE does not worsen and `both_beat_zf=True`. Guard against over-smoothing: if the best SSIM comes with a clear NMSE/PSNR_uni drop, prefer the more moderate weight. Record the winning `λ₃`.

---

### Task 10: Phase 3 — P1 + P3 joint + final figure/table (empirical protocol)

**Goal:** Run the full method at the winning `(λ₁, λ₃)`, confirm at full budget, produce the ablation table + final recon figure.

- [ ] **Step 1: Joint run at sweep budget** (substitute the Task 8/9 winners for `<L1>`/`<L3>`):

```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework && \
PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH= \
/home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle_arc.py \
  --exp p3_joint --gpu 7 --budget 0.5 --lambda_freqextrap <L1> --lambda_llr <L3> 2>&1 | tail -8
```

- [ ] **Step 2: Full-budget confirmation of the winner** (`--budget 1.0 --save_volumes`):

```bash
cd /data0/congcong/code/GS/Paired_ARC_GS_general_framework && \
PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH= \
/home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle_arc.py \
  --exp p3_joint_full --gpu 7 --budget 1.0 --lambda_freqextrap <L1> --lambda_llr <L3> --save_volumes 2>&1 | tail -8
```

- [ ] **Step 3: Build the ablation table.** From `results_arc_frontier/results_arc.tsv`: rows = `p0_baseline_aligned` / best `p1_fe*` / best `p3_llr*` / `p3_joint(_full)` / MC-GS reference (Task 3); columns = `PSNR_uni(T1,FLAIR)`, `NMSE(T1,FLAIR)`, `SSIM`, `HFEN`, `holdout_kspace_nrmse`, `both_beat_zf`.
Success check (design §2): (A) `p3_joint_full` unified-gauge PSNR ≥ MC-GS reference OR held-out NRMSE clearly below baseline; (B) SSIM/sharpness ≥ Phase-0 baseline AND ≥ MC-GS reference; both contrasts beat ZF on ssim∧nmse∧hfen.

- [ ] **Step 4: Final recon figure.** Render T1/FLAIR × (axial/coronal/sagittal) × [ZF | GT | Recon | |err| jet 0–0.1] from `results_arc_frontier/p3_joint_full_volumes.npz` (keys `recT1,gtT1,zfT1,recFL,gtFL,zfFL`, all `(256,220,256)` magnitude). Error panels: `jet`, fixed `vmin=0, vmax=0.1` normalized to GT peak (global figure convention). Save to `results_arc_frontier/final_recon_p1p3_T1_FLAIR.png`, then copy it to a fixed directory and report its **full absolute path** (the user views over Remote-SSH and cannot see files the agent only reads).

- [ ] **Step 5: Record results (no auto-commit of results).** Summarize the table + figure path for the user. Results artifacts (`*.npz`, `*.png`, `results_arc_frontier/`) are gitignored — do not commit them. Ask the user before committing any results-summary doc.

---

## Out of scope (this plan)

- **P2 complementary sampling** (design §6) — Stage-2, dataset-level, gated on Phase 1–3; its own spec → plan cycle if warranted.
- **De-shading / combine-gauge redefinition** and any **external/generative prior (MAISI)** — deferred (design §11).

## Self-Review notes

- **Spec coverage:** design §4 (P1) → Tasks 2, 6; §5 (P3) → Tasks 1, 5; §2 gauge + success → Tasks 3, 10; §7 ablation → Tasks 7–10; §3 architecture (attach at `compute_losses`) → Tasks 5, 6; §6 (P2) + §11 non-goals → "Out of scope".
- **Dependency order fixed:** all CLI knobs live in Task 4 (harness), which precedes the P3/P1 wiring tasks whose smoke tests use those knobs. Task 4 explicitly notes flags are inert until Tasks 5/6.
- **Baseline invariance** tested (Task 5 Step 5, Task 6 Step 7) so the Phase-0 ablation row is trustworthy.
- **Type consistency:** `cross_contrast_llr(x_p, x_c, block, stride, eps)`, `frequency_holdout_split(mask, phase_axes, r_acs, rho, generator)`, `psnr_unified(rec, gt, p)` — names/signatures identical across defining and consuming tasks; loss keys `"llr"`/`"freqextrap"` consistent across `arc_losses.total_loss`, `stages.STAGE_LOSS_TERMS`, `trainer.compute_losses`.
- **No placeholders:** every code step shows exact code; every run step shows exact command + expected output. (`<L1>`/`<L3>` in Task 10 are the sweep-determined winners from Tasks 8/9, not knowable at plan-time — the only intentional substitutions.)
