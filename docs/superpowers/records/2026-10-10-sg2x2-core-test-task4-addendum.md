# Task 4 addendum: controller rulings of 2026-10-10 (read after task-4-brief.md; where the two differ, this file wins)

## A. Coil-map fit check in scripts/sg2x2_core_verify.py (ruling on the Task 1 reviewer's Minor 1)

Nothing yet checks that the exported ESPIRiT maps fit the exported k-space (a transposed, flipped or coil-reordered
map set passes every existing test). Add this check, using the inventory's metric.

Add `import h5py` and `import numpy as np` to the imports, and these definitions above `check_export`:

```python
FIT_MAX = 0.06  # rank-1 residual in the RSS bin [0.35, 0.60) x P99.9; measured 0.021 (jiangyizhou T1), 0.029 (litianyu T1)
FIT_BINS = {"bin_35_60": (0.35, 0.60), "bin_60_150": (0.60, 1.50)}


def rank1_residual(images: torch.Tensor, maps: torch.Tensor) -> dict:
    """Relative misfit of coil images (coil, x, y, z) to coil maps of the same shape: sum |I_c - S_c x|^2 / sum |I_c|^2
    with x = sum conj(S_c) I_c / sum |S_c|^2, summed over the voxels of each bin of the RSS magnitude relative to its
    99.9th percentile (NaN for an empty bin)."""
    sos = torch.sum(torch.abs(maps) ** 2, dim=0).clamp_min(1e-12)
    x = torch.sum(torch.conj(maps) * images, dim=0) / sos
    err = torch.sum(torch.abs(images - maps * x) ** 2, dim=0)
    energy = torch.sum(torch.abs(images) ** 2, dim=0)
    rss = torch.sqrt(energy)
    p = float(np.quantile(rss.reshape(-1).numpy(), 0.999))
    out = {}
    for name, (lo, hi) in FIT_BINS.items():
        sel = (rss >= lo * p) & (rss < hi * p)
        den = float(energy[sel].sum())
        out[name] = float(err[sel].sum()) / den if den > 0 else float("nan")
    return out


def fit_residual(subject: str) -> dict:
    p = subject_paths(subject)
    out = {}
    for c in ("T1", "FLAIR"):
        with h5py.File(p["ksp"][c], "r") as f:
            k = torch.from_numpy(f["kspace"][()])
        with h5py.File(p["csm"][c], "r") as f:
            s = torch.from_numpy(f["map"][()])
        out[c] = rank1_residual(ifft3c(k), s)
    return out
```

In `main()`, per subject: `fit = fit_residual(s)`; store it as `res[s]["fit"] = fit`; extend `good` with
`and all(fit[c]["bin_35_60"] <= FIT_MAX for c in ("T1", "FLAIR"))` (a NaN fails, as intended); add a column
`fit T1 / FLAIR` to the header and to each printed row, formatted `{fit['T1']['bin_35_60']:.4f} / {fit['FLAIR']['bin_35_60']:.4f}`.
Mention the fit check in the module docstring (one sentence).

New test file `tests/test_sg2x2_core_verify.py` (write it first and see it fail on the missing names, as in the brief's
TDD order):

```python
"""Fit check of scripts/sg2x2_core_verify.py: on a noise-free phantom with RO != PE != SPE, aligned coil maps give a
zero residual; maps flipped along any spatial axis, or with the coil order reversed, exceed FIT_MAX.
Run: env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= $PY -m pytest tests/test_sg2x2_core_verify.py -q -p no:cacheprovider
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sg2x2_core_verify import FIT_MAX, rank1_residual  # noqa: E402

SHAPE = (24, 20, 16)  # RO != PE != SPE, so a transposed or flipped map set cannot match by symmetry


def _phantom():
    X, Y, Z = torch.meshgrid(*[torch.linspace(-1, 1, n) for n in SHAPE], indexing="ij")
    inside = ((X - 0.15) / 0.7) ** 2 + ((Y + 0.1) / 0.6) ** 2 + ((Z - 0.05) / 0.65) ** 2 <= 1
    x = torch.where(inside, 0.3 + 0.35 * (X + 1), torch.zeros_like(X)).to(torch.complex64)  # ramp: both bins populated
    centres = [(0.9, 0.2, -0.5), (-0.7, 0.8, 0.3), (0.1, -0.9, 0.7), (-0.4, -0.3, -0.9)]
    prof = [torch.exp(-((X - a) ** 2 + (Y - b) ** 2 + (Z - c) ** 2) / 0.8) * torch.exp(1j * (0.7 * i + 0.9 * X - 0.4 * Y + 0.3 * i * Z))
            for i, (a, b, c) in enumerate(centres)]
    maps = torch.stack(prof).to(torch.complex64)
    maps = maps / torch.sqrt(torch.sum(torch.abs(maps) ** 2, dim=0))
    return maps * x, maps


def test_aligned_maps_fit():
    images, maps = _phantom()
    r = rank1_residual(images, maps)
    assert r["bin_35_60"] < 1e-6 and r["bin_60_150"] < 1e-6


@pytest.mark.parametrize("dim", [0, 1, 2, 3])  # 0 = coil order reversed; 1, 2, 3 = flipped along RO, PE, SPE
def test_misaligned_maps_fail(dim):
    images, maps = _phantom()
    assert rank1_residual(images, maps.flip(dim))["bin_35_60"] > FIT_MAX
```

Report the four misaligned residuals in your report. If any of them is not above FIT_MAX, stop and report it
(NEEDS_CONTEXT) instead of changing FIT_MAX or the phantom.

## B. Main queue: the 84 core lines only (ruling)

The development subject's seeds 43 / 44 and the 1.8M supplement are already run by the dev-seeds queue (by 08:15, 19 of its 24
runs had finished with rc=0; the rest are running or re-queued), so `scripts/sg2x2_queue_main.txt` must not list them again.
In the brief's generator snippet: drop the two statements that append the dev lines, and use these header lines:

```
# sg2x2 core test: seed 42 of the six cells for the 14 subjects (development subject first).
# Lines: [KEY=VAL ...] EXP CELL ITERS SEED
```
Check: `grep -vc '^#' scripts/sg2x2_queue_main.txt` gives 84 (not 108); the duplicate-name check stays empty.

## C. Export logs (ruling)

The export ran as one process per subject, three at a time (OMP_NUM_THREADS=16, nice 10), from a launch script in the
session scratchpad. There is no `export_run_20261010.log`. Read the start / done lines from
`/data2/congcong/data/sg2x2_core_h5/logs/export_<subject>.log`, the final line from `logs/export_all.log`
("ALL EXPORTS FINISHED"), and `logs/export_failed.txt` (present only if a subject failed). Before the verification run,
confirm `logs/export_all.log` holds that line and `logs/export_failed.txt` does not exist; otherwise stop and report.

## D. Launch procedure in the record and in STATUS (ruling)

Write the controller's step 1 as: "once the dev-seeds runner (`scripts/sg2x2_dev_seeds.txt`) has launched its last
line, stop the runner's own bash process (its launched trainers keep running), or let it drain; start the smoke queue
only after that. Never run two queue runners at once: both can claim the same idle card (the card-sharing stall of
2026-10-09)." Steps 2 to 4 as in the brief; step 3 re-arms the stall watcher on `results_arc/sg2x2_queue_main.log`.
