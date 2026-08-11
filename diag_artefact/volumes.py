"""Loading the diagnosis volumes and carving the three regions every metric is scoped to.

The regions are not cosmetic. WM is where the patchiness lives and where §14b's selectivity was
calibrated; air is where P4 is decided and is the one region this pipeline gives no suppression to
(ecalib -c 0.0 leaves SoS==1 over the whole FOV, and the GT uses the same conjugate-SENSE combine,
so the GT's air already carries full-amplitude noise).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

CONTRAST_KEYS = {"T1": ("recT1", "gtT1", "zfT1"), "FLAIR": ("recFL", "gtFL", "zfFL")}


def load_volumes(npz_path, contrast: str = "T1") -> dict:
    if contrast not in CONTRAST_KEYS:
        raise ValueError(f"contrast must be one of {sorted(CONTRAST_KEYS)}, got {contrast!r}")
    rec_k, gt_k, zf_k = CONTRAST_KEYS[contrast]
    d = np.load(npz_path)
    missing = [k for k in (rec_k, gt_k, zf_k) if k not in d.files]
    if missing:
        raise KeyError(f"{npz_path} is missing {missing}; has {d.files}")
    return {"rec": d[rec_k], "gt": d[gt_k], "zf": d[zf_k]}


def brain_mask(gt: np.ndarray, rel: float = 0.05) -> np.ndarray:
    """PUBLISHED DEFINITION (`.claude/memory.md` §14b; producer script `rc_figure.py`, the script
    that actually generated the `selectivity` numbers memory §14b quotes — recovered and verified
    2026-08-11 after an earlier prose-only port of this function failed to reproduce them). Simple
    threshold against the volume's own peak, nothing else.

    Do NOT "improve" this with a percentile-of-99.9 threshold, `binary_fill_holes`, or a
    largest-connected-component step — an earlier, more careful-looking version of this function
    did exactly that and it silently diverged from the calibration this project depends on.
    """
    return gt > rel * float(gt.max())


def wm_mask(gt: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """PUBLISHED DEFINITION (`.claude/memory.md` §14b; producer script `rc_figure.py`). STRICT
    (not >=/<=) [P55, P90] band of the in-brain intensity distribution, then a STRONG
    Gaussian-smoothed erosion (`gaussian_filter(mask, sigma=2) > 0.95`) that keeps only deep WM
    interior, discarding near-boundary voxels.

    That strength matters: near-boundary voxels carry disproportionately large gradients, and
    `selectivity`'s high quantile (q=0.999) is dominated by whichever handful of voxels are most
    extreme, so a milder erosion changes the measurement's answer, not just its noise. Do NOT
    "improve" this into `ndimage.grey_opening(size=(2,2,2))` (a much weaker, differently-shaped
    regularization) — that was tried and it reproduced the LOW gradient quantiles but not the high
    ones, which is exactly where `selectivity` is read.
    """
    vals = gt[brain]
    lo, hi = np.percentile(vals, 55), np.percentile(vals, 90)
    m = brain & (gt > lo) & (gt < hi)
    return ndimage.gaussian_filter(m.astype(np.float32), 2.0) > 0.95


def air_mask(gt: np.ndarray, brain: np.ndarray, dilate: int = 8) -> np.ndarray:
    grown = ndimage.binary_dilation(brain, iterations=int(dilate))
    return ~grown
