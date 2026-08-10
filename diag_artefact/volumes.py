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


def brain_mask(gt: np.ndarray, rel: float = 0.15) -> np.ndarray:
    thr = rel * float(np.percentile(gt, 99.9))
    m = gt > thr
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def wm_mask(gt: np.ndarray, brain: np.ndarray) -> np.ndarray:
    vals = gt[brain]
    lo, hi = np.percentile(vals, 55), np.percentile(vals, 90)
    m = brain & (gt >= lo) & (gt <= hi)
    return ndimage.grey_opening(m.astype(np.float32), size=(2, 2, 2)) > 0.5


def air_mask(gt: np.ndarray, brain: np.ndarray, dilate: int = 8) -> np.ndarray:
    grown = ndimage.binary_dilation(brain, iterations=int(dilate))
    return ~grown
