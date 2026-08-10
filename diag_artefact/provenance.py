"""M0 — which volume produced LOOK_T1_axial.png.

The figure is 08-03 while the newest npz is 08-01 17:05, so it was re-drawn from an existing
volume by a script that was never written to disk. Identifying that volume is what lets §0's
numbers (selectivity, S, PSNR) attach to the picture the user is pointing at.
"""
from __future__ import annotations

import numpy as np


def extract_panel(png_path, n_panels: int = 4, panel_index: int = 2) -> np.ndarray:
    """Crop the i-th of n equal-width panels out of the figure and return grey in [0,1]."""
    import matplotlib.image as mpimg

    img = mpimg.imread(str(png_path))
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0
    # trim the uniform white margins matplotlib leaves around the axes row
    col_ink = (img < 0.98).any(axis=0)
    row_ink = (img < 0.98).any(axis=1)
    c0, c1 = int(np.argmax(col_ink)), len(col_ink) - int(np.argmax(col_ink[::-1]))
    r0, r1 = int(np.argmax(row_ink)), len(row_ink) - int(np.argmax(row_ink[::-1]))
    img = img[r0:r1, c0:c1]
    w = img.shape[1] // n_panels
    return img[:, panel_index * w:(panel_index + 1) * w]


def _resize_to(a: np.ndarray, shape) -> np.ndarray:
    from scipy import ndimage

    zoom = (shape[0] / a.shape[0], shape[1] / a.shape[1])
    return ndimage.zoom(a, zoom, order=1)


def match_panel(panel_gray: np.ndarray, candidates: dict, slice_index: int | None = None) -> list:
    """Rank candidate 2-D arrays by |Pearson r| against the rendered panel.

    `candidates` maps a label to either a 2-D slice or a 3-D volume; when 3-D, `slice_index`
    selects `vol[:, :, slice_index]` (the PE plane, readout axis 2).
    """
    ranked = []
    for label, arr in candidates.items():
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 3:
            if slice_index is None:
                raise ValueError("slice_index is required for 3-D candidates")
            a = a[:, :, int(slice_index)]
        a = a / (a.max() + 1e-12)
        p = panel_gray / (panel_gray.max() + 1e-12)
        if a.shape != p.shape:
            a = _resize_to(a, p.shape)
        av, pv = a.ravel() - a.mean(), p.ravel() - p.mean()
        r = float(av @ pv / (np.linalg.norm(av) * np.linalg.norm(pv) + 1e-12))
        ranked.append((label, abs(r)))
    return sorted(ranked, key=lambda t: -t[1])
