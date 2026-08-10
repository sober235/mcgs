"""M0 — which volume produced LOOK_T1_axial.png.

The figure is 08-03 while the newest npz is 08-01 17:05, so it was re-drawn from an existing
volume by a script that was never written to disk. Identifying that volume is what lets §0's
numbers (selectivity, S, PSNR) attach to the picture the user is pointing at.
"""
from __future__ import annotations

import numpy as np


def extract_panel(png_path, n_panels: int = 4, panel_index: int = 2, keep_rgb: bool = False) -> np.ndarray:
    """Locate and crop the i-th of n dark-background image panels from the figure.

    Uses darkness (pixel value < 0.5) to detect the actual image blocks rather than
    assuming equal widths; this correctly handles MRI figures with inter-panel gutters.
    Returns the panel as a float array in [0,1], or RGB if keep_rgb=True.
    """
    import matplotlib.image as mpimg

    img = mpimg.imread(str(png_path))
    img_rgb = None
    if img.ndim == 3:
        img_rgb = img[..., :3].astype(np.float64)
        if img_rgb.max() > 1.0:
            img_rgb /= 255.0
        img = img_rgb.mean(axis=2)
    else:
        img = img.astype(np.float64)
        if img.max() > 1.0:
            img /= 255.0

    # Detect dark image blocks (MRI = black background)
    dark = img < 0.5
    col_dark_frac = dark.mean(axis=0)
    row_dark_frac = dark.mean(axis=1)

    # Find columns and rows with >30% dark pixels (the image content)
    dark_cols = np.where(col_dark_frac > 0.3)[0]
    dark_rows = np.where(row_dark_frac > 0.3)[0]

    if len(dark_cols) == 0 or len(dark_rows) == 0:
        raise ValueError("Could not detect any dark image blocks in the figure")

    # Identify contiguous runs of dark columns (the n panel blocks)
    col_runs = []
    run_start = dark_cols[0]
    for i in range(1, len(dark_cols)):
        if dark_cols[i] - dark_cols[i-1] > 1:  # gap detected
            col_runs.append((run_start, dark_cols[i-1]))
            run_start = dark_cols[i]
    col_runs.append((run_start, dark_cols[-1]))

    if len(col_runs) != n_panels:
        raise ValueError(
            f"Expected {n_panels} image blocks but found {len(col_runs)} "
            f"(dark column runs: {col_runs})"
        )

    if panel_index >= len(col_runs):
        raise ValueError(f"panel_index {panel_index} out of range [0, {len(col_runs)-1}]")

    c0, c1 = col_runs[panel_index]
    r0, r1 = dark_rows[0], dark_rows[-1]

    if keep_rgb and img_rgb is not None:
        return img_rgb[r0:r1+1, c0:c1+1].astype(np.float32)
    else:
        return img[r0:r1+1, c0:c1+1].astype(np.float32)


def _resize_to(a: np.ndarray, shape) -> np.ndarray:
    from scipy import ndimage

    if a.ndim == 2:
        zoom = (shape[0] / a.shape[0], shape[1] / a.shape[1])
        return ndimage.zoom(a, zoom, order=1)
    elif a.ndim == 3:
        # For RGB or other 3-channel arrays, resize separately per channel
        zoom = (shape[0] / a.shape[0], shape[1] / a.shape[1])
        out = np.zeros((shape[0], shape[1], a.shape[2]), dtype=a.dtype)
        for c in range(a.shape[2]):
            out[:, :, c] = ndimage.zoom(a[:, :, c], zoom, order=1)
        return out
    else:
        raise ValueError(f"_resize_to expects 2D or 3D array, got {a.ndim}D")


def match_panel(panel_gray: np.ndarray, candidates: dict, slice_index: int | None = None) -> list:
    """Rank candidate 2-D arrays by |Pearson r| against the rendered panel.

    `candidates` maps a label to either a 2-D slice or a 3-D volume; when 3-D, `slice_index`
    selects `vol[:, :, slice_index]` (the PE plane, readout axis 2), rotated 90 deg
    to match the rendered orientation.
    """
    ranked = []
    for label, arr in candidates.items():
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 3:
            if slice_index is None:
                raise ValueError("slice_index is required for 3-D candidates")
            a = a[:, :, int(slice_index)]
            # Apply the rotation that the render pipeline uses (fig_recon_view.py)
            a = np.rot90(a)
        a = a / (a.max() + 1e-12)
        p = panel_gray / (panel_gray.max() + 1e-12)
        if a.shape != p.shape:
            a = _resize_to(a, p.shape)
        av, pv = a.ravel() - a.mean(), p.ravel() - p.mean()
        r = float(av @ pv / (np.linalg.norm(av) * np.linalg.norm(pv) + 1e-12))
        ranked.append((label, abs(r)))
    return sorted(ranked, key=lambda t: -t[1])


def check_calibration(panel_gray: np.ndarray, candidates: dict, slice_index: int | None = None) -> tuple:
    """Verify that panel 0 (gt) and panel 1 (zf) correlate with their known sources at r > 0.95.

    Returns (r_gt, r_zf) if both pass, raises ValueError if either fails.
    This is a mandatory gate: if extraction is broken, the ranking is meaningless.
    """
    ranked_all = match_panel(panel_gray, candidates, slice_index=slice_index)
    if len(ranked_all) < 2:
        raise ValueError("Need at least 2 candidates to verify calibration")

    # The top match is the expected source
    best_label, best_r = ranked_all[0]
    if best_r < 0.95:
        raise ValueError(
            f"Calibration failed: best candidate {best_label} correlates at only r={best_r:.4f}, "
            f"below threshold 0.95. Extraction or rotation may be broken."
        )

    return best_r, ranked_all[1][1]


def _render_error_panel(rec_slice: np.ndarray, gt_slice: np.ndarray, gt_volume_max: float | None = None) -> np.ndarray:
    """Render an error map (|rec - gt| normalized to [0, 1]) through jet colormap.

    Uses the project's fixed error-map convention: vmin=0, vmax=0.1 (normalized to GT peak).
    Normalises by gt_volume_max (the peak of the FULL 3D GT volume), not the slice peak,
    to comply with the convention that error maps must NEVER be autoscaled.

    Args:
        rec_slice: 2D reconstruction slice
        gt_slice: 2D ground-truth slice
        gt_volume_max: Peak of the full 3D GT volume. If None, uses slice peak (legacy/unsafe).

    Returns:
        RGB (3-channel) array after rot90 to match rendered orientation.
    """
    import matplotlib as mpl

    # Use volume-level normalization (convention-compliant) not slice-level (which is autoscaling)
    if gt_volume_max is None or gt_volume_max < 1e-12:
        gt_volume_max = np.abs(gt_slice).max()  # fallback, but this is autoscaling

    if gt_volume_max < 1e-12:
        error = np.zeros_like(gt_slice)
    else:
        error = np.abs(rec_slice - gt_slice) / gt_volume_max

    # Clip to [0, 0.1] convention (never autoscale)
    error_clipped = np.clip(error, 0, 0.1) / 0.1  # normalize to [0, 1] for colormap

    # Apply jet colormap
    cmap = mpl.colormaps["jet"]
    error_rgb = cmap(error_clipped)[:, :, :3]  # extract RGB, drop alpha

    # Apply rot90 to match the rendering orientation used in fig_recon_view.py
    error_rgb = np.rot90(error_rgb)

    return error_rgb.astype(np.float32)


def match_error_panel(panel_rgb: np.ndarray, candidates: dict, slice_index: int | None = None) -> list:
    """Rank candidate volumes by |Pearson r| of error-map panels in RGB space.

    Each candidate's error is rendered through jet [0, 0.1] and compared to the panel.
    Normalises by the VOLUME peak of each GT (per-candidate), not per-slice, to comply
    with the convention that error maps must never be autoscaled.

    Candidates dict should map labels to dicts with "rec" and "gt" keys.
    """
    ranked = []
    for label, cand_data in candidates.items():
        if isinstance(cand_data, dict) and "rec" in cand_data and "gt" in cand_data:
            rec_vol = np.asarray(cand_data["rec"], dtype=np.float64)
            gt_vol = np.asarray(cand_data["gt"], dtype=np.float64)
        else:
            raise ValueError(f"Candidate {label} must be a dict with 'rec' and 'gt' keys")

        if slice_index is None:
            raise ValueError("slice_index is required for error-panel matching")

        rec_slice = rec_vol[:, :, int(slice_index)]
        gt_slice = gt_vol[:, :, int(slice_index)]

        # Compute the VOLUME-level GT peak (convention-compliant normalization)
        gt_volume_max = float(np.abs(gt_vol).max())

        # Render error map through jet with volume-level normalization
        error_rgb = _render_error_panel(rec_slice, gt_slice, gt_volume_max=gt_volume_max)

        # Resize if needed
        p = panel_rgb / (panel_rgb.max() + 1e-12)
        if error_rgb.shape != p.shape:
            error_rgb = _resize_to(error_rgb, p.shape)

        # Compute Pearson correlation in 3D RGB space
        e_flat = error_rgb.ravel() - error_rgb.mean()
        p_flat = p.ravel() - p.mean()
        r = float(e_flat @ p_flat / (np.linalg.norm(e_flat) * np.linalg.norm(p_flat) + 1e-12))
        ranked.append((label, abs(r)))

    return sorted(ranked, key=lambda t: -t[1])


def check_error_calibration(cand_label: str, cand_data: dict, slice_index: int | None = None) -> float:
    """Verify that a candidate's error map renders and self-correlates near 1.0.

    This is the error-panel calibration stake: rendering a volume's own error map against
    itself should yield r ≈ 1.0, confirming the rendering pipeline works.
    Uses volume-level GT normalization (convention-compliant).
    """
    if slice_index is None:
        raise ValueError("slice_index is required for error calibration")

    rec_vol = np.asarray(cand_data["rec"], dtype=np.float64)
    gt_vol = np.asarray(cand_data["gt"], dtype=np.float64)

    rec_slice = rec_vol[:, :, int(slice_index)]
    gt_slice = gt_vol[:, :, int(slice_index)]

    # Compute volume-level GT peak
    gt_volume_max = float(np.abs(gt_vol).max())

    # Render error map (twice, to verify bit-stability)
    error_rgb = _render_error_panel(rec_slice, gt_slice, gt_volume_max=gt_volume_max)

    # Self-correlate: render the same thing again and compare
    error_rgb2 = _render_error_panel(rec_slice, gt_slice, gt_volume_max=gt_volume_max)

    e_flat = error_rgb.ravel() - error_rgb.mean()
    e2_flat = error_rgb2.ravel() - error_rgb2.mean()
    r_self = float(e_flat @ e2_flat / (np.linalg.norm(e_flat) * np.linalg.norm(e2_flat) + 1e-12))

    return abs(r_self)
