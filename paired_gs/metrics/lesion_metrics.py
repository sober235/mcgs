"""
Lesion-specific evaluation metrics for contrast-enhanced MRI reconstruction.

Provides:
  - compute_lesion_cnr(image, lesion_mask, bg_mask): CNR at user-specified or auto-detected ROIs
  - compute_lbc(image_pre, image_post, lesion_mask, bg_mask): lesion-to-background contrast change pre vs post
  - compute_enhancement_ratio(image_pre, image_post, eps): per-voxel E_R map
  - compute_fwhm_line_profile(image, p0, p1, num_samples): line profile + FWHM
  - auto_detect_lesion_roi(image_post, image_pre, ...): heuristic ROI detection
    from |post|-|pre| difference
  - compute_lesion_metrics_bundle(image_pre, image_post, **kwargs): top-level eval
    that returns all metrics

Conventions
-----------
- Inputs are real-valued magnitude volumes (3D numpy arrays). Complex inputs are
  converted to magnitude via ``numpy.abs``.
- Volumes are assumed to share a common spatial grid (same shape) for pre/post pairs.
- ROI masks are boolean numpy arrays of identical shape to the image.

References
----------
- ROI auto-detection follows the convention used in ``metrics.py`` and
  ``baselines/aggregate_ce_methods.py::shared_roi``: in-FOV voxels selected via
  intensity quantiles to avoid CSM-zeroed exterior voxels.
- Enhancement ratio: standard DCE-MRI definition E_R = (S_post - S_pre) / S_pre
  (Tofts/Kermode 1991; Padhani 2002).
- FWHM via two-sided half-max crossing on a 1D line profile (standard line-spread
  function characterization for MRI sharpness).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage as ndi

ArrayLike = Union[np.ndarray, "torch.Tensor"]  # noqa: F821


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_magnitude_numpy(data) -> np.ndarray:
    """Convert torch tensor / complex array / real array to float magnitude numpy."""
    try:
        import torch  # local import to keep numpy-only callers light

        if torch.is_tensor(data):
            data = data.detach().cpu()
            if data.is_complex():
                data = torch.abs(data)
            return data.numpy().astype(np.float64, copy=False)
    except ImportError:
        pass
    arr = np.asarray(data)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    return arr.astype(np.float64, copy=False)


def _ensure_same_shape(a: np.ndarray, b: np.ndarray, names=("a", "b")) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch between {names[0]} ({a.shape}) and {names[1]} ({b.shape})."
        )


def _safe_std(values: np.ndarray) -> float:
    """Robust noise std: std, then MAD/0.6745, then mean(|·|), else 0."""
    if values.size == 0:
        return 0.0
    s = float(values.std())
    if s > 1e-12:
        return s
    mad = float(np.median(np.abs(values - np.median(values))))
    if mad > 1e-12:
        return mad / 0.6745
    return float(np.mean(np.abs(values)))


# ---------------------------------------------------------------------------
# Auto ROI detection
# ---------------------------------------------------------------------------


def auto_detect_lesion_roi(
    image_post: ArrayLike,
    image_pre: Optional[ArrayLike] = None,
    diff_quantile: float = 0.95,
    bbox_size: Sequence[int] = (10, 10, 10),
    min_component_voxels: int = 8,
    smooth_sigma: float = 1.0,
) -> Dict[str, object]:
    """Auto-detect a candidate lesion ROI from |post| - |pre| difference map.

    Pipeline:
      1. compute diff = magnitude(post) - magnitude(pre) (clamped at 0)
      2. optional Gaussian smoothing (sigma=smooth_sigma) to suppress noise
      3. threshold at ``diff_quantile`` of all voxels
      4. label connected components, take the largest by voxel count (ignoring
         components below ``min_component_voxels``)
      5. compute centroid; build a small bbox of size ``bbox_size`` centred on it

    If ``image_pre`` is ``None``, uses ``image_post`` itself thresholded at
    ``diff_quantile`` (no subtraction).

    Returns
    -------
    dict with keys:
      ``diff_map`` (np.ndarray) : diff map (smoothed if applicable)
      ``thresh_mask`` (np.ndarray bool) : voxels above threshold
      ``component_mask`` (np.ndarray bool) : largest connected component
      ``bbox_mask`` (np.ndarray bool) : bbox-shaped mask centred on centroid
      ``centroid`` (tuple of int) : voxel index of centroid
      ``threshold`` (float) : intensity threshold
    """
    post = _to_magnitude_numpy(image_post)
    if image_pre is None:
        diff = post
    else:
        pre = _to_magnitude_numpy(image_pre)
        _ensure_same_shape(post, pre, ("post", "pre"))
        diff = np.clip(post - pre, 0.0, None)

    if smooth_sigma and smooth_sigma > 0:
        diff_smooth = ndi.gaussian_filter(diff, sigma=float(smooth_sigma))
    else:
        diff_smooth = diff

    flat = diff_smooth.flatten()
    if not np.any(flat > 0):
        # nothing to detect — fall back to volume centre
        ctr = tuple(s // 2 for s in post.shape)
        bbox_mask = _bbox_mask(post.shape, ctr, bbox_size)
        return dict(
            diff_map=diff_smooth,
            thresh_mask=np.zeros_like(post, dtype=bool),
            component_mask=np.zeros_like(post, dtype=bool),
            bbox_mask=bbox_mask,
            centroid=ctr,
            threshold=0.0,
        )

    threshold = float(np.quantile(flat, diff_quantile))
    thresh_mask = diff_smooth >= threshold
    if not thresh_mask.any():
        ctr = tuple(s // 2 for s in post.shape)
        bbox_mask = _bbox_mask(post.shape, ctr, bbox_size)
        return dict(
            diff_map=diff_smooth,
            thresh_mask=thresh_mask,
            component_mask=np.zeros_like(post, dtype=bool),
            bbox_mask=bbox_mask,
            centroid=ctr,
            threshold=threshold,
        )

    labels, n_comp = ndi.label(thresh_mask)
    if n_comp == 0:
        ctr = tuple(s // 2 for s in post.shape)
    else:
        sizes = ndi.sum(thresh_mask, labels, index=np.arange(1, n_comp + 1))
        sizes[sizes < min_component_voxels] = 0
        if sizes.max() == 0:
            ctr = tuple(s // 2 for s in post.shape)
            component_mask = np.zeros_like(post, dtype=bool)
        else:
            best_label = int(np.argmax(sizes)) + 1
            component_mask = labels == best_label
            cog = ndi.center_of_mass(component_mask)
            ctr = tuple(int(round(c)) for c in cog)

    if "component_mask" not in locals():
        component_mask = np.zeros_like(post, dtype=bool)

    bbox_mask = _bbox_mask(post.shape, ctr, bbox_size)
    return dict(
        diff_map=diff_smooth,
        thresh_mask=thresh_mask,
        component_mask=component_mask,
        bbox_mask=bbox_mask,
        centroid=ctr,
        threshold=threshold,
    )


def _bbox_mask(
    shape: Sequence[int], centre: Sequence[int], size: Sequence[int]
) -> np.ndarray:
    """Build a boolean bbox mask of given size centred on ``centre``, clipped to volume."""
    if len(shape) != len(centre) or len(shape) != len(size):
        raise ValueError("shape, centre, size must have the same length")
    mask = np.zeros(shape, dtype=bool)
    slices = []
    for s, c, sz in zip(shape, centre, size):
        half = max(int(sz) // 2, 1)
        lo = max(int(c) - half, 0)
        hi = min(int(c) + half, int(s))
        slices.append(slice(lo, hi))
    mask[tuple(slices)] = True
    return mask


def _auto_background_mask(
    image: np.ndarray,
    exclude_mask: np.ndarray,
    bg_low_q: float = 0.05,
    bg_high_q: float = 0.20,
    min_voxels: int = 256,
) -> np.ndarray:
    """Background mask = low-quantile band of in-FOV voxels, excluding the lesion ROI.

    Robust to nearly-uniform volumes: if the chosen band yields too few voxels
    (or none after morphological erosion), the band is widened toward the
    median. Erosion is skipped when it would wipe out the mask.
    """
    in_fov = image > 1e-12
    in_fov = in_fov & ~exclude_mask
    vals = image[in_fov]
    if vals.size == 0:
        return np.zeros_like(image, dtype=bool)

    lo = float(np.quantile(vals, bg_low_q))
    hi = float(np.quantile(vals, bg_high_q))
    bg = in_fov & (image >= lo) & (image <= hi)

    # widen band if too sparse (e.g. uniform background phantom)
    if int(bg.sum()) < min_voxels:
        lo = float(np.quantile(vals, max(bg_low_q - 0.05, 0.0)))
        hi = float(np.quantile(vals, min(bg_high_q + 0.30, 0.95)))
        bg = in_fov & (image >= lo) & (image <= hi)

    # erode only if it preserves a useful number of voxels
    if bg.any():
        bg_eroded = ndi.binary_erosion(bg, iterations=1)
        if int(bg_eroded.sum()) >= min_voxels:
            bg = bg_eroded
    return bg


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def compute_lesion_cnr(
    image: ArrayLike,
    lesion_mask: np.ndarray,
    bg_mask: np.ndarray,
) -> float:
    """Lesion CNR = |mean(lesion) - mean(bg)| / std(bg).

    Returns ``nan`` if either mask is empty; ``inf`` if background std is zero.
    """
    img = _to_magnitude_numpy(image)
    if lesion_mask.shape != img.shape or bg_mask.shape != img.shape:
        raise ValueError("lesion_mask and bg_mask must have the same shape as image")
    if not lesion_mask.any() or not bg_mask.any():
        return float("nan")
    sig = float(img[lesion_mask].mean())
    bg = float(img[bg_mask].mean())
    sigma_bg = _safe_std(img[bg_mask])
    if sigma_bg <= 1e-12:
        return float("inf")
    return float(abs(sig - bg) / sigma_bg)


def compute_lesion_snr(
    image: ArrayLike,
    lesion_mask: np.ndarray,
    bg_mask: np.ndarray,
) -> float:
    """SNR_lesion = mean(image[lesion]) / std(image[bg]).

    Linear-domain SNR computed as the ratio of the lesion mean signal to the
    background noise std. Use ``compute_lesion_snr_db`` for the dB form.

    Returns ``nan`` if either mask is empty; ``inf`` if background std is zero.
    """
    img = _to_magnitude_numpy(image)
    if lesion_mask.shape != img.shape or bg_mask.shape != img.shape:
        raise ValueError("lesion_mask and bg_mask must have the same shape as image")
    if not lesion_mask.any() or not bg_mask.any():
        return float("nan")
    sig = float(np.abs(img[lesion_mask]).mean())
    nstd = _safe_std(np.abs(img[bg_mask])) + 1e-12
    if nstd <= 1e-12:
        return float("inf")
    return float(sig / nstd)


def compute_lesion_snr_db(
    image: ArrayLike,
    lesion_mask: np.ndarray,
    bg_mask: np.ndarray,
) -> float:
    """Lesion SNR in dB = 20*log10( mean(|lesion|) / std(|bg|) ).

    Returns ``nan`` if either mask is empty; ``inf`` if SNR is non-finite.
    """
    snr = compute_lesion_snr(image, lesion_mask, bg_mask)
    if not np.isfinite(snr) or snr <= 0:
        return float(snr) if not np.isnan(snr) else float("nan")
    return float(20.0 * np.log10(snr))


def compute_lbc(
    image_pre: ArrayLike,
    image_post: ArrayLike,
    lesion_mask: np.ndarray,
    bg_mask: np.ndarray,
) -> Dict[str, float]:
    """Lesion-to-background contrast pre vs post.

    LBC = (mean(lesion) - mean(bg)) / mean(bg).

    Returns dict with ``lbc_pre``, ``lbc_post``, ``delta_lbc`` (post-pre).
    Negative values are valid (lesion darker than background pre-contrast).
    """
    pre = _to_magnitude_numpy(image_pre)
    post = _to_magnitude_numpy(image_post)
    _ensure_same_shape(pre, post, ("pre", "post"))
    if lesion_mask.shape != pre.shape or bg_mask.shape != pre.shape:
        raise ValueError("masks must match image shape")
    if not lesion_mask.any() or not bg_mask.any():
        return {
            "lbc_pre": float("nan"),
            "lbc_post": float("nan"),
            "delta_lbc": float("nan"),
        }

    def _lbc(x: np.ndarray) -> float:
        s = float(x[lesion_mask].mean())
        b = float(x[bg_mask].mean())
        if abs(b) < 1e-12:
            return float("inf") if abs(s - b) > 0 else 0.0
        return float((s - b) / b)

    lbc_pre = _lbc(pre)
    lbc_post = _lbc(post)
    return {
        "lbc_pre": lbc_pre,
        "lbc_post": lbc_post,
        "delta_lbc": lbc_post - lbc_pre,
    }


def compute_enhancement_ratio(
    image_pre: ArrayLike,
    image_post: ArrayLike,
    eps: float = 1e-6,
    floor_quantile: float = 0.05,
) -> np.ndarray:
    """Per-voxel enhancement ratio E_R = (S_post - S_pre) / S_pre.

    A voxel-wise floor (``floor_quantile`` of nonzero pre values) is added to the
    denominator to avoid divide-by-zero in low-signal regions. Voxels with both
    pre and post below the floor are set to 0.
    """
    pre = _to_magnitude_numpy(image_pre)
    post = _to_magnitude_numpy(image_post)
    _ensure_same_shape(pre, post, ("pre", "post"))

    nz = pre > eps
    if not nz.any():
        return np.zeros_like(pre)
    floor = max(float(np.quantile(pre[nz], floor_quantile)), eps)
    denom = np.where(pre > floor, pre, floor)
    er = (post - pre) / denom
    # zero out very low-signal regions (both pre and post below floor)
    low = (pre < floor) & (post < floor)
    er[low] = 0.0
    return er.astype(np.float32)


def compute_fwhm_line_profile(
    image: ArrayLike,
    p0: Sequence[float],
    p1: Sequence[float],
    num_samples: int = 100,
    order: int = 1,
) -> Dict[str, object]:
    """Sample a 1D line profile along ``p0 -> p1`` and compute FWHM.

    Uses scipy.ndimage.map_coordinates with linear interpolation by default.

    FWHM is computed by:
      1. find the maximum of the profile
      2. compute half-max value = 0.5 * max
      3. find left and right crossings of the half-max via linear interpolation
         between samples
      4. return the distance between crossings (in voxel units along the line)

    Returns dict with ``profile`` (np.ndarray), ``fwhm`` (float, voxels along
    the line), ``fwhm_world`` (float, in voxel-distance units = ``fwhm * step``
    where step is the per-sample distance), ``peak_value``, ``peak_index``.

    If the profile does not cross half-max on both sides, ``fwhm`` is NaN.
    """
    img = _to_magnitude_numpy(image)
    if len(p0) != img.ndim or len(p1) != img.ndim:
        raise ValueError(
            f"p0 and p1 must have ndim={img.ndim} entries (one per axis)"
        )
    p0_arr = np.asarray(p0, dtype=np.float64)
    p1_arr = np.asarray(p1, dtype=np.float64)
    t = np.linspace(0.0, 1.0, int(num_samples))
    coords = p0_arr[:, None] * (1 - t)[None, :] + p1_arr[:, None] * t[None, :]
    # map_coordinates expects shape (ndim, N)
    profile = ndi.map_coordinates(img, coords, order=int(order), mode="nearest")
    profile = np.asarray(profile, dtype=np.float64)

    if profile.size == 0:
        return dict(
            profile=profile,
            fwhm=float("nan"),
            fwhm_world=float("nan"),
            peak_value=float("nan"),
            peak_index=-1,
        )

    peak_idx = int(np.argmax(profile))
    peak_val = float(profile[peak_idx])
    if peak_val <= 0:
        return dict(
            profile=profile,
            fwhm=float("nan"),
            fwhm_world=float("nan"),
            peak_value=peak_val,
            peak_index=peak_idx,
        )
    half = 0.5 * peak_val

    # left crossing
    left = float("nan")
    for i in range(peak_idx, 0, -1):
        if profile[i] >= half >= profile[i - 1]:
            # linear interp between i-1 and i
            denom = profile[i] - profile[i - 1]
            if abs(denom) < 1e-12:
                left = float(i - 1)
            else:
                frac = (half - profile[i - 1]) / denom
                left = float((i - 1) + frac)
            break
    # right crossing
    right = float("nan")
    for i in range(peak_idx, len(profile) - 1):
        if profile[i] >= half >= profile[i + 1]:
            denom = profile[i + 1] - profile[i]
            if abs(denom) < 1e-12:
                right = float(i)
            else:
                frac = (half - profile[i]) / denom
                right = float(i + frac)
            break

    if not (np.isfinite(left) and np.isfinite(right)):
        fwhm_samples = float("nan")
    else:
        fwhm_samples = right - left

    # convert sample units to voxel-distance units: each sample step covers
    # ||p1-p0|| / (num_samples-1) voxels along the line
    seg_len = float(np.linalg.norm(p1_arr - p0_arr))
    step = seg_len / max(int(num_samples) - 1, 1)
    fwhm_world = fwhm_samples * step if np.isfinite(fwhm_samples) else float("nan")

    return dict(
        profile=profile,
        fwhm=float(fwhm_samples),
        fwhm_world=float(fwhm_world),
        peak_value=peak_val,
        peak_index=peak_idx,
    )


# ---------------------------------------------------------------------------
# Top-level bundle
# ---------------------------------------------------------------------------


def compute_lesion_metrics_bundle(
    image_pre: ArrayLike,
    image_post: ArrayLike,
    lesion_mask: Optional[np.ndarray] = None,
    bg_mask: Optional[np.ndarray] = None,
    diff_quantile: float = 0.95,
    bbox_size: Sequence[int] = (10, 10, 10),
    smooth_sigma: float = 1.0,
    bg_low_q: float = 0.05,
    bg_high_q: float = 0.20,
    fwhm_line: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    fwhm_num_samples: int = 100,
    return_arrays: bool = True,
) -> Dict[str, object]:
    """Top-level bundle of lesion-specific metrics.

    Parameters
    ----------
    image_pre, image_post : array-like
        Magnitude (or complex; will be magnitude'd) volumes of identical shape.
    lesion_mask, bg_mask : np.ndarray bool, optional
        Pre-supplied ROI masks. If ``lesion_mask`` is ``None`` it is auto-detected
        from |post| - |pre|. If ``bg_mask`` is ``None`` it is built from a
        low-intensity quantile band of in-FOV voxels (excluding the lesion ROI).
    diff_quantile, bbox_size, smooth_sigma : auto-detection params (see
        ``auto_detect_lesion_roi``).
    bg_low_q, bg_high_q : auto-background params.
    fwhm_line : ((p0), (p1)), optional
        Endpoints (in voxel coordinates) for a line profile FWHM computation. If
        ``None``, a default vertical line is drawn through the lesion centroid
        on the dominant axis.
    fwhm_num_samples : number of profile samples.
    return_arrays : if True, include diff_map and E_R in the returned dict.

    Returns
    -------
    dict containing scalar metrics:
        lesion_cnr_pre, lesion_cnr_post, lbc_pre, lbc_post, delta_lbc,
        er_mean, er_max, er_p99, fwhm_main, fwhm_main_world,
        n_lesion_voxels, n_bg_voxels, lesion_centroid (tuple).
    Plus arrays (if return_arrays):
        diff_map (np.ndarray), er_map (np.ndarray), lesion_mask (np.ndarray bool),
        bg_mask (np.ndarray bool), fwhm_profile (np.ndarray).
    """
    pre = _to_magnitude_numpy(image_pre)
    post = _to_magnitude_numpy(image_post)
    _ensure_same_shape(pre, post, ("pre", "post"))

    # ROI auto-detection if needed
    detection: Optional[Dict[str, object]] = None
    if lesion_mask is None:
        detection = auto_detect_lesion_roi(
            post,
            pre,
            diff_quantile=diff_quantile,
            bbox_size=bbox_size,
            smooth_sigma=smooth_sigma,
        )
        # prefer the connected component mask if it has voxels; else bbox
        comp = detection["component_mask"]
        lesion_mask = comp if comp.any() else detection["bbox_mask"]
        centroid = detection["centroid"]
        diff_map = detection["diff_map"]
    else:
        # compute diff for return-array convenience
        diff_map = np.clip(post - pre, 0.0, None)
        if smooth_sigma and smooth_sigma > 0:
            diff_map = ndi.gaussian_filter(diff_map, sigma=float(smooth_sigma))
        # centroid of supplied lesion mask
        if lesion_mask.any():
            cog = ndi.center_of_mass(lesion_mask)
            centroid = tuple(int(round(c)) for c in cog)
        else:
            centroid = tuple(s // 2 for s in pre.shape)

    if bg_mask is None:
        bg_mask = _auto_background_mask(
            post, exclude_mask=lesion_mask, bg_low_q=bg_low_q, bg_high_q=bg_high_q
        )

    n_lesion = int(lesion_mask.sum())
    n_bg = int(bg_mask.sum())

    # CNR pre and post
    cnr_pre = compute_lesion_cnr(pre, lesion_mask, bg_mask)
    cnr_post = compute_lesion_cnr(post, lesion_mask, bg_mask)

    # SNR pre and post (linear and dB)
    snr_pre = compute_lesion_snr(pre, lesion_mask, bg_mask)
    snr_post = compute_lesion_snr(post, lesion_mask, bg_mask)
    snr_pre_db = compute_lesion_snr_db(pre, lesion_mask, bg_mask)
    snr_post_db = compute_lesion_snr_db(post, lesion_mask, bg_mask)

    # LBC
    lbc = compute_lbc(pre, post, lesion_mask, bg_mask)

    # Enhancement ratio map
    er_map = compute_enhancement_ratio(pre, post)
    if lesion_mask.any():
        er_in_lesion = er_map[lesion_mask]
        er_mean = float(er_in_lesion.mean())
        er_max = float(er_in_lesion.max())
        er_p99 = float(np.percentile(er_in_lesion, 99))
    else:
        er_mean = er_max = er_p99 = float("nan")

    # FWHM line profile
    if fwhm_line is None and lesion_mask.any():
        # default: vertical line on axis 0 spanning ±1.5 * bbox half-extent
        c = centroid
        half = max(int(bbox_size[0]) // 2, 1) * 3
        p0 = list(c)
        p1 = list(c)
        p0[0] = max(0, c[0] - half)
        p1[0] = min(post.shape[0] - 1, c[0] + half)
        fwhm_line = (tuple(p0), tuple(p1))

    if fwhm_line is not None:
        fwhm_res = compute_fwhm_line_profile(
            post,
            fwhm_line[0],
            fwhm_line[1],
            num_samples=int(fwhm_num_samples),
        )
    else:
        fwhm_res = dict(
            profile=np.zeros(0),
            fwhm=float("nan"),
            fwhm_world=float("nan"),
            peak_value=float("nan"),
            peak_index=-1,
        )

    out: Dict[str, object] = {
        "lesion_cnr_pre": cnr_pre,
        "lesion_cnr_post": cnr_post,
        "delta_lesion_cnr": (cnr_post - cnr_pre)
        if (np.isfinite(cnr_pre) and np.isfinite(cnr_post))
        else float("nan"),
        "lesion_snr_pre": snr_pre,
        "lesion_snr_post": snr_post,
        "lesion_snr_pre_db": snr_pre_db,
        "lesion_snr_post_db": snr_post_db,
        "delta_lesion_snr_db": (snr_post_db - snr_pre_db)
        if (np.isfinite(snr_pre_db) and np.isfinite(snr_post_db))
        else float("nan"),
        "lbc_pre": lbc["lbc_pre"],
        "lbc_post": lbc["lbc_post"],
        "delta_lbc": lbc["delta_lbc"],
        "er_mean": er_mean,
        "er_max": er_max,
        "er_p99": er_p99,
        "fwhm_main": fwhm_res["fwhm"],
        "fwhm_main_world": fwhm_res["fwhm_world"],
        "fwhm_peak_value": fwhm_res["peak_value"],
        "n_lesion_voxels": n_lesion,
        "n_bg_voxels": n_bg,
        "lesion_centroid": tuple(int(c) for c in centroid),
        "fwhm_line": (
            tuple(int(v) for v in fwhm_line[0]) if fwhm_line is not None else None,
            tuple(int(v) for v in fwhm_line[1]) if fwhm_line is not None else None,
        ),
    }
    if return_arrays:
        out["diff_map"] = diff_map.astype(np.float32)
        out["er_map"] = er_map.astype(np.float32)
        out["lesion_mask"] = np.asarray(lesion_mask, dtype=bool)
        out["bg_mask"] = np.asarray(bg_mask, dtype=bool)
        out["fwhm_profile"] = np.asarray(fwhm_res["profile"], dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Convenience scalar-only filter for JSON serialization
# ---------------------------------------------------------------------------


def lesion_metrics_scalars(bundle: Dict[str, object]) -> Dict[str, object]:
    """Strip arrays from a bundle so it can be JSON-serialised."""
    drop = {"diff_map", "er_map", "lesion_mask", "bg_mask", "fwhm_profile"}
    out: Dict[str, object] = {}
    for k, v in bundle.items():
        if k in drop:
            continue
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out
