"""Building control-B masks that differ from the shipped one ONLY in RNG seed.

The roll is not hard-coded. It is derived by comparing the shipped unaligned mask against the
shipped ALIGNED one, so whatever transform produced the shipped alignment is reproduced exactly
rather than re-argued from memory's (2,24,-6).

Both spec-section-5 gates are ABSOLUTE bounds (human ruling on this task: the spec's
`|net_AF - 11.83| < 0.05` governs a proposed 1%-relative deviation and REJECTED it). No relative
tolerance ships here.

A third gate not in the written spec is also enforced: control B is supposed to differ from the
shipped mask only in RNG seed, so a regenerated pattern that happens to come out identical to the
shipped one would make control B vacuous — it would measure nothing while appearing to run fine.
`assert_differs_from_shipped` catches that before anything is launched.
"""
from __future__ import annotations

import numpy as np

AF_ABS_TOL = 0.05   # spec section 5, absolute: |net_AF - 11.83| < 0.05


def derive_roll(unaligned: np.ndarray, aligned: np.ndarray) -> tuple:
    """Recover the integer roll taking `unaligned` to `aligned` by circular cross-correlation."""
    if unaligned.shape != aligned.shape:
        raise ValueError(f"shape mismatch {unaligned.shape} vs {aligned.shape}")
    fa = np.fft.fftn(aligned.astype(np.float64))
    fb = np.fft.fftn(unaligned.astype(np.float64))
    xc = np.real(np.fft.ifftn(fa * np.conj(fb)))
    peak = np.unravel_index(int(np.argmax(xc)), xc.shape)
    return tuple(int(p) - s if p > s // 2 else int(p) for p, s in zip(peak, xc.shape))


def align_like_shipped(mask2d: np.ndarray, volume_shape: tuple, roll: tuple) -> np.ndarray:
    """Stack a 2-D PE-plane mask along readout axis 2, then apply `roll`.

    Mirrors dataset.py::_conform_mask_shape — transpose if the 2-D shape does not already match
    (volume_shape[0], volume_shape[1]), then broadcast along the missing axis.
    """
    h, w = mask2d.shape
    if (h, w) != (volume_shape[0], volume_shape[1]):
        mask2d = mask2d.T
    m3 = np.repeat(mask2d[:, :, None], volume_shape[2], axis=2).astype(np.float32)
    return np.roll(m3, shift=roll, axis=(0, 1, 2))


def assert_control_mask(mask3d: np.ndarray, dc_index, shipped_net_af: float) -> dict:
    """Both gates from spec §5. Raises AssertionError; returns the recorded readings on success."""
    net_af = float(mask3d.size / max(mask3d.sum(), 1e-9))
    dc0, dc1 = int(dc_index[0]), int(dc_index[1])
    at_dc = float(mask3d[dc0, dc1].mean())
    assert at_dc > 0.999, (
        f"mask @ DC is {at_dc:.3f}, must be 1 — the ACS does not land on the data's k-space DC "
        f"(this is exactly the §8 registration bug)"
    )
    abs_delta = abs(net_af - shipped_net_af)
    assert abs_delta < AF_ABS_TOL, (
        f"net_AF {net_af:.4f} vs shipped {shipped_net_af:.4f} -> |delta| {abs_delta:.4f} exceeds "
        f"the absolute bar {AF_ABS_TOL} (spec section 5); sampling rates are not comparable"
    )
    return {"net_af": net_af, "abs_delta_af": abs_delta, "mask_at_dc": at_dc}


def diff_fraction(mask3d: np.ndarray, shipped3d: np.ndarray) -> float:
    """Fraction of positions (thresholded at 0.5) where `mask3d` and `shipped3d` disagree."""
    if mask3d.shape != shipped3d.shape:
        raise ValueError(f"shape mismatch {mask3d.shape} vs {shipped3d.shape}")
    a = mask3d > 0.5
    b = shipped3d > 0.5
    return float(np.count_nonzero(a != b)) / a.size


def assert_differs_from_shipped(mask3d: np.ndarray, shipped3d: np.ndarray,
                                 min_diff_frac: float = 0.05) -> dict:
    """Extra gate (not in the written spec): control B must be a genuinely different mask
    instance, not — by an RNG or path bug — the same arrangement as the shipped mask under a new
    name. Raises AssertionError; returns the recorded reading on success."""
    diff_frac = diff_fraction(mask3d, shipped3d)
    assert diff_frac > min_diff_frac, (
        f"mask differs from the shipped mask in only {diff_frac:.4f} of positions "
        f"(need > {min_diff_frac}); control B would be vacuous — it would measure nothing "
        f"while appearing to run fine"
    )
    return {"diff_frac_vs_shipped": diff_frac}
