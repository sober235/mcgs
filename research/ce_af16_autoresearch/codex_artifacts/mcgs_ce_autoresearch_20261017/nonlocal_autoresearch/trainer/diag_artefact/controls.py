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

Also here: `assert_target_af_mask`, the same DC+rate gate generalised for a control mask with no
shipped instance to rate-match (control A's repair, an AF=2 "mild acceleration" mask); and
`assert_control_is_not_vacuous`, a POST-RUN gate (not a mask gate) catching the failure mode that
made control A's original all-ones run vacuous — hard-DC with (1-m)=0 discards the model output
entirely, so the scored reconstruction comes back numerically identical to ground truth.
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


def assert_target_af_mask(mask3d: np.ndarray, dc_index, target_af: float, tol: float) -> dict:
    """DC + acceleration gate for a control mask with no shipped instance to rate-match.

    `assert_control_mask` above hard-codes the module constant `AF_ABS_TOL=0.05`, which encodes
    spec §5's requirement that control B match the SHIPPED AF=12 mask's own rate. Control A's
    AF=2 "mild acceleration" plane has no shipped counterpart to match — it is a new target
    chosen by the human partner, not a re-draw of an existing rate — so this generalises the same
    two checks (mask@DC==1, net_AF close to a target) with an explicit `tol` the caller documents,
    rather than reusing a tolerance calibrated for a different regime. Raises AssertionError;
    returns the recorded readings on success.
    """
    net_af = float(mask3d.size / max(mask3d.sum(), 1e-9))
    dc0, dc1 = int(dc_index[0]), int(dc_index[1])
    at_dc = float(mask3d[dc0, dc1].mean())
    assert at_dc > 0.999, (
        f"mask @ DC is {at_dc:.3f}, must be 1 — the ACS does not land on the data's k-space DC "
        f"(this is exactly the §8 registration bug)"
    )
    abs_delta = abs(net_af - target_af)
    assert abs_delta < tol, (
        f"net_AF {net_af:.4f} vs target {target_af:.4f} -> |delta| {abs_delta:.4f} exceeds "
        f"the documented tolerance {tol}; this draw is not close enough to the intended "
        f"acceleration to call the control what it claims to be"
    )
    return {"net_af": net_af, "abs_delta_af": abs_delta, "mask_at_dc": at_dc,
            "target_af": target_af, "tol": tol}


def scored_recon_max_abs_diff(rec: np.ndarray, gt: np.ndarray) -> float:
    """Max absolute per-voxel difference between a scored reconstruction and ground truth."""
    if rec.shape != gt.shape:
        raise ValueError(f"shape mismatch {rec.shape} vs {gt.shape}")
    return float(np.max(np.abs(rec.astype(np.float64) - gt.astype(np.float64))))


# Documented reasoning (not tuned to any specific run): the failure this gate exists to catch —
# an all-ones mask under hard-DC (`k_dc = m*y + (1-m)*k_model`) — discards the model entirely, so
# rec==y==gt to float32 rounding. Measured on the real casualty, `ctrlA_lam010_volumes.npz`:
# max|rec-gt|=8.34e-07 at a GT peak of 1.887, i.e. a *relative* max-diff of 4.4e-07 — squarely at
# the float32 noise floor (eps~1.19e-7). NONVACUITY_REL_TOL=1e-4 sits ~2.5 orders of magnitude
# above that floor (comfortable margin against numerical noise) while still being far below the
# residual any genuinely-trained reconstruction leaves (this project's error maps are conventionally
# plotted on a [0, 0.1]-of-peak scale — three orders of magnitude above this floor).
NONVACUITY_REL_TOL = 1e-4


def assert_control_is_not_vacuous(rec: np.ndarray, gt: np.ndarray,
                                   rel_tol: float = NONVACUITY_REL_TOL) -> dict:
    """A control run must have actually measured something: the scored reconstruction must
    differ materially from ground truth. An identity result (as with the all-ones control A run,
    where hard-DC with (1-m)=0 discards the model's output and simply hands back the measured
    data) means the run carries no information about the thing being tested, however clean its
    printed metrics look. Raises AssertionError; returns the recorded readings on success.
    """
    peak = float(np.max(np.abs(gt)))
    diff = scored_recon_max_abs_diff(rec, gt)
    rel = diff / peak if peak > 0 else diff
    assert rel > rel_tol, (
        f"scored recon is numerically identical to GT: max|rec-gt|={diff:.3e} "
        f"({rel:.3e} of the GT peak {peak:.4f}), at or below the non-vacuity floor {rel_tol:.1e}. "
        f"This is the ctrlA all-ones failure mode: hard-DC with (1-m)=0 discards the model's "
        f"output entirely, so the run measured nothing."
    )
    return {"max_abs_diff": diff, "gt_peak": peak, "rel_max_abs_diff": rel}


def assert_differs_from_shipped(mask3d: np.ndarray, shipped3d: np.ndarray,
                                 min_diff_frac: float = 0.05) -> dict:
    """Extra gate (not in the written spec): control B must be a genuinely different mask
    instance, not — by an RNG or path bug — the same arrangement as the shipped mask under a new
    name. Raises AssertionError; returns the recorded reading on success.

    `min_diff_frac=0.05` (5%): two independent draws at this project's ~8.45% sampling density
    are expected to disagree at roughly `2*p*(1-p) ~= 15.5%` of positions (Bernoulli(p) vs an
    independent Bernoulli(p), a little lower in practice since both share an identical forced-ACS
    region); a near-identical/duplicated mask — the failure this gate exists to catch — reads
    near 0%. 5% sits safely below the expected genuine-difference value and safely above the
    failure case it is meant to catch.
    """
    diff_frac = diff_fraction(mask3d, shipped3d)
    assert diff_frac > min_diff_frac, (
        f"mask differs from the shipped mask in only {diff_frac:.4f} of positions "
        f"(need > {min_diff_frac}); control B would be vacuous — it would measure nothing "
        f"while appearing to run fine"
    )
    return {"diff_frac_vs_shipped": diff_frac}
