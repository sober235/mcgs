"""A control mask that is not registered to DC measures the §8 bug instead of the thing asked.

§8: the shipped mask's ACS was misregistered from the data's k-space DC by ~22 px on a PE axis and
the pipeline captured only 27% (T1) of k-space energy; the whole "AF=12 ceiling" was that bug. A
freshly generated vdPoisMex mask puts its ACS at the matrix centre, so it inherits the same offset
unless it is rolled. These assertions are what stop control B from repeating it.

Both spec-section-5 gates in `assert_control_mask` are ABSOLUTE bounds (human ruling on this task:
the spec's `|net_AF - 11.83| < 0.05` governs; a relative-tolerance deviation was proposed and
REJECTED). No REL_AF_TOL ships anywhere in this module.

The last two tests cover a gate the written spec does not have but this task requires: control B is
supposed to differ from the shipped mask ONLY in RNG seed, so `assert_differs_from_shipped` catches
the degenerate case where the "new" mask is (by bug) the same arrangement as the shipped one — a
control that would measure nothing while appearing to run fine.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from diag_artefact.controls import (
    align_like_shipped,
    assert_control_is_not_vacuous,
    assert_control_mask,
    assert_differs_from_shipped,
    assert_target_af_mask,
    derive_roll,
    scored_recon_max_abs_diff,
)


def test_derive_roll_recovers_a_known_shift():
    rng = np.random.default_rng(0)
    base = (rng.random((16, 14, 16)) > 0.9).astype(np.float32)
    base[8, 7, 8] = 1.0
    shift = (2, 3, -1)
    rolled = np.roll(base, shift=shift, axis=(0, 1, 2))
    assert derive_roll(base, rolled) == shift
    print("PASS test_derive_roll_recovers_a_known_shift")


def test_align_like_shipped_stacks_and_rolls():
    m2 = np.zeros((14, 16), np.float32)
    m2[7, 8] = 1.0
    out = align_like_shipped(m2, (16, 14, 16), roll=(0, 1, 0))
    assert out.shape == (16, 14, 16)
    # DEVIATION FROM BRIEF (self-adjudicated, intent-preserving, matches ledger precedent): the
    # brief wrote `out[:, 8, 8]`, which checks axis 0 vs axis 2 — the wrong pairing. The docstring's
    # own words ("broadcast along readout axis 2") and dataset.py::_conform_mask_shape (readout_axis
    # is always 2 in every shipped config) both say axis 2 is the broadcast axis, and the point after
    # roll=(0,1,0) lands at (8, 8, *): `out[8, 8, :]` is all-ones, `out[:, 8, 8]` is a single 1 among
    # zeros. Fixed the index order to match the assertion's own stated intent; implementation untouched.
    assert out[8, 8, :].all(), "the single sampled point must broadcast along readout axis 2"
    print("PASS test_align_like_shipped_stacks_and_rolls")


def test_assert_control_mask_rejects_a_mask_that_misses_dc():
    m = np.zeros((16, 14, 16), np.float32)
    m[..., :] = 0.0
    m[4, 4, :] = 1.0                      # sampled, but not at DC
    with pytest.raises(AssertionError, match="DC"):
        assert_control_mask(m, dc_index=(8, 7), shipped_net_af=11.832)
    print("PASS test_assert_control_mask_rejects_a_mask_that_misses_dc")


def test_assert_control_mask_rejects_a_mismatched_sampling_rate():
    m = np.zeros((16, 14, 16), np.float32)
    m[8, 7, :] = 1.0                      # hits DC, but net AF is nowhere near 11.83
    with pytest.raises(AssertionError, match="net_AF"):
        assert_control_mask(m, dc_index=(8, 7), shipped_net_af=11.832)
    print("PASS test_assert_control_mask_rejects_a_mismatched_sampling_rate")


def test_assert_control_mask_accepts_a_valid_mask_and_returns_the_expected_dict():
    # Happy-path coverage: a gate that spuriously ALWAYS raises would pass every test above
    # (they only exercise the two rejection paths) without this one.
    m = np.zeros((10, 10, 10), np.float32)
    m[5, 5, :] = 1.0                      # sampled at DC, broadcast along readout axis 2
    m[0, 0, :] = 1.0                      # a couple more sampled positions elsewhere
    m[1, 1, :] = 1.0
    net_af = float(m.size / m.sum())      # exact, by construction -> abs_delta_af must be 0.0
    rep = assert_control_mask(m, dc_index=(5, 5), shipped_net_af=net_af)
    assert rep == {"net_af": net_af, "abs_delta_af": 0.0, "mask_at_dc": 1.0}
    print("PASS test_assert_control_mask_accepts_a_valid_mask_and_returns_the_expected_dict")


def test_assert_differs_from_shipped_rejects_an_identical_pattern():
    rng = np.random.default_rng(0)
    m2 = (rng.random((14, 16)) > 0.9).astype(np.float32)
    m3 = np.repeat(m2[:, :, None], 5, axis=2)
    with pytest.raises(AssertionError, match="differ"):
        assert_differs_from_shipped(m3, m3.copy())
    print("PASS test_assert_differs_from_shipped_rejects_an_identical_pattern")


def test_assert_differs_from_shipped_accepts_a_genuinely_different_seed():
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(99)
    a2 = (rng_a.random((14, 16)) > 0.9).astype(np.float32)
    b2 = (rng_b.random((14, 16)) > 0.9).astype(np.float32)
    a3 = np.repeat(a2[:, :, None], 5, axis=2)
    b3 = np.repeat(b2[:, :, None], 5, axis=2)
    rep = assert_differs_from_shipped(a3, b3)
    assert rep["diff_frac_vs_shipped"] > 0.05
    print("PASS test_assert_differs_from_shipped_accepts_a_genuinely_different_seed")


def test_assert_target_af_mask_rejects_a_mask_that_misses_dc():
    # Control A's AF=2 mask has no shipped instance to rate-match, so it uses its own gate
    # (`assert_target_af_mask`) with a caller-supplied target/tol instead of the AF=12-specific
    # `AF_ABS_TOL` module constant. Same DC requirement as `assert_control_mask`.
    m = np.zeros((16, 14, 16), np.float32)
    m[4, 4, :] = 1.0                      # sampled, but not at DC
    with pytest.raises(AssertionError, match="DC"):
        assert_target_af_mask(m, dc_index=(8, 7), target_af=2.0, tol=0.15)
    print("PASS test_assert_target_af_mask_rejects_a_mask_that_misses_dc")


def test_assert_target_af_mask_rejects_outside_the_documented_tolerance():
    m = np.zeros((16, 14, 16), np.float32)
    m[8, 7, :] = 1.0                      # hits DC, but net AF is nowhere near 2.0
    with pytest.raises(AssertionError, match="target"):
        assert_target_af_mask(m, dc_index=(8, 7), target_af=2.0, tol=0.15)
    print("PASS test_assert_target_af_mask_rejects_outside_the_documented_tolerance")


def test_assert_target_af_mask_accepts_within_the_documented_tolerance():
    # half the plane sampled -> net_AF exactly 2.0, well within any sane tolerance
    m = np.zeros((10, 10, 10), np.float32)
    m[5, 5, :] = 1.0                      # sampled at DC
    m[0:4, 0, :] = 1.0                    # 4 more sampled rows -> 5/10 rows sampled overall
    net_af = float(m.size / m.sum())
    rep = assert_target_af_mask(m, dc_index=(5, 5), target_af=net_af, tol=0.05)
    assert rep == {"net_af": net_af, "abs_delta_af": 0.0, "mask_at_dc": 1.0,
                    "target_af": net_af, "tol": 0.05}
    print("PASS test_assert_target_af_mask_accepts_within_the_documented_tolerance")


def test_scored_recon_max_abs_diff_is_the_max_over_all_voxels():
    gt = np.zeros((4, 4, 4), np.float32)
    rec = gt.copy()
    rec[1, 2, 3] = 0.5
    rec[0, 0, 0] = 0.1
    assert scored_recon_max_abs_diff(rec, gt) == pytest.approx(0.5)
    print("PASS test_scored_recon_max_abs_diff_is_the_max_over_all_voxels")


def test_assert_control_is_not_vacuous_rejects_the_ctrlA_all_ones_failure_mode():
    # Reproduces the exact failure this gate exists to catch: with an all-ones mask, hard-DC
    # (k_dc = m*y + (1-m)*k_model) sets (1-m)=0 everywhere, so the scored reconstruction is the
    # measured data back out again -- identical to GT to float32 precision. Measured on the real
    # ctrlA_lam010_volumes.npz: max|rec-gt|=8.34e-07 at a GT peak of 1.887 (rel 4.4e-07).
    rng = np.random.default_rng(0)
    gt = (rng.random((32, 32, 32)).astype(np.float32)) * 1.887
    noise = (rng.random((32, 32, 32)).astype(np.float32) - 0.5) * 2 * 8.34e-07
    rec = gt + noise
    with pytest.raises(AssertionError, match="identical"):
        assert_control_is_not_vacuous(rec, gt)
    print("PASS test_assert_control_is_not_vacuous_rejects_the_ctrlA_all_ones_failure_mode")


def test_assert_control_is_not_vacuous_accepts_a_materially_different_recon():
    rng = np.random.default_rng(0)
    gt = rng.random((32, 32, 32)).astype(np.float32) * 1.887
    rec = gt + rng.normal(scale=0.02, size=gt.shape).astype(np.float32)
    rep = assert_control_is_not_vacuous(rec, gt)
    assert rep["max_abs_diff"] > 1e-4 * rep["gt_peak"]
    print("PASS test_assert_control_is_not_vacuous_accepts_a_materially_different_recon")


SHIPPED_MASK_DIR = "/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/mask"
SHIPPED_UNALIGNED = f"{SHIPPED_MASK_DIR}/mask3d_vdpois_af12_transverse.mat"
SHIPPED_ALIGNED = f"{SHIPPED_MASK_DIR}/mask3d_vdpois_af12_transverse_ALIGNED.mat"


@pytest.mark.skipif(not (os.path.exists(SHIPPED_UNALIGNED) and os.path.exists(SHIPPED_ALIGNED)),
                    reason="shipped AF=12 masks are bring-your-own (not in the repo)")
def test_derive_roll_pins_the_real_shipped_unaligned_to_aligned_pair():
    """`derive_roll` was only ever tested against a synthetic roll it constructed itself (final
    review I7). The whole point of the function is that control masks reproduce the SHIPPED
    alignment exactly rather than re-arguing memory section 8's (2, 24, -6) from prose, so pin it
    against the real pair it is used on.

    The answer is (2, 24, 0), not (2, 24, -6): both shipped masks are exactly uniform along axis 2
    (the fully-sampled readout axis), so ANY axis-2 shift is a provable no-op and the recovered
    roll is the canonical representative of that equivalence class. The assertion below proves it
    both ways -- the derived roll reproduces the aligned mask exactly, and so does the memory
    value.
    """
    from scipy import io as sio

    unal = sio.loadmat(SHIPPED_UNALIGNED)["mask"].astype(np.float32)
    alig = sio.loadmat(SHIPPED_ALIGNED)["mask"].astype(np.float32)
    roll = derive_roll(unal, alig)
    assert roll == (2, 24, 0), roll
    assert np.array_equal(np.roll(unal, shift=roll, axis=(0, 1, 2)), alig)
    assert np.array_equal(np.roll(unal, shift=(2, 24, -6), axis=(0, 1, 2)), alig), \
        "memory section 8's (2, 24, -6) must be equivalent -- axis 2 is uniform"
    print("PASS test_derive_roll_pins_the_real_shipped_unaligned_to_aligned_pair")
