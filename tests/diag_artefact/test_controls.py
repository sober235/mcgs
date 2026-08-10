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
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from diag_artefact.controls import (
    align_like_shipped,
    assert_control_mask,
    assert_differs_from_shipped,
    derive_roll,
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


if __name__ == "__main__":
    test_derive_roll_recovers_a_known_shift()
    test_align_like_shipped_stacks_and_rolls()
    test_assert_control_mask_rejects_a_mask_that_misses_dc()
    test_assert_control_mask_rejects_a_mismatched_sampling_rate()
    test_assert_differs_from_shipped_rejects_an_identical_pattern()
    test_assert_differs_from_shipped_accepts_a_genuinely_different_seed()
    print("ALL PASS")
