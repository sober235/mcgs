"""Every metric is checked against an input whose answer is known before it is trusted on data.

This is the §14b discipline applied to the tooling itself: the S index ordered runs the way the eye
does while measuring global smoothing rather than staircasing, so "it ranks correctly" is not
evidence that a metric measures the thing it is named after.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from diag_artefact.provenance import match_panel, extract_panel, check_calibration


def test_match_panel_ranks_the_true_source_first():
    rng = np.random.default_rng(0)
    truth = rng.random((64, 64)).astype(np.float32)
    decoy = rng.random((64, 64)).astype(np.float32)
    # the panel is the truth after the render pipeline: normalise then quantise to 8-bit
    panel = np.round(truth / truth.max() * 255).astype(np.float32) / 255.0
    ranked = match_panel(panel, {"truth": truth, "decoy": decoy})
    assert ranked[0][0] == "truth", f"expected truth first, got {ranked}"
    assert ranked[0][1] > 0.99, f"true source correlation too low: {ranked[0][1]:.4f}"
    assert ranked[1][1] < 0.5, f"decoy correlation implausibly high: {ranked[1][1]:.4f}"
    print("PASS test_match_panel_ranks_the_true_source_first")


def test_calibration_stake_on_synthetic_figure():
    """Test that match_panel applies rotation to 3D volumes and check_calibration requires r > 0.95."""
    # Create synthetic test data
    rng = np.random.default_rng(42)

    # Create a 3D volume with a known source at z=160
    # The volume is (256, 220, 256) like the real data
    source_vol = np.zeros((256, 220, 256), dtype=np.float32)
    source_slice = (rng.random((256, 220)) * 0.3).astype(np.float32)  # dark image
    source_vol[:, :, 160] = source_slice

    # The panel is the rot90 of the source slice (because fig_recon_view.py uses rot90)
    panel = np.round(np.rot90(source_slice) / np.rot90(source_slice).max() * 255).astype(np.float32) / 255.0

    # Create decoy volumes with random content
    decoy_vol = np.random.RandomState(43).random((256, 220, 256)).astype(np.float32) * 0.3

    # Test 1: match_panel should rank the source volume first (when given 3D volumes with slice_index)
    # because match_panel applies rot90 to 3D candidates
    candidates = {"source": source_vol, "decoy": decoy_vol}
    ranked = match_panel(panel, candidates, slice_index=160)
    assert ranked[0][0] == "source", f"Expected source first, got {ranked[0][0]}"
    assert ranked[0][1] > 0.95, f"Source correlation too low: {ranked[0][1]:.4f}"
    print(f"  match_panel correctly ranks 3D source (with rot90) first: r={ranked[0][1]:.4f}")

    # Test 2: check_calibration should raise if top correlation is below 0.95
    low_corr_candidates = {
        "bad": np.random.RandomState(44).random((256, 220, 256)).astype(np.float32),
        "worse": np.random.RandomState(45).random((256, 220, 256)).astype(np.float32)
    }
    try:
        check_calibration(panel, low_corr_candidates, slice_index=160)
        raise AssertionError("check_calibration should have raised ValueError for low correlation")
    except ValueError as e:
        assert "below threshold" in str(e), f"Expected threshold error, got: {e}"
        print(f"  check_calibration correctly rejects low correlation: {e}")

    # Test 3: check_calibration should pass when top correlation > 0.95
    good_candidates = {"source": source_vol, "decoy": decoy_vol}
    try:
        r1, r2 = check_calibration(panel, good_candidates, slice_index=160)
        assert r1 > 0.95, f"Expected r1 > 0.95, got {r1:.4f}"
        print(f"  check_calibration correctly passes good correlation: r1={r1:.4f}, r2={r2:.4f}")
    except ValueError as e:
        raise AssertionError(f"check_calibration should pass for good source: {e}")

    print("PASS test_calibration_stake_on_synthetic_figure")


if __name__ == "__main__":
    test_match_panel_ranks_the_true_source_first()
    test_calibration_stake_on_synthetic_figure()
    print("ALL PASS")
