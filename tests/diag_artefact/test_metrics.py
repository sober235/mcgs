"""Every metric is checked against an input whose answer is known before it is trusted on data.

This is the §14b discipline applied to the tooling itself: the S index ordered runs the way the eye
does while measuring global smoothing rather than staircasing, so "it ranks correctly" is not
evidence that a metric measures the thing it is named after.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from diag_artefact.provenance import (
    match_panel, extract_panel, check_calibration,
    match_error_panel, check_error_calibration, _render_error_panel
)
from diag_artefact.metrics import air_energy_curve, psf_spread, autocorr_profile, ncc, surrogate_band


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


def _mri_like_panel(rng, shape=(80, 100), bg=0.0):
    """A bright rectangular block on a dark background, with the SAME darkness fraction at
    every column (unlike a circular blob, whose chord height -- and so its column-wise darkness
    fraction -- varies with x and would fragment the dark-block run right at its own center).
    Mimics the black-background MRI panels extract_panel's dark-block detector is built to find.
    """
    arr = np.full(shape, bg, dtype=np.float32)
    r0, r1 = int(shape[0] * 0.30), int(shape[0] * 0.70)
    c0, c1 = int(shape[1] * 0.10), int(shape[1] * 0.90)
    arr[r0:r1, c0:c1] = 0.5 + rng.random((r1 - r0, c1 - c0)).astype(np.float32) * 0.4
    return np.clip(arr, 0.0, 1.0)


def test_extract_panel_recovers_synthetic_blocks():
    """extract_panel is where two of the T1-artefact-diagnosis's original defects lived (margin
    trimming that swallowed panel titles, and equal-width splitting that straddled the gutters
    between panels) -- yet it had zero coverage from an actual rendered PNG; every prior test
    hand-built in-memory arrays and called the matchers directly, skipping PNG parsing, the <0.5
    darkness threshold and the contiguous-run block detection entirely. This renders a real
    4-panel figure -- with per-panel titles AND a suptitle, so the title-trimming trap is
    genuinely present -- and checks extract_panel recovers each block's own content, and raises
    (rather than silently returning a wrong crop) when the panel count doesn't match.
    """
    import tempfile
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 4
    panels = [_mri_like_panel(np.random.default_rng(100 + i)) for i in range(n)]
    titles = ["ground truth", "zero-fill", "reconstruction", "error |rec-gt|"]

    with tempfile.TemporaryDirectory() as tmpdir:
        png_path = os.path.join(tmpdir, "synthetic_fig.png")
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        for ax, arr, title in zip(axes, panels, titles):
            ax.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle("SYNTHETIC T1 AXIAL z=160  (title-trimming trap fixture)")
        fig.savefig(png_path, dpi=100)
        plt.close(fig)

        candidates = {f"panel_{j}": panels[j] for j in range(n)}
        for i in range(n):
            extracted = extract_panel(png_path, n_panels=n, panel_index=i)
            ranked = match_panel(extracted, candidates)
            assert ranked[0][0] == f"panel_{i}", f"panel {i}: expected best match panel_{i}, got {ranked}"
            assert ranked[0][1] > 0.9, f"panel {i}: recovered content correlates too weakly: {ranked}"
            if len(ranked) > 1:
                # the panels share a common dark background, so absolute correlations are all
                # high (~0.96-0.99); what must hold is that the TRUE source is clearly ahead.
                assert ranked[1][1] < ranked[0][1] - 0.01, (
                    f"panel {i}: not clearly distinguishable from the next-best match: {ranked}"
                )
            print(f"  panel {i} ({titles[i]}): recovered r={ranked[0][1]:.4f} vs true source")

        # A wrong expected panel count must raise, not silently return a mis-cropped block.
        try:
            extract_panel(png_path, n_panels=5, panel_index=2)
            raise AssertionError("extract_panel should have raised for a wrong panel count")
        except ValueError as e:
            assert "5" in str(e) or "block" in str(e).lower(), f"unexpected error message: {e}"
            print(f"  extract_panel correctly raises on wrong panel count: {e}")

    print("PASS test_extract_panel_recovers_synthetic_blocks")


def test_error_panel_ranking():
    """Error-panel ranking must discriminate between a good and a poor reconstruction, and the
    ranker must be using the VOLUME-WIDE GT peak for normalisation, not the per-slice peak that
    fix round 3 retired as an autoscaling / convention violation.

    The GT volume's peak is deliberately placed on slice 50 -- far from the slice-160 content
    everything else lives on -- and made materially larger than slice 160's own peak, so the two
    normalisations diverge and this test can actually tell them apart. (In the original fixture
    only slice 160 was populated, so the volume-wide and per-slice peaks were numerically
    identical and this test could not have caught fix round 3's bug even by design.)
    """
    rng = np.random.default_rng(123)

    gt_vol = np.zeros((256, 220, 256), dtype=np.float32)
    gt_slice = (rng.random((256, 220)) * 0.5 + 0.2).astype(np.float32)  # brain-like contrast, peak ~0.7
    gt_vol[:, :, 160] = gt_slice
    far_peak = float(gt_slice.max()) * 6.0
    gt_vol[:, :, 50] = far_peak  # volume-wide peak lives on a slice nobody is looking at
    gt_volume_max = float(np.abs(gt_vol).max())
    assert gt_volume_max > 3 * float(gt_slice.max()), \
        "fixture sanity: volume peak must dominate the slice-160 peak"

    # Candidate 1: good reconstruction (small error)
    rec1_vol = np.zeros((256, 220, 256), dtype=np.float32)
    rec1_slice = gt_slice + rng.normal(0, 0.01, gt_slice.shape).astype(np.float32)
    rec1_vol[:, :, 160] = rec1_slice

    # Candidate 2: poor reconstruction (large error)
    rec2_vol = np.zeros((256, 220, 256), dtype=np.float32)
    rec2_slice = gt_slice + rng.normal(0, 0.05, gt_slice.shape).astype(np.float32)
    rec2_vol[:, :, 160] = rec2_slice

    # The reference "real panel": rendered the way the convention requires -- normalised to the
    # volume-wide GT peak, not the slice peak.
    reference_panel = _render_error_panel(rec1_slice, gt_slice, gt_volume_max=gt_volume_max)
    print(f"  Error map shape: {reference_panel.shape}, "
          f"range: [{reference_panel.min():.3f}, {reference_panel.max():.3f}]")

    candidates = {
        "good": {"rec": rec1_vol, "gt": gt_vol},
        "poor": {"rec": rec2_vol, "gt": gt_vol},
    }
    ranked = match_error_panel(reference_panel, candidates, slice_index=160)
    assert ranked[0][0] == "good", f"Expected good first, got {ranked[0][0]}"
    assert ranked[1][1] < ranked[0][1], f"Poor should rank lower than good: {ranked[1][1]:.4f} vs {ranked[0][1]:.4f}"
    gap = ranked[0][1] - ranked[1][1]
    assert gap > 0.01, f"Gap between good and poor too small: {gap:.4f} (need >0.01)"
    print(f"  Error-panel ranking: good r={ranked[0][1]:.4f}, poor r={ranked[1][1]:.4f}, gap={gap:.4f}")

    # The ranker must be using the VOLUME-WIDE peak: match_error_panel's internal rendering of
    # "good" was built (inside match_error_panel) with the same gt_volume_max as reference_panel
    # was built with here, so the match should be near-exact.
    r_correct = ranked[0][1]
    assert r_correct > 0.999, f"volume-wide-normalised self-match should be near-exact: r={r_correct:.6f}"

    # What the OLD (per-slice, autoscaling) normalisation would have produced for "good",
    # rendered against the SAME reference panel. match_error_panel no longer takes this code
    # path; this exists purely so the r_correct > 0.999 assertion above is demonstrably
    # meaningful -- reverting match_error_panel to per-slice normalisation collapses its
    # reading to roughly this value, well under the 0.999 bound.
    wrong_panel = _render_error_panel(rec1_slice, gt_slice, gt_volume_max=None)  # fallback = per-slice peak
    wv = wrong_panel.ravel() - wrong_panel.mean()
    rv = reference_panel.ravel() - reference_panel.mean()
    r_wrong = float(wv @ rv / (np.linalg.norm(wv) * np.linalg.norm(rv) + 1e-12))
    # Empirically (jet is close to monotonic over the low error range both normalisations land
    # in) the two renderings only diverge by ~0.07-0.09 in Pearson r, not dramatically -- but
    # the divergence is real and reproducible, so a 0.03 margin is a safe, non-tautological bar.
    assert r_wrong < r_correct - 0.03, (
        f"per-slice and volume-wide normalisation should render visibly differently: "
        f"r_correct={r_correct:.4f}, r_wrong={r_wrong:.4f}"
    )
    print(f"  normalisation check: r_correct(volume-wide)={r_correct:.6f}, "
          f"r_wrong(per-slice, would-be-bug)={r_wrong:.6f}")

    print("PASS test_error_panel_ranking")


def test_error_panel_calibration_stake_has_a_working_negative_control():
    """check_error_calibration must be a stake that CAN fail.

    The retired version called _render_error_panel twice with identical arguments and
    correlated the two results -- a pure, deterministic function compared to itself is r=1.0
    by construction, incapable of detecting anything, and it never touched the real PNG panel.

    The replacement renders a POSITIVE reading (the claimed slice) and a NEGATIVE reading (a
    control that should NOT match -- here, the same candidate rendered from a clearly different
    slice) against the same real panel. This test proves the stake actually discriminates: it
    must pass when the claim is correct, and -- with the claimed/control roles swapped -- it
    must correctly report failure. A tautological stake could not do the second half.
    """
    rng = np.random.default_rng(7)

    gt_vol = np.zeros((256, 220, 256), dtype=np.float32)
    rec_vol = np.zeros((256, 220, 256), dtype=np.float32)
    true_slice, other_slice = 160, 60

    gt_true = (rng.random((256, 220)) * 0.5 + 0.2).astype(np.float32)
    gt_other = (rng.random((256, 220)) * 0.5 + 0.2).astype(np.float32)
    gt_vol[:, :, true_slice] = gt_true
    gt_vol[:, :, other_slice] = gt_other
    rec_vol[:, :, true_slice] = gt_true + rng.normal(0, 0.01, gt_true.shape).astype(np.float32)
    rec_vol[:, :, other_slice] = gt_other + rng.normal(0, 0.01, gt_other.shape).astype(np.float32)

    gt_volume_max = float(np.abs(gt_vol).max())
    # The "real panel" is rendered from the true slice, exactly as LOOK_T1_axial.png's panel 3
    # was rendered from whichever volume/slice actually produced it.
    panel = _render_error_panel(
        rec_vol[:, :, true_slice], gt_vol[:, :, true_slice], gt_volume_max=gt_volume_max
    )
    cand_data = {"rec": rec_vol, "gt": gt_vol}

    # Correctly claimed slice must beat the wrong-slice control.
    result = check_error_calibration(panel, cand_data, slice_index=true_slice, negative_slice_index=other_slice)
    assert result["passed"], f"stake should pass when the claimed slice is the real source: {result}"
    assert result["positive"] > result["negative"], result
    print(f"  correct claim: positive={result['positive']:.4f}, negative={result['negative']:.4f}, "
          f"margin={result['margin']:.4f}")

    # Swap the roles: claim the WRONG slice, use the true source as the "negative" control.
    # The stake must now correctly report failure -- proving it is not a tautology.
    swapped = check_error_calibration(panel, cand_data, slice_index=other_slice, negative_slice_index=true_slice)
    assert not swapped["passed"], f"stake should FAIL when the claimed slice is wrong: {swapped}"
    assert swapped["positive"] < swapped["negative"], swapped
    print(f"  wrong claim:   positive={swapped['positive']:.4f}, negative={swapped['negative']:.4f}, "
          f"margin={swapped['margin']:.4f}")

    print("PASS test_error_panel_calibration_stake_has_a_working_negative_control")


def test_psf_separates_uniform_undersampling_from_blue_noise():
    n = 128
    uniform = np.zeros((n, n), np.float32)
    uniform[::4, :] = 1.0                        # every 4th PE line -> discrete replicas
    rng = np.random.default_rng(0)
    blue = (rng.random((n, n)) < 0.25).astype(np.float32)   # same sampling rate, no structure
    u, b = psf_spread(uniform), psf_spread(blue)
    assert u["peak_sidelobe"] > 0.5, f"uniform must show a discrete replica, got {u['peak_sidelobe']:.3f}"
    assert b["peak_sidelobe"] < 0.1, f"random must be diffuse, got {b['peak_sidelobe']:.3f}"
    print("PASS test_psf_separates_uniform_undersampling_from_blue_noise")


def test_air_energy_ratio_is_one_when_the_recon_leaves_air_alone():
    rng = np.random.default_rng(1)
    gt = np.abs(rng.normal(0, 0.01, (32, 32, 32))).astype(np.float32)
    gt[12:20, 12:20, 12:20] = 1.0
    brain = gt > 0.5
    same = gt.copy()
    curve = air_energy_curve(same, gt, brain, thresholds=(0.02, 0.05, 0.10))
    for row in curve:
        assert abs(row["ratio_rec"] - 1.0) < 1e-5, f"identical volumes must give 1.0, got {row}"

    # Verify the curve actually varies with threshold: at t=0.02 we get different voxels than t>=0.05
    n_voxels = [row["n_voxels"] for row in curve]
    assert len(set(n_voxels)) > 1, f"threshold sweep must select different voxels, got {n_voxels}"

    leaky = gt.copy()
    leaky[~brain] *= 3.0
    curve2 = air_energy_curve(leaky, gt, brain, thresholds=(0.02, 0.05, 0.10))
    for row in curve2:
        # 3x amplitude scaling means 3^2 = 9x energy scaling; this pins energy semantics
        # (an unsquared amplitude-only implementation would give ratio=3.0, which must FAIL)
        assert 8.9 < row["ratio_rec"] < 9.1, f"3x amplitude must give ~9.0 energy ratio, got {row['ratio_rec']:.6f}"
    print("PASS test_air_energy_ratio_is_one_when_the_recon_leaves_air_alone")


def test_ncc_is_one_for_identical_fields_and_zero_for_independent_ones():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, (24, 24, 24))
    b = rng.normal(0, 1, (24, 24, 24))
    m = np.ones((24, 24, 24), bool)
    assert abs(ncc(a, a, m) - 1.0) < 1e-9
    assert abs(ncc(a, b, m)) < 0.1
    print("PASS test_ncc_is_one_for_identical_fields_and_zero_for_independent_ones")


def test_white_noise_falls_inside_its_surrogate_band_and_smooth_noise_does_not():
    # n=64 is required for statistical power: the band estimates a 97.5th percentile, and
    # under exchangeability P(original exceeds the band at one lag) ≈ 1/(n+1) ≈ 1.5% at n=64.
    # The test checks 8 lags, so per-lag exceedance rate compounds to ~12% if uncorrelated; lags
    # ARE correlated, but this still requires n >> 24 to avoid flakiness. n=64 gives margin.
    rng = np.random.default_rng(3)
    m = np.ones((32, 32, 32), bool)
    white = rng.normal(0, 1, (32, 32, 32))
    lo, hi = surrogate_band(white, m, n=64, seed=0)
    p = autocorr_profile(white, m)
    inside = np.all((p[1:] >= lo[1:]) & (p[1:] <= hi[1:]))
    assert inside, f"white noise must sit inside its own surrogate band:\n{p}\n{lo}\n{hi}"

    from scipy import ndimage as ndi
    smooth = ndi.gaussian_filter(rng.normal(0, 1, (32, 32, 32)), 2.0)
    lo2, hi2 = surrogate_band(smooth, m, n=64, seed=0)
    p2 = autocorr_profile(smooth, m)
    assert p2[1] > hi2[1], f"smooth noise must escape the band at lag 1: {p2[1]:.3f} vs {hi2[1]:.3f}"
    print("PASS test_white_noise_falls_inside_its_surrogate_band_and_smooth_noise_does_not")


if __name__ == "__main__":
    test_match_panel_ranks_the_true_source_first()
    test_calibration_stake_on_synthetic_figure()
    test_extract_panel_recovers_synthetic_blocks()
    test_error_panel_ranking()
    test_error_panel_calibration_stake_has_a_working_negative_control()
    test_psf_separates_uniform_undersampling_from_blue_noise()
    test_air_energy_ratio_is_one_when_the_recon_leaves_air_alone()
    test_ncc_is_one_for_identical_fields_and_zero_for_independent_ones()
    test_white_noise_falls_inside_its_surrogate_band_and_smooth_noise_does_not()
    print("ALL PASS")
