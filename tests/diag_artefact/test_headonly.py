"""Head-only metrics (spec 2026-09-05 §2.1). The reference values in gates.json and every candidate
are scored by THIS code, so the tests pin the definitions, not just the plumbing."""
import numpy as np
import pytest
from scipy import ndimage

from diag_artefact.headonly import (grain_ratio, head_hfen, head_mask, head_psnr, head_ssim,
                                    log_kernel, score_headonly)


def test_head_mask_is_five_percent_of_p999(phantom):
    head = head_mask(phantom)
    p999 = np.quantile(phantom, 0.999)
    assert head.dtype == bool
    assert np.array_equal(head, phantom > 0.05 * p999)
    assert 0.2 < head.mean() < 0.6          # the ball, not the whole box, not empty


def test_identical_volumes_are_perfect(phantom):
    s = score_headonly(phantom, phantom)
    assert s["head_psnr"] == float("inf")
    assert s["head_ssim"] == pytest.approx(1.0, abs=1e-9)
    assert s["head_hfen"] == pytest.approx(0.0, abs=1e-12)
    assert s["grain_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert s["gmax"] == pytest.approx(float(phantom.max()))


def test_head_psnr_ignores_background_error(phantom):
    head = head_mask(phantom)
    rec = phantom.copy()
    rec[~head] += 0.3                           # wreck the background only
    assert head_psnr(rec, phantom, head, float(phantom.max())) == float("inf")


def test_head_hfen_ranks_high_frequency_error_above_smooth_error(phantom):
    rng = np.random.default_rng(3)
    head = head_mask(phantom)
    gmax = float(phantom.max())
    hf = (phantom + np.where(head, rng.normal(0, 0.02, phantom.shape), 0.0)).astype(np.float32)
    lf = (phantom + np.where(head, 0.02 * ndimage.gaussian_filter(rng.normal(0, 1, phantom.shape), 6.0), 0.0)).astype(np.float32)
    assert head_hfen(phantom, phantom, head, gmax) == 0.0
    assert head_hfen(hf, phantom, head, gmax) > head_hfen(lf, phantom, head, gmax)


def test_head_psnr_formula_and_no_rescaling(phantom):
    head = head_mask(phantom)
    rec = phantom * 1.1                          # a pure gain must COST psnr (no LS amplitude projection)
    gmax = float(phantom.max())
    mse = float(np.mean((rec[head].astype(np.float64) - phantom[head].astype(np.float64)) ** 2))
    expect = 20 * np.log10(gmax) - 10 * np.log10(mse)
    assert head_psnr(rec, phantom, head, gmax) == pytest.approx(expect, abs=1e-9)
    assert np.isfinite(expect)


def test_head_ssim_is_mean_of_map_inside_head(phantom):
    from skimage.metrics import structural_similarity
    rng = np.random.default_rng(1)
    rec = (phantom + rng.normal(0, 0.02, phantom.shape)).astype(np.float32)
    head = head_mask(phantom)
    gmax = float(phantom.max())
    _, smap = structural_similarity(rec, phantom, data_range=gmax, full=True)
    assert head_ssim(rec, phantom, head, gmax) == pytest.approx(float(smap[head].mean()), abs=1e-9)
    assert 0.3 < head_ssim(rec, phantom, head, gmax) < 1.0


def test_log_kernel_matches_paired_gs_definition():
    k = log_kernel(1.5, 15)
    assert k.shape == (15, 15, 15)
    assert abs(k.sum()) < 1e-12                 # DC-removed
    assert k[7, 7, 7] < 0                       # centre of a LoG is negative
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "paired_gs"))
        from metrics.metrics import _log3d_kernel
    except Exception:
        pytest.skip("paired_gs metrics not importable here")
    assert np.allclose(k, _log3d_kernel(1.5, 15), atol=1e-15)


def test_grain_ratio_orders_smooth_and_noisy(phantom):
    rng = np.random.default_rng(2)
    smooth = ndimage.gaussian_filter(phantom, 1.0).astype(np.float32)
    noisy = (phantom + rng.normal(0, 0.03, phantom.shape)).astype(np.float32)
    assert grain_ratio(smooth, phantom) < 1.1  # smooth should have less or similar grain (numerical tolerance)
    assert grain_ratio(noisy, phantom) > 1.0


def test_score_headonly_returns_python_floats(phantom):
    s = score_headonly(phantom * 0.98, phantom)
    for k in ("head_psnr", "head_ssim", "head_hfen", "grain_ratio", "head_frac", "gmax"):
        assert type(s[k]) is float, k
