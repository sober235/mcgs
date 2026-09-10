import numpy as np
import pytest

from cardiac_fr.gauge import (N_NYQ, af, error_map, heldout_split, image_metrics, kspace_nrmse,
                              monotone_worse, norm_scale, shot_subset)


def test_nyquist_and_af():
    assert N_NYQ == 19704
    assert af(19704) == 1.0 and af(9852) == 2.0


def test_heldout_split_is_disjoint_and_deterministic():
    ids = np.arange(0, 3000, 3)
    ho, pool = heldout_split(ids, 0.05, seed=0)
    assert ho.size == 50 and pool.size == 950
    assert np.intersect1d(ho, pool).size == 0 and np.array_equal(np.union1d(ho, pool), ids)
    assert np.array_equal(ho, heldout_split(ids, 0.05, seed=0)[0])
    assert not np.array_equal(ho, heldout_split(ids, 0.05, seed=1)[0])


def test_shot_subset_takes_whole_shots_close_to_target():
    lin = np.arange(22000)
    pool = lin[(lin % 22 != 0) & (lin % 7 != 3)]          # non-SI, some spokes missing
    sub = shot_subset(pool, target_af=12)
    assert abs(sub.size - N_NYQ / 12) <= 21
    assert np.isin(sub, pool).all()
    for s in np.unique(sub // 22):                         # every pool spoke of a chosen shot is in
        np.testing.assert_array_equal(np.sort(sub[sub // 22 == s]), pool[pool // 22 == s])


def _blob(n=24):
    x = np.linspace(-1, 1, n)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return np.exp(-3 * (X ** 2 + Y ** 2 + Z ** 2)).astype(np.float32)


def test_image_metrics_order_with_noise():
    ref, rng = _blob(), np.random.default_rng(0)
    small = ref + 0.01 * rng.standard_normal(ref.shape).astype(np.float32)
    large = ref + 0.05 * rng.standard_normal(ref.shape).astype(np.float32)
    box = ((6, 18), (6, 18), (6, 18))
    ms, ml = image_metrics(small, ref, heart_box=box), image_metrics(large, ref, heart_box=box)
    assert ms["body_psnr"] > ml["body_psnr"] and ms["body_ssim"] > ml["body_ssim"]
    assert ms["body_hfen"] < ml["body_hfen"] and ms["nmse"] < ml["nmse"]
    assert ms["heart_psnr"] > ml["heart_psnr"] and 0 < ms["body_frac"] <= 1


def test_kspace_nrmse_and_error_map():
    y = np.ones(10, np.complex64)
    assert kspace_nrmse(y, y) == 0.0 and kspace_nrmse(2 * y, y) == pytest.approx(1.0)
    ref = _blob()
    assert error_map(ref * 1.1, ref).max() == pytest.approx(0.1 * ref.max() / norm_scale(ref), rel=1e-5)


def test_monotone_worse():
    assert monotone_worse({8: 30.0, 12: 28.0, 16: 27.5}, higher_is_better=True)
    assert not monotone_worse({8: 30.0, 12: 30.5, 16: 27.5}, higher_is_better=True)
    assert monotone_worse({8: 0.1, 12: 0.2, 16: 0.3}, higher_is_better=False)
