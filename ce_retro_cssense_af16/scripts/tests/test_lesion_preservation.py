"""unit tests for the lesion-preservation measures -- synthetic volumes, no topic data."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lesion_preservation import (cnr, contrast, edge_width, gmag, nrmse_in,  # noqa: E402
                                 psnr_in, signed_distance)


def ball(shape, centre, radius):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2 <= radius ** 2


def test_contrast_of_a_dark_lesion_is_positive_and_scale_free():
    v = np.ones((10, 10, 10))
    core = np.zeros((10, 10, 10), bool); core[2:5, 2:5, 2:5] = True
    wm = np.zeros((10, 10, 10), bool); wm[6:10, 6:10, 6:10] = True
    v[core] = 0.4
    assert contrast(v, core, wm) == pytest.approx(0.6)
    assert contrast(v * 137.0, core, wm) == pytest.approx(0.6)   # a global gain cannot move it


def test_contrast_is_zero_when_the_lesion_matches_white_matter():
    v = np.ones((8, 8, 8))
    core = np.zeros((8, 8, 8), bool); core[1:3, 1:3, 1:3] = True
    wm = np.zeros((8, 8, 8), bool); wm[5:8, 5:8, 5:8] = True
    assert contrast(v, core, wm) == pytest.approx(0.0)


def test_cnr_falls_when_noise_is_added_at_constant_contrast():
    rng = np.random.default_rng(0)
    shape = (24, 24, 24)
    core = np.zeros(shape, bool); core[4:8, 4:8, 4:8] = True
    wm = np.zeros(shape, bool); wm[12:24, 12:24, 12:24] = True
    clean = np.ones(shape); clean[core] = 0.4
    noisy = clean + rng.normal(0, 0.05, shape)
    assert contrast(noisy, core, wm) == pytest.approx(contrast(clean, core, wm), abs=0.02)
    assert cnr(noisy, core, wm) < 0.2 * cnr(clean, core, wm)


def test_psnr_and_nrmse_are_exact_on_a_known_error():
    gt = np.full((6, 6, 6), 2.0)
    rec = gt.copy()
    roi = np.zeros((6, 6, 6), bool); roi[0:3, 0:3, 0:3] = True
    rec[roi] = 2.0 + 0.1
    assert psnr_in(rec, gt, roi, peak=2.0) == pytest.approx(20 * np.log10(2.0) - 10 * np.log10(0.01))
    assert nrmse_in(rec, gt, roi) == pytest.approx(0.1 / 2.0)
    assert psnr_in(gt, gt, roi, peak=2.0) == float("inf")


def test_signed_distance_sign_convention():
    core = np.zeros((20, 20, 20), bool); core[8:12, 8:12, 8:12] = True
    sd = signed_distance(core)
    assert sd[10, 10, 10] < 0          # inside
    assert sd[0, 0, 0] > 0             # outside
    assert abs(sd[core].max()) <= 2.0


def test_edge_width_grows_with_blur():
    """a sharp dark ball, then the same ball blurred: the 20-80% rise must get wider, never narrower"""
    shape = (40, 40, 40)
    core = ball(shape, (20, 20, 20), 6)
    sharp = np.ones(shape); sharp[core] = 0.3
    sd = signed_distance(core)
    w_sharp = edge_width(sharp, sd)
    w_blur = edge_width(ndi.gaussian_filter(sharp, 1.5), sd)
    assert np.isfinite(w_sharp) and np.isfinite(w_blur)
    assert w_blur > w_sharp + 0.2


def test_edge_width_is_nan_when_there_is_no_rise():
    shape = (30, 30, 30)
    core = ball(shape, (15, 15, 15), 5)
    flat = np.ones(shape)
    assert np.isnan(edge_width(flat, signed_distance(core)))


def test_gmag_matches_the_forward_difference_definition():
    v = np.zeros((4, 4, 4))
    v[1, 1, 1] = 1.0
    g = gmag(v)
    assert g[0, 1, 1] == pytest.approx(1.0, abs=1e-5)     # difference along the first axis
    assert g[1, 1, 1] == pytest.approx(np.sqrt(3.0), abs=1e-5)
