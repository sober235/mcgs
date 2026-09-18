"""unit tests for the geometry helpers of medsam_seg.py -- no model, no topic data."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medsam_seg import dice, hd95, largest_component, prompt_box, to_uint8_window  # noqa: E402


def cube(shape, lo, hi):
    m = np.zeros(shape, bool)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return m


def test_dice_identity_and_disjoint():
    a = cube((20, 20, 20), (5, 5, 5), (15, 15, 15))
    assert dice(a, a) == pytest.approx(1.0)
    b = cube((20, 20, 20), (15, 15, 15), (20, 20, 20))
    assert dice(a, b) == pytest.approx(0.0)


def test_dice_half_overlap():
    """two equal-sized sets sharing exactly half their voxels -> 2*|A&B| / (|A|+|B|) = 0.5"""
    a = cube((20, 20, 20), (0, 0, 0), (10, 20, 20))
    b = cube((20, 20, 20), (5, 0, 0), (15, 20, 20))
    assert dice(a, b) == pytest.approx(0.5)


def test_dice_both_empty_is_nan():
    z = np.zeros((4, 4, 4), bool)
    assert np.isnan(dice(z, z))


def test_hd95_identity_is_zero():
    a = cube((30, 30, 30), (10, 10, 10), (20, 20, 20))
    assert hd95(a, a) == pytest.approx(0.0)


def test_hd95_shifted_cube():
    """a cube against the same cube shifted 3 voxels: the 95th percentile of the surface distances is ~3"""
    a = cube((40, 40, 40), (10, 10, 10), (25, 25, 25))
    b = cube((40, 40, 40), (13, 10, 10), (28, 25, 25))
    assert 2.0 <= hd95(a, b) <= 4.0


def test_hd95_empty_is_nan():
    a = cube((10, 10, 10), (2, 2, 2), (5, 5, 5))
    assert np.isnan(hd95(a, np.zeros((10, 10, 10), bool)))


def test_prompt_box_grows_the_reference_bounding_box():
    ref = cube((60, 60, 60), (20, 25, 30), (30, 35, 40))
    z0, z1, y0, y1, x0, x1 = prompt_box(ref, margin=4)
    assert (z0, z1) == (16, 33)          # 20-4 .. 29+4, inclusive
    assert (y0, y1) == (21, 38)
    assert (x0, x1) == (26, 43)


def test_prompt_box_clips_to_the_grid():
    ref = cube((20, 20, 20), (0, 0, 0), (3, 3, 20))
    z0, z1, y0, y1, x0, x1 = prompt_box(ref, margin=5)
    assert (z0, y0, x0) == (0, 0, 0)
    assert (z1, y1, x1) == (7, 7, 19)


def test_window_is_fixed_and_saturates():
    """the same physical intensity must map to the same grey level in every arm: a fixed window, not per-slice min-max"""
    img = np.array([[0.0, 1.0, 2.0, 4.0]], float)
    out = to_uint8_window(img, top=2.0)
    assert out.dtype == np.uint8
    # mid-window is 127.5 grey levels and the mapping rounds (half-to-even) rather than truncating
    assert out.tolist() == [[0, 128, 255, 255]]


def test_window_is_scale_covariant_after_wm_normalisation():
    """two volumes differing by a global gain give the identical 8-bit image once each is divided by its own WM median"""
    a = np.array([[0.2, 0.4, 0.8]], float)
    b = a * 137.0
    assert np.array_equal(to_uint8_window(a / np.median(a), top=2.0),
                          to_uint8_window(b / np.median(b), top=2.0))


def test_largest_component_drops_specks():
    """the stacked 2-D masks carry stray specks; only the main body should survive"""
    m = np.zeros((20, 20, 20), bool)
    m[5:12, 5:12, 5:12] = True          # the body
    m[0, 0, 0] = True                   # a speck
    m[18:20, 18:20, 18] = True          # another
    out = largest_component(m)
    assert out.sum() == 7 ** 3
    assert not out[0, 0, 0] and not out[18, 18, 18]


def test_largest_component_keeps_a_single_blob_and_survives_empty():
    m = np.zeros((8, 8, 8), bool)
    m[2:5, 2:5, 2:5] = True
    assert np.array_equal(largest_component(m), m)
    z = np.zeros((4, 4, 4), bool)
    assert not largest_component(z).any()


def test_largest_component_is_26_connected():
    """two cubes touching only at a corner are one 26-connected object, not two"""
    m = np.zeros((12, 12, 12), bool)
    m[2:5, 2:5, 2:5] = True
    m[5:8, 5:8, 5:8] = True             # shares only the corner voxel diagonal
    assert largest_component(m).sum() == 2 * 27
