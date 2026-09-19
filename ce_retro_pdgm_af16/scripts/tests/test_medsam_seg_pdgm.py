"""unit tests for the PDGM segmentation helpers -- no model, no topic data."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "ce_retro_cssense_af16" / "scripts"))
import medsam_seg as uih  # noqa: E402  the UIH topic's copy, pinned against ours below
from medsam_seg_pdgm import (MARGIN, MIN_AREA, TARGETS, dice, hd95, largest_component,  # noqa: E402
                             slice_boxes, to_uint8_window)


def blob(shape, lo, hi):
    m = np.zeros(shape, bool)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return m


def test_transcribed_helpers_match_the_uih_copy():
    """the four helpers are copied, not imported; they must not drift apart"""
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.random((16, 16, 16)) > 0.6
        b = rng.random((16, 16, 16)) > 0.6
        assert dice(a, b) == uih.dice(a, b)
        assert largest_component(a).tolist() == uih.largest_component(a).tolist()
        if a.any() and b.any():
            assert hd95(a, b) == uih.hd95(a, b)
    img = rng.random((8, 8)) * 4
    assert np.array_equal(to_uint8_window(img), uih.to_uint8_window(img))


def test_slice_boxes_are_per_slice_and_grown_by_margin():
    ref = np.zeros((40, 40, 6), bool)
    ref[10:20, 12:22, 2] = True          # 100 voxels on slice 2
    ref[15:30, 5:25, 3] = True           # a different, larger cross-section on slice 3
    boxes = slice_boxes(ref)
    assert sorted(boxes) == [2, 3]
    assert boxes[2] == (10 - MARGIN, 12 - MARGIN, 19 + MARGIN, 21 + MARGIN)
    assert boxes[3] == (15 - MARGIN, 5 - MARGIN, 29 + MARGIN, 24 + MARGIN)


def test_slice_boxes_skip_slices_below_min_area():
    ref = np.zeros((30, 30, 4), bool)
    ref[5:10, 5:10, 1] = True             # 5 x 5 = 25 voxels, above MIN_AREA=20
    ref[5:9, 5:9, 2] = True               # 4 x 4 = 16 voxels, below
    assert MIN_AREA == 20
    assert sorted(slice_boxes(ref)) == [1]


def test_slice_boxes_clip_to_the_grid():
    ref = np.zeros((20, 20, 3), bool)
    ref[0:6, 15:20, 1] = True
    (r0, c0, r1, c1) = slice_boxes(ref)[1]
    assert (r0, c0) == (0, 15 - MARGIN)
    assert (r1, c1) == (5 + MARGIN, 19)


def test_slice_boxes_empty_reference_gives_no_prompt():
    assert slice_boxes(np.zeros((10, 10, 3), bool)) == {}


def test_target_label_sets():
    """the BraTS label convention of this dataset: 1 necrotic, 2 oedema, 4 enhancing"""
    assert TARGETS["tc"] == (1, 4)
    assert TARGETS["et"] == (4,)
    assert TARGETS["wt"] == (1, 2, 4)


def test_dice_and_hd95_on_known_geometry():
    a = blob((30, 30, 30), (10, 10, 10), (20, 20, 20))
    assert dice(a, a) == pytest.approx(1.0)
    assert hd95(a, a) == pytest.approx(0.0)
    b = blob((30, 30, 30), (13, 10, 10), (23, 20, 20))
    assert 2.0 <= hd95(a, b) <= 4.0
