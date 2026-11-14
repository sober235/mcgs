import numpy as np
import pytest
import scipy.ndimage as ndi

import lesions
from helpers import wm_phantom


def _setup():
    t1 = wm_phantom()
    brain = lesions.brain_mask(t1, erode=8, cut_below=30)
    return t1, brain, lesions.pick_sites(t1, brain)


def test_brain_mask_is_inside_the_head_and_cut_below():
    t1 = wm_phantom()
    brain = lesions.brain_mask(t1, erode=8, cut_below=30)
    assert brain.sum() > 20000 and not brain[t1 < 0.05].any()
    assert not brain[int(round(ndi.center_of_mass(t1 > 0.1)[0])) + 30:].any()


def test_sites_layout():
    t1, brain, sites = _setup()
    assert [s["contrast"] for s in sites] == ["T2", "T2", "T2", "T2", "T1"]
    assert [s["diameter_mm"] for s in sites] == [4.0, 6.0, 8.0, 12.0, 8.0]
    assert [s["gain"] for s in sites] == [1.0, 0.9, 0.75, 0.6, -0.4]
    depth = ndi.distance_transform_edt(brain)
    cen = np.array([s["center_vox"] for s in sites], dtype=float)
    for s in sites:
        assert depth[tuple(s["center_vox"])] >= s["diameter_mm"] / 2 + 3
    d = np.linalg.norm(cen[:, None] - cen[None], axis=-1)
    assert d[np.triu_indices(5, 1)].min() >= 25
    assert sites == lesions.pick_sites(t1, brain)                         # deterministic


def test_insert_changes_only_its_own_contrast():
    t1, brain, sites = _setup()
    v2, mask = lesions.insert(t1.astype(complex), sites, "T1")
    assert sorted(np.unique(mask).tolist()) == [0, 1, 2, 3, 4, 5]
    core = ndi.binary_erosion(mask == 5, iterations=2)
    assert core.sum() >= 7
    assert np.abs(v2)[core].mean() / np.abs(t1)[core].mean() == pytest.approx(0.6, abs=0.02)
    other = (mask > 0) & (mask < 5)
    assert np.array_equal(np.abs(v2)[other], np.abs(t1.astype(complex))[other])   # T2-only lesions leave T1 alone
    w2, mask2 = lesions.insert(t1.astype(complex), sites, "T2")
    assert np.array_equal(mask, mask2)
    big = ndi.binary_erosion(mask == 4, iterations=3)
    assert np.abs(w2)[big].mean() / np.abs(t1)[big].mean() == pytest.approx(1.6, abs=0.03)
    assert np.array_equal(w2[mask == 0][:1000], t1.astype(complex)[mask == 0][:1000]) or \
        np.abs(w2 - t1)[ndi.binary_dilation(mask > 0, iterations=6) == 0].max() < 1e-6


def test_insert_keeps_phase():
    t1, brain, sites = _setup()
    v = t1 * np.exp(0.3j)
    v2, _ = lesions.insert(v, sites, "T2")
    np.testing.assert_allclose(np.angle(v2[t1 > 0.1]), 0.3, atol=1e-9)
