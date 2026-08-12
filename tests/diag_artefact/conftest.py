"""The one calibration phantom every diag_artefact test builds on.

It used to be copy-pasted into `test_volumes.py` and `test_report.py`, kept in sync by a comment
saying "same fixture as test_volumes.py's `_phantom`" -- which is exactly the arrangement that
silently diverges the first time one copy is tuned. The masks under test are extremely sensitive
to the phantom's texture (see the docstring below), so a divergence would not fail loudly; it
would quietly change what the tests are testing.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def make_phantom(shape=(64, 60, 64), seed=0):
    """A bright plateau (=white matter) inside a dim rim and around a dim core, on a noisy dark field.

    The rim is DIMMER than the plateau on purpose. wm_mask selects [P55, P90] of the in-brain
    distribution (project memory 14b), so the plateau standing in for white matter has to occupy
    that band; a rim brighter than the plateau would take P90 instead. This is also the real T1
    ordering: bright WM interior, dark CSF/skull rim.

    The plateau's texture is spatially correlated (gaussian_filter sigma=6, amplitude 0.10) rather
    than i.i.d.: the published `wm_mask` erodes with `gaussian_filter(mask, 2) > 0.95`, a strong
    erosion that only survives where the [P55,P90]-band voxels form a big, spatially SOLID region.
    With i.i.d. plateau noise the band picks out a salt-and-pepper subset and the erosion eats all
    of it, leaving an empty WM mask -- not because the property under test became false, but
    because the fixture stopped resembling the thing the mask is designed for. The background
    field stays exactly flat, so brain_mask's corner-exclusion test is undisturbed.
    """
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    c = [s // 2 for s in shape]
    r = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
    vol = np.zeros(shape, np.float32)
    vol[r < 25] = 0.20          # dim rim (CSF / skull-like)
    vol[r < 22] = 0.60          # "white matter" plateau
    vol[r < 10] = 0.35          # dim core (ventricle-like)
    ball = r < 25
    texture = ndimage.gaussian_filter(rng.normal(0, 1, shape), 6.0)
    texture = texture / texture.std() * 0.10
    vol += np.where(ball, texture, 0.0).astype(np.float32)   # spatially-correlated, ball-only
    vol += rng.normal(0, 0.01, shape).astype(np.float32)     # small i.i.d. floor
    return np.abs(vol).astype(np.float32)


@pytest.fixture
def phantom():
    return make_phantom()
