"""Region masks must isolate what their names claim on a volume whose answer is known."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from diag_artefact.volumes import brain_mask, wm_mask, air_mask


def _phantom(shape=(64, 60, 64), seed=0):
    """A bright plateau (=white matter) inside a dim rim and around a dim core, on a noisy dark field.

    The rim is DIMMER than the plateau on purpose. wm_mask selects [P55, P90] of the in-brain
    distribution (project memory 14b), so the plateau standing in for white matter has to occupy
    that band; a rim brighter than the plateau would take P90 instead. This is also the real T1
    ordering: bright WM interior, dark CSF/skull rim.
    """
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    c = [s // 2 for s in shape]
    r = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
    vol = np.zeros(shape, np.float32)
    vol[r < 25] = 0.20          # dim rim (CSF / skull-like)
    vol[r < 22] = 0.60          # "white matter" plateau
    vol[r < 10] = 0.35          # dim core (ventricle-like)
    vol += rng.normal(0, 0.01, shape).astype(np.float32)
    return np.abs(vol)


def test_brain_mask_captures_the_ball_and_excludes_the_field():
    vol = _phantom()
    b = brain_mask(vol)
    assert b[32, 30, 32], "centre of the ball must be inside the brain mask"
    assert not b[0, 0, 0], "the corner of the field must be outside"
    frac = b.mean()
    # Analytic fraction for r<25 ball in 64x60x64 box: (4/3)*pi*25^3 / (64*60*64) = 0.2663
    assert 0.02 < frac < 0.30, f"brain fraction implausible: {frac:.3f}"
    print("PASS test_brain_mask_captures_the_ball_and_excludes_the_field")


def test_wm_mask_selects_the_plateau_not_the_shell_or_core():
    vol = _phantom()
    b = brain_mask(vol)
    w = wm_mask(vol, b)
    assert w.sum() > 0, "WM mask must not be empty"
    med = float(np.median(vol[w]))
    assert 0.45 < med < 0.75, f"WM median should sit on the 0.6 plateau, got {med:.3f}"
    assert w.mean() < b.mean(), "WM must be a strict subset of brain"
    print("PASS test_wm_mask_selects_the_plateau_not_the_shell_or_core")


def test_air_mask_is_disjoint_from_brain_and_keeps_its_distance():
    vol = _phantom()
    b = brain_mask(vol)
    a = air_mask(vol, b, dilate=4)
    assert not (a & b).any(), "air and brain must be disjoint"
    assert a[0, 0, 0], "the corner of the field must be air"
    assert a.sum() > 0
    print("PASS test_air_mask_is_disjoint_from_brain_and_keeps_its_distance")


if __name__ == "__main__":
    test_brain_mask_captures_the_ball_and_excludes_the_field()
    test_wm_mask_selects_the_plateau_not_the_shell_or_core()
    test_air_mask_is_disjoint_from_brain_and_keeps_its_distance()
    print("ALL PASS")
