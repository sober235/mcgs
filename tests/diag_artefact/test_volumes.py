"""Region masks must isolate what their names claim on a volume whose answer is known."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from diag_artefact.volumes import brain_mask, wm_mask, air_mask, load_volumes
from tests.diag_artefact.conftest import make_phantom as _phantom


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

    # fix round 1 (2026-08-11): wm_mask's blur-threshold step can leak a voxel through purely from
    # its neighbours' density even when the voxel's OWN intensity sits outside the [P55,P90] band
    # (or outside brain) -- the published recipe guards against this by intersecting the eroded
    # result back with the band mask. Assert that subset property directly so a regression (e.g.
    # dropping that intersection again) is caught here instead of only showing up as a drift in the
    # real-data calibration numbers.
    vals = vol[b]
    lo, hi = np.percentile(vals, 55), np.percentile(vals, 90)
    band = b & (vol > lo) & (vol < hi)
    assert not (w & ~band).any(), "WM mask must contain no voxel outside the [P55,P90] band"
    print("PASS test_wm_mask_selects_the_plateau_not_the_shell_or_core")


def test_air_mask_is_disjoint_from_brain_and_keeps_its_distance():
    vol = _phantom()
    b = brain_mask(vol)
    a = air_mask(vol, b, dilate=4)
    assert not (a & b).any(), "air and brain must be disjoint"
    assert a[0, 0, 0], "the corner of the field must be air"
    assert a.sum() > 0
    print("PASS test_air_mask_is_disjoint_from_brain_and_keeps_its_distance")


def test_load_volumes_maps_contrasts_correctly():
    """load_volumes must map contrast names to the correct npz keys without mixing T1/FLAIR."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "test.npz")

        # Create distinguishable test arrays
        rec_t1 = np.ones((256, 220, 256), dtype=np.float32) * 1.0
        gt_t1 = np.ones((256, 220, 256), dtype=np.float32) * 2.0
        zf_t1 = np.ones((256, 220, 256), dtype=np.float32) * 3.0
        rec_fl = np.ones((256, 220, 256), dtype=np.float32) * 4.0
        gt_fl = np.ones((256, 220, 256), dtype=np.float32) * 5.0
        zf_fl = np.ones((256, 220, 256), dtype=np.float32) * 6.0

        # Save to npz
        np.savez(npz_path, recT1=rec_t1, gtT1=gt_t1, zfT1=zf_t1,
                 recFL=rec_fl, gtFL=gt_fl, zfFL=zf_fl)

        # Test T1 contrast
        vols_t1 = load_volumes(npz_path, contrast="T1")
        assert np.allclose(vols_t1["rec"], 1.0), "T1 rec should be 1.0"
        assert np.allclose(vols_t1["gt"], 2.0), "T1 gt should be 2.0"
        assert np.allclose(vols_t1["zf"], 3.0), "T1 zf should be 3.0"

        # Test FLAIR contrast
        vols_fl = load_volumes(npz_path, contrast="FLAIR")
        assert np.allclose(vols_fl["rec"], 4.0), "FLAIR rec should be 4.0"
        assert np.allclose(vols_fl["gt"], 5.0), "FLAIR gt should be 5.0"
        assert np.allclose(vols_fl["zf"], 6.0), "FLAIR zf should be 6.0"

        print("PASS test_load_volumes_maps_contrasts_correctly")


def test_load_volumes_rejects_unknown_contrast():
    """load_volumes must raise ValueError for unknown contrast."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "test.npz")
        np.savez(npz_path, recT1=np.zeros((256, 220, 256), dtype=np.float32))

        try:
            load_volumes(npz_path, contrast="T2")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "T2" in str(e), f"Error message should mention T2: {e}"

        print("PASS test_load_volumes_rejects_unknown_contrast")


def test_load_volumes_rejects_missing_keys():
    """load_volumes must raise KeyError when required keys are missing."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "test.npz")
        # Create npz with only FLAIR keys, not T1
        np.savez(npz_path, recFL=np.zeros((256, 220, 256), dtype=np.float32),
                 gtFL=np.zeros((256, 220, 256), dtype=np.float32),
                 zfFL=np.zeros((256, 220, 256), dtype=np.float32))

        try:
            load_volumes(npz_path, contrast="T1")
            assert False, "Should have raised KeyError"
        except KeyError as e:
            assert "recT1" in str(e) or "missing" in str(e).lower(), f"Error should mention missing T1 keys: {e}"

        print("PASS test_load_volumes_rejects_missing_keys")


def test_load_volumes_rejects_mismatched_shapes_and_returns_float32():
    """Final review (minors): `load_volumes` validated neither dtype nor shape, so an npz whose
    three volumes disagree would have surfaced as a broadcasting error inside a statistic rather
    than as "this npz is not what you think it is"."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        good = os.path.join(tmpdir, "good.npz")
        np.savez(good, recT1=np.zeros((8, 8, 8), dtype=np.float64),
                 gtT1=np.zeros((8, 8, 8), dtype=np.float64),
                 zfT1=np.zeros((8, 8, 8), dtype=np.float64))
        vols = load_volumes(good, "T1")
        assert all(v.dtype == np.float32 for v in vols.values()), \
            {k: v.dtype for k, v in vols.items()}

        bad = os.path.join(tmpdir, "bad.npz")
        np.savez(bad, recT1=np.zeros((8, 8, 8), dtype=np.float32),
                 gtT1=np.zeros((8, 8, 7), dtype=np.float32),
                 zfT1=np.zeros((8, 8, 8), dtype=np.float32))
        try:
            load_volumes(bad, "T1")
            assert False, "mismatched volume shapes must raise"
        except ValueError as e:
            assert "common 3-D" in str(e), e
    print("PASS test_load_volumes_rejects_mismatched_shapes_and_returns_float32")


if __name__ == "__main__":
    test_brain_mask_captures_the_ball_and_excludes_the_field()
    test_wm_mask_selects_the_plateau_not_the_shell_or_core()
    test_air_mask_is_disjoint_from_brain_and_keeps_its_distance()
    test_load_volumes_maps_contrasts_correctly()
    test_load_volumes_rejects_unknown_contrast()
    test_load_volumes_rejects_missing_keys()
    test_load_volumes_rejects_mismatched_shapes_and_returns_float32()
    print("ALL PASS")
