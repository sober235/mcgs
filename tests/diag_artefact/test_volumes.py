"""Region masks must isolate what their names claim on a volume whose answer is known."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy import ndimage
from diag_artefact.volumes import brain_mask, wm_mask, air_mask, load_volumes


def _phantom(shape=(64, 60, 64), seed=0):
    """A bright plateau (=white matter) inside a dim rim and around a dim core, on a noisy dark field.

    The rim is DIMMER than the plateau on purpose. wm_mask selects [P55, P90] of the in-brain
    distribution (project memory 14b), so the plateau standing in for white matter has to occupy
    that band; a rim brighter than the plateau would take P90 instead. This is also the real T1
    ordering: bright WM interior, dark CSF/skull rim.

    UPDATED (2026-08-11, alongside the switch to the published brain_mask/wm_mask): the published
    wm_mask erodes with `gaussian_filter(mask, sigma=2) > 0.95`, a STRONG erosion that only
    survives where the [P55,P90]-band voxels form a big, spatially SOLID region. The original
    fixture gave the plateau i.i.d. per-voxel noise (std 0.01); since real anatomy is spatially
    correlated but i.i.d. noise is not, the P55-P90 band picked out a scattered, salt-and-pepper
    subset of the plateau instead of a solid shell, and the strong erosion ate all of it (WM mask
    came back empty) -- not because the property under test became false, but because the fixture
    stopped resembling the thing the mask is designed for. Fix: give the plateau (and only the
    plateau/rim/core -- the background field stays exactly flat, so the brain-mask corner-exclusion
    test below is undisturbed) a smoothed, spatially-correlated texture (sigma=6, amplitude 0.10)
    instead of i.i.d. noise, so the P55-P90 band forms large contiguous patches that the strong
    erosion can have deep interior in -- while keeping the original property intact: WM should
    still be selected from the plateau's own value range, not the rim's or the core's.
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
    vol += rng.normal(0, 0.01, shape).astype(np.float32)     # small i.i.d. floor, as before
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


if __name__ == "__main__":
    test_brain_mask_captures_the_ball_and_excludes_the_field()
    test_wm_mask_selects_the_plateau_not_the_shell_or_core()
    test_air_mask_is_disjoint_from_brain_and_keeps_its_distance()
    test_load_volumes_maps_contrasts_correctly()
    test_load_volumes_rejects_unknown_contrast()
    test_load_volumes_rejects_missing_keys()
    print("ALL PASS")
