"""run_all() is the piece task-9-brief.md's Step 3 promises but ships no code for -- it wires
M1-M5 and the controls into the `readings` dict `adjudicate` consumes. These tests exercise its
novel logic (the permutation-null significance tests behind P1/P2/P3, and the patchiness
attribution's agree/disagree rule) directly against synthetic arrays whose answer is known, the
same discipline `test_metrics.py` applies to the lower-level metrics themselves.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from diag_artefact.report import (
    _controls_from_manifest,
    _p1,
    _p1_all_lags_false_positive_rate,
    _p2,
    _p3,
    _patchiness,
    _permutation_null_ncc,
    _resolve_checkpoint,
    adjudicate,
    run_all,
)
from diag_artefact.volumes import brain_mask, wm_mask


def test_permutation_null_ncc_centers_near_zero_for_an_unrelated_reference():
    rng = np.random.default_rng(0)
    field = rng.normal(0, 1, (24, 24, 24))
    ref = rng.normal(0, 1, (24, 24, 24))  # independent of field
    m = np.ones((24, 24, 24), bool)
    null = _permutation_null_ncc(field, ref, m, n=64, seed=1)
    assert null.shape == (64,)
    assert abs(float(null.mean())) < 0.05, f"null distribution should center near 0: {null.mean():.4f}"
    print("PASS test_permutation_null_ncc_centers_near_zero_for_an_unrelated_reference")


def test_p1_all_lags_false_positive_rate_matches_the_independently_measured_value():
    # Deterministic (fixed seeds): reproduces the 11/20 = 0.55 quoted in task-9-brief.md's
    # limitation #2, independently re-derived here rather than hard-coded from the brief's text.
    rate = _p1_all_lags_false_positive_rate()
    assert rate == 0.55, f"expected the documented 0.55, got {rate}"
    print("PASS test_p1_all_lags_false_positive_rate_matches_the_independently_measured_value")


def _phantom_gt(shape=(64, 60, 64), seed=0):
    """Same fixture as test_volumes.py's `_phantom` (proven to give a non-empty wm_mask under the
    published, strongly-eroded [P55,P90]-band definition): a bright plateau (white matter) inside
    a dim rim, around a dim core, on a noisy dark field, with a spatially-correlated (not i.i.d.)
    texture so the WM band forms large contiguous patches the strong erosion has interior in.
    """
    from scipy import ndimage
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    c = [s // 2 for s in shape]
    r = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
    vol = np.zeros(shape, np.float32)
    vol[r < 25] = 0.20
    vol[r < 22] = 0.60
    vol[r < 10] = 0.35
    ball = r < 25
    texture = ndimage.gaussian_filter(rng.normal(0, 1, shape), 6.0)
    texture = texture / texture.std() * 0.10
    vol += np.where(ball, texture, 0.0).astype(np.float32)
    vol += rng.normal(0, 0.01, shape).astype(np.float32)
    return np.abs(vol).astype(np.float32)


def test_p1_true_when_residual_is_incoherent_and_shares_zfs_family():
    """Seed choice is not incidental: this project measured (task-9-brief.md limitation #2, and
    `_p1_all_lags_false_positive_rate` above) that P1's own literal all-lags-inside-band rule only
    passes ~55% of the time even on GENUINE white noise -- so a random seed has roughly even odds
    of making the stake fail through no fault of the code (verified empirically: seeds 0-29 split
    close to that ratio). Seed 8 is one of the seeds where a truly incoherent construction reads as
    incoherent, which is what this test needs to isolate; the point being tested is `_p1`'s
    combination logic, not the rule's own pass rate (that is `_p1_all_lags_false_positive_rate`'s
    job, above).
    """
    gt = _phantom_gt()
    b = brain_mask(gt)
    wm = wm_mask(gt, b)
    assert wm.sum() > 50, "fixture sanity: need a real WM region"

    rng = np.random.default_rng(8)
    err_zf = np.zeros_like(gt, dtype=np.float64)
    err_zf[wm] = rng.normal(0, 0.05, int(wm.sum()))          # pure white noise -- zf's aliasing
    err_rec = np.zeros_like(gt, dtype=np.float64)
    err_rec[wm] = 0.4 * err_zf[wm] + rng.normal(0, 0.01, int(wm.sum()))  # smaller, same family

    r = _p1(err_rec, err_zf, wm)
    assert r["stake_zf_inside_band_all_lags"] is True, r
    assert r["ncc_high_relative_to_null"] is True, r
    assert r["verdict"] == "true", r
    print("PASS test_p1_true_when_residual_is_incoherent_and_shares_zfs_family")


def test_p1_false_when_residual_has_structure_unrelated_to_zf():
    from scipy import ndimage
    gt = _phantom_gt()
    b = brain_mask(gt)
    wm = wm_mask(gt, b)

    # seed 8 again (see the docstring on the "true" test above): the zf stake construction is a
    # coin flip under the all-lags rule regardless of seed, and 8 is one where it reads incoherent.
    rng = np.random.default_rng(8)
    err_zf = np.zeros_like(gt, dtype=np.float64)
    err_zf[wm] = rng.normal(0, 0.05, int(wm.sum()))  # zf's own aliasing: still incoherent

    # a SMOOTH (spatially structured), independent field -- coherent structure unrelated to zf
    smooth = ndimage.gaussian_filter(rng.normal(0, 1, gt.shape), 3.0)
    err_rec = np.where(wm, smooth, 0.0)

    r = _p1(err_rec, err_zf, wm)
    assert r["stake_zf_inside_band_all_lags"] is True, r
    assert r["inside_band_all_lags"] is False, r
    assert r["ncc_high_relative_to_null"] is False, r
    assert r["verdict"] == "false", r
    print("PASS test_p1_false_when_residual_has_structure_unrelated_to_zf")


def test_p1_insufficient_when_the_zf_stake_itself_fails():
    from scipy import ndimage
    gt = _phantom_gt()
    b = brain_mask(gt)
    wm = wm_mask(gt, b)

    rng = np.random.default_rng(9)
    # zf's OWN residual is given real coherent structure -- an implausible mask, but the point is
    # to make the stake fail and check the metric is discarded rather than trusted anyway.
    err_zf = np.where(wm, ndimage.gaussian_filter(rng.normal(0, 1, gt.shape), 3.0), 0.0)
    err_rec = np.zeros_like(gt, dtype=np.float64)
    err_rec[wm] = rng.normal(0, 0.05, int(wm.sum()))

    r = _p1(err_rec, err_zf, wm)
    assert r["stake_zf_inside_band_all_lags"] is False, r
    assert r["verdict"] == "insufficient", r
    print("PASS test_p1_insufficient_when_the_zf_stake_itself_fails")


def test_p3_true_for_an_injected_half_voxel_displacement():
    from diag_artefact.metrics import half_voxel_shift
    gt = _phantom_gt()
    b = brain_mask(gt)
    wm = wm_mask(gt, b)

    shifted = half_voxel_shift(gt.astype(np.float64), axis=0)
    err_rec = shifted - gt.astype(np.float64)  # exactly what a real sub-voxel shift looks like

    r = _p3(err_rec, gt, wm)
    assert r["verdict"] == "true", r
    assert r["axes"]["axis0"]["significant"] is True, r["axes"]
    print("PASS test_p3_true_for_an_injected_half_voxel_displacement")


def test_p3_false_for_independent_noise():
    gt = _phantom_gt()
    b = brain_mask(gt)
    wm = wm_mask(gt, b)
    rng = np.random.default_rng(10)
    err_rec = np.zeros_like(gt, dtype=np.float64)
    err_rec[wm] = rng.normal(0, 0.05, int(wm.sum()))

    r = _p3(err_rec, gt, wm)
    assert r["verdict"] == "false", r
    assert all(not ax["significant"] for ax in r["axes"].values()), r["axes"]
    print("PASS test_p3_false_for_independent_noise")


def _write_t1_npz(path, rec, gt, zf):
    np.savez(path, recT1=rec.astype(np.float32), gtT1=gt.astype(np.float32),
             zfT1=zf.astype(np.float32), recFL=rec.astype(np.float32),
             gtFL=gt.astype(np.float32), zfFL=zf.astype(np.float32))


def test_p2_true_when_ctrlb_error_pattern_is_independent_of_shipped(tmp_path):
    gt = _phantom_gt()
    b = brain_mask(gt)
    rng = np.random.default_rng(12)
    rec_shipped = gt + np.where(b, rng.normal(0, 0.05, gt.shape), 0.0).astype(np.float32)

    # control B: a genuinely DIFFERENT (independent) error pattern on the same anatomy
    rec_ctrlB = gt + np.where(b, rng.normal(0, 0.05, gt.shape), 0.0).astype(np.float32)
    ctrlB_path = tmp_path / "ctrlB_volumes.npz"
    _write_t1_npz(ctrlB_path, rec_ctrlB, gt, gt)

    r = _p2(rec_shipped, gt, b, "T1", str(ctrlB_path))
    assert r["low"] is True, r
    assert r["verdict"] == "true", r
    print("PASS test_p2_true_when_ctrlb_error_pattern_is_independent_of_shipped")


def test_p2_false_when_ctrlb_error_pattern_matches_shipped(tmp_path):
    gt = _phantom_gt()
    b = brain_mask(gt)
    rng = np.random.default_rng(13)
    rec_shipped = gt + np.where(b, rng.normal(0, 0.05, gt.shape), 0.0).astype(np.float32)

    # control B: the SAME error pattern (as if the mask instance made no difference)
    ctrlB_path = tmp_path / "ctrlB_volumes.npz"
    _write_t1_npz(ctrlB_path, rec_shipped, gt, gt)

    r = _p2(rec_shipped, gt, b, "T1", str(ctrlB_path))
    assert r["low"] is False, r
    assert r["verdict"] == "false", r
    print("PASS test_p2_false_when_ctrlb_error_pattern_matches_shipped")


def test_p2_insufficient_when_ctrlb_is_not_available():
    gt = _phantom_gt()
    b = brain_mask(gt)
    r = _p2(gt, gt, b, "T1", None)
    assert r["verdict"] == "insufficient", r
    print("PASS test_p2_insufficient_when_ctrlb_is_not_available")


def test_patchiness_attribution_agrees_when_both_lambda_arms_move_the_same_direction(tmp_path):
    gt = _phantom_gt()
    b = brain_mask(gt)
    wm = wm_mask(gt, b)
    rng = np.random.default_rng(14)

    # shipped: a selective-suppression-like recon (flatten small gradients hard)
    def _selective(gt, strength, seed):
        from scipy import ndimage as ndi
        rr = np.random.default_rng(seed)
        g = np.abs(np.gradient(gt, axis=0))
        soft = ndi.gaussian_filter(gt, strength)
        out = np.where(g < np.quantile(g, 0.7), soft, gt)
        return (out + rr.normal(0, 0.001, gt.shape)).astype(np.float32)

    rec_shipped = _selective(gt, 3.0, 1)
    rec_lam010 = _selective(gt, 0.3, 2)   # much less selective suppression -> lower selectivity
    rec_lam060 = _selective(gt, 0.5, 3)   # still less than shipped -> agrees in direction

    p010 = tmp_path / "ctrlA_lam010_volumes.npz"
    p060 = tmp_path / "ctrlA_lam060_volumes.npz"
    _write_t1_npz(p010, rec_lam010, gt, gt)
    _write_t1_npz(p060, rec_lam060, gt, gt)

    out = _patchiness(rec_shipped, gt, wm, "T1", str(p010), str(p060),
                      rc_lam000_path="", zf=gt)
    assert out["agree"] is True, out
    assert out["attribution"] != "control A undecided", out
    print("PASS test_patchiness_attribution_agrees_when_both_lambda_arms_move_the_same_direction")


def test_patchiness_attribution_undecided_when_lambda_arms_disagree(tmp_path):
    gt = _phantom_gt()
    b = brain_mask(gt)
    wm = wm_mask(gt, b)

    def _selective(gt, strength, seed):
        from scipy import ndimage as ndi
        rr = np.random.default_rng(seed)
        g = np.abs(np.gradient(gt, axis=0))
        soft = ndi.gaussian_filter(gt, strength)
        out = np.where(g < np.quantile(g, 0.7), soft, gt)
        return (out + rr.normal(0, 0.001, gt.shape)).astype(np.float32)

    rec_shipped = _selective(gt, 2.0, 1)
    rec_lam010 = _selective(gt, 0.3, 2)   # less selective than shipped
    rec_lam060 = _selective(gt, 4.0, 3)   # MORE selective than shipped -- disagreement

    p010 = tmp_path / "ctrlA_lam010_volumes.npz"
    p060 = tmp_path / "ctrlA_lam060_volumes.npz"
    _write_t1_npz(p010, rec_lam010, gt, gt)
    _write_t1_npz(p060, rec_lam060, gt, gt)

    out = _patchiness(rec_shipped, gt, wm, "T1", str(p010), str(p060),
                      rc_lam000_path="", zf=gt)
    assert out["agree"] is False, out
    assert out["attribution"] == "control A undecided", out
    print("PASS test_patchiness_attribution_undecided_when_lambda_arms_disagree")


def test_patchiness_undecided_when_controls_are_missing():
    gt = _phantom_gt()
    b = brain_mask(gt)
    wm = wm_mask(gt, b)
    out = _patchiness(gt, gt, wm, "T1", None, None, rc_lam000_path="", zf=gt)
    assert out["attribution"] == "control A undecided", out
    print("PASS test_patchiness_undecided_when_controls_are_missing")


def test_run_all_wires_everything_and_adjudicate_consumes_it_without_error(tmp_path):
    gt = _phantom_gt()
    b = brain_mask(gt)
    rng = np.random.default_rng(15)
    rec = gt + np.where(b, rng.normal(0, 0.05, gt.shape), 0.0).astype(np.float32)
    zf = gt + np.where(b, rng.normal(0, 0.10, gt.shape), 0.0).astype(np.float32)

    npz_path = tmp_path / "shipped_volumes.npz"
    _write_t1_npz(npz_path, rec, gt, zf)

    # a tiny synthetic .mat mask so P4 does not depend on the real amax data mount
    from scipy import io as sio
    mask_path = tmp_path / "mask.mat"
    mask3d = (rng.random((40, 40, 8)) < 0.3).astype(np.float32)
    sio.savemat(str(mask_path), {"mask": mask3d})

    readings = run_all(str(npz_path), "T1", {"mask_path": str(mask_path)})
    for key in ("P1", "P2", "P3", "P4", "patchiness"):
        assert key in readings, readings.keys()
    assert readings["P2"]["verdict"] == "insufficient"  # no ctrlB given
    assert readings["patchiness"]["attribution"] == "control A undecided"  # no control A given

    verdicts = adjudicate(readings)
    for key in ("P1", "P2", "P3", "P4"):
        assert verdicts[key] in ("true", "false", "insufficient"), verdicts
    print("PASS test_run_all_wires_everything_and_adjudicate_consumes_it_without_error")


def test_resolve_checkpoint_falls_back_when_m0_did_not_resolve(tmp_path):
    m0 = tmp_path / "m0.json"
    m0.write_text('{"resolved": null}')
    assert _resolve_checkpoint(str(m0)) == "results_arc/tvs4k_e040_volumes.npz"
    print("PASS test_resolve_checkpoint_falls_back_when_m0_did_not_resolve")


def test_resolve_checkpoint_uses_the_resolved_candidate_when_m0_did_resolve(tmp_path):
    m0 = tmp_path / "m0.json"
    m0.write_text('{"resolved": "results_arc/some_other_volumes.npz"}')
    assert _resolve_checkpoint(str(m0)) == "results_arc/some_other_volumes.npz"
    print("PASS test_resolve_checkpoint_uses_the_resolved_candidate_when_m0_did_resolve")


def test_controls_from_manifest_never_cites_the_vacuous_or_not_run_entries(tmp_path):
    """The manifest's own `ctrlA_lam010` key is the ALL-ONES completed_vacuous run -- task-9-
    brief.md point 3 is explicit that it must not be cited for anything. `_controls_from_manifest`
    must map its OWN `controls["ctrlA_lam010"]` slot to the genuine AF=2 run instead, and must
    leave a not_run entry out entirely rather than pointing at a nonexistent file.
    """
    manifest = {
        "runs": {
            "ctrlA_lam010": {"npz": "results_arc/ctrlA_lam010_volumes.npz", "exists": True,
                             "status": "completed_vacuous"},
            "ctrlA_lam120": {"npz": "results_arc/ctrlA_lam120_volumes.npz", "exists": False,
                             "status": "not_run"},
            "ctrlB_seed1": {"npz": "results_arc/ctrlB_seed1_volumes.npz", "exists": True,
                            "status": "completed_genuine"},
            "ctrlA_af2_lam010": {"npz": "results_arc/ctrlA_af2_lam010_volumes.npz", "exists": True,
                                 "status": "completed_genuine"},
            "ctrlA_af2_lam060": {"npz": "results_arc/ctrlA_af2_lam060_volumes.npz", "exists": True,
                                 "status": "completed_genuine"},
        }
    }
    p = tmp_path / "manifest.json"
    p.write_text(__import__("json").dumps(manifest))

    controls = _controls_from_manifest(str(p))
    assert controls["ctrlB"] == "results_arc/ctrlB_seed1_volumes.npz", controls
    assert controls["ctrlA_lam010"] == "results_arc/ctrlA_af2_lam010_volumes.npz", controls
    assert controls["ctrlA_lam060"] == "results_arc/ctrlA_af2_lam060_volumes.npz", controls
    assert "ctrlA_lam010_volumes.npz" != controls["ctrlA_lam010"]  # the vacuous npz, by name, must not appear
    print("PASS test_controls_from_manifest_never_cites_the_vacuous_or_not_run_entries")


def test_controls_from_manifest_returns_none_for_a_run_that_is_not_completed_genuine(tmp_path):
    manifest = {"runs": {"ctrlB_seed1": {"npz": "x.npz", "exists": True, "status": "not_run"}}}
    p = tmp_path / "manifest.json"
    p.write_text(__import__("json").dumps(manifest))
    controls = _controls_from_manifest(str(p))
    assert controls["ctrlB"] is None, controls
    print("PASS test_controls_from_manifest_returns_none_for_a_run_that_is_not_completed_genuine")


if __name__ == "__main__":
    import tempfile
    test_permutation_null_ncc_centers_near_zero_for_an_unrelated_reference()
    test_p1_all_lags_false_positive_rate_matches_the_independently_measured_value()
    test_p1_true_when_residual_is_incoherent_and_shares_zfs_family()
    test_p1_false_when_residual_has_structure_unrelated_to_zf()
    test_p1_insufficient_when_the_zf_stake_itself_fails()
    test_p3_true_for_an_injected_half_voxel_displacement()
    test_p3_false_for_independent_noise()
    with tempfile.TemporaryDirectory() as td:
        test_p2_true_when_ctrlb_error_pattern_is_independent_of_shipped(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_p2_false_when_ctrlb_error_pattern_matches_shipped(Path(td))
    test_p2_insufficient_when_ctrlb_is_not_available()
    with tempfile.TemporaryDirectory() as td:
        test_patchiness_attribution_agrees_when_both_lambda_arms_move_the_same_direction(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_patchiness_attribution_undecided_when_lambda_arms_disagree(Path(td))
    test_patchiness_undecided_when_controls_are_missing()
    with tempfile.TemporaryDirectory() as td:
        test_run_all_wires_everything_and_adjudicate_consumes_it_without_error(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_resolve_checkpoint_falls_back_when_m0_did_not_resolve(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_resolve_checkpoint_uses_the_resolved_candidate_when_m0_did_resolve(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_controls_from_manifest_never_cites_the_vacuous_or_not_run_entries(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_controls_from_manifest_returns_none_for_a_run_that_is_not_completed_genuine(Path(td))
    print("ALL PASS")
