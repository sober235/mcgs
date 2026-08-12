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
    _p4,
    _patchiness,
    _permutation_null_ncc,
    _resolve_checkpoint,
    _figure_verdict_lines,
    adjudicate,
    adjudication_figure,
    main,
    run_all,
    spec_literal_verdicts,
)
from diag_artefact.volumes import brain_mask, wm_mask
from tests.diag_artefact.conftest import make_phantom as _phantom_gt


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


def test_p3_pre_registered_rule_detects_an_injected_half_voxel_displacement():
    """POSITIVE CONTROL for the rule spec section 2 actually registers for P3 -- "non-zero on at
    least one axis beyond what the half-pixel-shift positive control calibrates as detectable" --
    which `spec_literal_verdict` applies (final review I2). Feed in exactly a half-voxel shift
    residual and that rule must read `true`, with the shifted axis's ratio at 1.0 by construction
    (it IS the calibration) and the other two axes far below it.

    The two NULLS both abstain here, and that is recorded rather than asserted away:
      - the permutation null fails its exchangeability stake (a displacement residual is smooth and
        structured, exactly as on the real data), so the implemented rule returns "insufficient";
      - the circular-shift null cannot certify it either, because WM is a sparse, spatially
        concentrated mask (see `_circular_shift_null_ncc`'s docstring: the median overlap of the WM
        mask with a randomly-rolled copy of itself is zero voxels on the real volume), so the
        conservative variant is wide.
    That is the point of keeping this test: the pre-registered rule answers a question both
    substituted nulls lose, on a case whose ground truth is known.
    """
    from diag_artefact.metrics import half_voxel_shift
    gt = _phantom_gt()
    b = brain_mask(gt)
    wm = wm_mask(gt, b)

    shifted = half_voxel_shift(gt.astype(np.float64), axis=0)
    err_rec = shifted - gt.astype(np.float64)  # exactly what a real sub-voxel shift looks like

    r = _p3(err_rec, gt, wm)
    assert r["spec_literal_verdict"] == "true", r["spec_literal_basis"]
    ratios = r["spec_literal_basis"]["real_over_shift_ratio_per_axis"]
    assert abs(ratios["axis0"] - 1.0) < 1e-9, ratios      # the shifted axis IS the calibration
    assert ratios["axis1"] < 0.5 and ratios["axis2"] < 0.5, ratios
    assert r["permutation_null_valid"] is False, r
    assert r["verdict"] == "insufficient", r
    print("PASS test_p3_pre_registered_rule_detects_an_injected_half_voxel_displacement")


def test_p3_false_for_independent_noise_when_the_exchangeability_stake_passes():
    """Seed choice is not incidental (same reasoning as the P1 seed-8 test above): whether a given
    i.i.d.-noise realisation satisfies P1's own all-lags exchangeability check is itself a near
    coin flip (see `_p1_all_lags_false_positive_rate`). Seed 0 is one where it passes, which is
    what this test needs to isolate: the `false` branch reachable when the stake is valid, not the
    stake's own pass rate.
    """
    gt = _phantom_gt()
    b = brain_mask(gt)
    wm = wm_mask(gt, b)
    rng = np.random.default_rng(0)
    err_rec = np.zeros_like(gt, dtype=np.float64)
    err_rec[wm] = rng.normal(0, 0.05, int(wm.sum()))

    r = _p3(err_rec, gt, wm)
    assert r["permutation_null_valid"] is True, r
    assert r["verdict"] == "false", r
    assert all(not ax["significant_under_permutation_null"] for ax in r["axes"].values()), r["axes"]
    print("PASS test_p3_false_for_independent_noise_when_the_exchangeability_stake_passes")


def _write_t1_npz(path, rec, gt, zf):
    np.savez(path, recT1=rec.astype(np.float32), gtT1=gt.astype(np.float32),
             zfT1=zf.astype(np.float32), recFL=rec.astype(np.float32),
             gtFL=gt.astype(np.float32), zfFL=zf.astype(np.float32))


def test_p2_true_when_ctrlb_error_pattern_is_independent_of_shipped(tmp_path):
    gt = _phantom_gt()
    b = brain_mask(gt)
    rng = np.random.default_rng(12)
    rec_shipped = gt + np.where(b, rng.normal(0, 0.05, gt.shape), 0.0).astype(np.float32)
    zf_shipped = gt + np.where(b, rng.normal(0, 0.08, gt.shape), 0.0).astype(np.float32)

    # control B: a genuinely DIFFERENT (independent) error pattern on the same anatomy
    rec_ctrlB = gt + np.where(b, rng.normal(0, 0.05, gt.shape), 0.0).astype(np.float32)
    ctrlB_path = tmp_path / "ctrlB_volumes.npz"
    _write_t1_npz(ctrlB_path, rec_ctrlB, gt, gt)

    r = _p2(rec_shipped, gt, zf_shipped, b, "T1", str(ctrlB_path))
    assert r["low"] is True, r
    assert r["verdict"] == "true", r
    assert "anatomy_control" in r and "ncc_highpass_ctrlB_vs_shipped" in r["anatomy_control"], r
    assert "ncc_zf_vs_shipped" in r, r
    print("PASS test_p2_true_when_ctrlb_error_pattern_is_independent_of_shipped")


def test_p2_false_when_ctrlb_error_pattern_matches_shipped(tmp_path):
    gt = _phantom_gt()
    b = brain_mask(gt)
    rng = np.random.default_rng(13)
    rec_shipped = gt + np.where(b, rng.normal(0, 0.05, gt.shape), 0.0).astype(np.float32)
    zf_shipped = gt + np.where(b, rng.normal(0, 0.08, gt.shape), 0.0).astype(np.float32)

    # control B: the SAME error pattern (as if the mask instance made no difference)
    ctrlB_path = tmp_path / "ctrlB_volumes.npz"
    _write_t1_npz(ctrlB_path, rec_shipped, gt, gt)

    r = _p2(rec_shipped, gt, zf_shipped, b, "T1", str(ctrlB_path))
    assert r["low"] is False, r
    assert r["verdict"] == "false", r
    print("PASS test_p2_false_when_ctrlb_error_pattern_matches_shipped")


def test_p2_insufficient_when_ctrlb_is_not_available():
    gt = _phantom_gt()
    b = brain_mask(gt)
    r = _p2(gt, gt, gt, b, "T1", None)
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

    p010 = tmp_path / "ctrlA_af2_lam010_volumes.npz"
    p060 = tmp_path / "ctrlA_af2_lam060_volumes.npz"
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

    p010 = tmp_path / "ctrlA_af2_lam010_volumes.npz"
    p060 = tmp_path / "ctrlA_af2_lam060_volumes.npz"
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


def test_patchiness_gauge_correction_can_flip_raw_agreement_into_disagreement(tmp_path):
    """Fix round 1 (C2.3, the metric-gauge trap): reproduces, on synthetic data with EMPIRICALLY
    verified parameters, exactly the pattern found on the real diagnosis -- both arms read a LOWER
    raw selectivity than shipped (the old code called this "agree"), but arm060's own zf-anchor is
    much lower than shipped's zf-anchor (a different acquisition, memory.md section 3's gauge
    trap), so once each arm is normalised by its OWN zf-anchor, arm060's normalised selectivity is
    HIGHER than shipped's while arm010's is still lower -- the arms disagree once correctly gauged,
    and `agree` must be False (raw_agree stays True, proving the flip is real, not a coincidence
    of some other code path).
    """
    from scipy import ndimage as ndi

    gt = _phantom_gt()
    b = brain_mask(gt)
    wm = wm_mask(gt, b)

    def _selective(gt, strength, seed):
        rr = np.random.default_rng(seed)
        g = np.abs(np.gradient(gt, axis=0))
        soft = ndi.gaussian_filter(gt, strength)
        out = np.where(g < np.quantile(g, 0.7), soft, gt)
        return (out + rr.normal(0, 0.001, gt.shape)).astype(np.float32)

    def _zf(gt, blur, sigma, seed):
        rr = np.random.default_rng(seed)
        base = ndi.gaussian_filter(gt, blur) if blur > 0 else gt
        return (base + rr.normal(0, sigma, gt.shape)).astype(np.float32)

    rec_shipped = _selective(gt, 2.0, 1)
    zf_shipped = _zf(gt, 0.0, 0.05, 100)          # shipped's own zf-anchor -- reads ~0.98
    rec010 = _selective(gt, 0.5, 2)               # raw selectivity clearly below shipped
    zf010 = _zf(gt, 0.0, 0.05, 101)               # same acquisition family as shipped -- ~0.92
    rec060 = _selective(gt, 0.7, 3)               # raw selectivity SLIGHTLY below shipped too
    zf060 = _zf(gt, 1.0, 0.02, 102)               # a genuinely different acquisition -- ~0.91,
    #                                                enough lower than shipped's to flip the sign
    #                                                once rec060/zf060 is compared against rec_shipped/zf_shipped

    p010 = tmp_path / "ctrlA_af2_lam010_volumes.npz"
    p060 = tmp_path / "ctrlA_af2_lam060_volumes.npz"
    _write_t1_npz(p010, rec010, gt, zf010)
    _write_t1_npz(p060, rec060, gt, zf060)

    out = _patchiness(rec_shipped, gt, wm, "T1", str(p010), str(p060),
                      rc_lam000_path="", zf=zf_shipped)

    assert out["raw_selectivity_delta_lam010_vs_shipped"] < 0, out
    assert out["raw_selectivity_delta_lam060_vs_shipped"] < 0, out
    assert out["raw_agree_UNCORRECTED_do_not_use_for_attribution"] is True, out  # the trap: LOOKS like agreement

    assert out["gauge_normalized_delta_lam010_vs_shipped"] < 0, out
    assert out["gauge_normalized_delta_lam060_vs_shipped"] > 0, out             # flips sign once gauge-corrected
    assert out["agree"] is False, out
    assert out["attribution"] == "control A undecided", out
    print("PASS test_patchiness_gauge_correction_can_flip_raw_agreement_into_disagreement")


def _selective_recon(gt, strength, seed):
    """A reconstruction that flattens small gradients and keeps large ones -- the selective
    suppression memory section 14b names, i.e. the patchiness these metrics are meant to see."""
    from scipy import ndimage as ndi
    rr = np.random.default_rng(seed)
    g = np.abs(np.gradient(gt, axis=0))
    soft = ndi.gaussian_filter(gt, strength)
    out = np.where(g < np.quantile(g, 0.7), soft, gt)
    return (out + rr.normal(0, 0.001, gt.shape)).astype(np.float32)


def _synthetic_mask_mat(tmp_path, rng, shape=(40, 40, 8), density=0.3):
    from scipy import io as sio
    mask_path = tmp_path / "mask.mat"
    sio.savemat(str(mask_path), {"mask": (rng.random(shape) < density).astype(np.float32)})
    return str(mask_path)


def test_p4_reports_the_air_curve_its_stakes_and_a_reason_naming_the_blocking_clause(tmp_path):
    """`_p4` had no unit test at all before the final review (I7) -- only `adjudicate` was tested,
    against hand-written readings. Its own wiring (air curve, both stakes, the PSF sweep, and the
    reason string that must name the clause actually blocking the verdict) was never exercised.
    """
    gt = _phantom_gt()
    b = brain_mask(gt)
    rng = np.random.default_rng(21)
    rec = gt + np.where(b, rng.normal(0, 0.02, gt.shape), 0.0).astype(np.float32)
    zf = gt + rng.normal(0, 0.05, gt.shape).astype(np.float32)  # noisy everywhere, air included

    r = _p4(rec, gt, zf, b, _synthetic_mask_mat(tmp_path, rng))

    assert len(r["air_curve"]) == 4 and len(r["stake_air_curve_zf"]) == 4, r["air_curve"]
    n_vox = [row["n_voxels"] for row in r["air_curve"]]
    assert n_vox == sorted(n_vox), f"the air threshold sweep must be monotone in voxel count: {n_vox}"
    assert len(set(n_vox)) > 1, f"the sweep must actually sweep, not repeat one voxel set: {n_vox}"

    # stake 1: zf (pure noise, no prior) must read above the reconstruction at every threshold
    assert r["stake_gate"]["air_curve_zf_reads_above_rec_at_every_threshold"] is True, r["stake_gate"]
    # stake 2: the uniform-mask control must be more discrete than the random one
    assert r["stake_gate"]["psf_uniform_control_more_discrete_than_shipped"] is True, r["stake_gate"]
    assert r["stake_gate"]["passed"] is True, r["stake_gate"]

    assert set(r["psf_radius_sweep"]) == {2, 3, 4, 5, 6}, r["psf_radius_sweep"]
    assert "PSF term NEVER ENTERS" in r["reason"], r["reason"]
    assert r["spec_literal_verdict"] in ("true", "false"), r
    print("PASS test_p4_reports_the_air_curve_its_stakes_and_a_reason_naming_the_blocking_clause")


def test_every_recorded_stake_can_change_an_outcome(tmp_path):
    """META-TEST for the defect class the final review found (I1), rather than for its instances:
    a `stake_*` reading that is computed, recorded, and then never consulted by anything.

    For each stake, the same proposition is evaluated twice -- once as measured, once with the
    stake driven to its failing value -- and the OUTCOME must differ. A stake that cannot change
    an outcome is decoration, however prominently it is reported.

    P1's zf stake and P3's exchangeability stake have their own dedicated tests above
    (`test_p1_insufficient_when_the_zf_stake_itself_fails`,
    `test_p3_pre_registered_rule_detects_an_injected_half_voxel_displacement`); the three covered
    here are the ones that were purely decorative before this review.
    """
    from scipy import ndimage as ndi

    gt = _phantom_gt()
    b = brain_mask(gt)
    wm = wm_mask(gt, b)
    rng = np.random.default_rng(31)

    # ---- P4's stake gate: passing -> the pre-registered rule decides; failing -> insufficient
    readings_true = {"P4": {"air_curve": [{"threshold": 0.01, "ratio_rec": 5.0, "n_voxels": 10}],
                            "psf_peak_sidelobe": 0.05,
                            "stake_gate": {"passed": True}}}
    assert adjudicate(readings_true)["P4"] == "true", adjudicate(readings_true)
    readings_stake_down = {"P4": {**readings_true["P4"], "stake_gate": {"passed": False}}}
    assert adjudicate(readings_stake_down)["P4"] == "insufficient", adjudicate(readings_stake_down)

    # ---- P2's zf-reference stake. Shipped and ctrlB share an error pattern (-> "false"), but if
    # the shipped mask's OWN zero-fill error resembles the shipped error even more strongly, the
    # comparison no longer supports "indifferent to the mask instance".
    shared = np.where(b, ndi.gaussian_filter(rng.normal(0, 1, gt.shape), 1.5), 0.0)
    rec_shipped = (gt + 0.05 * shared).astype(np.float32)
    rec_ctrlB = (gt + 0.05 * shared + np.where(b, rng.normal(0, 0.01, gt.shape), 0.0)).astype(np.float32)
    zf_weak = (gt + np.where(b, rng.normal(0, 0.08, gt.shape), 0.0)).astype(np.float32)
    zf_strong = (gt + 0.05 * shared * 1.001).astype(np.float32)  # nearly the shipped error itself
    ctrlB_path = tmp_path / "ctrlB_volumes.npz"
    _write_t1_npz(ctrlB_path, rec_ctrlB, gt, gt)

    passing = _p2(rec_shipped, gt, zf_weak, b, "T1", str(ctrlB_path))
    failing = _p2(rec_shipped, gt, zf_strong, b, "T1", str(ctrlB_path))
    assert passing["stake_exceeds_zf_reference"] is True, passing
    assert passing["verdict"] == "false", passing
    assert failing["stake_exceeds_zf_reference"] is False, failing
    assert failing["verdict"] == "insufficient", failing

    # ---- P2's anatomy-control stake: a correlation carried ENTIRELY by the shared low-frequency
    # envelope must not be allowed to read as "indifferent to the mask instance".
    envelope = ndi.gaussian_filter(np.where(b, 1.0, 0.0), 8.0)
    rec_env = (gt + 0.05 * envelope + np.where(b, rng.normal(0, 0.01, gt.shape), 0.0)).astype(np.float32)
    ctrlB_env = (gt + 0.05 * envelope + np.where(b, rng.normal(0, 0.01, gt.shape), 0.0)).astype(np.float32)
    env_path = tmp_path / "ctrlB_env_volumes.npz"
    _write_t1_npz(env_path, ctrlB_env, gt, gt)
    envelope_only = _p2(rec_env, gt, zf_weak, b, "T1", str(env_path))
    assert envelope_only["anatomy_control"]["stake_highpass_clears_null"] is False, envelope_only
    assert envelope_only["verdict"] == "insufficient", envelope_only

    # ---- M5's stake in _patchiness: an rc_lam000 control that reads MORE patchy than the run
    # under test disqualifies the scalar, so no attribution may be asserted from it -- even when
    # the two lambda arms agree, which is exactly the real-data situation.
    rec_shipped_sel = _selective_recon(gt, 3.0, 1)
    p010 = tmp_path / "ctrlA_af2_lam010_volumes.npz"
    p060 = tmp_path / "ctrlA_af2_lam060_volumes.npz"
    _write_t1_npz(p010, _selective_recon(gt, 0.3, 2), gt, gt)
    _write_t1_npz(p060, _selective_recon(gt, 0.5, 3), gt, gt)
    no_stake = _patchiness(rec_shipped_sel, gt, wm, "T1", str(p010), str(p060),
                           rc_lam000_path="", zf=gt)
    assert no_stake["agree"] is True and no_stake["attribution"] != "control A undecided", no_stake

    rc_path = tmp_path / "rc_lam000_volumes.npz"
    _write_t1_npz(rc_path, _selective_recon(gt, 6.0, 4), gt, gt)  # MORE selective than shipped
    with_failing_stake = _patchiness(rec_shipped_sel, gt, wm, "T1", str(p010), str(p060),
                                     rc_lam000_path=str(rc_path), zf=gt)
    assert with_failing_stake["stake_rc_lam000_selectivity_discriminates"] is False, with_failing_stake
    assert with_failing_stake["agree"] is True, "the sign rule itself must be untouched"
    assert with_failing_stake["attribution"] == "control A undecided", with_failing_stake
    print("PASS test_every_recorded_stake_can_change_an_outcome")


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
    brief.md point 3 is explicit that it must not be cited for anything. Fix round 1 (I2) renamed
    the RETURNED dict's key from `ctrlA_lam010` to `ctrlA_af2_lam010` specifically so it can never
    be misread as citing the manifest's vacuous entry just because the two used to share a name;
    `_controls_from_manifest` must map `controls["ctrlA_af2_lam010"]` to the genuine AF=2 run, and
    must leave a not_run entry out entirely rather than pointing at a nonexistent file.
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
    assert controls["ctrlA_af2_lam010"] == "results_arc/ctrlA_af2_lam010_volumes.npz", controls
    assert controls["ctrlA_af2_lam060"] == "results_arc/ctrlA_af2_lam060_volumes.npz", controls
    assert "ctrlA_lam010" not in controls  # the OLD, ambiguous key must not exist at all
    assert "results_arc/ctrlA_lam010_volumes.npz" not in controls.values()  # the vacuous npz must not appear
    print("PASS test_controls_from_manifest_never_cites_the_vacuous_or_not_run_entries")


def test_controls_from_manifest_returns_none_for_a_run_that_is_not_completed_genuine(tmp_path):
    manifest = {"runs": {"ctrlB_seed1": {"npz": "x.npz", "exists": True, "status": "not_run"}}}
    p = tmp_path / "manifest.json"
    p.write_text(__import__("json").dumps(manifest))
    controls = _controls_from_manifest(str(p))
    assert controls["ctrlB"] is None, controls
    print("PASS test_controls_from_manifest_returns_none_for_a_run_that_is_not_completed_genuine")


def test_adjudication_figure_stamps_the_verdicts_on_the_panel(tmp_path):
    """The figure carried no outcome at all before the final review (I6) -- three images and a
    title. Assert the stamp exists and names every proposition, including the spec-literal reading
    where it differs from the implemented one."""
    gt = _phantom_gt(shape=(64, 60, 200))
    rng = np.random.default_rng(41)
    vols = {"gt": gt,
            "rec": gt + rng.normal(0, 0.02, gt.shape).astype(np.float32),
            "zf": gt + rng.normal(0, 0.05, gt.shape).astype(np.float32)}
    readings = {
        "P1": {"stake_zf_autocorr_profile": [1.0, 0.4], "ncc_rec_zf": 0.45,
               "spec_literal_verdict": "false"},
        "P2": {"ncc_ctrlB_vs_shipped": 0.39, "gating_null": "circular_shift",
               "gating_null_hi97_5": 0.08, "spec_literal_verdict": "false"},
        "P3": {"spec_literal_basis": {"max_real_over_shift_ratio": 0.016},
               "spec_literal_verdict": "false"},
        "P4": {"air_curve": [{"threshold": 0.005, "ratio_rec": 2.9, "n_voxels": 10},
                             {"threshold": 0.04, "ratio_rec": 0.48, "n_voxels": 20}],
               "spec_literal_verdict": "false"},
    }
    verdicts = {"P1": "insufficient", "P2": "false", "P3": "insufficient", "P4": "insufficient"}

    lines = _figure_verdict_lines(readings, verdicts)
    assert len(lines) == 4, lines
    assert any("spec-literal rule: FALSE" in line for line in lines), lines
    assert all(any(key in line for line in lines) for key in ("P1", "P2", "P3", "P4")), lines

    out = tmp_path / "adjudication.png"
    path = adjudication_figure(vols, {}, str(out), verdicts=verdicts, readings=readings)
    assert Path(path).exists() and Path(path).stat().st_size > 10_000, path
    print("PASS test_adjudication_figure_stamps_the_verdicts_on_the_panel")


def test_main_runs_end_to_end_and_writes_both_artefacts(tmp_path, monkeypatch):
    """`main()` -- the committed driver that produces verdicts.json and the figure -- had no test
    (final review I7). Everything it wires (M0 resolution, the manifest gate, run_all, adjudicate,
    the spec-literal block, the JSON header, the figure) is exercised here on synthetic volumes.
    """
    import diag_artefact.report as report_mod

    gt = _phantom_gt(shape=(64, 60, 200))
    b = brain_mask(gt)
    rng = np.random.default_rng(42)
    rec = gt + np.where(b, rng.normal(0, 0.05, gt.shape), 0.0).astype(np.float32)
    zf = gt + np.where(b, rng.normal(0, 0.10, gt.shape), 0.0).astype(np.float32)
    npz_path = tmp_path / "shipped_volumes.npz"
    _write_t1_npz(npz_path, rec, gt, zf)

    m0 = tmp_path / "m0.json"
    m0.write_text(__import__("json").dumps({"unique_match": True, "resolved": str(npz_path)}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(__import__("json").dumps({"runs": {}}))
    # do not reach for the real (multi-GB) rc_lam000 volumes from a unit test
    monkeypatch.setattr(report_mod, "DEFAULT_RC_LAM000", "")

    mask_path = _synthetic_mask_mat(tmp_path, rng)
    rc = main(["--contrast", "T1", "--m0-json", str(m0), "--manifest-json", str(manifest),
               "--out-dir", str(tmp_path / "out"), "--npz", str(npz_path)])
    assert rc == 0

    import json as _json
    report = _json.loads((tmp_path / "out" / "verdicts.json").read_text())
    assert report["generated_by"].startswith("diag_artefact/report.py"), report["generated_by"]
    assert len(report["commit"]) >= 7, report["commit"]
    assert set(report["verdicts"]) == {"P1", "P2", "P3", "P4"}, report["verdicts"]
    assert set(report["spec_literal_verdicts"]) == {"P1", "P2", "P3", "P4"}, report
    assert report["readings"]["provenance"]["checkpoint_used"] == str(npz_path)
    assert (tmp_path / "out" / "adjudication.png").exists()
    print("PASS test_main_runs_end_to_end_and_writes_both_artefacts")


def test_spec_literal_verdicts_are_reported_for_every_proposition():
    readings = {"P1": {"spec_literal_verdict": "false"}, "P2": {"spec_literal_verdict": "false"},
                "P3": {"spec_literal_verdict": "false"}, "P4": {"spec_literal_verdict": "false"},
                "patchiness": {}}
    got = spec_literal_verdicts(readings)
    assert got == {"P1": "false", "P2": "false", "P3": "false", "P4": "false"}, got
    # a proposition that never recorded one degrades to "insufficient", never to a guess
    assert spec_literal_verdicts({"P1": {}})["P1"] == "insufficient"
    print("PASS test_spec_literal_verdicts_are_reported_for_every_proposition")


if __name__ == "__main__":
    import tempfile
    test_permutation_null_ncc_centers_near_zero_for_an_unrelated_reference()
    test_p1_all_lags_false_positive_rate_matches_the_independently_measured_value()
    test_p1_true_when_residual_is_incoherent_and_shares_zfs_family()
    test_p1_false_when_residual_has_structure_unrelated_to_zf()
    test_p1_insufficient_when_the_zf_stake_itself_fails()
    test_p3_pre_registered_rule_detects_an_injected_half_voxel_displacement()
    test_p3_false_for_independent_noise_when_the_exchangeability_stake_passes()
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
        test_patchiness_gauge_correction_can_flip_raw_agreement_into_disagreement(Path(td))
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
    with tempfile.TemporaryDirectory() as td:
        test_p4_reports_the_air_curve_its_stakes_and_a_reason_naming_the_blocking_clause(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_every_recorded_stake_can_change_an_outcome(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_adjudication_figure_stamps_the_verdicts_on_the_panel(Path(td))
    test_spec_literal_verdicts_are_reported_for_every_proposition()
    print("ALL PASS  (test_main_runs_end_to_end_and_writes_both_artefacts needs pytest's "
          "monkeypatch fixture -- run it with pytest)")
