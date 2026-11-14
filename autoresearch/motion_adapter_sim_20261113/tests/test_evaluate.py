import numpy as np
import pytest

import evaluate
import geom
from common import NATIVE_SHIFT_MM
from helpers import ball_mask

SHAPE = (64, 64, 48)


def fake_table(t2, af="af12", t1=None):
    """{(cond, arm): T2 head PSNR} -> table rows in the format of evaluate.image_table."""
    t1 = t1 or {}
    return [dict(af=af, cond=c, arm=a, seed=42, exp=f"ma_{af}_{c}_{a}",
                 T1=dict(head_psnr=t1[(c, a)], head_psnr_eval=t1[(c, a)]) if (c, a) in t1 else None,
                 T2=dict(head_psnr=v, head_psnr_eval=v)) for (c, a), v in t2.items()]


def test_normalised_delta():
    t = fake_table({("C0", "A2"): 36.30, ("E2", "A2"): 36.25, ("C0", "I-T2"): 36.33, ("E2", "I-T2"): 36.31})
    assert evaluate.normalised_delta(t, "af12", "E2", "A2")["T2"] == pytest.approx(-0.03, abs=1e-6)


def test_normalised_delta_uses_a0_at_c0_when_the_arm_has_no_c0_run():
    t = fake_table({("C0", "A0"): 36.30, ("E2", "O"): 36.20, ("C0", "I-T2"): 36.33, ("E2", "I-T2"): 36.33})
    d = evaluate.normalised_delta(t, "af12", "E2", "O")
    assert d["T2"] == pytest.approx(-0.10, abs=1e-6) and any("A0" in n for n in d["notes"])


def test_normalised_delta_t1_is_unnormalised_without_i_t1():
    t = fake_table({("C0", "A2"): 36.30, ("E2", "A2"): 36.25, ("C0", "I-T2"): 36.33, ("E2", "I-T2"): 36.31},
                   t1={("C0", "A2"): 35.80, ("E2", "A2"): 35.78})
    d = evaluate.normalised_delta(t, "af12", "E2", "A2")
    assert d["T1"] == pytest.approx(-0.02, abs=1e-6) and any("T1" in n for n in d["notes"])


def test_tre_zero_for_exact_estimate_and_differential():
    truth = geom.rigid_from_pivot((0.5, 0.5, 1.5), (0.8, -0.5, 0.3))
    native = geom.Rigid(np.eye(3), np.asarray(NATIVE_SHIFT_MM) + np.array([0.01, -0.02, 0.03]))
    est = truth.compose(native)
    r = evaluate.rigid_motion_errors(est, truth, c0_est=native, brain=ball_mask())
    assert r["tre_mean_mm_differential"] < 1e-6 and r["rot_err_deg"] < 1e-6
    assert 0.0 < r["tre_mean_mm_absolute"] < 0.1            # the assumed native shift differs by (0.01, -0.02, 0.03)


def test_tre_sees_a_wrong_rotation():
    truth = geom.rigid_from_pivot((0.5, 0.5, 1.5), (0.8, -0.5, 0.3))
    native = geom.Rigid.identity()
    wrong = geom.rigid_from_pivot((0.5, 0.5, 1.0), (0.8, -0.5, 0.3))
    r = evaluate.rigid_motion_errors(wrong, truth, c0_est=native, brain=ball_mask())
    assert r["rot_err_deg"] == pytest.approx(0.5, abs=1e-3) and r["tre_mean_mm_differential"] > 0.3


def test_est_field_equals_the_adapter_transform():
    torch = pytest.importorskip("torch")
    import adapter
    ad = adapter.MotionAdapter(SHAPE, rigid=True, nonrigid=True, n_segments=3, ref_segment=1)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        ad.theta.copy_(torch.tensor([0.01, -0.02, 0.015]))
        ad.t_mm.copy_(torch.tensor([0.7, -0.3, 0.2]))
        ad.ctrl.copy_(0.2 * torch.randn(ad.ctrl.shape, generator=g))
        ad.seg_theta.copy_(0.01 * torch.randn(3, 3, generator=g))
        ad.seg_t_mm.copy_(0.3 * torch.randn(3, 3, generator=g))
    st = ad.state()
    for seg in (None, 0, 1, 2):
        want = adapter.dense_displacement(ad, SHAPE, seg=seg)
        got = evaluate.est_field(st, SHAPE, seg=seg)
        assert np.abs(got - want).max() < 2e-4
    u = evaluate.est_field(st, SHAPE, part="nonrigid")
    assert np.abs(u).max() > 0.05 and np.abs(u.reshape(-1, 3).mean(0)).max() < 0.3


def test_segment_errors():
    est = [dict(theta_deg=[0.0, 0.0, 0.5], t_mm=[0.2, 0.0, 0.0]), dict(theta_deg=[0.0, 0.0, 0.0], t_mm=[0.0, 0.0, 0.0])]
    true_mean = [geom.Rigid(geom.rotmat(np.deg2rad([0.0, 0.0, 0.8])), np.array([0.2, 0.0, 0.0])), geom.Rigid.identity()]
    e = evaluate.segment_errors(est, true_mean, ref_seg=1, brain=ball_mask())
    assert e["seg_rot_err_deg"][0] == pytest.approx(0.3, abs=1e-3) and e["seg_t_err_mm"][0] < 1e-9
    assert e["seg_rot_err_deg_max"] == pytest.approx(0.3, abs=1e-3) and e["seg_tre_mean_mm"][1] < 1e-9


def lesion_phantom(gain=0.8, shape=SHAPE):
    gt = np.ones(shape, dtype=np.float32)
    idx = np.indices(shape).astype(np.float32)
    d = np.sqrt(sum((idx[k] - c) ** 2 for k, c in enumerate((30, 34, 22))))
    mask = d <= 4.0
    gt[mask] = 1.0 + gain
    return gt, mask


def test_contrast_recovery():
    gt, mask = lesion_phantom(gain=0.8)
    rec = gt.copy()
    rec[mask] = gt[mask] * (1.4 / 1.8)                      # the lesion keeps half of its contrast
    assert evaluate.contrast_recovery(rec, gt, mask) == pytest.approx(0.5, abs=0.05)
    assert evaluate.contrast_recovery(gt, gt, mask) == pytest.approx(1.0, abs=1e-6)


def test_centroid_shift():
    gt, mask = lesion_phantom(gain=0.8)
    rec = np.roll(gt, 1, axis=1)
    assert evaluate.centroid_shift_mm(rec, gt, mask) == pytest.approx(1.0, abs=0.15)
    assert evaluate.centroid_shift_mm(gt, gt, mask) < 1e-6


def fake_summary(s1_dpsnr=-0.01, af="af12"):
    return {af: dict(zero={"A2": dict(dpsnr_T1=-0.01, dpsnr_T2=s1_dpsnr, t_minus_native_mm=0.1, rot_deg=0.05,
                                      nonrigid_brain_max_mm=0.1, seg_max_deg=None, seg_max_mm=None)},
                     delta={}, motion={}, lesions={}, table=[])}


def test_criteria_thresholds():
    c = evaluate.criteria(fake_summary(s1_dpsnr=-0.06))
    assert c["S1"]["psnr"]["pass"] is False and c["S1"]["psnr"]["threshold"] == -0.05
    c = evaluate.criteria(fake_summary(s1_dpsnr=-0.04))
    assert c["S1"]["psnr"]["pass"] is True and c["S1"]["nonrigid"]["pass"] is True


def test_criteria_s4_recovered_fraction():
    s = fake_summary()
    s["af12"]["table"] = fake_table({("C0", "A0"): 36.3, ("E4b", "A0"): 35.3, ("E4b", "A3"): 35.9})
    s["af12"]["motion"] = {"ma_af12_E4b_A3": dict(seg_rot_err_deg_max=0.1, seg_t_err_mm_max=0.1)}
    c = evaluate.criteria(s)
    assert c["S4"]["E4b"]["recovered"]["value"] == pytest.approx(0.6, abs=1e-6) and c["S4"]["E4b"]["recovered"]["pass"] is True
    assert c["S4"]["E4b"]["segments"]["pass"] is True


def test_criteria_report_missing_inputs_instead_of_passing():
    c = evaluate.criteria(fake_summary())
    assert c["S2"]["E2"]["tre"]["pass"] is None and c["S3"]["pass"] is None


def test_tagged_development_rows_stay_out_of_the_main_table_values():
    half = dict(head_psnr=36.0, head_ssim=0.9)
    led = [dict(exp="ma_af12_C0_A2", af="af12", cond="C0", arm="A2", seed=42, plain=half, contrast=half),
           dict(exp="ma_af12_C0_A2_r1", af="af12", cond="C0", arm="A2", seed=42, tag="r1", plain=half,
                contrast=dict(half, head_psnr=30.0))]
    t = evaluate.image_table(led)
    assert [r["tag"] for r in t] == ["", "r1"]
    assert evaluate._val(t, "af12", "C0", "A2", "T2") == 36.0
    assert evaluate._val(t, "af12", "C0", "A2", "T2", tag="r1") == 30.0
