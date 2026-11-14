import json
import sys

import numpy as np
import pytest

import geom
import simulate
from common import SRC
from helpers import fft3c_np, small_src


def combine(ksp, csm):
    return (np.conj(csm) * simulate.ifft3c(ksp)).sum(0)


def test_zero_motion_is_bit_identical():
    src = small_src()
    out = simulate.build("C0", src, small=True)
    assert np.array_equal(out["ksp_T2"], src["ksp_T2"]) and "ksp_T1" not in out


def test_shutter_zero_set_identical():
    src = small_src()
    out = simulate.build("E1a", src, small=True)
    assert out["ksp_T2"].dtype == np.complex64 and out["ksp_T2"].shape == src["ksp_T2"].shape
    assert np.array_equal(np.all(out["ksp_T2"] == 0, axis=0), np.all(src["ksp_T2"] == 0, axis=0))
    assert not np.array_equal(out["ksp_T2"], src["ksp_T2"])


def test_integer_shift_matches_roll_of_coil_combined():
    src = small_src(shutter=False)
    out = simulate.build_custom(src, inter=geom.Rigid(np.eye(3), np.array([2.0, 0, 0])), small=True)
    m0, m1 = combine(src["ksp_T2"], src["csm_T2"]), combine(out["ksp_T2"], src["csm_T2"])
    np.testing.assert_allclose(m1, np.roll(m0, 2, 0), atol=2e-3 * np.abs(m0).max())


def test_fft_convention_matches_trainer():
    if not (SRC / "paired_gs/data/transforms.py").exists():
        pytest.skip("frozen trainer source not on this machine")
    import torch
    sys.path.insert(0, str(SRC / "paired_gs"))
    from data.transforms import fft3c, ifft3c
    rng = np.random.default_rng(0)
    x = (rng.normal(size=(4, 16, 16, 8)) + 1j * rng.normal(size=(4, 16, 16, 8))).astype(np.complex64)
    np.testing.assert_allclose(simulate.fft3c(x), fft3c(torch.from_numpy(x)).numpy(), atol=1e-5)
    np.testing.assert_allclose(simulate.ifft3c(x), ifft3c(torch.from_numpy(x)).numpy(), atol=1e-5)
    np.testing.assert_allclose(simulate.fft3c(x), fft3c_np(x), atol=1e-5)


def test_intra_lines_come_from_their_train_pose():
    src = small_src()
    out = simulate.build("E4a", src, small=True)
    sch = out["schedule"]
    assert np.array_equal(out["gt_ksp_T2"], src["ksp_T2"])               # no inter-scan motion: reference pose = input
    f = sch["train_time_frac"][np.clip(sch["train_of"], 0, None)]
    late = (sch["train_of"] >= 0) & (f >= 0.60)
    early = (sch["train_of"] >= 0) & (f < 0.60)
    assert sch["train_of"][out["ksp_T2"].shape[2] // 2, out["ksp_T2"].shape[3] // 2] == sch["ref_train"]
    assert np.array_equal(out["ksp_T2"][:, :, early], src["ksp_T2"][:, :, early])     # before the step: untouched
    assert not np.allclose(out["ksp_T2"][:, :, late], src["ksp_T2"][:, :, late])       # after it: another pose
    assert np.array_equal(np.all(out["ksp_T2"] == 0, axis=0), np.all(src["ksp_T2"] == 0, axis=0))


def test_intra_drift_matches_a_single_pose_simulation_on_one_train():
    src = small_src()
    out = simulate.build("E4b", src, small=True)
    sch, truth = out["schedule"], out["truth"]
    n = sch["n_trains"] - 1
    one = simulate.build_custom(src, inter=geom.Rigid.from_json(truth["train_poses"][n]), small=True)
    sel = sch["train_of"] == n
    np.testing.assert_allclose(out["ksp_T2"][:, :, sel], one["ksp_T2"][:, :, sel], atol=1e-6)
    assert len(truth["segment_mean_poses"]) == 8 and truth["ref_seg"] == sch["ref_seg"]


def test_e4c_reference_pose_is_the_inter_scan_pose():
    src = small_src()
    out = simulate.build("E4c", src, small=True)
    e1a = simulate.build("E1a", src, small=True)
    np.testing.assert_allclose(out["gt_ksp_T2"], e1a["ksp_T2"], atol=1e-6)
    sel = out["schedule"]["train_of"] == out["schedule"]["ref_train"]
    np.testing.assert_allclose(out["ksp_T2"][:, :, sel], e1a["ksp_T2"][:, :, sel], atol=1e-6)


def test_e3_lesions_in_both_contrasts_and_masks():
    src = small_src()
    out = simulate.build("E3", src, small=True)
    e2 = simulate.build("E2", src, small=True)
    assert "ksp_T1" in out and not np.array_equal(out["ksp_T1"], src["ksp_T1"]) and "ksp_T1" not in e2
    m_t1, m_t1_0 = np.abs(combine(out["ksp_T1"], src["csm_T1"])), np.abs(combine(src["ksp_T1"], src["csm_T1"]))
    lab = out["lesion_mask_canon"]
    assert m_t1[lab == 5].mean() < 0.85 * m_t1_0[lab == 5].mean()          # the T1-only lesion is dark on T1
    assert abs(m_t1[lab == 4].mean() / m_t1_0[lab == 4].mean() - 1) < 0.02  # a T2-only lesion is absent from T1
    m_t2, m_t2_0 = np.abs(combine(out["ksp_T2"], src["csm_T2"])), np.abs(combine(e2["ksp_T2"], src["csm_T2"]))
    lab2 = out["lesion_mask_T2"]
    assert m_t2[lab2 == 4].mean() > 1.3 * m_t2_0[lab2 == 4].mean()         # bright on T2, at its moved position
    assert abs(m_t2[lab2 == 5].mean() / m_t2_0[lab2 == 5].mean() - 1) < 0.03
    np.testing.assert_allclose(out["disp_T2"], e2["disp_T2"])              # same motion as E2


def test_truth_json_has_no_clock_fields():
    out = simulate.build("E4b", small_src(), small=True)
    text = json.dumps(out["truth"]).lower()
    assert not any(k in text for k in ("utc", "timestamp", "elapsed", "date", "seconds"))
    assert out["truth"]["inter"] is None and len(out["truth"]["train_poses"]) == out["schedule"]["n_trains"]


def test_qc_figures_show_the_brain_mask_and_every_lesion_site(tmp_path):
    src = small_src()
    out = simulate.build("E3", src, small=True)
    simulate._qc_png(tmp_path / "qc_E3.png", "E3", out, src)
    assert (tmp_path / "qc_E3.png").stat().st_size > 20000
    assert (tmp_path / "qc_E3_lesions.png").stat().st_size > 20000
    out = simulate.build("E1a", src, small=True)
    simulate._qc_png(tmp_path / "qc_E1a.png", "E1a", out, src)
    assert (tmp_path / "qc_E1a.png").exists() and not (tmp_path / "qc_E1a_lesions.png").exists()
