import hashlib
import json

import numpy as np
import pytest

import common as c1
import geom
import lesions
import msim
import schedule
import simulate
import warp
from helpers import small_src
from mcommon import CONDITIONS2, CONTAM_MM, ORDER_NAME, ORDERS
from test_order import ellipse_support

REF = {"E4a": "462f89c2e0b119db350cfc59fc64da53fe13be1be7c7d3b1652ad3563765f8e0",      # computed on the server
       "E4c": "df1a4a73c3f4bb2aabd6aefe21d47f44376cf67faa39ca4a7a2332c666b6c49f"}      # before simulate.py changed


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _changed_trains(out):
    diff = np.any(out["ksp_T2"] != out["gt_ksp_T2"], axis=(0, 1))                  # (ky, kz)
    return sorted(set(out["schedule"]["train_of"][diff].tolist()))


def test_defaults_reproduce_v1():
    for cond, want in REF.items():
        out = simulate.build(cond, small_src(), small=True)
        assert _sha(out["ksp_T2"]) == want, cond
        assert out["truth"]["order"] == "linear"


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("cond", ["Ta", "Tc"])
def test_reference_train_untouched(cond, order):
    out = msim.build(cond, order, small_src(), small=True)
    sch = out["schedule"]
    sel = sch["train_of"] == sch["ref_train"]
    assert sel.any() and np.array_equal(out["ksp_T2"][:, :, sel], out["gt_ksp_T2"][:, :, sel])
    assert out["qc"]["reference_train_equals_reference_pose"] and out["qc"]["shutter_zero_set_identical"]
    assert out["truth"]["order"] == ORDER_NAME[order]


def test_pulse_changes_only_its_trains():
    out = msim.build("Ta", "il", small_src(), small=True)
    t = out["truth"]
    frac = out["schedule"]["train_time_frac"]
    pulse = [n for n, f in enumerate(frac) if 0.40 <= f < 0.45]
    assert len(pulse) >= 1
    assert _changed_trains(out) == pulse
    assert np.flatnonzero(t["contaminated"]).tolist() == pulse
    assert len(t["train_brain_disp_mm"]) == t["n_trains"] and min(t["train_brain_disp_mm"]) == 0.0
    assert all(d > CONTAM_MM for n, d in enumerate(t["train_brain_disp_mm"]) if n in pulse)
    json.dumps(t)                                                                # the truth must serialise


def test_reference_train_outside_pulses():
    sup = ellipse_support()
    for cond, spec in CONDITIONS2.items():
        for order in ORDERS:
            sch = schedule.build(sup, order=ORDER_NAME[order])
            f = float(sch["train_time_frac"][sch["ref_train"]])
            for e in spec["intra"] or []:
                if e["kind"] == "pulse":
                    assert not (e["at"] <= f < e["at"] + e["dur"]), (cond, order, f, e)


def test_tb_residual_below_contamination_line():
    path = c1.DATA / "E1a/brain.npy"
    if not path.exists():
        pytest.skip("needs the brain mask of the first version (server)")
    brain = np.load(path)
    ev = CONDITIONS2["Tb"]["intra"]
    frac = np.arange(110) / 109.0
    in_pulse = [any(e["kind"] == "pulse" and e["at"] <= f < e["at"] + e["dur"] for e in ev) for f in frac]
    for ref in (5, 100):                                    # reference before the first pulse / after the last one
        rel = schedule.poses(ev, frac, ref)
        for n in (0, 50, 109):
            assert not in_pulse[n]
            assert simulate.brain_disp_mm(rel[n], brain) + 0.1 * 0.5 < CONTAM_MM, (ref, n)      # + soft bound
        assert simulate.brain_disp_mm(rel[int(np.argmax(in_pulse))], brain) > CONTAM_MM


def test_truth_states():
    out = msim.build("Tc", "il", small_src(), small=True)
    t, frac = out["truth"], out["schedule"]["train_time_frac"]
    st = np.asarray(t["state_of_train"])
    first = int(np.argmax(frac >= 0.60))
    assert sorted(set(st.tolist())) == [0, 1] and (st[:first] == 0).all() and (st[first:] == 1).all()
    assert t["ref_state"] == int(st[t["ref_train"]])
    poses = [geom.Rigid.from_json(p) for p in t["state_poses"]]
    assert np.allclose(poses[t["ref_state"]].R, np.eye(3)) and np.allclose(poses[t["ref_state"]].t, 0.0)
    step = geom.rigid_from_pivot((0.0, 0.0, 1.0), (0.5, 0.0, 0.0))
    want = step if t["ref_state"] == 0 else step.inverse()
    other = poses[1 - t["ref_state"]]
    assert np.allclose(other.R, want.R, atol=1e-9) and np.allclose(other.t, want.t, atol=1e-9)
    assert not any(t["contaminated_in_state"])              # every train is explained by its state's pose
    assert np.array_equal(np.asarray(t["contaminated"]), st != t["ref_state"])
    for cond in ("Ta", "Tb"):
        t = msim.build(cond, "il", small_src(), small=True)["truth"]
        assert set(t["state_of_train"]) == {0} and t["ref_state"] == 0
        assert t["contaminated_in_state"] == t["contaminated"]


def test_soft_field_scales_with_amp():
    src = small_src()
    opts = simulate.SMALL
    t1 = np.abs(simulate.combine(src["ksp_T1"], src["csm_T1"], 1))
    src["brain"] = lesions.brain_mask(t1, erode=opts["brain_erode"], cut_below=opts["brain_cut"])
    soft = warp.rbf_field(23, 1.5, src["brain"], 0.5, shape=t1.shape, sigma_mm=opts["sigma_mm"])
    ev = CONDITIONS2["Ta"]["intra"]
    kw = dict(intra_events=ev, small=True, order="interleaved")
    with_soft = simulate.build_custom(src, soft_u=soft, **kw)
    rigid_only = simulate.build_custom(src, **kw)
    no_amp = simulate.build_custom(src, soft_u=soft, **dict(kw, intra_events=[dict(e, soft=0.0) for e in ev]))
    assert np.array_equal(no_amp["ksp_T2"], rigid_only["ksp_T2"])
    assert _changed_trains(with_soft) == _changed_trains(rigid_only)
    sel = np.isin(with_soft["schedule"]["train_of"], _changed_trains(with_soft))
    assert not np.array_equal(with_soft["ksp_T2"][:, :, sel], rigid_only["ksp_T2"][:, :, sel])
    assert np.array_equal(with_soft["ksp_T2"][:, :, ~sel], rigid_only["ksp_T2"][:, :, ~sel])
    assert with_soft["truth"]["soft_amp"][_changed_trains(with_soft)[0]] == 1.0


def test_static_conditions_have_a_schedule_and_a_clean_truth():
    src = small_src()
    for cond in ("C0", "E2"):
        out = msim.build(cond, "il", src, small=True)
        assert "ksp_T2" not in out and "gt_ksp_T2" not in out
        t = out["truth"]
        assert t["order"] == "interleaved" and not any(t["contaminated"]) and set(t["state_of_train"]) == {0}
        assert out["schedule"]["train_of"].shape == src["ksp_T2"].shape[2:] and out["brain"].dtype == bool
        assert t["v1_data"] == CONDITIONS2[cond]["v1_data"]


def test_lesion_condition_has_both_contrasts():
    out = msim.build("TbL", "il", small_src(), small=True)
    assert "ksp_T1" in out and out["lesion_mask_canon"].max() >= 1 and out["truth"]["lesion_sites"]
