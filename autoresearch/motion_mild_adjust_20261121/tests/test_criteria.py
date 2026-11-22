"""Criteria S1-S4 of the second version on synthetic summaries (no files, no trainer)."""
import copy

import mcommon
import mevaluate as ev
from mcommon import ARMS2

T1_0, T2_0 = 35.80, 36.30                          # B0 on C0


def _row(order, cond, arm, t1, t2):
    single = ARMS2[arm]["trainer"] == "single"
    return dict(exp=mcommon.exp_name("af12", order, cond, arm), af="af12", order=order, cond=cond, arm=arm, seed=42,
                tag="", T1=None if single else dict(head_psnr=t1), T2=dict(head_psnr=t2, head_psnr_eval=t2))


def _name(order, cond, arm):
    return mcommon.exp_name("af12", order, cond, arm)


def _w(recall=None, false_rate=0.0, frac_w1=1.0):
    return dict(recall=recall, false_rate=false_rate, frac_w1=frac_w1, auc=None, n_contaminated=0, n_down=0)


def _states(n=1, n_true=1, rot=None, t=None, b=None):
    return dict(n_states=n, n_true=n_true, boundary_err_trains=b, rot_err_deg_max=rot, t_err_mm_max=t,
                tre_mean_mm_max=None)


def _inter(rigid_part, full):
    e = lambda v: dict(tre_mean_mm=v, tre_max_mm=2 * v, nonrigid_rmse_mm=0.1)
    return dict(rigid_part=e(rigid_part), rigid=e(0.35), full=e(full), nonrigid_brain_max_mm=0.4)


def good():
    """A summary in which every criterion is met."""
    rows, weights, states, inter = [], {}, {}, {}
    rows.append(_row("il", "C0", "B0", T1_0, T2_0))
    rows.append(_row("il", "C0", "I-T2", None, 36.25))
    for order, arms in (("il", ("B1", "B2", "B12", "B123", "I-T2+w")), ("kz", ("B2", "B123"))):
        for a in arms:
            rows.append(_row(order, "C0", a, T1_0 - 0.01, (36.25 if a == "I-T2+w" else T2_0) - 0.01))
            weights[_name(order, "C0", a)] = _w()
            if ARMS2[a]["m3"]:
                states[_name(order, "C0", a)] = _states()
    # inter-scan: E2
    rows += [_row("il", "E2", "B0", T1_0 - 0.02, T2_0 - 0.03), _row("il", "E2", "I-T2", None, 36.25),
             _row("il", "E2", "B1", T1_0 - 0.02, T2_0 - 0.02), _row("il", "E2", "B123", T1_0 - 0.02, T2_0 - 0.03)]
    for a in ("B1", "B123"):
        inter[_name("il", "E2", a)] = _inter(0.08, 0.25)
    # intra-scan, interleaved: loss 1.3 dB, the full arm takes back 0.8 dB
    for cond in ("Ta", "Tb", "Tc", "E2Tb"):
        rows += [_row("il", cond, "B0", T1_0 - 0.01, T2_0 - 1.3), _row("il", cond, "Ow", T1_0, T2_0 - 0.4),
                 _row("il", cond, "B123", T1_0 - 0.01, T2_0 - 0.5), _row("il", cond, "B2", T1_0 - 0.01, T2_0 - 0.6)]
        weights[_name("il", cond, "B123")] = _w(recall=0.95, false_rate=0.01, frac_w1=0.9)
        states[_name("il", cond, "B123")] = _states(2, 2, 0.2, 0.2, 1) if cond == "Tc" else _states()
    rows.append(_row("il", "Tc", "Op", T1_0, T2_0 - 0.3))
    # lesions on TbL
    rows += [_row("il", "TbL", "B0", T1_0, T2_0 - 1.3), _row("il", "TbL", "B123", T1_0, T2_0 - 0.5)]
    les = lambda rec: [dict(label=1, contrast="T2", diameter_mm=6.0, contrast_recovery=rec, centroid_shift_mm=0.2,
                            crosstalk=0.004, crosstalk_tol=0.01),
                       dict(label=2, contrast="T1", diameter_mm=6.0, contrast_recovery=0.9, centroid_shift_mm=0.2,
                            crosstalk=-0.002, crosstalk_tol=0.01)]
    lesions = dict(B0=les(0.70), B123=les(0.78))
    # linear order: reported only
    for cond in ("Ta", "Tc"):
        rows += [_row("kz", cond, "B0", T1_0, T2_0 - 3.0), _row("kz", cond, "B123", T1_0, T2_0 - 3.0 + 0.01)]
    reg = dict(il=dict(C0=dict(theta_deg=[0.29, -0.04, -0.31], t_mm=[0.52, 0.43, -0.16])))
    return dict(table=rows, weights=weights, states=states, inter=inter, lesions=lesions, reg=reg)


def _set(s, order, cond, arm, t1=None, t2=None):
    for r in s["table"]:
        if (r["order"], r["cond"], r["arm"]) == (order, cond, arm):
            if t1 is not None:
                r["T1"]["head_psnr"] = t1
            if t2 is not None:
                r["T2"].update(head_psnr=t2, head_psnr_eval=t2)
            return
    raise KeyError((order, cond, arm))


def test_all_criteria_pass_on_the_good_summary():
    c = ev.criteria(good())
    assert c["S1"]["pass"] is True and c["S2"]["pass"] is True and c["S3"]["pass"] is True and c["S4"]["pass"] is True
    assert set(c["S1"]["arms"]) == {"il:B1", "il:B2", "il:B12", "il:B123", "il:I-T2+w", "kz:B2", "kz:B123"}
    it = c["S1"]["arms"]["il:B2"]["dT2"]
    assert abs(it["value"] + 0.01) < 1e-9 and it["threshold"] == -0.05 and it["sense"] == ">=" and it["pass"] is True
    # a single-contrast arm is compared with the single-contrast model without weights, and has no T1
    assert abs(c["S1"]["arms"]["il:I-T2+w"]["dT2"]["value"] + 0.01) < 1e-9
    assert c["S1"]["arms"]["il:I-T2+w"]["dT1"]["value"] is None
    # reported, not judged: the registration of the unmoved head against the native offset of the data
    assert "pass" not in c["S1"]["registration"] and c["S1"]["registration"]["rot_deg"] > 0.4


def test_s1_fails_on_downweighted_clean_data():
    s = good()
    s["weights"][_name("il", "C0", "B2")]["frac_w1"] = 0.95
    c = ev.criteria(s)
    assert c["S1"]["arms"]["il:B2"]["frac_w1"]["pass"] is False and c["S1"]["pass"] is False
    assert c["S1"]["arms"]["il:B12"]["frac_w1"]["pass"] is True


def test_s1_fails_on_a_second_state_and_on_a_psnr_loss():
    s = good()
    s["states"][_name("kz", "C0", "B123")] = _states(2, 1)
    c = ev.criteria(s)
    assert c["S1"]["arms"]["kz:B123"]["n_states"]["pass"] is False and c["S1"]["pass"] is False
    s = good()
    _set(s, "il", "C0", "B1", t1=T1_0 - 0.06)
    c = ev.criteria(s)
    assert c["S1"]["arms"]["il:B1"]["dT1"]["pass"] is False and c["S1"]["arms"]["il:B1"]["dT2"]["pass"] is True


def test_s1_is_undecided_when_a_run_is_missing():
    s = good()
    s["table"] = [r for r in s["table"] if not (r["cond"] == "C0" and r["arm"] == "B12")]
    c = ev.criteria(s)
    assert c["S1"]["arms"]["il:B12"]["dT2"]["pass"] is None and c["S1"]["pass"] is None


def test_s2_items():
    c = ev.criteria(good())["S2"]
    b1 = c["arms"]["B1"]
    assert b1["tre_rigid"]["value"] == 0.08 and b1["tre_rigid"]["threshold"] == 0.2 and b1["tre_rigid"]["pass"] is True
    assert b1["tre_full"]["value"] == 0.25 and b1["tre_full"]["threshold"] == 0.3
    # normalised delta: T1 = (E2 - C0) of the arm; T2 additionally minus the same difference of I-T2 (here 0)
    assert abs(b1["delta_T1"]["value"] - (-0.01)) < 1e-9 and abs(b1["delta_T2"]["value"] - (-0.01)) < 1e-9
    s = good()
    s["inter"][_name("il", "E2", "B123")] = _inter(0.25, 0.35)
    _set(s, "il", "E2", "B123", t2=T2_0 - 0.2)
    c = ev.criteria(s)["S2"]
    assert c["arms"]["B123"]["tre_rigid"]["pass"] is False and c["arms"]["B123"]["tre_full"]["pass"] is False
    assert c["arms"]["B123"]["delta_T2"]["pass"] is False and c["pass"] is False
    b0 = ev.criteria(good())["S2"]["arms"]["B0"]                           # the unchanged model: reported, not judged
    assert abs(b0["delta_T2"]["value"] + 0.03) < 1e-9 and b0["delta_T2"]["pass"] is None and b0["pass"] is None


def test_s3_skips_small_loss():
    s = good()
    _set(s, "il", "Ta", "B0", t2=T2_0 - 0.05)
    c = ev.criteria(s)["S3"]
    assert c["conds"]["Ta"]["recovered"]["value"] is None and c["conds"]["Ta"]["recovered"]["pass"] is None
    assert c["conds"]["Ta"]["pass"] is None and "not significant" in c["conds"]["Ta"]["note"]
    assert abs(c["conds"]["Ta"]["recovered"]["loss"] - 0.05) < 1e-9
    assert c["conds"]["Tb"]["pass"] is True and c["pass"] is True          # the skipped condition does not count


def test_s3_pass_and_fail():
    c = ev.criteria(good())["S3"]
    tb = c["conds"]["Tb"]
    assert abs(tb["recovered"]["value"] - 0.8 / 1.3) < 1e-9 and tb["recovered"]["pass"] is True
    assert tb["recall"]["pass"] is True and tb["false_rate"]["pass"] is True and tb["pass"] is True
    assert c["conds"]["Tc"]["pose_deg"]["pass"] is True and c["conds"]["Tc"]["pose_mm"]["pass"] is True
    assert "pose_deg" not in tb
    assert abs(c["conds"]["Tc"]["others"]["Op"] - 1.0 / 1.3) < 1e-9        # upper bounds are reported
    s = good()
    _set(s, "il", "Tc", "B123", t2=T2_0 - 1.0)                           # takes back 0.3 of 1.3 dB
    s["states"][_name("il", "Tc", "B123")] = _states(2, 2, 0.2, 0.4, 1)
    s["weights"][_name("il", "Tb", "B123")] = _w(recall=0.85, false_rate=0.03, frac_w1=0.9)
    c = ev.criteria(s)["S3"]
    assert c["conds"]["Tc"]["recovered"]["pass"] is False and c["conds"]["Tc"]["pose_mm"]["pass"] is False
    assert c["conds"]["Tc"]["pose_deg"]["pass"] is True and c["conds"]["Tc"]["pass"] is False
    assert c["conds"]["Tb"]["recall"]["pass"] is False and c["conds"]["Tb"]["false_rate"]["pass"] is False
    assert c["pass"] is False


def test_s3_states_not_found_fails_the_pose_items():
    s = good()
    s["states"][_name("il", "Tc", "B123")] = _states(1, 2)                # the shift was not detected
    c = ev.criteria(s)["S3"]["conds"]["Tc"]
    assert c["pose_deg"]["pass"] is False and c["pose_mm"]["pass"] is False and c["n_states"]["pass"] is False


def test_s4_lesions():
    c = ev.criteria(good())["S4"]
    assert abs(c["recovery_vs_B0"]["value"]) < 1e-9                       # the worst lesion: 0.90 against 0.90
    assert c["recovery_vs_B0"]["threshold"] == -0.05 and c["recovery_vs_B0"]["pass"] is True
    assert c["crosstalk"]["pass"] is True and c["pass"] is True
    s = good()
    s["lesions"]["B123"][0]["contrast_recovery"] = 0.60                  # 0.10 below B0
    s["lesions"]["B123"][1]["crosstalk"] = 0.05
    c = ev.criteria(s)["S4"]
    assert abs(c["recovery_vs_B0"]["value"] + 0.10) < 1e-9 and c["recovery_vs_B0"]["pass"] is False
    assert c["crosstalk"]["pass"] is False and c["pass"] is False
    s = good()
    s["lesions"] = {}
    assert ev.criteria(s)["S4"]["pass"] is None


def test_kz_reported_not_judged():
    s = good()
    _set(s, "kz", "Tc", "B123", t2=T2_0 - 3.1)
    c = ev.criteria(s)
    kz = c["kz"]
    assert abs(kz["Tc"]["value"] + 0.1) < 1e-9 and kz["Tc"]["no_harm"] is False and kz["Tc"]["pass"] is None
    assert abs(kz["Ta"]["value"] - 0.01) < 1e-9 and kz["Ta"]["no_harm"] is True and kz["Ta"]["pass"] is None
    assert "pass" not in kz                                               # no verdict for the linear order
    assert c["S3"]["pass"] is True                                        # and it does not touch S3


def test_summary_has_no_clock_keys():
    s = good()
    s["criteria"] = ev.criteria(copy.deepcopy(s))
    assert ev.clock_keys(s) == []
    bad = dict(a=[dict(elapsed_s=1.0)], run=dict(timestamp_utc="x", fine=1), train_time=2, updated=3)
    assert sorted(ev.clock_keys(bad)) == ["a[0].elapsed_s", "run.timestamp_utc", "train_time", "updated"]
