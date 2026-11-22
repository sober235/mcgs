import mcommon as m
import common as c1


def test_matrix_rows_are_defined():
    for part, rows in m.MATRIX2.items():
        for order, cond, arm in rows:
            assert order in m.ORDERS and cond in m.CONDITIONS2 and arm in m.ARMS2, (part, order, cond, arm)
    assert len(m.FORMAL) == len(set(m.FORMAL)) == 43
    assert ("il", "C0", "B0") in m.FORMAL
    dev = {"DTa", "DTc", "D2"}
    assert not any(cond in dev for _, cond, _ in m.FORMAL)
    assert all(cond in dev | {"C0"} for _, cond, _ in m.MATRIX2["feas"] + m.MATRIX2["dev"])
    assert all(cond == "C0" for _, cond, _ in m.MATRIX2["gate"])


def test_conditions_and_arms_have_their_keys():
    for name, spec in m.CONDITIONS2.items():
        assert sorted(spec) == ["inter", "intra", "lesions", "nonrigid", "v1_data"], name
        for e in spec["intra"] or []:
            assert e["kind"] in ("pulse", "step") and len(e["theta_deg"]) == 3 and len(e["t_mm"]) == 3
    for name, arm in m.ARMS2.items():
        assert sorted(arm) == ["m1", "m2", "m3", "slot", "trainer"], name
        assert arm["m2"] in ("monitor", "adaptive", "truth") and arm["m3"] in (None, "detect", "truth")
    assert m.CONDITIONS2["E2"]["v1_data"] == "E2" and m.CONDITIONS2["D2"]["v1_data"] == "D2"
    assert m.CONDITIONS2["E2Tb"]["inter"] == c1._E1A and m.CONDITIONS2["E2Tb"]["nonrigid"] == c1._NR_E2
    assert m.CONDITIONS2["Tc"]["intra"][0]["at"] == 0.60 and m.CONDITIONS2["TbL"]["lesions"] is True


def test_exp_name():
    assert m.exp_name("af12", "il", "Ta", "B2") == "mm_af12_il_Ta_B2"
    assert m.exp_name("af12", "kz", "Tc", "I-T2+w", seed=7, tag="d1") == "mm_af12_kz_Tc_I-T2+w_s7_d1"


def test_roots_are_separate_from_the_first_version():
    assert m.ROOT2 != c1.ROOT and str(m.ROOT2).endswith("mcgs_motion_mild_adjust_20261121")
    assert m.RES2 == m.TR2 / "results_arc" and m.DATA2 == m.ROOT2 / "data"
    assert m.THRESH2["feas_auc"] == 0.95 and m.THRESH2["s3_recover"] == 0.5 and m.CONTAM_MM == 0.2
