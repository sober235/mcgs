import json

import numpy as np

import mevaluate as ev


def _row(order, cond, arm, t2, t1=35.0, tag="", seed=42):
    single = arm.startswith("I-")
    s = dict(head_psnr=t2, head_ssim=0.9, nmse=0.01)
    return dict(exp=f"mm_af12_{order}_{cond}_{arm}" + (f"_{tag}" if tag else ""), af="af12", order=order, cond=cond,
                arm=arm, seed=seed, tag=tag, plain=s if single else dict(head_psnr=t1, head_ssim=0.9, nmse=0.01),
                contrast=None if single else s)


def _scores(wdc=None, raw=None):
    """Stand-in for the files: exp -> head PSNR of the kept-lines image / of the raw render."""
    def f(exp, key, single=False):
        d = (wdc or {}) if key == "rec_w_dc" else (raw or {})
        return None if exp not in d else dict(head_psnr=d[exp], head_ssim=0.9, head_hfen=0.1)
    return f


def test_eval_key_by_arm():
    led = [_row("il", "Ta", "B0", 34.0), _row("il", "Ta", "B2", 34.2), _row("il", "Ta", "Ow", 34.1),
           _row("il", "Ta", "I-T2", 33.9), _row("il", "Ta", "I-T2+w", 34.0)]
    wdc = {"mm_af12_il_Ta_B2": 35.5, "mm_af12_il_Ta_Ow": 35.8, "mm_af12_il_Ta_I-T2+w": 35.1}
    raw = {r["exp"]: 33.0 for r in led}
    t = ev.table(led, scores=_scores(wdc, raw))
    assert ev.val(t, "il", "Ta", "B0") == 34.0                     # scorer's value: every sampled line inserted
    assert ev.val(t, "il", "Ta", "B2") == 35.5 and ev.val(t, "il", "Ta", "Ow") == 35.8
    assert ev.val(t, "il", "Ta", "B2", key="head_psnr") == 34.2 and ev.val(t, "il", "Ta", "B2", key="head_psnr_raw") == 33.0
    assert ev.val(t, "il", "Ta", "I-T2") == 33.9 and ev.val(t, "il", "Ta", "I-T2+w") == 35.1
    assert ev.val(t, "il", "Ta", "B0", c="T1") == 35.0 and ev.val(t, "il", "Ta", "I-T2", c="T1") is None
    assert ev.val(t, "il", "Tb", "B0") is None
    row = [r for r in t if r["arm"] == "B2"][0]
    assert row["T2"]["head_psnr_wdc"] == 35.5 and row["order"] == "il"


def test_last_ledger_row_of_an_experiment_wins_and_tags_are_separate():
    led = [_row("il", "DTa", "B2", 30.0, tag="d1"), _row("il", "DTa", "B2", 31.0, tag="d2"),
           _row("il", "DTa", "B2", 32.0, tag="d2")]
    t = ev.table(led, scores=_scores({r["exp"]: r["plain"]["head_psnr"] + 0.5 for r in led}))
    assert len(t) == 2
    assert ev.val(t, "il", "DTa", "B2", key="head_psnr", tag="d1") == 30.0
    assert ev.val(t, "il", "DTa", "B2", key="head_psnr", tag="d2") == 32.0
    assert ev.val(t, "il", "DTa", "B2", key="head_psnr") is None


def test_recovered_fraction():
    led = [_row("il", "C0", "B0", 36.0), _row("il", "Ta", "B0", 34.0), _row("il", "Ta", "B2", 34.1)]
    t = ev.table(led, scores=_scores({"mm_af12_il_Ta_B2": 35.5}))
    r = ev.recovered(t, "il", "Ta", "B2")
    assert abs(r["loss"] - 2.0) < 1e-12 and abs(r["value"] - 0.75) < 1e-12
    assert r["b0_c0"] == 36.0 and r["b0"] == 34.0 and r["arm_psnr"] == 35.5
    assert ev.recovered(t, "il", "Ta", "B123") is None             # the arm has not run


def test_recovered_uses_the_untagged_baseline_for_a_tagged_run():
    led = [_row("il", "C0", "B0", 36.0), _row("il", "DTa", "B0", 35.0), _row("il", "DTa", "B2", 35.0, tag="d1")]
    t = ev.table(led, scores=_scores({"mm_af12_il_DTa_B2_d1": 35.6}))
    assert abs(ev.recovered(t, "il", "DTa", "B2", tag="d1")["value"] - 0.6) < 1e-9


def test_recovered_of_a_single_contrast_arm_uses_its_own_baseline():
    led = [_row("il", "C0", "I-T2", 36.2), _row("il", "Tb", "I-T2", 34.2), _row("il", "Tb", "I-T2+w", 34.0),
           _row("il", "C0", "B0", 36.0), _row("il", "Tb", "B0", 34.5)]
    t = ev.table(led, scores=_scores({"mm_af12_il_Tb_I-T2+w": 35.2}))
    r = ev.recovered(t, "il", "Tb", "I-T2+w", base="I-T2")
    assert abs(r["loss"] - 2.0) < 1e-9 and abs(r["value"] - 0.5) < 1e-9


def test_recovered_none_when_loss_small():
    led = [_row("il", "C0", "B0", 36.0), _row("il", "Ta", "B0", 35.95), _row("il", "Ta", "B2", 35.9)]
    t = ev.table(led, scores=_scores({"mm_af12_il_Ta_B2": 35.99}))
    r = ev.recovered(t, "il", "Ta", "B2")
    assert abs(r["loss"] - 0.05) < 1e-9 and r["value"] is None


def _mm(w, z=None, history=None):
    return dict(m2=dict(mode="adaptive", w=list(w), z=None if z is None else list(z), history=history or []))


def test_weight_detection_counts():
    cont = np.zeros(110, dtype=bool)
    cont[:10] = True
    w = np.ones(110)
    w[:9] = 0.0                                                     # 9 of 10 contaminated trains below 0.5
    w[50] = 0.3                                                     # one clean train below 0.5
    w[60] = 0.8                                                     # partly down-weighted, still kept
    z = np.where(cont, 10.0, 0.0)
    d = ev.weight_detection(_mm(w, z), dict(contaminated=cont.tolist()))
    assert abs(d["recall"] - 0.9) < 1e-12 and abs(d["false_rate"] - 0.01) < 1e-12
    assert abs(d["frac_w1"] - 99 / 110) < 1e-12 and d["auc"] == 1.0
    clean = ev.weight_detection(_mm(np.ones(110), np.zeros(110)), dict(contaminated=[False] * 110))
    assert clean["recall"] is None and clean["false_rate"] == 0.0 and clean["frac_w1"] == 1.0 and clean["auc"] is None
    assert ev.weight_detection(_mm(np.ones(110)), dict(contaminated=cont.tolist()))["auc"] is None     # no statistic yet


def _feas(auc_last, recover, loss=2.0):
    cont = np.zeros(110, dtype=bool)
    cont[40:47] = True
    good = np.where(cont, 9.0, 0.0)
    rng = np.random.default_rng(0)
    poor = rng.normal(size=110)
    z_last = good if auc_last else poor
    mm = _mm(np.ones(110), z_last, history=[dict(it=600, z=poor.tolist()), dict(it=700, z=z_last.tolist())])
    led = [_row("il", "C0", "B0", 36.0), _row("il", "DTa", "B0", 36.0 - loss), _row("il", "DTa", "Ow", 30.0)]
    t = ev.table(led, scores=_scores({"mm_af12_il_DTa_Ow": 36.0 - loss + recover * loss}))
    return ev.feasibility(t, mm=mm, truth=dict(contaminated=cont.tolist()))


def test_feasibility_decision():
    f = _feas(True, 0.6)
    assert f["decision"] == "go" and f["pass_auc"] and f["pass_recover"] and f["auc"] == 1.0
    assert abs(f["recover"] - 0.6) < 1e-9 and abs(f["loss"] - 2.0) < 1e-9
    assert [h["it"] for h in f["auc_history"]] == [600, 700] and f["auc_history"][0]["auc"] < 0.95
    assert _feas(False, 0.6)["decision"] == "stop" and not _feas(False, 0.6)["pass_auc"]
    assert _feas(True, 0.4)["decision"] == "stop" and not _feas(True, 0.4)["pass_recover"]
    small = _feas(True, 0.6, loss=0.05)
    assert small["decision"] == "stop: motion too small to measure" and small["recover"] is None
    json.dumps(f)


def _tc_truth():
    import geom
    state = [0] * 66 + [1] * 44
    moved = geom.Rigid(geom.rotmat(np.deg2rad([0.0, 0.0, 1.0])), np.array([0.5, 0.0, 0.0]))
    return dict(state_of_train=state, ref_state=0, state_poses=[geom.Rigid.identity().to_json(), moved.to_json()],
                contaminated=[bool(s) for s in state])


def _m3(first, theta=(0.0, 0.0, 1.1), t=(0.7, 0.0, 0.0)):
    zero = dict(theta_deg=[0.0, 0.0, 0.0], t_mm=[0.0, 0.0, 0.0])
    return dict(mode="detect", n=2, boundaries=[first], state_of_train=[0] * first + [1] * (110 - first),
                final_poses=[zero, dict(theta_deg=list(theta), t_mm=list(t))])


def test_state_errors():
    from helpers import ball_mask
    brain = ball_mask((32, 32, 24), 8)
    e = ev.state_errors(dict(m3=_m3(68)), _tc_truth(), brain)
    assert e["n_states"] == 2 and e["n_true"] == 2 and e["boundary_err_trains"] == 2
    assert abs(e["rot_err_deg_max"] - 0.1) < 1e-6 and abs(e["t_err_mm_max"] - 0.2) < 1e-9
    none = ev.state_errors(dict(m3=dict(mode="detect", n=1, boundaries=[], state_of_train=[0] * 110, final_poses=None)),
                           _tc_truth(), brain)
    assert none["n_states"] == 1 and none["n_true"] == 2 and none["boundary_err_trains"] is None
    assert none["rot_err_deg_max"] is None and none["t_err_mm_max"] is None
    assert ev.state_errors(dict(m3=None), _tc_truth(), brain) is None


def test_detection_counts_state_members_as_handled():
    truth = _tc_truth()
    ones = dict(m2=dict(mode="adaptive", w=[1.0] * 110, z=None, history=[]))
    d = ev.weight_detection(dict(ones, m3=_m3(66)), truth)             # 40 % of the trains in state 1, all weights 1
    assert d["recall"] == 1.0 and d["false_rate"] == 0.0 and d["frac_w1"] == 1.0
    d = ev.weight_detection(dict(ones, m3=_m3(64)), truth)             # the boundary two trains early
    assert d["recall"] == 1.0 and abs(d["false_rate"] - 2 / 66) < 1e-12
    assert ev.weight_detection(dict(ones, m3=None), truth)["recall"] == 0.0
    assert ev.weight_detection(ones, truth)["recall"] == 0.0           # files written before M3 have no m3 key


def test_false_rate_unmoved_leaves_out_clean_trains_with_a_true_lasting_shift():
    # Tb-like truth: 8 contaminated trains, a lasting 0.144 mm step from train 81 on (below CONTAM_MM, so "clean"),
    # one true state. M3 puts trains 81-109 into a second state, M2 down-weights exactly the 8 contaminated trains.
    cont = np.zeros(110, dtype=bool)
    cont[[28, 29, 30, 31, 77, 78, 79, 80]] = True
    disp = np.zeros(110)
    disp[cont] = 1.888
    disp[81:] = 0.144
    truth = dict(contaminated=cont.tolist(), train_brain_disp_mm=disp.tolist(), state_of_train=[0] * 110)
    w = np.where(cont, 0.0, 1.0)
    d = ev.weight_detection(dict(_mm(w), m3=_m3(81)), truth)
    assert d["recall"] == 1.0
    assert abs(d["false_rate"] - 29 / 102) < 1e-12                    # the pre-registered item is unchanged
    assert d["false_rate_unmoved"] == 0.0 and d["n_clean_moved"] == 29 and d["n_clean_moved_handled"] == 29
    early = ev.weight_detection(dict(_mm(w), m3=_m3(79)), truth)       # trains 79, 80 are contaminated anyway
    assert early["false_rate_unmoved"] == 0.0
    wrong = ev.weight_detection(dict(_mm(w), m3=_m3(60)), truth)       # boundary 21 trains too early
    assert abs(wrong["false_rate_unmoved"] - 17 / 73) < 1e-12         # trains 60-76 = 17 of the 73 unmoved clean ones


def test_false_rate_unmoved_without_per_train_displacement_equals_false_rate():
    d = ev.weight_detection(dict(_mm(np.ones(110)), m3=_m3(64)), _tc_truth())
    assert abs(d["false_rate"] - 2 / 66) < 1e-12 and d["false_rate_unmoved"] == d["false_rate"]
    assert d["n_clean_moved"] == 0


def test_inter_errors():
    import geom
    from helpers import ball_mask
    shape = (32, 32, 24)
    brain = ball_mask(shape, 8, offset=(0.0, 0.0, 0.0))
    pre = geom.Rigid(np.eye(3), np.array([0.5, 0.1, -0.14]))             # what the registration finds without motion
    g = geom.rigid_from_pivot((0.4, -0.3, 1.0), (0.8, -0.5, 0.3), pivot_offset_mm=(0.0, 0.0, 0.0))
    exact = g.compose(pre)
    st = dict(theta_deg=np.rad2deg(geom.axis_angle(exact.R)).tolist(), t_mm=exact.t.tolist(), ctrl_mm=None,
              segments=[], ref_segment=0)
    e = ev.inter_errors(st, g, None, pre, brain)
    assert e["rigid"]["tre_mean_mm"] < 1e-9 and e["full"]["tre_mean_mm"] < 1e-9 and e["nonrigid_brain_max_mm"] == 0.0
    assert e["rigid_part"]["tre_mean_mm"] < 1e-9
    st0 = dict(st, theta_deg=[0.0, 0.0, 0.0], t_mm=pre.t.tolist())        # the pose of the unmoved head
    e0 = ev.inter_errors(st0, g, None, pre, brain)
    r = geom.grid_mm(shape)[brain]
    want = float(np.linalg.norm(g.apply(pre.apply(r)) - pre.apply(r), axis=-1).mean())
    assert abs(e0["rigid"]["tre_mean_mm"] - want) < 1e-9 and want > 0.5
    # a smooth non-rigid truth on top: the rigid part alone leaves it, a matching control grid removes most of it
    disp = np.zeros(shape + (3,))
    disp[..., 0] = 0.6 * (geom.grid_mm(shape)[..., 1] / 16.0)             # shear-like, 0.6 mm at the edge
    total = (g.apply(geom.grid_mm(shape) + disp) - geom.grid_mm(shape))   # displacement field of (g o (1 + u))
    e1 = ev.inter_errors(st, None, total, pre, brain)
    assert 0.05 < e1["rigid"]["tre_mean_mm"] < 0.6 and e1["rigid_part"] is None
    # with the truth's rigid part given too: the estimated rigid part is judged against it alone
    e1r = ev.inter_errors(st, g, total, pre, brain)
    assert e1r["rigid_part"]["tre_mean_mm"] < 1e-9 and abs(e1r["rigid"]["tre_mean_mm"] - e1["rigid"]["tre_mean_mm"]) < 1e-12
    assert ev.inter_errors(st0, g, total, pre, brain)["rigid_part"]["tre_mean_mm"] > 0.5
    ctrl = np.zeros((5, 5, 5, 3))
    ctrl[..., 0] = 0.6 * np.linspace(-1, 1, 5)[None, :, None]             # the same shear on the control grid
    e2 = ev.inter_errors(dict(st, ctrl_mm=ctrl.tolist()), None, total, pre, brain)
    assert e2["full"]["tre_mean_mm"] < 0.5 * e1["rigid"]["tre_mean_mm"]
    assert abs(e2["rigid"]["tre_mean_mm"] - e1["rigid"]["tre_mean_mm"]) < 1e-9 and e2["nonrigid_brain_max_mm"] > 0.1


def test_state_errors_of_the_initial_poses_and_of_no_correction():
    from helpers import ball_mask
    brain = ball_mask((32, 32, 24), 8)
    m3 = dict(_m3(66), init_poses=[dict(theta_deg=[0.0, 0.0, 0.0], t_mm=[0.0, 0.0, 0.0]),
                                   dict(theta_deg=[0.0, 0.0, 1.0], t_mm=[0.5, 0.3, 0.0])])
    e = ev.state_errors(dict(m3=m3), _tc_truth(), brain, poses="init_poses")
    assert abs(e["t_err_mm_max"] - 0.3) < 1e-9 and e["rot_err_deg_max"] < 1e-6
    e0 = ev.state_errors(dict(m3=m3), _tc_truth(), brain, poses="identity")   # what no state pose at all would leave
    assert abs(e0["rot_err_deg_max"] - 1.0) < 1e-6 and abs(e0["t_err_mm_max"] - 0.5) < 1e-9
    assert e0["tre_mean_mm_max"] > e["tre_mean_mm_max"] > 0.0


def test_lesion_rows_from_volumes():
    """The lesion numbers of the first version, computed from volumes held in memory (so that the second version can
    use them with its own file names): recovery in the own contrast, crosstalk in the other one against a run
    without lesions."""
    import evaluate as ev1
    from helpers import ball_mask
    shape = (40, 40, 32)
    rng = np.random.default_rng(0)
    m = ball_mask(shape, 3.0, offset=(0.0, 0.0, 0.0))
    gt2 = np.ones(shape) + 0.5 * m                       # a T2 lesion, gain +0.5
    rec2 = np.ones(shape) + 0.4 * m                      # recovered to 80 %
    gt1 = np.ones(shape)
    rec1 = gt1 + 0.02 * m + 0.001 * rng.standard_normal(shape)        # 2 % of it leaks into T1
    rec1_ref = gt1 + 0.001 * rng.standard_normal(shape)
    truth = dict(lesion_sites=[dict(contrast="T2", diameter_mm=6.0, gain=0.5)])
    masks = dict(T1=m.astype(np.int16), T2=m.astype(np.int16))
    rows = ev1.lesion_rows(dict(T1=(rec1, gt1), T2=(rec2, gt2)), dict(T1=(rec1_ref, gt1), T2=(gt2, gt2)), truth, masks)
    assert len(rows) == 1 and rows[0]["label"] == 1 and rows[0]["contrast"] == "T2"
    assert abs(rows[0]["contrast_recovery"] - 0.8) < 1e-9
    assert abs(rows[0]["crosstalk"] - 0.02) < 2e-3 and rows[0]["crosstalk"] > rows[0]["crosstalk_tol"] > 0.0
    solo = ev1.lesion_rows(dict(T2=(rec2, gt2)), {}, truth, masks)     # a single-contrast run: no other contrast
    assert abs(solo[0]["contrast_recovery"] - 0.8) < 1e-9 and solo[0]["crosstalk"] is None


def test_detection_without_intra_scan_truth_counts_every_train_as_clean():
    """Conditions taken from the first version (E2, D2) have no per-train truth: every train is clean."""
    mm = dict(m2=dict(mode="adaptive", w=[1.0] * 108 + [0.2, 0.3], z=None, history=[]), m3=None)
    d = ev.weight_detection(mm, dict(inter=dict(R=[], t=[])))
    assert d["recall"] is None and abs(d["false_rate"] - 2 / 110) < 1e-12 and d["n_contaminated"] == 0


def test_scores_are_computed_once_per_result_file(tmp_path):
    """Scoring one volume takes minutes on the server; a result file is scored once and the scores are kept beside
    it. A result file of another size (a rerun) is scored again."""
    calls = []

    def compute(key):
        calls.append(key)
        return dict(head_psnr=30.0 + len(calls), head_ssim=0.9)
    cache = tmp_path / "x_evalscores.json"
    a = ev.cached_scores(cache, "rec_raw", 1000, compute)
    b = ev.cached_scores(cache, "rec_raw", 1000, compute)
    assert a == b == dict(head_psnr=31.0, head_ssim=0.9) and calls == ["rec_raw"]
    c = ev.cached_scores(cache, "rec_w_dc", 1000, compute)                 # another array of the same file
    assert c["head_psnr"] == 32.0 and ev.cached_scores(cache, "rec_raw", 1000, compute) == a
    d = ev.cached_scores(cache, "rec_raw", 1200, compute)                  # the result file changed
    assert d["head_psnr"] == 33.0 and calls == ["rec_raw", "rec_w_dc", "rec_raw"]
    assert ev.cached_scores(cache, "rec_w_dc", 1200, compute)["head_psnr"] == 34.0   # the old array went with it
    assert ev.clock_keys(__import__("json").loads(cache.read_text())) == []
