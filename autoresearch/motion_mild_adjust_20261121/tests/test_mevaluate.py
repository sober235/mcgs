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
