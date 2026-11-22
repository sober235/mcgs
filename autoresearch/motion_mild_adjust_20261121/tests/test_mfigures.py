import numpy as np

import mfigures


def _inputs():
    cont = np.zeros(110, dtype=bool)
    cont[40:47] = True
    rng = np.random.default_rng(0)
    z = rng.normal(size=110)
    z[cont] += 9.0
    feas = dict(auc=1.0, auc_history=[dict(it=600, auc=0.71), dict(it=700, auc=0.93), dict(it=4000, auc=1.0)],
                loss=1.2, recover=0.62, b0_c0=36.3, b0=35.1, ow=35.844, n_contaminated=7, n_trains=110,
                pass_auc=True, pass_recover=True, decision="go")
    return feas, dict(m2=dict(z=z.tolist(), settings=dict(z0=4.0, z1=8.0))), dict(contaminated=cont.tolist())


def test_feas_figure_writes_png(tmp_path):
    feas, mm, truth = _inputs()
    out = mfigures.feas_figure(feas, mm, truth, tmp_path / "feas.png")
    assert out.exists() and out.stat().st_size > 20_000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_feas_figure_handles_a_failed_gate(tmp_path):
    feas, mm, truth = _inputs()
    feas.update(recover=None, loss=0.04, decision="stop: motion too small to measure", pass_recover=False, ow=None)
    assert mfigures.feas_figure(feas, mm, truth, tmp_path / "f.png").exists()


def _summary():
    import mevaluate as ev
    from test_criteria import good
    s = good()
    s["criteria"] = ev.criteria(s)
    s["recovered"] = {}
    for r in s["table"]:
        if r["cond"] != "C0":
            s["recovered"].setdefault(f"{r['order']}:{r['cond']}", {})[r["arm"]] = ev.recovered(
                s["table"], r["order"], r["cond"], r["arm"], base="I-T2" if r["arm"].startswith("I-") else "B0")
    return s


def _png_ok(p):
    return p.exists() and p.stat().st_size > 20000 and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_recovered_figure(tmp_path):
    s = _summary()
    data = mfigures.recovered_data(s)
    assert [c for c, _ in data["il"]] == ["Ta", "Tb", "Tc", "E2Tb"] and [c for c, _ in data["kz"]] == ["Ta", "Tc"]
    tb = dict(data["il"])["Tb"]
    assert abs(tb["B123"] - 0.8 / 1.3) < 1e-9 and abs(tb["Ow"] - 0.9 / 1.3) < 1e-9 and "Op" not in tb
    assert abs(dict(data["il"])["Tc"]["Op"] - 1.0 / 1.3) < 1e-9
    assert _png_ok(mfigures.recovered_figure(s, tmp_path / "fig_recovered.png"))


def test_recovered_figure_marks_a_condition_with_too_small_a_loss(tmp_path):
    s = _summary()
    for arm in list(s["recovered"]["il:Ta"]):                 # the loss is a property of the condition
        s["recovered"]["il:Ta"][arm] = dict(s["recovered"]["il:Ta"][arm], value=None, loss=0.05)
    data = mfigures.recovered_data(s)
    assert dict(data["il"])["Ta"] == {"_loss": 0.05}
    assert _png_ok(mfigures.recovered_figure(s, tmp_path / "fig_recovered.png"))


def test_trains_figure(tmp_path):
    n = 110
    state = [0] * 66 + [1] * 44
    disp = [0.0] * 30 + [0.6, 0.9, 0.5] + [0.0] * 33 + [0.8] * 44
    truth = dict(train_brain_disp_mm=disp, state_of_train=state, contaminated=[d > 0.2 for d in disp])
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, n)
    z[30:33] += 9
    w = np.ones(n)
    w[30:33] = 0.0
    mm = dict(m2=dict(mode="adaptive", w=w.tolist(), z=z.tolist(), settings=dict(z0=4.0, z1=8.0), history=[]),
              m3=dict(n=2, boundaries=[68], state_of_train=[0] * 68 + [1] * 42))
    p = mfigures.trains_figure(truth, mm, "il / Tb, B123", tmp_path / "fig_trains.png")
    assert _png_ok(p)
    # a run without states and without a statistic still draws (monitor arm before the first computation)
    mm0 = dict(m2=dict(mode="monitor", w=[1.0] * n, z=None, history=[]), m3=None)
    assert _png_ok(mfigures.trains_figure(truth, mm0, "il / Tb, B0", tmp_path / "fig_trains0.png"))


def test_images_figure(tmp_path):
    rng = np.random.default_rng(1)
    gt = np.zeros((24, 64, 48), dtype=np.float32)
    gt[:, 12:52, 8:40] = 1.0 + 0.2 * rng.random((24, 40, 32)).astype(np.float32)
    recs = [("B0", gt + 0.05 * rng.standard_normal(gt.shape).astype(np.float32), 33.21),
            ("B123", gt + 0.02 * rng.standard_normal(gt.shape).astype(np.float32), 35.02),
            ("Ow", gt + 0.01 * rng.standard_normal(gt.shape).astype(np.float32), 35.60)]
    p = mfigures.images_figure(gt, recs, 12, "il / Tb, T2W", tmp_path / "fig_images.png")
    assert _png_ok(p)
    assert mfigures.error_scale(gt, [r for _, r, _ in recs], 12) > 0.0


def test_align_figure(tmp_path):
    assert _png_ok(mfigures.align_figure(_summary(), tmp_path / "fig_align_E2.png"))


def test_worst_slice_is_inside_the_brain():
    gt = np.zeros((10, 8, 8))
    rec = gt.copy()
    rec[2] += 5.0                                   # a large error outside the brain does not count
    rec[6, 3:5, 3:5] += 1.0
    brain = np.zeros(gt.shape, dtype=bool)
    brain[4:8, 2:6, 2:6] = True
    assert mfigures.worst_slice(gt, rec, brain) == 6
