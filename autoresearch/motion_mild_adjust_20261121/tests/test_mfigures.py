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


def _slab(seed=1):
    """A 24 x 64 x 48 block with an edge inside it; returns (gt, rng)."""
    rng = np.random.default_rng(seed)
    gt = np.zeros((24, 64, 48), dtype=np.float32)
    gt[:, 8:56, 6:42] = 1.0
    gt[:, 8:56, 24:42] = 0.5                         # a tissue boundary along column 24
    return gt, rng


def _noisy(gt, rng, sd):
    return (gt + sd * rng.standard_normal(gt.shape) * (gt > 0)).astype(np.float32)


def test_record_numbers_bold_the_best_column_that_does_not_use_the_truth():
    gt, rng = _slab()
    cells = {"ZF": _noisy(gt, rng, 0.2), "MC-GS": _noisy(gt, rng, 0.05), "True motion": _noisy(gt, rng, 0.001),
             "MC-GS + M1-M3": _noisy(gt, rng, 0.02)}
    r = mfigures.record_numbers(dict(gt=gt, cells=cells), 12, truth_cols=("True motion",))
    assert set(r["num"]) == set(cells)
    assert r["best"] == {"psnr": "MC-GS + M1-M3", "ssim": "MC-GS + M1-M3", "nmse": "MC-GS + M1-M3"}
    assert r["num"]["True motion"]["psnr"] > r["num"]["MC-GS + M1-M3"]["psnr"]      # the bound is better, not bold
    assert r["head"].shape == (64, 48) and r["head"].sum() == 48 * 36
    assert abs(r["gmax"] - 1.0) < 1e-6


def test_box_origin_keeps_the_box_inside_the_slice():
    score = np.zeros((64, 48))
    score[62, 46] = 1.0
    assert mfigures.box_origin(score, 16) == (48, 32)
    score[:] = 0.0
    score[1, 1] = 1.0
    assert mfigures.box_origin(score, 16) == (0, 0)
    score[:] = 0.0
    score[30, 20] = 1.0
    assert mfigures.box_origin(score, 16) == (22, 12)       # centred on the peak when there is room


def test_arrow_point_is_where_the_other_columns_err_more_than_the_last():
    gt, _ = _slab()
    g2 = gt[12].astype(np.float64)
    flat = np.full(g2.shape, 0.01)
    bump = flat.copy()
    bump[28:33, 22:27] = 0.08                       # on the boundary along column 24
    errs = {"ZF": np.full(g2.shape, 0.5), "MC-GS": bump, "MC-GS + M2": bump, "True motion": flat, "MC-GS + M1-M3": flat}
    x, y, gap = mfigures.arrow_point(g2, errs, ["MC-GS", "MC-GS + M2"], "MC-GS + M1-M3", (10, 4), 40)
    assert abs((x + 4) - 24) <= 3 and abs((y + 10) - 30) <= 3 and gap > 0.0
    assert mfigures.arrow_point(g2, errs, [], "MC-GS + M1-M3", (10, 4), 40) is None
    assert mfigures.arrow_point(g2, {"MC-GS": bump}, ["MC-GS"], "MC-GS + M1-M3", (10, 4), 40) is None


def test_record_figure_draws_rows_with_a_missing_cell_and_a_mask(tmp_path):
    import json
    gt, rng = _slab()
    cols = ("ZF", "GS-single", "MC-GS", "True motion", "MC-GS + M1-M3")
    full = {c: _noisy(gt, rng, sd) for c, sd in zip(cols, (0.2, 0.06, 0.05, 0.01, 0.02))}
    kmask = (rng.random((48, 24)) < 0.2).astype(np.int8)
    kmask[10:14] *= 2                               # lines acquired during the motion
    rows = [dict(name="Brief motion (Ta)", contrast="T1W", gt=gt, kmask=np.ones((48, 24), np.int8),
                 cells={c: v for c, v in full.items() if c != "GS-single"}, blank={"GS-single": "T2W only"}),
            dict(name="Brief motion (Ta)", contrast="T2W", gt=gt, kmask=kmask, cells=full, tags={"True motion": "arm Ow"})]
    p = mfigures.record_figure(rows, cols, 12, (10, 4), 40, "Mild motion inside the T2W scan", ["one part", "another part"],
                               tmp_path / "fig_images_record.png", truth_cols=("True motion",))
    assert _png_ok(p)
    rec = json.loads(p.with_suffix(".json").read_text())
    assert rec["slice"] == 12 and rec["box"] == dict(y0=10, x0=4, edge=40) and len(rec["rows"]) == 2
    assert "GS-single" not in rec["rows"][0]["numbers"] and rec["rows"][1]["best"]["psnr"] == "MC-GS + M1-M3"
    assert rec["rows"][1]["arrow"] is not None


def test_caption_is_broken_between_words_when_one_part_is_wider_than_the_figure():
    part = " ".join(f"word{i}" for i in range(80))
    lines = mfigures._wrap(["short part", part], 5.0, 11)
    assert len(lines) > 2
    assert " ".join(lines).split() == ("short part;   " + part).split()        # nothing lost, order kept
    assert max(len(x) for x in lines) < 110                                    # 5 in at 11 pt holds about 80 letters


def test_view_box_is_the_head_with_a_margin_inside_the_slice():
    head = np.zeros((64, 48), dtype=bool)
    head[20:40, 2:30] = True
    other = np.zeros((64, 48), dtype=bool)
    other[25:45, 10:20] = True
    assert mfigures.view_box([head, other], 5) == (15, 50, 0, 35)             # y0, y1, x0, x1; clipped at the left edge


def test_mask_codes_mark_only_sampled_lines():
    train_of = np.array([[0, 0, 1], [1, 2, 2], [-1, 2, 0]])                    # the order covers the whole support
    mask = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])                         # the lines that were sampled
    codes = mfigures.mask_codes(mask, train_of, [False, True, False])
    assert codes.tolist() == [[1, 0, 2], [0, 1, 0], [1, 1, 0]]
    assert mfigures.mask_codes(mask, None, None).tolist() == mask.tolist()     # T1W: no motion to mark


def test_align_figure(tmp_path):
    assert _png_ok(mfigures.align_figure(_summary(), tmp_path / "fig_align_E2.png"))


def test_worst_slice_is_inside_the_brain():
    gt = np.zeros((10, 8, 8))
    rec = gt.copy()
    rec[2] += 5.0                                   # a large error outside the brain does not count
    rec[6, 3:5, 3:5] += 1.0
    brain = np.zeros(gt.shape, dtype=bool)
    brain[4:8, 2:6, 2:6] = True
    k, e = mfigures.worst_slice([(gt, rec), (gt, gt)], brain)
    assert k == 6 and e.shape == (8, 8) and e[3, 3] == 1.0 and e[0, 0] == 0.0
