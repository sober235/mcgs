import numpy as np

import geom
import schedule
from mcommon import CONDITIONS2


def ellipse_support(shape=(256, 176)):
    ky, kz = np.meshgrid(np.arange(shape[0]) - shape[0] // 2, np.arange(shape[1]) - shape[1] // 2, indexing="ij")
    return (ky / (shape[0] / 2)) ** 2 + (kz / (shape[1] / 2)) ** 2 <= 1.0


def test_linear_is_unchanged():
    sup = ellipse_support()
    a, b = schedule.build(sup), schedule.build(sup, order="linear")
    assert sorted(a) == sorted(b)
    for k in a:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k
    first = np.argwhere(a["train_of"] == 0)
    assert first[:, 1].max() <= np.argwhere(a["train_of"] == 1)[:, 1].min()        # still kz outer


def test_interleaved_spreads_every_train():
    sup = ellipse_support()
    s = schedule.build(sup, order="interleaved")
    lin = schedule.build(sup)
    assert sorted(s) == sorted(lin) and s["n_trains"] == lin["n_trains"]
    assert (s["train_of"][~sup] == -1).all() and s["train_of"][sup].min() == 0
    kz = np.argwhere(s["train_of"] == 3)[:, 1]
    assert kz.max() - kz.min() > 0.8 * np.ptp(np.argwhere(sup)[:, 1])              # one train spans the kz range
    assert np.ptp(np.bincount(s["train_of"][sup])) <= 1                            # trains differ by one point at most
    assert s["train_of"][128, 88] == s["ref_train"]
    flat = s["train_of"].T.reshape(-1)                                             # raster: kz outer, ky inner
    assert np.array_equal(flat[flat >= 0], np.arange(int(sup.sum())) % s["n_trains"])


def test_unknown_order_is_refused():
    try:
        schedule.build(ellipse_support(), order="spiral")
    except ValueError:
        return
    raise AssertionError("an unknown order must raise ValueError")


def test_pulse_returns_to_reference():
    ev = [dict(kind="pulse", at=0.40, dur=0.05, theta_deg=(0.0, 0.0, 1.0), t_mm=(0.5, 0.0, 0.0))]
    frac = np.arange(101) / 100.0
    rel = schedule.poses(ev, frac, ref_train=0)
    want = geom.rigid_from_pivot((0.0, 0.0, 1.0), (0.5, 0.0, 0.0))
    inside = [n for n, f in enumerate(frac) if 0.40 <= f < 0.45]
    assert inside == [40, 41, 42, 43, 44]
    for n in range(101):
        if n in inside:
            assert np.allclose(rel[n].R, want.R) and np.allclose(rel[n].t, want.t)
        else:
            assert np.allclose(rel[n].R, np.eye(3)) and np.allclose(rel[n].t, 0.0)


def test_soft_amp():
    frac = np.arange(201) / 200.0
    a = schedule.soft_amp(CONDITIONS2["Tb"]["intra"], frac)
    for f, v in zip(frac, a):
        if 0.25 <= f < 0.29 or 0.70 <= f < 0.74:
            want = 1.0
        elif f >= 0.74:
            want = 0.1
        else:
            want = 0.0
        assert abs(v - want) < 1e-12, (f, v, want)
    assert schedule.soft_amp([dict(kind="drift", theta_deg=(0, 0, 1), t_mm=(0, 0, 0))], frac).max() == 0.0
