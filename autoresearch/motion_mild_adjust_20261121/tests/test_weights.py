import numpy as np

import schedule
import weights as W
from test_order import ellipse_support

Z0, Z1, CAP = 4.0, 8.0, 0.25


def _w(rho, groups=None):
    return W.weights_from_z(W.robust_z(rho, groups), Z0, Z1, CAP)


def test_clean_trains_all_weight_one():
    for seed in range(20):
        rho = 1.0 + 0.03 * np.random.default_rng(seed).normal(size=110)
        assert (_w(rho) == 1.0).all(), seed


def test_contaminated_trains_are_found():
    rng = np.random.default_rng(1)
    rho = 1.0 + 0.03 * rng.normal(size=110)
    bad = np.array([40, 41, 42, 43, 44, 45])
    rho[bad] *= 1.5
    w = _w(rho)
    truth = np.zeros(110, dtype=bool)
    truth[bad] = True
    assert (w[bad] == 0.0).all() and (w[~truth] == 1.0).all()
    assert W.auc(W.robust_z(rho), truth) == 1.0


def test_ramp_and_cap():
    z = np.zeros(100)
    z[:3] = [Z0, 0.5 * (Z0 + Z1), Z1]
    assert W.weights_from_z(z, Z0, Z1, CAP)[:3].tolist() == [1.0, 0.5, 0.0]
    z = np.zeros(100)
    z[:40] = np.linspace(20.0, 9.0, 40)                    # 40 % above the line
    w = W.weights_from_z(z, Z0, Z1, CAP)
    assert (w < 1.0).sum() == 25 and (w[:25] == 0.0).all() and (w[25:] == 1.0).all()      # the largest z stay down


def test_degenerate_trains_keep_weight_one():
    assert (_w(np.full(50, 1.3)) == 1.0).all()             # MAD = 0
    rng = np.random.default_rng(2)
    rho = 1.0 + 0.03 * rng.normal(size=60)
    with_nan = rho.copy()
    with_nan[[3, 17]] = np.nan                             # trains without a sampled point
    z_ref = W.robust_z(np.delete(rho, [3, 17]))
    z = W.robust_z(with_nan)
    assert np.isfinite(z).all() and z[3] == 0.0 and z[17] == 0.0
    assert np.allclose(np.delete(z, [3, 17]), z_ref)
    assert (_w(with_nan)[[3, 17]] == 1.0).all()
    assert (_w(np.array([1.0, np.nan, np.nan])) == 1.0).all()


def test_groups_are_scored_separately():
    rng = np.random.default_rng(3)
    rho = np.concatenate([1.0 + 0.03 * rng.normal(size=60), 2.0 + 0.06 * rng.normal(size=40)])
    groups = np.repeat([0, 1], [60, 40])
    assert (_w(rho, groups) == 1.0).all()                  # each state has its own level
    assert (_w(rho) < 1.0).any()                           # scored as one group, the higher level is down-weighted


def _kspace(order, seed=0):
    sup = ellipse_support((96, 64))
    sch = schedule.build(sup, etl=64, order=order)
    ky, kz = np.meshgrid(np.arange(96) - 48, np.arange(64) - 32, indexing="ij")
    r = np.sqrt((ky / 48.0) ** 2 + (kz / 32.0) ** 2)
    rng = np.random.default_rng(seed)
    e = 100.0 ** (-r) * (1.0 + 0.05 * rng.normal(size=r.shape))          # energy falls by 100 from centre to edge
    sampled = sup & (rng.random(r.shape) < 0.25)
    return sch, e, sampled


def test_shell_normalisation():
    for order in ("linear", "interleaved"):
        sch, e, sampled = _kspace(order)
        rho = W.train_stat(e, sampled, sch["train_of"], sch["n_trains"], W.shell_index(e.shape, 24))
        assert rho.shape == (sch["n_trains"],) and np.isfinite(rho).all()
        assert (_w(rho) == 1.0).all(), order
        bad = [5, 6]
        e2 = e.copy()
        e2[np.isin(sch["train_of"], bad)] *= 2.0
        w = _w(W.train_stat(e2, sampled, sch["train_of"], sch["n_trains"], W.shell_index(e.shape, 24)))
        assert (w[bad] == 0.0).all() and (np.delete(w, bad) == 1.0).all(), order


def test_train_without_samples_is_nan():
    sch, e, sampled = _kspace("linear")
    sampled = sampled & (sch["train_of"] != 4)
    rho = W.train_stat(e, sampled, sch["train_of"], sch["n_trains"], W.shell_index(e.shape, 24))
    assert np.isnan(rho[4]) and np.isfinite(np.delete(rho, 4)).all()


def test_shell_index():
    s = W.shell_index((96, 64), 24)
    assert s.shape == (96, 64) and s.min() == 0 and s.max() == 23 and s[48, 32] == 0
    assert s[48, 0] >= s[48, 16] >= s[48, 31]


def test_changepoint_found():
    for seed in range(10):
        rng = np.random.default_rng(seed)
        rho = 1.0 + 0.03 * rng.normal(size=110)
        rho[66:] *= 1.2
        cp = W.changepoints(rho, 0.10, 8.0, 3)
        assert len(cp) == 1 and abs(cp[0] - 66) <= 1, (seed, cp)


def test_two_changepoints_and_the_limit():
    rng = np.random.default_rng(0)
    rho = 1.0 + 0.02 * rng.normal(size=120)
    rho[40:80] += 0.3
    rho[80:] += 0.6
    cp = W.changepoints(rho, 0.10, 8.0, 3)
    assert len(cp) == 2 and abs(cp[0] - 40) <= 1 and abs(cp[1] - 80) <= 1
    assert len(W.changepoints(rho, 0.10, 8.0, 2)) == 1     # max_states bounds the number of states
    assert W.changepoints(rho, 0.10, 8.0, 1) == []


def test_changepoints_none_on_stationary_noise():
    for seed in range(200):
        rho = 1.0 + 0.03 * np.random.default_rng(seed).standard_t(3, size=110)
        assert W.changepoints(rho, 0.10, 8.0, 3) == [], seed


def test_pulses_are_not_states():
    rng = np.random.default_rng(4)
    rho = 1.0 + 0.03 * rng.normal(size=110)
    rho[27:32] *= 1.6
    rho[77:82] *= 1.6
    assert W.changepoints(rho, 0.10, 8.0, 3) == []


def test_short_tail_is_not_a_state():
    rng = np.random.default_rng(5)
    rho = 1.0 + 0.03 * rng.normal(size=110)
    rho[104:] *= 1.5                                       # 6 trains < 10 % of 110
    assert W.changepoints(rho, 0.10, 8.0, 3) == []


def test_auc():
    assert W.auc(np.array([0.1, 0.2, 0.9, 0.8]), np.array([False, False, True, True])) == 1.0
    assert W.auc(np.array([0.9, 0.8, 0.1, 0.2]), np.array([False, False, True, True])) == 0.0
    assert W.auc(np.array([1.0, 1.0, 1.0, 1.0]), np.array([False, False, True, True])) == 0.5
    assert np.isnan(W.auc(np.array([1.0, 2.0]), np.array([False, False])))


def test_median_statistic_ignores_a_few_strong_lines():
    """In real data the residual near the k-space centre differs strongly from line to line, so a clean train can
    hold a few lines far above their shell's level (measured: such trains reach z = 11 with the sum, without any
    motion). The median over the lines of energy / expected energy does not react to them and still finds a train
    whose lines are all raised."""
    sch, e, sampled = _kspace("interleaved")
    tr, n, shells = sch["train_of"], sch["n_trains"], W.shell_index(e.shape, 24)
    odd, bad = 9, [5, 6]
    e2 = e.copy()
    for y, z in np.argwhere(sampled & (tr == odd))[:2]:
        e2[y, z] *= 40.0
    e2[np.isin(tr, bad)] *= 2.0
    w_sum = _w(W.train_stat(e2, sampled, tr, n, shells))
    assert w_sum[odd] == 0.0                                              # the sum is carried by the two lines
    rho = W.train_stat(e2, sampled, tr, n, shells, kind="median")
    w = _w(rho)
    assert rho.shape == (n,) and w[odd] == 1.0 and (w[bad] == 0.0).all() and (np.delete(w, bad) == 1.0).all()
    assert np.array_equal(W.train_stat(e2, sampled, tr, n, shells, kind="sum"), W.train_stat(e2, sampled, tr, n, shells))
    gone = sampled & (tr != 4)
    assert np.isnan(W.train_stat(e2, gone, tr, n, shells, kind="median")[4])
    try:
        W.train_stat(e2, sampled, tr, n, shells, kind="mean")
        assert False, "an unknown kind must be refused"
    except ValueError:
        pass
