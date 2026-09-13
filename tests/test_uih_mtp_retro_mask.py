"""Retrospective AF-12 mask selection, trimming and export (uih_mtp.retro_mask)."""
import numpy as np
import pytest

SHAPE = (40, 20)
CENTRE = (20, 10)


def _support():
    """An elliptical support truncated in PE, like the MTP scan's acquired band."""
    pe, spe = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    ell = ((pe - CENTRE[0]) / 19.0) ** 2 + ((spe - CENTRE[1]) / 9.5) ** 2 <= 1.0
    return ell & (pe < 30)


def _draw(rng, n):
    """A random pattern with n points anywhere in the plane."""
    m = np.zeros(SHAPE, dtype=bool)
    idx = rng.choice(SHAPE[0] * SHAPE[1], size=n, replace=False)
    m.reshape(-1)[idx] = True
    return m


def test_select_draw_takes_the_smallest_qualifying_draw():
    from uih_mtp.retro_mask import select_draw
    sup = _support()
    rng = np.random.default_rng(0)
    pool = []
    for name, n in (("a", 40), ("b", 400), ("c", 200)):
        while True:
            m = _draw(rng, n)
            if (m & sup).sum() >= 1:
                break
        pool.append((dict(name=name, nnz=int(m.sum())), m))
    counts = {meta["name"]: int((m & sup).sum()) for meta, m in pool}
    target = min(counts["b"], counts["c"])
    meta, mask = select_draw(pool, sup, target)
    assert meta["name"] == min((k for k, v in counts.items() if v >= target), key=lambda k: (counts[k], k))
    assert int((mask & sup).sum()) == counts[meta["name"]]


def test_select_draw_raises_when_no_draw_reaches_the_target():
    from uih_mtp.retro_mask import select_draw
    sup = _support()
    rng = np.random.default_rng(1)
    pool = [(dict(name="a"), _draw(rng, 20))]
    with pytest.raises(ValueError, match="target"):
        select_draw(pool, sup, 10_000)


def test_trim_keeps_exactly_the_budget_and_drops_the_outermost():
    from uih_mtp.retro_mask import trim_to_budget
    sup = _support()
    mask = sup.copy()                       # every support point sampled
    target = 100
    out = trim_to_budget(mask, sup, target, CENTRE)
    assert out.sum() == target
    assert not (out & ~sup).any()
    r_keep = np.hypot(*(np.nonzero(out)[i] - CENTRE[i] for i in (0, 1)))
    dropped = mask & ~out
    r_drop = np.hypot(*(np.nonzero(dropped)[i] - CENTRE[i] for i in (0, 1)))
    assert r_keep.max() <= r_drop.min() + 1e-9


def test_trim_never_drops_a_protected_point():
    from uih_mtp.retro_mask import calib_square, trim_to_budget
    sup = _support()
    cal = calib_square(SHAPE, CENTRE, 6) & sup
    mask = sup.copy()
    out = trim_to_budget(mask, sup, int(cal.sum()) + 5, CENTRE, protect=cal)
    assert (out & cal).sum() == cal.sum()
    assert out.sum() == cal.sum() + 5


def test_trim_is_deterministic():
    from uih_mtp.retro_mask import trim_to_budget
    sup = _support()
    a = trim_to_budget(sup.copy(), sup, 90, CENTRE, seed=0)
    b = trim_to_budget(sup.copy(), sup, 90, CENTRE, seed=0)
    assert np.array_equal(a, b)


def test_calib_square_is_centred_and_the_right_size():
    from uih_mtp.retro_mask import calib_square
    cal = calib_square(SHAPE, CENTRE, 6)
    assert cal.sum() == 36
    pe, spe = np.nonzero(cal)
    assert pe.min() == CENTRE[0] - 3 and pe.max() == CENTRE[0] + 2
    assert spe.min() == CENTRE[1] - 3 and spe.max() == CENTRE[1] + 2


def test_exports_have_the_keys_shapes_and_dtypes_the_framework_reads(tmp_path):
    import h5py
    import scipy.io
    from uih_mtp.bartio import read_cfl
    from uih_mtp.retro_mask import (broadcast_readout, write_mask_h5, write_mask_mat,
                                    write_pattern_cfl, write_support_h5)
    sup = _support()
    omega = sup & (np.arange(SHAPE[0])[:, None] % 3 == 0)
    n_ro = 5
    write_mask_mat(tmp_path / "m.mat", omega)
    got = scipy.io.loadmat(tmp_path / "m.mat")["mask"]
    assert got.shape == SHAPE and got.dtype == np.float32 and np.array_equal(got > 0.5, omega)

    m3 = broadcast_readout(omega, n_ro)
    assert m3.shape == (n_ro, *SHAPE) and m3.dtype == np.uint8
    assert np.array_equal(m3[0].astype(bool), omega) and (m3 == m3[0:1]).all()
    write_mask_h5(tmp_path / "m.h5", m3)
    with h5py.File(tmp_path / "m.h5", "r") as f:
        assert f["mask"].dtype == np.uint8 and np.array_equal(np.array(f["mask"]), m3)

    write_support_h5(tmp_path / "s.h5", sup)
    with h5py.File(tmp_path / "s.h5", "r") as f:
        assert f["support"].dtype == np.uint8 and np.array_equal(np.array(f["support"]).astype(bool), sup)

    write_pattern_cfl(str(tmp_path / "pat"), omega, n_ro)
    pat = read_cfl(str(tmp_path / "pat"))
    assert pat.shape == (n_ro, *SHAPE, 1) and pat.dtype == np.complex64
    assert np.array_equal(pat[..., 0].real.astype(bool), np.broadcast_to(omega, (n_ro, *SHAPE)))
    assert pat.imag.max() == 0.0


def test_load_support_splits_the_plane(tmp_path):
    from uih_mtp.retro_mask import load_support
    sup = _support()
    np.savez(tmp_path / "masks.npz", m_acq=sup & (np.arange(SHAPE[0])[:, None] % 2 == 0),
             m_grappa=sup & (np.arange(SHAPE[0])[:, None] % 2 == 1), m_zero=~sup,
             envelope=sup, acs=np.array([12, 16]))
    got_sup, got_zero, acs = load_support(tmp_path / "masks.npz")
    assert np.array_equal(got_sup, sup) and np.array_equal(got_zero, ~sup)
    assert not (got_sup & got_zero).any() and (got_sup | got_zero).all()
    assert acs == (12, 16)
