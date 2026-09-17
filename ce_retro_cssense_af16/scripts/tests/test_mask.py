import numpy as np
import pytest

import make_mask_af16 as M


def test_poisson_is_deterministic_rectangular_with_full_acs(tmp_path):
    a = M.poisson(4.0, 1, tmp_path)
    b = M.poisson(4.0, 1, tmp_path / "again")
    assert a.shape == (M.NY, M.NZ) and a.dtype == bool
    assert np.array_equal(a, b)
    assert M.acs_block(a).all()
    assert a[M.radius() > 1.0].sum() > 0            # no elliptical shutter: the corners are sampled too


def test_trim_keeps_acs_and_hits_target():
    rng = np.random.default_rng(0)
    m = rng.random((M.NY, M.NZ)) < 0.08
    y0, z0 = M.NY // 2 - M.ACS // 2, M.NZ // 2 - M.ACS // 2
    m[y0:y0 + M.ACS, z0:z0 + M.ACS] = True
    t = M.trim_to_target(m, M.radius(), M.TARGET)
    assert t.sum() == M.TARGET
    assert M.acs_block(t).all()
    assert not (t & ~m).any()                      # trimming only removes points


def test_find_mask_exact_af16(tmp_path):
    m, info = M.find_mask(workdir=tmp_path)
    assert m.sum() == M.TARGET == 2552
    assert M.acs_block(m).all()
    assert info["af"] == pytest.approx(16.0)
    assert isinstance(info["n_bart_calls"], int) and info["n_bart_calls"] > 0
    assert info["n_bart_calls"] == len(info["visited"])
    assert len(info["visited"]) > 0
    assert all(isinstance(a, float) and isinstance(n, int) for a, n in info["visited"])
    assert info["count_raw"] - info["trimmed"] == M.TARGET
