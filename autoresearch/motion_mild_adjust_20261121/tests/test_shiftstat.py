"""The linearised shift of every echo train (M3's detection statistic) on a small phantom with exact Fourier shifts."""
import numpy as np
import pytest

import shiftstat

SHAPE = (32, 32, 24)


def _fft(x):
    ax = (-3, -2, -1)
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)


def _shifted_k(k, d):
    """k-space of the object moved by d voxels (f(x - d)): a phase ramp."""
    ramp = 1.0
    for j, n in enumerate(SHAPE):
        kk = (np.arange(n) - n // 2) / n
        sh = [1, 1, 1]
        sh[j] = n
        ramp = ramp * np.exp(-2j * np.pi * kk.reshape(sh) * d[j])
    return k * ramp


def _setup(seed=0, n_trains=12):
    from helpers import blob_at, phantom
    rng = np.random.default_rng(seed)
    vol = phantom(SHAPE, seed=3) * blob_at((16, 16, 12), SHAPE, sigma=6.0)
    csm = np.stack([blob_at((8, 16, 12), SHAPE, sigma=14.0), blob_at((24, 16, 12), SHAPE, sigma=14.0)]).astype(np.complex128)
    mask = (rng.random(SHAPE[1:]) < 0.5).astype(np.float64)                 # sampled (ky, kz) lines
    train_of = np.where(mask > 0, rng.integers(0, n_trains, SHAPE[1:]), -1)  # every train spans k-space
    return vol, csm, mask, train_of, n_trains


def _measure(vol, csm, mask, train_of, shift_of_train):
    k0 = _fft(csm * vol)
    y = np.zeros_like(k0)
    for t, d in enumerate(shift_of_train):
        sel = (train_of == t)[None, None]
        y = y + np.where(sel, _shifted_k(k0, d), 0.0)
    return y * mask


def test_shift_of_every_train_is_recovered():
    vol, csm, mask, train_of, n = _setup()
    true = np.zeros((n, 3))
    true[5:] = (0.30, -0.20, 0.40)                                          # a lasting shift from train 5 on
    y = _measure(vol, csm, mask, train_of, true)
    sums = shiftstat.line_sums_np(vol, csm, y, mask, _fft)
    d = shiftstat.train_shift(sums, train_of, n)
    assert d.shape == (n, 3)
    assert np.abs(d[:5]).max() < 1e-6                                       # unmoved trains: exactly no shift
    assert np.abs(d[5:] - true[5:]).max() < 0.08, d[5:]                     # linearisation error only (one train)
    assert np.abs(np.median(d[5:], 0) - true[5]).max() < 0.02


def test_a_model_between_the_two_positions_sees_both_states_with_opposite_signs():
    """What defeats the residual energy: the model sits between the two positions. The difference of the shifts
    of the two states is still the whole shift."""
    vol, csm, mask, train_of, n = _setup()
    true = np.zeros((n, 3))
    true[6:] = (0.0, 0.4, -0.3)
    y = _measure(vol, csm, mask, train_of, true)
    mid = _ifft(_shifted_k(_fft(vol), (0.0, 0.2, -0.15)))                 # the model half-way between the positions
    sums = shiftstat.line_sums_np(mid, csm, y, mask, _fft)
    d = shiftstat.train_shift(sums, train_of, n)
    lo, hi = np.median(d[:6], 0), np.median(d[6:], 0)
    assert np.abs((hi - lo) - true[6]).max() < 0.03, (lo, hi)
    half = np.array([0.0, 0.2, -0.15])                                      # opposite signs, half the shift each
    assert np.abs(lo + half).max() < 0.03 and np.abs(hi - half).max() < 0.03


def _ifft(k):
    ax = (-3, -2, -1)
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k, axes=ax), axes=ax, norm="ortho"), axes=ax)


def test_empty_train_gives_nan():
    vol, csm, mask, train_of, n = _setup()
    train_of = np.where(train_of == 3, -1, train_of)
    y = _measure(vol, csm, mask * (train_of >= 0), train_of, np.zeros((n, 3)))
    d = shiftstat.train_shift(shiftstat.line_sums_np(vol, csm, y, mask * (train_of >= 0), _fft), train_of, n)
    assert np.isnan(d[3]).all() and np.isfinite(np.delete(d, 3, axis=0)).all()


def test_torch_sums_match_numpy():
    torch = pytest.importorskip("torch")
    vol, csm, mask, train_of, n = _setup()
    true = np.zeros((n, 3))
    true[4:] = (0.2, 0.1, -0.3)
    y = _measure(vol, csm, mask, train_of, true)
    ax = (-3, -2, -1)
    tfft = lambda x: torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=ax), dim=ax, norm="ortho"), dim=ax)
    a = shiftstat.line_sums_np(vol, csm, y, mask, _fft)
    b = shiftstat.line_sums(torch.from_numpy(vol), torch.from_numpy(csm), torch.from_numpy(y),
                            torch.from_numpy(mask), tfft, chunk=1)
    for k in ("A0", "A1", "A2", "Q0", "Q1", "E"):
        np.testing.assert_allclose(b[k], a[k], rtol=1e-9, atol=1e-12)
    # the trainer holds the sampling mask per coil and voxel: the same sums
    full = torch.from_numpy(mask).reshape(1, 1, *SHAPE[1:]).expand(csm.shape[0], *SHAPE).contiguous()
    c = shiftstat.line_sums(torch.from_numpy(vol), torch.from_numpy(csm), torch.from_numpy(y), full, tfft)
    np.testing.assert_allclose(shiftstat.train_shift(c, train_of, n), shiftstat.train_shift(a, train_of, n), atol=1e-9)


def test_changepoints_of_a_vector_series():
    rng = np.random.default_rng(0)
    d = 0.02 * rng.standard_normal((110, 3))
    d[50:, 1] -= 0.33
    d[50:, 2] -= 0.48
    assert shiftstat.changepoints_vec(d, 0.10, 8.0, 3) == [50]
    assert shiftstat.changepoints_vec(0.02 * rng.standard_normal((110, 3)), 0.10, 8.0, 3) == []
    d[30:33] += 2.0                                                         # three trains during a movement: no state
    assert shiftstat.changepoints_vec(d, 0.10, 8.0, 3) == [50]
    d[5] = np.nan                                                           # a train without data
    assert shiftstat.changepoints_vec(d, 0.10, 8.0, 3) == [50]
    two = 0.02 * rng.standard_normal((110, 3))
    two[40:, 0] += 0.3
    two[80:, 2] += 0.4
    assert shiftstat.changepoints_vec(two, 0.10, 8.0, 3) == [40, 80]
    assert shiftstat.changepoints_vec(two, 0.10, 8.0, 2) in ([40], [80])     # at most max_states - 1


def test_state_shifts_are_medians_against_the_reference_state():
    rng = np.random.default_rng(1)
    d = 0.02 * rng.standard_normal((110, 3)) + np.array([0.1, 0.0, -0.1])    # the model itself is a little off
    d[60:] += (0.0, -0.33, -0.48)
    state = np.array([0] * 60 + [1] * 50)
    s = shiftstat.state_shifts(d, state, 2)
    assert s.shape == (2, 3) and not s[0].any()
    assert np.abs(s[1] - np.array([0.0, -0.33, -0.48])).max() < 0.02


def test_changepoints_vec_none_on_heavy_tailed_stationary_noise():
    """Review Focus 3 on the detector of the frozen settings (per-train shift, three components)."""
    for seed in range(200):
        d = 0.03 * np.random.default_rng(seed).standard_t(3, size=(110, 3))
        assert shiftstat.changepoints_vec(d, 0.10, 8.0, 3) == [], seed
