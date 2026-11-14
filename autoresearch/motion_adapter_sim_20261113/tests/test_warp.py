import numpy as np
import pytest

import geom
import warp
from helpers import ball_mask, blob_at, centroid, phantom


def test_integer_shift_equals_roll():
    v = phantom()
    np.testing.assert_allclose(warp.fourier_shift(v, (3, -2, 1)), np.roll(v, (3, -2, 1), (0, 1, 2)), atol=1e-5)


def test_rigid_roundtrip_under_half_percent():
    v = phantom()
    g = geom.rigid_from_pivot((1.0, 1.0, 3.0), (1.6, -1.0, 0.6), pivot_offset_mm=(15, 5, 0))
    back = warp.rigid(warp.rigid(v, g), g.inverse())
    m = np.abs(v) > 0.2 * np.abs(v).max()
    assert np.sqrt((np.abs(back - v)[m] ** 2).mean()) < 0.005 * np.sqrt((np.abs(v)[m] ** 2).mean())


def test_rigid_moves_content_forward():
    v = blob_at((32, 32, 24))
    g = geom.Rigid(np.eye(3), np.array([4.0, 0, 0]))
    assert centroid(np.abs(warp.rigid(v, g)))[0] == pytest.approx(36.0, abs=0.05)


def test_rigid_rotation_moves_points_as_the_transform_says():
    shape = (64, 64, 48)
    c = (np.asarray(shape) - 1) / 2.0
    r0 = np.array([12.0, -6.0, 4.0])
    g = geom.Rigid(geom.rotmat(np.deg2rad([0.0, 0.0, 10.0])), np.array([1.0, 0.5, -2.0]))
    out = warp.rigid(blob_at(c + r0, shape), g)
    np.testing.assert_allclose(centroid(np.abs(out)) - c, g.apply(r0), atol=0.05)


def test_rbf_field_limits_and_zero_mean():
    brain = ball_mask()
    u = warp.rbf_field(7, 1.5, brain, 0.8, shape=(64, 64, 48), sigma_mm=(8.0, 14.0))
    n = np.linalg.norm(u, axis=-1)
    assert u.shape == (64, 64, 48, 3)
    assert n.max() == pytest.approx(1.5, rel=0.15) and n[brain].max() <= 0.8 + 1e-6
    assert np.abs(u.reshape(-1, 3).mean(0)).max() < 1e-6
    assert np.argmax(n.max(axis=(1, 2))) > 64 * 0.55                    # the peak sits in the lower part
    assert not np.array_equal(u, warp.rbf_field(101, 1.5, brain, 0.8, shape=(64, 64, 48), sigma_mm=(8.0, 14.0)))


def test_nonrigid_then_rigid_moves_points_as_displacement_says():
    shape = (64, 64, 48)
    c = (np.asarray(shape) - 1) / 2.0
    brain = ball_mask()
    u = warp.rbf_field(7, 1.5, brain, 0.8, shape=shape, sigma_mm=(8.0, 14.0))
    g = geom.rigid_from_pivot((0.5, 0.5, 1.5), (0.8, -0.5, 0.3), pivot_offset_mm=(15, 5, 0))
    disp = warp.displacement(u, g, shape=shape)
    for r0 in ([14.0, 3.0, -2.0], [-8.0, -5.0, 6.0]):
        i0 = np.round(c + r0).astype(int)
        out = warp.nonrigid_then_rigid(blob_at(i0, shape, sigma=2.5), u, g)
        np.testing.assert_allclose(centroid(np.abs(out)) - i0, disp[tuple(i0)], atol=0.06)


def test_nonrigid_zero_field_equals_rigid():
    v = phantom()
    g = geom.rigid_from_pivot((0.5, 0.5, 1.5), (0.8, -0.5, 0.3), pivot_offset_mm=(15, 5, 0))
    a = warp.nonrigid_then_rigid(v, np.zeros((*v.shape, 3)), g)
    m = np.abs(v) > 0.2 * np.abs(v).max()
    assert np.abs(a - warp.rigid(v, g))[m].max() < 0.005 * np.abs(v).max()


def _noisy(shape=(64, 64, 48), seed=3, ball=True):
    """Like real data: object plus noise up to Nyquist, fading to zero near the array border. ball=True keeps the
    noise inside the inscribed k-space ball (what a rotation can carry without loss); ball=False is white noise on
    the whole k-space cube, whose corners leave the cube when the image is rotated."""
    import scipy.ndimage as ndi
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    if ball:
        f2 = sum(np.fft.fftfreq(n).reshape([-1 if a == i else 1 for a in range(3)]) ** 2 for i, n in enumerate(shape))
        noise = np.fft.ifftn(np.fft.fftn(noise) * (f2 < 0.47 ** 2))
    w = np.zeros(shape)
    w[10:-10, 10:-10, 10:-10] = 1.0
    return (phantom(shape) + 0.05 * noise) * ndi.gaussian_filter(w, 2.0)


def _hf_fraction(v):
    k = np.fft.fftn(v)
    f = np.sqrt(sum(np.fft.fftfreq(n).reshape([-1 if a == i else 1 for a in range(3)]) ** 2 for i, n in enumerate(v.shape)))
    return float((np.abs(k[f > 0.25]) ** 2).sum() / (np.abs(k) ** 2).sum())


def test_rigid_roundtrip_is_exact_for_noisy_data():
    v = _noisy()
    g = geom.rigid_from_pivot((0.5, 0.5, 1.5), (0.8, -0.5, 0.3), pivot_offset_mm=(15, 5, 0))
    back = warp.rigid(warp.rigid(v, g), g.inverse())
    inner = (slice(12, -12),) * 3
    assert np.sqrt((np.abs(back - v)[inner] ** 2).mean()) < 0.005 * np.sqrt((np.abs(v)[inner] ** 2).mean())
    assert _hf_fraction(warp.rigid(v, g)) > 0.97 * _hf_fraction(v)        # the noise floor is not smoothed away


def test_rigid_roundtrip_loses_only_the_cube_corners_of_white_noise():
    v, clean = _noisy(ball=False), _noisy(ball=False, seed=3) - _noisy(ball=False, seed=4)
    g = geom.rigid_from_pivot((0.5, 0.5, 1.5), (0.8, -0.5, 0.3), pivot_offset_mm=(15, 5, 0))
    back = warp.rigid(warp.rigid(v, g), g.inverse())
    inner = (slice(14, -14),) * 3
    noise_rms = np.sqrt((np.abs(clean)[inner] ** 2).mean() / 2)           # rms of one noise realisation
    assert np.sqrt((np.abs(back - v)[inner] ** 2).mean()) < 0.15 * noise_rms


def test_nonrigid_keeps_high_frequencies_of_noisy_data():
    v = _noisy()
    brain = ball_mask()
    u = warp.rbf_field(7, 1.5, brain, 0.8, shape=v.shape, sigma_mm=(8.0, 14.0))
    g = geom.rigid_from_pivot((0.5, 0.5, 1.5), (0.8, -0.5, 0.3), pivot_offset_mm=(15, 5, 0))
    out = warp.nonrigid_then_rigid(v, u, g)
    assert _hf_fraction(out) > 0.97 * _hf_fraction(v)
