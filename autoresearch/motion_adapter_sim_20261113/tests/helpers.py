"""Small synthetic volumes shared by the tests."""
import numpy as np
import scipy.ndimage as ndi


def _grid(shape):
    return np.stack(np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij"), axis=-1)


def blob_at(center, shape=(64, 64, 48), sigma=3.0):
    d2 = ((_grid(shape) - np.asarray(center, dtype=np.float64)) ** 2).sum(-1)
    return np.exp(-d2 / (2 * sigma ** 2)).astype(np.complex128)


def phantom(shape=(64, 64, 48), seed=0, n=14, sigma=(3.5, 6.0)):
    """Smooth (effectively band-limited) complex object well inside the FOV: a sum of Gaussian blobs with phase."""
    rng = np.random.default_rng(seed)
    c = (np.asarray(shape) - 1) / 2.0
    v = np.zeros(shape, dtype=np.complex128)
    for _ in range(n):
        cen = c + rng.uniform(-0.22, 0.22, 3) * np.asarray(shape)
        v += rng.uniform(0.5, 1.0) * np.exp(1j * rng.uniform(-0.6, 0.6)) * blob_at(cen, shape, rng.uniform(*sigma))
    return v


def centroid(mag):
    return np.array(ndi.center_of_mass(mag))


def ball_mask(shape=(64, 64, 48), radius=14.0, offset=(-6.0, 0.0, 0.0)):
    c = (np.asarray(shape) - 1) / 2.0 + np.asarray(offset)
    return ((_grid(shape) - c) ** 2).sum(-1) <= radius ** 2


def wm_phantom(shape=(112, 112, 88)):
    """T1-like magnitude: scalp shell, darker rim (grey matter / CSF), bright interior (white matter), with texture."""
    g = _grid(shape)
    c = (np.asarray(shape) - 1) / 2.0
    rad = np.asarray(shape) * np.array([0.44, 0.44, 0.44])
    rho = np.sqrt((((g - c) / rad) ** 2).sum(-1))
    v = np.zeros(shape)
    v[rho <= 1.0] = 0.9            # scalp
    v[rho <= 0.93] = 0.15          # skull / CSF gap
    v[rho <= 0.86] = 0.55          # grey matter
    v[rho <= 0.74] = 1.0           # white matter
    rng = np.random.default_rng(1)
    v *= 1.0 + 0.03 * ndi.gaussian_filter(rng.normal(size=shape), 2.0) / 0.05
    return ndi.gaussian_filter(v, 0.7)
