"""Image warps of the simulator (numpy / scipy). All functions return vol o T^-1: the content of vol moved forward
by T (geom.py conventions). Deliberately a different numerical route from the adapter (which moves Gaussians and
re-renders), and a different non-rigid basis (a few large Gaussian RBFs)."""
import numpy as np
import scipy.ndimage as ndi

import geom
from common import N

PAD = 16
ORDER = 5            # quintic B-spline: the lowest blur scipy offers


def fourier_shift(vol, shift_mm):
    """Shift the content of a complex volume by shift_mm (voxels = mm) with a linear phase in k-space (periodic)."""
    k = np.fft.fftn(vol)
    for ax, s in enumerate(shift_mm):
        if s == 0:
            continue
        f = np.fft.fftfreq(vol.shape[ax])
        shape = [1, 1, 1]
        shape[ax] = -1
        k = k * np.exp(-2j * np.pi * f * s).reshape(shape)
    return np.fft.ifftn(k)


def _resample(vol, coords_mm):
    """vol sampled at positions r (mm, centre origin), given as an array (*shape, 3)."""
    shape = vol.shape
    c = (np.asarray(shape) - 1) / 2.0
    padded = np.pad(vol, PAD)
    idx = np.moveaxis(coords_mm + c + PAD, -1, 0)
    out = ndi.map_coordinates(padded.real, idx, order=ORDER, mode="constant", cval=0.0).astype(np.complex128)
    if np.iscomplexobj(vol):
        out += 1j * ndi.map_coordinates(padded.imag, idx, order=ORDER, mode="constant", cval=0.0)
    return out


def rigid(vol, g):
    """vol o g^-1. Rotation by spline resampling, translation by an exact Fourier shift."""
    vol = np.asarray(vol)
    if np.allclose(g.R, np.eye(3), atol=1e-14):
        rot = vol.astype(np.complex128)
    else:
        rot = _resample(vol, geom.grid_mm(vol.shape) @ g.R)          # rot(r) = vol(R^T r); row-vector form r @ R
    return fourier_shift(rot, g.t)


def rbf_field(seed, peak_mm, brain, brain_max_mm, shape=N, sigma_mm=(30.0, 50.0)):
    """Smooth displacement field u (*shape, 3), mm: three Gaussian RBFs in the lower part of the FOV (neck, jaw),
    one inside the brain; zero mean; peak norm = peak_mm; norm inside the brain mask <= brain_max_mm."""
    rng = np.random.default_rng(seed)
    r = geom.grid_mm(shape)
    half = np.asarray(shape) / 2.0
    u = np.zeros((*shape, 3))
    for k in range(4):
        if k < 3:
            cen = np.array([rng.uniform(0.40, 0.85) * half[0], rng.uniform(-0.5, 0.5) * half[1],
                            rng.uniform(-0.5, 0.5) * half[2]])
            amp = rng.uniform(0.7, 1.0)
        else:
            vox = np.argwhere(brain)[rng.integers(int(brain.sum()))]
            cen = vox - (np.asarray(shape) - 1) / 2.0
            amp = rng.uniform(0.3, 0.5)
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        s = rng.uniform(*sigma_mm)
        u += amp * np.exp(-((r - cen) ** 2).sum(-1) / (2 * s * s))[..., None] * d
    soft = ndi.gaussian_filter(brain.astype(np.float64), min(sigma_mm) / 4.0)
    soft /= soft.max()

    def shaped(a):
        f = u * (1.0 - (1.0 - a) * soft)[..., None]
        f = f - f.reshape(-1, 3).mean(0)
        return f * (peak_mm / np.linalg.norm(f, axis=-1).max())

    lo, hi = 0.0, 1.0
    f = shaped(hi)
    if np.linalg.norm(f, axis=-1)[brain].max() > brain_max_mm:
        assert np.linalg.norm(shaped(lo), axis=-1)[brain].max() <= brain_max_mm, "brain limit cannot be met"
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if np.linalg.norm(shaped(mid), axis=-1)[brain].max() <= brain_max_mm:
                lo = mid
            else:
                hi = mid
        f = shaped(lo)
    return f


def _interp_field(u, coords_mm):
    c = (np.asarray(u.shape[:3]) - 1) / 2.0
    idx = np.moveaxis(coords_mm + c, -1, 0)
    return np.stack([ndi.map_coordinates(u[..., k], idx, order=1, mode="nearest") for k in range(3)], axis=-1)


def nonrigid_then_rigid(vol, u, g, iters=8):
    """vol o T^-1 for T(r) = g(r + u(r)): invert g exactly, invert r -> r + u(r) by fixed-point iteration, then one
    spline resampling."""
    vol = np.asarray(vol)
    s = g.inverse().apply(geom.grid_mm(vol.shape))
    r = s.copy()
    for _ in range(iters):
        r = s - _interp_field(u, r)
    return _resample(vol, r)


def displacement(u, g, shape=N):
    """T(r) - r on the voxel grid, T(r) = g(r + u(r))."""
    r = geom.grid_mm(shape)
    return g.apply(r + u) - r
