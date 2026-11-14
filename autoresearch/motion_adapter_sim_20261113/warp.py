"""Image warps of the simulator (numpy / scipy). All functions return vol o T^-1: the content of vol moved forward
by T (geom.py conventions). Deliberately a different numerical route from the adapter (which moves Gaussians and
re-renders), and a different non-rigid basis (a few large Gaussian RBFs)."""
import numpy as np
import scipy.ndimage as ndi

import geom
from common import N

PAD = 16
ORDER = 5            # quintic B-spline on a 2x oversampled grid (non-rigid part only)


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


def _upsample2(vol):
    """Exact (Fourier zero-padding) 2x upsampling of a complex volume; sample j of the result sits at j/2."""
    n = np.asarray(vol.shape)
    k = np.fft.fftshift(np.fft.fftn(vol))
    big = np.zeros(tuple(2 * n), dtype=np.complex128)
    lo = n - n // 2
    big[lo[0]:lo[0] + n[0], lo[1]:lo[1] + n[1], lo[2]:lo[2] + n[2]] = k
    return np.fft.ifftn(np.fft.ifftshift(big)) * 8.0


def _resample(vol, coords_mm):
    """vol sampled at positions r (mm, centre origin), given as an array (*shape, 3). The volume is first upsampled
    2x exactly, so the spline's roll-off falls above the data's own band and the noise floor is kept."""
    shape = vol.shape
    c = (np.asarray(shape) - 1) / 2.0
    fine = _upsample2(np.pad(vol, PAD))
    idx = np.moveaxis(2.0 * (coords_mm + c + PAD), -1, 0)
    out = ndi.map_coordinates(fine.real, idx, order=ORDER, mode="constant", cval=0.0).astype(np.complex128)
    if np.iscomplexobj(vol):
        out += 1j * ndi.map_coordinates(fine.imag, idx, order=ORDER, mode="constant", cval=0.0)
    return out


def _shear(v, axis, other, k):
    """Shift every line along `axis` by k * (x_other - centre): an exact, invertible Fourier shear."""
    f = np.fft.fftfreq(v.shape[axis]).reshape([-1 if a == axis else 1 for a in range(3)])
    x = (np.arange(v.shape[other]) - (v.shape[other] - 1) / 2.0).reshape([-1 if a == other else 1 for a in range(3)])
    return np.fft.ifft(np.fft.fft(v, axis=axis) * np.exp(-2j * np.pi * f * k * x), axis=axis)


def _rotate_plane(v, i, j, angle):
    """Rotate the content by `angle`, axis i towards axis j, as three shears (Paeth): R = Sx Sy Sx."""
    if angle == 0.0:
        return v
    k1, k2 = -np.tan(angle / 2.0), np.sin(angle)
    return _shear(_shear(_shear(v, i, j, k1), j, i, k2), i, j, k1)


def rigid(vol, g):
    """vol o g^-1, exactly invertible: the rotation is R = R_axis2 R_axis1 R_axis0, each a three-shear Fourier
    rotation about the array centre (band-limited interpolation, no smoothing), the translation a Fourier shift."""
    from scipy.spatial.transform import Rotation
    vol = np.asarray(vol)
    out = vol.astype(np.complex128)
    if not np.allclose(g.R, np.eye(3), atol=1e-14):
        a0, a1, a2 = Rotation.from_matrix(g.R).as_euler("xyz")          # R = R2(a2) R1(a1) R0(a0)
        out = np.pad(out, PAD)
        out = _rotate_plane(out, 1, 2, a0)                               # about axis 0: axis 1 -> axis 2
        out = _rotate_plane(out, 2, 0, a1)                               # about axis 1: axis 2 -> axis 0
        out = _rotate_plane(out, 0, 1, a2)                               # about axis 2: axis 0 -> axis 1
        out = out[PAD:-PAD, PAD:-PAD, PAD:-PAD]
    return fourier_shift(out, g.t)


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
    """vol o T^-1 for T(r) = g(r + u(r)): the small non-rigid part by fixed-point inversion and one (oversampled)
    spline resampling, then the rigid part exactly."""
    vol = np.asarray(vol)
    if not np.any(u):
        return rigid(vol, g)
    s0 = geom.grid_mm(vol.shape)
    r = s0.copy()
    for _ in range(iters):
        r = s0 - _interp_field(u, r)
    return rigid(_resample(vol, r), g)


def displacement(u, g, shape=N):
    """T(r) - r on the voxel grid, T(r) = g(r + u(r))."""
    r = geom.grid_mm(shape)
    return g.apply(r + u) - r
