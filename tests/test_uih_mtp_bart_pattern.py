"""`bart pics -p` is honoured: measured zeros are not the same as unsampled (stage-2 spec, section 5)."""
import os
import subprocess

import numpy as np
import pytest

BART = "/home/congcongliu/src/bart/bart"
pytestmark = pytest.mark.skipif(not os.path.exists(BART), reason="BART not installed")


def _pics(tmp, ksp, maps, pattern=None, lam=0.001, iters=10):
    from uih_mtp.bartio import read_cfl, write_cfl
    tmp.mkdir(parents=True, exist_ok=True)
    write_cfl(str(tmp / "k"), ksp)
    write_cfl(str(tmp / "m"), maps)
    cmd = [BART, "pics"]
    if pattern is not None:
        write_cfl(str(tmp / "p"), pattern)
        cmd += ["-p", str(tmp / "p")]
    cmd += ["-l1", "-r", str(lam), "-i", str(iters), "-S", str(tmp / "k"), str(tmp / "m"), str(tmp / "out")]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         env=dict(os.environ, OMP_NUM_THREADS="2"))
    assert res.returncode == 0, res.stdout + res.stderr
    return np.squeeze(read_cfl(str(tmp / "out")))


def _case(seed=0):
    rng = np.random.default_rng(seed)
    nx, ny, nz, nc = 16, 12, 1, 4
    x = rng.standard_normal((nx, ny, nz)) + 1j * rng.standard_normal((nx, ny, nz))
    maps = (rng.standard_normal((nx, ny, nz, nc)) + 1j * rng.standard_normal((nx, ny, nz, nc))).astype(np.complex64)
    maps /= np.sqrt((np.abs(maps) ** 2).sum(-1, keepdims=True))
    k = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x[..., None] * maps, axes=(0, 1, 2)),
                                    axes=(0, 1, 2), norm="ortho"), axes=(0, 1, 2))
    omega = np.zeros((nx, ny, nz), dtype=bool)
    omega[:, ::2] = True
    zeros = np.zeros((nx, ny, nz), dtype=bool)
    zeros[:, 1::4] = True                          # "measured zero" columns, disjoint from omega
    return k.astype(np.complex64) * omega[..., None], maps, omega, zeros


def test_pattern_of_the_nonzero_support_reproduces_the_inferred_pattern(tmp_path):
    ksp, maps, omega, _ = _case()
    a = _pics(tmp_path / "a", ksp, maps)
    b = _pics(tmp_path / "b", ksp, maps, pattern=omega.astype(np.complex64)[..., None])
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5 * np.abs(a).max())


def test_adding_measured_zeros_to_the_pattern_changes_the_result(tmp_path):
    ksp, maps, omega, zeros = _case()
    a = _pics(tmp_path / "a", ksp, maps)
    c = _pics(tmp_path / "c", ksp, maps, pattern=(omega | zeros).astype(np.complex64)[..., None])
    rel = np.linalg.norm(c - a) / np.linalg.norm(a)
    assert rel > 1e-3, rel
