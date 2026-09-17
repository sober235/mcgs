"""Shared fixtures. tiny_topic builds a miniature topic (3 coils, 16x16x16 grid, one Poisson-like mask) under tmp_path
and points RETRO_TOPIC at it, so Pair / scorers / extractors can be tested without the 5 GB data and without a GPU."""
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]          # scripts/: the modules under test live here
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def fft3c(x):
    ax = (-3, -2, -1)
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)


@pytest.fixture
def tiny_topic(tmp_path, monkeypatch):
    G, NC = (16, 16, 16), 3
    rng = np.random.default_rng(0)
    zz, yy, xx = np.meshgrid(*[np.linspace(-1, 1, g) for g in G], indexing="ij")
    head = (zz ** 2 + yy ** 2 + xx ** 2) < 0.8
    S = np.stack([np.exp(-((xx - c) ** 2 + yy ** 2)) * np.exp(1j * 0.3 * c * zz) for c in (-0.5, 0.0, 0.5)]).astype(np.complex64)
    S /= np.sqrt((np.abs(S) ** 2).sum(0, keepdims=True))
    top = tmp_path / "topic"
    for d in ("sim/gs", "sim/mask", "recon", "logs", "figures"):
        (top / d).mkdir(parents=True)
    m2 = rng.random((16, 16)) < 0.35
    m2[6:10, 6:10] = True
    m3 = np.broadcast_to(m2[None], G).astype(np.uint8)
    with h5py.File(top / "sim/mask/mask_af16_3d.h5", "w") as f:
        f.create_dataset("mask", data=m3)
    gts, scales = {}, {}
    for scan, amp in (("pre", 1.0), ("post", 1.3)):
        x = (amp * head * (1.0 + 0.2 * rng.random(G)) * np.exp(1j * 0.1 * yy)).astype(np.complex64)
        y = fft3c(S * x[None]).astype(np.complex64)
        with h5py.File(top / f"sim/gs/{scan}_kspace.h5", "w") as f:
            f.create_dataset("kspace", data=y)
        with h5py.File(top / f"sim/gs/{scan}_csm.h5", "w") as f:
            f.create_dataset("map", data=S)
        gts[scan] = x
        scales[f"scale_{scan}"] = float(np.quantile(np.abs(x), 0.999))
    np.savez(top / "sim/gt.npz", mask2d=m2, **gts, **scales)
    monkeypatch.setenv("RETRO_TOPIC", str(top))
    return dict(root=top, G=G, NC=NC, S=S, m2=m2, gts=gts, scales=scales)
