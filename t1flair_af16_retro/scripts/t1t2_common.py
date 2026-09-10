"""t1t2_common.py -- data loading and the scoring gauge shared by every non-GS arm here.

The load/normalise/score code is transcribed VERBATIM from scripts/recon_classical.py (which
produced the ZF / CG-SENSE / CS-SENSE numbers) so that ZS-SSDU, 3D-DIP and 3D-INR land on
exactly the same gauge. scripts/check_gauge.py re-scores an already-exported classical volume
through this module and asserts it reproduces the recorded numbers.

Gauge:
    GT      = sum_c conj(S_c) . ifft3c(y_full_c)        from the FULL k-space
    scale   = P99.9 of |GT|, applied to GT and to y
    metrics = paired_gs.metrics.evaluate_reconstruction (whole@max)
              + diag_artefact.score_headonly, with hfen / sharp_ratio / grain transcribed
              verbatim from arc/cycle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.fft as tfft
import torch.nn.functional as F

ROOT = Path("/data0/congcong/code/GS/mcgs_af16_wt")
if str(ROOT / "paired_gs") not in sys.path:
    sys.path.insert(0, str(ROOT / "paired_gs"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from metrics import evaluate_reconstruction                      # noqa: E402
from diag_artefact.headonly import score_headonly                # noqa: E402

R = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_retro")


# ---- centred orthonormal FFTs ---------------------------------------------------------------
def _c(fn, x, dims):
    return tfft.fftshift(fn(tfft.ifftshift(x, dim=dims), dim=dims, norm="ortho"), dim=dims)


def fft3c(x):  return _c(tfft.fftn,  x, (-3, -2, -1))
def ifft3c(x): return _c(tfft.ifftn, x, (-3, -2, -1))
def fft2c(x):  return _c(tfft.fftn,  x, (-2, -1))
def ifft2c(x): return _c(tfft.ifftn, x, (-2, -1))
def ifft1c(x, d): return _c(tfft.ifftn, x, (d,))


# ---- metrics transcribed verbatim from arc/cycle.py (do not "improve") ----------------------
def hfen(rec, gt):
    def lap(v):
        return (6.0 * v - torch.roll(v, 1, -3) - torch.roll(v, -1, -3)
                - torch.roll(v, 1, -2) - torch.roll(v, -1, -2)
                - torch.roll(v, 1, -1) - torch.roll(v, -1, -1))
    g = gt / (gt.max() + 1e-9); r = rec / (gt.max() + 1e-9); lg = lap(g)
    return float(torch.linalg.norm(lap(r) - lg) / (torch.linalg.norm(lg) + 1e-9))


def _gmag(v):
    gx = torch.zeros_like(v); gy = torch.zeros_like(v); gz = torch.zeros_like(v)
    gx[:-1] = v[1:] - v[:-1]; gy[:, :-1] = v[:, 1:] - v[:, :-1]; gz[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
    return torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)


def sharp_ratio(rec, gt):
    brain = gt > 0.05 * gt.max()
    return float(_gmag(rec)[brain].mean() / (_gmag(gt)[brain].mean() + 1e-9))


def _quantile(t, q):
    if t.numel() <= 16_000_000:
        return torch.quantile(t, q)
    return torch.tensor(float(np.quantile(t.detach().cpu().numpy(), q)), device=t.device, dtype=t.dtype)


def grain(rec, gt):
    g = gt / (gt.max() + 1e-9); r = rec / (gt.max() + 1e-9)
    sm = F.avg_pool3d(r[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    hp = r - sm
    brain = g > 0.05 * g.max()
    gg = _gmag(g)
    thr = _quantile(gg[brain], 0.30)
    flat = brain & (gg <= thr)
    return float(hp[flat].std())


# ---- data -----------------------------------------------------------------------------------
class Pair:
    """Everything a reconstruction needs, already on the device and in the normalised gauge."""

    def __init__(self, contrast: str, dev, chunk: int = 4):
        self.contrast, self.dev, self.chunk = contrast, dev, chunk
        with h5py.File(R / f"data/ksp_{contrast}.h5", "r") as f:
            y_full = torch.from_numpy(f["kspace"][:]).to(torch.complex64)
        with h5py.File(R / f"data/csm_espirit_{contrast}.h5", "r") as f:
            S = torch.from_numpy(f["map"][:]).to(torch.complex64)
        with h5py.File(R / "data/mask_af16_3d.h5", "r") as f:
            m3 = torch.from_numpy(f["mask"][:]).to(torch.float32)
        assert y_full.shape == S.shape, (y_full.shape, S.shape)
        self.y_full = y_full.to(dev); self.S = S.to(dev)
        self.m3 = (m3 > 0.5).to(dev)
        self.NC = self.y_full.shape[0]
        self.G = tuple(self.m3.shape)
        # the mask is broadcast along the readout (axis 0); every 2-D method below relies on it
        assert bool((self.m3 == self.m3[0:1]).all()), "mask is not constant along the readout axis"
        self.m2 = self.m3[0]                                   # (ky, kz) = (256, 176)

        gt_c = self.sense_adj(self.y_full)                      # GT from the FULL k-space
        self.scale = float(np.quantile(gt_c.abs().reshape(-1).cpu().numpy(), 0.999))
        self.gt_c = gt_c / self.scale
        self.y_full = self.y_full / self.scale
        self.gt = self.gt_c.abs()
        self.y = self.y_full * self.m3.unsqueeze(0)             # the retrospectively acquired data

    # -- operators ---------------------------------------------------------------------------
    def sense_adj(self, k):
        acc = torch.zeros(self.m3.shape, dtype=torch.complex64, device=self.dev)
        for i in range(0, self.NC, self.chunk):
            acc += (self.S[i:i + self.chunk].conj() * ifft3c(k[i:i + self.chunk])).sum(0)
        return acc

    def AHA(self, x, mask=None):
        m = self.m3 if mask is None else mask
        acc = torch.zeros_like(x)
        for i in range(0, self.NC, self.chunk):
            acc += (self.S[i:i + self.chunk].conj()
                    * ifft3c(fft3c(self.S[i:i + self.chunk] * x) * m)).sum(0)
        return acc

    # -- scoring -----------------------------------------------------------------------------
    def score(self, x_complex, method: str, seconds: float = 0.0, **extra) -> dict:
        rec = x_complex.abs()
        met = evaluate_reconstruction(x_complex, self.gt_c, compute_extended=False)
        ho = score_headonly(rec.cpu().numpy(), self.gt.cpu().numpy())
        return dict(method=method, contrast=self.contrast,
                    psnr=float(met["psnr"]), ssim=float(met["ssim"]), nmse=float(met["nmse"]),
                    hfen=hfen(rec, self.gt), sharp=sharp_ratio(rec, self.gt),
                    grain=grain(rec, self.gt), seconds=seconds,
                    **{k: float(v) for k, v in ho.items()}, **extra)
