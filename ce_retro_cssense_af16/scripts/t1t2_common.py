"""t1t2_common.py -- paths, data loading and the scoring gauge shared by every arm of this topic.

Kept under the jyz file name so recon_joint_cssense.py / recon_spirit.py / recon_zsssdu.py / recon_dip_inr.py import
it unchanged. Differences from the jyz AF16 module (Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/scripts):
  * the ground truth is the CS-SENSE volume of the prospective pair (sim/gt.npz), not a SENSE adjoint of full k-space;
  * the two scans are called pre / post; the noisy synthetic k-space is sim/gs/<scan>_kspace.h5 (unmasked, the mask
    is applied here), maps sim/gs/<scan>_csm.h5 (key map), mask sim/mask/mask_af16_3d.h5 (key mask);
  * two extra scores: head_nmse (head-masked NMSE on magnitudes) and psnr_cplx (head-masked complex PSNR);
  * RETRO_TOPIC / RETRO_TRAINER override the topic root and the trainer snapshot (tests).
Gauge: scale = P99.9 of |GT| per scan (stored in gt.npz), applied to GT and to y; metrics =
paired_gs.metrics.evaluate_reconstruction (whole@max) + diag_artefact.headonly.score_headonly; hfen / sharp_ratio /
grain transcribed verbatim from arc/cycle.py (do not "improve").
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.fft as tfft
import torch.nn.functional as F

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
TRAINER = Path(os.environ.get("RETRO_TRAINER", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/trainer/_trainer_e778bf2"))
HERE = Path(__file__).resolve().parent
for _p in (str(TRAINER / "paired_gs"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from metrics import evaluate_reconstruction                      # noqa: E402
from diag_artefact.headonly import score_headonly, head_mask      # noqa: E402

SCANS = ("pre", "post")


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
    """Everything a reconstruction needs, on the device and in the normalised gauge."""

    def __init__(self, contrast: str, dev, chunk: int = 4):
        assert contrast in SCANS, contrast
        self.contrast, self.dev, self.chunk = contrast, dev, chunk
        with h5py.File(R / f"sim/gs/{contrast}_kspace.h5", "r") as f:
            y_full = torch.from_numpy(f["kspace"][:]).to(torch.complex64)
        with h5py.File(R / f"sim/gs/{contrast}_csm.h5", "r") as f:
            S = torch.from_numpy(f["map"][:]).to(torch.complex64)
        with h5py.File(R / "sim/mask/mask_af16_3d.h5", "r") as f:
            m3 = torch.from_numpy(f["mask"][:]).to(torch.float32)
        with np.load(R / "sim/gt.npz") as g:
            gt_c = torch.from_numpy(np.asarray(g[contrast])).to(torch.complex64)
            self.scale = float(g[f"scale_{contrast}"])
        assert y_full.shape == S.shape, (y_full.shape, S.shape)
        assert tuple(gt_c.shape) == tuple(m3.shape) == tuple(y_full.shape[1:]), (gt_c.shape, m3.shape, y_full.shape)
        self.y_full = (y_full / self.scale).to(dev); self.S = S.to(dev)
        self.m3 = (m3 > 0.5).to(dev)
        self.NC = self.y_full.shape[0]
        self.G = tuple(self.m3.shape)
        assert bool((self.m3 == self.m3[0:1]).all()), "mask is not constant along the readout axis"
        self.m2 = self.m3[0]                                   # (ky, kz) = (232, 176)
        self.gt_c = (gt_c / self.scale).to(dev)
        self.gt = self.gt_c.abs()
        self.y = self.y_full * self.m3.unsqueeze(0)             # the retrospectively acquired data

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

    def score(self, x_complex, method: str, seconds: float = 0.0, **extra) -> dict:
        rec = x_complex.abs()
        met = evaluate_reconstruction(x_complex, self.gt_c, compute_extended=False)
        gt_np = self.gt.cpu().numpy()
        ho = score_headonly(rec.cpu().numpy(), gt_np)
        head = torch.from_numpy(head_mask(gt_np)).to(self.dev)
        e_gt = float((self.gt[head] ** 2).sum())
        d_mag = float(((rec - self.gt)[head] ** 2).sum())
        mse_c = float((x_complex - self.gt_c)[head].abs().square().mean())
        gmax = float(self.gt.max())
        psnr_cplx = float("inf") if mse_c == 0.0 else float(20 * np.log10(gmax) - 10 * np.log10(mse_c))
        return dict(method=method, contrast=self.contrast,
                    psnr=float(met["psnr"]), ssim=float(met["ssim"]), nmse=float(met["nmse"]),
                    hfen=hfen(rec, self.gt), sharp=sharp_ratio(rec, self.gt),
                    grain=grain(rec, self.gt), seconds=seconds,
                    **{k: float(v) for k, v in ho.items()},
                    head_nmse=d_mag / e_gt, psnr_cplx=psnr_cplx, **extra)
