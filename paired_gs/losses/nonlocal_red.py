"""R_nl: non-local prior as regularisation-by-denoising with a lagged target (jyz AF16 phase 3, 2026-09-12).

Every prior in the recipe (edge-TV, wavelet L1) and in every comparison arm is local. The CS-MRI step above
wavelet + TV is non-local self-similarity (BM3D / BM4D class), and this term brings it in without a learned
prior: on the rendered magnitude of each contrast,

    R_nl(x)  = mean_B( (|x| - x_tilde)^2 ) / ( mean_B( x_tilde^2 ) + eps )      B = head bounding box
    x_tilde  = D( |x_k| )     refreshed at iteration nl_start and every nl_refresh iterations after it

x_tilde is DETACHED: between refreshes it is a constant, so the gradient is that of a quadratic pull of |x|
towards the denoised copy of an earlier render (the RED fixed point when the denoiser is a projection). The
normaliser makes the term dimensionless: scaling x and x_tilde by the same constant leaves it unchanged, so
lambda_nl does not depend on the k-space normalisation. D is BM4D (package bm4d 4.2.5, CPU) with
sigma = nl_sigma_rel * P99(|x_k| inside the head), run on the head bounding box only: the ZF head support the
trainer uses for gaussian.init_mode=head (|zf| > 0.05 * P99.9), dilated by 8 voxels, fixed for the run. The
denoiser is a plain callable (numpy crop, sigma) -> numpy crop, so tests inject fakes and never call BM4D.

The trainer builds nothing and imports nothing while lambda_nl (lambda_nl_plain / lambda_nl_contrast) is 0.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

HEAD_FRAC = 0.05           # the init_mode=head support rule: |zf| > HEAD_FRAC * P99.9(|zf|)
BBOX_DILATE = 8            # voxels added on every side of the head bounding box
_QUANTILE_MAX = 2 ** 24    # torch.quantile input limit

BBox = Tuple[slice, slice, slice]
Denoiser = Callable[[np.ndarray, float], np.ndarray]


def head_bbox(support: torch.Tensor, dilate: int = BBOX_DILATE) -> BBox:
    """Axis-aligned bounding box of the True voxels of `support` (D, H, W), grown by `dilate` voxels on every
    side and clipped to the volume. Raises when the support is empty."""
    if support.ndim != 3:
        raise ValueError(f"support must be (D, H, W), got {tuple(support.shape)}")
    sup = support.bool()
    if not bool(sup.any()):
        raise RuntimeError("head support is empty; check the ZF images / head_frac")
    out = []
    for ax in range(3):
        other = tuple(a for a in range(3) if a != ax)
        nz = torch.nonzero(sup.any(dim=other[1]).any(dim=other[0]), as_tuple=False).flatten()
        lo = max(int(nz.min()) - dilate, 0)
        hi = min(int(nz.max()) + 1 + dilate, int(sup.shape[ax]))
        out.append(slice(lo, hi))
    return tuple(out)


def crop(vol: torch.Tensor, bbox: BBox) -> torch.Tensor:
    return vol[bbox]


def uncrop(c: torch.Tensor, bbox: BBox, shape: Tuple[int, int, int]) -> torch.Tensor:
    """Paste the crop back into a zero volume of `shape`; exact inverse of `crop` inside the box."""
    out = torch.zeros(tuple(shape), dtype=c.dtype, device=c.device)
    out[bbox] = c
    return out


def nonlocal_red(vol: torch.Tensor, target: torch.Tensor, bbox: BBox, eps: float = 1e-12) -> torch.Tensor:
    """mean_B((|vol| - target)^2) / (mean_B(target^2) + eps); `target` is the crop-sized constant x_tilde."""
    m = vol[bbox].abs()
    if target.shape != m.shape:
        raise ValueError(f"target {tuple(target.shape)} does not match the crop {tuple(m.shape)}")
    t = target.to(m.dtype)
    return (m - t).square().mean() / (t.square().mean() + eps)


def make_denoiser(name: str, profile: str = "np", threads: int = 32) -> Denoiser:
    """'bm4d' (the real one; profile 'np' or 'lc', `threads` CPU threads), or the test fakes 'fake_identity'
    (returns the crop) and 'fake_blur' (1-voxel Gaussian blur, sigma ignored)."""
    if name == "bm4d":
        if profile not in ("np", "lc"):
            raise ValueError(f"nl_profile must be 'np' or 'lc', got {profile!r}")
        import bm4d
        from bm4d.profiles import BM4DProfileLC
        pro = bm4d.BM4DProfile() if profile == "np" else BM4DProfileLC()   # 4.2.5 resolves no 'lc' string
        pro.num_threads = int(threads)

        def run(z: np.ndarray, sigma: float) -> np.ndarray:
            return np.asarray(bm4d.bm4d(z, float(sigma), pro), dtype=np.float32)
        return run
    if name == "fake_identity":
        return lambda z, sigma: z.copy()
    if name == "fake_blur":
        from scipy.ndimage import gaussian_filter
        return lambda z, sigma: gaussian_filter(z, 1.0)
    raise ValueError(f"nl_denoiser must be 'bm4d', 'fake_identity' or 'fake_blur', got {name!r}")


class NonLocalRED:
    """Holds the fixed bounding box, the denoiser and one lagged target per contrast.

    due(it)                 -> True at nl_start and every nl_refresh iterations after it
    refresh(name, vol, it)  -> rebuild the target of `name` from the current render (no grad); returns the log info
    self(name, vol)         -> R_nl for that contrast, or 0 before its first refresh
    """

    def __init__(self, support: torch.Tensor, denoiser: Denoiser, sigma_rel: float = 0.04,
                 start: int = 500, refresh: int = 500, dilate: int = BBOX_DILATE, eps: float = 1e-12):
        if start < 1 or refresh < 1:
            raise ValueError(f"nl_start and nl_refresh must be >= 1, got {start} / {refresh}")
        self.bbox = head_bbox(support, dilate)
        self.head_crop = support.bool()[self.bbox]           # head voxels of the crop (sigma is read here)
        self.crop_shape = tuple(self.head_crop.shape)
        self.denoiser = denoiser
        self.sigma_rel = float(sigma_rel)
        self.start = int(start)
        self.refresh_every = int(refresh)
        self.eps = float(eps)
        self.targets: Dict[str, torch.Tensor] = {}
        self.info: Dict[str, dict] = {}

    def due(self, it: int) -> bool:
        return it >= self.start and (it - self.start) % self.refresh_every == 0

    def bbox_str(self) -> str:
        return "x".join(f"{s.start}:{s.stop}" for s in self.bbox)

    @torch.no_grad()
    def refresh(self, name: str, vol: torch.Tensor, it: int) -> dict:
        t0 = time.time()
        c = vol.detach()[self.bbox].abs()
        head = c[self.head_crop.to(c.device)].float()
        if head.numel() > _QUANTILE_MAX:
            head = head[:: head.numel() // _QUANTILE_MAX + 1]
        sigma = self.sigma_rel * float(torch.quantile(head, 0.99))
        z = np.ascontiguousarray(c.float().cpu().numpy(), dtype=np.float32)
        out = self.denoiser(z, sigma)
        target = torch.as_tensor(np.ascontiguousarray(out, dtype=np.float32), device=vol.device)
        if tuple(target.shape) != tuple(c.shape):
            raise RuntimeError(f"denoiser returned {tuple(target.shape)}, expected {tuple(c.shape)}")
        self.targets[name] = target
        self.info[name] = dict(iter=int(it), sigma=sigma, crop=tuple(c.shape), seconds=time.time() - t0)
        return self.info[name]

    def __call__(self, name: str, vol: torch.Tensor) -> torch.Tensor:
        target = self.targets.get(name)
        if target is None:
            return torch.zeros((), device=vol.device, dtype=(vol.real if vol.is_complex() else vol).dtype)
        return nonlocal_red(vol, target, self.bbox, self.eps)
