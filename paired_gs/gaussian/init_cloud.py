"""Cold-start cloud initialisation from the measured (zero-filled) images.

Why this exists. `JointTrainer._setup_model` scatters a fresh cloud UNIFORMLY over [-1, 1]^3
when no warm-start checkpoint is given. On the (256, 256, 176) 26-coil bundle the head occupies
~28 % of that box, so ~72 % of the Gaussians start in air, where their density (0.1 * zf) is
~0 and the tile voxelizer's position gradient (proportional to density) is ~0 too: they never
move and never contribute. The shipped recipe never met this because its 576902-Gaussian warm
start had already been fitted onto anatomy.

Everything here reads only the zero-filled SENSE images of the two channels (measured data,
no ground truth), so the init is as legitimate as the warm start it replaces.

Modes (`gaussian.init_mode`):
  uniform    the historical behaviour, kept elsewhere (this module is not called).
  head       uniform over the voxels whose union-ZF magnitude exceeds head_frac * P99.9.
  intensity  inside the same support, probability proportional to the union-ZF magnitude.
"""
from __future__ import annotations

from typing import Tuple

import torch


def union_magnitude(zf_p: torch.Tensor, zf_c: torch.Tensor) -> torch.Tensor:
    """Per-voxel max of the two channels' ZF magnitudes, each normalised by its own P99.9,
    so a voxel counts as head if EITHER contrast lights it up."""
    mp, mc = zf_p.abs(), zf_c.abs()
    sp = torch.quantile(mp.flatten()[:: max(1, mp.numel() // 2_000_000)], 0.999).clamp_min(1e-12)
    sc = torch.quantile(mc.flatten()[:: max(1, mc.numel() // 2_000_000)], 0.999).clamp_min(1e-12)
    return torch.maximum(mp / sp, mc / sc)


def head_support(mag: torch.Tensor, head_frac: float = 0.05) -> torch.Tensor:
    """Boolean support: mag > head_frac * P99.9(mag). Same rule as the head-only scorer, but on
    the measured ZF image rather than the ground truth."""
    p999 = torch.quantile(mag.flatten()[:: max(1, mag.numel() // 2_000_000)], 0.999)
    return mag > head_frac * p999


def sample_positions(mag: torch.Tensor, num_points: int, mode: str, head_frac: float,
                     generator: torch.Generator | None = None,
                     jitter: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Draw `num_points` voxel indices inside the head support and return
    (voxel_coords (N,3) long, positions (N,3) float in [-1, 1]).

    `head`      uniform over the support (with replacement only if the support is smaller than N).
    `intensity` proportional to mag inside the support.
    Positions are the voxel centres plus, when `jitter`, a uniform sub-voxel offset in
    (-0.5, 0.5) voxel so that duplicates do not sit on top of each other.
    """
    if mode not in ("head", "intensity"):
        raise ValueError(f"init_mode must be 'head' or 'intensity', got {mode!r}")
    dev = mag.device
    shape = tuple(mag.shape)
    sup = head_support(mag, head_frac)
    cand = torch.nonzero(sup.flatten(), as_tuple=False).squeeze(-1)
    if cand.numel() == 0:
        raise RuntimeError("head support is empty; check the ZF images / head_frac")
    if mode == "head":
        w = torch.ones(cand.numel(), device=dev)
    else:
        w = mag.flatten()[cand].clamp_min(0.0)
    replacement = cand.numel() < num_points
    pick = torch.multinomial(w, num_points, replacement=replacement, generator=generator)
    idx = cand[pick]
    d, h, wd = shape
    z = idx // (h * wd)
    rem = idx % (h * wd)
    y = rem // wd
    x = rem % wd
    vox = torch.stack([z, y, x], dim=-1)
    off = torch.rand(num_points, 3, device=dev, generator=generator) - 0.5 if jitter else torch.zeros(num_points, 3, device=dev)
    pos = torch.stack([
        ((z.float() + 0.5 + off[:, 0]) / d) * 2 - 1,
        ((y.float() + 0.5 + off[:, 1]) / h) * 2 - 1,
        ((x.float() + 0.5 + off[:, 2]) / wd) * 2 - 1,
    ], dim=-1).clamp_(-1.0, 1.0)
    return vox, pos


def scales_3nn(positions: torch.Tensor, shape: Tuple[int, int, int], scale_mult: float = 1.0,
               min_scale: float = 1e-4, max_scale: float = 0.03) -> torch.Tensor:
    """Isotropic scale = scale_mult * mean distance to the 3 nearest neighbours (normalised
    coordinates), so the cloud covers its own spacing regardless of how many points were drawn.
    Clamped to [min_scale, max_scale]: an isolated noise voxel that slipped into the support would
    otherwise get a neighbour distance of tens of voxels and become a blob (max_scale 0.03 is
    ~4 voxels on a 256 axis). Falls back to 2/max(shape) (one voxel) when scipy is unavailable."""
    try:
        from scipy.spatial import cKDTree
    except Exception:            # pragma: no cover
        base = 2.0 / max(shape)
        return torch.full((positions.shape[0], 3), base * scale_mult, device=positions.device)
    import numpy as np
    pos_np = positions.detach().cpu().numpy().astype("float32")
    k = min(4, pos_np.shape[0])
    dist, _ = cKDTree(pos_np).query(pos_np, k=k, workers=-1)
    nn = np.asarray(dist[:, 1:], dtype="float32")
    if nn.shape[1] < 3:
        nn = np.concatenate([nn, np.repeat(nn[:, -1:], 3 - nn.shape[1], axis=1)], axis=1)
    mean = np.clip(nn[:, :3].mean(axis=1, keepdims=True) * scale_mult, min_scale, max_scale)
    return torch.from_numpy(np.repeat(mean, 3, axis=1)).to(device=positions.device, dtype=torch.float32)


# --- frozen init support (sg2x2 shared-geometry ablation, 2026-10-09) ------------------------------------------
def load_init_support(path: str, volume_shape: Tuple[int, int, int], device) -> torch.Tensor:
    """A frozen boolean (D, H, W) support from an .npz (key 'support'), e.g. the development subject's union-ZF
    head support written by scripts/sg2x2_make_init_support.py. Refuses a shape that is not this volume's."""
    import numpy as np
    with np.load(path) as z:
        sup = np.asarray(z["support"]).astype(bool)
    if tuple(sup.shape) != tuple(volume_shape):
        raise ValueError(f"init support {path} has shape {tuple(sup.shape)}, the volume is {tuple(volume_shape)}")
    return torch.from_numpy(sup).to(device)


def sample_positions_in_support(support: torch.Tensor, num_points: int,
                                generator: torch.Generator | None = None,
                                jitter: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniform draw of `num_points` voxels inside a boolean support (same conventions as sample_positions: with
    replacement only when the support is smaller than N, voxel centre plus a sub-voxel jitter, [-1, 1] coords).
    The draw depends only on (support, num_points, generator): nothing of the subject's data enters."""
    dev = support.device
    cand = torch.nonzero(support.flatten(), as_tuple=False).squeeze(-1)
    if cand.numel() == 0:
        raise RuntimeError("init support is empty")
    w = torch.ones(cand.numel(), device=dev)
    pick = torch.multinomial(w, num_points, replacement=cand.numel() < num_points, generator=generator)
    idx = cand[pick]
    d, h, wd = tuple(support.shape)
    z = idx // (h * wd)
    rem = idx % (h * wd)
    y = rem // wd
    x = rem % wd
    vox = torch.stack([z, y, x], dim=-1)
    off = torch.rand(num_points, 3, device=dev, generator=generator) - 0.5 if jitter else torch.zeros(num_points, 3, device=dev)
    pos = torch.stack([
        ((z.float() + 0.5 + off[:, 0]) / d) * 2 - 1,
        ((y.float() + 0.5 + off[:, 1]) / h) * 2 - 1,
        ((x.float() + 0.5 + off[:, 2]) / wd) * 2 - 1,
    ], dim=-1).clamp_(-1.0, 1.0)
    return vox, pos
