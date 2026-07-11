"""v2 mask three-way split per PDF §1.5 (M = M^tr ∪ M^fb ∪ M^te).

Implements INF-01 per .planning/phases/07-v2-v2-infrastructure/07-CONTEXT.md:
    决策 C: ACS = 24×24 phase plane × full readout 全部固定进入 M^tr。
    决策 A: mask_split_seed 是独立 config 字段 (default 42),与 global --seed 解耦。
    决策 B: split 结果 additive — 调用方保留原 acquired mask,叠加三 mask dict。
    D-03 mitigation: split 后产物用 dict 而非额外的位置参数。

纯函数 + 落盘工具,无副作用,可被 MRIDataset / Auditor / CF 三方独立调用。
不依赖 MRIDataset (避免循环 import);ACS 几何与 v1 ``mask_generator.py`` / ``MRIDataset._add_acs_region`` 完全一致。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


__all__ = [
    "split_acquired_mask",
    "save_split_indices",
]


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #


def _validate_ratios(ratios: Tuple[float, float, float]) -> None:
    if len(ratios) != 3:
        raise ValueError(f"ratios must have exactly 3 entries, got {ratios}")
    s = float(sum(ratios))
    if not (0.999 <= s <= 1.001):
        raise ValueError(f"ratios must sum to 1.0 (±1e-3), got sum={s:.6f}: {ratios}")
    for i, r in enumerate(ratios):
        if r < 0 or r > 1:
            raise ValueError(f"ratios[{i}]={r} out of [0,1]")


def _validate_axes(
    n_dims: int,
    phase_axes: Tuple[int, int],
    readout_axis: int,
) -> None:
    if n_dims != 3:
        raise ValueError(f"mask must be 3D (D,H,W), got n_dims={n_dims}")
    if len(phase_axes) != 2:
        raise ValueError(f"phase_axes must contain 2 entries, got {phase_axes}")
    if readout_axis not in (0, 1, 2):
        raise ValueError(f"readout_axis must be in {{0,1,2}}, got {readout_axis}")
    if readout_axis in phase_axes:
        raise ValueError(
            f"readout_axis ({readout_axis}) must be distinct from phase_axes ({phase_axes})"
        )
    if sorted((readout_axis, *phase_axes)) != [0, 1, 2]:
        raise ValueError(
            f"readout_axis + phase_axes must cover all 3 spatial axes once, "
            f"got readout={readout_axis}, phase={phase_axes}"
        )


def _make_acs_mask_3d(
    volume_shape: Tuple[int, int, int],
    acs_size: Tuple[int, int],
    phase_axes: Tuple[int, int],
    readout_axis: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build the 3D ACS region mask = ``(acs_h × acs_w)`` phase-plane center × full readout.

    Returns a ``(D, H, W)`` float tensor with values ∈ {0, 1}. The ACS region is
    centered on each phase axis at ``size // 2`` and spans ``acs_size[k] // 2``
    voxels on either side (matching v1 ``MRIDataset._add_acs_region`` slicing
    convention). The readout axis is always fully covered.
    """

    _validate_axes(len(volume_shape), phase_axes, readout_axis)
    if acs_size[0] > volume_shape[phase_axes[0]] or acs_size[1] > volume_shape[phase_axes[1]]:
        raise ValueError(
            f"acs_size={acs_size} exceeds phase axis sizes "
            f"({volume_shape[phase_axes[0]]}, {volume_shape[phase_axes[1]]})"
        )

    mask = torch.zeros(volume_shape, dtype=dtype, device=device)
    index = [slice(None)] * 3
    # readout axis fully sampled
    index[readout_axis] = slice(None)
    for k, axis in enumerate(phase_axes):
        size = volume_shape[axis]
        center = size // 2
        half_acs = acs_size[k] // 2
        start = max(0, center - half_acs)
        end = min(size, center + half_acs)
        index[axis] = slice(start, end)
    mask[tuple(index)] = 1.0
    return mask


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def split_acquired_mask(
    mask: torch.Tensor,
    ratios: Tuple[float, float, float] = (0.60, 0.25, 0.15),
    *,
    seed: int = 42,
    preserve_acs: bool = True,
    acs_size: Tuple[int, int] = (24, 24),
    phase_axes: Tuple[int, int] = (0, 2),
    readout_axis: int = 1,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Split a binary acquired-mask into three disjoint masks (M^tr, M^fb, M^te).

    Parameters
    ----------
    mask
        ``(D, H, W)`` float tensor with values ∈ {0, 1} — the acquired full
        undersampling mask. Same shape as the volume.
    ratios
        Target ``(tr, fb, te)`` partition ratios of the **non-ACS** acquired
        voxels. Sum must be 1.0 (±1e-3). Default ``(0.60, 0.25, 0.15)`` per
        ``REQUIREMENTS.md`` INF-01.
    seed
        Independent RNG seed for the split, decoupled from any global ``--seed``
        (CONTEXT.md decision A). Default 42.
    preserve_acs
        If True (default), the ACS region (per ``acs_size`` on the phase plane,
        full readout) is forced 100% into M^tr — every ACS-acquired voxel has
        ``M^tr=1, M^fb=0, M^te=0`` (CONTEXT.md decision C).
    acs_size
        ``(acs_h, acs_w)`` phase-plane ACS box. Default ``(24, 24)``.
    phase_axes
        Two-element tuple of phase-encoding axes. Default ``(0, 2)`` (HYW layout).
    readout_axis
        Fully-sampled readout axis. Default ``1`` (HYW layout).
    device
        Optional device override; if None, output stays on ``mask.device``.

    Returns
    -------
    Dict[str, torch.Tensor]
        ``{'tr': Tensor, 'fb': Tensor, 'te': Tensor}`` — three ``(D, H, W)``
        float tensors with values ∈ {0, 1}. Guarantees:

        - ``M^tr + M^fb + M^te == mask`` everywhere
        - ``M^tr * M^fb == M^tr * M^te == M^fb * M^te == 0`` (disjoint)
        - ACS region with ``preserve_acs=True``: all-1 in M^tr, all-0 elsewhere

    Notes
    -----
    - Reproducibility: same ``mask`` + same ``seed`` + same axes/acs_size ⇒
      bit-identical output across runs (uses ``torch.Generator`` ``manual_seed``
      with ``torch.randperm`` — both deterministic on CPU).
    - The split operates on CPU (binary masks are tiny relative to data).
    - Per CONTEXT.md decision C, ACS = phase-plane box × **full readout**
      (NOT a 24³ cube), matching 3D Cartesian MRI convention.

    References
    ----------
    PDF §1.5 — Evidence-auditing requires a disjoint feedback hold-out (M^fb)
    to compute PSF-corrected residual evidence R_c that does not see training
    voxels. M^te is the final test hold-out for K_test=32 evaluation.
    """

    # ----- 1) input validation -----
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be torch.Tensor, got {type(mask)}")
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D (D,H,W), got shape {tuple(mask.shape)}")
    _validate_ratios(ratios)
    _validate_axes(mask.ndim, tuple(phase_axes), int(readout_axis))

    volume_shape: Tuple[int, int, int] = tuple(int(s) for s in mask.shape)  # type: ignore[assignment]

    # Coerce mask to bool {0,1}-valued (allow float ∈ {0,1} input)
    mask_bool = mask > 0.5
    if not torch.equal(mask.float(), mask_bool.float()):
        # Allow values that are floating-point but already binary; reject true greyscale
        unique_vals = torch.unique(mask)
        if not torch.all((unique_vals == 0) | (unique_vals == 1)):
            raise ValueError(
                f"mask must be binary ({{0,1}}), got unique values {unique_vals.tolist()}"
            )

    # ----- 2) build ACS mask (3D) on same device as input -----
    cpu_device = torch.device("cpu")
    mask_cpu = mask_bool.to(cpu_device)
    acs_mask = _make_acs_mask_3d(
        volume_shape=volume_shape,
        acs_size=tuple(int(a) for a in acs_size),  # type: ignore[arg-type]
        phase_axes=tuple(int(a) for a in phase_axes),  # type: ignore[arg-type]
        readout_axis=int(readout_axis),
        dtype=torch.float32,
        device=cpu_device,
    ).bool()

    # ----- 3) acquired-voxel indices, split into ACS / non-ACS -----
    acquired_idx = torch.nonzero(mask_cpu, as_tuple=False)  # (N_acq, 3)
    is_acs = acs_mask[acquired_idx[:, 0], acquired_idx[:, 1], acquired_idx[:, 2]]
    if not preserve_acs:
        is_acs = torch.zeros_like(is_acs)

    acs_indices = acquired_idx[is_acs]            # (N_acs_acq, 3)
    non_acs_indices = acquired_idx[~is_acs]       # (N_non_acs, 3)
    n_non_acs = int(non_acs_indices.shape[0])

    # ----- 4) deterministic random partition of non-ACS voxels -----
    gen = torch.Generator(device=cpu_device)
    gen.manual_seed(int(seed))
    perm = torch.randperm(n_non_acs, generator=gen)
    shuffled_non_acs = non_acs_indices[perm]

    # target bucket sizes; last bucket absorbs rounding error
    n_tr_target = int(round(n_non_acs * float(ratios[0])))
    n_fb_target = int(round(n_non_acs * float(ratios[1])))
    n_te_target = n_non_acs - n_tr_target - n_fb_target
    if n_te_target < 0:
        # round-up overflow; clip
        n_te_target = 0
        n_fb_target = n_non_acs - n_tr_target

    tr_non_acs = shuffled_non_acs[:n_tr_target]
    fb_non_acs = shuffled_non_acs[n_tr_target:n_tr_target + n_fb_target]
    te_non_acs = shuffled_non_acs[n_tr_target + n_fb_target:]

    # ----- 5) ACS voxels all enter M^tr -----
    tr_all_idx = torch.cat([acs_indices, tr_non_acs], dim=0) if acs_indices.numel() else tr_non_acs

    # ----- 6) materialize 3 masks via scatter -----
    def _scatter(indices: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(volume_shape, dtype=torch.float32, device=cpu_device)
        if indices.numel():
            out[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0
        return out

    m_tr = _scatter(tr_all_idx)
    m_fb = _scatter(fb_non_acs)
    m_te = _scatter(te_non_acs)

    # ----- 7) self-check invariants -----
    union = m_tr + m_fb + m_te
    if not torch.equal(union, mask_cpu.float()):
        # Should never trigger; surface as a hard error so future regressions catch us
        n_mismatch = int((union != mask_cpu.float()).sum().item())
        raise RuntimeError(
            f"split_acquired_mask internal error: union != acquired mask "
            f"(n_mismatch={n_mismatch}, n_acq={int(mask_cpu.sum().item())})"
        )
    if not torch.all(m_tr * m_fb == 0):
        raise RuntimeError("split_acquired_mask: M^tr ∩ M^fb is non-empty")
    if not torch.all(m_tr * m_te == 0):
        raise RuntimeError("split_acquired_mask: M^tr ∩ M^te is non-empty")
    if not torch.all(m_fb * m_te == 0):
        raise RuntimeError("split_acquired_mask: M^fb ∩ M^te is non-empty")
    if preserve_acs:
        acs_acq = (acs_mask.float() * mask_cpu.float())
        if not torch.equal(m_tr * acs_acq, acs_acq):
            raise RuntimeError("split_acquired_mask: ACS not fully retained in M^tr")

    # ----- 8) move to requested device -----
    target_device = device if device is not None else mask.device
    return {
        "tr": m_tr.to(target_device),
        "fb": m_fb.to(target_device),
        "te": m_te.to(target_device),
    }


def save_split_indices(
    out_dir: str,
    splits: Dict[str, torch.Tensor],
    seed: int,
    ratios: Tuple[float, float, float],
    acquired_mask: torch.Tensor,
    filename: str = "split_indices.npz",
) -> Path:
    """Dump split indices + provenance to ``<out_dir>/<filename>`` for post-hoc audit.

    Saves 6 fields in the npz archive:

    - ``tr_idx``, ``fb_idx``, ``te_idx`` — flat int64 indices into the volume
      (via ``np.ravel_multi_index``), shape ``(N_k,)`` each
    - ``seed`` — int32 array of shape () (the split seed)
    - ``ratio_tuple`` — float64 array of shape (3,) (the ratios used)
    - ``acquired_mask_hash`` — string array of shape () (SHA-256 hexdigest of the
      acquired mask's raw bytes, for verifying which mask the split came from)

    Parameters
    ----------
    out_dir
        Directory to write into. Created if missing.
    splits
        Output of :func:`split_acquired_mask` — ``{'tr', 'fb', 'te'}`` dict.
    seed
        Split seed (recorded for reproducibility).
    ratios
        Split ratios tuple (recorded for reproducibility).
    acquired_mask
        The source acquired mask, used to compute the provenance hash.
    filename
        Output filename; default ``split_indices.npz``.

    Returns
    -------
    pathlib.Path
        Full path to the written ``.npz`` file.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / filename

    volume_shape = tuple(int(s) for s in acquired_mask.shape)

    def _flat_indices(m: torch.Tensor) -> np.ndarray:
        nz = m.detach().cpu().numpy().nonzero()  # tuple of arrays
        if any(arr.size == 0 for arr in nz):
            return np.empty(0, dtype=np.int64)
        return np.ravel_multi_index(nz, volume_shape).astype(np.int64)

    acquired_np = acquired_mask.detach().cpu().numpy()
    acq_hash = hashlib.sha256(np.ascontiguousarray(acquired_np).tobytes()).hexdigest()

    np.savez(
        str(target),
        tr_idx=_flat_indices(splits["tr"]),
        fb_idx=_flat_indices(splits["fb"]),
        te_idx=_flat_indices(splits["te"]),
        seed=np.array(int(seed), dtype=np.int64),
        ratio_tuple=np.array([float(r) for r in ratios], dtype=np.float64),
        acquired_mask_hash=np.array(acq_hash),
    )
    return target
