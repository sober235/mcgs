"""3D GRAPPA for Cartesian data accelerated along one phase-encoding axis with an ACS block.

Conventions: k-space arrays are (coil, RO, PE, SPE), DC-centred; the acquisition mask is (PE, SPE).
Acceleration R along PE: the acquired lines outside the ACS block share one parity (``y % R``), the
missing lines are the other residues. Sources for a missing line are the R-1 … nearest acquired lines on
each side (``dy`` offsets), a small window along RO (``dx``) and SPE (``dz``), all coils. Weights are
calibrated on the ACS block by Tikhonov-regularised least squares and applied to every missing line inside
the elliptical envelope of the acquired pattern; acquired lines are never modified, positions outside the
envelope (corner cut, partial Fourier) are left at zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class GrappaResult:
    kspace: torch.Tensor             # (C, X, Y, Z) complex64, CPU
    m_grappa: np.ndarray             # (Y, Z) bool, interpolated positions
    m_zero: np.ndarray               # (Y, Z) bool, left empty by design
    envelope: np.ndarray             # (Y, Z) bool, acquired | interpolated
    weights: torch.Tensor            # (K*C, C) complex64, CPU
    info: dict = field(default_factory=dict)


def acquired_parity(m_acq: np.ndarray, acs_lines: tuple[int, int], R: int) -> int:
    """Residue ``y % R`` shared by the acquired lines outside the ACS block."""
    y0, y1 = acs_lines
    lines = np.where(m_acq.any(1))[0]
    outside = lines[(lines < y0) | (lines >= y1)]
    residues = np.unique(outside % R)
    if len(residues) != 1:
        raise ValueError(f"acquired lines outside the ACS block have residues {residues} mod {R}")
    return int(residues[0])


def target_mask(m_acq: np.ndarray, R: int = 2) -> np.ndarray:
    """Missing (PE, SPE) positions that lie between two acquired lines at most ``R`` apart, inside the
    SPE span common to both neighbours (the elliptical envelope). Lines carrying any sample are acquired
    lines and are never targets; their unsampled SPE positions are outside the envelope."""
    Y, Z = m_acq.shape
    lines = np.where(m_acq.any(1))[0]
    zmin = np.full(Y, Z); zmax = np.full(Y, -1)
    for y in lines:
        zs = np.where(m_acq[y])[0]
        zmin[y], zmax[y] = zs.min(), zs.max()
    targets = np.zeros_like(m_acq)
    for y in range(lines.min(), lines.max() + 1):
        if m_acq[y].any():
            continue
        lo = lines[lines < y]; hi = lines[lines > y]
        if len(lo) == 0 or len(hi) == 0:
            continue
        a, b = lo.max(), hi.min()
        if b - a > R:
            continue
        z0, z1 = max(zmin[a], zmin[b]), min(zmax[a], zmax[b])
        if z1 >= z0:
            targets[y, z0:z1 + 1] = True
    return targets


def _offsets(R: int, n_pe_sources: int, dx: int, dz: int, target_parity: int, acquired_parity: int):
    """Source offsets (dy, dx, dz): the ``n_pe_sources`` nearest PE offsets that land on acquired lines
    (for R=2: -1, +1, -3, +3, …) times a (2dx+1) x (2dz+1) window along RO and SPE."""
    dys = []
    j = 1
    while len(dys) < n_pe_sources:
        for d in (-j, j):
            if (target_parity + d) % R == acquired_parity and len(dys) < n_pe_sources:
                dys.append(d)
        j += 1
    dys = sorted(dys)
    return [(dy, ddx, ddz) for dy in dys for ddx in range(-dx, dx + 1) for ddz in range(-dz, dz + 1)]


def _sources(kp: torch.Tensor, offs, xs: torch.Tensor, y: int, zs: torch.Tensor, pad):
    """Source matrix for target line ``y`` at the (xs x zs) grid from the zero-padded k-space ``kp``.
    Returns (len(xs)*len(zs), K*C) with feature order (offset-major, coil-minor)."""
    px, py, pz = pad
    cols = []
    for (dy, ddx, ddz) in offs:
        line = kp[:, :, py + y + dy]                                            # (C, Xp, Zp) view
        cols.append(line[:, px + xs + ddx][:, :, pz + zs + ddz])                # (C, nx, nz)
    S = torch.stack(cols, 0)                                                   # (K, C, nx, nz)
    K, C, nx, nz = S.shape
    return S.reshape(K * C, nx * nz).transpose(0, 1)


def _calibration_lines(acs_lines: tuple[int, int], R: int, target_parity: int, dys) -> list[int]:
    """Target-parity lines of the ACS block whose every PE source lies inside the block."""
    y0, y1 = acs_lines
    return [y for y in range(y0, y1) if y % R == target_parity and all(y0 <= y + dy < y1 for dy in dys)]


def _pad(k: torch.Tensor, pad):
    px, py, pz = pad
    C, X, Y, Z = k.shape
    kp = torch.zeros((C, X + 2 * px, Y + 2 * py, Z + 2 * pz), dtype=k.dtype, device=k.device)
    kp[:, px:px + X, py:py + Y, pz:pz + Z] = k
    return kp


def calibrate(k: torch.Tensor, m_acq: np.ndarray, acs_lines: tuple[int, int], R: int, offs, pad,
              target_parity: int, lam_rel: float) -> tuple[torch.Tensor, dict]:
    """Tikhonov least-squares GRAPPA weights from the ACS block. Every equation uses only acquired
    source and target positions (the ACS block may be cut by the elliptical envelope)."""
    C, X, Y, Z = k.shape
    y0, y1 = acs_lines
    px, py, pz = pad
    dys = sorted(set(o[0] for o in offs))
    kp = _pad(k, pad)
    xs = torch.arange(px, X - px, device=k.device)
    G = torch.zeros((len(offs) * C, len(offs) * C), dtype=torch.complex128, device=k.device)
    B = torch.zeros((len(offs) * C, C), dtype=torch.complex128, device=k.device)
    n_eq = 0; lines_used = []
    m = torch.from_numpy(m_acq).to(k.device)
    for y in _calibration_lines(acs_lines, R, target_parity, dys):
        valid = m[y].clone()
        for (dy, _, ddz) in offs:
            shifted = torch.zeros_like(valid)
            lo, hi = max(0, -ddz), min(Z, Z - ddz)
            shifted[lo:hi] = m[y + dy][lo + ddz:hi + ddz]
            valid &= shifted
        zs = torch.where(valid)[0]
        if len(zs) == 0:
            continue
        S = _sources(kp, offs, xs, y, zs, pad)
        T = kp[:, :, py + y][:, px + xs][:, :, pz + zs].reshape(C, -1).transpose(0, 1)
        G += (S.conj().transpose(0, 1) @ S).to(torch.complex128)
        B += (S.conj().transpose(0, 1) @ T).to(torch.complex128)
        n_eq += S.shape[0]; lines_used.append(y)
    if n_eq < G.shape[0]:
        raise ValueError(f"calibration underdetermined: {n_eq} equations for {G.shape[0]} unknowns")
    lam = lam_rel * (torch.real(torch.trace(G)) / G.shape[0])
    W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype, device=G.device), B)
    info = dict(n_equations=int(n_eq), calibration_lines=lines_used, lam=float(lam), n_unknowns=int(G.shape[0]))
    return W.to(torch.complex64), info


def apply(k: torch.Tensor, targets: np.ndarray, W: torch.Tensor, offs, pad) -> torch.Tensor:
    """Fill every target position from the acquired data; returns a new tensor."""
    C, X, Y, Z = k.shape
    px, py, pz = pad
    kp = _pad(k, pad)
    out = k.clone()
    xs = torch.arange(X, device=k.device)
    for y in np.where(targets.any(1))[0]:
        zs = torch.from_numpy(np.where(targets[y])[0]).to(k.device)
        S = _sources(kp, offs, xs, int(y), zs, pad)
        est = (S @ W).transpose(0, 1).reshape(C, X, len(zs))
        out[:, :, int(y)][:, :, zs] = est
    return out


def grappa3d(k: torch.Tensor, m_acq: torch.Tensor | np.ndarray, acs_lines: tuple[int, int], R: int = 2,
             n_pe_sources: int = 4, dx: int = 2, dz: int = 2, lam_rel: float = 1e-3,
             device: torch.device | None = None) -> GrappaResult:
    """Complete the missing PE lines of ``k`` (C, X, Y, Z) with 3D GRAPPA."""
    m_acq = np.asarray(m_acq.cpu().numpy() if isinstance(m_acq, torch.Tensor) else m_acq, dtype=bool)
    device = device or k.device
    k = k.to(device=device, dtype=torch.complex64)
    parity_acq = acquired_parity(m_acq, acs_lines, R)
    targets = target_mask(m_acq, R)
    tgt_res = np.unique(np.where(targets.any(1))[0] % R)
    if len(tgt_res) != 1:
        raise ValueError(f"targets have residues {tgt_res} mod {R}; only one missing residue is supported")
    target_parity = int(tgt_res[0])
    offs = _offsets(R, n_pe_sources, dx, dz, target_parity, parity_acq)
    pad = (dx, max(abs(o[0]) for o in offs), dz)
    W, info = calibrate(k, m_acq, acs_lines, R, offs, pad, target_parity, lam_rel)
    out = apply(k, targets, W, offs, pad)
    envelope = m_acq | targets
    info.update(dict(R=R, n_pe_sources=n_pe_sources, dx=dx, dz=dz, offsets=len(offs), lam_rel=lam_rel,
                     acquired_parity=parity_acq, target_parity=target_parity, acs_lines=list(acs_lines),
                     n_targets=int(targets.sum()), n_acquired=int(m_acq.sum())))
    return GrappaResult(kspace=out.cpu(), m_grappa=targets, m_zero=~envelope, envelope=envelope,
                        weights=W.cpu(), info=info)


def acs_holdout_check(k: torch.Tensor, m_acq: torch.Tensor | np.ndarray, acs_lines: tuple[int, int], R: int = 2,
                      n_pe_sources: int = 4, dx: int = 2, dz: int = 2, lam_rel: float = 1e-3,
                      device: torch.device | None = None) -> float:
    """Out-of-sample check: hide the missing-parity lines of the upper half of the ACS block, calibrate on
    the lower half, predict the hidden lines and return the relative k-space error against the measured
    values."""
    m_acq = np.asarray(m_acq.cpu().numpy() if isinstance(m_acq, torch.Tensor) else m_acq, dtype=bool)
    device = device or k.device
    k = k.to(device=device, dtype=torch.complex64)
    y0, y1 = acs_lines
    mid = y0 + (y1 - y0) // 2
    parity_acq = acquired_parity(m_acq, acs_lines, R)
    hidden = np.zeros_like(m_acq)
    for y in range(mid, y1):
        if y % R != parity_acq:
            hidden[y] = m_acq[y]
    m_cal = m_acq & ~hidden
    k_cal = k * torch.from_numpy(m_cal).to(device)[None, None]
    targets = target_mask(m_cal, R) & hidden
    target_parity = int(np.unique(np.where(targets.any(1))[0] % R)[0])
    offs = _offsets(R, n_pe_sources, dx, dz, target_parity, parity_acq)
    pad = (dx, max(abs(o[0]) for o in offs), dz)
    W, _ = calibrate(k_cal, m_cal, (y0, mid), R, offs, pad, target_parity, lam_rel)
    est = apply(k_cal, targets, W, offs, pad)
    t = torch.from_numpy(targets).to(device)
    num = torch.linalg.vector_norm(est[:, :, t] - k[:, :, t])
    den = torch.linalg.vector_norm(k[:, :, t])
    return float(num / den)


def acs_leave_one_out(k: torch.Tensor, m_acq: torch.Tensor | np.ndarray, acs_lines: tuple[int, int], R: int = 2,
                      n_pe_sources: int = 4, dx: int = 2, dz: int = 2, lam_rel: float = 1e-3,
                      device: torch.device | None = None) -> dict:
    """Leave-one-line-out check of the GRAPPA kernel: for every calibration target line, hide it, calibrate on
    the remaining lines, predict it from its (acquired-parity) neighbours and report the relative k-space
    error against the measurement. Returns ``lines``, ``per_line`` and ``mean``."""
    m_acq = np.asarray(m_acq.cpu().numpy() if isinstance(m_acq, torch.Tensor) else m_acq, dtype=bool)
    device = device or k.device
    k = k.to(device=device, dtype=torch.complex64)
    parity_acq = acquired_parity(m_acq, acs_lines, R)
    target_parity = int(np.unique(np.where(target_mask(m_acq, R).any(1))[0] % R)[0])
    offs = _offsets(R, n_pe_sources, dx, dz, target_parity, parity_acq)
    pad = (dx, max(abs(o[0]) for o in offs), dz)
    lines = _calibration_lines(acs_lines, R, target_parity, sorted(set(o[0] for o in offs)))
    errors = []
    for y in lines:
        m_cal = m_acq.copy(); m_cal[y] = False
        k_cal = k * torch.from_numpy(m_cal).to(device)[None, None]
        W, _ = calibrate(k_cal, m_cal, acs_lines, R, offs, pad, target_parity, lam_rel)
        targets = np.zeros_like(m_acq); targets[y] = m_acq[y]
        est = apply(k_cal, targets, W, offs, pad)
        t = torch.from_numpy(targets).to(device)
        errors.append(float(torch.linalg.vector_norm(est[:, :, t] - k[:, :, t]) / torch.linalg.vector_norm(k[:, :, t])))
    return dict(lines=lines, per_line=errors, mean=float(np.mean(errors)))
