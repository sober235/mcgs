"""Known-zero k-space region Z and coil-noise prewhitening W — the Phase 1 physics levers
(docs/superpowers/specs/2026-09-05-af12-to-af10-quality-design.md §4, as amended by appendix A, 2026-09-06).

On this data the vendor's k-space support is a HARD shutter in the phase-encoding (PE) plane: whole readout
lines are numerically zero (|y| ~ 1e-20 of the data scale; nothing between 1e-12 and 1e-4) outside an
axis-aligned region built from rectangular padding cuts and/or an ellipse centred on the DC. Z is estimated
from the ACQUIRED lines only; ZERO_REL, the grid steps and EXPAND_STEP are numerical resolutions, not knobs.

    Z_PE = padding cuts (a side is cut only where the nonzero lines end on a straight, wide edge and every
           sampled line beyond it is zero)
         ∪ exterior of the fitted DC-centred ellipse (expanded until no sampled nonzero line lies outside)
    Z_PE never contains a sampled nonzero line.      Z = Z_PE broadcast along the readout axis.

Σ (prewhitening, R2) comes from the readout-end samples of the sampled nonzero lines, where the per-sample
energy sits on the noise floor for both contrasts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

ZERO_REL = 1e-10            # a sampled line is numerically zero if max|y| over coils & readout < ZERO_REL * global max
COARSE_STEP_PX = 4.0        # ellipse semi-axis grid, coarse stage
GRID_STEP_PX = 0.5          # ellipse semi-axis grid, fine stage (±COARSE_STEP_PX around the coarse optimum)
EXPAND_STEP = 0.005         # safety expansion resolution of the ellipse
EXPAND_MAX = 2.0            # give the ellipse up beyond this expansion
CUT_SPAN_MIN = 0.5          # a padding edge must carry nonzero lines over >= this fraction of the other axis' extent
READOUT_END_FRAC = 0.125    # Σ source: outer 1/8 of the readout indices at both ends
MIN_NOISE_SAMPLES = 2000


@dataclass
class KnownZeroInfo:
    dc: tuple                    # (i, j, k) array index of the sampled-energy peak (PE axes + readout)
    ro_axis: int
    zero_threshold: float        # absolute max|y| below which a sampled line is numerically zero
    cuts: dict                   # axis{0,1}_{lo,hi}: index bound of the zero padding (lo: indices < bound are zero;
                                 # hi: indices > bound are zero), or None
    ellipse: Optional[tuple]     # (a_px, b_px, expand) about dc on the PE axes, or None
    n_sampled_lines: int
    n_zero_lines: int
    n_zero_explained: int        # sampled zero lines inside Z
    n_nonzero_in_z: int          # sampled nonzero lines inside Z — 0 by construction
    frac_z: float                # fraction of the PE plane in Z
    frac_sampled_in_z: float     # |Z ∩ Ω| / |Ω| over lines


def _pe_lines(kspace: torch.Tensor, mask: torch.Tensor, ro_axis: int):
    """Per-PE-line max|y| and energy over the SAMPLED readout entries only; (P0,P1) each, plus sampled (P0,P1)."""
    m_r = torch.movedim(mask > 0.5, ro_axis, -1)
    a_r = torch.movedim(kspace.abs().amax(0), ro_axis, -1) * m_r
    e_r = torch.movedim((kspace.real.square() + kspace.imag.square()).sum(0), ro_axis, -1) * m_r
    return a_r.amax(-1), e_r.sum(-1), m_r.any(-1)


def _rect_cuts(nonzero: np.ndarray, zero: np.ndarray, span_min: float = CUT_SPAN_MIN) -> dict:
    """Per-side padding cuts. A side is cut only if (1) at least one sampled zero line lies beyond the bounding
    box of the sampled nonzero lines on that side (every sampled line there is zero by construction) and (2) the
    edge row/column of the bounding box carries nonzero lines over >= span_min of the other axis' nonzero extent
    (a straight padding edge, not the tip of an ellipse)."""
    cuts = {}
    for ax in (0, 1):
        other = 1 - ax
        nz_idx = np.nonzero(nonzero.any(axis=other))[0]
        z_idx = np.nonzero(zero.any(axis=other))[0]
        if nz_idx.size == 0:
            cuts[f"axis{ax}_lo"] = cuts[f"axis{ax}_hi"] = None
            continue
        lo, hi = int(nz_idx.min()), int(nz_idx.max())
        ext_o = np.nonzero(nonzero.any(axis=ax))[0]
        extent = int(ext_o.max() - ext_o.min() + 1)

        def wide(idx):
            line = np.nonzero(np.take(nonzero, idx, axis=ax))[0]
            return line.size > 0 and (line.max() - line.min() + 1) >= span_min * extent

        cuts[f"axis{ax}_lo"] = lo if (np.any(z_idx < lo) and wide(lo)) else None
        cuts[f"axis{ax}_hi"] = hi if (np.any(z_idx > hi) and wide(hi)) else None
    return cuts


def _rect_zero(shape, cuts: dict) -> np.ndarray:
    z = np.zeros(shape, bool)
    if cuts["axis0_lo"] is not None:
        z[:cuts["axis0_lo"], :] = True
    if cuts["axis0_hi"] is not None:
        z[cuts["axis0_hi"] + 1:, :] = True
    if cuts["axis1_lo"] is not None:
        z[:, :cuts["axis1_lo"]] = True
    if cuts["axis1_hi"] is not None:
        z[:, cuts["axis1_hi"] + 1:] = True
    return z


def _fit_ellipse(di: np.ndarray, dj: np.ndarray, is_nonzero: np.ndarray, is_target: np.ndarray,
                 half0: float, half1: float):
    """Axis-aligned ellipse about the DC on the SAMPLED lines: (a, b, expand) or None.
    di/dj: PE offsets of the sampled lines; is_nonzero: sampled nonzero lines (must end up outside Z);
    is_target: sampled zero lines the padding cuts do not explain. Misclassification = nonzero outside the
    ellipse + target inside it; 'no ellipse' competes with misclassification = |target|. The winner is then
    expanded until no sampled nonzero line lies outside it, and dropped if it then explains no target line."""
    n_target = int(is_target.sum())
    if n_target == 0:
        return None
    di2, dj2 = di.astype(np.float64) ** 2, dj.astype(np.float64) ** 2

    def mis(a, b):
        outside = di2 / (a * a) + dj2 / (b * b) >= 1.0
        return int((outside & is_nonzero).sum() + (~outside & is_target).sum())

    best = (n_target, None)
    for a in np.arange(COARSE_STEP_PX, 1.5 * half0 + COARSE_STEP_PX, COARSE_STEP_PX):
        for b in np.arange(COARSE_STEP_PX, 1.5 * half1 + COARSE_STEP_PX, COARSE_STEP_PX):
            m_ = mis(a, b)
            if m_ < best[0]:
                best = (m_, (float(a), float(b)))
    if best[1] is None:
        return None
    a0, b0 = best[1]
    for a in np.arange(a0 - COARSE_STEP_PX, a0 + COARSE_STEP_PX + 1e-9, GRID_STEP_PX):
        for b in np.arange(b0 - COARSE_STEP_PX, b0 + COARSE_STEP_PX + 1e-9, GRID_STEP_PX):
            if a <= 0 or b <= 0:
                continue
            m_ = mis(a, b)
            if m_ < best[0]:
                best = (m_, (float(a), float(b)))
    a, b = best[1]
    expand = 1.0
    while expand <= EXPAND_MAX:
        outside = di2 / (a * expand) ** 2 + dj2 / (b * expand) ** 2 >= 1.0
        if not bool((outside & is_nonzero).any()):
            break
        expand += EXPAND_STEP
    else:
        return None
    if int((outside & is_target).sum()) == 0:
        return None
    return (a, b, float(round(expand, 6)))


def estimate_known_zero(kspace: torch.Tensor, mask: torch.Tensor, ro_axis: int):
    """Z from the acquired lines only (appendix A, R1). kspace (C,D,H,W) complex, already masked; mask (D,H,W)."""
    assert kspace.ndim == 4 and mask.ndim == 3 and tuple(kspace.shape[1:]) == tuple(mask.shape)
    dev = kspace.device
    amax, energy, sampled = _pe_lines(kspace, mask, ro_axis)
    P0, P1 = sampled.shape
    thr = ZERO_REL * float(amax.max())
    sampled_np = sampled.cpu().numpy()
    zero = (sampled & (amax < thr)).cpu().numpy()
    nonzero = sampled_np & ~zero
    flat = int(torch.argmax(energy))
    dc0, dc1 = flat // P1, flat % P1
    m_r = torch.movedim(mask > 0.5, ro_axis, -1)
    e_ro = (torch.movedim((kspace.real.square() + kspace.imag.square()).sum(0), ro_axis, -1) * m_r).sum((0, 1))
    dc2 = int(torch.argmax(e_ro))
    cuts = _rect_cuts(nonzero, zero)
    z_pe = _rect_zero((P0, P1), cuts)
    si, sj = np.nonzero(sampled_np)
    ell = _fit_ellipse(si - dc0, sj - dc1, nonzero[si, sj], (zero & ~z_pe)[si, sj], P0 / 2, P1 / 2)
    if ell is not None:
        a, b, s = ell
        ii, jj = np.meshgrid(np.arange(P0) - dc0, np.arange(P1) - dc1, indexing="ij")
        z_pe |= (ii / (a * s)) ** 2 + (jj / (b * s)) ** 2 >= 1.0
    z_pe &= ~nonzero                                                  # never override a measured nonzero line
    Rn = int(mask.shape[ro_axis])
    Z = torch.movedim(torch.from_numpy(z_pe).to(dev).unsqueeze(-1).expand(P0, P1, Rn), -1, ro_axis).contiguous()
    dc = [0, 0, 0]
    pe = [a_ for a_ in range(3) if a_ != ro_axis]
    dc[pe[0]], dc[pe[1]], dc[ro_axis] = int(dc0), int(dc1), dc2
    info = KnownZeroInfo(dc=tuple(dc), ro_axis=int(ro_axis), zero_threshold=float(thr), cuts=cuts, ellipse=ell,
                         n_sampled_lines=int(sampled_np.sum()), n_zero_lines=int(zero.sum()),
                         n_zero_explained=int((zero & z_pe).sum()), n_nonzero_in_z=int((nonzero & z_pe).sum()),
                         frac_z=float(z_pe.mean()),
                         frac_sampled_in_z=float((z_pe & sampled_np).sum() / max(int(sampled_np.sum()), 1)))
    return Z, info


def estimate_noise_cov(kspace: torch.Tensor, mask: torch.Tensor, Z: torch.Tensor, info: KnownZeroInfo, *,
                       min_samples: int = MIN_NOISE_SAMPLES, end_frac: float = READOUT_END_FRAC):
    """Coil noise covariance Σ (C×C) from the readout-end samples (outer `end_frac` of the readout indices at
    both ends) of the sampled NONZERO lines (appendix A, R2). `Z` is accepted for interface compatibility and
    not used. Returns (Σ | None, winfo) with a uniform winfo key set; None means W = I."""
    ro = info.ro_axis
    m_r = torch.movedim(mask > 0.5, ro, -1)                           # (P0,P1,R)
    y_r = torch.movedim(kspace, ro + 1, -1)                           # (C,P0,P1,R)
    Rn = m_r.shape[-1]
    n_end = max(int(round(end_frac * Rn)), 1)
    a_line = (y_r.abs().amax(0) * m_r).amax(-1)
    nonzero = m_r.any(-1) & (a_line >= info.zero_threshold)
    sel = torch.zeros_like(m_r)
    sel[..., :n_end] = True
    sel[..., Rn - n_end:] = True
    sel = sel & m_r & nonzero.unsqueeze(-1)
    vals = y_r[:, sel]                                                # (C, N)
    empty = {"coil_corr_max_offdiag": None, "coil_corr_mean_offdiag": None, "eig_ratio": None, "noise_level": None}
    if vals.shape[1] < min_samples or float(vals.abs().max()) < info.zero_threshold:
        return None, {"sigma_source": "identity", "n_noise_samples": int(vals.shape[1]), **empty}
    vals = vals.to(torch.complex128)
    Sigma = (vals @ vals.conj().T) / vals.shape[1]
    d = torch.sqrt(torch.diagonal(Sigma).real)
    corr = (Sigma / (d.view(-1, 1) * d.view(1, -1))).abs()
    corr.fill_diagonal_(0.0)
    ev = torch.linalg.eigvalsh(0.5 * (Sigma + Sigma.conj().T)).real
    Cn = Sigma.shape[0]
    winfo = {"sigma_source": "readout_ends", "n_noise_samples": int(vals.shape[1]),
             "coil_corr_max_offdiag": float(corr.max()), "coil_corr_mean_offdiag": float(corr.sum() / (Cn * (Cn - 1))),
             "eig_ratio": float(ev.max() / ev.clamp_min(1e-30).min()),
             "noise_level": float(torch.diagonal(Sigma).real.mean())}
    return Sigma.to(kspace.dtype), winfo


def whitening_matrix(Sigma: torch.Tensor, eig_floor: float = 1e-6) -> torch.Tensor:
    """W = Σ^{-1/2} · sqrt(mean diag Σ): whitens the coil dimension while keeping the mean variance,
    so Σ = σ²I gives W = I exactly and lambda_geo keeps its meaning."""
    S = 0.5 * (Sigma + Sigma.conj().T).to(torch.complex128)
    evals, evecs = torch.linalg.eigh(S)
    evals = evals.real.clamp_min(eig_floor * float(evals.real.max()))
    W = evecs @ torch.diag((evals ** -0.5).to(evecs.dtype)) @ evecs.conj().T
    scale = torch.sqrt(torch.diagonal(S).real.mean())
    return (W * scale).to(Sigma.dtype)


def build_fit_tensors(kspace: torch.Tensor, mask: torch.Tensor, csm: torch.Tensor, Z: torch.Tensor, W):
    """Fit-side tensors (spec §4.1/§4.2):
        mask_fit  = mask ∪ Z                 (loss and hard-DC)
        kspace_dc = kspace with Z zeroed     (hard-DC, original coil basis)
        kspace_fit = W · kspace_dc           (loss only; == kspace_dc when W is None)
        csm_fit   = W · csm                  (loss only; == csm when W is None)
    The DC-loss denominator is NOT produced here: the caller keeps |Omega| = mask.sum()."""
    Z = Z.bool()
    mask_fit = torch.maximum(mask, Z.to(mask.dtype))
    kspace_dc = kspace * (~Z).unsqueeze(0).to(kspace.dtype)
    if W is None:
        return mask_fit, kspace_dc, kspace_dc, csm
    Wk = W.to(kspace.dtype)
    kspace_fit = torch.einsum("ij,jdhw->idhw", Wk, kspace_dc)
    csm_fit = torch.einsum("ij,jdhw->idhw", W.to(csm.dtype), csm)
    return mask_fit, kspace_dc, kspace_fit, csm_fit
