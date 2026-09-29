"""Scoring for arc/run_spec.py (spec 2026-09-28 §3.2).

score_channel() is arc/cycle.py's score_contrast() with two constants taken from the spec's `scoring`
block -- the anatomy-mask threshold (0.05 * P99.9 in cycle.py and diag_artefact/headonly.py) and whether
the brain white-matter grain metric N is computed -- and with torch.quantile replaced by numpy above
16M elements. hfen/sharp_ratio/grain are verbatim copies (tests/test_spec_scoring.py compares the source).
With anatomy_rel=0.05 and grain_ratio=True the numbers equal cycle.py's (tests/test_scoring_equivalence.py).
arc/cycle.py and diag_artefact/headonly.py are not modified.

The anatomy mask (scoring.anatomy_rel) drives head_psnr, head_ssim, head_hfen, head_nmse, head_frac,
head_psnr_cplx and cos_phase. comp, beat, sharp and grain use brain constants and ignore anatomy_rel: sharp and
grain use their own 0.05 * max mask, beat compares whole-volume SSIM / NMSE / HFEN with the zero-filled image,
and comp weighs whole-volume metrics with weights tuned on the brain. So comp, and the json's combined and
both_beat built from comp and beat, are not selection criteria outside the brain. grain_ratio (N) is the brain
white-matter metric with its own masks, switched by scoring.grain_ratio.
Keys added here that cycle.py does not have: head_nmse (sum over the anatomy mask of (rec - gt)^2 / gt^2, the
knee's anatomy-region NMSE, spec §5.4), ssim_mode / zf_ssim_mode ("3d", or "slicewise" when
evaluate_reconstruction fell back to slice-wise SSIM).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "paired_gs"):         # paired_gs ends up first: a root-level metrics.py or data/ cannot
    while str(_p) in sys.path:                # shadow it, whatever sys.path the importer had
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))
from data.transforms import fft3c, ifft3c  # noqa: E402
from metrics import evaluate_reconstruction  # noqa: E402
from diag_artefact.headonly import grain_ratio as _grain_ratio  # noqa: E402
from diag_artefact.headonly import head_hfen, head_psnr, head_ssim  # noqa: E402

QUANTILE_TORCH_MAX = 16_000_000


def quantile(t: torch.Tensor, q: float) -> torch.Tensor:
    """torch.quantile below 16M elements (bit-identical to cycle.py); numpy above (torch caps at 2**24)."""
    flat = t.reshape(-1)
    if flat.numel() <= QUANTILE_TORCH_MAX:
        return torch.quantile(flat, q)
    return torch.tensor(float(np.quantile(flat.detach().cpu().numpy(), q)), dtype=flat.dtype, device=flat.device)


# --- verbatim copies of arc/cycle.py hfen / sharp_ratio / grain (grain: quantile() instead of torch.quantile)
def hfen(rec, gt):
    """3D-Laplacian high-freq error norm vs GT (lower = fewer aliasing/speckle residuals)."""
    def lap(v):
        return (6.0 * v - torch.roll(v, 1, -3) - torch.roll(v, -1, -3)
                - torch.roll(v, 1, -2) - torch.roll(v, -1, -2)
                - torch.roll(v, 1, -1) - torch.roll(v, -1, -1))
    g = gt / (gt.max() + 1e-9)
    r = rec / (gt.max() + 1e-9)
    lg = lap(g)
    return float(torch.linalg.norm(lap(r) - lg) / (torch.linalg.norm(lg) + 1e-9))


def sharp_ratio(rec, gt):
    """Edge-sharpness ratio: mean per-voxel gradient magnitude rec/GT over the brain mask.
    ~1.0 = matched sharpness; <1 = blurred (under-shoots GT edges); >1 = over-sharp/noisy.
    Explicit anti-blur diagnostic so the optimizer can't trade aliasing for edge softening."""
    def gm(v):
        gx = torch.zeros_like(v); gy = torch.zeros_like(v); gz = torch.zeros_like(v)
        gx[:-1] = v[1:] - v[:-1]; gy[:, :-1] = v[:, 1:] - v[:, :-1]; gz[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
        return torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)
    brain = gt > 0.05 * gt.max()
    return float(gm(rec)[brain].mean() / (gm(gt)[brain].mean() + 1e-9))


def grain(rec, gt):
    """Non-anatomical grain: std of the high-pass residual (rec - smooth(rec)) over GT-FLAT brain
    voxels (where GT gradient is in the lowest 30%). Isolates speckle/grain in homogeneous tissue
    from genuine edge structure, so it is separable from sharp_ratio (which lives at edges).
    Lower = cleaner flat regions. Normalised by GT max."""
    import torch.nn.functional as F
    g = gt / (gt.max() + 1e-9)
    r = rec / (gt.max() + 1e-9)
    sm = F.avg_pool3d(r[None, None], kernel_size=3, stride=1, padding=1)[0, 0]   # local mean
    hp = r - sm                                                                   # high-pass residual
    def gmag(v):
        gx = torch.zeros_like(v); gy = torch.zeros_like(v); gz = torch.zeros_like(v)
        gx[:-1] = v[1:] - v[:-1]; gy[:, :-1] = v[:, 1:] - v[:, :-1]; gz[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
        return torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)
    brain = g > 0.05 * g.max()
    gg = gmag(g)
    thr = quantile(gg[brain], 0.30)            # GT-flat = lowest-30% gradient inside brain
    flat = brain & (gg <= thr)
    return float(hp[flat].std())
# --------------------------------------------------------------------------------------------------------


def score_anatomy(rec: np.ndarray, gt: np.ndarray, anatomy_rel: float, with_grain_ratio: bool) -> dict:
    """diag_artefact.headonly.score_headonly with the mask threshold as a parameter (brain: 0.05)."""
    rec = np.asarray(rec, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    head = gt > anatomy_rel * float(np.quantile(gt, 0.999))
    gmax = float(gt.max())
    g = gt[head].astype(np.float64)
    d = rec[head].astype(np.float64) - g
    return {
        "head_psnr": head_psnr(rec, gt, head, gmax),
        "head_ssim": head_ssim(rec, gt, head, gmax),
        "head_hfen": head_hfen(rec, gt, head, gmax),
        "head_nmse": float(np.sum(d * d) / np.sum(g * g)),
        "grain_ratio": _grain_ratio(rec, gt) if with_grain_ratio else None,
        "head_frac": float(head.mean()),
        "gmax": gmax,
    }


def _ssim_mode(met: dict) -> str:
    """metrics.evaluate_reconstruction switches to slice-wise SSIM on any exception; ssim_std is its only mark."""
    return "slicewise" if "ssim_std" in met else "3d"


def score_channel(tr, which, dev, anatomy_rel, with_grain_ratio):
    """arc/cycle.py score_contrast (lines 212-267) with the anatomy threshold and grain switch as parameters."""
    if which == "plain":
        csm, y, mask, ds = tr.csm_p, tr.kspace_p, tr.mask_p, tr.ds_p
        mask_fit, y_dc, tag = tr.mask_fit_p, tr.kspace_dc_p, "p"
    else:
        csm, y, mask, ds = tr.csm_c, tr.kspace_c, tr.mask_c, tr.ds_c
        mask_fit, y_dc, tag = tr.mask_fit_c, tr.kspace_dc_c, "c"
    gt = ds.get_data()["ground_truth"].to(dev)
    m = (mask if mask.ndim == 3 else mask[0]).to(dev).float()
    mf = (mask_fit if mask_fit.ndim == 3 else mask_fit[0]).to(dev).float()
    with torch.no_grad():
        x = tr._render(which)
        k_model = fft3c(csm * x.unsqueeze(0))
        y_full = ds.kspace_multicoil.to(dev)
        unobserved = (1.0 - m).unsqueeze(0)
        z = (mf - m).clamp_min(0.0).unsqueeze(0)
        unobserved_error = ((k_model - y_full) * unobserved).abs().square().sum()
        unobserved_energy = (y_full * unobserved).abs().square().sum().clamp_min(1e-12)
        nmse_unobs = float(unobserved_error / unobserved_energy)
        e_unobs = float(torch.sqrt(unobserved_error / unobserved_energy))
        unobs_x = unobserved * (1.0 - z)
        unobs_error_xz = ((k_model - y_full) * unobs_x).abs().square().sum()
        unobs_energy_xz = (y_full * unobs_x).abs().square().sum().clamp_min(1e-12)
        e_unobs_excl_z = float(torch.sqrt(unobs_error_xz / unobs_energy_xz))
        k_dc = mf.unsqueeze(0) * y_dc + (1.0 - mf).unsqueeze(0) * k_model
        kz_energy = float((k_dc * z).abs().square().sum())
        x_dc = (ifft3c(k_dc) * csm.conj()).sum(0)
        gm = gt.abs()
        hm = gm > anatomy_rel * quantile(gm.reshape(-1).float(), 0.999)
        dz = (x_dc - gt)[hm]
        head_psnr_cplx = float(20 * torch.log10(gm.max()) - 10 * torch.log10(dz.abs().square().mean()))
        wph = (x_dc.abs() * gm)[hm]
        cos_phase = float((wph * torch.cos(torch.angle(x_dc[hm]) - torch.angle(gt[hm]))).sum()
                          / wph.sum().clamp_min(1e-12))
        met = evaluate_reconstruction(x_dc, gt, compute_extended=False)
        hf = hfen(x_dc.abs(), gt.abs())
        sr = sharp_ratio(x_dc.abs(), gt.abs())
        gr = grain(x_dc.abs(), gt.abs())
        ho = score_anatomy(x_dc.abs().cpu().numpy(), gt.abs().cpu().numpy(), anatomy_rel, with_grain_ratio)
        x_zf = (ifft3c(m.unsqueeze(0) * y) * csm.conj()).sum(0)
        mz = evaluate_reconstruction(x_zf, gt, compute_extended=False)
        zf_hf = hfen(x_zf.abs(), gt.abs())
    psnr, ssim, nmse = float(met["psnr"]), float(met["ssim"]), float(met["nmse"])
    comp = psnr + 25.0 * ssim - 60.0 * nmse - 25.0 * hf - 40.0 * max(0.0, 1.0 - sr) - 200.0 * gr
    beat = (ssim >= float(mz["ssim"])) and (nmse <= float(mz["nmse"])) and (hf <= zf_hf)
    return dict(psnr=psnr, ssim=ssim, nmse=nmse, hfen=hf, sharp=sr, grain=gr,
                head_psnr_cplx=head_psnr_cplx, cos_phase=cos_phase,
                e_unobs=e_unobs, e_unobs_excl_z=e_unobs_excl_z, nmse_unobs=nmse_unobs, comp=comp, beat=beat,
                ssim_mode=_ssim_mode(met), zf_ssim_mode=_ssim_mode(mz),
                zf=(float(mz["psnr"]), float(mz["ssim"]), float(mz["nmse"]), zf_hf),
                kz_energy=kz_energy, physics=tr.physics_info.get(tag, {}), **ho)
