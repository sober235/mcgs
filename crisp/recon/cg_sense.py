"""复数 CG-SENSE。求解 (A^H A + lam I) x = A^H y,A = P_Omega F S。

同一实例既作重建器,也作 spec §4.3 要求的"所有方法共用的 DC 算子"。
"""
from __future__ import annotations

import torch

from crisp.metrics.coil_domain import fft3c, ifft3c


def _normal_op(x, csm, mask, lam):
    k = mask.unsqueeze(0) * fft3c(csm * x.unsqueeze(0))
    return (ifft3c(k) * csm.conj()).sum(0) + lam * x


def cg_sense(y, csm, mask, lam: float = 0.0, iters: int = 30, x0=None,
             device: str | None = None) -> torch.Tensor:
    if device is not None:
        y, csm, mask = y.to(device), csm.to(device), mask.to(device)
    b = (ifft3c(mask.unsqueeze(0) * y) * csm.conj()).sum(0)
    x = torch.zeros_like(b) if x0 is None else x0.clone().to(b.device)
    r = b - _normal_op(x, csm, mask, lam)
    p = r.clone()
    rs = torch.vdot(r.reshape(-1), r.reshape(-1)).real
    for _ in range(iters):
        if rs.sqrt() < 1e-12:
            break
        ap = _normal_op(p, csm, mask, lam)
        alpha = rs / torch.vdot(p.reshape(-1), ap.reshape(-1)).real.clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = torch.vdot(r.reshape(-1), r.reshape(-1)).real
        p = r + (rs_new / rs.clamp_min(1e-30)) * p
        rs = rs_new
    return x
