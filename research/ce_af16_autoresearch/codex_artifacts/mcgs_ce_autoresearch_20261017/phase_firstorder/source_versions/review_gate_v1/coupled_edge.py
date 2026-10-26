"""Exact source cross-edge alignment with the anchor's detached gates frozen.

This is the trainer method, including its stride-16 quantiles and circular
centered differences. It deliberately does not use losses.pair_prior.
"""
import copy
import math

import torch


def _validate(x):
    if not isinstance(x, torch.Tensor) or not x.is_complex():
        raise TypeError("paired image must be complex")
    if x.ndim != 4 or x.shape[0] != 2 or any(n < 2 for n in x.shape[1:]):
        raise ValueError("paired image must have shape (2,D,H,W), each dimension >=2")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("paired image must be finite")


def _gradient(x):
    gd = (torch.roll(x, -1, 0) - torch.roll(x, 1, 0)) * .5
    gh = (torch.roll(x, -1, 1) - torch.roll(x, 1, 1)) * .5
    gw = (torch.roll(x, -1, 2) - torch.roll(x, 1, 2)) * .5
    return torch.stack([gd, gh, gw], dim=0)


class FrozenEdge:
    @property
    def info(self):
        return copy.deepcopy(self._info)

    def merit(self, x):
        _validate(x)
        if (tuple(x.shape), x.dtype, x.device) != self._signature:
            raise ValueError("paired image must match anchor shape, dtype, and device")
        gp, gc = _gradient(x[0].abs()), _gradient(x[1].abs())
        dot = (gp * gc).sum(dim=0)
        ell = 1.0 - dot.pow(2) / ((gp * gp).sum(dim=0) * (gc * gc).sum(dim=0) + 1e-8)
        return (self._weight * ell).sum() / (self._weight.sum() + 1e-8)


def build_frozen_edge(x0, *, p=.9, kappa=30.):
    _validate(x0)
    if not math.isfinite(p) or not 0 <= p <= 1 or not math.isfinite(kappa) or kappa < 0:
        raise ValueError("p must be in [0,1] and kappa must be finite and nonnegative")
    with torch.no_grad():
        gp, gc = _gradient(x0[0].abs()), _gradient(x0[1].abs())
        gpm, gcm = gp.norm(dim=0), gc.norm(dim=0)
        tau_p = torch.quantile(gpm.flatten()[::16], p)
        tau_c = torch.quantile(gcm.flatten()[::16], p)
        weight = (torch.sigmoid(kappa * (gpm - tau_p)) * torch.sigmoid(kappa * (gcm - tau_c))).detach()
    result = FrozenEdge()
    result._signature = (tuple(x0.shape), x0.dtype, x0.device)
    result._weight = weight.clone()
    result._info = {"p": float(p), "kappa": float(kappa), "tau_plain": float(tau_p),
                    "tau_contrast": float(tau_c), "weight_sum": float(weight.sum()),
                    "quantile_stride": 16, "gates_frozen_within_outer_step": True}
    return result
