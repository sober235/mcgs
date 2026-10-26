"""Frozen source-faithful image priors and complex-linear quadratic majorizers.

The full quadratic is homogeneous: ``quadratic(x) + majorizer_constant`` bounds
``merit(x)`` on the declared structural support and touches at construction.
All edge weights, dephasing factors, Young ratios and cycle shifts are frozen.
The wavelet adjoint must be obtained by VJP; its inverse is not used here.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import importlib.util
import math
from pathlib import Path
import ptwt
import torch


_FLOOR = 1e-12
_SOURCE = Path(__file__).resolve().parent.parent / "nonlocal_autoresearch/trainer/paired_gs/losses"


class PhaseZeroObstruction(ValueError):
    """An active phase edge has exactly one zero without a structural guarantee."""

    def __init__(self, count: int):
        self.count = count
        super().__init__(f"{count} nonstructural single-zero phase pairs have no finite touching Young majorizer")


@lru_cache(maxsize=1)
def _source_edge():
    """Load only the read-only source loss, without importing the CUDA trainer."""
    spec = importlib.util.spec_from_file_location("_density_mm_source_edge_tv", _SOURCE / "edge_tv.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _channels(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or x.dtype not in (torch.complex64, torch.complex128):
        raise TypeError("image must be a complex64 or complex128 tensor")
    if x.ndim not in (3, 4) or any(n < 2 for n in x.shape[-3:]) or (x.ndim == 4 and x.shape[0] == 0):
        raise ValueError("image must have shape (D,H,W) or (C,D,H,W), with each spatial dimension >= 2")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("image contains nonfinite values")
    return x.unsqueeze(0) if x.ndim == 3 else x


def _shift(value) -> tuple[int, int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("wavelet_shift must contain three integers")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in value):
        raise ValueError("wavelet_shift must contain three integers")
    return tuple(value)


def wavelet_details(x: torch.Tensor, shifts=(1, 1, 1)) -> tuple[torch.Tensor, ...]:
    """Public complex-linear db4/level1 detail operator, exactly as in C.

    The source stacks real and imaginary parts per channel, rolls that real
    batch, calls ptwt.wavedec3(mode='periodic'), and excludes the coarse band.
    Returning complex detail bands allows callers to obtain its adjoint by VJP.
    """
    volumes = _channels(x)
    if any(n < 6 for n in volumes.shape[-3:]):
        raise ValueError("source db4 periodic level1 transform needs spatial dimensions >= 6")
    shifts = _shift(shifts)
    parts = [part for volume in volumes for part in (volume.real, volume.imag)]
    batch = torch.roll(torch.stack(parts, 0), shifts=shifts, dims=(1, 2, 3))
    coefficients = ptwt.wavedec3(batch, "db4", level=1, mode="periodic")
    result = []
    for detail in coefficients[1:]:
        for band in detail.values():
            pair = band.reshape(len(volumes), 2, *band.shape[1:])
            result.append(torch.complex(pair[:, 0], pair[:, 1]))
    return tuple(result)


def _per_channel(value, count: int, name: str, device, dtype, *, positive=False):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        values = [float(value)] * count
    else:
        try:
            values = [float(v) for v in value]
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be a scalar or a sequence of channel values") from error
        if len(values) != count:
            raise ValueError(f"{name} needs {count} channel values")
    if any(not math.isfinite(v) or v < 0 or (positive and v == 0) for v in values):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    tensor = torch.tensor(values, device=device, dtype=dtype)
    underflow = any(value > 0 and float(converted) == 0 for value, converted in zip(values, tensor))
    if not bool(torch.isfinite(tensor).all()) or underflow or (positive and bool((tensor <= 0).any())):
        raise ValueError(f"{name} cannot be represented in the image dtype")
    return values, tensor


def _m2(x):
    # Same two real squares as the source; abs().square() adds avoidable rounding.
    return x.real.square() + x.imag.square()


def _pairs(x, axis):
    lo = [slice(None)] * 4
    hi = [slice(None)] * 4
    lo[axis + 1] = slice(None, -1)
    hi[axis + 1] = slice(1, None)
    return x[tuple(lo)], x[tuple(hi)]


def _stats(values):
    selected = [v.detach().reshape(-1) for v in values if v.numel()]
    if not selected:
        return {"min": 0., "max": 0., "max_over_min_positive": 0.}
    low = min(float(v.min()) for v in selected)
    high = max(float(v.max()) for v in selected)
    positive = [v[v > 0] for v in selected]
    positive = [v for v in positive if v.numel()]
    min_positive = min(float(v.min()) for v in positive) if positive else 0.
    return {"min": low, "max": high,
            "max_over_min_positive": high / min_positive if min_positive else 0.}


class FrozenPrior:
    """Evaluate one immutable construction; evaluations never refresh its weights."""

    @property
    def info(self):
        return copy.deepcopy(self._info)

    @property
    def wavelet_transform_batches(self):
        return self._wavelet_transform_batches

    def _wavelet_details(self, x):
        self._wavelet_transform_batches += 1
        return wavelet_details(x, self._wavelet_shift)

    def _validate(self, x):
        volumes = _channels(x)
        if tuple(x.shape) != self._shape or x.dtype != self._dtype or x.device != self._device:
            raise ValueError("image shape, dtype and device must match the frozen construction")
        if self._support is not None and bool((volumes[~self._support] != 0).any()):
            raise ValueError("image violates declared structural support")
        return volumes

    def merit(self, x):
        """Actual nonlinear source penalties with the frozen weights and shifts."""
        x = self._validate(x)
        m = x.abs()
        # Preserve a differentiable zero for entirely disabled priors.
        total = x.real.sum() * 0
        if self._tv is not None:
            base = m[:, :-1, :-1, :-1]
            dx = m[:, 1:, :-1, :-1] - base
            dy = m[:, :-1, 1:, :-1] - base
            dz = m[:, :-1, :-1, 1:] - base
            gn = torch.sqrt(dx.square() + dy.square() + dz.square() + _FLOOR)
            value = self._edge.charbonnier(gn, self._tv_eps)
            total = total + (self._tv_weights * value).flatten(1).mean(1).mul(self._lambda_tv).sum() * 1.5
        for axis, (edge_weight, coefficient, sqrt_w, inverse_sqrt_w) in enumerate(self._phase):
            left, right = _pairs(x, axis)
            ml, mr = _pairs(m, axis)
            perp = torch.sqrt(torch.clamp((right - left).abs().square() - (mr - ml).square(), min=0) + _FLOOR)
            value = self._edge.charbonnier(perp, self._phase_eps)
            total = total + (edge_weight * value).flatten(1).mean(1).mul(self._lambda_phase).sum()
        if self._wave is not None:
            per_channel = torch.zeros_like(self._lambda_wavelet)
            for band in self._wavelet_details(x):
                per_channel = per_channel + torch.sqrt(_m2(band) + _FLOOR).flatten(1).sum(1)
            total = total + (per_channel * self._lambda_wavelet).sum() / self._voxels
        return total

    def quadratic(self, x, *, include_phase=True):
        """Homogeneous Hermitian PSD form; skipping phase is not a full bound.

        The default preserves the source touching quadratic. include_phase=False
        skips positive phase terms without subtractive cancellation; the merit
        and its full majorizer_constant are unchanged and must not be paired
        with the reduced form to claim touching/majorization.
        """
        if type(include_phase) is not bool:
            raise ValueError("include_phase must be a boolean")
        x = self._validate(x)
        total = x.real.sum() * 0
        if self._tv is not None:
            dephased = self._q.conj() * x
            base = dephased[:, :-1, :-1, :-1]
            dx = dephased[:, 1:, :-1, :-1] - base
            dy = dephased[:, :-1, 1:, :-1] - base
            dz = dephased[:, :-1, :-1, 1:] - base
            total = total + (self._tv * (_m2(dx) + _m2(dy) + _m2(dz))).sum()
        for axis, (edge_weight, coefficient, sqrt_w, inverse_sqrt_w) in enumerate(self._phase if include_phase else ()):
            left, right = _pairs(x, axis)
            difference = sqrt_w * left - inverse_sqrt_w * right
            total = total + (coefficient * _m2(difference)).sum()
        if self._wave is not None:
            for coefficient, band in zip(self._wave, self._wavelet_details(x)):
                total = total + (coefficient * _m2(band)).sum()
        return total


def build_frozen_prior(x0, *, lambda_tv, lambda_phase, lambda_wavelet, kappa,
                       tv_eps=.04, phase_eps=.04, w_floor=0.,
                       wavelet_shift=(1, 1, 1), structural_support=None,
                       tv_vector=True, wavelet="db4", wavelet_levels=1,
                       wavelet_on="complex", wavelet_eps=0.):
    """Freeze canonical source priors; strengths sum over channels, never average.

    A scalar lambda/kappa broadcasts deliberately; a sequence supplies each
    channel explicitly. Structural support is a caller-proved bool image mask.
    False entries must be exactly zero for every allowed future image. The
    single-zero phase exception is omitted only under that guarantee; its
    original source smoothing-floor contribution remains in the merit/constant.
    """
    if tv_vector is not True or wavelet != "db4" or wavelet_levels != 1 or wavelet_on != "complex" or wavelet_eps != 0:
        raise ValueError("unsupported source mode: require vector TV and complex db4 level1 wavelet at eps=0")
    for name, value in (("tv_eps", tv_eps), ("phase_eps", phase_eps), ("w_floor", w_floor)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if w_floor > 1:
        raise ValueError("w_floor must lie in [0,1]")
    initial = _channels(x0).detach().clone()
    shifts = _shift(wavelet_shift)
    prior = FrozenPrior()
    prior._wavelet_transform_batches = 0
    prior._shape, prior._dtype, prior._device = tuple(x0.shape), x0.dtype, x0.device
    prior._voxels = initial[0].numel()
    count = len(initial)
    real_dtype = initial.real.dtype
    parameters = {}
    for name, value in (("lambda_tv", lambda_tv), ("lambda_phase", lambda_phase),
                        ("lambda_wavelet", lambda_wavelet), ("kappa", kappa)):
        values, tensor = _per_channel(value, count, name, initial.device, real_dtype, positive=name == "kappa")
        parameters[name] = values
        setattr(prior, "_" + name, tensor)
    support = None
    if structural_support is not None:
        if not isinstance(structural_support, torch.Tensor) or structural_support.dtype != torch.bool or tuple(structural_support.shape) != tuple(x0.shape) or structural_support.device != x0.device:
            raise ValueError("structural support must be a bool tensor matching x0 shape and device")
        support = structural_support.detach().clone()
        support = support.unsqueeze(0) if support.ndim == 3 else support
        if bool((initial[~support] != 0).any()):
            raise ValueError("x0 violates declared structural support")
    prior._support = support
    prior._edge = _source_edge()
    prior._tv_eps, prior._phase_eps, prior._wavelet_shift = tv_eps, phase_eps, shifts
    prior._tv, prior._phase, prior._wave = None, [], None
    prior._q = None
    magnitude = initial.abs()
    channel_shape = (count, 1, 1, 1)
    kappas = prior._kappa.reshape(channel_shape)
    if bool((prior._lambda_tv > 0).any()):
        base = magnitude[:, :-1, :-1, :-1]
        delta = (magnitude[:, 1:, :-1, :-1] - base,
                 magnitude[:, :-1, 1:, :-1] - base,
                 magnitude[:, :-1, :-1, 1:] - base)
        gn = torch.sqrt(sum(v.square() for v in delta) + _FLOOR)
        prior._tv_weights = prior._edge._edge_weight(gn, kappas, w_floor)
        denominator = torch.sqrt(gn.square() + tv_eps * tv_eps) if tv_eps > 0 else gn
        cells = gn[0].numel()
        prior._tv = prior._lambda_tv.reshape(channel_shape) * 1.5 * prior._tv_weights / (2 * cells * denominator)
        prior._q = torch.where(magnitude > 0, initial / torch.where(magnitude > 0, magnitude, 1), torch.ones_like(initial))
    zero_count, both_zero, omissions, ratios = 0, 0, 0, []
    if bool((prior._lambda_phase > 0).any()):
        for axis in range(3):
            left, right = _pairs(initial, axis)
            ml, mr = _pairs(magnitude, axis)
            active = (prior._lambda_phase > 0).reshape(channel_shape).expand_as(ml)
            one_zero = ((ml == 0) ^ (mr == 0)) & active
            both_zero += int(((ml == 0) & (mr == 0) & active).sum())
            omit = torch.zeros_like(one_zero)
            if support is not None:
                sl, sr = _pairs(support, axis)
                omit = one_zero & (((ml == 0) & ~sl) | ((mr == 0) & ~sr))
            zero_count += int((one_zero & ~omit).sum())
            omissions += int(omit.sum())
            regular = (ml > 0) & (mr > 0) & active
            # Double ratios retain tiny nonzero magnitudes; no clipping or fake touching.
            ratio = torch.ones_like(ml, dtype=torch.float64)
            ratio[regular] = mr[regular].double() / ml[regular].double()
            if not bool(torch.isfinite(ratio).all()) or bool((ratio <= 0).any()):
                raise ValueError("Young ratios are not finite positive; rescale or use a wider image dtype")
            ratios.append(ratio[regular])
            sqrt_w = ratio.sqrt().to(real_dtype)
            inverse_sqrt_w = ratio.rsqrt().to(real_dtype)
            phase0 = torch.clamp((right - left).abs().square() - (mr - ml).square(), min=0)
            denominator = torch.sqrt(torch.sqrt(phase0 + _FLOOR).square() + phase_eps * phase_eps) if phase_eps > 0 else torch.sqrt(phase0 + _FLOOR)
            weight = prior._edge._edge_weight(mr - ml, kappas, w_floor)
            pairs = ml[0].numel()
            coefficient = prior._lambda_phase.reshape(channel_shape) * weight / (2 * pairs * denominator)
            coefficient = coefficient.masked_fill(omit | ~active, 0)
            prior._phase.append((weight, coefficient, sqrt_w, inverse_sqrt_w))
        if zero_count:
            raise PhaseZeroObstruction(zero_count)
    if bool((prior._lambda_wavelet > 0).any()):
        bands = prior._wavelet_details(initial)
        prior._wave = tuple(prior._lambda_wavelet.reshape(channel_shape) /
                            (2 * prior._voxels * torch.sqrt(_m2(band) + _FLOOR))
                            for band in bands)
    curvature = ([] if prior._tv is None else [prior._tv]) + [p[1] for p in prior._phase] + ([] if prior._wave is None else list(prior._wave))
    if any(not bool(torch.isfinite(v).all()) for v in curvature) or any(not bool(torch.isfinite(p[k]).all()) for p in prior._phase for k in (2, 3)):
        raise ValueError("nonfinite quadratic coefficients; rescale or use a wider image dtype")
    prior.majorizer_constant = (prior.merit(x0) - prior.quadratic(x0)).detach()
    if not bool(torch.isfinite(prior.majorizer_constant)):
        raise ValueError("nonfinite majorizer constant; rescale or use a wider image dtype")
    ratio_info = _stats(ratios)
    prior._info = dict(parameters, shape=list(x0.shape), channels=count, dtype=str(x0.dtype),
                       tv_eps=tv_eps, phase_eps=phase_eps, w_floor=w_floor,
                       tv_vector=True, tv_vector_scale=1.5, tv_vector_floor=_FLOOR,
                       phase_floor=_FLOOR, wavelet="db4", wavelet_levels=1,
                       wavelet_on="complex", wavelet_eps=0., wavelet_floor=_FLOOR,
                       wavelet_shift=list(shifts), channel_reduction="sum",
                       tv_shared_base_cells=math.prod(n - 1 for n in initial.shape[-3:]),
                       phase_axis_pair_counts=[_pairs(magnitude, axis)[0][0].numel() for axis in range(3)],
                       wavelet_normalizer_voxels=prior._voxels,
                       zero_magnitude_voxels=int((magnitude == 0).sum()),
                       structural_zero_voxels=int((~support).sum()) if support is not None else 0,
                       phase={"both_zero_pairs": both_zero, "nonstructural_single_zero_pairs": zero_count,
                              "structural_omitted_pairs": omissions, "min_ratio": ratio_info["min"],
                              "max_ratio": ratio_info["max"], "ratio_clipped": False},
                       quadratic_coefficient_stats=_stats(curvature),
                       majorizer_constant=float(prior.majorizer_constant),
                       source_edge_tv_sha256=hashlib.sha256((_SOURCE / "edge_tv.py").read_bytes()).hexdigest(),
                       source_wavelet_sha256=hashlib.sha256((_SOURCE / "wavelet_l1.py").read_bytes()).hexdigest())
    return prior
