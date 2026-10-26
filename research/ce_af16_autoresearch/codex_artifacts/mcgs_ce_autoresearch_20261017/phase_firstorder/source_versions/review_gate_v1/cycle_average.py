"""Uniform mean of the eight original level1 cycle-spin wavelet priors.

TV and phase penalties occur once. Each frozen wavelet phase has one eighth
of the original per-channel strength. This is the exact discrete expectation
of the source 0/1 shifts; padded periodic db4 is not claimed to be fully
translation invariant. No solver, checkpoint or GPU launch is imported here.
"""
from __future__ import annotations

import copy
import itertools

from surrogates import build_frozen_prior, _per_channel


_SHIFTS = tuple(itertools.product((0, 1), repeat=3))


class FrozenCycleAveragePrior:
    """Immutable composition with the same merit/quadratic interface as a prior."""

    def __init__(self, base, phases, original_wavelet_strengths):
        self._parts = (base, *phases)
        self._constant = sum(part.majorizer_constant for part in self._parts).detach()
        info = base.info
        # There is deliberately no single wavelet_shift in this construction.
        info.pop("wavelet_shift")
        # Base-only curvature statistics would misdescribe this combined form;
        # each component's statistics remain in its explicit information block.
        info.pop("quadratic_coefficient_stats")
        info.update(
            mode="uniform8",
            lambda_wavelet=list(original_wavelet_strengths),
            wavelet_shifts=[list(shift) for shift in _SHIFTS],
            wavelet_phase_count=8,
            wavelet_transforms_per_evaluation=8 if any(original_wavelet_strengths) else 0,
            wavelet_phase_weights=[1. / 8] * 8,
            base_tv_phase=base.info,
            wavelet_phases=[part.info for part in phases],
            majorizer_constant=float(self._constant),
        )
        self._info = info

    @property
    def info(self):
        return copy.deepcopy(self._info)

    @property
    def majorizer_constant(self):
        return self._constant.clone()

    @property
    def wavelet_transform_batches(self):
        """Actual cumulative source transform calls, including anchor construction."""
        return sum(part.wavelet_transform_batches for part in self._parts)

    def merit(self, x):
        """Sum one frozen TV/phase penalty and the eight source wavelet phases."""
        return sum(part.merit(x) for part in self._parts)

    def quadratic(self, x, *, include_phase=True):
        """Sum positive forms, optionally omit phase curvature without a bound."""
        return sum(part.quadratic(x, include_phase=include_phase) for part in self._parts)


def build_cycle_average_prior(x0, *, lambda_tv, lambda_phase, lambda_wavelet,
                              kappa, **source_options):
    """Compose the reviewed source-faithful priors without a single-shift option.

    Canonical source options, including structural support, are forwarded to
    every component and unsupported options retain the source builder's error.
    The source's private per-channel validator is reused so the *original*
    wavelet strength must fit the image dtype before division; each divided
    strength is then independently validated by its source builder.
    """
    if "wavelet_shift" in source_options:
        raise ValueError("wavelet_shift must be omitted for the explicit uniform8 average")
    base = build_frozen_prior(x0, lambda_tv=lambda_tv, lambda_phase=lambda_phase,
                             lambda_wavelet=0., kappa=kappa, **source_options)
    base_info = base.info
    original, _ = _per_channel(lambda_wavelet, base_info["channels"],
                               "lambda_wavelet", x0.device, x0.real.dtype)
    strengths = [value / 8 for value in original]
    phases = tuple(build_frozen_prior(x0, lambda_tv=0., lambda_phase=0.,
                                     lambda_wavelet=strengths, kappa=base_info["kappa"],
                                     wavelet_shift=shift, **source_options)
                   for shift in _SHIFTS)
    return FrozenCycleAveragePrior(base, phases, original)
