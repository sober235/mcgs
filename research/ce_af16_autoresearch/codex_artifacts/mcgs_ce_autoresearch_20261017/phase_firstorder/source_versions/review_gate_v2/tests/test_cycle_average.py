"""CPU algebra checks for the exact mean of the eight source cycle phases."""
import copy
import importlib
import importlib.util
import itertools
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "nonlocal_autoresearch/trainer/paired_gs/losses"
SHIFTS = tuple(itertools.product((0, 1), repeat=3))


@pytest.fixture(scope="module")
def cycle():
    path = ROOT / "cycle_average.py"
    assert path.exists(), "eight-phase prior has not been implemented"
    sys.path.insert(0, str(ROOT))
    try:
        spec = importlib.util.spec_from_file_location("density_mm_cycle_average", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ROOT))


@pytest.fixture(scope="module")
def shipped():
    package = types.ModuleType("density_mm_cycle_reference_losses")
    package.__path__ = [str(SOURCE)]
    sys.modules[package.__name__] = package
    return tuple(importlib.import_module(package.__name__ + "." + name)
                 for name in ("edge_tv", "phase_tv", "wavelet_l1"))


def image(dtype=torch.complex128, shape=(2, 6, 7, 8), seed=47):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype) + (1.1 + .7j)


def settings(**changes):
    options = dict(lambda_tv=(.1, .1), lambda_phase=(.03, .03),
                   lambda_wavelet=(.12, .1), kappa=(.6, .8))
    options.update(changes)
    return options


def source_mean(x, shipped, *, tv=(.1, .1), phase=(.03, .03),
                wave=(.12, .1), kappa=(.6, .8)):
    edge, phase_source, wave_source = shipped
    vols = [x] if x.ndim == 3 else list(x)
    expected = sum(t * edge.edge_preserving_tv(v, k, 0., .04, True)
                   + p * phase_source.phase_tv(v, k, 0., .04)
                   for v, t, p, k in zip(vols, tv, phase, kappa))
    expected += sum(wave_source.wavelet_l1(vols, "db4", 1, 0., shift,
                                           "complex", wave)
                    for shift in SHIFTS) / 8
    return expected


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_mean_matches_actual_source_value_and_gradient_with_tv_phase_once(cycle, shipped, dtype):
    """Catches a missing phase, wrong 1/8 strength or repeated TV/phase penalty."""
    x0 = image(dtype)
    prior = cycle.build_cycle_average_prior(x0, **settings())
    point = x0.clone().requires_grad_()
    actual = prior.merit(point)
    expected = source_mean(point, shipped)
    tol = 4e-6 if dtype == torch.complex64 else 3e-12
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol * .02)
    actual_gradient = torch.autograd.grad(actual, point)[0]
    source_gradient = torch.autograd.grad(expected, point)[0]
    torch.testing.assert_close(actual_gradient, source_gradient,
                               rtol=5e-5 if dtype == torch.complex64 else 3e-11,
                               atol=3e-8 if dtype == torch.complex64 else 3e-13)


def test_average_is_not_a_single_fixed_phase_on_padded_nontrivial_image(cycle, shipped):
    """Catches retaining a fixed111 wavelet term instead of the original expectation."""
    x0 = image()
    prior = cycle.build_cycle_average_prior(x0, **settings(lambda_tv=0., lambda_phase=0.))
    expected = source_mean(x0, shipped, tv=(0., 0.), phase=(0., 0.))
    fixed = shipped[2].wavelet_l1(list(x0), "db4", 1, 0., (1, 1, 1),
                                 "complex", (.12, .1))
    assert abs(float(expected - fixed)) > 1e-4
    torch.testing.assert_close(prior.merit(x0), expected, rtol=2e-13, atol=2e-13)


def test_combined_quadratic_bounds_touches_and_is_tangent(cycle):
    """Catches summing the homogeneous forms but omitting or duplicating constants."""
    x0 = image()
    prior = cycle.build_cycle_average_prior(x0, **settings())
    point = x0.clone().requires_grad_()
    merit = prior.merit(point)
    bound = prior.quadratic(point) + prior.majorizer_constant
    torch.testing.assert_close(merit, bound, rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(torch.autograd.grad(merit, point)[0],
                               torch.autograd.grad(bound, point)[0],
                               rtol=3e-10, atol=3e-12)
    for seed, scale in ((5, .05), (6, .5), (7, 2.), (8, 20.)):
        candidate = image(seed=seed) * scale
        gap = prior.quadratic(candidate) + prior.majorizer_constant - prior.merit(candidate)
        assert float(gap) >= -3e-12


def test_projected_small_fit_has_complex_linear_hermitian_psd_normal(cycle):
    """Catches a real-linear or non-Hermitian sum when used in a complex density fit."""
    generator = torch.Generator().manual_seed(28)
    basis = torch.randn((216, 5), generator=generator, dtype=torch.complex128)
    density = torch.randn((5,), generator=generator, dtype=torch.complex128)
    data = torch.randn((7, 5), generator=generator, dtype=torch.complex128)
    weights = torch.linspace(.2, 1.1, 7, dtype=torch.float64)
    x0 = (basis @ density).reshape(6, 6, 6)
    prior = cycle.build_cycle_average_prior(x0, lambda_tv=.1, lambda_phase=.03,
                                             lambda_wavelet=.12, kappa=.8)

    def quadratic(rho):
        return (weights * (data @ rho).abs().square()).sum() + prior.quadratic(
            (basis @ rho).reshape_as(x0))

    def normal(rho):
        point = rho.detach().requires_grad_()
        return torch.autograd.grad(quadratic(point), point)[0] / 2

    identity = torch.eye(5, dtype=torch.complex128)
    matrix = torch.stack([normal(e) for e in identity], 1)
    torch.testing.assert_close(matrix, matrix.H, rtol=2e-12, atol=2e-12)
    assert float(torch.linalg.eigvalsh(matrix).min()) > 0.
    other = torch.randn((5,), generator=generator, dtype=torch.complex128)
    a, b = .3 + .7j, -.4 + .2j
    torch.testing.assert_close(normal(a * density + b * other),
                               a * normal(density) + b * normal(other),
                               rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(quadratic(density),
                               torch.vdot(density, matrix @ density).real,
                               rtol=3e-12, atol=3e-12)


def test_scalar_single_volume_interface_retains_original_per_scan_strength(cycle, shipped):
    x0 = image(shape=(6, 7, 8))
    prior = cycle.build_cycle_average_prior(x0, lambda_tv=.1, lambda_phase=.03,
                                             lambda_wavelet=.12, kappa=.6)
    expected = source_mean(x0, shipped, tv=(.1,), phase=(.03,), wave=(.12,), kappa=(.6,))
    torch.testing.assert_close(prior.merit(x0), expected, rtol=3e-12, atol=3e-12)


def test_kappa_single_pass_iterable_is_normalized_before_component_reuse(cycle, shipped):
    """Catches consuming a source-accepted channel iterable again for each phase."""
    x0 = image()
    prior = cycle.build_cycle_average_prior(x0, **settings(kappa=iter((.6, .8))))
    torch.testing.assert_close(prior.merit(x0), source_mean(x0, shipped))


def test_dynamic_wavelet_transform_counter_includes_all_eight_actual_phases(cycle, monkeypatch):
    """Catches caching construction work or charging one transform for eight calls."""
    import ptwt
    transform = ptwt.wavedec3
    actual_calls = []

    def observed_transform(*args, **kwargs):
        actual_calls.append(1)
        return transform(*args, **kwargs)

    monkeypatch.setattr(ptwt, "wavedec3", observed_transform)
    x0 = image()
    prior = cycle.build_cycle_average_prior(x0, **settings())
    construction = prior.wavelet_transform_batches
    assert construction == len(actual_calls) and construction > 0
    prior.merit(x0)
    assert prior.wavelet_transform_batches == construction + 8
    assert prior.wavelet_transform_batches == len(actual_calls)
    prior.quadratic(x0)
    assert prior.wavelet_transform_batches == construction + 16
    assert prior.wavelet_transform_batches == len(actual_calls)
    _ = prior.info
    _ = prior.majorizer_constant
    assert prior.wavelet_transform_batches == construction + 16
    disabled = cycle.build_cycle_average_prior(x0, **settings(lambda_wavelet=0.))
    disabled.merit(x0)
    disabled.quadratic(x0)
    assert disabled.wavelet_transform_batches == 0


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_inputs_outputs_and_information_keep_dtype_and_are_not_aliased(cycle, dtype):
    """Catches altered anchor/support/strengths or refreshed/externally mutable metadata."""
    x0 = image(dtype)
    support = torch.ones_like(x0, dtype=torch.bool)
    strengths = [.12, .1]
    snapshot = x0.clone(), support.clone(), strengths[:]
    prior = cycle.build_cycle_average_prior(x0, **settings(
        lambda_wavelet=strengths, structural_support=support))
    candidate = image(dtype, seed=19)
    before = prior.merit(candidate).clone(), prior.quadratic(candidate).clone()
    info = copy.deepcopy(prior.info)
    assert before[0].dtype == x0.real.dtype
    assert before[1].dtype == x0.real.dtype
    assert prior.majorizer_constant.dtype == x0.real.dtype
    torch.testing.assert_close(x0, snapshot[0], rtol=0., atol=0.)
    assert torch.equal(support, snapshot[1])
    assert strengths == snapshot[2]
    x0.zero_()
    support.zero_()
    strengths[0] = 90.
    leaked = prior.info
    leaked["lambda_wavelet"][0] = 90.
    leaked["base_tv_phase"]["phase"]["structural_omitted_pairs"] = 900
    torch.testing.assert_close(prior.merit(candidate), before[0], rtol=0., atol=0.)
    torch.testing.assert_close(prior.quadratic(candidate), before[1], rtol=0., atol=0.)
    assert prior.info == info
    assert info["mode"] == "uniform8"
    assert info["wavelet_phase_count"] == 8
    assert info["wavelet_transforms_per_evaluation"] == 8
    assert info["lambda_tv"] == [.1, .1]
    assert info["lambda_phase"] == [.03, .03]
    assert info["lambda_wavelet"] == [.12, .1]
    assert info["channel_reduction"] == "sum"
    assert info["wavelet_shifts"] == [list(s) for s in SHIFTS]
    assert info["base_tv_phase"]["lambda_wavelet"] == [0., 0.]
    for phase, shift in zip(info["wavelet_phases"], SHIFTS):
        assert phase["lambda_tv"] == [0., 0.]
        assert phase["lambda_phase"] == [0., 0.]
        assert phase["lambda_wavelet"] == [.015, .0125]
        assert phase["wavelet_shift"] == list(shift)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_zero_floor_matches_source_mean_and_has_finite_constant(cycle, shipped, dtype):
    """Catches omitting the source wavelet/TV/phase smoothing floor at exactly zero."""
    x0 = torch.zeros((2, 6, 7, 8), dtype=dtype)
    prior = cycle.build_cycle_average_prior(x0, **settings())
    expected = source_mean(x0, shipped)
    torch.testing.assert_close(prior.merit(x0), expected,
                               rtol=3e-6 if dtype == torch.complex64 else 2e-12, atol=0.)
    torch.testing.assert_close(prior.quadratic(x0) + prior.majorizer_constant,
                               expected, rtol=3e-6 if dtype == torch.complex64 else 2e-12, atol=0.)
    assert torch.isfinite(prior.majorizer_constant)


def test_nonstructural_one_zero_obstruction_propagates(cycle):
    x0 = image()
    x0[0, 0, 0, 0] = 0.
    with pytest.raises(ValueError, match="3 nonstructural single-zero"):
        cycle.build_cycle_average_prior(x0, **settings())


def test_structural_support_is_respected_by_entire_composition(cycle):
    """Catches dropping support when constructing a wavelet-only component."""
    x0 = image()
    support = torch.ones_like(x0, dtype=torch.bool)
    support[0, 0, 0, 0] = False
    x0[~support] = 0.
    prior = cycle.build_cycle_average_prior(x0, **settings(structural_support=support))
    candidate = image(seed=21)
    candidate[~support] = 0.
    assert float(prior.quadratic(candidate) + prior.majorizer_constant - prior.merit(candidate)) >= 0.
    assert prior.info["base_tv_phase"]["phase"]["structural_omitted_pairs"] == 3
    candidate[~support] = .001
    with pytest.raises(ValueError, match="structural support"):
        prior.merit(candidate)
    with pytest.raises(ValueError, match="structural support"):
        prior.quadratic(candidate)


@pytest.mark.parametrize("shift", [(1, 1, 1), (0, 0, 0), None])
def test_caller_single_phase_is_rejected_even_if_it_matches_an_averaged_phase(cycle, shift):
    with pytest.raises(ValueError, match="wavelet_shift"):
        cycle.build_cycle_average_prior(image(), **settings(wavelet_shift=shift))


@pytest.mark.parametrize("option", [dict(tv_vector=False), dict(wavelet="db2"),
                                    dict(wavelet_levels=2), dict(wavelet_on="magnitude"),
                                    dict(wavelet_eps=.01), dict(unknown_setting=True)])
def test_unsupported_source_settings_are_not_silently_ignored(cycle, option):
    with pytest.raises((ValueError, TypeError)):
        cycle.build_cycle_average_prior(image(), **settings(**option))


@pytest.mark.parametrize("option", [dict(lambda_wavelet=-.1), dict(lambda_wavelet=(.1,)),
                                    dict(lambda_wavelet=float("nan")), dict(lambda_wavelet=True),
                                    dict(kappa=0.), dict(lambda_tv=-.1), dict(w_floor=2.),
                                    dict(phase_eps=float("nan"))])
def test_source_strength_and_parameter_validation_propagates(cycle, option):
    with pytest.raises((ValueError, TypeError)):
        cycle.build_cycle_average_prior(image(), **settings(**option))


def test_original_strength_dtype_overflow_and_eighth_strength_underflow_fail(cycle):
    x0 = image(torch.complex64)
    for strength in (1e39, 1e-45):
        with pytest.raises(ValueError, match="image dtype"):
            cycle.build_cycle_average_prior(x0, **settings(lambda_wavelet=strength))


def test_disabled_wavelets_allow_small_images_without_changing_tv_phase(cycle, shipped):
    x0 = image(shape=(2, 2, 3, 4))
    prior = cycle.build_cycle_average_prior(x0, **settings(lambda_wavelet=0.))
    expected = sum(
        .1 * shipped[0].edge_preserving_tv(v, k, 0., .04, True)
        + .03 * shipped[1].phase_tv(v, k, 0., .04)
        for v, k in zip(x0, (.6, .8)))
    torch.testing.assert_close(prior.merit(x0), expected)
    assert prior.info["wavelet_phase_count"] == 8
    assert prior.info["wavelet_transforms_per_evaluation"] == 0


def test_enabled_wavelets_keep_source_minimum_shape_rejection(cycle):
    with pytest.raises(ValueError, match="spatial dimensions >= 6"):
        cycle.build_cycle_average_prior(image(shape=(2, 5, 6, 7)), **settings())


def test_evaluation_rejects_wrong_shape_dtype_nonfinite_and_noncomplex(cycle):
    x0 = image()
    prior = cycle.build_cycle_average_prior(x0, **settings())
    for bad in (x0[:, :-1], x0.real, x0.to(torch.complex64), x0 * float("nan")):
        with pytest.raises((ValueError, TypeError)):
            prior.merit(bad)
        with pytest.raises((ValueError, TypeError)):
            prior.quadratic(bad)
