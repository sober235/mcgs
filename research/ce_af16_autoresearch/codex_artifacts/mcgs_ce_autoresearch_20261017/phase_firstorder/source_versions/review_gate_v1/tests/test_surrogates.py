"""CPU checks of the shipped priors and their frozen quadratic majorizers."""
import copy
import importlib
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "nonlocal_autoresearch/trainer/paired_gs/losses"


@pytest.fixture(scope="module")
def priors():
    path = ROOT / "surrogates.py"
    assert path.exists(), "frozen image priors have not been implemented"
    spec = importlib.util.spec_from_file_location("density_mm_surrogates", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def shipped():
    package = types.ModuleType("density_mm_reference_losses")
    package.__path__ = [str(SOURCE)]
    sys.modules[package.__name__] = package
    return tuple(importlib.import_module(package.__name__ + "." + name)
                 for name in ("edge_tv", "phase_tv", "wavelet_l1"))


def image(dtype=torch.complex128, shape=(2, 6, 7, 8), seed=31):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype) + (1.1 + .7j)


def build(priors, x0, **kwargs):
    settings = dict(lambda_tv=(.7, 1.1), lambda_phase=(.2, .4),
                    lambda_wavelet=(.01, .03), kappa=(.6, .8))
    if x0.shape[0] == 1:
        settings = {k: v[:1] for k, v in settings.items()}
    settings.update(kwargs)
    return priors.build_frozen_prior(x0, **settings)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_values_match_actual_source_per_channel_normalization(priors, shipped, dtype):
    """Catches omitted vector factor, axis means, wave floor, shift or channel weights."""
    x0 = image(dtype)
    prior = build(priors, x0)
    edge, phase, wave = shipped
    expected = sum(tv * edge.edge_preserving_tv(v, k, 0, .04, True)
                   + ptv * phase.phase_tv(v, k, 0, .04)
                   for v, tv, ptv, k in zip(x0, (.7, 1.1), (.2, .4), (.6, .8)))
    expected += wave.wavelet_l1(list(x0), "db4", 1, 0, (1, 1, 1),
                                "complex", (.01, .03))
    torch.testing.assert_close(prior.merit(x0), expected, rtol=3e-6, atol=2e-7)
    torch.testing.assert_close(prior.quadratic(x0) + prior.majorizer_constant,
                               expected, rtol=3e-6, atol=2e-7)


@pytest.mark.parametrize("term", ["tv", "phase", "wavelet", "all"])
def test_global_bound_touches_and_is_tangent_at_nonzero_images(priors, term):
    """Catches reversed IRLS bounds or incorrect complex dephasing/Young ratios."""
    x0 = image()
    enabled = {"lambda_" + name: ((.4, .6) if name == term or term == "all" else (0., 0.))
               for name in ("tv", "phase", "wavelet")}
    prior = build(priors, x0, **enabled)
    point = x0.detach().requires_grad_()
    true = prior.merit(point)
    quadratic = prior.quadratic(point) + prior.majorizer_constant
    torch.testing.assert_close(true, quadratic, rtol=2e-12, atol=2e-12)
    g = torch.autograd.grad(true, point)[0]
    h = torch.autograd.grad(quadratic, point)[0]
    torch.testing.assert_close(g, h, rtol=2e-9, atol=2e-10)
    for seed in range(8):
        point = image(seed=100 + seed) * (10 ** (seed / 3 - 1))
        assert float(prior.quadratic(point) + prior.majorizer_constant - prior.merit(point)) >= -2e-11


def test_wavelet_public_linear_helper_matches_source_and_has_vjp_adjoint(priors, shipped):
    """Catches a mismatched detail transform or assuming inverse equals adjoint."""
    x = image(shape=(1, 6, 7, 8))
    y = image(shape=x.shape, seed=71)
    wx = priors.wavelet_details(x, (1, 0, 1))
    wy = priors.wavelet_details(y, (1, 0, 1))
    combined = priors.wavelet_details((.3 + .4j) * x - y, (1, 0, 1))
    for a, b, c in zip(wx, wy, combined):
        torch.testing.assert_close(c, (.3 + .4j) * a - b)
    source = shipped[2].wavelet_l1(list(x), "db4", 1, 0, (1, 0, 1), "complex")
    actual = sum(torch.sqrt(w.abs().square() + 1e-12).sum() for w in wx) / x[0].numel()
    torch.testing.assert_close(actual, source)
    point = x.detach().requires_grad_()
    output = priors.wavelet_details(point, (1, 0, 1))
    test_duals = tuple(torch.randn_like(w) for w in output)
    adjoint = torch.autograd.grad(output, point, grad_outputs=test_duals)[0]
    lhs = sum((a.conj() * b).sum() for a, b in zip(wx, test_duals))
    rhs = (x.conj() * adjoint).sum()
    torch.testing.assert_close(lhs, rhs, rtol=2e-12, atol=2e-12)


def test_quadratic_defines_complex_hermitian_psd_matrix(priors):
    """Catches real-linear curvature, non-Hermitian signs and negative weights."""
    x0 = image(shape=(1, 2, 2, 2))
    prior = build(priors, x0, lambda_wavelet=(0.,))
    n = x0.numel()
    def normal(x):
        point = x.detach().requires_grad_()
        return torch.autograd.grad(prior.quadratic(point), point)[0] / 2
    eye = torch.eye(n, dtype=x0.dtype)
    matrix = torch.stack([normal(e.reshape_as(x0)).reshape(-1) for e in eye], 1)
    torch.testing.assert_close(matrix, matrix.H, rtol=2e-13, atol=2e-13)
    assert float(torch.linalg.eigvalsh(matrix).min()) >= -2e-13
    for e in eye:
        torch.testing.assert_close(normal((1j * e).reshape_as(x0)),
                                   1j * normal(e.reshape_as(x0)))
    point = image(shape=x0.shape, seed=9)
    torch.testing.assert_close(prior.quadratic(point),
                               torch.vdot(point.flatten(), matrix @ point.flatten()).real)


def test_zero_magnitude_dephasing_and_both_zero_phase_bounds(priors):
    """Zeros have valid value bounds without making a differentiability claim."""
    x0 = torch.zeros((1, 3, 4, 5), dtype=torch.complex128)
    prior = build(priors, x0, lambda_wavelet=(0.,))
    torch.testing.assert_close(prior.merit(x0), prior.quadratic(x0) + prior.majorizer_constant)
    assert prior.info["phase"]["both_zero_pairs"] > 0
    for seed in range(3):
        candidate = image(shape=x0.shape, seed=seed)
        assert float(prior.quadratic(candidate) + prior.majorizer_constant - prior.merit(candidate)) >= 0


def test_single_zero_magnitude_tv_still_touches_without_a_gradient_claim(priors):
    x0 = image(shape=(1, 3, 4, 5))
    x0[0, 0, 0, 0] = 0
    prior = build(priors, x0, lambda_phase=(0.,), lambda_wavelet=(0.,))
    torch.testing.assert_close(prior.quadratic(x0) + prior.majorizer_constant, prior.merit(x0))
    for seed in range(4):
        candidate = image(shape=x0.shape, seed=seed)
        assert float(prior.quadratic(candidate) + prior.majorizer_constant - prior.merit(candidate)) >= 0


def test_single_volume_scalar_interface_equals_batched_channel_sum(priors):
    x0 = image()
    options = dict(lambda_tv=.7, lambda_phase=.2, lambda_wavelet=.03, kappa=.6)
    batch = priors.build_frozen_prior(x0, **options)
    parts = [priors.build_frozen_prior(v, **options) for v in x0]
    candidate = image(seed=72)
    torch.testing.assert_close(batch.merit(candidate),
                               sum(prior.merit(v) for prior, v in zip(parts, candidate)))
    torch.testing.assert_close(batch.quadratic(candidate),
                               sum(prior.quadratic(v) for prior, v in zip(parts, candidate)))
    torch.testing.assert_close(batch.majorizer_constant, sum(prior.majorizer_constant for prior in parts))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_zero_tv_and_phase_source_floors_with_and_without_outer_smoothing(priors, shipped, dtype):
    x0 = torch.zeros((1, 3, 4, 5), dtype=dtype)
    for eps in (0., .04):
        prior = build(priors, x0, lambda_tv=1., lambda_phase=1., lambda_wavelet=0.,
                      tv_eps=eps, phase_eps=eps)
        expected = (shipped[0].edge_preserving_tv(x0[0], .6, 0, eps, True)
                    + shipped[1].phase_tv(x0[0], .6, 0, eps))
        torch.testing.assert_close(prior.merit(x0), expected, rtol=3e-15 if dtype == torch.complex128 else 2e-7, atol=0)
        torch.testing.assert_close(prior.quadratic(x0) + prior.majorizer_constant,
                                   expected, rtol=3e-15 if dtype == torch.complex128 else 2e-7, atol=0)


def test_nonstructural_single_zero_phase_endpoint_fails_with_count(priors):
    x0 = image(shape=(1, 3, 4, 5))
    x0[0, 0, 0, 0] = 0
    with pytest.raises(priors.PhaseZeroObstruction, match="3.*nonstructural"):
        build(priors, x0, lambda_wavelet=(0.,))


def test_structural_single_zero_phase_edges_are_omitted_only_on_allowed_domain(priors, shipped):
    x0 = image(shape=(1, 3, 4, 5))
    x0[0, 0, 0, 0] = 0
    support = torch.ones_like(x0, dtype=torch.bool)
    support[0, 0, 0, 0] = False
    prior = build(priors, x0, lambda_tv=(0.,), lambda_wavelet=(0.,), structural_support=support)
    assert prior.info["phase"]["structural_omitted_pairs"] == 3
    candidate = image(shape=x0.shape, seed=8)
    candidate[~support] = 0
    assert float(prior.quadratic(candidate) + prior.majorizer_constant - prior.merit(candidate)) >= -1e-12
    assert torch.isfinite(prior.merit(candidate))
    candidate[~support] = .001
    with pytest.raises(ValueError, match="structural support"):
        prior.merit(candidate)
    with pytest.raises(ValueError, match="structural support"):
        prior.quadratic(candidate)


def test_large_young_ratios_are_recorded_without_clipping(priors):
    x0 = torch.ones((1, 2, 2, 2), dtype=torch.complex128)
    x0[0, 0, 0, 0] = 1e-25
    prior = build(priors, x0, lambda_tv=(0.,), lambda_wavelet=(0.,))
    assert prior.info["phase"]["max_ratio"] >= 1e24
    assert prior.info["phase"]["ratio_clipped"] is False
    torch.testing.assert_close(prior.quadratic(x0) + prior.majorizer_constant, prior.merit(x0))


def test_exact_zero_wavelet_floor_matches_source_in_both_dtypes(priors, shipped):
    for dtype in (torch.complex64, torch.complex128):
        x0 = torch.zeros((1, 6, 7, 8), dtype=dtype)
        prior = build(priors, x0, lambda_tv=(0.,), lambda_phase=(0.,), lambda_wavelet=(1.,))
        expected = shipped[2].wavelet_l1(list(x0), "db4", 1, 0, (1, 1, 1), "complex")
        torch.testing.assert_close(prior.merit(x0), expected, rtol=0, atol=0)


def test_evaluating_and_mutating_inputs_does_not_refresh_frozen_weights(priors):
    x0 = image()
    prior = build(priors, x0)
    information = copy.deepcopy(prior.info)
    candidate = image(seed=7)
    merit = prior.merit(candidate).clone()
    quad = prior.quadratic(candidate).clone()
    x0.zero_()
    for _ in range(2):
        torch.testing.assert_close(prior.merit(candidate), merit, rtol=0, atol=0)
        torch.testing.assert_close(prior.quadratic(candidate), quad, rtol=0, atol=0)
    assert prior.info == information


@pytest.mark.parametrize("option", [dict(tv_vector=False), dict(wavelet="db2"),
                                    dict(wavelet_levels=2), dict(wavelet_on="magnitude"),
                                    dict(wavelet_eps=.01)])
def test_unsupported_source_modes_are_rejected(priors, option):
    with pytest.raises(ValueError, match="unsupported"):
        build(priors, image(), **option)


@pytest.mark.parametrize("option", [dict(kappa=0), dict(lambda_tv=-1), dict(w_floor=2),
                                    dict(phase_eps=float("nan")), dict(lambda_phase=(1.,)),
                                    dict(wavelet_shift=(0, 1, .5))])
def test_invalid_parameters_fail_before_a_solve(priors, option):
    with pytest.raises((ValueError, TypeError)):
        build(priors, image(), **option)


def test_nonfinite_wrong_shape_dtype_and_invalid_support_are_rejected(priors):
    x0 = image()
    prior = build(priors, x0)
    for bad in (x0[:, :-1], x0.real, x0.to(torch.complex64), x0 * float("nan")):
        with pytest.raises((ValueError, TypeError)):
            prior.merit(bad)
        with pytest.raises((ValueError, TypeError)):
            prior.quadratic(bad)
    for bad in (x0.real, x0 * float("nan"), x0[:, :1]):
        with pytest.raises((ValueError, TypeError)):
            build(priors, bad)
    with pytest.raises(ValueError, match="structural support"):
        build(priors, x0, structural_support=torch.zeros_like(x0, dtype=torch.bool))


def test_source_db4_small_grid_limitation_is_reported_explicitly(priors):
    with pytest.raises(ValueError, match="db4.*6"):
        build(priors, image(shape=(1, 4, 5, 6)))


def test_positive_strength_underflow_cannot_silently_disable_a_prior(priors):
    with pytest.raises(ValueError, match="represented"):
        build(priors, image(dtype=torch.complex64), lambda_phase=1e-100)
