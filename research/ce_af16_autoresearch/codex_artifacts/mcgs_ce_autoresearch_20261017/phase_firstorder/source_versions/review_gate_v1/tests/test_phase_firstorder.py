"""New phase-first-order behavior, tested against real source priors/old driver."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

import runner
import surrogates
from cycle_average import build_cycle_average_prior


def old_module(name, filename):
    path = Path(__file__).resolve().parents[2] / "frequency_lowreg_mm" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def image(dtype):
    real = torch.float64 if dtype == torch.complex128 else torch.float32
    values = torch.linspace(.02, 1.1, 54, dtype=real).reshape(2, 3, 3, 3)
    phase = torch.linspace(-.17, .23, 54, dtype=real).reshape_as(values)
    return (values * torch.exp(1j * phase)).to(dtype)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_skip_phase_positive_terms_preserves_merit_and_nonzero_phase_gradient(dtype):
    x = image(dtype).requires_grad_()
    prior = surrogates.build_frozen_prior(x.detach(), lambda_tv=0., lambda_phase=.03,
                                          lambda_wavelet=0., kappa=.6)
    before = prior.merit(x)
    g, = torch.autograd.grad(before, x)
    assert float(g.abs().sum()) > 0
    # No remaining Q terms: this is independently the zero homogeneous form.
    reduced = prior.quadratic(x, include_phase=False)
    rg, = torch.autograd.grad(reduced, x)
    assert float(reduced) == 0 and torch.count_nonzero(rg) == 0
    torch.testing.assert_close(prior.merit(x), before, rtol=0, atol=0)
    assert float(prior.quadratic(torch.ones_like(x))) > 0


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_default_full_quadratic_matches_frozen_L_exactly(dtype):
    old = old_module("_phase_frozen_L_surrogates", "surrogates.py")
    x = image(dtype)
    settings = dict(lambda_tv=(.1,.1), lambda_phase=.03, lambda_wavelet=0., kappa=(.6,.8))
    current, previous = (module.build_frozen_prior(x, **settings) for module in (surrogates, old))
    point = (x * (.8 + .2j)).requires_grad_()
    a, b = current.quadratic(point), previous.quadratic(point)
    ga, = torch.autograd.grad(a, point)
    gb, = torch.autograd.grad(b, point)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)


def test_uniform8_forwards_skip_without_extra_wavelet_transforms():
    x = torch.linspace(.1, 2., 432, dtype=torch.float64).reshape(2,6,6,6).to(torch.complex128)
    prior = build_cycle_average_prior(x, lambda_tv=0., lambda_phase=.03,
                                      lambda_wavelet=(.12,.1), kappa=.6)
    initial = prior.wavelet_transform_batches
    reduced = prior.quadratic(x, include_phase=False)
    assert prior.wavelet_transform_batches - initial == 8
    expected = sum(part.quadratic(x) for part in prior._parts[1:])
    torch.testing.assert_close(reduced, expected, rtol=0, atol=0)


class ImagePhysics:
    def __init__(self, dtype=torch.complex128):
        self.initial_density = image(dtype).reshape(2,-1).clone()
        self.target = self.initial_density * (.85 + .03j)
        self.counters = {"render_pairs":0,"renderer_calls":0}

    def render(self, value):
        self.counters["render_pairs"] += 1
        self.counters["renderer_calls"] += 2
        return tuple(value.reshape(2,3,3,3))

    def structural_support(self):
        self.render(torch.ones_like(self.initial_density))
        return tuple(torch.ones(3,3,3,dtype=torch.bool) for _ in range(2))

    def data_loss(self, pre, post, zero_target=False):
        residual = torch.stack((pre,post)).reshape(2,-1)
        if not zero_target:
            residual = residual-self.target
        return residual.abs().square().sum()/27


SETTINGS = dict(lambda_tv=(.1,.1),lambda_phase=.03,lambda_wavelet=0.,kappa=(.6,.8))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_default_full_driver_update_matches_frozen_L(dtype):
    previous = old_module("_phase_frozen_L_runner", "runner.py")
    cfg = dict(outer_steps=1,cg_steps=8,true_residual_diagnostics=True)
    current = runner.optimize_density(ImagePhysics(dtype), prior_kwargs=SETTINGS,
                                      config=runner.SolverConfig(**cfg))
    old = previous.optimize_density(ImagePhysics(dtype), prior_kwargs=SETTINGS,
                                    config=previous.SolverConfig(**cfg))
    torch.testing.assert_close(current.density,old.density,rtol=0,atol=0)
    assert current.trace[0]["damping"] == old.trace[0]["damping"]


def test_first_order_keeps_full_probe_eta_and_skips_only_CG_endpoint_curvature():
    operator = ImagePhysics()
    flags = []
    class ObservedRealPrior:
        def __init__(self, real): self.real=real
        @property
        def info(self): return self.real.info
        @property
        def wavelet_transform_batches(self): return self.real.wavelet_transform_batches
        def merit(self, x): return self.real.merit(x)
        def quadratic(self, x, *, include_phase=True):
            flags.append(include_phase)
            return self.real.quadratic(x,include_phase=include_phase)
    def factory(x,**kwargs):
        return ObservedRealPrior(surrogates.build_frozen_prior(x,**kwargs))
    cfg=runner.SolverConfig(outer_steps=1,cg_steps=8,true_residual_diagnostics=True)
    original=runner.optimize_density(ImagePhysics(),prior_kwargs=SETTINGS,config=cfg)
    actual=runner.optimize_density(operator,prior_kwargs=SETTINGS,config=cfg,
                                   prior_factory=factory,phase_curvature_mode="first_order")
    row=actual.trace[0]
    assert row["damping"] == original.trace[0]["damping"]
    assert flags == [True]+[False]*(row["cg_matvecs"]+1)
    assert actual.work["full_curvature_normal_evaluations"] == 1
    assert actual.work["phase_first_order_normal_evaluations"] == row["cg_matvecs"]+1
    assert row["true_residual"]["phase_curvature_mode"] == "first_order"
    assert row["directional_derivative"] < 0 and row["merit_after"] < row["merit_before"]
    assert actual.work["vjp_pairs"] == actual.work["normal_evaluations"]+1
    assert actual.work["render_pairs"] == operator.counters["render_pairs"]


def test_first_order_zero_mode_exposes_zero_role_counts_without_work():
    result=runner.optimize_density(ImagePhysics(),prior_kwargs=SETTINGS,
           config=runner.SolverConfig(outer_steps=0),phase_curvature_mode="first_order")
    assert result.work["full_curvature_normal_evaluations"] == 0
    assert result.work["phase_first_order_normal_evaluations"] == 0
    assert result.work["render_pairs"] == result.work["vjp_pairs"] == 0


def test_bad_phase_mode_refused_before_any_physics():
    operator=ImagePhysics()
    with pytest.raises(ValueError,match="phase"):
        runner.optimize_density(operator,prior_kwargs=SETTINGS,phase_curvature_mode="ignored")
    assert operator.counters["render_pairs"] == 0


def test_default_uniform8_complete_update_matches_frozen_L():
    from test_phase_protocol import TinyPhysics
    previous=old_module("_phase_frozen_L_runner_wave", "runner.py")
    settings={**SETTINGS,"lambda_wavelet":(.12,.1)}
    cfg=dict(outer_steps=2,cg_steps=4,true_residual_diagnostics=True)
    current=runner.optimize_density(TinyPhysics(),prior_kwargs=settings,
                  config=runner.SolverConfig(**cfg),wavelet_mode="uniform8")
    old=previous.optimize_density(TinyPhysics(),prior_kwargs=settings,
                  config=previous.SolverConfig(**cfg),wavelet_mode="uniform8")
    torch.testing.assert_close(current.density,old.density,rtol=0,atol=0)
    assert [row["damping"] for row in current.trace] == [row["damping"] for row in old.trace]
    assert current.work["wavelet_transform_batches"] == old.work["wavelet_transform_batches"]


@pytest.mark.parametrize("dtype",[torch.complex64,torch.complex128])
def test_reduced_actual_uniform8_form_is_complex_Hermitian_PSD(dtype):
    real=torch.float32 if dtype == torch.complex64 else torch.float64
    amplitude=torch.linspace(.02,1.,432,dtype=real).reshape(2,6,6,6)
    angle=torch.linspace(-.1,.2,432,dtype=real).reshape_as(amplitude)
    anchor=(amplitude*torch.exp(1j*angle)).to(dtype)
    full=build_cycle_average_prior(anchor,lambda_tv=(.1,.1),lambda_phase=.03,
                                   lambda_wavelet=(.12,.1),kappa=(.6,.8))
    phase=surrogates.build_frozen_prior(anchor,lambda_tv=0.,lambda_phase=.03,
                                       lambda_wavelet=0.,kappa=(.6,.8))
    indices=torch.tensor([0,3,71,100,200,230,333,430])
    def expand(vector):
        output=torch.zeros_like(anchor).flatten()
        return output.scatter(0,indices,vector).reshape_as(anchor)
    def apply(vector,form,skip=False):
        point=vector.detach().clone().requires_grad_(True)
        value=form.quadratic(expand(point),include_phase=not skip)
        return torch.autograd.grad(value,point)[0]/2
    eye=torch.eye(8,dtype=dtype)
    reduced=torch.stack([apply(v,full,True) for v in eye],1)
    all_terms=torch.stack([apply(v,full) for v in eye],1)
    phase_terms=torch.stack([apply(v,phase) for v in eye],1)
    tolerance=dict(rtol=2e-5,atol=2e-5) if dtype == torch.complex64 else dict(rtol=2e-12,atol=2e-12)
    torch.testing.assert_close(reduced,reduced.H,**tolerance)
    # Compare the positive sum: subtraction in complex64 can erase a small
    # phase block beside much larger wavelet IRLS coefficients.
    algebra_tolerance=8*torch.finfo(real).eps
    torch.testing.assert_close(all_terms,reduced+phase_terms,rtol=algebra_tolerance,atol=algebra_tolerance)
    assert float(torch.linalg.eigvalsh(reduced.to(torch.complex128)).min()) >= -tolerance["atol"]
    for vector in eye:
        torch.testing.assert_close(apply(1j*vector,full,True),1j*apply(vector,full,True),**tolerance)
    # Data-null directions become SPD only after positive increment damping.
    assert float(torch.linalg.eigvalsh(reduced.to(torch.complex128)+.001*torch.eye(8)).min())>0


def test_first_order_failed_actual_Armijo_keeps_source_density_and_counts_attempt():
    physics=ImagePhysics()
    original=physics.initial_density.clone()
    original_loss=physics.data_loss
    physics.data_loss=lambda *args,**kwargs:1e-4*original_loss(*args,**kwargs)
    settings={**SETTINGS,"lambda_tv":0.}
    result=runner.optimize_density(physics,prior_kwargs=settings,edge_weight=0.,
             config=runner.SolverConfig(outer_steps=1,cg_steps=96,backtracks=1,true_residual_diagnostics=True),
             phase_curvature_mode="first_order")
    assert result.status == "backtracking_failed" and result.accepted_steps == 0
    row=result.trace[0]
    assert row["directional_derivative"]<0 and row["accepted"] is False
    assert len(row["backtracking"]) == result.work["merit_forward_evaluations"] == 1
    assert row["backtracking"][0]["merit"]>row["merit_before"]
    assert result.work["full_curvature_normal_evaluations"] == 1
    assert result.work["phase_first_order_normal_evaluations"] == row["cg_matvecs"]+1
    torch.testing.assert_close(result.density,original,rtol=0,atol=0)
    torch.testing.assert_close(physics.initial_density,original,rtol=0,atol=0)
