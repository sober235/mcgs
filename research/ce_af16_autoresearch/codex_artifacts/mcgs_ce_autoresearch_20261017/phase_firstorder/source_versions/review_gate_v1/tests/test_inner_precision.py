"""CPU evidence that endpoint diagnostics measure, but never change, a solve."""
import copy
import dataclasses
import math

import pytest
import torch

import runner
import run_gpu as launch
from test_driver import ExplicitPhysics, PRIOR
from test_run_gpu import supervisor


def test_endpoint_residual_matches_direct_frozen_damped_matrix(monkeypatch):
    captured = []
    actual = runner.solve_pcg
    def observed(apply, rhs, **kwargs):
        result = actual(apply, rhs, **kwargs)
        captured.append((rhs.detach().clone(), result.x.detach().clone()))
        return result
    monkeypatch.setattr(runner, "solve_pcg", observed)
    physics = ExplicitPhysics()
    result = runner.optimize_density(physics, prior_kwargs=PRIOR, edge_weight=0.,
        config=runner.SolverConfig(outer_steps=1, cg_steps=3, cg_rtol=1e-12,
                                   true_residual_diagnostics=True))
    row = result.trace[0]
    rhs, delta = captured[0]
    # This fixture's actual data loss has H=diag(weights**2)+eta I.
    direct = rhs - (physics.weights.double().square() + row["damping"]) * delta
    norm = math.sqrt(runner._dot(direct, direct))
    evidence = row["true_residual"]
    assert evidence["diagnostic_only"] is True
    assert evidence["norm"] == pytest.approx(norm, rel=1e-13, abs=1e-14)
    assert evidence["rhs_norm"] == pytest.approx(row["gradient_norm"] * .5, rel=1e-14)
    assert evidence["rhs_norm"] == row["cg_residual_history"][0]
    assert evidence["relative"] == evidence["norm"] / evidence["rhs_norm"]
    assert evidence["recursive_norm"] == row["cg_residual_history"][-1]
    assert evidence["relative_norm_discrepancy"] == abs(evidence["norm"]-evidence["recursive_norm"])/evidence["rhs_norm"]
    assert evidence["normal_evaluations"] == result.work["true_residual_evaluations"] == 1
    assert result.work["normal_evaluations"] == row["cg_matvecs"] + 2
    assert result.work["render_pairs"] == physics.counters["render_pairs"]
    assert evidence["elapsed_seconds"] > 0
    assert result.work["true_residual_elapsed_seconds"] == evidence["elapsed_seconds"]


def test_endpoint_diagnostic_does_not_change_delta_or_acceptance():
    cfg = runner.SolverConfig(outer_steps=2, cg_steps=3, cg_rtol=1e-12)
    original = runner.optimize_density(ExplicitPhysics(), prior_kwargs=PRIOR, edge_weight=0., config=cfg)
    measured = runner.optimize_density(ExplicitPhysics(), prior_kwargs=PRIOR, edge_weight=0.,
        config=dataclasses.replace(cfg, true_residual_diagnostics=True))
    torch.testing.assert_close(measured.density, original.density, rtol=0, atol=0)
    assert measured.status == original.status and measured.accepted_steps == original.accepted_steps == 2
    for before, after in zip(original.trace, measured.trace):
        assert {k:v for k,v in after.items() if k != "true_residual"} == before
    for key in ("vjp_pairs", "normal_evaluations", "render_pairs"):
        assert measured.work[key] == original.work[key] + 2
    assert measured.work["renderer_calls"] == original.work["renderer_calls"] + 4


def test_zero_diagnostic_is_no_work_and_budget_never_widens():
    zero = runner.optimize_density(ExplicitPhysics(), prior_kwargs={},
        config=runner.SolverConfig(outer_steps=0, true_residual_diagnostics=True))
    assert zero.work["true_residual_evaluations"] == zero.work["true_residual_elapsed_seconds"] == 0
    result = runner.optimize_density(ExplicitPhysics(), prior_kwargs=PRIOR, edge_weight=0.,
        config=runner.SolverConfig(outer_steps=1,cg_steps=3,max_vjp_pairs=5,
                                   true_residual_diagnostics=True))
    assert result.status == "vjp_budget" and result.accepted_steps == 0
    assert result.work["vjp_pairs"] == 5 and result.work["true_residual_evaluations"] == 0


@pytest.mark.parametrize("factor", [1.,.75,.5])
def test_precision_profile_and_actual_tiny_command_bind96(factor):
    if factor != 1.:
        with pytest.raises(ValueError):launch.strength_protocol(1.,factor)
        return
    protocol = launch.strength_protocol(1.,factor)
    assert protocol["cg_steps"] == 96 and protocol["true_residual_diagnostics"] is True
    assert protocol == runner.strength_protocol(1.,factor)
    args = launch.parser().parse_args(["--stage","tiny","--exp","precision_tiny","--local-factor",str(factor),
                                      "--review-gate","review.json"])
    command = launch._command(args,"launch.json")
    assert command[command.index("--cg-steps")+1] == "96"


def test_precision_trace_validator_rejects_forged_residual_and_work():
    cfg=runner.SolverConfig(outer_steps=1,cg_steps=96,true_residual_diagnostics=True)
    result=runner.optimize_density(ExplicitPhysics(),prior_kwargs=PRIOR,edge_weight=0.,config=cfg)
    launch.validate_precision_trace(result.trace,result.work,dataclasses.asdict(cfg))
    forged=copy.deepcopy(result.trace)
    forged[0]["true_residual"]["relative"] += .01
    with pytest.raises(ValueError,match="residual"):
        launch.validate_precision_trace(forged,result.work,dataclasses.asdict(cfg))
    work=copy.deepcopy(result.work);work["normal_evaluations"] -= 1
    with pytest.raises(ValueError,match="work"):
        launch.validate_precision_trace(result.trace,work,dataclasses.asdict(cfg))


def test_five_actual96_sweeps_count495_vjp_without_widening_caps():
    physics=ExplicitPhysics()
    physics.weights=torch.logspace(-3,3,27,dtype=torch.float64)
    cfg=runner.SolverConfig(outer_steps=5,cg_steps=96,cg_rtol=1e-30,damping_rel=1e-8,
                            true_residual_diagnostics=True)
    result=runner.optimize_density(physics,prior_kwargs=PRIOR,edge_weight=0.,config=cfg)
    assert result.status == "complete" and result.accepted_steps == 5
    assert all(row["cg_iterations"] == row["cg_matvecs"] == 96 and row["cg_status"] == "maxiter"
               and row["cg_converged"] is False and row["alpha"] == 1. for row in result.trace)
    assert result.work["normal_evaluations"] == 490
    assert result.work["vjp_pairs"] == 495
    assert result.work["render_pairs"] == physics.counters["render_pairs"] == 501
    assert result.work["renderer_calls"] == 1002 and result.work["true_residual_evaluations"] == 5
    launch.validate_precision_trace(result.trace,result.work,dataclasses.asdict(cfg))


def test_true_H_counts_actual_eight_phase_wavelets(monkeypatch):
    import surrogates
    physics=ExplicitPhysics()
    torch.manual_seed(133)
    physics.initial_density=torch.randn(2,216,dtype=torch.complex128)+1
    physics.target=torch.randn_like(physics.initial_density)
    physics.weights=torch.linspace(.5,2,216)
    def render(rho):
        physics.counters["render_pairs"] += 1
        physics.counters["renderer_calls"] += 2
        return rho[0].reshape(6,6,6),rho[1].reshape(6,6,6)
    physics.render=render
    def loss(p,c,zero_target=False):
        value=torch.stack((p,c)).reshape(2,216)
        return ((value-(0 if zero_target else physics.target))*physics.weights).abs().square().sum()
    physics.data_loss=loss
    observed=[]
    actual=surrogates.wavelet_details
    def counted(*args,**kwargs):
        observed.append(1)
        return actual(*args,**kwargs)
    monkeypatch.setattr(surrogates,"wavelet_details",counted)
    result=runner.optimize_density(physics,prior_kwargs=dict(PRIOR,lambda_wavelet=(.12,.1)),edge_weight=0.,
        wavelet_mode="uniform8",config=runner.SolverConfig(outer_steps=2,cg_steps=4,true_residual_diagnostics=True))
    assert result.status == "complete" and result.accepted_steps == 2
    assert result.work["wavelet_transform_batches"] == len(observed) == 176
    assert [row["wavelet_transform_batches"] for row in result.trace] == [88,88]


@pytest.mark.parametrize("stage",["tiny","zero","solve"])
def test_old80_request_is_rejected_at_every_stage_before_launch(supervisor,stage):
    supervisor.args.stage=stage;supervisor.args.cg_steps=80
    with pytest.raises(ValueError,match="cg_steps96"):
        launch.validate_stage_gates(supervisor.args,supervisor.hashes,supervisor.invariants)


def test_formal_request_cannot_choose_another_factor_than_recipe(supervisor,monkeypatch):
    supervisor.args.stage="solve";supervisor.args.local_factor=1.
    supervisor.args.cuda_operator_gate=supervisor.verify/"tiny"
    supervisor.args.zero_audit_gate=supervisor.verify/"zero"
    supervisor.args.selected_recipe=supervisor.verify/"recipe"
    monkeypatch.setattr(launch,"validate_observed_gate",lambda *a,**k:None)
    monkeypatch.setattr(launch,"validate_recipe",lambda *a,**k:{"local_factor_grid":[.5]})
    with pytest.raises(ValueError,match="root-selected"):
        launch.validate_stage_gates(supervisor.args,supervisor.hashes,supervisor.invariants)


def test_actual_callback_path_preserves_trace_and_checks_each_cumulative_work():
    callbacks=[]
    cfg=runner.SolverConfig(outer_steps=2,cg_steps=96,true_residual_diagnostics=True)
    result=runner.optimize_density(ExplicitPhysics(),prior_kwargs=PRIOR,edge_weight=0.,config=cfg,callback=callbacks.append)
    assert callbacks != result.trace  # Real JSONL includes cumulative work.
    launch.validate_precision_callbacks(callbacks,result.trace,result.work,dataclasses.asdict(cfg))
    forged=copy.deepcopy(callbacks)
    forged[0]["work"]["true_residual_elapsed_seconds"] += 1.
    with pytest.raises(ValueError,match="work"):
        launch.validate_precision_callbacks(forged,result.trace,result.work,dataclasses.asdict(cfg))
