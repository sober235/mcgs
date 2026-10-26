"""Actual C coefficient/source checks for the declared TV/wavelet family."""
import ast
import importlib.util
import itertools
from pathlib import Path
import sys
import types

import pytest
import torch

import runner
import run_gpu as launch
from cycle_average import build_cycle_average_prior
from coupled_edge import build_frozen_edge
from gpu_preflight import check_prior_mode
from test_surrogates import shipped
from test_run_gpu import supervisor,own_selection,fake_worker

ROOT=Path(__file__).resolve().parents[1]
PGS=ROOT.parent/"nonlocal_autoresearch/trainer/paired_gs"


def test_explicit_strength_protocol_is_frequency_fixed_and_factor_bound():
    expected={"frequency_t":1.,"local_factor":1.,"wavelet_mode":"uniform8","wavelet_phase_count":8,"fixed_wavelet_shift":None,
              "cg_steps":96,"true_residual_diagnostics":True,**launch.phase_protocol()}
    assert runner.strength_protocol(1.,1.) == launch.strength_protocol(1.,1.) == expected
    with pytest.raises(ValueError): runner.strength_protocol(.5,.75)
    with pytest.raises(ValueError): launch.strength_protocol(1.,3.)


def config_and_options(factor,dtype):
    cfg=runner._load_base().build_config("audit",0,ROOT/"outputs/not_created_cpu_fixture",local_factor=factor,nlambda=0.)
    generator=torch.Generator().manual_seed(235)
    x=torch.randn((2,6,7,8),dtype=dtype,generator=generator)+(1.2+.8j)
    sys.path.insert(0,str(PGS))
    trainer=types.SimpleNamespace(config=cfg,zf_p=x[0],zf_c=x[1])
    options=runner.source_prior_kwargs(trainer,wavelet_mode="uniform8")
    return cfg,x,options


@pytest.mark.parametrize("dtype",[torch.complex64,torch.complex128])
@pytest.mark.parametrize("factor",[1.,.75,.5])
def test_actual_C_config_value_gradient_and_Q_tangency(factor,dtype,shipped):
    cfg,x,options=config_and_options(factor,dtype)
    edge,phase,wave=shipped
    assert options["lambda_tv"] == [.1*factor,.1*factor]
    assert options["lambda_wavelet"] == [.12*factor,.1*factor]
    assert options["lambda_phase"] == .03
    assert cfg["loss"]["lambda_edge"] == .001 and cfg["loss"]["edge_p"] == .9 and cfg["loss"]["edge_kappa"] == 30.
    assert options["kappa"] == [edge.resolve_edge_kappa(x[0],.85),edge.resolve_edge_kappa(x[1],.85)]
    assert options["tv_eps"] == options["phase_eps"] == .04
    prior=build_cycle_average_prior(x,**options)
    point=x.detach().clone().requires_grad_(True)
    actual=prior.merit(point)
    expected=sum(tv*edge.edge_preserving_tv(vol,k,0.,.04,True)+.03*phase.phase_tv(vol,k,0.,.04)
                 for vol,tv,k in zip(point,options["lambda_tv"],options["kappa"]))
    expected += sum(wave.wavelet_l1(list(point),"db4",1,0.,shift,"complex",options["lambda_wavelet"])
                    for shift in itertools.product((0,1),repeat=3))/8.
    tolerance=dict(rtol=5e-6,atol=2e-7) if dtype == torch.complex64 else dict(rtol=2e-11,atol=2e-12)
    torch.testing.assert_close(actual,expected,**tolerance)
    torch.testing.assert_close(prior.quadratic(point)+prior.majorizer_constant,expected,**tolerance)
    grad_actual=torch.autograd.grad(actual,point,retain_graph=True)[0]
    grad_source=torch.autograd.grad(expected,point,retain_graph=True)[0]
    grad_Q=torch.autograd.grad(prior.quadratic(point),point)[0]
    torch.testing.assert_close(grad_actual,grad_source,**tolerance)
    torch.testing.assert_close(grad_Q,grad_source,**tolerance)
    info=prior.info
    assert info["base_tv_phase"]["lambda_phase"] == [.03,.03]
    assert info["base_tv_phase"]["lambda_tv"] == [.1*factor,.1*factor]
    for component in info["wavelet_phases"]:
        assert component["lambda_tv"] == component["lambda_phase"] == [0.,0.]
        assert component["lambda_wavelet"] == [.12*factor/8.,.1*factor/8.]


@pytest.mark.parametrize("factor",[1.,.75,.5])
def test_cross_edge_source_value_gradient_and_kappa_are_unchanged(factor):
    cfg,x,options=config_and_options(factor,torch.complex128)
    path=PGS/"trainers/joint_trainer.py"
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name == "JointTrainer")
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == "_edge_align_loss")
    namespace={"torch":torch}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),"exec"),namespace)
    point=x.detach().clone().requires_grad_(True)
    fixed=build_frozen_edge(point.detach(),p=.9,kappa=30.)
    actual=.001*fixed.merit(point)
    expected=cfg["loss"]["lambda_edge"]*namespace["_edge_align_loss"](None,point[0],point[1],.9,30.)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    left=torch.autograd.grad(actual,point,retain_graph=True)[0]
    right=torch.autograd.grad(expected,point)[0]
    torch.testing.assert_close(left,right,rtol=2e-12,atol=2e-13)


@pytest.mark.parametrize("factor",[1.,.75,.5])
def test_tiny_checks_actual_scaled_wavelet_and_describes_zero_TV_phase(factor):
    _,x,_=config_and_options(factor,torch.complex128)
    result=check_prior_mode(x,wavelet_mode="uniform8",local_factor=factor)
    assert result["passed"] is True and result["local_factor"] == factor
    assert result["lambda_wavelet"] == [.12*factor,.1*factor]
    assert result["lambda_tv"] == result["lambda_phase"] == [0.,0.]
    assert result["tested_terms"] == ["wavelet_mean8"] and result["nonzero_TV_phase_tested"] is False


def test_actual_factor_bound_tiny_gate_cannot_be_reused_at_other_factor(supervisor,monkeypatch):
    supervisor.args.local_factor=1.
    own_selection(supervisor);fake_worker(monkeypatch,supervisor)
    launch.execute(supervisor.args,[])
    path=supervisor.verify/"trial_completed.json"
    launch.validate_observed_gate(path,"tiny",supervisor.hashes,supervisor.invariants,frequency_t=1.,local_factor=1.)
    with pytest.raises(ValueError,match="protocol mismatch|factor1"):
        launch.validate_observed_gate(path,"tiny",supervisor.hashes,supervisor.invariants,frequency_t=1.,local_factor=.5)
    command=launch.read(supervisor.verify/"trial_launch.json")["command"]
    assert command[command.index("--local-factor")+1] == "1.0"
