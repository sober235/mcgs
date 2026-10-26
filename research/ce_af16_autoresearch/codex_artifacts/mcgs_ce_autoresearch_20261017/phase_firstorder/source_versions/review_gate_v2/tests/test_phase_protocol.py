"""Clinical protocol binds phase roles, real work and non-Adam artifact identity."""
import copy
import functools
import json

import pytest
import torch

import runner
import run_gpu
from test_phase_firstorder import ImagePhysics, SETTINGS


PROTOCOL = {"phase_curvature_mode":"first_order","damping_curvature_mode":"full",
            "update_class":"phase_first_order_quasi_newton","full_MM_majorization":False}


class TinyPhysics(ImagePhysics):
    def __init__(self):
        super().__init__()
        amplitude=torch.linspace(.02,1.,432,dtype=torch.float64).reshape(2,-1)
        angle=torch.linspace(-.1,.2,432,dtype=torch.float64).reshape(2,-1)
        self.initial_density=(amplitude*torch.exp(1j*angle)).to(torch.complex128)
        self.target=self.initial_density*(.85+.03j)
    def render(self,value):
        self.counters["render_pairs"]+=1;self.counters["renderer_calls"]+=2
        return tuple(value.reshape(2,6,6,6))
    def structural_support(self):
        self.render(torch.ones_like(self.initial_density))
        return tuple(torch.ones(6,6,6,dtype=torch.bool) for _ in range(2))


@functools.lru_cache(maxsize=1)
def numerical_tiny_fixture():
    """Actual CPU witness reused only by mocked-supervisor unit fixtures."""
    import gpu_preflight
    return gpu_preflight.check_phase_firstorder(TinyPhysics(),local_factor=1.)


def test_parent_and_child_fixed_single_candidate_protocol():
    for module in (runner,run_gpu):
        actual=module.strength_protocol(1.,1.)
        assert all(actual.get(k) == v for k,v in PROTOCOL.items())
        for value in (.75,.5,True):
            with pytest.raises(ValueError): module.strength_protocol(1.,value)


def test_real_phase_trace_roles_and_tamper_rejection():
    cfg=runner.SolverConfig(outer_steps=2,cg_steps=4,true_residual_diagnostics=True)
    result=runner.optimize_density(ImagePhysics(),prior_kwargs=SETTINGS,config=cfg,
                                   phase_curvature_mode="first_order")
    run_gpu.validate_phase_trace(result.trace,result.work,zero=False)
    for key in ("phase_first_order_normal_evaluations","full_curvature_normal_evaluations"):
        bad=copy.deepcopy(result.work);bad[key]+=1
        with pytest.raises(ValueError):run_gpu.validate_phase_trace(result.trace,bad,zero=False)
    for target in ("full_rayleigh","reduced_cg","reduced_endpoint"):
        trace=copy.deepcopy(result.trace);trace[0]["curvature_evaluations"][target]+=1
        with pytest.raises(ValueError):run_gpu.validate_phase_trace(trace,result.work,zero=False)


def test_checkpoint_discloses_actual_phase_update_and_cannot_be_Adam_resumed(tmp_path):
    cfg=runner.SolverConfig(outer_steps=0,true_residual_diagnostics=True)
    result=runner.optimize_density(ImagePhysics(),prior_kwargs=SETTINGS,config=cfg,
                                   wavelet_mode="uniform8",phase_curvature_mode="first_order")
    path=tmp_path/"density_solver_nonresumable.pth"
    runner.write_solver_checkpoint(path,torch.nn.Linear(2,2),result,
           source_checkpoint="source4000",source_checkpoint_sha256="bound-source",frequency_t=1.)
    artifact=torch.load(path,map_location="cpu",weights_only=True)
    assert artifact["variant"] == "phase_firstorder"
    assert all(artifact.get(k) == v for k,v in PROTOCOL.items())
    assert artifact["resumable_by_original_adam"] is False
    assert not {"iter","optim_state"}&set(artifact)


def test_tiny_phase_path_uses_actual_nonzero_phase_and_reduced_solver():
    import gpu_preflight
    # CPU actual source operator; the same check runs on actual CUDA toy operator.
    operator=TinyPhysics()
    check=gpu_preflight.check_phase_firstorder(operator,local_factor=1.)
    assert check["passed"] is True
    assert check["phase_gradient_norm"] > 0
    assert check["source_phase_gradient_relative_error"] <= 2e-5
    assert check["accepted_outer_steps"] == check["solver_config"]["outer_steps"] == 2
    assert check["solver_config"]["cg_steps"] == 4
    assert all(check.get(k) == v for k,v in PROTOCOL.items())
    run_gpu.validate_phase_trace(check["trace"],check["solver_counters"],zero=False)
    run_gpu.validate_tiny_phase_check(check,device="cpu")
    for key in ("phase_gradient_norm","phase_full_probe_gradient_norm"):
        bad=copy.deepcopy(check);bad[key]=0.
        with pytest.raises(ValueError):run_gpu.validate_tiny_phase_check(bad,device="cpu")
