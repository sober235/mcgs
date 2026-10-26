"""Actual source FFT/SENSE equations, fixed CSM lambda and bounded outcomes."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
C=ROOT.parent/'nonlocal_autoresearch/trainer/paired_gs/data/transforms.py'
spec=importlib.util.spec_from_file_location('_prox_test_source_fft',C)
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)


def implementation():
    path=ROOT/'physics.py'
    assert path.is_file(),'proximal physics is not implemented'
    spec=importlib.util.spec_from_file_location('_prox_physics_tested',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def fixture(dtype):
    generator=torch.Generator().manual_seed(235)
    shape=(4,4,4)
    csm=torch.stack((torch.ones(shape,dtype=dtype),torch.full(shape,2j,dtype=dtype)))
    csm[:,0,0,0]=0
    mask=((torch.arange(4)[:,None]+torch.arange(4)[None,:])%2==0).unsqueeze(0).expand(shape)
    data=torch.randn((2,*shape),generator=generator,dtype=dtype)
    anchor=torch.randn(shape,generator=generator,dtype=dtype)
    return csm,data,mask,anchor


def matrix(csm,mask):
    shape=mask.shape;eye=torch.eye(mask.numel(),dtype=csm.dtype)
    return torch.stack([(source.fft3c(csm*v.reshape(shape))*mask).flatten() for v in eye],1)


@pytest.mark.parametrize('dtype',[torch.complex64,torch.complex128])
def test_source_forward_adjoint_normal_and_fixed_rule(dtype):
    module=implementation();csm,data,mask,x=fixture(dtype)
    operator=module.SenseOperator(csm,data,mask,chunk_size=1)
    a=matrix(csm,mask)
    tolerance=dict(rtol=2e-6,atol=2e-6) if dtype==torch.complex64 else dict(rtol=2e-12,atol=2e-12)
    torch.testing.assert_close(operator.forward(x).flatten(),a@x.flatten(),**tolerance)
    torch.testing.assert_close(operator.adjoint(data).flatten(),a.H@data.flatten(),**tolerance)
    torch.testing.assert_close(operator.normal(x).flatten(),a.H@a@x.flatten(),**tolerance)
    assert operator.L==5. and operator.lambda_requested==operator.lambda_effective==.5
    assert not operator.support[0,0,0] and int(operator.support.sum())==63


@pytest.mark.parametrize('dtype',[torch.complex64,torch.complex128])
def test_actual_J_gradient_directsolve_and_true_endpoint(dtype):
    module=implementation();csm,data,mask,raw=fixture(dtype)
    operator=module.SenseOperator(csm,data,mask,chunk_size=1)
    result=module.proximal_dc(operator,raw)
    assert result.status=='converged'
    anchor=raw*operator.support
    a=matrix(csm,mask).to(torch.complex128)
    lam=operator.lambda_effective
    h=a.H@a+lam*torch.eye(mask.numel(),dtype=a.dtype)
    rhs=a.H@data.to(torch.complex128).flatten()+lam*anchor.to(torch.complex128).flatten()
    direct=torch.linalg.solve(h,rhs)
    tolerance=dict(rtol=3e-5,atol=3e-5) if dtype==torch.complex64 else dict(rtol=3e-9,atol=3e-9)
    torch.testing.assert_close(result.image.to(torch.complex128).flatten(),direct,**tolerance)
    assert torch.count_nonzero(result.image[~operator.support])==0
    record=result.record
    assert record['initial_anchor_nonzero'] is True
    assert record['initial_residual_normal_evaluations']==1
    assert record['n_coils']==2 and record['chunk_size']==1
    assert record['cg_matvecs']==record['cg_iterations']+1
    assert record['true_relative_residual']<=record['true_residual_tolerance']
    assert record['J_after']<=record['J_anchor'] and record['directional_derivative']<0
    point=anchor.detach().clone().requires_grad_(True)
    independent=.5*((source.fft3c(csm*point)*mask-data*mask).abs().square().sum())
    independent=independent+.5*lam*(point-anchor).abs().square().sum()
    gradient,=torch.autograd.grad(independent,point)
    expected_gradient=(a.H@(a@anchor.flatten().to(a.dtype)-data.flatten().to(a.dtype))+lam*(anchor-anchor).flatten().to(a.dtype)).reshape_as(point)
    torch.testing.assert_close(gradient.to(a.dtype),expected_gradient,**tolerance)
    work=result.work
    assert work['rhs_adjoint_calls']==1 and work['normal_calls']==record['cg_matvecs']<=32
    assert work['terminal_normal_calls']==1 and work['objective_calls']==2
    assert work['fft_batches']==2*(record['cg_matvecs']+1+2)
    assert work['ifft_batches']==2*(1+record['cg_matvecs']+1)
    assert work['gaussian_renders']==work['gaussian_vjps']==0


def test_initial_normal_consumes_budget_and_no_retry_is_hidden():
    module=implementation();csm,data,mask,raw=fixture(torch.complex128)
    operator=module.SenseOperator(csm,data,mask,chunk_size=1)
    result=module.proximal_dc(operator,raw,config=module.ProxConfig(max_matvecs=1))
    assert result.status=='budget_exhausted' and result.record['cg_matvecs']==1
    assert result.record['cg_iterations']==0 and result.work['normal_calls']==1
    torch.testing.assert_close(result.image,raw*operator.support,rtol=0,atol=0)


def test_stationarity_is_measured_and_not_invented_as_an_update():
    module=implementation();csm,data,mask,raw=fixture(torch.complex128)
    supported=raw.clone();supported[0,0,0]=0
    measured=source.fft3c(csm*supported)*mask
    operator=module.SenseOperator(csm,measured,mask,chunk_size=1)
    result=module.proximal_dc(operator,raw)
    assert result.status=='stationary' and result.record['cg_iterations']==0
    assert result.record['J_after']<=1e-20
    torch.testing.assert_close(result.image,supported,rtol=0,atol=0)


@pytest.mark.parametrize('bad',['nonbinary','zeroCSM','NaN'])
def test_invalid_physics_refused_before_solve(bad):
    module=implementation();csm,data,mask,_=fixture(torch.complex64)
    if bad=='nonbinary':mask=mask.float();mask[0,0,0]=.5
    elif bad=='zeroCSM':csm.zero_()
    else:data[0,0,0,0]=complex(float('nan'),0)
    with pytest.raises(ValueError):module.SenseOperator(csm,data,mask)
