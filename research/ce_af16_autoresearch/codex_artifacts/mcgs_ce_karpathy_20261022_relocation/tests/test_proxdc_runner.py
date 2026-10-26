"""Actual frozen proximal physics on tiny data; no clinical arrays or GPU."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def module():
    import proxdc_runner
    return proxdc_runner


def fixture(dtype=torch.complex64):
    shape=(2,2,2); g=torch.Generator().manual_seed(26)
    csm=torch.stack((torch.ones(shape,dtype=dtype),torch.full(shape,2j,dtype=dtype)))
    csm[:,0,0,0]=0
    mask=torch.tensor([[[0,1],[1,0]],[[1,0],[0,1]]],dtype=torch.float32)
    data=torch.randn((2,*shape),dtype=dtype,generator=g)*mask
    raw=torch.randn(shape,dtype=dtype,generator=g)
    return csm,data,mask,raw


@pytest.mark.parametrize('dtype',[torch.complex64,torch.complex128])
def test_exact_old_physics_solves_raw_anchor_with_fixed_lambda_and_explicit_work(dtype):
    m=module();ph=m.physics();csm,y,mask,raw=fixture(dtype)
    op=ph.SenseOperator(csm,y,mask,chunk_size=1)
    result=ph.proximal_dc(op,raw,config=ph.ProxConfig(max_matvecs=32))
    n=mask.numel();eye=torch.eye(n,dtype=dtype)
    a=torch.stack([(ph.transforms.fft3c(csm*e.reshape(mask.shape))*mask).flatten() for e in eye],1).to(torch.complex128)
    anchor=(raw*op.support).flatten().to(torch.complex128)
    direct=torch.linalg.solve(a.H@a+.5*torch.eye(n,dtype=torch.complex128),
        a.H@y.flatten().to(torch.complex128)+.5*anchor)
    torch.testing.assert_close(result.image.flatten().to(torch.complex128),direct,rtol=3e-5,atol=3e-5)
    assert op.L==5. and op.lambda_requested==op.lambda_effective==.5
    assert result.status=='converged'
    record=result.record;work=result.work
    assert record['cg_matvecs']==record['cg_iterations']+1<=32
    assert work['normal_calls']==record['cg_matvecs']
    assert work['terminal_normal_calls']==work['rhs_adjoint_calls']==1
    assert work['objective_calls']==2
    assert work['fft_batches']==2*(record['cg_matvecs']+3)
    assert work['ifft_batches']==2*(record['cg_matvecs']+2)
    assert record['J_after']<=record['J_anchor']
    assert record['true_endpoint_passed'] and record['outside_support_exact_zero']


def fake_trainer(*, coils=2):
    m=module();helper=m.helpers();ph=m.physics();csm,y,mask,raw=fixture()
    helper.source_paths()
    if coils != 2:
        extra=torch.zeros((coils-2,*csm.shape[1:]),dtype=csm.dtype)
        csm=torch.cat((csm,extra));y=torch.cat((y,extra.clone()))
    class Trainer:
        def _render(self,channel):
            return raw.clone() if channel=='plain' else raw.clone()*(.8+.1j)
    t=Trainer();t.device=torch.device('cpu')
    for tag in ('p','c'):
        for key,value in dict(csm=csm,kspace=y,mask=mask,csm_fit=csm,kspace_fit=y,mask_fit=mask,kspace_dc=y).items():
            setattr(t,key+'_'+tag,value)
    gauges={scan:dict(factor_scale=3.,scale_trainer=6.,scale_gauge=2.) for scan in ('pre','post')}
    refs={scan:m.topic_array(helper.base_runner().export_trainer_volume(t,channel)[0],gauges[scan])
          for scan,channel in (('pre','plain'),('post','contrast'))}
    return t,gauges,refs


def test_capture_is_same_render_restores_hooks_and_saves_solver_image_without_second_harddc(tmp_path):
    m=module();trainer,gauges,refs=fake_trainer()
    original_attrs=set(vars(trainer));render=trainer._render
    result=m.reconstruct(trainer,refs,gauges,tmp_path,'prox',m.progress_state())
    assert set(vars(trainer))==original_attrs
    assert trainer._render==render
    assert all(row['passed'] for row in result['canonical_parity'].values())
    raw=np.load(tmp_path/'raw_pre_complex.npy');canonical=np.load(tmp_path/'canonical_pre_complex.npy')
    assert not np.array_equal(raw,canonical)
    csm,y,mask,anchor=fixture()
    expected=m.physics().proximal_dc(m.physics().SenseOperator(csm,y,mask),anchor).image
    np.testing.assert_array_equal(np.load(tmp_path/'candidate_pre_complex.npy'),m.topic_array(expected,gauges['pre']))
    assert result['work_total']['gaussian_renders']==2
    assert result['scans']['pre']['work']['gaussian_renders']==0


def test_both_scan_canonical_checks_precede_any_solve(tmp_path,monkeypatch):
    m=module();trainer,gauges,refs=fake_trainer();refs['post']=refs['post']*1.01
    def forbidden(*args,**kwargs):pytest.fail('solve started before both G2 controls passed')
    monkeypatch.setattr(m.physics(),'proximal_dc',forbidden)
    with pytest.raises(ValueError,match='canonical'):
        m.reconstruct(trainer,refs,gauges,tmp_path,'prox',m.progress_state())
    assert not (tmp_path/'candidate_pre_complex.npy').exists()


def test_failed_solver_is_not_exported_or_retried(tmp_path,monkeypatch):
    m=module();trainer,gauges,refs=fake_trainer();ph=m.physics();actual=ph.proximal_dc;calls=[]
    def exhausted(operator,raw,**kwargs):
        calls.append(1)
        return actual(operator,raw,config=ph.ProxConfig(max_matvecs=1))
    monkeypatch.setattr(ph,'proximal_dc',exhausted)
    progress=m.progress_state()
    with pytest.raises(ValueError,match='solve|budget'):
        m.reconstruct(trainer,refs,gauges,tmp_path,'prox',progress)
    assert len(calls)==1
    assert progress['scans']['pre']['record']['status']=='budget_exhausted'
    assert not (tmp_path/'candidate_pre_complex.npy').exists()


@pytest.mark.parametrize('change',[dict(mask_fit_p=torch.ones(2,2,2)),
    dict(kspace_dc_p=torch.ones(2,2,2,2,dtype=torch.complex64))])
def test_acquisition_mismatch_refuses_export(tmp_path,change):
    m=module();trainer,gauges,refs=fake_trainer()
    for key,value in change.items():setattr(trainer,key,value)
    with pytest.raises(ValueError):m.reconstruct(trainer,refs,gauges,tmp_path,'canonical',m.progress_state())


def test_tiny_cpu_uses_the_same_fixed_physics_without_claiming_g2_restore():
    result=module().tiny_stage(torch.device('cpu'))
    assert result['tiny_checks_passed']
    assert all(row['record']['cg_max_matvecs']==32 for row in result['scans'].values())


def test_tiny_failure_retains_completed_and_current_scan_work(monkeypatch):
    m=module();ph=m.physics();actual=ph.proximal_dc;calls=[]
    def fail_second(op,raw,**kwargs):
        calls.append(1)
        if len(calls)==2:
            return actual(op,raw,config=ph.ProxConfig(max_matvecs=1))
        return actual(op,raw,**kwargs)
    monkeypatch.setattr(ph,'proximal_dc',fail_second)
    progress=m.progress_state()
    with pytest.raises(ValueError):m.tiny_stage(torch.device('cpu'),progress=progress)
    assert progress['scans']['pre']['record']['status']=='converged'
    assert progress['scans']['post']['record']['status']=='budget_exhausted'
    assert progress['scans']['post']['work']['normal_calls']==1


def test_failure_work_is_preserved_even_when_endpoint_record_is_nonfinite(tmp_path):
    m=module();progress=dict(scans=dict(pre=dict(work=dict(normal_calls=4),record=dict(J_after=float('nan')))))
    m.write_progress(tmp_path,progress)
    saved=json.loads((tmp_path/'progress.json').read_text())
    assert saved['scans']['pre']['work']['normal_calls']==4
    assert saved['scans']['pre']['record']['J_after']=='nan'
    assert not (tmp_path/'metrics.json').exists()
