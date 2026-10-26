"""Source-acquired mask and exact-weight evidence for the isolated FM operator."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

from density_operator import DensityLinearOperator
from frequency_weights import flatten_acquired_rms
from test_density_operator import TinyTrainer, assert_nested_equal


ROOT = Path(__file__).resolve().parents[1]


def frequency_trainer(dtype=torch.float64):
    trainer = TinyTrainer(dtype)
    def build(self, high_override=None):
        high = 2.8 if high_override is None else high_override
        return torch.tensor([.3, .3, 1., 1., 1., high, high, high],
                            dtype=torch.float32).reshape(self.volume_shape)
    trainer._build_freq_weight = types.MethodType(build, trainer)
    return trainer


def test_frequency_t0_control_uses_original_weights_exactly():
    trainer = frequency_trainer()
    op = DensityLinearOperator(trainer, frequency_t=0.)
    for source, effective in zip(op._source_frequency_weights, op._frequency_weights):
        assert effective is source
        assert torch.equal(source, effective)
    assert op.frequency_summary()['frequency_t'] == 0.


@pytest.mark.parametrize('t', [.5, 1.])
def test_both_scans_flatten_on_original_acquired_mask_not_known_zero_fit_mask(t):
    trainer = frequency_trainer()
    before = copy.deepcopy(trainer.optimizer.state_dict())
    old_model = {n:p.detach().clone() for n,p in trainer.model.named_parameters()}
    op = DensityLinearOperator(trainer,frequency_t=t)
    summary = op.frequency_summary()
    for index, (scan,tag) in enumerate((('pre','p'),('post','c'))):
        source = op._source_frequency_weights[index]
        expected = flatten_acquired_rms(source,getattr(trainer,'mask_'+tag),t)
        wrong = flatten_acquired_rms(source,getattr(trainer,'mask_fit_'+tag),t)
        assert torch.equal(op._frequency_weights[index],expected)
        assert not torch.allclose(expected,wrong)
        row=summary['scans'][scan]
        assert row['acquired_count'] == (3 if tag=='p' else 4)
        assert row['relative_mass_error'] <= 2e-6
    assert_nested_equal(trainer.optimizer.state_dict(),before)
    for n,p in trainer.model.named_parameters():assert torch.equal(p,old_model[n])


def test_t0_loss_gradient_normal_and_CG_trajectory_equal_reviewed_D_v3():
    from pcg import solve_pcg
    source = ROOT.parent/'density_mm/density_operator.py'
    # P is a fresh exact FM clone; retain the original D-control provenance
    # from the immutable FM starting manifest rather than inventing a D hash.
    manifest=json.loads((ROOT.parent/'frequency_mm/evidence/starting_copy_manifest.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['source_sha256']['density_operator.py']
    spec=importlib.util.spec_from_file_location('_frequency_D_control',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    trainer=frequency_trainer();old=module.DensityLinearOperator(trainer);new=DensityLinearOperator(trainer,frequency_t=0.)
    def loss_gradient(op):
        x=op.initial_density.detach().clone().requires_grad_(True)
        loss=op.data_loss(*op.render(x));gradient,=torch.autograd.grad(loss,x)
        return loss.detach(),gradient.detach()
    assert all(torch.equal(a,b) for a,b in zip(loss_gradient(old),loss_gradient(new)))
    def normal(op,x):
        x=x.detach().clone().requires_grad_(True)
        loss=.5*op.data_loss(*op.render(x),zero_target=True)
        out,=torch.autograd.grad(loss,x)
        return out.detach()+.01*x.detach()
    assert torch.equal(normal(old,old.initial_density),normal(new,new.initial_density))
    rhs=-loss_gradient(old)[1]/2
    a=solve_pcg(lambda x:normal(old,x),rhs,maxiter=6,rtol=1e-12)
    b=solve_pcg(lambda x:normal(new,x),rhs,maxiter=6,rtol=1e-12)
    assert torch.equal(a.x,b.x) and a.residual_history==b.residual_history
    assert (a.iterations,a.matvecs,a.status)==(b.iterations,b.matvecs,b.status)
    # Also exercise accepted MM directions/backtracking using the reviewed D
    # solve body, rather than treating CG parity alone as full solve parity.
    from runner import SolverConfig,optimize_density
    spec=importlib.util.spec_from_file_location('_frequency_D_solver_control',ROOT.parent/'density_mm/runner.py')
    source_solver=importlib.util.module_from_spec(spec);sys.modules[spec.name]=source_solver;spec.loader.exec_module(source_solver)
    prior=dict(lambda_tv=0.,lambda_phase=0.,lambda_wavelet=0.,kappa=1.)
    config=dict(outer_steps=2,cg_steps=6,damping_rel=.01)
    previous=source_solver.optimize_density(old,prior_kwargs=prior,edge_weight=.001,
        config=source_solver.SolverConfig(**config),wavelet_mode='uniform8')
    current=optimize_density(new,prior_kwargs=prior,edge_weight=.001,
        config=SolverConfig(**config),wavelet_mode='uniform8')
    assert torch.equal(previous.density,current.density)
    assert previous.trace==current.trace and previous.work==current.work
    assert (previous.accepted_steps,previous.status)==(current.accepted_steps,current.status)


@pytest.mark.parametrize('t', [0., .5, 1.])
def test_weighted_merit_and_zero_normal_use_same_explicit_squared_weights(t):
    trainer=frequency_trainer();op=DensityLinearOperator(trainer,frequency_t=t)
    x=op.initial_density.detach().clone().requires_grad_(True);images=op.render(x)
    direct=0.;zero=0.
    for tag,image,weight in zip(('p','c'),images,op._frequency_weights):
        args=(getattr(trainer,'csm_fit_'+tag),getattr(trainer,'kspace_fit_'+tag),getattr(trainer,'mask_fit_'+tag),weight)
        direct=direct+trainer._kspace_loss(image,*args,norm=getattr(trainer,'_dc_norm_'+tag))
        zero=zero+trainer._kspace_loss(image,args[0],args[1].new_zeros(()),args[2],weight,norm=getattr(trainer,'_dc_norm_'+tag))
    assert torch.equal(op.data_loss(*images),direct)
    assert torch.equal(op.data_loss(*images,zero_target=True),zero)
    H=[]
    for index in range(x.numel()):
        basis=torch.zeros_like(x).flatten();basis[index]=1;basis=basis.reshape_as(x).requires_grad_(True)
        loss=.5*op.data_loss(*op.render(basis),zero_target=True)
        product,=torch.autograd.grad(loss,basis);H.append(product.detach().flatten())
    matrix=torch.stack(H,dim=1)
    torch.testing.assert_close(matrix,matrix.mH,rtol=1e-12,atol=1e-12)
    assert float(torch.linalg.eigvalsh(matrix).min()) > 0
    actual,=torch.autograd.grad(.5*op.data_loss(*images,zero_target=True),x)
    torch.testing.assert_close(actual.flatten(),matrix@x.flatten(),rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('t',[0.,.5,1.])
def test_evidence_arrays_independently_recompute_hashes_and_mass_without_overwrite(tmp_path,t):
    op=DensityLinearOperator(frequency_trainer(),frequency_t=t)
    value=op.save_frequency_evidence(tmp_path)
    assert value==json.loads((tmp_path/'frequency_weights/manifest.json').read_text())
    assert value['mass_relative_tolerance']==2e-6
    for scan,row in value['scans'].items():
        arrays={}
        for name in ('source_weight','effective_weight','acquired_mask'):
            info=row[name];path=Path(info['path'])
            assert path.is_relative_to(tmp_path/'frequency_weights')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==info['file_sha256']
            array=np.load(path,allow_pickle=False);arrays[name]=array
            assert hashlib.sha256(array.tobytes()).hexdigest()==info['byte_sha256']
            assert list(array.shape)==info['shape']
        mask=arrays['acquired_mask'].astype(bool)
        mass=[np.square(arrays[name][mask].astype(np.float64)).mean() for name in ('source_weight','effective_weight')]
        assert row['source_mean_squared_weight']==mass[0]
        assert row['effective_mean_squared_weight']==mass[1]
        assert abs(mass[1]/mass[0]-1)<=2e-6
        if t==0:assert np.array_equal(arrays['source_weight'],arrays['effective_weight'])
    with pytest.raises(FileExistsError):op.save_frequency_evidence(tmp_path)


@pytest.mark.parametrize('defect',['missing_mask','fitmask_only','empty_acquired','nonbinary','none_weight','negative_weight','bad_t'])
def test_invalid_or_missing_original_frequency_inputs_fail_without_fallback(defect):
    trainer=frequency_trainer();t=.5
    if defect in ('missing_mask','fitmask_only'):del trainer.mask_p
    elif defect=='empty_acquired':trainer.mask_c.zero_()
    elif defect=='nonbinary':trainer.mask_p[0,0,0]=.5
    elif defect=='none_weight':trainer._build_freq_weight=lambda *args,**kwargs:None
    elif defect=='negative_weight':trainer._build_freq_weight=lambda *args,**kwargs:-torch.ones(trainer.volume_shape)
    elif defect=='bad_t':t=.25
    with pytest.raises((ValueError,TypeError)):DensityLinearOperator(trainer,frequency_t=t)


def test_tiny_source_t0_parity_helper_uses_original_loss_normal_and_exact_weights():
    from gpu_preflight import check_t0_source_parity
    result=check_t0_source_parity(DensityLinearOperator(frequency_trainer(),frequency_t=.5))
    assert result['passed'] is True and result['weights_exact_equal'] is True
    assert result['source_loss_absolute_difference']==0.
    assert result['zero_loss_absolute_difference']==0.
    assert result['normal_relative_difference']==0.


def test_actual_source_builder_3D_shape_and_all_radial_bands_with_original_masks():
    trainer=TinyTrainer(torch.float32);trainer.volume_shape=(9,9,9)
    # Keep the verbatim C _build_freq_weight method installed by TinyTrainer.
    for tag in ('p','c'):
        acquired=torch.zeros(trainer.volume_shape)
        acquired[4,4,4]=1.;acquired[4,4,5]=1.;acquired[0,0,0]=1.
        setattr(trainer,'mask_'+tag,acquired)
        setattr(trainer,'mask_fit_'+tag,torch.ones(trainer.volume_shape))
        setattr(trainer,'csm_fit_'+tag,torch.ones(2,*trainer.volume_shape,dtype=torch.complex64))
        setattr(trainer,'kspace_fit_'+tag,torch.zeros(2,*trainer.volume_shape,dtype=torch.complex64))
        setattr(trainer,'_dc_norm_'+tag,3.)
    op=DensityLinearOperator(trainer,frequency_t=.5)
    for index,high in enumerate((2.2,2.8)):
        source=op._source_frequency_weights[index]
        assert tuple(source.shape)==(9,9,9)
        torch.testing.assert_close(source[4,4,4],torch.tensor(.3),rtol=0.,atol=0.)
        assert source[4,4,5]==1. and source[0,0,0]==torch.tensor(high)
        expected=flatten_acquired_rms(source,op._acquired_masks[index],.5)
        assert torch.equal(op._frequency_weights[index],expected)
        assert op.frequency_summary()['scans']['pre' if index==0 else 'post']['acquired_count']==3


@pytest.mark.parametrize('t',[0.,.5,1.])
def test_actual_tiny_fixture_has_all_source_and_acquired_bands_and_fit_only_samples(t,tmp_path):
    from gpu_preflight import TINY_SHAPE,tiny_acquisition_masks,tiny_frequency_bands
    trainer=TinyTrainer(torch.float32);trainer.volume_shape=TINY_SHAPE
    generator=torch.Generator(device='cpu').manual_seed(1729)
    for tag in ('p','c'):
        acquired,fit=tiny_acquisition_masks(TINY_SHAPE,device=torch.device('cpu'),generator=generator)
        assert not torch.equal(acquired,fit) and bool(((fit==1)&(acquired==0)).any())
        setattr(trainer,'mask_'+tag,acquired);setattr(trainer,'mask_fit_'+tag,fit)
        setattr(trainer,'csm_fit_'+tag,torch.ones(2,*TINY_SHAPE,dtype=torch.complex64))
        setattr(trainer,'kspace_fit_'+tag,torch.zeros(2,*TINY_SHAPE,dtype=torch.complex64))
        setattr(trainer,'_dc_norm_'+tag,float(acquired.sum()))
    op=DensityLinearOperator(trainer,frequency_t=t)
    for source,effective,mask in zip(op._source_frequency_weights,op._frequency_weights,op._acquired_masks):
        assert len(torch.unique(source))==len(torch.unique(source[mask.bool()]))==3
        assert torch.equal(source,effective) is (t==0.)
    assert max(row['relative_mass_error'] for row in op.frequency_summary()['scans'].values())<=2e-6
    bands=tiny_frequency_bands(op)
    for scan in ('pre','post'):
        assert len(bands[scan]['bands'])==3 and bands[scan]['fit_only_count']>0
        assert all(row['acquired_count']>0 for row in bands[scan]['bands'])
        assert bands[scan]['effective_identity'] is (t==0.)
    from runner import validate_saved_frequency_evidence
    proof=validate_saved_frequency_evidence(tmp_path,op.save_frequency_evidence(tmp_path),t,op)
    assert proof['passed'] is True and proof['frequency_t']==t
    for scan in ('pre','post'):
        assert len(proof['scans'][scan]['source_levels'])==len(proof['scans'][scan]['acquired_source_levels'])==3
        assert proof['scans'][scan]['effective_identity'] is (t==0.)
