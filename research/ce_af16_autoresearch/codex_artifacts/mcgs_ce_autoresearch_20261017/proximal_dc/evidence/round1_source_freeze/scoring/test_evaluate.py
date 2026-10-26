"""CPU-only scorer contracts; no clinical outputs, GPU, or observed gates."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import types

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent


def utility():
    path=HERE/'evaluate.py'
    assert path.is_file(),'new proximal scorer is absent'
    spec=importlib.util.spec_from_file_location('_proximal_scoring_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def anchor_fixture(tmp_path,e):
    rows=[dict(name='old_result',method_variant='older',score=-.4,accepted=False),
        dict(name='q96',method_variant='inner_precision_mm',frequency_t=1.,cg_steps=96,true_residual_diagnostics=True,
             score=-.3,accepted=False,strict_success=False),
        *[dict(name=('adamflat_g1_v1' if g==1 else 'adamflat_g2_v1'),method_variant='frequency_adam',frequency_t=1.,
               geometry_factor=g,score=(-.2 if g==1 else -.3),accepted=g==1,strict_success=False) for g in (1.,2.)],
        *[dict(name=f'lower{f}',method_variant='frequency_lowreg_mm',frequency_t=1.,local_factor=f,cg_steps=96,
               true_residual_diagnostics=True,score=-.3,accepted=False,strict_success=False) for f in (.75,.5)],
        dict(name='phase1',method_variant='phase_firstorder',frequency_t=1.,local_factor=1.,cg_steps=96,
             true_residual_diagnostics=True,phase_curvature_mode='first_order',damping_curvature_mode='full',
             update_class='phase_first_order_quasi_newton',full_MM_majorization=False,score=-.3,accepted=False,strict_success=False)]
    previous=dict(status='completed_real_trial_snapshot',trials=rows,
        inner_precision_family=dict(frequency_t=1.,cg_steps=96,true_residual_diagnostics=True,completed_candidates=['q96']),
        frequency_adam_family=dict(frequency_t=1.,declared_geometry_factors=[1.,2.],completed_geometry_factors=[1.,2.]),
        frequency_lowreg_family=dict(frequency_t=1.,declared_local_factors=[.75,.5],completed_local_factors=[.75,.5],
            cg_steps=96,true_residual_diagnostics=True,completed_candidates=['lower0.75','lower0.5']),
        phase_firstorder_family=dict(frequency_t=1.,declared_local_factors=[1.],completed_local_factors=[1.],
            completed_candidates=['phase1'],cg_steps=96,true_residual_diagnostics=True,phase_curvature_mode='first_order',
            damping_curvature_mode='full',update_class='phase_first_order_quasi_newton',full_MM_majorization=False),
        status_counts=dict(completed=len(rows),accepted=1,rejected=len(rows)-1,failed=1),
        setup_failures=[{'name':'old_refusal','measured_cost':13}],failed_trials=[{'name':'old_failure','measured_cost':17}])
    summary=put(tmp_path/'latest_summary.json',previous)
    current=put(tmp_path/'development_incumbent.json',dict(exp='adamflat_g1_v1',score=-.2,strict_success=False,
        validated_summary=str(summary),validated_summary_sha256=digest(summary)))
    e.CURRENT_INCUMBENT=current
    recipe=put(tmp_path/'recipe.json',dict(status='selected',selected_by='root',**e.profile(),minimum_score_improvement=1e-4,
        completed_prior_summary=dict(path=str(summary),sha256=digest(summary)),
        incumbent=dict(path=str(current),sha256=digest(current),score=-.2)))
    return recipe,summary,current


def test_fixed_profile_identifies_H14500_export_only_not_density_or_Adam_training():
    e=utility();p=e.profile()
    assert p==dict(method_variant='mcgs_prox_dc',method_label='MC-GS+proxDC',source_model='adamflat_g1_v1',source_iteration=4500,
        frequency_t=1.,geometry_factor=1.,local_factor=1.,lambda_ratio=.1,cg_max_matvecs=32,cg_rtol=1e-5,
        terminal_normal_evaluations=1,training_updates=0,working_dtype='complex64',
        true_residual_tolerance=2e-5,J_relative_tolerance=2e-6,J_absolute_tolerance=1e-10)


def test_latest_anchor_preserves_completed_history_and_records_zero_new_training(tmp_path):
    e=utility();recipe,_,_=anchor_fixture(tmp_path,e);anchor=e.load_anchor(recipe)
    previous=copy.deepcopy(anchor['previous']);old=copy.deepcopy(previous)
    trial=dict(name='unit_proximal_result',**e.profile(),score=-.19)
    assert e.apply_acceptance(previous,[trial],anchor['score'])==-.19
    assert previous['trials'][:-1]==old['trials'] and trial['accepted'] is True
    assert trial['training_updates']==0 and trial['experiment_class']=='reconstruction_postprocess'
    for key in ('setup_failures','failed_trials','inner_precision_family','frequency_adam_family',
                'frequency_lowreg_family','phase_firstorder_family'):assert previous[key]==old[key]


@pytest.mark.parametrize('attack',['stale_global','copied_global','missing_PF','PF_row','PF_strict','PF_profile',
    'incomplete_Q','Q_strict','H_order','H_strict','L_order','L_strict','current_strict','model_selection',
    'source4000','lambda_sweep','wrong_budget','training','wrong_score','stale_summary'])
def test_stale_or_incomplete_history_or_changed_fixed_recipe_cannot_authorize_scoring(tmp_path,attack):
    e=utility();recipe,summary,current=anchor_fixture(tmp_path,e);r=json.loads(recipe.read_text())
    if attack=='stale_global':current.write_text('{}')
    elif attack=='copied_global':r['incumbent']['path']=str(put(tmp_path/'copied.json',json.loads(current.read_text())))
    elif attack=='stale_summary':summary.write_text('{}')
    elif attack in ('missing_PF','PF_row','PF_strict','PF_profile','incomplete_Q','Q_strict','H_order','H_strict','L_order','L_strict'):
        p=json.loads(summary.read_text())
        if attack=='missing_PF':del p['phase_firstorder_family']
        elif attack=='PF_row':p['trials'].pop();p['status_counts']['completed']-=1
        elif attack=='PF_strict':p['trials'][-1]['strict_success']=True
        elif attack=='PF_profile':p['trials'][-1]['phase_curvature_mode']='full'
        elif attack=='incomplete_Q':p['inner_precision_family']['completed_candidates']=[]
        elif attack=='Q_strict':p['trials'][1]['strict_success']=True
        elif attack=='H_order':p['frequency_adam_family']['completed_geometry_factors']=[2.,1.]
        elif attack=='H_strict':p['trials'][2]['strict_success']=True
        elif attack=='L_order':p['frequency_lowreg_family']['completed_local_factors']=[.5,.75]
        else:p['trials'][-2]['strict_success']=True
        put(summary,p);r['completed_prior_summary']['sha256']=digest(summary)
        c=json.loads(current.read_text());c['validated_summary_sha256']=digest(summary);put(current,c);r['incumbent']['sha256']=digest(current)
    elif attack=='current_strict':
        c=json.loads(current.read_text());c['strict_success']=True;put(current,c);r['incumbent']['sha256']=digest(current)
    elif attack=='model_selection':r['source_model']='some_other_model'
    elif attack=='source4000':r['source_iteration']=4000
    elif attack=='lambda_sweep':r['lambda_ratio']=.2
    elif attack=='wrong_budget':r['cg_max_matvecs']=64
    elif attack=='training':r['training_updates']=500
    else:r['incumbent']['score']=-.1
    put(recipe,r)
    with pytest.raises((ValueError,FileNotFoundError)):e.load_anchor(recipe)


@pytest.mark.parametrize('number',[0,2])
def test_only_one_reconstruction_result_can_be_aggregated(number):
    e=utility();p=dict(trials=[],status_counts=dict(completed=0,accepted=0,rejected=0))
    rows=[dict(name=f'unit{i}',**e.profile(),score=-.1) for i in range(number)]
    with pytest.raises(ValueError):e.apply_acceptance(p,rows,-.2)
    assert not p['trials']


def test_missing_preformal_reference_freeze_is_not_created(tmp_path,monkeypatch):
    e=utility();path=tmp_path/'missing_freeze.json';monkeypatch.setattr(e,'REFERENCE_FREEZE',path)
    with pytest.raises(ValueError):e.existing_reference_freeze({'input_manifest':{}},{})
    assert not path.exists()


def test_canonical_array_check_is_complex_no_gain_fit_and_reference_is_readonly(tmp_path):
    e=utility();reference=(np.arange(24).reshape(2,3,4)+1j).astype(np.complex64)
    old=reference.copy();candidate=reference.copy()
    result=e.canonical_array_check(candidate,reference)
    assert result['complex_relative_l2']==0. and np.array_equal(reference,old)
    with pytest.raises(ValueError):e.canonical_array_check(candidate*np.complex64(1.001),reference)
    with pytest.raises(ValueError):e.canonical_array_check(candidate*np.complex64(1j),reference)


def test_CSM_only_lambda_is_independently_double_accumulated_and_rounded_once(tmp_path):
    import h5py
    e=utility();path=tmp_path/'map.h5'
    values=np.array([.1+.2j,.3+.4j,1.3+.2j],dtype=np.complex64)[:,None,None,None]*np.ones((3,2,2,2),np.complex64)
    values[:,0,0,0]=0.;values[:,1,1,1]*=np.complex64(2.)
    with h5py.File(path,'w') as stream:stream['map']=values
    before=digest(path);actual=e.csm_power_definition(path)
    power=np.sum(np.abs(values.astype(np.complex128))**2,axis=0,dtype=np.float64)
    assert actual['support_count']==7 and actual['shape']==list(values.shape)
    assert actual['L']==pytest.approx(float(power.max()),rel=1e-15)
    assert actual['lambda_requested']==.1*actual['L']
    assert actual['lambda_effective']==float(np.float32(actual['lambda_requested']))
    assert actual['sha256']==digest(path)==before


@pytest.mark.parametrize('defect',['zero','nonfinite','real_dtype','shape'])
def test_invalid_CSM_does_not_get_a_fitted_scale_or_positive_floor(tmp_path,defect):
    import h5py
    e=utility();path=tmp_path/'map.h5';values=np.ones((2,2,2,2),np.complex64)
    if defect=='zero':values*=0.
    elif defect=='nonfinite':values[0,0,0,0]=complex(float('nan'),0.)
    elif defect=='real_dtype':values=values.real.copy()
    else:values=values[0]
    with h5py.File(path,'w') as stream:stream['map']=values
    with pytest.raises(ValueError):e.csm_power_definition(path)


def scan_fixture(e,stationary=False):
    import torch
    physics=e.load(e.ROOT/'physics.py','_proximal_scorer_real_physics_fixture')
    generator=torch.Generator().manual_seed(238)
    csm=torch.randn((3,2,2,2),generator=generator,dtype=torch.complex64)
    csm[:,0,0,0]=0.
    data=torch.randn(csm.shape,generator=generator,dtype=torch.complex64)
    mask=torch.zeros((2,2,2),dtype=torch.bool) if stationary else torch.ones((2,2,2),dtype=torch.bool)
    raw=torch.randn((2,2,2),generator=generator,dtype=torch.complex64)
    operator=physics.SenseOperator(csm,data,mask,chunk_size=2)
    result=physics.proximal_dc(operator,raw)
    power=(csm.real.double().square()+csm.imag.double().square()).sum(0)
    proof=dict(L=float(power.max()),support_count=int((power>0).sum()),shape=list(csm.shape))
    return result.record,result.work,proof


@pytest.mark.parametrize('stationary',[False,True])
def test_actual_physics_records_validate_with_initial_residual_and_stationarity(stationary):
    e=utility();record,work,power=scan_fixture(e,stationary)
    assert e.validate_scan(record,work,power)==work
    assert record['cg_matvecs']==record['cg_iterations']+1
    if stationary:assert record['status']=='stationary' and record['cg_iterations']==0


@pytest.mark.parametrize('defect',['lambda_rule','rounding','dtype','L','support','initial_count','normal_count','terminal_count',
    'hidden_gaussian','fft_count','role_count','cg_stop','true_relative','true_tolerance','J_tolerance','J_arithmetic',
    'J_increase','bad_status','false_converged','gradient','outside_support','training'])
def test_changed_or_incomplete_proximal_record_is_never_a_verified_result(defect):
    e=utility();record,work,power=scan_fixture(e)
    if defect=='lambda_rule':record['lambda_requested']*=2
    elif defect=='rounding':record['lambda_effective']*=2
    elif defect=='dtype':record['dtype']='torch.complex128'
    elif defect=='L':record['L']*=2
    elif defect=='support':record['support_count']+=1
    elif defect=='initial_count':record['initial_residual_normal_evaluations']=0
    elif defect=='normal_count':work['normal_calls']+=1
    elif defect=='terminal_count':work['terminal_normal_calls']=0
    elif defect=='hidden_gaussian':work['gaussian_vjps']=1
    elif defect=='fft_count':work['fft_batches']+=1
    elif defect=='role_count':work['roles']['rhs']['ifft_batches']+=1
    elif defect=='cg_stop':record['cg_residual_history'][-1]=2e-5*record['rhs_norm']
    elif defect=='true_relative':record['true_relative_residual']*=2
    elif defect=='true_tolerance':record['true_residual_tolerance']=.01
    elif defect=='J_tolerance':record['J_check_tolerance']=1.
    elif defect=='J_arithmetic':record['data_after']+=1.
    elif defect=='J_increase':record['J_after']=record['J_anchor']+1.
    elif defect=='bad_status':record['status']='budget_exhausted'
    elif defect=='false_converged':record['cg_converged']=False
    elif defect=='gradient':record['directional_derivative']=1.
    elif defect=='outside_support':record['outside_support_exact_zero']=False
    else:record['training_updates']=1
    with pytest.raises(ValueError):e.validate_scan(record,work,power)


def test_nvgen_is_refused_before_numeric_score_inputs_are_read(tmp_path):
    e=utility();nvgen='/home/congcongliu/anaconda3/envs/nvgen/bin/python'
    result=subprocess.run([nvgen,'-X','utf8','-B',str(HERE/'evaluate.py'),'score','--exp','unit_missing',
        '--selected-recipe',str(tmp_path/'missing.json'),'--output',str(tmp_path/'not_created')],capture_output=True,text=True,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'})
    assert result.returncode!=0 and 'GSTMI' in result.stderr
    assert not (tmp_path/'not_created').exists()


@pytest.mark.parametrize('defect',[None,'head','changed_input'])
def test_calibration_emits_exact_parent_proof_only_after_rendered_arrays_are_frozen(tmp_path,monkeypatch,defect):
    e=utility();m=e.load(e.ROOT/'run_gpu.py','_proximal_calibration_wire_parent')
    monkeypatch.setattr(e,'ROOT',tmp_path);monkeypatch.setattr(e,'S',tmp_path/'scoring')
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'VERIFY',tmp_path/'verification')
    row=dict(head_psnr=30.,head_ssim=.8,head_nmse=.1,head_hfen=.2)
    published={};model_files={};artifacts={}
    for scan in ('pre','post'):
        score=put(tmp_path/f'published_{scan}.json',dict(rows=[row]));array=tmp_path/f'canonical_{scan}.npy';array.write_bytes(b'frozen synthetic array descriptor')
        published[scan]=dict(score_reference=m.descriptor(score),canonical_array=m.descriptor(array),head_psnr=30.)
        model_files[str(score)]=digest(score);model_files[str(array)]=digest(array);artifacts[str(array)]=digest(array)
    raw=put(m.VERIFY/'unit_rendered.json',dict(control_bindings={}))
    I=dict(input_manifest={},accepted_h1=dict(definition_manifest={'path':'fixed','sha256':'definition'},
        files=model_files,definition=dict(scans=published)))
    source=tmp_path/'reviewed_source.py';source.write_text('# synthetic source binding\n');hashes={str(source):digest(source)}
    monkeypatch.setattr(m,'snapshot_invariants',lambda:I);monkeypatch.setattr(m,'control_bindings',lambda a:{})
    monkeypatch.setattr(e,'context',lambda:(m,hashes,I));monkeypatch.setattr(e,'scoring_implementations',lambda:{})
    monkeypatch.setattr(e.report,'load_references',lambda *a:{})
    monkeypatch.setattr(e,'existing_reference_freeze',lambda *a:{})
    proof=dict(complex_relative_l2=0.,complex_relative_l2_tolerance=1e-5,amplitude_or_phase_fit=False)
    evidence=dict(bound_artifacts_sha256={str(raw):digest(raw),**artifacts},scans={s:dict(canonical_check=proof) for s in ('pre','post')})
    monkeypatch.setattr(e,'validate_actual_run',lambda *a,**k:(tmp_path/'outputs/unit',{},evidence,artifacts,types.SimpleNamespace()))
    def heads(paths):
        assert raw.is_file() and not (m.VERIFY/'unit_completed.json').exists()
        if defect=='changed_input':Path(published['pre']['canonical_array']['path']).write_bytes(b'changed after freeze')
        return {s:{**row,'head_psnr':30.002 if defect=='head' and s=='post' else 30.} for s in ('pre','post')}
    monkeypatch.setattr(e,'head_scores',heads);output=tmp_path/'scoring/calibration_unit'
    if defect:
        with pytest.raises(ValueError):e.calibrate('unit',output)
        assert not (output/'calibration.json').exists()
    else:
        e.calibrate('unit',output)
        value=m.validate_calibration(m.descriptor(output/'calibration.json'),'unit',hashes,I,artifacts)
        assert value['rendered_proof']==m.descriptor(raw) and value['scans']['pre']['head_metrics']['psnr']==30.


def test_calibration_never_reads_GT_when_actual_rendered_validation_fails(tmp_path,monkeypatch):
    e=utility();monkeypatch.setattr(e,'scoring_implementations',lambda:{})
    monkeypatch.setattr(e,'context',lambda:(None,None,None));called=[]
    def refuse(*args,**kwargs):raise ValueError('rendered arrays are missing or changed')
    monkeypatch.setattr(e,'validate_actual_run',refuse);monkeypatch.setattr(e,'head_scores',lambda *a:called.append('GT'))
    with pytest.raises(ValueError,match='rendered arrays'):e.calibrate('missing',tmp_path/'never_created')
    assert not called and not (tmp_path/'never_created').exists()
