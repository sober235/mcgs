"""Focused CPU-only independent-array and anchor/aggregation contracts."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import os
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parent
PHASE = dict(phase_curvature_mode='first_order',damping_curvature_mode='full',
             update_class='phase_first_order_quasi_newton',full_MM_majorization=False)


def utility():
    path=ROOT/'evaluate.py'
    if not path.is_file():pytest.fail('Independent FM scorer implementation is absent')
    spec=importlib.util.spec_from_file_location('fm_scoring_test_evaluator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_arrays(tmp_path,t=.5):
    folder=tmp_path/'frequency_weights';folder.mkdir()
    value={'version':1,'frequency_t':t,'mass_relative_tolerance':2e-6,'scans':{}}
    sources,masks={},{}
    for scan,high in (('pre',2.2),('post',2.8)):
        source=np.asarray([.3,.3,1,1,1,high,high,high],dtype=np.float32).reshape(2,2,2)
        mask=np.asarray([1,0,1,1,0,1,0,1],dtype=np.float32).reshape(source.shape)
        selected=mask.astype(bool);c=np.sqrt(np.mean(source[selected].astype(np.float64)**2))
        mixed=(1-t)*source.astype(np.float64)+t*c
        effective=(mixed*c/np.sqrt(np.mean(mixed[selected]**2))).astype(np.float32) if t else source.copy()
        sources[scan]=source.copy();masks[scan]=mask.copy()
        row={'acquired_count':int(selected.sum()),'c':float(c),'source_mean_squared_weight':float(np.mean(source[selected].astype(np.float64)**2)),
             'effective_mean_squared_weight':float(np.mean(effective[selected].astype(np.float64)**2))}
        row['relative_mass_error']=abs(row['effective_mean_squared_weight']/row['source_mean_squared_weight']-1)
        row['t0_exact_equal']=True if t==0 else None
        for name,array in (('source_weight',source),('effective_weight',effective),('acquired_mask',mask)):
            path=folder/f'{scan}_{name}.npy';np.save(path,array,allow_pickle=False)
            row[name]={'path':str(path),'shape':list(array.shape),'dtype':'torch.float32','device':'cpu',
                       'file_sha256':digest(path),'byte_sha256':hashlib.sha256(array.tobytes()).hexdigest()}
        value['scans'][scan]=row
    put(folder/'manifest.json',value)
    return value,sources,masks


def test_independent_frequency_array_validator_recomputes_both_scan_formulas(tmp_path):
    e=utility();value,sources,masks=fixture_arrays(tmp_path)
    result=e.validate_frequency_arrays(value,tmp_path,.5,sources,masks)
    assert result['passed'] is True
    assert max(row['relative_mass_error'] for row in result['scans'].values())<=2e-6


@pytest.mark.parametrize('attack',['unflattened','fit_mask','raw_hash','source','frequency_t','mass_stat'])
def test_consistent_metadata_cannot_hide_wrong_formula_or_original_mask(tmp_path,attack):
    e=utility();value,sources,masks=fixture_arrays(tmp_path)
    row=value['scans']['post']
    if attack in ('unflattened','fit_mask','source'):
        name={'unflattened':'effective_weight','fit_mask':'acquired_mask','source':'source_weight'}[attack]
        path=Path(row[name]['path']);array=np.load(path)
        if attack=='unflattened':array=sources['post'].copy()
        elif attack=='fit_mask':array=np.ones_like(array)
        else:array*=1.1
        np.save(path,array,allow_pickle=False)
        row[name].update(file_sha256=digest(path),byte_sha256=hashlib.sha256(array.tobytes()).hexdigest())
        if attack=='unflattened':row['effective_mean_squared_weight']=row['source_mean_squared_weight'];row['relative_mass_error']=0.
    elif attack=='raw_hash':row['source_weight']['byte_sha256']='0'*64
    elif attack=='frequency_t':value['frequency_t']=1.
    else:row['c']*=1.1
    put(tmp_path/'frequency_weights/manifest.json',value)
    with pytest.raises(ValueError):e.validate_frequency_arrays(value,tmp_path,.5,sources,masks)


def anchor_fixture(tmp_path,e):
    rows=[{'name':'old_best','score':-.2,'accepted':True},
        *[dict(name=f'geo{f}',method_variant='geometry_lr_restart',geometry_factor=float(f),score=-.3,accepted=False) for f in (2,4,8)],
        *[dict(name=f'fm{t}',method_variant='frequency_mm',frequency_t=t,score=-.3,accepted=False) for t in (.5,1.)],
        *[dict(name=f'prior_strength{f}',method_variant='frequency_mm',frequency_t=1.,local_factor=float(f),score=-.3,accepted=False) for f in (2,4)],
        dict(name='precision96',method_variant='inner_precision_mm',frequency_t=1.,local_factor=1.,cg_steps=96,
             true_residual_diagnostics=True,strict_success=False,score=-.3,accepted=False),
        *[dict(name=f'flat_Adam{g}',method_variant='frequency_adam',frequency_t=1.,geometry_factor=float(g),
               strict_success=False,score=-.3,accepted=False) for g in (1,2)],
        *[dict(name=f'lowreg{f}',method_variant='frequency_lowreg_mm',frequency_t=1.,local_factor=f,cg_steps=96,
               true_residual_diagnostics=True,strict_success=False,score=-.3,accepted=False) for f in (.75,.5)]]
    prior=dict(status='completed_real_trial_snapshot',trials=rows,
        geometry_family={'completed_factors':[2.,4.,8.]},frequency_family={'completed_t':[.5,1.]},
        tvwave_family={'completed_local_factors':[2.,4.]},
        inner_precision_family={'frequency_t':1.,'cg_steps':96,'true_residual_diagnostics':True,'completed_candidates':['precision96']},
        frequency_adam_family={'frequency_t':1.,'declared_geometry_factors':[1.,2.],'completed_geometry_factors':[1.,2.]},
        frequency_lowreg_family={'frequency_t':1.,'declared_local_factors':[.75,.5],'completed_local_factors':[.75,.5],
            'cg_steps':96,'true_residual_diagnostics':True,'completed_candidates':[f'lowreg{f}' for f in (.75,.5)]},
        status_counts={'completed':len(rows),'accepted':1,'rejected':len(rows)-1},
        setup_failures=[{'name':'old_setup_failure','cost':17}],failed_trials=[{'name':'old_trial_failure','cost':23}])
    summary=put(tmp_path/'summary.json',prior)
    incumbent=put(tmp_path/'incumbent.json',dict(score=-.2,strict_success=False,validated_summary=str(summary),validated_summary_sha256=digest(summary)))
    e.CURRENT_INCUMBENT=incumbent
    recipe=put(tmp_path/'recipe.json',dict(status='selected',selected_by='root',frequency_t=1.,local_factor_grid=[1.],**PHASE,
        cg_steps=96,true_residual_diagnostics=True,minimum_score_improvement=1e-4,
        source_iteration=4000,outer_steps=5,max_vjp_pairs=500,max_render_pairs=700,max_merit_forwards=100,
        nlambda=0.,wavelet_mode='uniform8',damping_rel=.001,cg_rtol=.001,
        completed_prior_summary={'path':str(summary),'sha256':digest(summary)},
        incumbent={'path':str(incumbent),'sha256':digest(incumbent),'score':-.2}))
    return recipe,summary,incumbent


def test_anchor_uses_latest_completed_Q_H_L_and_preserves_every_prior_row(tmp_path):
    e=utility();recipe,_,_=anchor_fixture(tmp_path,e);anchor=e.load_anchor(recipe)
    previous=copy.deepcopy(anchor['previous']);old=copy.deepcopy(previous)
    trials=[dict(name='phase_firstorder',frequency_t=1.,local_factor=1.,cg_steps=96,true_residual_diagnostics=True,
        wavelet_mode='uniform8',wavelet_phase_count=8,fixed_wavelet_shift=None,score=-.19,**PHASE)]
    final=e.apply_acceptance(previous,trials,anchor['score'])
    assert final==-.19 and [t['accepted'] for t in trials]==[True]
    assert previous['trials'][:-1]==old['trials']
    assert previous['setup_failures']==old['setup_failures'] and previous['failed_trials']==old['failed_trials']


@pytest.mark.parametrize('attack',['missing_anchor','changed_anchor','incomplete_H','H_order','missing_H_row',
    'H_frequency','H_strict','Q_missing','Q_cap','Q_strict','L_missing','L_order','L_row','L_strict','L_profile',
    'current_strict','wrong_score','wrong_grid','single_factor','old_H_anchor','phase_mode','majorization','stale_copy','bool_factor',
    'missing_budget','changed_budget'])
def test_missing_stale_unresolved_or_strict_win_anchors_are_rejected(tmp_path,attack):
    e=utility();recipe,summary,current=anchor_fixture(tmp_path,e);value=json.loads(recipe.read_text())
    if attack=='missing_anchor':summary.unlink()
    elif attack=='changed_anchor':summary.write_text('{}')
    elif attack in ('incomplete_H','H_order','missing_H_row','H_frequency','H_strict','Q_missing','Q_cap','Q_strict',
                     'L_missing','L_order','L_row','L_strict','L_profile','old_H_anchor'):
        prior=json.loads(summary.read_text())
        if attack=='incomplete_H':prior['frequency_adam_family']['completed_geometry_factors']=[1.]
        elif attack=='H_order':prior['frequency_adam_family']['declared_geometry_factors']=[2.,1.]
        elif attack=='missing_H_row':prior['trials']=[t for t in prior['trials'] if t['name']!='flat_Adam2'];prior['status_counts']['completed']-=1
        elif attack=='H_frequency':next(t for t in prior['trials'] if t.get('method_variant')=='frequency_adam')['frequency_t']=0.
        elif attack=='H_strict':next(t for t in prior['trials'] if t.get('method_variant')=='frequency_adam')['strict_success']=True
        elif attack=='Q_missing':prior['inner_precision_family']['completed_candidates']=[]
        elif attack=='Q_cap':prior['inner_precision_family']['cg_steps']=80
        elif attack=='Q_strict':next(t for t in prior['trials'] if t.get('method_variant')=='inner_precision_mm')['strict_success']=True
        elif attack=='L_missing':del prior['frequency_lowreg_family']
        elif attack=='L_order':prior['frequency_lowreg_family']['completed_local_factors']=[.5,.75]
        elif attack=='L_row':prior['trials'].pop();prior['status_counts']['completed']-=1
        elif attack=='L_strict':prior['trials'][-1]['strict_success']=True
        elif attack=='L_profile':prior['trials'][-1]['cg_steps']=80
        else:
            del prior['frequency_lowreg_family'];prior['trials']=prior['trials'][:-2];prior['status_counts']['completed']-=2
        put(summary,prior);value['completed_prior_summary']['sha256']=digest(summary)
        incumbent=json.loads(current.read_text());incumbent['validated_summary_sha256']=digest(summary)
        put(current,incumbent);value['incumbent']['sha256']=digest(current)
    elif attack=='current_strict':
        incumbent=json.loads(current.read_text());incumbent['strict_success']=True;put(current,incumbent)
        value['incumbent']['sha256']=digest(current)
    elif attack=='wrong_score':value['incumbent']['score']=-.1
    elif attack=='wrong_grid':value['local_factor_grid']=[.5,.75]
    elif attack=='phase_mode':value['phase_curvature_mode']='full'
    elif attack=='majorization':value['full_MM_majorization']=True
    elif attack=='stale_copy':
        stale=put(tmp_path/'copied_incumbent.json',json.loads(current.read_text()))
        value['incumbent']['path']=str(stale);value['incumbent']['sha256']=digest(stale)
    elif attack=='bool_factor':value['local_factor_grid']=[True]
    elif attack=='missing_budget':del value['max_vjp_pairs']
    elif attack=='changed_budget':value['max_render_pairs']=701
    else:value['local_factor']=.75
    put(recipe,value)
    with pytest.raises((ValueError,FileNotFoundError)):e.load_anchor(recipe)
    if attack=='missing_anchor':assert not summary.exists()


def test_missing_reference_freeze_is_never_created(tmp_path,monkeypatch):
    e=utility();path=tmp_path/'missing_freeze.json';monkeypatch.setattr(e,'REFERENCE_FREEZE',path)
    monkeypatch.setattr(e.report,'report_reference_freeze',lambda *a,**k:pytest.fail('create-if-missing helper must never run'))
    with pytest.raises(ValueError):e.existing_reference_freeze({}, {})
    assert not path.exists()


def test_original_preformal_freeze_and_reviewed_D_source_are_readonly_fixed():
    e=utility()
    assert e.sha(e.REFERENCE_FREEZE)==e.REFERENCE_FREEZE_SHA
    assert e.read(e.REFERENCE_FREEZE)['created_before_formal_trials'] is True
    assert e.sha(e.D_SCORER)==e.D_SCORER_SHA


def tiny_prior(factor):
    return dict(passed=True,local_factor=factor,lambda_tv=[0.,0.],lambda_phase=[0.,0.],
        lambda_wavelet=[.12*factor,.1*factor],tested_terms=['wavelet_mean8'],nonzero_TV_phase_tested=False,
        wavelet_mode='uniform8',wavelet_phase_count=8,fixed_wavelet_shift=None,tolerance=2e-5,
        errors=dict(value_relative=1e-7,gradient_relative=1e-7,quadratic_gradient_relative=1e-7),
        wavelet_transform_batches=40,reference_wavelet_transform_batches=8,device='cuda:0')


@pytest.mark.parametrize('factor',[1.,.75,.5])
def test_tiny_evidence_explicitly_binds_scaled_wavelet_only_coverage(factor):
    e=utility();prior=tiny_prior(factor)
    assert e.validate_tiny_prior(prior,factor)['nonzero_TV_phase_tested'] is False


@pytest.mark.parametrize('defect',['factor','unscaled','tv','phase','overclaim','missing_scope','tolerance','error','work'])
def test_tiny_cannot_claim_nonzero_TV_phase_or_hide_unscaled_wavelet(defect):
    e=utility();prior=tiny_prior(.75)
    if defect=='factor':prior['local_factor']=.5
    elif defect=='unscaled':prior['lambda_wavelet']=[.12,.1]
    elif defect=='tv':prior['lambda_tv']=[.2,.2]
    elif defect=='phase':prior['lambda_phase']=[.03,.03]
    elif defect=='overclaim':prior['nonzero_TV_phase_tested']=True
    elif defect=='missing_scope':del prior['tested_terms']
    elif defect=='tolerance':prior['tolerance']=3e-5
    elif defect=='error':prior['errors']['gradient_relative']=3e-5
    else:prior['wavelet_transform_batches']=5
    with pytest.raises(ValueError):e.validate_tiny_prior(prior,.75)


def fm_protocol(e, factor=1.):
    common=e.load(e.D_SCORER,'_tvwave_fixture_readonly_D');namespace={'ev':common,'deepcopy':copy.deepcopy}
    import ast
    tree=ast.parse((e.D_SCORER.parent/'test_evaluate.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('prior_info','protocol')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'reviewed_D_test_fixture','exec'),namespace)
    p,m=namespace['protocol']('uniform8')
    for value in (p,m):value.update(mode='frequency_mm',method_variant='phase_firstorder',frequency_t=1.,local_factor=factor,
                                  cg_steps=96,true_residual_diagnostics=True,**PHASE)
    p['solver_config'].update(cg_steps=96,true_residual_diagnostics=True)
    m['solver_config']=copy.deepcopy(p['solver_config'])
    for value in (p,m):
        value['solver_counters'].update(render_pairs=31,renderer_calls=62,vjp_pairs=25,normal_evaluations=20,
            wavelet_transform_batches=360,true_residual_evaluations=5,true_residual_elapsed_seconds=.05,
            full_curvature_normal_evaluations=5,phase_first_order_normal_evaluations=15,**PHASE)
    p['operator_counters'].update(render_pairs=31,renderer_calls=62,data_loss_calls=30,kspace_loss_calls=60)
    for trace in p['trace']:
        trace.update(local_factor=factor,cg_status='converged',cg_converged=True,cg_residual_history=[1.,.01,.0005],gradient_norm=2.,
            damping_rayleigh=.002,damping=.000002,curvature_evaluations=dict(full_rayleigh=1,reduced_cg=2,reduced_endpoint=1),**PHASE,
            wavelet_transform_batches=72,true_residual=dict(diagnostic_only=True,rhs_norm=1.,norm=.0006,relative=.0006,
                recursive_norm=.0005,recursive_relative=.0005,relative_norm_discrepancy=abs(.0006-.0005),
                elapsed_seconds=.01,normal_evaluations=1,phase_curvature_mode='first_order'))
        prior=trace['prior'];prior['lambda_tv']=[.1*factor,.1*factor];prior['lambda_wavelet']=[.12*factor,.1*factor]
        prior['base_tv_phase']['lambda_tv']=[.1*factor,.1*factor]
        fixed=dict(kappa=[.025,.026],tv_eps=.04,phase_eps=.04,w_floor=0.,tv_vector=True,
                   wavelet='db4',wavelet_levels=1,wavelet_on='complex',wavelet_eps=0.,channel_reduction='sum')
        for part in (prior,prior['base_tv_phase'],*prior['wavelet_phases']):part.update(fixed)
        for part in prior['wavelet_phases']:part['lambda_wavelet']=[.12*factor/8,.1*factor/8]
        trace['edge']=dict(p=.9,kappa=30.,quantile_stride=16,gates_frozen_within_outer_step=True,
                           tau_plain=.07,tau_contrast=.08,weight_sum=100.)
    return p,m


@pytest.mark.parametrize('factor',[1.])
def test_reviewed_solver_semantics_accept_actual_scaled_priors_without_rewriting(factor):
    e=utility();p,m=fm_protocol(e,factor);before=copy.deepcopy((p,m))
    work=e.validate_protocol(p,m,1.,factor)
    assert work['vjp_pairs']==25 and work['wavelet_transform_batches']==360
    assert (p,m)==before


@pytest.mark.parametrize('surface',['provenance','metrics','work','trace'])
@pytest.mark.parametrize('key',list(PHASE))
def test_phase_protocol_cannot_be_relabelled_at_any_record_surface(surface,key):
    e=utility();p,m=fm_protocol(e)
    row={'provenance':p,'metrics':m,'work':p['solver_counters'],'trace':p['trace'][2]}[surface]
    row[key]=True if key=='full_MM_majorization' else 'wrong'
    if surface=='work':m['solver_counters']=copy.deepcopy(row)
    with pytest.raises(ValueError):e.validate_protocol(p,m,1.,1.)


@pytest.mark.parametrize('defect',['roles','full_counter','reduced_counter','endpoint_operator','eta','rayleigh','fake_bool',
                                  'nonformal_factor','full_mode_checkpoint'])
def test_full_damping_reduced_CG_and_endpoint_evidence_must_match_actual_work(defect):
    e=utility();p,m=fm_protocol(e)
    if defect=='roles':p['trace'][0]['curvature_evaluations']['reduced_cg']=1
    elif defect=='full_counter':
        for row in (p,m):row['solver_counters']['full_curvature_normal_evaluations']=4
    elif defect=='reduced_counter':
        for row in (p,m):row['solver_counters']['phase_first_order_normal_evaluations']=16
    elif defect=='endpoint_operator':p['trace'][0]['true_residual']['phase_curvature_mode']='full'
    elif defect=='eta':p['trace'][0]['damping']*=.5
    elif defect=='rayleigh':p['trace'][0]['damping_rayleigh']=-1.
    elif defect=='fake_bool':p['full_MM_majorization']=0
    elif defect=='nonformal_factor':
        p,m=fm_protocol(e,.75)
        with pytest.raises(ValueError):e.validate_protocol(p,m,1.,.75)
        return
    else:p['method_variant']=m['method_variant']='frequency_mm'
    with pytest.raises(ValueError):e.validate_protocol(p,m,1.,1.)


@pytest.mark.parametrize('defect',['missing','diagnostic_only','rhs','norm','relative','recursive_norm',
    'recursive_relative','discrepancy','time','count','work_count','work_time','top_cap','top_diagnostic','metric_config'])
def test_precision_diagnostics_and_counted_work_cannot_be_forged(defect):
    e=utility();p,m=fm_protocol(e);diagnostic=p['trace'][0]['true_residual']
    if defect=='missing':del p['trace'][0]['true_residual']
    elif defect=='diagnostic_only':diagnostic['diagnostic_only']=False
    elif defect=='rhs':diagnostic['rhs_norm']=2.
    elif defect=='norm':diagnostic['norm']=float('nan')
    elif defect=='relative':diagnostic['relative']=.1
    elif defect=='recursive_norm':diagnostic['recursive_norm']=.01
    elif defect=='recursive_relative':diagnostic['recursive_relative']=.01
    elif defect=='discrepancy':diagnostic['relative_norm_discrepancy']=0.
    elif defect=='time':diagnostic['elapsed_seconds']=-1.
    elif defect=='count':diagnostic['normal_evaluations']=0
    elif defect=='work_count':
        for value in (p,m):value['solver_counters']['true_residual_evaluations']=0
    elif defect=='work_time':
        for value in (p,m):value['solver_counters']['true_residual_elapsed_seconds']=.01
    elif defect=='top_cap':m['cg_steps']=80
    elif defect=='top_diagnostic':m['true_residual_diagnostics']=False
    else:m['solver_config']['cg_steps']=80
    with pytest.raises(ValueError):e.validate_protocol(p,m,1.,1.)


def full_precision_protocol(e):
    p,m=fm_protocol(e)
    for trace in p['trace']:
        trace.update(cg_iterations=96,cg_matvecs=96,cg_status='maxiter',cg_converged=False,
                     cg_residual_history=[1.]+[.1]*96,wavelet_transform_batches=824,
                     curvature_evaluations=dict(full_rayleigh=1,reduced_cg=96,reduced_endpoint=1))
        trace['true_residual'].update(norm=.11,relative=.11,recursive_norm=.1,recursive_relative=.1,
                                    relative_norm_discrepancy=abs(.11-.1))
    for value in (p,m):value['solver_counters'].update(normal_evaluations=490,vjp_pairs=495,
        render_pairs=501,renderer_calls=1002,wavelet_transform_batches=4120,phase_first_order_normal_evaluations=485)
    p['operator_counters'].update(render_pairs=501,renderer_calls=1002,data_loss_calls=500,kspace_loss_calls=1000)
    return p,m


def test_full_96_step_nonconvergence_preserves_true_residual_and_actual_maximum_work():
    e=utility();p,m=full_precision_protocol(e)
    work=e.validate_protocol(p,m,1.,1.)
    assert work['vjp_pairs']==495 and work['render_pairs']==501 and work['wavelet_transform_batches']==4120


def test_maxiter_label_cannot_hide_a_recursive_residual_below_the_stated_tolerance():
    e=utility();p,m=full_precision_protocol(e);trace=p['trace'][0]
    trace['cg_residual_history'][-1]=.0005
    trace['true_residual'].update(recursive_norm=.0005,recursive_relative=.0005,
                                relative_norm_discrepancy=abs(.11-.0005))
    with pytest.raises(ValueError):e.validate_protocol(p,m,1.,1.)


def test_true_residual_above_tolerance_is_reported_without_rewriting_recursive_convergence():
    e=utility();p,m=fm_protocol(e)
    p['trace'][0]['true_residual'].update(norm=.05,relative=.05,relative_norm_discrepancy=.0495)
    e.validate_protocol(p,m,1.,1.)
    assert p['trace'][0]['cg_converged'] is True


@pytest.mark.parametrize('defect',['frequency','mode','steps','cap','phase','prior','work','adam','geometry','no_update','termination'])
def test_protocol_refuses_partial_incompatible_or_unaccounted_solver_evidence(defect):
    e=utility();p,m=fm_protocol(e)
    if defect=='frequency':m['frequency_t']=.5
    elif defect=='mode':p['mode']='zero'
    elif defect=='steps':p['trace'].pop()
    elif defect=='cap':p['solver_config']['cg_steps']=80
    elif defect=='phase':p['wavelet_phase_count']=1
    elif defect=='prior':p['trace'][0]['prior']['base_tv_phase']['lambda_wavelet']=[.12,.1]
    elif defect=='work':p['operator_counters']['data_loss_calls']+=1
    elif defect=='adam':p['after']['adam']['density_real']['step']=4001
    elif defect=='geometry':p['after']['parameter_sha256']['positions']='changed'
    elif defect=='no_update':p['after']['parameter_sha256']['density_real']=p['before']['parameter_sha256']['density_real']
    else:p['trace'][0].update(cg_status='maxiter',cg_converged=False)
    with pytest.raises(ValueError):e.validate_protocol(p,m,1.,1.)


@pytest.mark.parametrize('defect',['declared_factor','trace_factor','unscaled_tv','unscaled_wave','base_tv','phase_strength',
    'phase_wave','duplicated_tv','duplicated_phase','eps','phase_kappa','outer_kappa','edge_kappa','edge_gate'])
def test_actual_factor_and_fixed_source_prior_metadata_cannot_be_misdeclared(defect):
    e=utility();p,m=fm_protocol(e);prior=p['trace'][0]['prior']
    if defect=='declared_factor':m['local_factor']=.5
    elif defect=='trace_factor':p['trace'][4]['local_factor']=.5
    elif defect=='unscaled_tv':prior['lambda_tv']=[.075,.075]
    elif defect=='unscaled_wave':prior['lambda_wavelet']=[.09,.075]
    elif defect=='base_tv':prior['base_tv_phase']['lambda_tv']=[.075,.075]
    elif defect=='phase_strength':prior['lambda_phase']=[.06,.06]
    elif defect=='phase_wave':prior['wavelet_phases'][7]['lambda_wavelet']=[.09/8,.075/8]
    elif defect=='duplicated_tv':prior['wavelet_phases'][7]['lambda_tv']=[.2,.2]
    elif defect=='duplicated_phase':prior['wavelet_phases'][7]['lambda_phase']=[.03,.03]
    elif defect=='eps':prior['base_tv_phase']['tv_eps']=.08
    elif defect=='phase_kappa':prior['wavelet_phases'][7]['kappa']=[.05,.052]
    elif defect=='outer_kappa':p['trace'][4]['prior']['kappa']=[.05,.052]
    elif defect=='edge_kappa':p['trace'][0]['edge']['kappa']=60.
    else:p['trace'][0]['edge']['gates_frozen_within_outer_step']=False
    with pytest.raises(ValueError):e.validate_protocol(p,m,1.,1.)


@pytest.mark.parametrize('defect',['status','runtime','armijo','merit_increase','backtracking','residual_history',
    'normal_work','wavelet_work','Adam_flag','geometry_flag','source_sha'])
def test_cloned_solver_guards_reject_actual_descent_work_and_source_defects(defect):
    e=utility();p,m=fm_protocol(e)
    if defect=='status':p['status']='partial'
    elif defect=='runtime':p['total_elapsed_seconds']=float('nan')
    elif defect=='armijo':p['trace'][0]['directional_derivative']=-1e9
    elif defect=='merit_increase':p['trace'][0]['merit_after']=2.
    elif defect=='backtracking':p['trace'][0]['backtracking'][0]['alpha']=.5
    elif defect=='residual_history':p['trace'][0]['cg_residual_history'].pop()
    elif defect=='normal_work':p['solver_counters']['normal_evaluations']+=1
    elif defect=='wavelet_work':p['trace'][0]['wavelet_transform_batches']-=1
    elif defect=='Adam_flag':p['original_adam_unchanged']=False
    elif defect=='geometry_flag':p['geometry_unchanged']=False
    else:p['source_checkpoint_sha256']='wrong'
    with pytest.raises(ValueError):e.validate_protocol(p,m,1.,1.)


def trace_file_fixture(p):
    work=dict(render_pairs=1,renderer_calls=2,vjp_pairs=0,gradient_evaluations=0,normal_evaluations=0,
              merit_forward_evaluations=0,support_forward_pairs=1,wavelet_mode='uniform8',wavelet_transform_batches=0,
              true_residual_evaluations=0,true_residual_elapsed_seconds=0.,full_curvature_normal_evaluations=0,
              phase_first_order_normal_evaluations=0,**PHASE)
    result=[]
    for trace in p['trace']:
        work['gradient_evaluations']+=1;work['normal_evaluations']+=4;work['merit_forward_evaluations']+=1
        work['vjp_pairs']+=5;work['render_pairs']+=6;work['renderer_calls']=2*work['render_pairs']
        work['wavelet_transform_batches']+=72
        work['true_residual_evaluations']+=1;work['true_residual_elapsed_seconds']+=.01
        work['full_curvature_normal_evaluations']+=1;work['phase_first_order_normal_evaluations']+=3
        result.append({**copy.deepcopy(trace),'work':dict(work)})
    return result


def test_cloned_trace_guard_recomputes_every_cumulative_work_record():
    e=utility();p,_=fm_protocol(e);actual=trace_file_fixture(p)
    e.validate_trace_file(p,actual)
    actual[2]['work']['vjp_pairs']+=1
    with pytest.raises(ValueError):e.validate_trace_file(p,actual)


@pytest.fixture(params=[1.,.75,.5])
def zero_config_fixture(tmp_path,request):
    e=utility();import yaml
    factor=request.param;run=tmp_path/f'factor{factor}_zero';run.mkdir()
    base=e.load(e.C/'runner.py','_tvwave_zero_config_fixture_C')
    config=base.build_config('audit',0,run,local_factor=factor,nlambda=0.)
    config.setdefault('mode',{})['strict_cuda']=True
    config['frequency_overrides']={'frequency_t':1.,'mask_source':'original acquired mask_p/mask_c; never fit masks',
        'utility_sha256':'a150298dd8de4fcaa4ab77d7b60e00c79c8f6b1da8ad5448c3fe3c7182806933'}
    config['inner_precision_overrides']={'cg_steps':96,'true_residual_diagnostics':True}
    config['phase_firstorder_overrides']=dict(PHASE)
    path=run/'actual_config.yaml';path.write_text(yaml.safe_dump(config))
    hashes={str(e.FM/'frequency_weights.py'):config['frequency_overrides']['utility_sha256']}
    return e,run,path,config,hashes,factor


def test_actual_zero_configuration_is_rebuilt_from_original_source_factor(zero_config_fixture):
    e,run,path,config,hashes,factor=zero_config_fixture
    tv,wave_pre,wave_post={1.:(.1,.12,.1),.75:(.07500000000000001,.09,.07500000000000001),.5:(.05,.06,.05)}[factor]
    assert config['loss']['tv_weight']==tv
    assert config['loss']['lambda_wav_plain']==wave_pre and config['loss']['lambda_wav']==wave_post
    assert config['loss']['lambda_ptv']==.03 and config['loss']['lambda_edge']==.001
    proof=e.validate_actual_config(run,1.,factor,hashes)
    assert proof['sha256']==digest(path) and proof['matches_original_C_build_config'] is True


@pytest.mark.parametrize('defect',['wrong_factor','unscaled_coefficients','phase','edge','frequency','strict_cuda',
    'different_run','missing_precision','wrong_cg','disabled_diagnostics','missing_phase_protocol','full_phase',
    'reduced_damping','majorization','majorization_integer'])
def test_zero_config_cannot_hide_wrong_factor_or_changed_fixed_priors(zero_config_fixture,defect):
    e,run,path,config,hashes,factor=zero_config_fixture;import yaml
    if defect=='wrong_factor':
        base=e.load(e.C/'runner.py','_tvwave_wrong_factor_C')
        wrong=base.build_config('audit',0,run,local_factor=(.5 if factor==1. else 1.),nlambda=0.)
        wrong.setdefault('mode',{})['strict_cuda']=True
        wrong['frequency_overrides']=config['frequency_overrides']
        wrong['inner_precision_overrides']=config['inner_precision_overrides']
        wrong['phase_firstorder_overrides']=config['phase_firstorder_overrides'];config=wrong
    elif defect=='unscaled_coefficients':
        config['loss'].update(tv_weight=.15,lambda_wav_plain=.18,lambda_wav=.15)
    elif defect=='phase':config['loss']['lambda_ptv']=.06
    elif defect=='edge':config['loss']['lambda_edge']=.002
    elif defect=='frequency':config['frequency_overrides']['frequency_t']=.5
    elif defect=='strict_cuda':config['mode']['strict_cuda']=False
    elif defect=='different_run':config['output']['output_dir']='wrong_run'
    elif defect=='missing_precision':del config['inner_precision_overrides']
    elif defect=='wrong_cg':config['inner_precision_overrides']['cg_steps']=80
    elif defect=='disabled_diagnostics':config['inner_precision_overrides']['true_residual_diagnostics']=False
    elif defect=='missing_phase_protocol':del config['phase_firstorder_overrides']
    elif defect=='full_phase':config['phase_firstorder_overrides']['phase_curvature_mode']='full'
    elif defect=='reduced_damping':config['phase_firstorder_overrides']['damping_curvature_mode']='first_order'
    elif defect=='majorization':config['phase_firstorder_overrides']['full_MM_majorization']=True
    else:config['phase_firstorder_overrides']['full_MM_majorization']=0
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):e.validate_actual_config(run,1.,factor,hashes)


def test_source_reconstruction_uses_original_H5_mask_and_kspace_header_only(tmp_path,monkeypatch):
    e=utility();import h5py
    data=tmp_path/'data.h5';maskpath=tmp_path/'mask.h5'
    with h5py.File(data,'w') as f:f.create_dataset('kspace',shape=(2,9,9,9),dtype=np.complex64)
    mask=np.zeros((9,9,9),np.float32);mask[4,4,4]=mask[4,4,5]=mask[0,0,0]=1
    with h5py.File(maskpath,'w') as f:f.create_dataset('mask',data=mask)
    cfg={'data':{channel:{'data_path':str(data),'mask_path':str(maskpath),'presampled':False} for channel in ('plain','contrast')},
         'loss':{'kspace_freq_adaptive':True,'kspace_center_fraction':.05,'kspace_center_weight':.3,
                 'kspace_mid_weight':1.,'kspace_high_weight':2.8,'kspace_high_weight_plain':2.2}}
    manifest={'files':{str(p):digest(p) for p in (data,maskpath)}}
    old_getitem=h5py.Dataset.__getitem__
    def forbid_measurement_values(self,*args,**kwargs):
        assert self.name!='/kspace','measurement values must not enter source weight reconstruction'
        return old_getitem(self,*args,**kwargs)
    monkeypatch.setattr(h5py.Dataset,'__getitem__',forbid_measurement_values)
    sources,masks,domains=e.reconstruct_source_arrays(cfg,manifest)
    for scan,high in (('pre',2.2),('post',2.8)):
        assert np.array_equal(masks[scan],mask)
        assert sources[scan][4,4,4]==np.float32(.3)
        assert sources[scan][4,4,5]==1. and sources[scan][0,0,0]==np.float32(high)
        assert domains[scan]['measurement_values_read'] is False


def test_nvgen_cannot_enter_numeric_scorer_even_when_anchor_missing():
    result=subprocess.run(['/home/congcongliu/anaconda3/envs/nvgen/bin/python','-X','utf8','-B',str(ROOT/'evaluate.py'),
        'score','--exp','missing','--local-factor','1','--selected-recipe','/missing/recipe.json','--output','/missing/output'],
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
    assert result.returncode!=0 and 'actual GSTMI' in result.stderr


@pytest.fixture
def state_fixture(tmp_path,monkeypatch):
    e=utility();import torch
    shapes={n:([2,3] if i<2 else [2,4] if i==2 else [2]) for i,n in enumerate(e.NAMES)}
    models={n:torch.full(shapes[n],i+.25,dtype=torch.float32) for i,n in enumerate(e.NAMES)}
    moments={i:{'step':torch.tensor(4000.),'exp_avg':torch.full_like(models[n],.1),'exp_avg_sq':torch.full_like(models[n],.02)}
             for i,n in enumerate(e.NAMES)}
    nominal=(.0003,.002,.0003,.015,.015,.015,.015)
    groups=[{'params':[i],'initial_lr':nominal[i],'lr':nominal[i]*.9995**4000,'betas':(.9,.999)} for i in range(7)]
    def tensor_digest(x):return hashlib.sha256(x.numpy().tobytes()).hexdigest()
    def snapshot(model):
        hashes={n:tensor_digest(t) for n,t in model.items()}
        result={'parameter_sha256':hashes,'adam':{n:{'step':4000,'exp_avg_sha256':tensor_digest(moments[i]['exp_avg']),
                   'exp_avg_sq_sha256':tensor_digest(moments[i]['exp_avg_sq'])} for i,n in enumerate(e.NAMES)}}
        for key,names in (('model_sha256',e.NAMES),('geometry_sha256',e.NAMES[:3]),('density_sha256',e.NAMES[3:])):
            result[key]=hashlib.sha256(''.join(hashes[n] for n in names).encode()).hexdigest()
        return result
    before=snapshot(models);source=tmp_path/'source.pth'
    torch.save({'iter':4000,'model_state':models,'optim_state':{'state':moments,'param_groups':groups}},source)
    final={n:v.clone() for n,v in models.items()};final['density_real']+=.01;after=snapshot(final)
    provenance={'source_checkpoint':str(source),'before':before,'after':after,'frequency_t':1.,'local_factor':1.,
                'cg_steps':96,'true_residual_diagnostics':True,'frequency_evidence':{'frequency_t':1.},
                'execution':{'declared':'fixture'},'solver_counters':{'fixture':1},'accepted_outer_steps':5}
    saved={'variant':'phase_firstorder','source_iteration':4000,'resumable_by_original_adam':False,
           'source_checkpoint':str(source),'source_checkpoint_sha256':digest(source),'frequency_t':1.,'local_factor':1.,
           'cg_steps':96,'true_residual_diagnostics':True,
           'frequency_evidence':provenance['frequency_evidence'],'execution':provenance['execution'],
           'solver_counters':provenance['solver_counters'],'accepted_outer_steps':5,'model_state':final,
           'wavelet_mode':'uniform8','wavelet_phase_count':8,'fixed_wavelet_shift':None}
    provenance.update(PHASE);saved.update(PHASE)
    path=tmp_path/'density_solver_nonresumable.pth';torch.save(saved,path)
    monkeypatch.setattr(e,'SOURCE_SHA',digest(source));monkeypatch.setattr(e,'SHAPES',shapes)
    real_read=e.read
    monkeypatch.setattr(e,'read',lambda p:{'before':before} if Path(p)==e.R/'block_opt/outputs/all_500/metrics.json' else real_read(p))
    return e,provenance,path


def test_all_seven_source_model_Adam_and_saved_density_bindings_are_recomputed(state_fixture):
    e,provenance,path=state_fixture
    result=e.validate_saved_state(path.parent,provenance,1.,1.)
    assert result['all_seven_final_tensor_hashes_match'] is True
    assert result['original_model_and_Adam_restored_exactly'] is True


@pytest.mark.parametrize('defect',['final_tensor','missing_tensor','geometry','adam_metadata','resume_keys',
    'frequency','factor','cg_cap','diagnostics','execution','variant','phase','damping','majorization'])
def test_saved_state_cannot_hide_a_changed_tensor_moment_or_resumable_profile(state_fixture,defect):
    e,p,path=state_fixture;import torch
    saved=torch.load(path,weights_only=True)
    if defect=='final_tensor':saved['model_state']['density_imag_contrast'][0]+=1
    elif defect=='missing_tensor':del saved['model_state']['density_real']
    elif defect=='geometry':p['after']['geometry_sha256']='wrong'
    elif defect=='adam_metadata':p['after']['adam']['density_real']['step']=4001
    elif defect=='resume_keys':saved['optim_state']={}
    elif defect=='frequency':saved['frequency_t']=.5
    elif defect=='factor':saved['local_factor']=.5
    elif defect=='cg_cap':saved['cg_steps']=80
    elif defect=='diagnostics':saved['true_residual_diagnostics']=False
    elif defect=='execution':saved['execution']={'changed':True}
    elif defect=='variant':saved['variant']='frequency_mm'
    elif defect=='phase':saved['phase_curvature_mode']='full'
    elif defect=='damping':saved['damping_curvature_mode']='first_order'
    else:saved['full_MM_majorization']=True
    torch.save(saved,path)
    with pytest.raises(ValueError):e.validate_saved_state(path.parent,p,1.,1.)


def phase_tiny_fixture(e):
    p,_=fm_protocol(e);trace=copy.deepcopy(p['trace'][:2]);work=trace_file_fixture(p)[1]['work']
    work.update(support_forward_pairs=0,render_pairs=12,renderer_calls=24)
    config=dict(p['solver_config']);config.update(outer_steps=2,cg_steps=4,max_vjp_pairs=40,max_render_pairs=100,max_merit_forwards=24)
    control=copy.deepcopy(trace[:1]);control[0].update(cg_iterations=1,cg_matvecs=1,cg_residual_history=[1.,.0005],
        wavelet_transform_batches=64)
    for key in (*PHASE,'curvature_evaluations'):control[0].pop(key)
    control[0]['true_residual'].pop('phase_curvature_mode')
    old=dict(render_pairs=5,renderer_calls=10,vjp_pairs=4,gradient_evaluations=1,normal_evaluations=3,
        merit_forward_evaluations=1,support_forward_pairs=0,wavelet_mode='uniform8',wavelet_transform_batches=64,
        true_residual_evaluations=1,true_residual_elapsed_seconds=.01)
    names=('render_pairs','renderer_calls','adjoint_calls','vjp_calls','data_loss_calls','kspace_loss_calls')
    before=dict.fromkeys(names,0);source={**before,'render_pairs':1,'renderer_calls':2}
    after={**before,'render_pairs':18,'renderer_calls':36,'data_loss_calls':17,'kspace_loss_calls':34}
    for row in (*trace,*control):
        info=row['prior']
        for part in (info,info['base_tv_phase'],*info['wavelet_phases']):part['kappa']=[.6,.8]
    return dict(PHASE,passed=True,local_factor=1.,tolerance=2e-5,device='cuda:0',source_density_unchanged=True,
        lambda_tv=[.1,.1],lambda_phase=[.03,.03],lambda_wavelet=[.12,.1],phase_gradient_norm=.003,
        source_phase_gradient_relative_error=1e-7,source_phase_value_relative_error=1e-7,phase_full_probe_gradient_norm=.1,
        phase_reduced_probe_gradient_norm=0.,full_control_eta_relative_error=0.,phase_image_vjp_evaluations=4,
        solver_config=config,solver_status='complete',accepted_outer_steps=2,solver_counters=work,trace=trace,
        full_control_solver_config={**config,'outer_steps':1,'cg_steps':1},full_control_solver_status='complete',
        full_control_solver_counters=old,full_control_trace=control,source_check_operator_delta=source,
        operator_counters_before=before,operator_counters_after=after)


def test_actual_tiny_requires_nonzero_phase_source_gradient_and_reduced_solver_proof():
    e=utility();value=phase_tiny_fixture(e);before=copy.deepcopy(value)
    e.validate_phase_firstorder_tiny(value)
    assert value==before and value['lambda_phase']==[.03,.03]


def test_phase_tiny_and_formal_profile_match_actual_parent_stdlib_API():
    e=utility();parent=e.load(e.FM/'run_gpu.py','_PF_scoring_parent_contract_fixture')
    assert parent.strength_protocol(1.,1.)==e.profile(1.,1.)
    value=phase_tiny_fixture(e)
    parent.validate_tiny_phase_check(value)
    e.validate_phase_firstorder_tiny(value)
    p,m=fm_protocol(e)
    parent.validate_phase_trace(p['trace'],p['solver_counters'])
    assert e.validate_protocol(p,m,1.,1.)==p['solver_counters']


@pytest.mark.parametrize('defect',['zero_gradient','source_error','phase_lambda','TV_lambda','wave_lambda','disabled_phase',
    'full_endpoint','full_CG','full_counter','armijo','partial','wrong_cap','missing_trace','false_majorization',
    'full_control','control_eta','phase_branch','source_counters','source_density','image_vjps'])
def test_wave_only_or_relabelled_tiny_cannot_replace_actual_phase_firstorder_check(defect):
    e=utility();value=phase_tiny_fixture(e)
    if defect=='zero_gradient':value['phase_gradient_norm']=0.
    elif defect=='source_error':value['source_phase_gradient_relative_error']=.01
    elif defect=='phase_lambda':value['lambda_phase']=[0.,0.]
    elif defect=='TV_lambda':value['lambda_tv']=[0.,0.]
    elif defect=='wave_lambda':value['lambda_wavelet']=[0.,0.]
    elif defect=='disabled_phase':value['phase_curvature_mode']='full'
    elif defect=='full_endpoint':value['trace'][0]['true_residual']['phase_curvature_mode']='full'
    elif defect=='full_CG':value['trace'][0]['curvature_evaluations']['reduced_cg']=0
    elif defect=='full_counter':value['solver_counters']['full_curvature_normal_evaluations']+=1
    elif defect=='armijo':value['trace'][0]['merit_after']=value['trace'][0]['merit_before']+1.
    elif defect=='partial':value['solver_status']='partial'
    elif defect=='wrong_cap':value['solver_config']['cg_steps']=96
    elif defect=='missing_trace':value['trace'].pop()
    elif defect=='false_majorization':value['full_MM_majorization']=True
    elif defect=='full_control':value['full_control_trace'][0]['phase_curvature_mode']='first_order'
    elif defect=='control_eta':value['full_control_trace'][0]['damping']*=.5
    elif defect=='phase_branch':value['phase_reduced_probe_gradient_norm']=.01
    elif defect=='source_counters':value['operator_counters_after']['render_pairs']+=1
    elif defect=='source_density':value['source_density_unchanged']=False
    else:value['phase_image_vjp_evaluations']=0
    with pytest.raises(ValueError):e.validate_phase_firstorder_tiny(value)


def test_actual_setup_and_failed_formal_attempts_stay_separate_from_completed_rows(tmp_path,monkeypatch):
    e=utility();monkeypatch.setattr(e,'FM',tmp_path);verify=tmp_path/'verification';verify.mkdir()
    for name,stage in (('tiny_bad','tiny'),('formal_bad','solve')):
        put(verify/f'{name}_selection.json',{'parent_pid':10,'reservation_token':name,'stage':stage})
        put(verify/f'{name}_failure.json',{'status':'failed','parent_pid':10,'reservation_token':name,
            'stage':stage,'worker_pid':11,'returncode':17,'error':'actual fixture failure'})
    setup,formal,bound=e.observed_failures()
    assert len(setup)==len(formal)==1 and len(bound)==4
    assert setup[0]['counted_as_completed_training'] is formal[0]['counted_as_completed_training'] is False


def test_aggregate_rejects_missing_duplicate_or_unselected_phase_firstorder_candidates():
    e=utility()
    for factors in ([],[.75],[.5],[1.,1.],[1.,.5],[.75,.5,.5]):
        previous={'trials':[],'status_counts':{'completed':0,'accepted':0,'rejected':0}}
        trials=[dict(frequency_t=1.,local_factor=f,cg_steps=96,true_residual_diagnostics=True,
                     wavelet_mode='uniform8',wavelet_phase_count=8,fixed_wavelet_shift=None,score=-.1,**PHASE) for f in factors]
        with pytest.raises(ValueError):e.apply_acceptance(previous,trials,-.2)
        assert not previous['trials']


def test_single_candidate_aggregate_appends_new_family_without_replacing_Q_H_L_or_failure_costs(tmp_path,monkeypatch):
    """Policy wiring only: all scoring functions use owned synthetic stand-ins."""
    e=utility();recipe,_,_=anchor_fixture(tmp_path,e);anchor=e.load_anchor(recipe)
    anchor['previous']['enhancement_fidelity']={'rows':[{'method':'old_roi'}],'inputs':{'old_roi':{'fixture':1}}}
    original=copy.deepcopy(anchor['previous']);runtime=e.numerical_runtime();invariants={'fixture':'not a real gate'}
    folder=tmp_path/'scoring/unit_candidate';folder.mkdir(parents=True)
    protocol=e.profile(1.,1.);counts={'fixture_strict_planes':0};head={'pre':{'fixture':1},'post':{'fixture':2}}
    score={'score':-.19,'strict_success':False,'planes':['unit_only'],'best_plane':'unit_only','whole_head_metrics':head,
           'method_variant':'phase_firstorder',**protocol}
    trial={'name':'unit_phase_candidate','method_variant':'phase_firstorder',**protocol,'score':-.19,'strict_success':False,
        'plane_counts':counts,'whole_head_metrics':head,'anchor_incumbent':anchor['score'],'improves_anchor':True,
        'anchor_bindings':anchor['bindings'],'solver_evidence':{'budget_disclosure':'synthetic unit data'}}
    enhancement={'rows':[{'method':trial['name']}],'inputs':{trial['name']:{'fixture':2}}}
    for name,value in (('trial.json',trial),('objective.json',score),('enhancement.json',enhancement),
                       ('lesion_allmetrics.json',{'fixture':'no clinical scores'})):put(folder/name,value)
    done={'status':'verified','returncode':0,'numerical_runtime':runtime,'anchor_bindings':anchor['bindings'],
        'invariants_before_and_after':invariants,'observed_artifact_sha256_before_and_after':{},
        'exp':trial['name'],**protocol}
    for name,key in (('trial.json','trial_sha256'),('objective.json','objective_sha256'),
                     ('enhancement.json','enhancement_sha256'),('lesion_allmetrics.json','raw_sha256')):done[key]=digest(folder/name)
    put(folder/'completed.json',done)
    monkeypatch.setattr(e,'S',tmp_path/'scoring')
    monkeypatch.setattr(e,'load_anchor',lambda p:copy.deepcopy(anchor))
    monkeypatch.setattr(e,'scoring_implementations',lambda:{})
    monkeypatch.setattr(e,'load',lambda *a:SimpleNamespace(snapshot_invariants=lambda:invariants))
    monkeypatch.setattr(e,'observed_failures',lambda:([],[],{}))
    monkeypatch.setattr(e,'objective',SimpleNamespace(SCANS=('pre','post'),load_baseline=lambda p:{},score=lambda *a:score))
    monkeypatch.setattr(e,'report',SimpleNamespace(plane_counts=lambda v:counts,head_metrics=lambda v:v))
    output=tmp_path/'scoring/combined_unit';e.aggregate([folder],recipe,output)
    result=json.loads((output/'summary.json').read_text());family=result['phase_firstorder_family']
    assert family['completed_candidates']==['unit_phase_candidate']
    assert family['declared_local_factors']==family['completed_local_factors']==[1.]
    assert all(family[k]==v for k,v in PHASE.items())
    assert result['trials'][:-1]==original['trials'] and result['status_counts']['completed']==len(original['trials'])+1
    for key in ('inner_precision_family','frequency_adam_family','frequency_lowreg_family','setup_failures','failed_trials'):
        assert result[key]==original[key]
    assert result['enhancement_fidelity']['rows'][0]==original['enhancement_fidelity']['rows'][0]


def actual_route_commands(tmp_path,monkeypatch):
    e=utility();supervisor=e.load(e.FM/'run_gpu.py','_fm_relative_gate_regression')
    monkeypatch.chdir(e.FM)
    flags=('--review-gate','--cuda-operator-gate','--zero-audit-gate','--selected-recipe')
    argv=['--stage','solve','--exp','command_fixture','--frequency-t','1','--local-factor','1','--cg-steps','96']
    bindings={}
    for index,flag in enumerate(flags):
        path=put(tmp_path/f'gate{index}.json',{'fixture':'path comparison only; not a passed gate'})
        argv.extend((flag,os.path.relpath(path,e.FM)));bindings[str(path)]=digest(path)
    args=supervisor.parser().parse_args(argv)
    launch=tmp_path/'launch.json';actual=supervisor._command(args,launch)
    request=supervisor._request(args)
    from argparse import Namespace
    absolute=Namespace(**vars(args))
    for flag in flags:
        name=flag[2:].replace('-','_');setattr(absolute,name,Path(request[name]))
    expected=supervisor._command(absolute,launch)
    return e,actual,expected,bindings


def test_real_frozen_parser_relative_gate_paths_match_saved_absolute_request(tmp_path,monkeypatch):
    e,actual,expected,bindings=actual_route_commands(tmp_path,monkeypatch)
    assert actual!=expected
    proof=e.validate_child_command(actual,expected,bindings)
    assert proof['normalized_command']==expected
    assert proof['relative_gate_resolution_root']==str(e.FM)
    assert proof['historical_cwd_recorded'] is False


@pytest.mark.parametrize('defect',['frequency','factor','budget','diagnostic_flag','wrong_path','wrong_hash','duplicate_flag',
                                  'phase_flag','missing_phase_flag'])
def test_gate_path_equivalence_does_not_relax_other_tokens_paths_or_hashes(tmp_path,monkeypatch,defect):
    e,actual,expected,bindings=actual_route_commands(tmp_path,monkeypatch)
    if defect=='frequency':actual[actual.index('--frequency-t')+1]='0.5'
    elif defect=='factor':actual[actual.index('--local-factor')+1]='0.5'
    elif defect=='budget':actual[actual.index('--cg-steps')+1]='80'
    elif defect=='diagnostic_flag':actual.remove('--true-residual-diagnostics')
    elif defect=='wrong_path':actual[actual.index('--review-gate')+1]=str(put(tmp_path/'wrong.json',{}))
    elif defect=='wrong_hash':Path(next(iter(bindings))).write_text('{"changed":true}')
    elif defect=='phase_flag':actual[actual.index('--phase-curvature-mode')+1]='full'
    elif defect=='missing_phase_flag':
        index=actual.index('--phase-curvature-mode');del actual[index:index+2]
    else:actual.extend(('--review-gate',actual[actual.index('--review-gate')+1]))
    with pytest.raises(ValueError):e.validate_child_command(actual,expected,bindings)
