"""Declared lower-prior grid and dynamic completed-H gate, CPU only."""
import copy
import dataclasses
import json

import pytest

import runner
import run_gpu as launch
from test_driver import ExplicitPhysics,PRIOR
from test_run_gpu import supervisor


def test_factor_trace_binds_metadata_without_rewriting_numeric_evidence():
    cfg=runner.SolverConfig(outer_steps=1,cg_steps=3,true_residual_diagnostics=True)
    result=runner.optimize_density(ExplicitPhysics(),prior_kwargs=PRIOR,edge_weight=0.,config=cfg,
                                   phase_curvature_mode="first_order")
    row=result.trace[0];before=copy.deepcopy(row)
    tagged=runner.factor_trace(row,1.)
    assert tagged['local_factor']==1. and row==before
    assert {k:v for k,v in tagged.items() if k!='local_factor'}==before
    assert all(tagged[k]==v for k,v in runner.phase_protocol().items())


def test_factor_trace_cannot_relabel_an_actual_full_helper_update():
    cfg=runner.SolverConfig(outer_steps=1,cg_steps=3,true_residual_diagnostics=True)
    result=runner.optimize_density(ExplicitPhysics(),prior_kwargs=PRIOR,edge_weight=0.,config=cfg)
    with pytest.raises(ValueError,match="phase"):
        runner.factor_trace(result.trace[0],1.)


@pytest.mark.parametrize('factor',[.75,.5])
def test_lower_profiles_and_all_child_commands_preserve_actual_fraction(factor):
    with pytest.raises(ValueError):runner.strength_protocol(1.,factor)
    with pytest.raises(ValueError):launch.strength_protocol(1.,factor)
    for stage in ('tiny','zero','solve'):
        args=launch.parser().parse_args(['--stage',stage,'--exp','not_run','--local-factor',str(factor),
                                        '--review-gate','unused','--cuda-operator-gate','unused'])
        with pytest.raises(ValueError):launch._command(args,'launch.json')


def quality_recipe(tmp_path,monkeypatch,defect=None):
    root=tmp_path/'L';root.mkdir();monkeypatch.setattr(launch,'ROOT',root)
    q={'name':'Q_cpu_fixture','method_variant':'inner_precision_mm','cg_steps':96,
       'true_residual_diagnostics':True,'strict_success':False}
    h=[{'name':f'H_cpu_fixture_{factor}','method_variant':'frequency_adam','frequency_t':1.,
        'geometry_factor':factor,'strict_success':False} for factor in (1.,2.)]
    l=[{'name':f'L_cpu_fixture_{factor}','method_variant':'frequency_lowreg_mm','frequency_t':1.,
        'local_factor':factor,'cg_steps':96,'true_residual_diagnostics':True,'strict_success':False} for factor in (.75,.5)]
    history={'status':'completed_real_trial_snapshot','inner_precision_family':{
        'frequency_t':1.,'cg_steps':96,'true_residual_diagnostics':True,'completed_candidates':[q['name']]},
        'frequency_adam_family':{'frequency_t':1.,'declared_geometry_factors':[1.,2.],
                                 'completed_geometry_factors':[1.,2.]},'trials':[q,*h,*l],
        'frequency_lowreg_family':{'frequency_t':1.,'declared_local_factors':[.75,.5],
            'completed_local_factors':[.75,.5],'completed_candidates':[v['name'] for v in l],
            'cg_steps':96,'true_residual_diagnostics':True}}
    if defect=='pending_H':history.pop('frequency_adam_family')
    elif defect=='one_H':history['frequency_adam_family']['completed_geometry_factors']=[1.];history['trials'].remove(h[1])
    elif defect=='H_order':history['frequency_adam_family']['completed_geometry_factors']=[2.,1.]
    elif defect=='pending_Q':history.pop('inner_precision_family')
    elif defect=='Q80':history['inner_precision_family']['cg_steps']=80
    elif defect=='strict_Q':q['strict_success']=True
    elif defect=='strict_H':h[1]['strict_success']=True
    elif defect=='pending_L':history.pop('frequency_lowreg_family')
    elif defect=='one_L':history['frequency_lowreg_family']['completed_local_factors']=[.75]
    elif defect=='L_order':history['frequency_lowreg_family']['completed_candidates'].reverse()
    elif defect=='strict_L':l[1]['strict_success']=True
    summary=tmp_path/'completed_summary.json';summary.write_text(json.dumps(history))
    source=tmp_path/'development_incumbent.json';score=-.123
    actual={'score':score,'strict_success':defect=='strict_current','validated_summary':str(summary),
            'validated_summary_sha256':launch.sha256(summary)}
    source.write_text(json.dumps(actual));hashes={'CPU':'sourcehash'};invariants={'CPU':'inputhash'}
    recipe={'status':'selected','selected_by':'root','code_sha256':hashes,'invariants':invariants,
        'frequency_t':1.,'local_factor_grid':[1.],'source_iteration':4000,'outer_steps':5,'cg_steps':96,
        **launch.phase_protocol(),
        'true_residual_diagnostics':True,'max_vjp_pairs':500,'max_render_pairs':700,'max_merit_forwards':100,
        'nlambda':0.,'wavelet_mode':'uniform8','damping_rel':.001,'cg_rtol':.001,'minimum_score_improvement':1e-4,
        'incumbent':{'path':str(source),'sha256':launch.sha256(source),'score':score},
        'completed_prior_summary':{'path':str(summary),'sha256':launch.sha256(summary)}}
    if defect=='grid_order':recipe['local_factor_grid']=[.5,.75]
    path=root/'recipe.json';path.write_text(json.dumps(recipe))
    return path,hashes,invariants


@pytest.mark.parametrize('defect',[None,'pending_H','one_H','H_order','pending_Q','Q80','strict_Q','strict_H','strict_current','grid_order','pending_L','one_L','L_order','strict_L'])
def test_formal_recipe_requires_complete_Q_H_and_no_strict_win(tmp_path,monkeypatch,defect):
    path,hashes,invariants=quality_recipe(tmp_path,monkeypatch,defect)
    if defect is None:
        assert launch.validate_recipe(path,hashes,invariants)['local_factor_grid']==[1.]
    else:
        with pytest.raises(ValueError):launch.validate_recipe(path,hashes,invariants)


def test_formal_stale_L_grid_is_not_a_phase_candidate(supervisor,monkeypatch):
    args=supervisor.args;args.stage='solve';args.local_factor=1.
    args.cuda_operator_gate=args.zero_audit_gate=args.selected_recipe=supervisor.review
    monkeypatch.setattr(launch,'validate_observed_gate',lambda *a,**k:None)
    monkeypatch.setattr(launch,'validate_recipe',lambda *a:{'local_factor_grid':[.75,.5]})
    with pytest.raises(ValueError,match='root-selected'):
        launch.validate_stage_gates(args,supervisor.hashes,supervisor.invariants)
