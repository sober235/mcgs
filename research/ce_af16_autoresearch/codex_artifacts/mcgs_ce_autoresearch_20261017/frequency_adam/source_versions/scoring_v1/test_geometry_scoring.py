"""Focused CPU fixtures; no reconstruction measurements or GPU launches."""
from argparse import Namespace
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

SPEC = importlib.util.spec_from_file_location("geometry_scoring_entry",Path(__file__).with_name("evaluate.py"))
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)


def scalar_state(step, factor=1.):
    groups = []
    for i,(name,nominal) in enumerate(zip(ev.NAMES,ev.NOMINAL)):
        scale = factor if i < 3 else 1.
        groups.append(dict(name=name,initial_lr=nominal*scale,lr=nominal*.9995**4000*scale*.9995**(step-4000),
                           betas=[.9,.999],eps=1e-8,weight_decay=0.,step=float(step)))
    return dict(groups=groups,scheduler=dict(last_epoch=step,step_count=step+1,gamma=.9995,
                                           base_lrs=[g["initial_lr"] for g in groups],last_lrs=[g["lr"] for g in groups]))


def records(factor=2.):
    source,applied,final = scalar_state(4000),scalar_state(4000,factor),scalar_state(4500,factor)
    geometry = dict(status="complete",restore=dict(factor=factor,source_iteration=4000,is_noop=(factor==1.),
        geometry_groups=list(ev.NAMES[:3]),scheduler_clock_preserved=True,source=source,after=applied),
        after_restore=deepcopy(applied),final=final,planned=dict(geometry_factor=factor,geometry_groups=list(ev.NAMES[:3]),
        density_lr_factor=1.,betas=[.9,.999],apply_after_strict_restore_iteration=4000))
    def tensor_snapshot(step,actual):
        return dict(parameter_sha256={n:n+str(step) for n in ev.NAMES},
                    adam={n:dict(step=step,exp_avg_sha256="first"+str(step),exp_avg_sq_sha256="second"+str(step)) for n in ev.NAMES},
                    model_sha256="model"+str(step),geometry_restart=deepcopy(actual))
    metrics = dict(mode="all",method_variant="frequency_adam",frequency_t=1.,geometry_factor=factor,source_iteration=4000,end_iteration=4500,forward_backward_evaluations=500,
                   source_checkpoint_sha256=ev.SOURCE_SHA,before=tensor_snapshot(4000,applied),after=tensor_snapshot(4500,final))
    geometry.update(method_variant="frequency_adam",frequency_t=1.,geometry_factor=factor)
    provenance = deepcopy(metrics)
    provenance.update(status="complete",geometry_restart=deepcopy(geometry))
    return geometry,metrics,provenance


@pytest.mark.parametrize("factor",[1.,2.])
def test_only_three_geometry_rates_scale_for_all_declared_factors(factor):
    geometry,metrics,provenance = records(factor)
    source,applied,final = ev.validate_optimizer(geometry,metrics,provenance,factor)
    assert all(applied["groups"][i] == source["groups"][i] for i in range(3,7))
    assert final["scheduler"]["last_epoch"] == 4500


def test_density_rate_change_is_rejected():
    geometry,metrics,provenance = records()
    geometry["restore"]["after"]["groups"][3]["lr"] *= 2
    with pytest.raises(ValueError): ev.validate_optimizer(geometry,metrics,provenance,2.)


@pytest.mark.parametrize("stage",["source","after","final"])
def test_beta_change_is_rejected(stage):
    geometry,metrics,provenance = records()
    value = geometry[stage] if stage == "final" else geometry["restore"][stage]
    value["groups"][0]["betas"] = [.9,.99]
    provenance["geometry_restart"] = deepcopy(geometry)
    with pytest.raises(ValueError): ev.validate_optimizer(geometry,metrics,provenance,2.)


@pytest.mark.parametrize("status",["starting","restored","failed","partial"])
def test_partial_execution_is_never_completed(status):
    geometry,metrics,provenance = records()
    geometry["status"] = status
    with pytest.raises(ValueError,match="not a completed"): ev.validate_optimizer(geometry,metrics,provenance,2.)


def test_coherent_Adam4500_is_required_for_every_group():
    geometry,metrics,provenance = records()
    metrics["after"]["adam"]["rotations"]["step"] = 4000
    provenance["after"] = deepcopy(metrics["after"])
    with pytest.raises(ValueError,match="seven Adam groups"): ev.validate_optimizer(geometry,metrics,provenance,2.)


def test_original_scheduler_clock_cannot_reset():
    geometry,metrics,provenance = records()
    geometry["restore"]["source"]["scheduler"]["last_epoch"] = 0
    provenance["geometry_restart"] = deepcopy(geometry)
    with pytest.raises(ValueError,match="clocks"): ev.validate_optimizer(geometry,metrics,provenance,2.)


def trace_frequency_manifest():
    return {'frequency_t':1.,'scans':{scan:{name:{'byte_sha256':f'{scan}_{name}'}
        for name in ('source_weight','effective_weight','acquired_mask')} for scan in ('pre','post')}}


def trace_frequency_use(step):
    return dict(frequency_t=1.,effective_weight_byte_sha256={s:f'{s}_effective_weight' for s in ('pre','post')},
        source_weight_byte_sha256={s:f'{s}_source_weight' for s in ('pre','post')},
        acquired_mask_byte_sha256={s:f'{s}_acquired_mask' for s in ('pre','post')},installed_before_first_forward=True,
        actual_trainer_weights_match=True,loss_call_scans=['pre','post'],loss_calls_total=2*step,dc_norm={'pre':8.,'post':16.})


def traces(factor=2.):
    source = scalar_state(4000)
    rows = []
    for i,shift in enumerate(ev.original_shifts()):
        used = {g["name"]:g["lr"]*(factor if j < 3 else 1.)*.9995**i for j,g in enumerate(source["groups"])}
        rows.append(dict(mode="all",absolute_iteration=4001+i,wavelet_shift=shift,gradients_finite=True,
            full_gradient_group_count=7,updated_parameter_groups=list(ev.NAMES),loss=.1,unclipped_grad_norm=.2,
            postclip_gradient_norm=.2,learning_rates_used=used,learning_rates_next={n:v*.9995 for n,v in used.items()},
            **{"nonlocal":{"enabled":False},"frequency_use":trace_frequency_use(i+1)}))
    return rows,source


@pytest.mark.parametrize("factor",[1.,2.])
def test_all500_original_wavelet_draws_and_rates(factor):
    rows,source = traces(factor)
    ev.validate_trace(rows,factor,source,trace_frequency_manifest(),{"pre":8.,"post":16.})
    assert [r["wavelet_shift"] for r in rows[:4]] == [[1,1,1],[1,1,1],[1,0,0],[1,1,1]]


@pytest.mark.parametrize("defect",["short","wavelet","density_lr","clock","groups","nonfinite","nonlocal"])
def test_actual_trace_defects_are_rejected(defect):
    rows,source = traces()
    if defect == "short": rows.pop()
    elif defect == "wavelet": rows[210]["wavelet_shift"] = [9,9,9]
    elif defect == "density_lr": rows[499]["learning_rates_used"]["density_imag"] *= 2
    elif defect == "clock": rows[110]["absolute_iteration"] = 111
    elif defect == "groups": rows[100]["updated_parameter_groups"].pop()
    elif defect == "nonfinite": rows[99]["unclipped_grad_norm"] = float("nan")
    else: rows[20]["nonlocal"]["enabled"] = True
    with pytest.raises(ValueError): ev.validate_trace(rows,2.,source,trace_frequency_manifest(),{"pre":8.,"post":16.})


@pytest.mark.parametrize('defect',['missing','frequency','source_weight','effective_weight','mask','late_install',
    'reverted_weight','wrong_scan','loss_count','dc_norm'])
def test_every_original_Adam_step_requires_actual_bound_frequency_use(defect):
    rows,source=traces();value=rows[211]['frequency_use']
    if defect=='missing':del rows[211]['frequency_use']
    elif defect=='frequency':value['frequency_t']=0.
    elif defect=='source_weight':value['source_weight_byte_sha256']['pre']='changed'
    elif defect=='effective_weight':value['effective_weight_byte_sha256']['post']='changed'
    elif defect=='mask':value['acquired_mask_byte_sha256']['post']='fit_mask'
    elif defect=='late_install':value['installed_before_first_forward']=False
    elif defect=='reverted_weight':value['actual_trainer_weights_match']=False
    elif defect=='wrong_scan':value['loss_call_scans']=['pre','pre']
    elif defect=='loss_count':value['loss_calls_total']-=2
    else:value['dc_norm']['pre']*=2
    with pytest.raises(ValueError):ev.validate_trace(rows,2.,source,trace_frequency_manifest(),{'pre':8.,'post':16.})


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def anchor_fixture(tmp_path):
    trials=[dict(name='old_best',score=-.2,accepted=True),
        *[dict(name=f'geo{f}',method_variant='geometry_lr_restart',geometry_factor=float(f),score=-.3,accepted=False) for f in (2,4,8)],
        *[dict(name=f'freq{t}',method_variant='frequency_mm',frequency_t=t,score=-.3,accepted=False) for t in (.5,1.)],
        *[dict(name=f'tvwave{f}',method_variant='frequency_mm',frequency_t=1.,local_factor=float(f),score=-.3,accepted=False) for f in (2,4)],
        dict(name='precision96',method_variant='inner_precision_mm',frequency_t=1.,local_factor=1.,cg_steps=96,
             true_residual_diagnostics=True,strict_success=False,score=-.3,accepted=False)]
    prior=dict(status='completed_real_trial_snapshot',trials=trials,
        geometry_family={'completed_factors':[2.,4.,8.]},frequency_family={'completed_t':[.5,1.]},
        tvwave_family={'completed_local_factors':[2.,4.]},inner_precision_family={'completed_candidates':['precision96'],
            'frequency_t':1.,'cg_steps':96,'true_residual_diagnostics':True},
        status_counts={'completed':len(trials),'accepted':1,'rejected':len(trials)-1},
        setup_failures=[{'name':'old_failed_setup','cost':17}],failed_trials=[{'name':'old_failed_trial','cost':23}])
    summary=put(tmp_path/'summary.json',prior)
    incumbent=put(tmp_path/'incumbent.json',dict(score=-.2,strict_success=False,
        validated_summary=str(summary),validated_summary_sha256=digest(summary)))
    recipe=put(tmp_path/'recipe.json',dict(status='selected',selected_by='root',frequency_t=1.,geometry_factors=[1.,2.],
        minimum_score_improvement=1e-4,completed_prior_summary={'path':str(summary),'sha256':digest(summary)},
        incumbent={'path':str(incumbent),'sha256':digest(incumbent),'score':-.2}))
    return recipe,summary,incumbent


def test_dynamic_closed_Q_anchor_retains_all_prior_rows_failures_and_costs(tmp_path):
    recipe,_,_=anchor_fixture(tmp_path);value=ev.load_anchor(recipe)
    previous=deepcopy(value['previous']);original=deepcopy(previous)
    trials=[dict(name='flat_g1',geometry_factor=1.,frequency_t=1.,score=-.19),
            dict(name='flat_g2',geometry_factor=2.,frequency_t=1.,score=-.195)]
    best=ev.apply_acceptance(previous,trials,value['score'])
    assert best==-.19 and [t['accepted'] for t in trials]==[True,False]
    assert previous['trials'][:-2]==original['trials']
    assert previous['setup_failures']==original['setup_failures'] and previous['failed_trials']==original['failed_trials']


@pytest.mark.parametrize('attack',['missing','changed','unresolved_Q','missing_Q_row','wrong_Q_cap','wrong_Q_family','strict_Q','wrong_score','wrong_grid'])
def test_unresolved_or_stale_root_anchors_are_not_fabricated(tmp_path,attack):
    recipe,summary,current=anchor_fixture(tmp_path);value=json.loads(recipe.read_text())
    if attack=='missing':summary.unlink()
    elif attack=='changed':summary.write_text('{}')
    elif attack in ('unresolved_Q','missing_Q_row','wrong_Q_cap','wrong_Q_family','strict_Q'):
        previous=json.loads(summary.read_text())
        if attack=='unresolved_Q':previous['inner_precision_family']['completed_candidates']=[]
        elif attack=='missing_Q_row':previous['trials'].pop();previous['status_counts']['completed']-=1
        elif attack=='wrong_Q_cap':previous['trials'][-1]['cg_steps']=80
        elif attack=='wrong_Q_family':previous['inner_precision_family']['cg_steps']=80
        else:previous['trials'][-1]['strict_success']=True
        put(summary,previous);value['completed_prior_summary']['sha256']=digest(summary)
        incumbent=json.loads(current.read_text());incumbent['validated_summary_sha256']=digest(summary)
        put(current,incumbent);value['incumbent']['sha256']=digest(current)
    elif attack=='wrong_score':value['incumbent']['score']=-.1
    else:value['geometry_factors']=[2.,1.]
    put(recipe,value)
    with pytest.raises((ValueError,FileNotFoundError)):ev.load_anchor(recipe)


def test_tiny_anchor_improvement_does_not_qualify():
    previous=dict(trials=[],status_counts=dict(completed=0,accepted=0,rejected=0))
    trials=[dict(geometry_factor=1.,frequency_t=1.,score=-.19999),
            dict(geometry_factor=2.,frequency_t=1.,score=-.1998)]
    ev.apply_acceptance(previous,trials,-.2)
    assert [t['accepted'] for t in trials]==[False,True]


def test_actual_H_selection_request_schema_preserves_failures_without_completed_training(tmp_path,monkeypatch):
    monkeypatch.setattr(ev,'G',tmp_path)
    for name,stage in (('audit_failed','audit0'),('trial_failed','trial500')):
        put(tmp_path/f'verification/{name}_selection.json',dict(parent_pid=10,reservation_token=name,
            request={'stage':stage,'exp':name,'geometry_factor':1.,'frequency_t':1.}))
        put(tmp_path/f'verification/{name}_failure.json',dict(status='failed',stage=stage,parent_pid=10,
            reservation_token=name,worker_pid=11,returncode=17,error='actual fixture failure'))
    setup,formal,bindings=ev.observed_failures()
    assert len(setup)==len(formal)==1 and len(bindings)==4
    assert setup[0]['counted_as_completed_training'] is formal[0]['counted_as_completed_training'] is False


@pytest.mark.parametrize('defect',[None,'launch','completion','proof_before','proof_after','bytes'])
def test_actual_control_and_quality_anchor_bindings_must_survive_the_whole_run(tmp_path,defect):
    path=put(tmp_path/'anchor.json',{'fixture':'immutable quality anchor'})
    bindings={str(path):digest(path)}
    launch={'control_bindings':dict(bindings)};complete={'control_bindings':dict(bindings)}
    proof={'control_bindings_before':dict(bindings),'control_bindings_after':dict(bindings)}
    if defect=='launch':launch['control_bindings']={}
    elif defect=='completion':complete['control_bindings']={}
    elif defect=='proof_before':proof['control_bindings_before']={}
    elif defect=='proof_after':proof['control_bindings_after']={}
    elif defect=='bytes':put(path,{'fixture':'changed during the run'})
    if defect:
        with pytest.raises(ValueError):ev.validate_control_bindings(launch,complete,proof,bindings)
    else:assert ev.validate_control_bindings(launch,complete,proof,bindings)==bindings


@pytest.mark.parametrize('factors',[[2.,1.],[1.,1.],[1.,4.],[],[1.],[1.,2.,2.]])
def test_family_order_duplicates_and_undeclared_factors_are_rejected(factors):
    previous=dict(trials=[],status_counts=dict(completed=0,accepted=0,rejected=0))
    with pytest.raises(ValueError):
        ev.apply_acceptance(previous,[dict(geometry_factor=f,frequency_t=1.,score=-.09) for f in factors],-.2)


def test_live_child_command_schema_cpu_only():
    supervisor = ev.load(ev.G/"run_gpu.py","geometry_scoring_fixture_supervisor")
    assert set(supervisor.PRODUCTION) == {"overrides.py","runner.py","run_gpu.py","frequency_weights.py"}
    args = Namespace(stage="trial500",exp="fixture",geometry_factor=2.,frequency_t=1.)
    command = supervisor.child_command(args,ev.G/"verification/future_launch.json",7)
    assert command[:6] == [str(ev.ar.PYTHON),"-X","utf8","-B","-u",str(ev.G/"runner.py")]
    assert command[command.index("--geometry-factor")+1] == "2.0"
    assert command[command.index("--frequency-t")+1] == "1.0"


def checkpoint_fixture():
    helper = ev.load(ev.G/"runner.py","geometry_checkpoint_fixture_helper")
    observed = scalar_state(4500,2.)
    params = {n:torch.tensor([float(i),float(i+1)],dtype=torch.float32) for i,n in enumerate(ev.NAMES)}
    states = {i:dict(step=torch.tensor(4500.),exp_avg=torch.tensor([i*.01,i*.02]),
                     exp_avg_sq=torch.tensor([.001,.002])) for i in range(7)}
    groups = [{**g,"params":[i]} for i,g in enumerate(observed["groups"])]
    checkpoint = dict(iter=4500,model_state=params,optim_state=dict(param_groups=groups,state=states))
    def digest(t): return hashlib.sha256(memoryview(t.numpy()).cast("B")).hexdigest()
    snapshot = dict(parameter_sha256={n:digest(t) for n,t in params.items()},
                    adam={n:dict(step=4500,**{k+"_sha256":digest(states[i][k]) for k in ("exp_avg","exp_avg_sq")})
                          for i,n in enumerate(ev.NAMES)})
    shapes = {n:[2] for n in ev.NAMES}
    return helper,checkpoint,snapshot,observed,shapes


def test_actual_pure_checkpoint_helper_accepts_coherent_Adam4500_fixture():
    helper,checkpoint,snapshot,observed,shapes = checkpoint_fixture()
    assert helper.validate_checkpoint(checkpoint,shapes,4500,snapshot,observed)["all_seven_model_and_moment_hashes_match"] is True


@pytest.mark.parametrize("defect",["CG_nonresumable","model","moment","step","density_rate","state_alias"])
def test_actual_checkpoint_defects_are_rejected(defect):
    helper,checkpoint,snapshot,observed,shapes = checkpoint_fixture()
    if defect == "CG_nonresumable":
        checkpoint.pop("iter");checkpoint.pop("optim_state");checkpoint["source_iteration"] = 4000
    elif defect == "model": checkpoint["model_state"]["positions"] += 1
    elif defect == "moment": checkpoint["optim_state"]["state"][4]["exp_avg"] += 1
    elif defect == "step": checkpoint["optim_state"]["state"][5]["step"] = torch.tensor(4000.)
    elif defect == "density_rate": checkpoint["optim_state"]["param_groups"][6]["lr"] *= 2
    else: checkpoint["optim_state"]["param_groups"][1]["params"] = [0]
    with pytest.raises(ValueError): helper.validate_checkpoint(checkpoint,shapes,4500,snapshot,observed)


def frequency_record_fixture(steps=500,t=1.,factor=2.):
    evidence=trace_frequency_manifest();evidence['frequency_t']=t
    use=trace_frequency_use(steps);use['frequency_t']=t
    return dict(status='complete',method_variant='frequency_adam',frequency_t=t,geometry_factor=factor,
        frequency_evidence=evidence,frequency_array_check={'passed':True,'frequency_t':t},
        installed_before_first_forward=True,dc_norm={'pre':8.,'post':16.},loss_calls_total=2*steps,
        completed_steps=steps,final_frequency_use=use if steps else None,
        final_weights_check=dict(passed=True,actual_trainer_weights_match=True,
            source_weights_unchanged=True,original_masks_unchanged=True))


@pytest.mark.parametrize('steps,t,factor',[(0,1.,1.),(2,0.,1.),(500,1.,1.),(500,1.,2.)])
def test_frequency_record_binds_distinct_audit_noop_and_trial_protocols(steps,t,factor):
    record=frequency_record_fixture(steps,t,factor)
    embedded=[{'frequency_adam':deepcopy(record)} for _ in range(3)]
    assert ev.validate_frequency_record(record,embedded,steps,t,factor)['completed_steps']==steps


@pytest.mark.parametrize('defect',['partial','embedded','steps','losses','late','weights','source','mask','final_use'])
def test_final_frequency_record_cannot_hide_uninstalled_or_changed_actual_weights(defect):
    record=frequency_record_fixture();embedded=[{'frequency_adam':record} for _ in range(3)]
    if defect=='partial':record['status']='partial'
    elif defect=='embedded':embedded[2]={'frequency_adam':{}}
    elif defect=='steps':record['completed_steps']=499
    elif defect=='losses':record['loss_calls_total']=998
    elif defect=='late':record['installed_before_first_forward']=False
    elif defect=='weights':record['final_weights_check']['actual_trainer_weights_match']=False
    elif defect=='source':record['final_weights_check']['source_weights_unchanged']=False
    elif defect=='mask':record['final_weights_check']['original_masks_unchanged']=False
    else:record['final_frequency_use']['effective_weight_byte_sha256']['pre']='reverted'
    with pytest.raises(ValueError):ev.validate_frequency_record(record,embedded,500,1.,2.)


def checkpoint_recipe(t=1.,factor=2.):
    return dict(version=1,frequency_t=t,geometry_factor=factor,local_factor=1.,nlambda=0.,
        wavelet_protocol='original_single_phase',frequency_manifest_sha256='f'*64,requires_same_recipe_on_resume=True)


def test_ordinary_checkpoint_discloses_frequency_and_geometry_resume_recipe():
    value={'frequency_adam_recipe':checkpoint_recipe()}
    assert ev.validate_checkpoint_recipe(value,1.,2.,'f'*64)['requires_same_recipe_on_resume'] is True


@pytest.mark.parametrize('defect',['missing','frequency','geometry','manifest','wavelet','resume','local','NL'])
def test_checkpoint_cannot_claim_generic_original_frequency_or_wrong_resume_recipe(defect):
    value={'frequency_adam_recipe':checkpoint_recipe()};recipe=value['frequency_adam_recipe']
    if defect=='missing':value.clear()
    elif defect=='frequency':recipe['frequency_t']=0.
    elif defect=='geometry':recipe['geometry_factor']=1.
    elif defect=='manifest':recipe['frequency_manifest_sha256']='wrong'
    elif defect=='wavelet':recipe['wavelet_protocol']='uniform8'
    elif defect=='resume':recipe['requires_same_recipe_on_resume']=False
    elif defect=='local':recipe['local_factor']=2.
    else:recipe['nlambda']=.01
    with pytest.raises(ValueError):ev.validate_checkpoint_recipe(value,1.,2.,'f'*64)


@pytest.fixture(params=[('audit',0,1.,1.),('all',2,1.,0.),('all',500,1.,1.),('all',500,2.,1.)])
def source_config_fixture(tmp_path,request):
    import yaml
    mode,steps,factor,t=request.param;run=tmp_path/'actual_run';run.mkdir()
    base=ev.load(ev.C/'runner.py','frequency_adam_config_fixture_C')
    config=base.build_config(mode,steps,run,local_factor=1.,nlambda=0.,sigma_rel=.015)
    backend=config.setdefault('mode',{})
    config['campaign_overrides']['runtime_backend']={'mode.strict_cuda':{
        'source_present':'strict_cuda' in backend,'source':backend.get('strict_cuda'),'effective':True}}
    backend['strict_cuda']=True
    planned=dict(geometry_factor=factor,geometry_groups=list(ev.NAMES[:3]),density_lr_factor=1.,betas=[.9,.999],
        apply_after_strict_restore_iteration=4000,moment_policy='retain source Adam moments and step4000; no reset',
        scheduler_policy='retain clock; scale three current/initial/base/last LRs coherently')
    config['optimizer_overrides']=planned;config['campaign_overrides']['geometry_restart']=planned
    config['frequency_overrides']=dict(frequency_t=t,mask_source='original acquired mask_p/mask_c; never fit masks',
        utility_sha256='a150298dd8de4fcaa4ab77d7b60e00c79c8f6b1da8ad5448c3fe3c7182806933')
    path=run/'actual_config.yaml';path.write_text(yaml.safe_dump(config))
    return run,path,config,mode,steps,factor,t


def test_actual_control_and_formal_configs_are_independently_rebuilt(source_config_fixture):
    run,path,config,mode,steps,factor,t=source_config_fixture
    assert config['loss']['tv_weight']==.1 and config['loss']['lambda_ptv']==.03
    assert config['loss']['lambda_edge']==.001 and config['loss']['lambda_wav_plain']==.12
    assert ev.validate_actual_config(run,mode,steps,factor,t)['sha256']==digest(path)


@pytest.mark.parametrize('defect',['frequency','geometry','phase','edge','wavelet','utility','mask_source','strict'])
def test_configs_cannot_hide_changed_source_priors_or_control_frequency(source_config_fixture,defect):
    run,path,config,mode,steps,factor,t=source_config_fixture;import yaml
    if defect=='frequency':config['frequency_overrides']['frequency_t']=1.-t
    elif defect=='geometry':config['optimizer_overrides']['geometry_factor']=4.
    elif defect=='phase':config['loss']['lambda_ptv']=.06
    elif defect=='edge':config['loss']['lambda_edge']=.002
    elif defect=='wavelet':config['loss']['lambda_wav_plain']=.24
    elif defect=='utility':config['frequency_overrides']['utility_sha256']='wrong'
    elif defect=='mask_source':config['frequency_overrides']['mask_source']='fit masks'
    else:config['mode']['strict_cuda']=False
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):ev.validate_actual_config(run,mode,steps,factor,t)


def test_missing_reference_freeze_never_calls_create_or_writes(tmp_path,monkeypatch):
    target = tmp_path/"missing_reference_freeze.json"
    monkeypatch.setattr(ev,"REFERENCE_FREEZE",target)
    def forbid(*args,**kwargs): raise AssertionError("create-if-missing helper must never be called")
    monkeypatch.setattr(ev.report,"report_reference_freeze",forbid)
    with pytest.raises(ValueError,match="recreation is forbidden"):
        ev.existing_reference_freeze({}, {})
    assert not target.exists() and list(tmp_path.iterdir()) == []


def test_changed_reference_freeze_is_rejected_without_rewrite(tmp_path,monkeypatch):
    target = tmp_path/"reference_freeze.json"
    target.write_text(json.dumps({"created_before_formal_trials":True}))
    original = target.read_bytes()
    monkeypatch.setattr(ev,"REFERENCE_FREEZE",target)
    with pytest.raises(ValueError,match="missing/changed"):
        ev.existing_reference_freeze({}, {})
    assert target.read_bytes() == original


def test_existing_reference_freeze_uses_only_read_only_validator(tmp_path,monkeypatch):
    target = tmp_path/"reference_freeze.json"
    target.write_text(json.dumps({"created_before_formal_trials":True,"fixture":True}))
    original = target.read_bytes(); observed = []
    monkeypatch.setattr(ev,"REFERENCE_FREEZE",target)
    monkeypatch.setattr(ev,"REFERENCE_FREEZE_SHA",ev.sha(target))
    monkeypatch.setattr(ev.report,"_verify_reference_inputs",lambda *args:observed.append(args))
    monkeypatch.setattr(ev.report,"report_reference_freeze",lambda *a,**k:pytest.fail("writer called"))
    frozen = ev.existing_reference_freeze({"manifest":"fixture"},{"reference":"fixture"})
    assert frozen["fixture"] is True and len(observed) == 1 and target.read_bytes() == original


def test_nonpreformal_reference_freeze_cannot_be_relabelled(tmp_path,monkeypatch):
    target = tmp_path/"reference_freeze.json"
    target.write_text(json.dumps({"created_before_formal_trials":False}))
    monkeypatch.setattr(ev,"REFERENCE_FREEZE",target)
    monkeypatch.setattr(ev,"REFERENCE_FREEZE_SHA",ev.sha(target))
    with pytest.raises(ValueError,match="not preformal"):
        ev.existing_reference_freeze({}, {})


def test_runtime_guard_rejects_wrong_interpreter(monkeypatch):
    monkeypatch.setattr(sys,"executable","/tmp/not_GSTMI/python")
    with pytest.raises(ValueError,match="explicit GSTMI"):
        ev.numerical_runtime()


@pytest.mark.parametrize("action",["score","aggregate","original_shifts"])
def test_actual_NVGEN_refused_before_numerical_work_or_outputs(tmp_path,action):
    # The actual installed orchestration interpreter is intentionally not a
    # numerical backend; no GPU allocation or reconstruction data is read.
    wrong_python = Path("/home/congcongliu/anaconda3/envs/nvgen/bin/python")
    if action == "score":
        tail = [str(Path(ev.__file__)),"score","--exp","fixture_missing","--geometry-factor","2","--selected-recipe",str(tmp_path/"missing_recipe.json"),"--output",str(tmp_path/"out")]
    elif action == "aggregate":
        tail = [str(Path(ev.__file__)),"aggregate","--trial",str(tmp_path/"a"),"--trial",str(tmp_path/"b"),
                "--selected-recipe",str(tmp_path/"missing_recipe.json"),"--output",str(tmp_path/"out")]
    else:
        source = "import importlib.util; s=importlib.util.spec_from_file_location('wrong_runtime_probe',"+repr(str(Path(ev.__file__)))+"); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.original_shifts()"
        tail = ["-c",source]
    result = subprocess.run([str(wrong_python),"-X","utf8","-B",*tail],capture_output=True,text=True,
        env={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1","OMP_NUM_THREADS":"1"})
    assert result.returncode != 0 and "explicit GSTMI" in result.stderr
    assert not (tmp_path/"out").exists()


def utility():return ev

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


@pytest.mark.parametrize("t",[0.,1.])
def test_independent_frequency_array_validator_recomputes_both_scan_formulas(tmp_path,t):
    e=utility();value,sources,masks=fixture_arrays(tmp_path,t)
    result=e.validate_frequency_arrays(value,tmp_path,t,sources,masks)
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
