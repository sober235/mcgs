#!/usr/bin/env python3
"""Stdlib supervisor for one frozen H14500 image proximal-DC pilot."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parent
R=ROOT.parent
VERIFY=ROOT/'verification'
MODEL_DEFINITION=ROOT/'evidence/h1_definition_v1.json'
CURRENT_INCUMBENT=R/'development_incumbent.json'
PYTHON=Path('/home/congcongliu/anaconda3/envs/GSTMI/bin/python')
CPU_PYTHON=Path('/home/congcongliu/anaconda3/envs/nvgen/bin/python')
PRODUCTION=('physics.py','runner.py','run_gpu.py')
GROUPS=('positions','scales','rotations','density_real','density_imag','density_real_contrast','density_imag_contrast')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


# These functions keep their reviewed module globals. No rebinding of old ROOT.
_frozen=load(R/'phase_firstorder/run_gpu.py','_proximal_frozen_supervisor_utilities')
read,require,sha256,write_new=_frozen.read,_frozen.require,_frozen.sha256,_frozen.write_new
claim_lock,initial_visibility,process_start=_frozen.claim_lock,_frozen.initial_visibility,_frozen.process_start
load_campaign,select_resource,wait_owned_worker=_frozen.load_campaign,_frozen.select_resource,_frozen.wait_owned_worker
CLAIM_LOCK=_frozen.CLAIM_LOCK


def parent_runtime():
    require(not any(name in sys.modules for name in ('torch','numpy','frequency_weights')),'supervisor must remain stdlib-only')
    require(Path(sys.executable).resolve()==CPU_PYTHON.resolve() and sys.version_info[:3]==(3,11,15),
            'supervisor requires actual NVGEN Python3.11.15')
    return dict(python=sys.executable,python_version=list(sys.version_info[:3]),torch_imported=False,numpy_imported=False)


def stage_profile(stage):
    require(stage in ('tiny','canonical','prox'),'invalid proximal stage')
    return dict(method_variant='mcgs_prox_dc',method_label='MC-GS+proxDC',source_model='adamflat_g1_v1',source_iteration=4500,
        frequency_t=1.,geometry_factor=1.,local_factor=1.,lambda_ratio=.1,cg_max_matvecs=32,cg_rtol=1e-5,
        terminal_normal_evaluations=1,training_updates=0,working_dtype='complex64',
        true_residual_tolerance=2e-5,J_relative_tolerance=2e-6,J_absolute_tolerance=1e-10)


def validate_profile(value,stage):
    require(all(value.get(k)==v and (type(v) not in (int,float) or type(value.get(k)) in (int,float))
                for k,v in stage_profile(stage).items()),'fixed H14500 proximal profile differs')


def source_hashes():return {str(ROOT/name):sha256(ROOT/name) for name in PRODUCTION}


def dependency_hashes():
    paths=[R/'phase_firstorder'/name for name in _frozen.PRODUCTION]
    paths += [R/'frequency_adam'/name for name in ('runner.py','run_gpu.py','overrides.py','frequency_weights.py')]
    return {str(path):sha256(path) for path in paths}


def descriptor(path):return dict(path=str(Path(path).resolve()),sha256=sha256(path))


def model_definition(path=MODEL_DEFINITION,*,old_invariants=None):
    path=Path(path).resolve()
    require(path==MODEL_DEFINITION.resolve(),'only the independently fixed H1 model definition is allowed')
    before=sha256(path);value=read(path)
    require(all(value.get(k)==v for k,v in dict(version=1,source_model='adamflat_g1_v1',source_iteration=4500,
            frequency_t=1.,geometry_factor=1.,local_factor=1.,nlambda=0.,no_oracle_inputs=True).items()),'fixed accepted H14500 definition differs')
    require(value.get('source_run')==str(R/'frequency_adam/outputs/adamflat_g1_v1'),'source model run differs')
    files=value.get('files',{})
    require(isinstance(files,dict) and bool(files) and all(Path(p).is_absolute() for p in files),'H1 absolute file definition required')
    observed_files={p:sha256(p) for p in files}
    require(observed_files==files,
            'accepted H1 model/config/lineage/array bytes changed')
    for key in ('checkpoint','actual_config','lineage_completed'):
        d=value[key];require(files.get(d['path'])==d['sha256'],'model descriptor is outside its fixed file definition')
    recipe=value.get('frequency_adam_recipe',{})
    require(all(recipe.get(k)==v for k,v in dict(version=1,frequency_t=1.,geometry_factor=1.,local_factor=1.,nlambda=0.,
            wavelet_protocol='original_single_phase',requires_same_recipe_on_resume=True).items())
            and recipe.get('frequency_manifest_sha256') in files.values(),'H1 frequency checkpoint recipe differs')
    validation=value.get('checkpoint_validation',{});state=value.get('expected_model_state',{})
    shapes=validation.get('model_shapes',{});params=state.get('parameter_sha256',{});adam=state.get('adam',{})
    require(validation.get('iter')==4500 and validation.get('all_seven_model_and_moment_hashes_match') is True
            and set(shapes)==set(params)==set(adam)==set(GROUPS)
            and all(adam[g].get('step')==4500 and all(adam[g].get(k) for k in ('exp_avg_sha256','exp_avg_sq_sha256')) for g in GROUPS),
            'all seven actual H14500 model/moment clocks required')
    optimizer=value.get('expected_optimizer_actual',{});groups=optimizer.get('groups',[]);scheduler=optimizer.get('scheduler',{})
    require([g.get('name') for g in groups]==list(GROUPS) and all(g.get('step')==4500 for g in groups)
            and scheduler.get('last_epoch')==4500 and scheduler.get('step_count')==4501
            and state.get('geometry_restart')==optimizer,'ordinary H1 optimizer/scheduler declaration differs')
    gate=read(value['lineage_completed']['path'])
    require(gate.get('passed') is True and gate.get('status')=='complete' and gate.get('stage')=='trial500'
            and gate.get('returncode')==0 and gate.get('frequency_t')==gate.get('geometry_factor')==1.
            and all(files.get(p)==h for p,h in gate.get('code_sha256',{}).items())
            and bool(gate.get('artifacts')) and all(files.get(p)==h for p,h in gate['artifacts'].items()),'actual H1 completion lineage differs')
    proof_path=gate['post_run_invariants'];require(files.get(proof_path)==gate['post_run_invariants_sha256'],'H1 lineage proof is not bound')
    proof=read(proof_path);expected=dict(code_sha256=gate['code_sha256'],invariants=gate['invariants'])
    require(proof.get('status')=='verified' and proof.get('before')==proof.get('after')==expected,
            'H1 historical source/input proof differs')
    historical=gate['invariants'];references=historical.get('numerical_reference_artifacts',{})
    require(isinstance(references,dict) and bool(references),'historical H1 numerical references required')
    observed_references={p:sha256(p) for p in references}
    require(observed_references==references,
            'historical H1 numerical reference artifact bytes changed')
    if old_invariants is not None:
        require(all(historical.get(k)==v for k,v in old_invariants.items())
                and set(historical)-set(old_invariants)=={'numerical_reference_artifacts'},'H1 original input/source lineage differs')
    history=value['accepted_history_anchor'];require(sha256(history['path'])==history['sha256'],'fixed H1 accepted-history bytes changed')
    rows=[row for row in read(history['path']).get('trials',[]) if row.get('name')=='adamflat_g1_v1']
    require(len(rows)==1 and rows[0]==history['model_trial'] and rows[0].get('accepted') is True
            and rows[0].get('method_variant')=='frequency_adam' and rows[0].get('strict_success') is False,
            'fixed H1 model is not the actual accepted ordinary trial')
    require(set(value.get('scans',{}))=={'pre','post'},'both H1 reporting gauges required')
    for row in value['scans'].values():
        require(all(files.get(row[k]['path'])==row[k]['sha256'] for k in ('canonical_array','score_reference'))
                and all(type(row.get(k)) in (int,float) and math.isfinite(row[k]) and row[k]>0
                        for k in ('scale_trainer','scale_gauge','factor_scale'))
                and math.isclose(row['factor_scale'],row['scale_trainer']/row['scale_gauge'],rel_tol=1e-15),
                'published H1 canonical arrays or fixed reporting gauge differ')
    require(sha256(path)==before,'H1 definition changed during verification')
    return dict(definition_manifest=dict(path=str(path),sha256=before),definition=value,files=observed_files,
        lineage_reference_artifacts=observed_references)


def snapshot_invariants(model_definition_path=MODEL_DEFINITION):
    old=_frozen.snapshot_invariants()
    return {**old,'accepted_h1':model_definition(model_definition_path,old_invariants=old),
        'dependency_sources':dependency_hashes(),'scoring_sources':{str(ROOT/'scoring/evaluate.py'):sha256(ROOT/'scoring/evaluate.py')}}


def build_review_gate(report_path,*,actionable_findings,unresolved_findings,round):
    require(type(actionable_findings) is type(unresolved_findings) is int and actionable_findings==unresolved_findings==0
            and type(round) is int and 1<=round<=3,'independent review is not clean/bounded')
    path=Path(report_path).resolve();require(path.is_file(),'actual independent report required')
    return dict(status='pass',round=round,actionable_findings=0,unresolved_findings=0,
        code_sha256=source_hashes(),invariants=snapshot_invariants(),review_report=str(path),review_report_sha256=sha256(path))


def validate_review(path,hashes,invariants):
    gate=read(path)
    require(gate.get('status')=='pass' and gate.get('actionable_findings')==gate.get('unresolved_findings')==0
            and type(gate.get('round')) is int and 1<=gate['round']<=3
            and gate.get('code_sha256')==hashes and gate.get('invariants')==invariants
            and sha256(gate['review_report'])==gate.get('review_report_sha256'),'fresh clean review with full current invariants required')
    return gate


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--stage',choices=('tiny','canonical','prox'),required=True);p.add_argument('--exp',required=True)
    p.add_argument('--review-gate',type=Path,required=True);p.add_argument('--cuda-operator-gate',type=Path)
    p.add_argument('--canonical-control-gate',type=Path);p.add_argument('--selected-recipe',type=Path)
    p.add_argument('--model-definition',type=Path,default=MODEL_DEFINITION);p.add_argument('--preferred-gpu',type=int,default=5)
    p.add_argument('--reserved-gpu',type=int,help=argparse.SUPPRESS);p.add_argument('--reservation-token',help=argparse.SUPPRESS)
    p.add_argument('--claim-fd',type=int,help=argparse.SUPPRESS);return p


def request(args):
    names=('stage','exp','review_gate','cuda_operator_gate','canonical_control_gate','selected_recipe','model_definition')
    return {**{k:str(v.resolve()) if isinstance(v,Path) else v for k in names for v in (getattr(args,k),)},
        'output':str((ROOT/'outputs'/args.exp).resolve()),**stage_profile(args.stage)}


def args_from_request(value):
    args=parser().parse_args(['--stage',value['stage'],'--exp',value['exp'],'--review-gate',value['review_gate']])
    for key in ('cuda_operator_gate','canonical_control_gate','selected_recipe','model_definition'):
        setattr(args,key,Path(value[key]) if value.get(key) else None)
    require(request(args)==value,'launch request contains changed or additional protocol fields');return args


def control_bindings(args):
    files=[getattr(args,k) for k in ('review_gate','cuda_operator_gate','canonical_control_gate','selected_recipe','model_definition')
           if getattr(args,k) is not None]
    if args.stage=='prox':files += [CURRENT_INCUMBENT,Path(read(CURRENT_INCUMBENT)['validated_summary'])]
    return {str(Path(p).resolve()):sha256(p) for p in files}


def validate_history(recipe):
    """Root's latest completed history is a quality anchor, never a training input."""
    prior=recipe['completed_prior_summary'];incumbent=recipe['incumbent']
    path=Path(prior['path']);current_path=Path(incumbent['path'])
    require(path.is_absolute() and path.resolve().is_relative_to(R.resolve()) and current_path.resolve()==CURRENT_INCUMBENT.resolve()
            and sha256(path)==prior['sha256'] and sha256(current_path)==incumbent['sha256'],'latest summary/incumbent bytes differ')
    previous,current=read(path),read(current_path);rows=previous.get('trials',[])
    require(previous.get('status')=='completed_real_trial_snapshot' and bool(rows)
            and len({r['name'] for r in rows})==len(rows) and previous['status_counts']['completed']==len(rows)
            and all(type(r.get('score')) in (int,float) and math.isfinite(r['score']) and type(r.get('accepted')) is bool for r in rows),
            'actual completed history is incomplete; no fixed trial count is assumed')
    q=[r for r in rows if r.get('method_variant')=='inner_precision_mm'];qf=previous.get('inner_precision_family',{})
    hp=dict(frequency_t=1.,cg_steps=96,true_residual_diagnostics=True)
    require(len(q)==1 and qf.get('completed_candidates')==[r['name'] for r in q]
            and all(qf.get(k)==v and all(r.get(k)==v for r in q) for k,v in hp.items()),'completed Q96 family required')
    h=[r for r in rows if r.get('method_variant')=='frequency_adam'];hf=previous.get('frequency_adam_family',{})
    require([r.get('geometry_factor') for r in h]==[1.,2.] and all(r.get('frequency_t')==1. for r in h)
            and hf.get('frequency_t')==1. and hf.get('declared_geometry_factors')==hf.get('completed_geometry_factors')==[1.,2.]
            and h[0]['name']=='adamflat_g1_v1' and h[0].get('accepted') is True,'actual accepted H1 and completed ordered H1/H2 required')
    low=[r for r in rows if r.get('method_variant')=='frequency_lowreg_mm'];lf=previous.get('frequency_lowreg_family',{})
    require([r.get('local_factor') for r in low]==[.75,.5]
            and lf.get('declared_local_factors')==lf.get('completed_local_factors')==[.75,.5]
            and lf.get('completed_candidates')==[r['name'] for r in low]
            and all(lf.get(k)==v and all(r.get(k)==v for r in low) for k,v in hp.items()),'completed ordered L family required')
    pf=[r for r in rows if r.get('method_variant')=='phase_firstorder'];pff=previous.get('phase_firstorder_family',{})
    phase={**hp,'phase_curvature_mode':'first_order','damping_curvature_mode':'full',
        'update_class':'phase_first_order_quasi_newton','full_MM_majorization':False}
    require(len(pf)==1 and pf[0].get('local_factor')==1.
            and pff.get('declared_local_factors')==pff.get('completed_local_factors')==[1.]
            and pff.get('completed_candidates')==[r['name'] for r in pf]
            and all(pff.get(k)==v and all(r.get(k)==v for r in pf) for k,v in phase.items())
            and pff.get('full_MM_majorization') is False and pf[0].get('full_MM_majorization') is False,'completed actual PF family required')
    require(not any(r.get('method_variant')=='mcgs_prox_dc' for r in rows),'this one fixed proximal candidate was already completed')
    require(all(r.get('strict_success') is False for r in (*q,*h,*low,*pf)) and current.get('strict_success') is False,
            'strict win takes priority: repeat and matched improved GS-single first')
    score=current.get('score');accepted=[r['score'] for r in rows if r['accepted']]
    require(type(score) in (int,float) and math.isfinite(score) and score==incumbent.get('score') and bool(accepted)
            and max(accepted)==score and current.get('validated_summary')==str(path)
            and current.get('validated_summary_sha256')==prior['sha256'],'latest global quality anchor differs')
    return previous


def build_recipe():
    """Return a reviewable root recipe; never publish it or select a model/lambda."""
    current=read(CURRENT_INCUMBENT);summary=Path(current['validated_summary']);I=snapshot_invariants()
    value=dict(status='selected',selected_by='root',**stage_profile('prox'),minimum_score_improvement=1e-4,
        code_sha256=source_hashes(),invariants=I,model_definition=I['accepted_h1']['definition_manifest'],
        completed_prior_summary=descriptor(summary),incumbent={**descriptor(CURRENT_INCUMBENT),'score':current['score']})
    validate_history(value);return value


def validate_recipe(path,hashes,invariants):
    recipe=read(path);validate_profile(recipe,'prox')
    require(recipe.get('status')=='selected' and recipe.get('selected_by')=='root'
            and recipe.get('minimum_score_improvement')==1e-4 and recipe.get('code_sha256')==hashes
            and recipe.get('invariants')==invariants
            and recipe.get('model_definition')==invariants['accepted_h1']['definition_manifest'],'root recipe source/model/full-I differs')
    validate_history(recipe);return recipe


def validate_stage_gates(args,hashes,invariants):
    require(args.exp not in ('','.','..') and Path(args.exp).name==args.exp,'exp must be one fresh directory name')
    require(args.preferred_gpu>=0 and (args.reserved_gpu is None or args.reserved_gpu>=0),'invalid GPU index')
    require(args.model_definition.resolve()==MODEL_DEFINITION.resolve(),'fixed H1 model definition required')
    review=validate_review(args.review_gate,hashes,invariants)
    if args.stage!='tiny':
        require(args.cuda_operator_gate is not None,'actual tiny CUDA gate is required')
        validate_observed_gate(args.cuda_operator_gate,'tiny',hashes,invariants)
    if args.stage=='prox':
        require(args.canonical_control_gate is not None,'actual calibrated canonical gate is required')
        validate_observed_gate(args.canonical_control_gate,'canonical',hashes,invariants)
        require(args.selected_recipe is not None,'root selected current single proximal recipe required')
        validate_recipe(args.selected_recipe,hashes,invariants)
    else:require(args.selected_recipe is None,'controls do not select or fit a candidate recipe')
    return review


def command(args,launch_path):
    return [str(PYTHON),'-X','utf8','-B','-u',str(ROOT/'runner.py'),'--stage',args.stage,'--exp',args.exp,
        '--output',str(ROOT/'outputs'),'--launch-manifest',str(Path(launch_path).resolve()),
        '--review-gate',str(args.review_gate.resolve()),'--model-definition',str(args.model_definition.resolve())]


PHYSICS_COUNTERS=('rhs_adjoint_calls','normal_calls','terminal_normal_calls','objective_calls','forward_calls','adjoint_calls',
    'fft_batches','ifft_batches','coil_fft_transforms','coil_ifft_transforms','gaussian_renders','gaussian_vjps')


def validate_scan(record,work):
    """Stdlib scalar/work guard; scoring also reads frozen CSMs independently."""
    def finite(key,nonnegative=False):
        v=record.get(key);require(type(v) in (int,float) and math.isfinite(v) and (not nonnegative or v>=0),f'invalid {key}');return v
    def integer(key,minimum=0):
        v=record.get(key);require(type(v) is int and v>=minimum,f'invalid {key}');return v
    def close(a,b):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-30)
    L=finite('L');requested=finite('lambda_requested');effective=finite('lambda_effective')
    require(record.get('dtype')=='torch.complex64' and record.get('lambda_ratio')==.1 and L>0
            and requested==.1*L and effective==struct.unpack('f',struct.pack('f',requested))[0] and effective>0,
            'native CSM-only lambda differs')
    integer('support_count',1);coils=integer('n_coils',1);chunk=integer('chunk_size',1)
    require(chunk<=4 and type(record.get('initial_anchor_nonzero')) is bool,'coil chunk/initial anchor evidence differs')
    initial=integer('initial_residual_normal_evaluations');iterations=integer('cg_iterations');normal=integer('cg_matvecs')
    require(initial==int(record['initial_anchor_nonzero']) and normal==iterations+initial and normal<=32
            and record.get('cg_max_matvecs')==32 and record.get('cg_rtol')==1e-5,'CG budget includes initial residual')
    require(record.get('cg_converged') is True and record.get('cg_status')=='converged'
            and record.get('status')==('stationary' if iterations==0 else 'converged'),'failed/exhausted solve is not a completed pilot')
    rhs=finite('rhs_norm',True);norm=finite('true_residual_norm',True);relative=finite('true_relative_residual',True)
    history=record.get('cg_residual_history')
    require(isinstance(history,list) and len(history)==iterations+1
            and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in history) and history[-1]<=1e-5*rhs,
            'CG convergence history/RHS stopping differs')
    require(close(relative,norm/max(rhs,1e-30)) and relative<=2e-5 and record.get('true_residual_tolerance')==2e-5
            and record.get('true_endpoint_passed') is True,'true endpoint residual failed')
    ja,jf,da,df,p=[finite(k,True) for k in ('J_anchor','J_after','data_anchor','data_after','anchor_penalty')]
    tolerance=finite('J_check_tolerance',True);slope=finite('directional_derivative')
    require(record.get('J_relative_tolerance')==2e-6 and record.get('J_absolute_tolerance')==1e-10
            and close(tolerance,1e-10+2e-6*max(abs(ja),1e-30)) and close(ja,da) and close(jf,df+p)
            and jf<=ja+tolerance and record.get('J_check_passed') is True,'fixed J arithmetic/nonincrease differs')
    require((iterations==0 and slope==0) or (iterations>0 and slope<0),'nonstationary direction is not descent')
    require(record.get('outside_support_exact_zero') is True and record.get('terminal_normal_evaluations')==1
            and record.get('training_updates')==0,'support/terminal/zero-training evidence differs')
    chunks=(coils+chunk-1)//chunk
    expected=dict(rhs_adjoint_calls=1,normal_calls=normal,terminal_normal_calls=1,objective_calls=2,forward_calls=0,adjoint_calls=1,
        fft_batches=chunks*(normal+3),ifft_batches=chunks*(normal+2),coil_fft_transforms=coils*(normal+3),
        coil_ifft_transforms=coils*(normal+2),gaussian_renders=0,gaussian_vjps=0)
    require(set(work)==set(expected)|{'roles'} and all(type(work[k]) is int and work[k]==v for k,v in expected.items()),'proximal physics work differs')
    roles=dict(rhs=dict(fft_batches=0,ifft_batches=chunks),terminal=dict(fft_batches=chunks,ifft_batches=chunks),
        objective_anchor=dict(fft_batches=chunks,ifft_batches=0),objective_candidate=dict(fft_batches=chunks,ifft_batches=0))
    if initial:roles['initial_normal']=dict(fft_batches=chunks,ifft_batches=chunks)
    if iterations:roles['cg_normal']=dict(fft_batches=chunks*iterations,ifft_batches=chunks*iterations)
    require(isinstance(work['roles'],dict) and {k:v for k,v in work['roles'].items() if any(v.values())}==roles,'physics FFT role counts differ')
    return work


def validate_numeric_runtime(value):
    require(Path(value['python']).resolve()==PYTHON.resolve() and value.get('torch_version')=='2.7.1+cu118'
            and value.get('cuda_version')=='11.8' and value.get('python_version',[])[:2]==[3,11],
            'actual GSTMI Torch2.7.1+cu118/CUDA11.8 numerical runtime required')


def validate_states(value,stage,definition):
    before,after=value['source_state_before'],value['source_state_after']
    fields=('parameter_sha256','parameter_identities','requires_grad','gradient_sha256','gradient_identities','adam',
        'adam_identities','optimizer_groups','scheduler_state','config_sha256','model_sha256','geometry_sha256','density_sha256')
    require(all(k in before and before[k]==after.get(k) for k in fields),'source model/gradient/Adam/scheduler/config mutated')
    require(set(before['parameter_sha256'])==set(before['parameter_identities'])==set(GROUPS)
            and len({str(v) for v in before['parameter_identities'].values()})==7,'seven distinct original parameter identities required')
    expected_checks={'model_parameters_unchanged','parameter_identities_unchanged','gradients_unchanged','adam_unchanged',
        'adam_identities_unchanged','scheduler_unchanged','config_unchanged','no_training_steps','strict_cuda_prebuilt'}
    if stage!='tiny':expected_checks |= {'source_checkpoint_matches_H1','source_recipe_verified','fit_equals_acquired','canonical_complex_matches_H1'}
    require(expected_checks<=set(value.get('checks',{})) and all(value['checks'][k] is True for k in expected_checks),
            'measured source/restore/backend checks incomplete')
    require(value.get('forbidden_calls')==dict(trainer_train=0,optimizer_step=0,scheduler_step=0)
            and all(type(v) is int for v in value['forbidden_calls'].values()),'training/optimizer/scheduler calls are forbidden')
    if stage=='tiny':
        require(value.get('source_restore_performed') is False and value.get('restored_iteration') is None,
                'synthetic tiny must not claim H1 restore')
    else:
        require(value.get('source_restore_performed') is True and value.get('restored_iteration')==4500,'actual H14500 restore required')
        source=definition['expected_model_state']
        require(all(before[k]==source[k] for k in ('parameter_sha256','adam','model_sha256','geometry_sha256','density_sha256')),
                'restored seven model/moment hashes differ from accepted H1')
        expected=definition['expected_optimizer_actual'];groups=before['optimizer_groups'];scheduler=before['scheduler_state']
        require(len(groups)==7 and all(g.get('param_names')==[want['name']]
                and all(g.get(k)==v for k,v in want.items() if k not in ('step','name')) for g,want in zip(groups,expected['groups']))
                and scheduler.get('last_epoch')==4500 and scheduler.get('_step_count')==4501
                and scheduler.get('gamma')==expected['scheduler']['gamma']
                and scheduler.get('base_lrs')==expected['scheduler']['base_lrs']
                and scheduler.get('_last_lr')==expected['scheduler']['last_lrs'],'actual ordinary seven-group optimizer/scheduler differs')
        restore=value['restore_proof']
        require(restore.get('iteration')==4500 and restore.get('checkpoint')==definition['checkpoint']
                and restore.get('frequency_adam_recipe')==definition['frequency_adam_recipe']
                and restore.get('optimizer_actual')==definition['expected_optimizer_actual']
                and restore.get('parameter_identities_before')==restore.get('parameter_identities_after')==before['parameter_identities'],
                'public4500 restore/checkpoint recipe/parameter identity proof differs')


def validate_work_total(value,stage):
    scans=value['scans'];total=value['work_total']
    physics={k:sum(row['work'][k] for row in scans.values()) for k in PHYSICS_COUNTERS}
    identity={k:0 for k in PHYSICS_COUNTERS}
    if stage=='tiny':
        tiny=value['tiny_checks'];require(tiny.get('tolerance')==2e-5 and tiny.get('parameters_unchanged') is True,'actual tiny operator proof required')
        require(set(tiny.get('source_raw_reference_relative',{}))==set(tiny.get('physics',{}))=={'pre','post'},'both tiny source/physics scans required')
        for scan in ('pre','post'):
            error=tiny['source_raw_reference_relative'][scan];row=tiny['physics'][scan]
            require(type(error) in (int,float) and math.isfinite(error) and 0<=error<=2e-5,'CUDA/source Torch raw parity failed')
            for key in ('forward_adjoint_relative','normal_hermitian_relative','normal_linearity_relative'):
                v=row.get(key);require(type(v) in (int,float) and math.isfinite(v) and 0<=v<=2e-5,'tiny physics identity failed')
            require(type(row.get('normal_curvature_real')) in (int,float) and math.isfinite(row['normal_curvature_real'])
                    and row['normal_curvature_real']>0,'tiny normal positive curvature failed')
            w=row['work'];coils=scans[scan]['n_coils'];chunk=scans[scan]['record']['chunk_size'];batches=(coils+chunk-1)//chunk
            expected=dict(rhs_adjoint_calls=0,normal_calls=3,terminal_normal_calls=0,objective_calls=0,forward_calls=1,adjoint_calls=1,
                fft_batches=4*batches,ifft_batches=4*batches,coil_fft_transforms=4*coils,coil_ifft_transforms=4*coils,
                gaussian_renders=0,gaussian_vjps=0)
            require(all(type(w.get(k)) is int and w[k]==v for k,v in expected.items()),'tiny identity work is hidden or mixed with solver work')
            for k in identity:identity[k]+=w[k]
    coils=sum(row['n_coils'] for row in scans.values());reference=2 if stage=='tiny' else 0
    expected=dict(gaussian_renders=2+reference,gaussian_cuda_renders=2,gaussian_reference_renders=reference,gaussian_vjps=0,
        canonical_export_calls=2,canonical_fft_batches=2,canonical_ifft_batches=4,
        canonical_coil_fft_transforms=coils,canonical_coil_ifft_transforms=2*coils,
        total_fft_batches=2+physics['fft_batches']+identity['fft_batches'],
        total_ifft_batches=4+physics['ifft_batches']+identity['ifft_batches'],
        total_coil_fft_transforms=coils+physics['coil_fft_transforms']+identity['coil_fft_transforms'],
        total_coil_ifft_transforms=2*coils+physics['coil_ifft_transforms']+identity['coil_ifft_transforms'])
    require(set(total)==set(expected)|{'physics','tiny_identity_physics'}
            and all(type(total[k]) is int and total[k]==v for k,v in expected.items())
            and total['physics']==physics and total['tiny_identity_physics']==identity,'observed Gaussian/control/physics total counters differ')


def validate_launch(path,hashes,invariants):
    launch=read(path);stage=launch['stage'];validate_profile(launch,stage)
    require(launch.get('code_sha256')==hashes and launch.get('invariants')==invariants,'launch current source/full-I differs')
    args=args_from_request(launch['request']);validate_stage_gates(args,hashes,invariants)
    require(Path(path).resolve()==(VERIFY/f'{args.exp}_launch.json').resolve() and stage==args.stage
            and launch.get('command')==command(args,path) and launch.get('control_bindings')==control_bindings(args),
            'launch command/request/control bindings differ')
    runtime=launch['parent_runtime']
    require(Path(runtime['python']).resolve()==CPU_PYTHON.resolve() and runtime.get('python_version')==[3,11,15]
            and runtime.get('torch_imported') is False and runtime.get('numpy_imported') is False,'parent was not actual stdlib NVGEN')
    require(type(launch.get('parent_pid')) is int and launch['parent_pid']>0
            and launch.get('initial_visibility')==str(launch['resource']['gpu']),'actual reserved GPU visibility differs')
    return launch,args


def validate_child_artifacts(stage,directory,hashes,invariants,*,require_calibration=True,calibration_proof=None):
    directory=Path(directory).resolve();exp=directory.name
    require(directory==(ROOT/'outputs'/exp).resolve() and exp not in ('','.','..'),'child artifacts outside owned output')
    launch_path=VERIFY/f'{exp}_launch.json';launch,args=validate_launch(launch_path,hashes,invariants)
    require(args.stage==stage,'child stage differs from launch')
    value,metrics=read(directory/'provenance.json'),read(directory/'metrics.json')
    same=('stage','exp','code_sha256','invariants','launch_sha256','model_definition','control_bindings','numerical_runtime','device','backend',
        'source_restore_performed','restored_iteration','source_state_before','source_state_after','checks','forbidden_calls','scans','arrays','work_total')
    require(value.get('status')==metrics.get('status')=='complete' and all(k in value and value[k]==metrics.get(k) for k in same),
            'complete metrics/provenance disagree')
    if stage!='tiny':require(value.get('restore_proof')==metrics.get('restore_proof'),'metrics/provenance restore proofs disagree')
    else:require(value.get('tiny_checks')==metrics.get('tiny_checks'),'metrics/provenance actual tiny checks disagree')
    elapsed=value.get('total_elapsed_seconds')
    require(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=0 and metrics.get('total_elapsed_seconds')==elapsed,
            'actual nonnegative child runtime must match in metrics/provenance')
    for document in (value,metrics):validate_profile(document,stage)
    require(value['stage']==stage and value['exp']==exp and value['code_sha256']==hashes and value['invariants']==invariants
            and value['launch_sha256']==sha256(launch_path) and value['control_bindings']==launch['control_bindings']
            and value['model_definition']==invariants['accepted_h1']['definition_manifest'],'child immutable launch/model bindings differ')
    validate_numeric_runtime(value['numerical_runtime']);require(value['device']=='cuda:0','actual child must use reserved CUDA0')
    backend=value['backend']
    require(backend.get('prebuilt_extension')==invariants['prebuilt_extension'] and backend.get('backend')=='tile_cuda'
            and backend.get('strict_cuda') is True and backend.get('use_cuda') is True and backend.get('compiler_branch_bypassed') is True
            and backend.get('normal_tile_source')==str(R/'nonlocal_autoresearch/trainer/paired_gs/gaussian/tile_voxelizer.py'),
            'actual source strict prebuilt CUDA backend required')
    definition=invariants['accepted_h1']['definition'];validate_states(value,stage,definition)
    require(set(value.get('scans',{}))=={'pre','post'},'both independently reconstructed scans required')
    declared_arrays={}
    require(value.get('arrays')=={} if stage=='tiny' else set(value.get('arrays',{}))=={'pre','post'},'formal arrays in tiny or missing clinical arrays')
    for scan,row in value['scans'].items():
        require(type(row.get('n_coils')) is int and row['n_coils']>0,'actual source coil count required')
        if stage=='canonical':
            require(row.get('algorithm_executed') is False and row.get('record') is None
                    and row.get('work')==dict.fromkeys(PHYSICS_COUNTERS,0) and all(type(v) is int for v in row['work'].values()),
                    'canonical control must not fabricate CG records')
        else:
            require(row.get('algorithm_executed') is True and row['record']['n_coils']==row['n_coils'],'actual independent proximal solve missing')
            validate_scan(row['record'],row['work'])
        if stage=='tiny':continue
        gauge={k:definition['scans'][scan][k] for k in ('scale_trainer','scale_gauge','factor_scale')}
        require(row.get('gauge')==gauge,'fixed source-to-reporting gauge differs; no fitting allowed')
        check=row.get('canonical_check',{})
        require(check.get('tolerance')==1e-5 and type(check.get('relative_complex_l2')) in (int,float)
                and math.isfinite(check['relative_complex_l2']) and 0<=check['relative_complex_l2']<=1e-5
                and check.get('reference')==definition['scans'][scan]['canonical_array'],'published H1 canonical complex control failed')
        arrays=value['arrays'][scan];kinds={'raw','canonical','candidate'} if stage=='prox' else {'raw','canonical'}
        require(set(arrays)==kinds,'clinical exported array set differs')
        for kind in kinds:
            info=arrays[kind];path=directory/f'{kind}_{scan}_complex.npy'
            require(info.get('path')==str(path) and info.get('sha256')==sha256(path) and info.get('dtype')=='complex64'
                    and info.get('gauge')=='source_topic' and type(info.get('shape')) is list and len(info['shape'])==3
                    and all(type(n) is int and n>0 for n in info['shape']),'actual fixed-gauge array descriptor differs')
            declared_arrays[str(path)]=info['sha256']
    validate_work_total(value,stage)
    paths=[p for p in directory.rglob('*') if p.is_file()]
    require(all(p.suffix not in ('.pth','.pt','.npz') and (p.suffix!='.npy' or str(p) in declared_arrays) for p in paths),
            'checkpoint/model or undeclared array artifacts are forbidden')
    artifacts={str(p):sha256(p) for p in paths}
    if stage=='canonical' and require_calibration:
        proof=calibration_proof or descriptor(ROOT/'scoring'/f'calibration_{exp}'/'calibration.json')
        validate_calibration(proof,exp,hashes,invariants,artifacts)
    return artifacts


def validate_calibration(proof,exp,hashes,invariants,artifacts):
    path=Path(proof['path']);require(path.resolve().is_relative_to((ROOT/'scoring').resolve()) and sha256(path)==proof['sha256'],
            'canonical calibration proof bytes differ')
    value=read(path);validate_profile(value,'canonical');validate_numeric_runtime(value['numerical_runtime'])
    require(value.get('status')=='verified' and value.get('passed') is True and value.get('stage')=='canonical' and value.get('exp')==exp
            and value.get('code_sha256')==hashes and value.get('invariants')==invariants
            and value.get('bound_artifacts_sha256')==artifacts,'actual canonical calibration source/artifact bindings differ')
    rendered=VERIFY/f'{exp}_rendered.json'
    require(value.get('rendered_proof')==descriptor(rendered),'calibration does not bind actual raw rendered proof')
    raw=read(rendered);require(value.get('control_bindings')==raw.get('control_bindings'),'calibration control bindings differ')
    require(set(value.get('scans',{}))=={'pre','post'},'both canonical calibrations required')
    for row in value['scans'].values():
        require(row.get('complex_relative_l2_tolerance')==1e-5 and row.get('head_psnr_tolerance_db')==.001
                and row.get('amplitude_or_phase_fit') is False
                and all(type(row.get(k)) in (int,float) and math.isfinite(row[k]) and 0<=row[k]<=limit
                        for k,limit in (('complex_relative_l2',1e-5),('head_psnr_absolute_difference_db',.001))),
                'complex/head-PSNR canonical calibration failed')
    return value


def validate_observed_gate(path,stage,hashes,invariants):
    value=read(path);validate_profile(value,stage);exp=value['exp']
    require(Path(path).resolve()==(VERIFY/f'{exp}_completed.json').resolve() and value.get('stage')==stage
            and value.get('passed') is True and value.get('status')=='complete' and value.get('returncode')==0
            and type(value.get('worker_pid')) is int and value['worker_pid']>0 and value.get('code_sha256')==hashes
            and value.get('invariants')==invariants,'actual completed stage/source/full-I gate required')
    launch,args=validate_launch(VERIFY/f'{exp}_launch.json',hashes,invariants)
    require(value.get('request')==launch['request'] and value.get('control_bindings')==launch['control_bindings'],'gate launch/control request differs')
    proof_path=Path(value['post_run_invariants']);require(sha256(proof_path)==value['post_run_invariants_sha256'],'post-run invariant proof bytes changed')
    proof=read(proof_path);expected=dict(code_sha256=hashes,invariants=invariants);bound=value['control_bindings']
    require(proof.get('status')=='verified' and proof.get('before')==proof.get('after')==expected
            and proof.get('control_bindings_before')==proof.get('control_bindings_after')==bound
            and all(sha256(p)==h for p,h in bound.items()),'verified full before/after source/control proof differs')
    artifacts=validate_child_artifacts(stage,ROOT/'outputs'/exp,hashes,invariants,
        calibration_proof=value.get('calibration_proof'))
    require(value.get('artifacts')==artifacts,'completed stage artifacts changed');return value


def validate_rendered(exp,hashes,invariants):
    path=VERIFY/f'{exp}_rendered.json';value=read(path);validate_profile(value,'canonical')
    expected=dict(code_sha256=hashes,invariants=invariants)
    require(value.get('status')=='rendered' and value.get('stage')=='canonical' and value.get('exp')==exp
            and value.get('returncode')==0 and type(value.get('worker_pid')) is int and value['worker_pid']>0
            and value.get('before')==value.get('after')==expected and value.get('code_sha256')==hashes
            and value.get('invariants')==invariants,'actual raw canonical rendered proof required; no completed gate substitute')
    launch,args=validate_launch(VERIFY/f'{exp}_launch.json',hashes,invariants)
    require(value.get('launch')==descriptor(VERIFY/f'{exp}_launch.json') and value.get('request')==launch['request']
            and value.get('control_bindings')==control_bindings(args),'rendered proof launch/control bytes differ')
    artifacts=validate_child_artifacts('canonical',ROOT/'outputs'/exp,hashes,invariants,require_calibration=False)
    require(value.get('artifacts')==artifacts,'raw rendered artifacts changed');return value,artifacts


def validate_reservation(args,selection,hashes,invariants):
    require(args.reservation_token and selection.get('reservation_token')==args.reservation_token
            and selection.get('parent_pid')==os.getpid() and selection.get('process_start')==process_start(),
            'reservation is not owned by this exact CPU parent')
    require(selection.get('code_sha256')==hashes and selection.get('invariants')==invariants
            and selection.get('stage')==args.stage and selection.get('request')==request(args)
            and selection.get('control_bindings')==control_bindings(args),'reservation source/request/control changed')
    require(selection['resource']['gpu']==args.reserved_gpu
            and initial_visibility()==os.environ.get('CUDA_VISIBLE_DEVICES')==str(args.reserved_gpu),
            'reservation must exist in same PID parent initial visibility')
    require(args.claim_fd is not None and args.claim_fd>=3,'inherited shared claim lock required')
    actual,expected=os.fstat(args.claim_fd),CLAIM_LOCK.stat()
    require((actual.st_dev,actual.st_ino,actual.st_uid)==(expected.st_dev,expected.st_ino,os.getuid()),'claim lock descriptor differs')
    fcntl.flock(args.claim_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)


def invariant_proof(hashes,invariants,bound):
    proof=dict(status='checking',before=dict(code_sha256=hashes,invariants=invariants),after={},
        control_bindings_before=bound,control_bindings_after={})
    try:
        proof['after']=dict(code_sha256=source_hashes(),invariants=snapshot_invariants())
        proof['control_bindings_after']={p:sha256(p) for p in bound}
        require(proof['before']==proof['after'] and proof['control_bindings_after']==bound,'source/model/input/control changed during child')
        parent_runtime();proof['status']='verified'
    except BaseException as error:proof.update(status='failed',error=repr(error))
    return proof


def execute(args,full_argv):
    runtime=parent_runtime();hashes=source_hashes();invariants=snapshot_invariants(args.model_definition)
    validate_stage_gates(args,hashes,invariants);bound=control_bindings(args)
    require(not (ROOT/'outputs'/args.exp).exists(),'fresh child output required; preserve every previous run')
    campaign=load_campaign();selection_path=VERIFY/f'{args.exp}_selection.json'
    if args.reserved_gpu is None:
        require(args.claim_fd is None and args.reservation_token is None,'internal reservation options cannot be supplied initially')
        with claim_lock(CLAIM_LOCK) as fd:
            selected=select_resource(campaign,preferred=args.preferred_gpu,environment=os.environ)
            args.reservation_token=uuid.uuid4().hex
            write_new(selection_path,dict(parent_pid=os.getpid(),process_start=process_start(),reservation_token=args.reservation_token,
                timestamp=time.time(),stage=args.stage,**stage_profile(args.stage),request=request(args),resource=selected,
                code_sha256=hashes,invariants=invariants,control_bindings=bound))
            environment={**os.environ,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':str(selected['gpu'])}
            os.execve(sys.executable,[sys.executable,'-X','utf8','-B','-u',str(Path(__file__).resolve()),*full_argv,
                '--reserved-gpu',str(selected['gpu']),'--reservation-token',args.reservation_token,'--claim-fd',str(fd)],environment)
        raise RuntimeError('same-PID reexec unexpectedly returned')
    selection=read(selection_path);validate_reservation(args,selection,hashes,invariants)
    try:
        selected=select_resource(campaign,preferred=args.reserved_gpu,environment=os.environ)
        require(selected['gpu']==args.reserved_gpu,'reserved GPU changed')
        if selection['resource']['free']:require(selected['free'],'idle reserved GPU gained a workload; preserve it')
    finally:
        os.close(args.claim_fd);args.claim_fd=None
    launch_path=VERIFY/f'{args.exp}_launch.json';child_command=command(args,launch_path)
    launch=dict(timestamp=time.time(),parent_pid=os.getpid(),stage=args.stage,**stage_profile(args.stage),request=request(args),
        parent_runtime=runtime,reservation_token=args.reservation_token,initial_visibility=initial_visibility(),
        resource=selected,initial_resource=selection['resource'],command=child_command,code_sha256=hashes,
        invariants=invariants,control_bindings=bound)
    require(source_hashes()==hashes and snapshot_invariants(args.model_definition)==invariants
            and control_bindings(args)==bound,'source/model/input/control changed before numeric launch')
    write_new(launch_path,launch)
    environment={**os.environ,'PYTHONNOUSERSITE':'1','PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1',
        'CUDA_CACHE_DISABLE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    worker=None;error=None;returncode=None;artifacts=None;calibration=None
    with (VERIFY/f'{args.exp}_launch.log').open('x') as stream:
        try:
            worker=subprocess.Popen(child_command,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            write_new(VERIFY/f'{args.exp}_started.json',dict(parent_pid=os.getpid(),worker_pid=worker.pid,reservation_token=args.reservation_token))
            returncode=wait_owned_worker(worker)
            require(returncode==0,f'numeric child failed with return code {returncode}')
            artifacts=validate_child_artifacts(args.stage,ROOT/'outputs'/args.exp,hashes,invariants,require_calibration=False)
            if args.stage=='canonical':
                observed=invariant_proof(hashes,invariants,bound);require(observed['status']=='verified','post-render full source/model/input check failed')
                write_new(VERIFY/f'{args.exp}_rendered.json',dict(status='rendered',stage='canonical',exp=args.exp,
                    **stage_profile('canonical'),returncode=returncode,worker_pid=worker.pid,parent_pid=os.getpid(),
                    code_sha256=hashes,invariants=invariants,before=observed['before'],after=observed['after'],
                    control_bindings=bound,request=request(args),launch=descriptor(launch_path),artifacts=artifacts))
                output=ROOT/'scoring'/f'calibration_{args.exp}'
                calibrate=[str(PYTHON),'-X','utf8','-B','-u',str(ROOT/'scoring/evaluate.py'),'calibrate','--exp',args.exp,'--output',str(output)]
                calibration_worker=subprocess.Popen(calibrate,env={**environment,'CUDA_VISIBLE_DEVICES':''},stdout=stream,stderr=subprocess.STDOUT)
                require(wait_owned_worker(calibration_worker)==0,'read-only canonical calibration failed; preserve partial proof/log')
                calibration=descriptor(output/'calibration.json')
                artifacts=validate_child_artifacts('canonical',ROOT/'outputs'/args.exp,hashes,invariants,calibration_proof=calibration)
        except BaseException as failure:error=failure
        finally:
            if worker is not None and worker.poll() is None:
                try:wait_owned_worker(worker)
                except BaseException as failure:
                    if error is None:error=failure
            if worker is not None:returncode=worker.returncode
            proof=invariant_proof(hashes,invariants,bound)
            write_new(VERIFY/f'{args.exp}_post_run_invariants.json',proof)
            if proof['status']!='verified' and error is None:error=ValueError(proof.get('error','post-run proof failed'))
    if error is not None:
        error.worker_returncode=returncode;error.worker_pid=worker.pid if worker is not None else None;raise error
    proof_path=VERIFY/f'{args.exp}_post_run_invariants.json'
    completed=dict(passed=True,status='complete',stage=args.stage,exp=args.exp,**stage_profile(args.stage),
        timestamp=time.time(),parent_pid=os.getpid(),worker_pid=worker.pid,returncode=returncode,request=request(args),
        code_sha256=hashes,invariants=invariants,control_bindings=bound,parent_runtime=runtime,artifacts=artifacts,
        post_run_invariants=str(proof_path),post_run_invariants_sha256=sha256(proof_path))
    if calibration is not None:completed['calibration_proof']=calibration
    write_new(VERIFY/f'{args.exp}_completed.json',completed)


def record_failure(args,error):
    if not args.reservation_token:return
    try:selection=read(VERIFY/f'{args.exp}_selection.json')
    except (OSError,ValueError):return
    if (selection.get('parent_pid'),selection.get('process_start'),selection.get('reservation_token'))!=(os.getpid(),process_start(),args.reservation_token):return
    path=VERIFY/f'{args.exp}_failure.json'
    if path.exists():return
    child=ROOT/'outputs'/args.exp/'provenance.json';observation=None
    if child.is_file():
        try:
            partial=read(child)
            observation=dict(artifact=descriptor(child),status=partial.get('status'),work_total=partial.get('work_total'),
                total_elapsed_seconds=partial.get('total_elapsed_seconds'),validation_scope='failed partial observation; never a completed candidate')
        except (OSError,ValueError) as failure:observation=dict(read_error=repr(failure))
    write_new(path,dict(status='failed',stage=args.stage,exp=args.exp,**stage_profile(args.stage),timestamp=time.time(),
        parent_pid=os.getpid(),reservation_token=args.reservation_token,error=repr(error),
        returncode=getattr(error,'worker_returncode',None),worker_pid=getattr(error,'worker_pid',None),
        child_artifact=str(child),child_observation=observation))


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv==['--describe-runtime']:print(json.dumps(parent_runtime()));return
    args=parser().parse_args(argv)
    try:execute(args,argv)
    except BaseException as error:
        try:record_failure(args,error)
        except Exception as failure:print(f'Failure evidence could not be written: {failure!r}',file=sys.stderr,flush=True)
        raise


if __name__=='__main__':main()
