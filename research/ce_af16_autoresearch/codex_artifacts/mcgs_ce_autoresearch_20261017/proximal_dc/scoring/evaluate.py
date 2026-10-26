#!/usr/bin/env python3
"""Read-only calibration and frozen original59 scoring for MC-GS+proxDC."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime,timezone
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
R,S=ROOT.parent,ROOT/'scoring'
C=R/'nonlocal_autoresearch'
CURRENT_INCUMBENT=R/'development_incumbent.json'
GSTMI=Path('/home/congcongliu/anaconda3/envs/GSTMI/bin/python')
sys.path.insert(0,str(C))
import autoresearch as ar
import objective
import report

read,sha,write=objective.read_json,objective.sha256_file,ar.write_new
OBJECTIVE_SHA='cb9fa0c4ae2227947e06d3384559f31138086780aa729540b6ebfd3cc0a0bd41'
REPORT_SHA='fcce12451ba3983a00b232e4c7c34f927a7c7a6827949d9e0e5f8c657d20b800'
REFERENCE_FREEZE=C/'verification/report_reference_freeze.json'
REFERENCE_FREEZE_SHA='6ebe48fa543af773274f7d0c10bc0c21593669b9305784a8382fab7e9896e634'
SOURCES=('physics.py','runner.py','run_gpu.py')


def require(condition,message):
    if not condition:raise ValueError(message)


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value


def numerical_runtime():
    require(Path(sys.executable).resolve()==GSTMI.resolve(),'proximal numerical calibration/scoring requires actual GSTMI')
    import torch
    require(torch.__version__=='2.7.1+cu118' and torch.version.cuda=='11.8','actual GSTMI Torch2.7.1+cu118/CUDA11.8 required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','calibration/scoring is CPU-only with empty CUDA visibility')
    return dict(python=sys.executable,python_version=list(sys.version_info[:3]),torch_version=str(torch.__version__),cuda_version=torch.version.cuda)


def profile():
    return dict(method_variant='mcgs_prox_dc',method_label='MC-GS+proxDC',source_model='adamflat_g1_v1',source_iteration=4500,
        frequency_t=1.,geometry_factor=1.,local_factor=1.,lambda_ratio=.1,cg_max_matvecs=32,cg_rtol=1e-5,
        terminal_normal_evaluations=1,training_updates=0,working_dtype='complex64',
        true_residual_tolerance=2e-5,J_relative_tolerance=2e-6,J_absolute_tolerance=1e-10)


def validate_profile(value):
    require(all(value.get(k)==v for k,v in profile().items()),'fixed H14500 export-only proximal protocol differs')
    require(all(type(value.get(k)) in (int,float) for k,v in profile().items() if type(v) in (int,float)),
            'numeric protocol fields cannot be booleans')


def load_anchor(recipe_path):
    recipe=read(recipe_path);validate_profile(recipe)
    require(recipe.get('status')=='selected' and recipe.get('selected_by')=='root'
            and recipe.get('minimum_score_improvement')==1e-4,'root-selected single proximal recipe required')
    history,current_definition=recipe['completed_prior_summary'],recipe['incumbent']
    prior_path,current_path=Path(history['path']),Path(current_definition['path'])
    require(prior_path.is_absolute() and current_path.is_absolute() and current_path.resolve()==CURRENT_INCUMBENT.resolve()
            and sha(prior_path)==history['sha256'] and sha(current_path)==current_definition['sha256'],
            'actual latest summary/global incumbent bytes changed')
    previous,current=read(prior_path),read(current_path);rows=previous.get('trials',[])
    require(previous.get('status')=='completed_real_trial_snapshot' and bool(rows)
            and len({t['name'] for t in rows})==len(rows) and previous['status_counts']['completed']==len(rows)
            and all(type(t.get('score')) in (int,float) and math.isfinite(t['score']) and type(t.get('accepted')) is bool for t in rows),
            'latest actual completed history is incomplete; no trial count is hardcoded')
    q=[t for t in rows if t.get('method_variant')=='inner_precision_mm'];qf=previous.get('inner_precision_family',{})
    require(len(q)==1 and qf.get('completed_candidates')==[t['name'] for t in q]
            and all(qf.get(k)==v for k,v in dict(frequency_t=1.,cg_steps=96,true_residual_diagnostics=True).items())
            and all(t.get('frequency_t')==1. and t.get('cg_steps')==96 and t.get('true_residual_diagnostics') is True for t in q),
            'completed Q96 family and actual row required')
    h=[t for t in rows if t.get('method_variant')=='frequency_adam'];hf=previous.get('frequency_adam_family',{})
    require([t.get('geometry_factor') for t in h]==[1.,2.] and hf.get('frequency_t')==1.
            and hf.get('declared_geometry_factors')==hf.get('completed_geometry_factors')==[1.,2.]
            and all(t.get('frequency_t')==1. for t in h)
            and h[0]['name']=='adamflat_g1_v1' and h[0].get('accepted') is True,
            'both completed H factors and actual accepted fixed H1 model required')
    low=[t for t in rows if t.get('method_variant')=='frequency_lowreg_mm'];lf=previous.get('frequency_lowreg_family',{})
    require([t.get('local_factor') for t in low]==[.75,.5]
            and lf.get('declared_local_factors')==lf.get('completed_local_factors')==[.75,.5]
            and lf.get('completed_candidates')==[t['name'] for t in low]
            and all(lf.get(k)==v for k,v in dict(frequency_t=1.,cg_steps=96,true_residual_diagnostics=True).items())
            and all(t.get('frequency_t')==1. and t.get('cg_steps')==96 and t.get('true_residual_diagnostics') is True for t in low),
            'completed ordered lower-prior family required')
    pf=[t for t in rows if t.get('method_variant')=='phase_firstorder'];pff=previous.get('phase_firstorder_family',{})
    old_phase=dict(phase_curvature_mode='first_order',damping_curvature_mode='full',
        update_class='phase_first_order_quasi_newton',full_MM_majorization=False,frequency_t=1.,cg_steps=96,true_residual_diagnostics=True)
    require(len(pf)==1 and pff.get('declared_local_factors')==pff.get('completed_local_factors')==[1.]
            and pff.get('completed_candidates')==[t['name'] for t in pf]
            and all(pff.get(k)==v and all(t.get(k)==v for t in pf) for k,v in old_phase.items())
            and pf[0].get('local_factor')==1. and pff.get('full_MM_majorization') is False
            and pf[0].get('full_MM_majorization') is False,'completed actual phase-first-order family required')
    require(all(t.get('strict_success') is False for t in (*q,*h,*low,*pf)) and current.get('strict_success') is False,
            'strict win requires repeat and matched improved GS-single before another pilot')
    require(not any(t.get('method_variant')=='mcgs_prox_dc' for t in rows),'this one fixed proximal experiment was already completed')
    score=current.get('score');accepted=[t['score'] for t in rows if t['accepted']]
    require(type(score) in (int,float) and math.isfinite(score) and score==current_definition['score']
            and bool(accepted) and max(accepted)==score
            and current.get('validated_summary')==str(prior_path)
            and current.get('validated_summary_sha256')==history['sha256'],'current quality anchor/history differs')
    return dict(recipe=recipe,recipe_path=str(Path(recipe_path).resolve()),previous=previous,score=float(score),
        bindings={str(Path(recipe_path).resolve()):sha(recipe_path),str(prior_path):history['sha256'],str(current_path):current_definition['sha256']})


def apply_acceptance(previous,trials,anchor_score):
    require(len(trials)==1,'aggregate exactly one completed proximal reconstruction')
    trial=trials[0];validate_profile(trial)
    require(type(trial.get('score')) in (int,float) and math.isfinite(trial['score']),'nonfinite score')
    keep=ar.is_improvement(trial['score'],anchor_score)
    trial.update(completed_index=len(previous['trials'])+1,anchor_incumbent=anchor_score,incumbent_before=anchor_score,
        improves_anchor=keep,accepted=keep,decision='accepted' if keep else 'rejected',experiment_class='reconstruction_postprocess',
        decision_scope='one frozen H1 image postprocess; root promotes only after verified aggregate; no new Adam training')
    previous['trials'].append(trial);previous['status_counts']['completed']+=1
    previous['status_counts']['accepted' if keep else 'rejected']+=1
    return trial['score'] if keep else anchor_score


def canonical_array_check(candidate,reference):
    numerical_runtime()
    import numpy as np
    require(candidate.dtype==reference.dtype==np.complex64 and candidate.shape==reference.shape and candidate.ndim==3
            and bool(np.isfinite(candidate).all()) and bool(np.isfinite(reference).all()),'finite matching source complex64 arrays required')
    denominator=float(np.linalg.norm(reference.astype(np.complex128).ravel()))
    require(denominator>0,'published canonical reference is empty')
    relative=float(np.linalg.norm((candidate.astype(np.complex128)-reference.astype(np.complex128)).ravel())/denominator)
    require(math.isfinite(relative) and relative<=1e-5,'canonical complex L2 calibration failed; no gain or phase fit allowed')
    return dict(complex_relative_l2=relative,complex_relative_l2_tolerance=1e-5,amplitude_or_phase_fit=False)


def csm_power_definition(path,*,return_support=False):
    """Read only frozen CSMs, one coil at a time; no image/GT/ROI/solve input."""
    numerical_runtime()
    import h5py
    import numpy as np
    path=Path(path);before=sha(path)
    with h5py.File(path,'r') as stream:
        data=stream['map']
        require(data.ndim==4 and data.shape[0]>0 and data.dtype==np.dtype('complex64'),
                'source CSM must have complex64 coil,D,H,W shape')
        shape=list(data.shape);power=np.zeros(data.shape[1:],dtype=np.float64)
        for coil in range(data.shape[0]):
            value=data[coil]
            require(bool(np.isfinite(value).all()),'nonfinite CSM')
            power+=np.square(value.real,dtype=np.float64)
            power+=np.square(value.imag,dtype=np.float64)
    L=float(power.max());requested=.1*L;effective=float(np.float32(requested))
    require(math.isfinite(L) and L>0 and math.isfinite(effective) and effective>0,
            'positive representable CSM-only lambda required; no fitted scale/floor')
    require(sha(path)==before,'source CSM changed during independent scalar check')
    result=dict(path=str(path.resolve()),sha256=before,shape=shape,dtype='complex64',L=L,
        support_count=int(np.count_nonzero(power>0)),lambda_ratio=.1,lambda_requested=requested,lambda_effective=effective)
    return (result,power>0) if return_support else result


def validate_scan(record,work,power):
    """Independently check clinical scalar arithmetic and every counted FFT role."""
    import numpy as np
    def number(key,nonnegative=False):
        value=record.get(key)
        require(type(value) in (int,float) and math.isfinite(value) and (not nonnegative or value>=0),f'invalid {key}')
        return value
    def integer(key,minimum=0):
        value=record.get(key);require(type(value) is int and value>=minimum,f'invalid {key}');return value
    def close(a,b):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-30)
    require(record.get('dtype')=='torch.complex64' and record.get('lambda_ratio')==.1,'clinical native complex64/lambda rule differs')
    L=number('L');requested=number('lambda_requested');effective=number('lambda_effective')
    require(L>0 and close(L,power['L']) and requested==.1*L and effective==float(np.float32(requested)),
            'CSM-only double L and once-rounded native lambda differ')
    require(integer('support_count',1)==power['support_count'],'CSM support differs')
    coils=integer('n_coils',1);chunk=integer('chunk_size',1)
    require(coils==power['shape'][0] and chunk<=4,'source coil/chunk shape differs')
    require(type(record.get('initial_anchor_nonzero')) is bool,'actual anchor nonzero evidence missing')
    initial=integer('initial_residual_normal_evaluations')
    require(initial==int(record['initial_anchor_nonzero']),'initial residual normal is not counted')
    iterations=integer('cg_iterations');normal=integer('cg_matvecs')
    require(record.get('cg_max_matvecs')==32 and record.get('cg_rtol')==1e-5
            and normal==iterations+initial and normal<=32,'fixed CG budget or actual initial/iteration count differs')
    require(record.get('cg_converged') is True and record.get('cg_status')=='converged'
            and record.get('status')==('stationary' if iterations==0 else 'converged'),
            'failed or exhausted proximal result is not a verified candidate')
    rhs=number('rhs_norm',True);residual=number('true_residual_norm',True);relative=number('true_relative_residual',True)
    history=record.get('cg_residual_history')
    require(isinstance(history,list) and len(history)==iterations+1
            and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in history)
            and history[-1]<=1e-5*rhs,'CG history/relative-to-RHS stopping differs')
    require(close(relative,residual/max(rhs,1e-30)) and relative<=2e-5
            and record.get('true_residual_tolerance')==2e-5 and record.get('true_endpoint_passed') is True,
            'independent true endpoint residual failed')
    ja,after,da,df,penalty=[number(k,True) for k in ('J_anchor','J_after','data_anchor','data_after','anchor_penalty')]
    jt=number('J_check_tolerance',True)
    require(record.get('J_relative_tolerance')==2e-6 and record.get('J_absolute_tolerance')==1e-10
            and close(jt,1e-10+2e-6*max(abs(ja),1e-30))
            and close(ja,da) and close(after,df+penalty) and after<=ja+jt and record.get('J_check_passed') is True,
            'native-gauge fixed J arithmetic/nonincrease check failed')
    slope=number('directional_derivative')
    require((iterations==0 and slope==0) or (iterations>0 and slope<0),'nonstationary update must be a descent direction')
    require(record.get('outside_support_exact_zero') is True and record.get('terminal_normal_evaluations')==1
            and record.get('training_updates')==0,'support, terminal check, or zero-training protocol differs')
    chunks=(coils+chunk-1)//chunk
    expected=dict(rhs_adjoint_calls=1,normal_calls=normal,terminal_normal_calls=1,objective_calls=2,
        forward_calls=0,adjoint_calls=1,fft_batches=chunks*(normal+3),ifft_batches=chunks*(normal+2),
        coil_fft_transforms=coils*(normal+3),coil_ifft_transforms=coils*(normal+2),gaussian_renders=0,gaussian_vjps=0)
    require(set(work)==set(expected)|{'roles'} and all(type(work[k]) is int and work[k]==v for k,v in expected.items()),
            'actual proximal work/caps differ')
    roles=dict(rhs=dict(fft_batches=0,ifft_batches=chunks),
        objective_anchor=dict(fft_batches=chunks,ifft_batches=0),objective_candidate=dict(fft_batches=chunks,ifft_batches=0),
        terminal=dict(fft_batches=chunks,ifft_batches=chunks))
    if initial:roles['initial_normal']=dict(fft_batches=chunks,ifft_batches=chunks)
    if iterations:roles['cg_normal']=dict(fft_batches=chunks*iterations,ifft_batches=chunks*iterations)
    require(isinstance(work['roles'],dict) and {k:v for k,v in work['roles'].items() if any(v.values())}==roles,
            'FFT role counts differ from actual RHS/CG/terminal/objectives')
    return work


def scoring_implementations():
    expected={str(C/'objective.py'):OBJECTIVE_SHA,str(C/'report.py'):REPORT_SHA,
        str(REFERENCE_FREEZE):REFERENCE_FREEZE_SHA,str(ar.SCORER):ar.SCORER_HASH,str(ar.SCRIPTS/'arms.py'):ar.ARMS_HASH}
    require(all(sha(p)==h for p,h in expected.items()),'frozen original scoring/reference implementation changed')
    return expected


def existing_reference_freeze(invariants,references):
    require(REFERENCE_FREEZE.is_file() and sha(REFERENCE_FREEZE)==REFERENCE_FREEZE_SHA,
            'existing preformal reference freeze missing/changed; recreation forbidden')
    value=read(REFERENCE_FREEZE)
    require(value.get('created_before_formal_trials') is True,'reference freeze is not preformal')
    report._verify_reference_inputs(value,invariants['input_manifest'],ar.TOPIC,ar.ROI,references)
    return value


def reserve_output(output):
    path=Path(output).resolve()
    require(path.is_relative_to(S.resolve()) and path!=S.resolve(),'write only fresh proximal_dc/scoring outputs')
    path.mkdir(parents=True,exist_ok=False)
    return path


def head_scores(array_paths):
    """Original Pair metrics on already-frozen arrays; never called by physics."""
    numerical_runtime()
    import numpy as np
    import torch
    os.environ.update(RETRO_TOPIC=str(ar.TOPIC),RETRO_TRAINER=str(C/'trainer'))
    original=load(ar.SCRIPTS/'t1t2_common.py','_proximal_original_CPU_head_metrics')
    rows={}
    for scan in objective.SCANS:
        array=np.load(array_paths[scan],allow_pickle=False)
        require(array.dtype==np.complex64 and array.ndim==3 and bool(np.isfinite(array).all()),'frozen source-gauge complex64 array required')
        pair=original.Pair(scan,torch.device('cpu'))
        pair.gt_c=pair.gt_c.to(torch.complex128);pair.gt=pair.gt.to(torch.float64)
        rows[scan]=pair.score(torch.from_numpy(array).to(torch.complex128),'mcgs_prox_dc',seconds=0.)
        report.head_metrics(rows[scan])
        del pair,array
    return rows


def context():
    supervisor=load(ROOT/'run_gpu.py','_proximal_scoring_supervisor')
    hashes=supervisor.source_hashes();invariants=supervisor.snapshot_invariants()
    require(set(hashes)=={str(ROOT/name) for name in SOURCES} and supervisor.stage_profile('prox')==profile(),
            'three reviewed sources and exact protocol required')
    return supervisor,hashes,invariants


def control_artifacts(supervisor,args):
    """Bind actual control trees, including calibration and raw rendered proof."""
    files=supervisor.control_bindings(args)
    for path in (args.cuda_operator_gate,args.canonical_control_gate):
        if path is None:continue
        gate=read(path);files.update(gate['artifacts']);files[gate['post_run_invariants']]=gate['post_run_invariants_sha256']
        files.update(gate['control_bindings']);files[str(path)]=sha(path)
        files[str(supervisor.VERIFY/f"{gate['exp']}_launch.json")]=sha(supervisor.VERIFY/f"{gate['exp']}_launch.json")
        if gate.get('calibration_proof'):
            proof=gate['calibration_proof'];files[proof['path']]=proof['sha256']
            rendered=read(proof['path'])['rendered_proof'];files[rendered['path']]=rendered['sha256']
    return files


def validate_arrays_and_scans(run,stage,value,invariants):
    """No inverse solve: read immutable exports, CSMs and fixed reporting scales."""
    numerical_runtime()
    import numpy as np
    import yaml
    definition=invariants['accepted_h1']['definition'];config=yaml.safe_load(Path(definition['actual_config']['path']).read_text())
    checks={}
    for scan,channel in (('pre','plain'),('post','contrast')):
        row=value['scans'][scan];csm=Path(config['data'][channel]['csm_path'])
        require(str(csm) in invariants['input_manifest']['files'] and sha(csm)==invariants['input_manifest']['files'][str(csm)],
                'independent CSM must be an unchanged original frozen input')
        power,support=csm_power_definition(csm,return_support=True)
        require(row['n_coils']==power['shape'][0],'actual CSM coil count differs')
        arrays={}
        for kind,info in value['arrays'][scan].items():
            path=Path(info['path']);array=np.load(path,allow_pickle=False)
            require(path.resolve()==(Path(run)/f'{kind}_{scan}_complex.npy').resolve() and sha(path)==info['sha256']
                    and array.dtype==np.complex64 and list(array.shape)==info['shape']==power['shape'][1:]
                    and bool(np.isfinite(array).all()),'exported complex64 dtype/shape/finiteness/hash differs')
            arrays[kind]=array
        reference=np.load(definition['scans'][scan]['canonical_array']['path'],allow_pickle=False)
        canonical=canonical_array_check(arrays['canonical'],reference)
        require(math.isclose(canonical['complex_relative_l2'],row['canonical_check']['relative_complex_l2'],rel_tol=1e-6,abs_tol=1e-12),
                'recorded canonical complex relative error differs from independent reader')
        gauge={k:definition['scans'][scan][k] for k in ('scale_trainer','scale_gauge','factor_scale')}
        require(row['gauge']==gauge,'reporting gauge changed; no amplitude/phase fitting allowed')
        if stage=='prox':
            validate_scan(row['record'],row['work'],power)
            require(bool((arrays['candidate'][~support]==0).all()),'candidate contains values outside nonzero CSM support')
        else:require(row['record'] is None and row['algorithm_executed'] is False,'canonical control cannot contain an invented solve')
        checks[scan]=dict(csm_power=power,canonical_check=canonical,fixed_gauge=gauge,
            solver_record=row['record'],actual_work=row['work'],candidate_support_exact_zero=stage=='prox')
    return checks


def validate_total_work(value,stage):
    """Independent clinical domain sum, including source data preparation."""
    require(stage in ('canonical','prox'),'clinical work validation requires canonical/prox')
    scans=value['scans'];total=value['work_total'];coils=sum(r['n_coils'] for r in scans.values())
    construction=dict(fft_batches=2,ifft_batches=6,scalar_fft_transforms=2,scalar_ifft_transforms=0,
        coil_fft_transforms=0,coil_ifft_transforms=3*coils)
    require(total.get('construction')==construction and all(type(v) is int for v in total['construction'].values()),
            'normal-constructor scalar/coil transforms omitted or mislabeled')
    physics={k:sum(r['work'][k] for r in scans.values()) for k in total['physics']}
    require(total['physics']==physics and all(type(v) is int and v==0 for v in total['tiny_identity_physics'].values()),
            'clinical physics work differs or includes synthetic identity work')
    expected=dict(gaussian_renders=2,gaussian_cuda_renders=2,gaussian_reference_renders=0,gaussian_vjps=0,
        canonical_export_calls=2,canonical_fft_batches=2,canonical_ifft_batches=4,
        canonical_coil_fft_transforms=coils,canonical_coil_ifft_transforms=2*coils,
        total_fft_batches=4+physics['fft_batches'],total_ifft_batches=10+physics['ifft_batches'],
        total_coil_fft_transforms=coils+physics['coil_fft_transforms'],total_coil_ifft_transforms=5*coils+physics['coil_ifft_transforms'],
        total_scalar_fft_transforms=2,total_scalar_ifft_transforms=0)
    require(all(type(total.get(k)) is int and total[k]==v for k,v in expected.items()),'end-to-end source/export/solver work totals differ')
    return total


def validate_actual_run(exp,stage,supervisor,hashes,invariants,*,recipe_path=None,rendered=False):
    require(exp not in ('','.','..') and Path(exp).name==exp,'invalid experiment name')
    run=ROOT/'outputs'/exp;launch_path=supervisor.VERIFY/f'{exp}_launch.json'
    launch,args=supervisor.validate_launch(launch_path,hashes,invariants)
    require(args.stage==stage and args.exp==exp,'actual execution stage/exp differs')
    if recipe_path is not None:
        require(args.selected_recipe is not None and args.selected_recipe.resolve()==Path(recipe_path).resolve(),
                'scoring anchor differs from actual root execution recipe')
    if rendered:
        completed,artifacts=supervisor.validate_rendered(exp,hashes,invariants)
        proof_path=supervisor.VERIFY/f'{exp}_rendered.json'
    else:
        proof_path=supervisor.VERIFY/f'{exp}_completed.json'
        completed=supervisor.validate_observed_gate(proof_path,stage,hashes,invariants);artifacts=completed['artifacts']
    value,metrics=read(run/'provenance.json'),read(run/'metrics.json')
    validate_profile(value);validate_profile(metrics)
    definition=invariants['accepted_h1']['definition'];before=value['source_state_before'];after=value['source_state_after']
    require(before==after and before['parameter_sha256']==definition['expected_model_state']['parameter_sha256']
            and before['adam']==definition['expected_model_state']['adam']
            and all(v.get('step')==4500 for v in before['adam'].values())
            and value['source_restore_performed'] is True and value['restored_iteration']==4500,
            'actual H14500 model, all seven Adam states, or immutability differs')
    require(value['numerical_runtime']==metrics['numerical_runtime']==numerical_runtime(),'child/scorer actual GSTMI runtime differs')
    elapsed=value.get('total_elapsed_seconds')
    require(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=0
            and metrics.get('total_elapsed_seconds')==elapsed,'actual nonnegative execution runtime required')
    checks=validate_arrays_and_scans(run,stage,value,invariants)
    validate_total_work(value,stage)
    bound={**artifacts,**control_artifacts(supervisor,args),str(launch_path):sha(launch_path),str(proof_path):sha(proof_path)}
    if not rendered:bound[completed['post_run_invariants']]=completed['post_run_invariants_sha256']
    evidence=dict(stage=stage,**profile(),model_definition=invariants['accepted_h1']['definition_manifest'],
        source_checkpoint=definition['checkpoint'],scans=checks,actual_work=value['work_total'],total_elapsed_seconds=elapsed,
        bound_artifacts_sha256=bound,numerical_runtime=numerical_runtime(),
        budget_disclosure='One frozen H14500 export-only image postprocess: source construction/data preparation, canonical export and solver work are disclosed separately and included in totals. Per scan32CG matvecs including initial residual, plus one independent terminal normal. No new Adam updates, Gaussian VJP, checkpoint or equal-compute claim.')
    return run,value,evidence,artifacts,args


def calibrate(exp,output):
    runtime=numerical_runtime();implementations=scoring_implementations();supervisor,hashes,I=context()
    run,value,evidence,artifacts,args=validate_actual_run(exp,'canonical',supervisor,hashes,I,rendered=True)
    references=report.load_references(ar.TOPIC,I['input_manifest']);existing_reference_freeze(I,references)
    observed={**evidence['bound_artifacts_sha256'],**implementations,**hashes,
        **I['accepted_h1']['files'],str(__file__):sha(__file__)}
    output=reserve_output(output);write(output/'STARTED.json',dict(status='calibrating',exp=exp,stage='canonical',**profile()))
    head=head_scores({s:run/f'canonical_{s}_complex.npy' for s in objective.SCANS});scans={}
    for scan in objective.SCANS:
        published=I['accepted_h1']['definition']['scans'][scan];reference=read(published['score_reference']['path'])['rows']
        require(len(reference)==1 and report.head_metrics(reference[0])['psnr']==published['head_psnr'],'published H1 head reference differs')
        difference=abs(report.head_metrics(head[scan])['psnr']-published['head_psnr'])
        require(math.isfinite(difference) and difference<=.001,'canonical whole-head PSNR calibration failed')
        scans[scan]={**evidence['scans'][scan]['canonical_check'],'head_psnr_absolute_difference_db':difference,
            'head_psnr_tolerance_db':.001,'head_metrics':report.head_metrics(head[scan]),'reference':published['canonical_array']}
    require(all(sha(p)==h for p,h in observed.items()) and supervisor.snapshot_invariants()==I,
            'source/model/input/rendered arrays changed during canonical calibration')
    write(output/'calibration.json',dict(status='verified',passed=True,stage='canonical',exp=exp,**profile(),
        code_sha256=hashes,invariants=I,model_definition=I['accepted_h1']['definition_manifest'],control_bindings=supervisor.control_bindings(args),
        rendered_proof=supervisor.descriptor(supervisor.VERIFY/f'{exp}_rendered.json'),bound_artifacts_sha256=artifacts,
        scans=scans,numerical_runtime=runtime,observed_artifact_sha256_before_and_after=observed))


def score_trial(exp,recipe_path,output):
    runtime=numerical_runtime();anchor=load_anchor(recipe_path);implementations=scoring_implementations()
    supervisor,hashes,I=context();supervisor.validate_recipe(recipe_path,hashes,I)
    run,provenance,evidence,_,_=validate_actual_run(exp,'prox',supervisor,hashes,I,recipe_path=recipe_path)
    references=report.load_references(ar.TOPIC,I['input_manifest'])
    require(references==anchor['previous']['references'],'all ten audited references/opponents must be unchanged')
    freeze=existing_reference_freeze(I,references)
    observed={**evidence['bound_artifacts_sha256'],**anchor['bindings'],**implementations,**hashes,
        str(__file__):sha(__file__),str(ar.BASELINE):sha(ar.BASELINE)}
    output=reserve_output(output);write(output/'STARTED.json',dict(status='scoring',exp=exp,**profile(),anchor=anchor['bindings']))
    head=head_scores({s:run/f'candidate_{s}_complex.npy' for s in objective.SCANS})
    for scan in objective.SCANS:write(output/f'scores_{scan}.json',dict(rows=[head[scan]]))
    baseline=objective.load_baseline(ar.BASELINE);raw=output/'lesion_allmetrics.json'
    command=[str(GSTMI),'-X','utf8','-B',str(ar.SCORER),'--min_core','150','--mcgs_file',
        str(run/'candidate_{scan}_complex.npy'),'--out',str(raw)]
    started=time.perf_counter()
    with (output/'scorer.log').open('x') as stream:
        rc=subprocess.run(command,cwd=C,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','RETRO_TOPIC':str(ar.TOPIC),
            'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','PYTHONPATH':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
            stdout=stream,stderr=subprocess.STDOUT).returncode
    require(rc==0,'original all59 scorer failed; preserve partial output and its log')
    result=objective.score(read(raw),baseline,head)
    result.update(run_dir=str(run),**profile(),postprocess_evidence=evidence,
        candidate_sha256={s:sha(run/f'candidate_{s}_complex.npy') for s in objective.SCANS},scorer_command=command)
    write(output/'objective.json',result)
    trial=dict(name=exp,**profile(),experiment_class='reconstruction_postprocess',score=result['score'],strict_success=result['strict_success'],
        best_plane=result['best_plane'],plane_counts=report.plane_counts(result),whole_head_metrics={s:report.head_metrics(head[s]) for s in objective.SCANS},
        run_dir=str(run),objective=str(output/'objective.json'),objective_sha256=sha(output/'objective.json'),metrics_sha256=sha(run/'metrics.json'),
        source_checkpoint_sha256=I['accepted_h1']['definition']['checkpoint']['sha256'],postprocess_elapsed_seconds=provenance['total_elapsed_seconds'],
        postprocess_evidence=evidence,candidate=dict(name=exp,**profile()),anchor_incumbent=anchor['score'],anchor_bindings=anchor['bindings'],
        improves_anchor=ar.is_improvement(result['score'],anchor['score']),
        decision_scope='One root-selected fixed H1 image postprocess; root promotion awaits verified aggregate',
        runtime_scope='Measured restore/export/postprocess runtime; no new500-step training or equal-compute claim')
    enhancement=report.enhancement_bundle(ar.TOPIC,ar.ROI,references,[trial],I['input_manifest'],freeze)
    require(all(sha(p)==h for p,h in observed.items()) and supervisor.snapshot_invariants()==I,'actual candidate/source/control/anchor changed during scoring')
    load_anchor(recipe_path);write(output/'enhancement.json',enhancement);write(output/'trial.json',trial)
    write(output/'completed.json',dict(status='verified',exp=exp,**profile(),returncode=rc,anchor_bindings=anchor['bindings'],
        raw_sha256=sha(raw),trial_sha256=sha(output/'trial.json'),objective_sha256=sha(output/'objective.json'),
        enhancement_sha256=sha(output/'enhancement.json'),observed_artifact_sha256_before_and_after=observed,invariants_before_and_after=I,
        scorer_command=command,scorer_cwd=str(C),elapsed_seconds=time.perf_counter()-started,numerical_runtime=runtime,
        scoring_policy='All59 head-masked lesion-containing planes; both scans/all4metrics/all10opponents. Original thresholds;8ULP verifies recomputation only.'))


def observed_failures():
    setup,formal,bindings=[],[],{}
    for path in sorted((ROOT/'verification').glob('*_failure.json')):
        value=read(path);name=path.name[:-len('_failure.json')];selection_path=path.with_name(name+'_selection.json');selection=read(selection_path)
        require(value.get('status')=='failed' and value.get('stage') in ('tiny','canonical','prox')
                and value.get('reservation_token') and value.get('reservation_token')==selection.get('reservation_token')
                and value.get('parent_pid')==selection.get('parent_pid') and value.get('stage')==selection.get('stage'),
                'failure is not an actual owned proximal attempt')
        observation=value.get('child_observation') or {};verified=observation.get('binding_status')=='verified'
        row=dict(name=name,path=str(path),sha256=sha(path),stage=value['stage'],child_started=value.get('worker_pid') is not None,
            returncode=value.get('returncode'),error=value.get('error'),training_updates=0,
            child_observation=value.get('child_observation'),
            measured_elapsed_seconds=observation.get('elapsed_seconds') if verified else None,
            partial_work=observation.get('partial_work') if verified else None,
            scope='failed postprocess attempt; not completed' if value['stage']=='prox' else 'setup/control; not completed training')
        (formal if value['stage']=='prox' else setup).append(row);bindings.update({str(path):sha(path),str(selection_path):sha(selection_path)})
        observed=dict(observation.get('bound_artifacts_sha256',{}));info=observation.get('artifact')
        if info:
            require(info['path'] not in observed or observed[info['path']]==info['sha256'],'failed artifact descriptors disagree')
            observed[info['path']]=info['sha256']
        for artifact,digest in observed.items():
            require(Path(artifact).resolve().parent==(ROOT/'outputs'/name).resolve()
                    and Path(artifact).name in ('failure.json','provenance.json') and sha(artifact)==digest,
                    'failed observed artifact changed or is outside the owned failed run')
            bindings[artifact]=digest
    return setup,formal,bindings


def aggregate(directories,recipe_path,output):
    runtime=numerical_runtime();require(len(directories)==1,'aggregate exactly one completed proximal candidate')
    anchor=load_anchor(recipe_path);scoring_implementations();supervisor,hashes,I=context();supervisor.validate_recipe(recipe_path,hashes,I)
    directory=Path(directories[0]).resolve();require(directory.is_relative_to(S.resolve()) and directory!=S.resolve(),'scored trial outside owned scoring tree')
    done,trial=read(directory/'completed.json'),read(directory/'trial.json');validate_profile(done);validate_profile(trial)
    require(done.get('status')=='verified' and done.get('returncode')==0 and done.get('numerical_runtime')==runtime
            and done.get('anchor_bindings')==trial.get('anchor_bindings')==anchor['bindings']
            and done.get('invariants_before_and_after')==I,'partial/stale/different-anchor score is inadmissible')
    observed=dict(anchor['bindings'])
    for name,key in (('trial.json','trial_sha256'),('objective.json','objective_sha256'),('enhancement.json','enhancement_sha256'),('lesion_allmetrics.json','raw_sha256')):
        require(sha(directory/name)==done[key],'scored output bytes changed');observed[str(directory/name)]=done[key]
    require(all(sha(p)==h for p,h in done['observed_artifact_sha256_before_and_after'].items()),'actual scored artifact/source/control evidence changed')
    observed.update(done['observed_artifact_sha256_before_and_after'])
    stored=read(directory/'objective.json');validate_profile(stored)
    recomputed=objective.score(read(directory/'lesion_allmetrics.json'),objective.load_baseline(ar.BASELINE),stored['whole_head_metrics'])
    require(all(stored[k]==recomputed[k] for k in ('score','strict_success','planes','best_plane')),'objective recomputation differs')
    require(trial['name']==done['exp'] and trial['name'] not in {t['name'] for t in anchor['previous']['trials']}
            and trial['score']==recomputed['score'] and trial['strict_success']==recomputed['strict_success']
            and trial['plane_counts']==report.plane_counts(recomputed) and trial['anchor_incumbent']==anchor['score']
            and trial['improves_anchor']==ar.is_improvement(trial['score'],anchor['score'])
            and trial['source_checkpoint_sha256']==I['accepted_h1']['definition']['checkpoint']['sha256']
            and trial['experiment_class']=='reconstruction_postprocess'
            and trial['whole_head_metrics']=={s:report.head_metrics(stored['whole_head_metrics'][s]) for s in objective.SCANS},
            'actual postprocess trial/quality/source/decision differs')
    previous=deepcopy(anchor['previous']);final=apply_acceptance(previous,[trial],anchor['score'])
    setup,formal,failure_bindings=observed_failures();observed.update(failure_bindings)
    previous.setdefault('setup_failures',[]).extend(setup);previous.setdefault('proximal_dc_attempt_failures',[]).extend(formal)
    previous.setdefault('failed_trials',[]).extend({**row,'event':'trial_failed','exp':row['name'],'method_variant':'mcgs_prox_dc'} for row in formal)
    previous.setdefault('terminal_failed_trials',[]).extend(row['name'] for row in formal)
    previous['status_counts']['failed']=previous['status_counts'].get('failed',0)+len(formal)
    previous['status_counts']['failure_events']=previous['status_counts'].get('failure_events',0)+len(formal)
    enhancement=read(directory/'enhancement.json')
    previous['enhancement_fidelity']['rows'].append(next(r for r in enhancement['rows'] if r['method']==trial['name']))
    previous['enhancement_fidelity']['inputs'][trial['name']]=enhancement['inputs'][trial['name']]
    previous.update(status='completed_real_trial_snapshot',created_utc=datetime.now(timezone.utc).isoformat(),
        previous_summary=anchor['recipe']['completed_prior_summary'],artifacts={},numerical_runtime=runtime,
        proximal_dc_family={**profile(),'declared_candidate_count':1,'completed_candidates':[trial['name']],
            'initial_incumbent':anchor['score'],'final_incumbent':final,'anchor_bindings':anchor['bindings'],
            'experiment_class':'reconstruction_postprocess','global_promotion':'root only after verified aggregate'},
        postprocess_budget_disclosure=trial['postprocess_evidence']['budget_disclosure'])
    previous['claim_status']='Provisional: fresh repeat and matched improved GS-single required' if trial['strict_success'] else 'Target unmet'
    require(all(sha(p)==h for p,h in observed.items()) and supervisor.snapshot_invariants()==I,'source/candidate/control/anchor changed during aggregation')
    load_anchor(recipe_path);output=reserve_output(output);write(output/'summary.json',previous)
    write(output/'validator_proof.json',dict(status='verified',anchor_bindings=anchor['bindings'],all_observed_artifact_sha256=observed,
        invariants_before_and_after=I,numerical_runtime=runtime,
        experiment_count_policy='One actual image postprocess appended; all prior results/failures preserved. Training updates0; controls and archive rescoring are not training.'))


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);sub=p.add_subparsers(dest='action',required=True)
    for name in ('calibrate','score'):
        command=sub.add_parser(name,allow_abbrev=False);command.add_argument('--exp',required=True)
        command.add_argument('--output',type=Path,required=True)
        if name=='score':command.add_argument('--selected-recipe',type=Path,required=True)
    combined=sub.add_parser('aggregate',allow_abbrev=False);combined.add_argument('--trial',type=Path,action='append',required=True)
    combined.add_argument('--selected-recipe',type=Path,required=True);combined.add_argument('--output',type=Path,required=True)
    args=p.parse_args();numerical_runtime()
    if args.action=='calibrate':calibrate(args.exp,args.output)
    elif args.action=='score':score_trial(args.exp,args.selected_recipe,args.output)
    else:aggregate(args.trial,args.selected_recipe,args.output)


if __name__=='__main__':main()
