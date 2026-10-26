#!/usr/bin/env python3
"""Read-only phase-first-order validation and original59 fixed-anchor scoring."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types
from typing import Tuple

sys.dont_write_bytecode = True
FM = Path(__file__).resolve().parents[1]
R, S = FM.parent, FM / 'scoring'
C = R / 'nonlocal_autoresearch'
CURRENT_INCUMBENT = R / 'development_incumbent.json'
sys.path.insert(0, str(C))
import autoresearch as ar
import objective
import report

GSTMI = Path('/home/congcongliu/anaconda3/envs/GSTMI/bin/python')
SOURCE_SHA = '39e1632ff07e530fb7a3454e78698363a62a856f301c4159d3b4c9a784a48c9e'
OBJECTIVE_SHA = 'cb9fa0c4ae2227947e06d3384559f31138086780aa729540b6ebfd3cc0a0bd41'
REPORT_SHA = 'fcce12451ba3983a00b232e4c7c34f927a7c7a6827949d9e0e5f8c657d20b800'
REFERENCE_FREEZE = C / 'verification/report_reference_freeze.json'
REFERENCE_FREEZE_SHA = '6ebe48fa543af773274f7d0c10bc0c21593669b9305784a8382fab7e9896e634'
D_SCORER = R / 'density_mm/scoring/evaluate.py'
D_SCORER_SHA = '9cd8700380040aeafc67deb1369378ef5040166469f258930b8dbbfcb14edc39'
SHIFTS = [list(shift) for shift in itertools.product((0, 1), repeat=3)]
NAMES = ('positions','scales','rotations','density_real','density_imag','density_real_contrast','density_imag_contrast')
SOURCES = ('runner.py','coupled_edge.py','density_operator.py','surrogates.py','pcg.py','cycle_average.py',
           'gpu_preflight.py','run_gpu.py','frequency_weights.py')
SHAPES = {**{name:[3600000,3] for name in NAMES[:2]}, 'rotations':[3600000,4],
          **{name:[3600000] for name in NAMES[3:]}}
PHASE = {'phase_curvature_mode':'first_order','damping_curvature_mode':'full',
         'update_class':'phase_first_order_quasi_newton','full_MM_majorization':False}
read, sha, write = objective.read_json, objective.sha256_file, ar.write_new


def require(condition, message):
    if not condition: raise ValueError(message)


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def numerical_runtime():
    require(Path(sys.executable).resolve()==GSTMI.resolve(),'all TV/wavelet numerical validation/scoring requires the actual GSTMI interpreter')
    import torch
    require(torch.__version__=='2.7.1+cu118' and torch.version.cuda=='11.8','require actual GSTMI Torch2.7.1+cu118/CUDA11.8')
    return {'python':sys.executable,'python_version':list(sys.version_info[:3]),
            'torch_version':str(torch.__version__),'cuda_version':torch.version.cuda}


def profile(t,local_factor):
    require(type(t) in (int,float) and t==1.,'formal strength candidates require fixed frequency_t=1')
    require(type(local_factor) in (int,float) and local_factor in (1.,.75,.5),'local factor must be1/.75/.5')
    return {'frequency_t':1.,'local_factor':float(local_factor),'wavelet_mode':'uniform8',
            'wavelet_phase_count':8,'fixed_wavelet_shift':None,'cg_steps':96,'true_residual_diagnostics':True,**PHASE}


def validate_phase_metadata(value):
    require(all(value.get(k)==v for k,v in PHASE.items()) and value.get('full_MM_majorization') is False,
            'actual first-order phase/full damping/quasi-Newton protocol differs')


def existing_reference_freeze(manifest,references):
    require(REFERENCE_FREEZE.is_file() and sha(REFERENCE_FREEZE)==REFERENCE_FREEZE_SHA,
            'existing audited preformal reference freeze is missing/changed; recreation is forbidden')
    value=read(REFERENCE_FREEZE)
    require(value.get('created_before_formal_trials') is True,'reference freeze is not preformal')
    report._verify_reference_inputs(value,manifest,ar.TOPIC,ar.ROI,references)
    return value


def load_anchor(recipe_path):
    """Dynamic latest root selection after completed Q, H and ordered L families."""
    recipe=read(recipe_path)
    require(recipe.get('status')=='selected' and recipe.get('selected_by')=='root'
            and type(recipe.get('frequency_t')) in (int,float) and recipe['frequency_t']==1.
            and recipe.get('local_factor_grid')==[1.] and type(recipe['local_factor_grid'][0]) in (int,float)
            and 'local_factor' not in recipe and recipe.get('cg_steps')==96
            and recipe.get('true_residual_diagnostics') is True and recipe.get('minimum_score_improvement')==1e-4,
            'root-selected single phase-first-order factor1 recipe required')
    require(all(recipe.get(k)==v for k,v in dict(source_iteration=4000,outer_steps=5,max_vjp_pairs=500,
        max_render_pairs=700,max_merit_forwards=100,nlambda=0.,wavelet_mode='uniform8',damping_rel=.001,cg_rtol=.001).items()),
        'fixed original-source phase candidate recipe budgets/modes differ')
    validate_phase_metadata(recipe)
    history, incumbent=recipe['completed_prior_summary'],recipe['incumbent']
    prior_path,current_path=Path(history['path']),Path(incumbent['path'])
    require(prior_path.is_absolute() and current_path.is_absolute() and current_path.resolve()==CURRENT_INCUMBENT.resolve()
            and sha(prior_path)==history['sha256'] and sha(current_path)==incumbent['sha256'],
            'selected completed-summary/incumbent bytes changed or are missing')
    previous,current=read(prior_path),read(current_path)
    trials=previous.get('trials',[])
    require(previous.get('status')=='completed_real_trial_snapshot' and bool(trials)
            and len({t['name'] for t in trials})==len(trials)
            and previous.get('geometry_family',{}).get('completed_factors')==[2.,4.,8.]
            and previous.get('frequency_family',{}).get('completed_t')==[.5,1.]
            and previous.get('tvwave_family',{}).get('completed_local_factors')==[2.,4.]
            and previous.get('inner_precision_family',{}).get('frequency_t')==1.
            and previous.get('inner_precision_family',{}).get('cg_steps')==96
            and previous.get('inner_precision_family',{}).get('true_residual_diagnostics') is True
            and len(previous.get('inner_precision_family',{}).get('completed_candidates',[]))==1
            and previous.get('frequency_adam_family',{}).get('frequency_t')==1.
            and previous.get('frequency_adam_family',{}).get('declared_geometry_factors')==[1.,2.]
            and previous.get('frequency_adam_family',{}).get('completed_geometry_factors')==[1.,2.]
            and previous.get('frequency_lowreg_family',{}).get('frequency_t')==1.
            and previous.get('frequency_lowreg_family',{}).get('declared_local_factors')==[.75,.5]
            and previous.get('frequency_lowreg_family',{}).get('completed_local_factors')==[.75,.5]
            and previous.get('frequency_lowreg_family',{}).get('cg_steps')==96
            and previous.get('frequency_lowreg_family',{}).get('true_residual_diagnostics') is True,
            'wait for actual completed Q/H/ordered L families; prior counts are not hardcoded')
    require(all(type(t.get('score')) in (int,float) and math.isfinite(t['score'])
                and type(t.get('accepted')) is bool for t in trials),'prior completed trials need finite scores and actual decisions')
    geometry=[t['geometry_factor'] for t in trials if t.get('method_variant')=='geometry_lr_restart']
    require(geometry==[2.,4.,8.],'all actual geometry factor2/4/8 trials must remain in declared order')
    frequency=[t.get('frequency_t') for t in trials if t.get('method_variant')=='frequency_mm' and t.get('local_factor',1.)==1.]
    require(frequency==[.5,1.],'both actual frequency t.5/1 trials must remain in declared order')
    strength=[t.get('local_factor') for t in trials if t.get('method_variant')=='frequency_mm'
              and t.get('frequency_t')==1. and t.get('local_factor') in (2.,4.)]
    require(strength==[2.,4.],'both actual TV/wavelet factor2/4 trials must remain in declared order')
    precision=[t for t in trials if t.get('method_variant')=='inner_precision_mm']
    require([t['name'] for t in precision]==previous['inner_precision_family']['completed_candidates']
            and all(t.get('frequency_t')==1. and t.get('cg_steps')==96
                    and t.get('true_residual_diagnostics') is True and t.get('strict_success') is False
                    for t in precision),
            'actual completed inner-precision trial and family declaration differ')
    adam=[t for t in trials if t.get('method_variant')=='frequency_adam']
    require([t.get('geometry_factor') for t in adam]==[1.,2.]
            and all(t.get('frequency_t')==1. and t.get('strict_success') is False for t in adam),
            'both actual ordered H factors1/2 must be completed without a provisional strict win')
    lowreg=[t for t in trials if t.get('method_variant')=='frequency_lowreg_mm']
    require([t.get('local_factor') for t in lowreg]==[.75,.5]
            and [t['name'] for t in lowreg]==previous['frequency_lowreg_family'].get('completed_candidates')
            and all(t.get('frequency_t')==1. and t.get('cg_steps')==96
                    and t.get('true_residual_diagnostics') is True and t.get('strict_success') is False for t in lowreg),
            'both actual ordered L .75/.5 rows must match completed family without a provisional strict win')
    require(sha(C/'objective.py')==OBJECTIVE_SHA and sha(C/'report.py')==REPORT_SHA,'frozen scoring implementations changed')
    score=current.get('score')
    require(type(score) in (int,float) and math.isfinite(score) and score==incumbent['score']
            and current.get('strict_success') is False,'anchor changed or strict-win confirmation takes priority')
    require(current.get('validated_summary')==str(prior_path) and current.get('validated_summary_sha256')==history['sha256'],
            'current incumbent must bind the latest completed campaign summary')
    accepted=[t['score'] for t in trials if t.get('accepted') is True]
    require(bool(accepted) and all(type(v) in (int,float) and math.isfinite(v) for v in accepted)
            and max(accepted)==score and previous['status_counts']['completed']==len(trials),
            'selected incumbent and completed acceptance history disagree')
    return {'recipe_path':str(Path(recipe_path).resolve()),'recipe_sha256':sha(recipe_path),
            'recipe':recipe,'previous':previous,'score':float(score),
            'bindings':{str(prior_path):history['sha256'],str(current_path):incumbent['sha256'],
                        str(Path(recipe_path).resolve()):sha(recipe_path)}}


def validate_frequency_arrays(value,run,t,expected_sources,expected_masks):
    """Recompute saved arrays with independent NumPy math and frozen originals."""
    runtime=numerical_runtime()
    require(not isinstance(t,bool) and t in (0.,.5,1.),'frequency-array checks require declared t0/.5/1')
    import numpy as np
    folder=Path(run).resolve()/'frequency_weights'
    require(value==read(folder/'manifest.json') and value.get('version')==1 and value.get('frequency_t')==t
            and value.get('mass_relative_tolerance')==2e-6 and set(value.get('scans',{}))=={'pre','post'},
            'saved frequency manifest/profile differs')
    proof,files={}, {str(folder/'manifest.json'):sha(folder/'manifest.json')}
    for scan in ('pre','post'):
        row=value['scans'][scan];arrays={}
        for name in ('source_weight','effective_weight','acquired_mask'):
            info=row[name];path=folder/f'{scan}_{name}.npy'
            require(path.resolve().parent==folder and str(path)==info.get('path') and sha(path)==info.get('file_sha256'),
                    'frequency file path/bytes changed')
            array=np.load(path,allow_pickle=False)
            require(list(array.shape)==info.get('shape') and 'torch.'+str(array.dtype)==info.get('dtype')
                    and hashlib.sha256(array.tobytes(order='C')).hexdigest()==info.get('byte_sha256'),
                    'frequency raw dtype/shape/hash differs')
            arrays[name]=array;files[str(path)]=sha(path)
        source,effective,mask=(arrays[k] for k in ('source_weight','effective_weight','acquired_mask'))
        require(source.shape==effective.shape==mask.shape and source.dtype==effective.dtype
                and source.dtype.kind=='f' and np.isfinite(source).all() and np.isfinite(effective).all()
                and (source>=0).all() and np.isfinite(mask).all() and ((mask==0)|(mask==1)).all(),
                'invalid real nonnegative frequency arrays/binary mask')
        require(source.dtype==expected_sources[scan].dtype and np.array_equal(source,expected_sources[scan]),
                'source frequency weights differ from original frozen builder')
        require(mask.dtype==expected_masks[scan].dtype and np.array_equal(mask,expected_masks[scan]),
                'saved mask is not the original acquired mask')
        acquired=mask.astype(bool)
        require(acquired.any() and (source[acquired]>0).any(),'empty acquired positive source mass')
        original=source.astype(np.float64)
        def rms(a):
            scale=float(a.max());return scale*math.sqrt(float(np.mean((a/scale)**2,dtype=np.float64)))
        c=rms(original[acquired])
        if t==0.:expected=source
        elif t==1.:expected=np.full_like(source,c)
        else:
            mixed=(1-t)*original+t*c;expected=(mixed*c/rms(mixed[acquired])).astype(source.dtype)
        difference=float(np.max(np.abs(effective.astype(np.float64)-expected)/np.maximum(np.abs(expected),np.finfo(source.dtype).tiny)))
        tolerance=8*float(np.finfo(source.dtype).eps)
        before=float(np.mean(original[acquired]**2,dtype=np.float64))
        after=float(np.mean(effective[acquired].astype(np.float64)**2,dtype=np.float64));mass=abs(after/before-1)
        require(math.isfinite(difference) and difference<=tolerance and mass<=2e-6
                and (t!=0. or np.array_equal(source,effective)),'saved weights do not implement declared flatten/RMS formula')
        require(row['acquired_count']==int(acquired.sum())
                and math.isclose(c,row['c'],rel_tol=1e-12)
                and math.isclose(before,row['source_mean_squared_weight'],rel_tol=1e-12)
                and math.isclose(after,row['effective_mean_squared_weight'],rel_tol=1e-12)
                and math.isclose(mass,row['relative_mass_error'],rel_tol=1e-10,abs_tol=1e-14),
                'recorded scalar statistics disagree with independently read arrays')
        proof[scan]={'source_and_original_acquired_mask_exact':True,'acquired_count':int(acquired.sum()),
                     'source_mean_squared_weight':before,'effective_mean_squared_weight':after,'c':c,
                     'relative_mass_error':mass,'interpolation_max_relative_error':difference,
                     'interpolation_relative_tolerance':tolerance,'effective_identity':np.array_equal(source,effective)}
    return {'passed':True,'frequency_t':t,'mass_relative_tolerance':2e-6,'scans':proof,
            'bound_files_sha256':files,'numerical_runtime':runtime}


def reconstruct_source_arrays(config,manifest):
    """Original builder + external acquired-mask loader; kspace headers only."""
    numerical_runtime()
    import contextlib,io
    import h5py
    import numpy as np
    import scipy.io
    import torch
    require(torch.get_default_dtype()==torch.float32,'source builder requires the original float32 default')
    root=C/'trainer/paired_gs';namespace={'torch':torch,'np':np,'h5py':h5py,'scipy':__import__('scipy'),'Tuple':Tuple}
    methods={}
    for file,cls,names in ((root/'trainers/joint_trainer.py','JointTrainer',{'_build_freq_weight'}),
                           (root/'data/dataset.py','MRIDataset',{'_infer_coil_axis','_conform_mask_shape','_load_external_mask'})):
        tree=ast.parse(file.read_text(encoding='utf-8'));body=next(n.body for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
        selected=[n for n in body if isinstance(n,ast.FunctionDef) and n.name in names]
        require(len(selected)==len(names),'frozen source method set changed')
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(file),'exec'),namespace)
        methods.update({name:namespace[name] for name in names})
    sources,masks,domains={},{},{}
    for scan,channel in (('pre','plain'),('post','contrast')):
        data=config['data'][channel]
        require(data.get('presampled',True) is False and not data.get('center_kspace_dc',False),
                'source mask reconstruction needs frozen external-mask simulated layout without measurement-derived recentering')
        data_path,mask_path=Path(data['data_path']),Path(data['mask_path'])
        require(all(str(p) in manifest['files'] and sha(p)==manifest['files'][str(p)] for p in (data_path,mask_path)),
                'source domain/acquired-mask files are not the frozen inputs')
        with h5py.File(data_path,'r') as handle:
            key=next((k for k in ('kspace','ksp') if k in handle),next(iter(handle)))
            shape=tuple(n for n in handle[key].shape if n!=1)
        require(len(shape)==4,'unsupported source header layout; no measurement values are used for weights')
        stub=types.SimpleNamespace();axis=methods['_infer_coil_axis'](stub,shape)
        volume=shape[:axis]+shape[axis+1:];stub.volume_shape=volume;stub.num_coils=shape[axis]
        stub._conform_mask_shape=types.MethodType(methods['_conform_mask_shape'],stub)
        with contextlib.redirect_stdout(io.StringIO()):mask=methods['_load_external_mask'](stub,str(mask_path))
        masks[scan]=mask.contiguous().numpy()
        trainer=types.SimpleNamespace(config=config,volume_shape=volume,device=torch.device('cpu'))
        override=config['loss'].get('kspace_high_weight_plain') if scan=='pre' else None
        weight=methods['_build_freq_weight'](trainer,high_override=override)
        require(weight is not None,'source builder cannot silently substitute missing weights')
        sources[scan]=weight.detach().contiguous().numpy()
        domains[scan]={'volume_shape':list(volume),'num_coils':stub.num_coils,
                       'domain_header':str(data_path),'original_acquired_mask':str(mask_path),
                       'measurement_values_read':False,'GT_or_ROI_used':False}
    return sources,masks,domains


def apply_acceptance(previous,trials,anchor_score):
    require([t.get('local_factor') for t in trials]==[1.]
            and all(all(t.get(k)==v for k,v in profile(1.,t['local_factor']).items()) for t in trials),
            'aggregate exactly one selected phase-first-order factor1 candidate')
    for trial in trials:validate_phase_metadata(trial)
    incumbent=anchor_score
    for trial in trials:
        require(math.isfinite(trial['score']),'nonfinite completed score')
        trial.update(completed_index=len(previous['trials'])+1,anchor_incumbent=anchor_score,
                     improves_anchor=ar.is_improvement(trial['score'],anchor_score),incumbent_before=incumbent,
                     accepted=ar.is_improvement(trial['score'],incumbent))
        trial['decision']='accepted' if trial['accepted'] else 'rejected'
        trial['decision_scope']='single phase-first-order candidate; global promotion belongs to root after verified aggregation'
        previous['trials'].append(trial);previous['status_counts']['completed']+=1
        previous['status_counts']['accepted' if trial['accepted'] else 'rejected']+=1
        if trial['accepted']:incumbent=trial['score']
    return incumbent


def validate_prior_info(info, mode, local_factor):
    """Owned clone of reviewed D guards, with explicit TV/wavelet factor only."""
    tv, wave = [.1*local_factor,.1*local_factor], [.12*local_factor,.1*local_factor]
    common = {"lambda_tv":tv, "lambda_phase":[.03,.03], "lambda_wavelet":wave,
              "wavelet":"db4", "wavelet_levels":1, "wavelet_on":"complex", "wavelet_eps":0.,
              "channel_reduction":"sum"}
    require(all(info.get(k) == v for k,v in common.items()), "actual factor-scaled prior strengths/modes differ")
    if mode == "single":
        require(info.get("wavelet_shift") == [1,1,1] and "wavelet_phases" not in info,
                "single wavelet phase differs")
        return
    require("wavelet_shift" not in info and info.get("mode") == "uniform8"
            and info.get("wavelet_shifts") == SHIFTS and info.get("wavelet_phase_count") == 8
            and info.get("wavelet_phase_weights") == [.125] * 8
            and info.get("wavelet_transforms_per_evaluation") == 8, "explicit uniform8 phase composition differs")
    base = info["base_tv_phase"]
    require(base.get("lambda_tv") == tv and base.get("lambda_phase") == [.03,.03]
            and base.get("lambda_wavelet") == [0.,0.], "TV/phase must occur once outside wavelet average")
    phases = info["wavelet_phases"]
    require(len(phases) == 8, "eight actual wavelet phase components required")
    for phase, shift in zip(phases, SHIFTS):
        require(phase.get("wavelet_shift") == shift and phase.get("lambda_wavelet") == [v/8 for v in wave]
                and phase.get("lambda_tv") == [0.,0.] and phase.get("lambda_phase") == [0.,0.],
                "wavelet average coefficient or duplicated TV/phase differs")

    fixed = dict(tv_eps=.04,phase_eps=.04,w_floor=0.,tv_vector=True,
                 wavelet="db4",wavelet_levels=1,wavelet_on="complex",wavelet_eps=0.,channel_reduction="sum")
    kappas=info.get("kappa",[])
    require(len(kappas)==2 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in kappas),
            "positive fixed source-ZF kappas required")
    for part in (info,base,*phases):
        require(all(part.get(k)==v for k,v in fixed.items()) and part.get("kappa")==kappas,
                "fixed source modes/eps/kappas differ across prior components")


def validate_endpoint_residual(trace, *, phase_curvature_mode='first_order'):
    """Check honest scalar evidence for one counted, diagnostic-only frozen-H call."""
    value=trace.get('true_residual',{})
    fields={'diagnostic_only','rhs_norm','norm','relative','recursive_norm','recursive_relative',
            'relative_norm_discrepancy','elapsed_seconds','normal_evaluations'}
    require(phase_curvature_mode in ('first_order','full'),'unknown endpoint curvature mode')
    if phase_curvature_mode=='first_order':fields.add('phase_curvature_mode')
    require(set(value)==fields and value.get('diagnostic_only') is True
            and (phase_curvature_mode=='full' or value.get('phase_curvature_mode')=='first_order')
            and type(value.get('normal_evaluations')) is int and value['normal_evaluations']==1,
            'one actual diagnostic-only endpoint normal evaluation required')
    scalars=fields-{'diagnostic_only','normal_evaluations','phase_curvature_mode'}
    require(all(type(value.get(k)) in (int,float) and math.isfinite(value[k]) and value[k]>=0 for k in scalars)
            and value['rhs_norm']>0 and value['elapsed_seconds']>0,'invalid endpoint diagnostic scalars/time')
    history=trace['cg_residual_history'];rhs=value['rhs_norm']
    require(type(trace.get('gradient_norm')) in (int,float) and math.isfinite(trace['gradient_norm'])
            and math.isclose(rhs,.5*trace['gradient_norm'],rel_tol=1e-12,abs_tol=1e-15)
            and math.isclose(rhs,history[0],rel_tol=1e-12,abs_tol=1e-15),'endpoint RHS norm differs from actual gradient/initial residual')
    require(value['recursive_norm']==history[-1]
            and math.isclose(value['relative'],value['norm']/rhs,rel_tol=1e-12,abs_tol=1e-15)
            and math.isclose(value['recursive_relative'],history[-1]/rhs,rel_tol=1e-12,abs_tol=1e-15)
            and math.isclose(value['relative_norm_discrepancy'],abs(value['norm']-history[-1])/rhs,
                             rel_tol=1e-12,abs_tol=1e-15),'endpoint true/recursive residual arithmetic differs')
    return value


def validate_protocol(provenance, metrics, t, local_factor):
    """Owned reviewed-D work/Armijo guards; inspect actual factor-aware records."""
    wavelet_mode="uniform8"
    require(type(local_factor) in (int,float) and local_factor==1.,"formal phase-first-order candidate is selected factor1")
    require(provenance.get("status") == "complete", "execution is partial/no_update/failed/pending, not a completed trial")
    require(provenance.get("mode") == metrics.get("mode") == "frequency_mm", "not a frequency-MM strength solve")
    require(provenance.get("method_variant") == metrics.get("method_variant") == "phase_firstorder", "variant differs")
    require(provenance.get("source_iteration") == metrics.get("source_iteration") == 4000, "source iteration differs")
    require(provenance.get("source_checkpoint_sha256") == SOURCE_SHA, "source checkpoint differs")
    phase_profile = profile(t,local_factor)
    for value in (provenance,metrics):validate_phase_metadata(value)
    require(all(provenance.get(k) == v and metrics.get(k) == v for k,v in phase_profile.items()),
            "actual wavelet profile differs or silently ignores an individual phase")
    require(provenance.get('true_residual_diagnostics') is True and metrics.get('true_residual_diagnostics') is True,
            'actual precision diagnostic flag must be boolean true')
    require(provenance.get("solver_status") == metrics.get("solver_status") == "complete", "solver is not complete")
    require(type(provenance.get("total_elapsed_seconds")) in (int, float)
            and math.isfinite(provenance["total_elapsed_seconds"]) and provenance["total_elapsed_seconds"] > 0,
            "positive measured execution runtime required")
    cfg = provenance["solver_config"]
    expected = {"outer_steps":5, "cg_steps":96, "cg_rtol":.001, "damping_rel":.001,
                "max_vjp_pairs":500, "max_render_pairs":700, "max_merit_forwards":100,
                "backtracks":12, "backtrack_factor":.5, "armijo":.0001,"true_residual_diagnostics":True}
    require(cfg == metrics.get("solver_config") == expected and cfg.get("true_residual_diagnostics") is True,
            "predeclared 5 x 96 plus diagnostic solver controls differ")
    work = provenance["solver_counters"]
    require(work == metrics.get("solver_counters"), "solver counter records disagree")
    keys = {"render_pairs", "renderer_calls", "vjp_pairs", "gradient_evaluations", "normal_evaluations",
            "merit_forward_evaluations", "support_forward_pairs", "wavelet_transform_batches", "wavelet_mode",
            "true_residual_evaluations", "true_residual_elapsed_seconds","full_curvature_normal_evaluations",
            "phase_first_order_normal_evaluations",*PHASE}
    validate_phase_metadata(work)
    require(set(work) == keys and work["wavelet_mode"] == wavelet_mode
            and all(type(v) is int and v >= 0 for k,v in work.items()
                    if k not in {"wavelet_mode","true_residual_elapsed_seconds",*PHASE}), "invalid actual work counters")
    require(work["true_residual_evaluations"]==5 and type(work["true_residual_elapsed_seconds"]) in (int,float)
            and math.isfinite(work["true_residual_elapsed_seconds"])
            and 0<work["true_residual_elapsed_seconds"]<=provenance["total_elapsed_seconds"],
            "five timed actual endpoint diagnostic evaluations required")
    require(work["vjp_pairs"] <= 500 and work["render_pairs"] <= 700
            and work["merit_forward_evaluations"] <= 100, "actual work exceeds declared limits")
    require(work["renderer_calls"] == 2 * work["render_pairs"], "paired renderer count differs")
    require(work["vjp_pairs"] == work["gradient_evaluations"] + work["normal_evaluations"], "VJP accounting differs")
    require(work["render_pairs"] == work["vjp_pairs"] + work["merit_forward_evaluations"]
            + work["support_forward_pairs"], "render accounting differs")
    require(work["support_forward_pairs"] in (0, 1), "support setup work differs")
    traces = provenance["trace"]
    require(len(traces) == provenance.get("accepted_outer_steps") == metrics.get("accepted_outer_steps") == 5,
            "five actual accepted outer steps are required for completion")
    require(work["gradient_evaluations"] == 5, "full-gradient count differs")
    normals, merit_calls, wavelets, diagnostic_seconds = 0, 0, 0, 0.
    for index, trace in enumerate(traces, 1):
        validate_phase_metadata(trace)
        require(trace["outer_step"] == index and trace["accepted"] is True and trace["status"] == "accepted",
                "invalid actual accepted-step trace")
        require(trace.get("wavelet_mode") == wavelet_mode and trace.get("local_factor")==local_factor,
                "actual outer wavelet mode/factor differs")
        validate_prior_info(trace["prior"], wavelet_mode, local_factor)
        require(trace["prior"]["kappa"]==traces[0]["prior"]["kappa"],"source-ZF kappas changed between outer steps")
        edge=trace.get("edge",{})
        require(all(edge.get(k)==v for k,v in dict(p=.9,kappa=30.,quantile_stride=16,
                    gates_frozen_within_outer_step=True).items())
                and all(type(edge.get(k)) in (int,float) and math.isfinite(edge[k]) and edge[k]>=0
                        for k in ("tau_plain","tau_contrast","weight_sum")),"fixed cross-edge gates differ")
        require(type(trace["cg_iterations"]) is int and 1 <= trace["cg_iterations"] <= 96
                and trace["cg_matvecs"] == trace["cg_iterations"], "actual CG work differs")
        require(trace["cg_status"] in ("converged", "maxiter")
                and trace["cg_converged"] == (trace["cg_status"] == "converged"), "CG termination differs")
        require(len(trace["cg_residual_history"]) == trace["cg_iterations"] + 1
                and all(math.isfinite(v) and v >= 0 for v in trace["cg_residual_history"]), "invalid CG residual history")
        require(trace.get('curvature_evaluations')=={'full_rayleigh':1,'reduced_cg':trace['cg_matvecs'],'reduced_endpoint':1},
                'full damping probe/reduced CG/reduced endpoint roles differ')
        rayleigh,damping=trace.get('damping_rayleigh'),trace.get('damping')
        require(type(rayleigh) in (int,float) and math.isfinite(rayleigh) and rayleigh>=0
                and type(damping) in (int,float) and math.isfinite(damping) and damping>0
                and math.isclose(damping,cfg['damping_rel']*max(rayleigh,1e-12),rel_tol=1e-12,abs_tol=1e-18),
                'original full-curvature Rayleigh damping arithmetic differs')
        before, after, slope, alpha = (trace[k] for k in ("merit_before", "merit_after", "directional_derivative", "alpha"))
        require(all(math.isfinite(v) for v in (before, after, slope, alpha)) and slope < 0
                and after < before and after <= before + cfg["armijo"] * alpha * slope, "frozen merit did not decrease")
        backs = trace["backtracking"]
        require(1 <= len(backs) <= cfg["backtracks"] and backs[-1] == {"alpha":alpha, "merit":after},
                "accepted backtracking evidence differs")
        require([b["alpha"] for b in backs] == [.5 ** j for j in range(len(backs))]
                and all(math.isfinite(b["merit"]) for b in backs), "invalid merit evaluation history")
        diagnostic=validate_endpoint_residual(trace)
        diagnostic_seconds+=diagnostic["elapsed_seconds"]
        normals += trace["cg_matvecs"] + 2  # Damping probe and endpoint diagnostic are actual work.
        merit_calls += len(backs)
        # Anchor WT + construction merit/quadratic = 3; full-gradient merit = 1;
        # Rayleigh probe = 1, endpoint diagnostic = 1; plus CG and backtracking.
        actual_wt = phase_profile["wavelet_phase_count"] * (6 + trace["cg_matvecs"] + len(backs))
        require(type(trace.get("wavelet_transform_batches")) is int
                and trace["wavelet_transform_batches"] == actual_wt, "actual per-outer wavelet-transform work differs")
        wavelets += actual_wt
    require(work["normal_evaluations"] == normals and work["merit_forward_evaluations"] == merit_calls,
            "trace and total actual work disagree")
    require(work['full_curvature_normal_evaluations']==len(traces)
            and work['phase_first_order_normal_evaluations']==sum(t['cg_matvecs']+1 for t in traces)
            and work['normal_evaluations']==work['full_curvature_normal_evaluations']+work['phase_first_order_normal_evaluations'],
            'actual full/reduced normal work roles differ')
    require(work["wavelet_transform_batches"] == wavelets, "actual total wavelet-transform work differs")
    require(math.isclose(work["true_residual_elapsed_seconds"],diagnostic_seconds,rel_tol=1e-12,abs_tol=1e-15),
            "endpoint diagnostic total timing differs")
    operator = provenance["operator_counters"]
    require(operator == {"render_pairs":work["render_pairs"], "renderer_calls":work["renderer_calls"],
                         "adjoint_calls":0, "vjp_calls":0,
                         "data_loss_calls":work["vjp_pairs"] + work["merit_forward_evaluations"],
                         "kspace_loss_calls":2 * (work["vjp_pairs"] + work["merit_forward_evaluations"])},
            "independent operator and solver counters disagree")
    require(provenance.get("original_adam_unchanged") is True and provenance.get("geometry_unchanged") is True,
            "original Adam or geometry was modified")
    before, after = provenance["before"], provenance["after"]
    require(before["adam"] == after["adam"] and before["geometry_sha256"] == after["geometry_sha256"],
            "actual Adam/geometry hashes changed")
    require(set(before["parameter_sha256"]) == set(after["parameter_sha256"]) == set(NAMES), "seven model groups required")
    require(all(before["parameter_sha256"][n] == after["parameter_sha256"][n] for n in NAMES[:3]), "geometry tensors changed")
    require(all(before["adam"][n]["step"] == 4000 for n in NAMES), "source Adam clock changed")
    for trace in traces:
        converged=trace['cg_residual_history'][-1]<=cfg['cg_rtol']*trace['cg_residual_history'][0]
        require(trace['cg_converged'] is converged
                and trace['cg_status']==('converged' if converged else 'maxiter')
                and (converged or trace['cg_iterations']==96),'CG termination is incompatible with actual cap/tolerance')
    require(any(before['parameter_sha256'][n]!=after['parameter_sha256'][n] for n in NAMES[3:]),
            'completed scientific solve has no actual density update')
    return work


def validate_trace_file(provenance, trace_file):
    require(len(trace_file) == len(provenance["trace"]) == 5, "outer trace file length differs")
    work = dict(render_pairs=provenance["solver_counters"]["support_forward_pairs"],
                renderer_calls=2 * provenance["solver_counters"]["support_forward_pairs"],
                vjp_pairs=0, gradient_evaluations=0, normal_evaluations=0,
                merit_forward_evaluations=0, support_forward_pairs=provenance["solver_counters"]["support_forward_pairs"],
                wavelet_mode=provenance["wavelet_mode"], wavelet_transform_batches=0,
                true_residual_evaluations=0,true_residual_elapsed_seconds=0.,
                full_curvature_normal_evaluations=0,phase_first_order_normal_evaluations=0,**PHASE)
    for expected, actual in zip(provenance["trace"], trace_file):
        validate_phase_metadata(expected)
        require({k:v for k,v in actual.items() if k != "work"} == expected,
                "outer trace file disagrees with final provenance")
        diagnostic=validate_endpoint_residual(expected)
        normals, merits = expected["cg_matvecs"] + 2, len(expected["backtracking"])
        work["gradient_evaluations"] += 1
        work["normal_evaluations"] += normals
        work['full_curvature_normal_evaluations']+=1
        work['phase_first_order_normal_evaluations']+=expected['cg_matvecs']+1
        work["merit_forward_evaluations"] += merits
        work["vjp_pairs"] += 1 + normals
        work["render_pairs"] += 1 + normals + merits
        work["renderer_calls"] = 2 * work["render_pairs"]
        work["wavelet_transform_batches"] += expected["wavelet_transform_batches"]
        work["true_residual_evaluations"] += 1
        work["true_residual_elapsed_seconds"] += diagnostic["elapsed_seconds"]
        require(actual.get("work") == work, "actual per-outer cumulative work differs")
    require(work == provenance["solver_counters"], "final cumulative work differs")


def digest_tensor(tensor):
    return hashlib.sha256(memoryview(tensor.detach().cpu().contiguous().numpy()).cast("B")).hexdigest()


def validate_tiny_prior(prior,local_factor):
    """The actual CUDA tiny covers scaled wavelet value/gradient, with TV/phase0."""
    expected=dict(passed=True,local_factor=local_factor,lambda_tv=[0.,0.],lambda_phase=[0.,0.],
        lambda_wavelet=[.12*local_factor,.1*local_factor],tested_terms=['wavelet_mean8'],
        nonzero_TV_phase_tested=False,wavelet_mode='uniform8',wavelet_phase_count=8,
        fixed_wavelet_shift=None,tolerance=2e-5,wavelet_transform_batches=40,
        reference_wavelet_transform_batches=8,device='cuda:0')
    errors=prior.get('errors',{})
    require(prior.get('passed') is True and prior.get('nonzero_TV_phase_tested') is False
            and all(prior.get(k)==v for k,v in expected.items())
            and set(errors)=={'value_relative','gradient_relative','quadratic_gradient_relative'}
            and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=2e-5 for v in errors.values()),
            'actual tiny scaled wavelet/zero-TV-phase evidence differs')
    return {k:prior[k] for k in (*expected,'errors')}


def _validate_phase_tiny_solver(trace,work,cfg,*,first_order):
    """Direct arithmetic/work guard for actual tiny and unchanged full control."""
    require(len(trace)==cfg['outer_steps'] and work.get('gradient_evaluations')==len(trace)
            and work.get('support_forward_pairs')==0 and work.get('wavelet_mode')=='uniform8',
            'tiny actual outer/gradient/support work differs')
    metadata=set(PHASE) if first_order else set()
    counters={'render_pairs','renderer_calls','vjp_pairs','gradient_evaluations','normal_evaluations',
        'merit_forward_evaluations','support_forward_pairs','wavelet_transform_batches','true_residual_evaluations',
        'true_residual_elapsed_seconds','wavelet_mode'}
    if first_order:
        validate_phase_metadata(work);counters|=metadata|{'full_curvature_normal_evaluations','phase_first_order_normal_evaluations'}
    require(set(work)==counters and all(type(v) is int and v>=0 for k,v in work.items()
            if k not in metadata|{'wavelet_mode','true_residual_elapsed_seconds'}),'invalid tiny actual counters')
    normals=merits=wavelets=0;seconds=0.
    for index,row in enumerate(trace,1):
        require(row.get('outer_step')==index and row.get('accepted') is True and row.get('status')=='accepted'
                and row.get('wavelet_mode')=='uniform8','tiny solver partial or unsupported wavelet mode')
        validate_prior_info(row['prior'],'uniform8',1.)
        require(row['prior']['kappa']==[.6,.8],'tiny source phase/TV kappas differ')
        if first_order:
            validate_phase_metadata(row)
            require(row.get('curvature_evaluations')=={'full_rayleigh':1,'reduced_cg':row.get('cg_matvecs'),'reduced_endpoint':1},
                    'tiny CG/endpoint did not use actual reduced phase curvature')
        else:require(not (set(PHASE)|{'curvature_evaluations'})&set(row),'full helper control was relabelled')
        n=row.get('cg_matvecs');history=row.get('cg_residual_history',[])
        require(type(n) is int and n==row.get('cg_iterations') and 1<=n<=cfg['cg_steps']
                and len(history)==n+1 and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in history),
                'tiny actual CG budget/history differs')
        converged=history[-1]<=cfg['cg_rtol']*history[0]
        require(row.get('cg_converged') is converged and row.get('cg_status')==('converged' if converged else 'maxiter')
                and (converged or n==cfg['cg_steps']),'tiny actual CG termination differs')
        rayleigh,eta=row.get('damping_rayleigh'),row.get('damping')
        require(type(rayleigh) in (int,float) and math.isfinite(rayleigh) and rayleigh>=0
                and type(eta) in (int,float) and math.isfinite(eta) and eta>0
                and math.isclose(eta,.001*max(rayleigh,1e-12),rel_tol=1e-12,abs_tol=1e-18),
                'tiny original full Rayleigh damping differs')
        before,after,slope,alpha=(row.get(k) for k in ('merit_before','merit_after','directional_derivative','alpha'))
        require(all(type(v) in (int,float) and math.isfinite(v) for v in (before,after,slope,alpha)) and slope<0
                and after<before and after<=before+cfg['armijo']*alpha*slope,'tiny actual full-merit Armijo failed')
        backs=row.get('backtracking',[])
        require(1<=len(backs)<=cfg['backtracks'] and backs[-1]=={'alpha':alpha,'merit':after}
                and [b.get('alpha') for b in backs]==[.5**i for i in range(len(backs))]
                and all(type(b.get('merit')) in (int,float) and math.isfinite(b['merit']) for b in backs),
                'tiny actual backtracking evidence differs')
        diagnostic=validate_endpoint_residual(row,phase_curvature_mode='first_order' if first_order else 'full')
        normals+=n+2;merits+=len(backs);seconds+=diagnostic['elapsed_seconds']
        wt=8*(6+n+len(backs));wavelets+=wt
        require(row.get('wavelet_transform_batches')==wt,'tiny actual wavelet work differs')
    require(work['normal_evaluations']==normals and work['merit_forward_evaluations']==merits
            and work['vjp_pairs']==len(trace)+normals and work['render_pairs']==len(trace)+normals+merits
            and work['renderer_calls']==2*work['render_pairs'] and work['wavelet_transform_batches']==wavelets
            and work['true_residual_evaluations']==len(trace)
            and type(work['true_residual_elapsed_seconds']) in (int,float)
            and math.isclose(work['true_residual_elapsed_seconds'],seconds,rel_tol=1e-12,abs_tol=1e-15)
            and work['vjp_pairs']<=cfg['max_vjp_pairs'] and work['render_pairs']<=cfg['max_render_pairs']
            and merits<=cfg['max_merit_forwards'],'tiny solver/trace measured work differs')
    if first_order:require(work['full_curvature_normal_evaluations']==len(trace)
        and work['phase_first_order_normal_evaluations']==normals-len(trace),'tiny full/reduced work split differs')


def validate_phase_firstorder_tiny(value):
    validate_phase_metadata(value)
    require(value.get('passed') is True and value.get('local_factor')==1. and value.get('tolerance')==2e-5
            and value.get('device')=='cuda:0' and value.get('source_density_unchanged') is True
            and value.get('phase_image_vjp_evaluations')==4
            and value.get('lambda_tv')==[.1,.1] and value.get('lambda_phase')==[.03,.03]
            and value.get('lambda_wavelet')==[.12,.1],'actual nonzero phase tiny proof/source strengths differ')
    for key in ('phase_gradient_norm','phase_full_probe_gradient_norm'):
        require(type(value.get(key)) in (int,float) and math.isfinite(value[key]) and value[key]>0,
                'tiny must exercise nonzero source phase gradient/full curvature')
    require(value.get('phase_reduced_probe_gradient_norm')==0.,'tiny phase curvature was not actually omitted')
    for key in ('source_phase_gradient_relative_error','source_phase_value_relative_error','full_control_eta_relative_error'):
        require(type(value.get(key)) in (int,float) and math.isfinite(value[key]) and 0<=value[key]<=2e-5,
                'tiny actual source phase/full-control numerical parity failed')
    cfg={'outer_steps':2,'cg_steps':4,'cg_rtol':.001,'damping_rel':.001,'max_vjp_pairs':40,'max_render_pairs':100,
         'max_merit_forwards':24,'backtracks':12,'backtrack_factor':.5,'armijo':.0001,'true_residual_diagnostics':True}
    control_cfg={**cfg,'outer_steps':1,'cg_steps':1}
    require(value.get('solver_config')==cfg and value.get('full_control_solver_config')==control_cfg
            and value.get('solver_status')==value.get('full_control_solver_status')=='complete'
            and value.get('accepted_outer_steps')==2,'actual tiny/control solver config or completion differs')
    trace,control=value.get('trace',[]),value.get('full_control_trace',[])
    _validate_phase_tiny_solver(trace,value['solver_counters'],cfg,first_order=True)
    _validate_phase_tiny_solver(control,value['full_control_solver_counters'],control_cfg,first_order=False)
    eta,old=trace[0]['damping'],control[0]['damping']
    require(math.isclose(value['full_control_eta_relative_error'],abs(eta-old)/max(abs(eta),abs(old),1e-30),
                         rel_tol=1e-12,abs_tol=1e-15),'tiny full-control eta comparison differs')
    before,after,source=(value.get(k,{}) for k in ('operator_counters_before','operator_counters_after','source_check_operator_delta'))
    names={'render_pairs','renderer_calls','adjoint_calls','vjp_calls','data_loss_calls','kspace_loss_calls'}
    require(all(set(row)==names and all(type(v) is int and v>=0 for v in row.values()) for row in (before,after,source))
            and source['render_pairs'] in (1,2) and source['renderer_calls']==2*source['render_pairs']
            and all(source[k]==0 for k in names-{'render_pairs','renderer_calls'}),'tiny source-image/setup counter proof differs')
    works=(value['solver_counters'],value['full_control_solver_counters'])
    expected={k:source[k] for k in names}
    for work in works:
        for k in ('render_pairs','renderer_calls'):expected[k]+=work[k]
        expected['data_loss_calls']+=work['vjp_pairs']+work['merit_forward_evaluations']
        expected['kspace_loss_calls']+=2*(work['vjp_pairs']+work['merit_forward_evaluations'])
    require({k:after[k]-before[k] for k in names}==expected,'tiny actual source+solver+control work hidden or miscounted')
    return deepcopy(value)


def validate_saved_state(run,provenance,t,local_factor):
    numerical_runtime()
    import torch
    source_path=Path(provenance['source_checkpoint'])
    require(sha(source_path)==SOURCE_SHA,'original source4000 checkpoint bytes differ')
    source=torch.load(source_path,map_location='cpu',weights_only=True)
    require(source.get('iter')==4000 and set(source.get('model_state',{}))==set(NAMES),'source checkpoint schema differs')
    before,after=provenance['before'],provenance['after'];groups=source['optim_state']['param_groups'];ids=[]
    require(len(groups)==7,'source Adam must have seven original groups')
    for index,(name,group) in enumerate(zip(NAMES,groups)):
        tensor=source['model_state'][name]
        require(list(tensor.shape)==SHAPES[name] and tensor.dtype==torch.float32 and bool(torch.isfinite(tensor).all())
                and digest_tensor(tensor)==before['parameter_sha256'][name],'source model shape/dtype/hash differs: '+name)
        require(len(group['params'])==1 and tuple(group['betas'])==(.9,.999),'source Adam group/betas differ')
        ident=group['params'][0];ids.append(ident);state=source['optim_state']['state'][ident]
        nominal=(.0003,.002,.0003,.015,.015,.015,.015)[index]
        require(group['initial_lr']==nominal and math.isclose(group['lr'],nominal*.9995**4000,rel_tol=1e-10,abs_tol=1e-14)
                and float(state['step'])==before['adam'][name]['step']==4000,'original source Adam clock/rate differs')
        for key in ('exp_avg','exp_avg_sq'):
            value=state[key]
            require(value.shape==tensor.shape and value.dtype==tensor.dtype and bool(torch.isfinite(value).all())
                    and digest_tensor(value)==before['adam'][name][key+'_sha256'],'source Adam moment differs')
    require(len(set(ids))==7 and set(source['optim_state']['state'])==set(ids),'source Adam mapping is not one-to-one')
    require(before==read(R/'block_opt/outputs/all_500/metrics.json')['before'],'strict source snapshot differs from original control')
    del source
    path=Path(run)/'density_solver_nonresumable.pth';saved=torch.load(path,map_location='cpu',weights_only=True)
    require(not ({'iter','optim_state','optimizer_state'}&set(saved)) and saved.get('variant')=='phase_firstorder'
            and saved.get('source_iteration')==4000 and saved.get('resumable_by_original_adam') is False,
            'FM artifact is not the declared nonresumable density-CG schema')
    require(saved.get('source_checkpoint')==str(source_path) and saved.get('source_checkpoint_sha256')==SOURCE_SHA
            and saved.get('frequency_t')==t and saved.get('frequency_evidence')==provenance['frequency_evidence']
            and saved.get('execution')==provenance['execution'],'saved source/frequency/execution bindings differ')
    require(saved.get('solver_counters')==provenance['solver_counters']
            and saved.get('accepted_outer_steps')==provenance['accepted_outer_steps']
            and all(saved.get(k)==v for k,v in profile(t,local_factor).items()),'saved actual work/phase profile differs')
    validate_phase_metadata(saved)
    require(set(saved['model_state'])==set(NAMES),'saved model must contain all seven original tensors')
    for name,tensor in saved['model_state'].items():
        require(list(tensor.shape)==SHAPES[name] and tensor.dtype==torch.float32 and bool(torch.isfinite(tensor).all())
                and digest_tensor(tensor)==after['parameter_sha256'][name],'actual saved final tensor differs: '+name)
    for snapshot in (before,after):
        for key,names in (('model_sha256',NAMES),('geometry_sha256',NAMES[:3]),('density_sha256',NAMES[3:])):
            require(snapshot[key]==hashlib.sha256(''.join(snapshot['parameter_sha256'][n] for n in names).encode()).hexdigest(),
                    'composite state hash is inconsistent')
    require(before['adam']==after['adam'] and before['geometry_sha256']==after['geometry_sha256'],
            'original Adam/geometry changed')
    return {'path':str(path),'sha256':sha(path),'all_seven_final_tensor_hashes_match':True,
            'original_model_and_Adam_restored_exactly':True,'original_Adam_unchanged':True,
            'geometry_unchanged':True,'source_iteration':4000,'resumable_by_original_adam':False,**profile(t,local_factor)}


def frequency_fingerprint(value):
    return {scan:{name:{key:row[name][key] for key in ('byte_sha256','shape','dtype')}
                  for name in ('source_weight','effective_weight','acquired_mask')}
            for scan,row in value['scans'].items()}


def validate_child_command(actual,expected,bindings):
    """Resolve only gate tokens from the documented FM cwd; keep all else exact.

    Frozen FM launch records do not record cwd. Relative tokens are therefore
    accepted only when the README's explicit FM-root interpretation resolves
    to the saved absolute request and unchanged control-binding file hashes.
    """
    flags=('--review-gate','--cuda-operator-gate','--zero-audit-gate','--selected-recipe')
    counts={flag:expected.count(flag) for flag in flags}
    require(all(count in (0,1) and actual.count(flag)==count for flag,count in counts.items()),
            'missing/duplicate/unexpected child gate flag')
    def normalize(command):
        require(isinstance(command,list) and all(isinstance(v,str) for v in command),'invalid actual child argv')
        result=list(command)
        for flag in flags:
            if counts[flag]==0:continue
            require(result.count(flag)==1 and result.index(flag)+1<len(result),'missing/duplicate child gate flag')
            index=result.index(flag)+1;path=Path(result[index])
            resolved=(path if path.is_absolute() else FM/path).resolve()
            require(resolved.is_file() and str(resolved) in bindings and sha(resolved)==bindings[str(resolved)],
                    'child gate path/hash differs from its saved binding')
            result[index]=str(resolved)
        return result
    normalized=normalize(actual)
    require(normalized==normalize(expected),'actual GSTMI numerical child command differs beyond equivalent gate paths')
    return {'normalized_command':normalized,'relative_gate_resolution_root':str(FM.resolve()),
            'historical_cwd_recorded':False}


def validate_actual_config(run,t,local_factor,hashes):
    """Rebuild each clinical solve/zero YAML with its actual factor and run path."""
    import yaml
    profile(t,local_factor)
    run=Path(run);path=run/'actual_config.yaml'
    base=load(C/'runner.py','_tvwave_scoring_source_C_config')
    expected=base.build_config('audit',0,run,local_factor=local_factor,nlambda=0.)
    expected.setdefault('mode',{})['strict_cuda']=True
    expected['frequency_overrides']={'frequency_t':t,'mask_source':'original acquired mask_p/mask_c; never fit masks',
                                    'utility_sha256':hashes[str(FM/'frequency_weights.py')]}
    expected['inner_precision_overrides']={'cg_steps':96,'true_residual_diagnostics':True}
    expected['phase_firstorder_overrides']=dict(PHASE)
    actual=yaml.safe_load(path.read_text())
    require(actual==expected,'actual factor-scaled source/priors config differs')
    validate_phase_metadata(actual['phase_firstorder_overrides'])
    return {'path':str(path),'sha256':sha(path),**profile(t,local_factor),
            'matches_original_C_build_config':True}


def validate_actual_run(exp,t,local_factor,recipe_path,supervisor,invariants):
    runtime=numerical_runtime()
    import yaml
    run=FM/'outputs'/exp;hashes=supervisor.source_hashes();protocol=profile(t,local_factor)
    require(set(hashes)=={str(FM/name) for name in SOURCES},'all nine reviewed FM sources required')
    immutable={'density_operator.py':'86a3243a57bc38dc74f052f27bee30814d75d7460d93350dc53dd62583ef8bef',
        'coupled_edge.py':'0348f81227bb953001b13a636e7a92d8cfa0fb0b49fc244c5862e48987ef4ebe',
        'pcg.py':'224a1552873dd0486f3a820d986aee3e76e0376d8c2006077588788666800897',
        'frequency_weights.py':'a150298dd8de4fcaa4ab77d7b60e00c79c8f6b1da8ad5448c3fe3c7182806933'}
    require(all(hashes[str(FM/name)]==digest for name,digest in immutable.items()),'four immutable operator/edge/PCG/frequency files changed')
    complete_path=FM/'verification'/f'{exp}_completed.json';complete=read(complete_path)
    require(complete.get('passed') is True and complete.get('status')=='complete' and complete.get('returncode')==0
            and complete.get('stage')=='solve' and complete.get('code_sha256')==hashes and complete.get('invariants')==invariants
            and all(complete.get(k)==v for k,v in protocol.items()),'actual parent completion/source/frequency differs')
    launch_path=FM/'verification'/f'{exp}_launch.json';launch=read(launch_path)
    require(launch.get('stage')=='solve' and launch.get('code_sha256')==hashes and launch.get('invariants')==invariants
            and all(launch.get(k)==v for k,v in protocol.items()),'actual launch source/frequency differs')
    request=launch['parameters'];require(request==complete['parameters'] and request['exp']==exp
        and request['frequency_t']==t and request['local_factor']==local_factor,'parent bounded request differs')
    validate_phase_metadata(request)
    args=argparse.Namespace(**request,preferred_gpu=0,reserved_gpu=None)
    for name in ('review_gate','cuda_operator_gate','zero_audit_gate','selected_recipe'):
        setattr(args,name,Path(getattr(args,name)))
    require(args.selected_recipe.resolve()==Path(recipe_path).resolve(),'scorer anchor is not the root execution recipe')
    supervisor.validate_stage_gates(args,hashes,invariants)
    bound=launch['control_bindings']
    command_check=validate_child_command(launch['command'],supervisor._command(args,launch_path),bound)
    require(launch['initial_visibility']==str(launch['resource']['gpu']),'actual GPU reservation differs')
    require(launch['parent_runtime']['torch_imported'] is False
            and Path(launch['parent_runtime']['python']).resolve()==supervisor.CPU_PYTHON.resolve(),'parent was not stdlib-only NVGEN')
    require(bound==complete['control_bindings'] and all(sha(p)==h for p,h in bound.items()),
                                          'review/control/recipe/anchor binding changed')
    proof_path=Path(complete['post_run_invariants']);require(sha(proof_path)==complete['post_run_invariants_sha256'],'parent post-run proof changed')
    proof=read(proof_path);expected={'code_sha256':hashes,'invariants':invariants}
    require(proof.get('status')=='verified' and proof.get('before')==proof.get('after')==expected
            and proof.get('control_bindings_before')==proof.get('control_bindings_after')==bound,'parent before/after proofs differ')
    artifacts=supervisor.validate_child_artifacts('solve',run,hashes,invariants,**protocol)
    require(artifacts==complete['artifacts'],'actual solver artifacts changed')
    provenance,metrics=read(run/'provenance.json'),read(run/'metrics.json')
    require(provenance.get('exp')==exp and provenance.get('code_sha256')==hashes
            and provenance.get('source_checkpoint')==invariants['input_manifest']['source_checkpoint']
            and provenance.get('source_config_sha256')==invariants['source_config_sha256']
            and provenance.get('numerical_runtime')==metrics.get('numerical_runtime')==runtime,'actual source/config/GSTMI runtime differs')
    work=validate_protocol(provenance,metrics,t,local_factor)
    execution=provenance['execution']
    require(execution==metrics['execution'] and execution['child_command']==launch['command']
            and execution['source_code_sha256']==hashes
            and execution['declared_parameters']=={**protocol,'solver_config':provenance['solver_config'],
                                                   'local_factor':local_factor,'nlambda':0.},'actual declared execution parameters differ')
    validate_trace_file(provenance,[json.loads(line) for line in (run/'outer_trace.jsonl').read_text().splitlines()])
    formal_config=validate_actual_config(run,t,local_factor,hashes)
    backend=provenance['runtime_cuda']
    require(backend['prebuilt_extension']==invariants['prebuilt_extension'] and backend['backend']=='tile_cuda'
            and backend['strict_cuda'] is True and backend['use_cuda'] is True and backend['compiler_branch_bypassed'] is True,
            'clinical backend differs')
    source_config=yaml.safe_load(Path(invariants['input_manifest']['source_config']).read_text())
    sources,masks,domains=reconstruct_source_arrays(source_config,invariants['input_manifest'])
    frequency=validate_frequency_arrays(provenance['frequency_evidence'],run,t,sources,masks)
    saved=validate_saved_state(run,provenance,t,local_factor)
    zero_gate=read(args.zero_audit_gate)
    zero_run=Path(next(p for p in zero_gate['artifacts'] if Path(p).name=='provenance.json')).parent
    zero_prov,zero_metrics=read(zero_run/'provenance.json'),read(zero_run/'metrics.json')
    zero_config=validate_actual_config(zero_run,t,local_factor,hashes)
    zero_frequency=validate_frequency_arrays(zero_prov['frequency_evidence'],zero_run,t,sources,masks)
    require(frequency_fingerprint(provenance['frequency_evidence'])==frequency_fingerprint(zero_prov['frequency_evidence']),
            'formal weights/acquired masks differ from the actual same-t/factor zero control')
    require(provenance['resume']==zero_prov['resume'] and provenance['before']==zero_prov['before']
            and provenance['runtime_overrides']==zero_prov['runtime_overrides'],'source restore/scheduler/backend differs from zero control')
    zero_work=zero_prov['solver_counters']
    validate_phase_metadata(zero_work)
    require(zero_prov['before']==zero_prov['after'] and zero_work['wavelet_mode']=='uniform8'
            and all(v==0 for k,v in zero_work.items() if k not in {'wavelet_mode',*PHASE}),
            'zero control performed hidden optimization')
    tiny_gate=read(args.cuda_operator_gate)
    tiny_path=Path(next(p for p in tiny_gate['artifacts'] if Path(p).name=='preflight.json'));tiny=read(tiny_path)
    tiny_prior=validate_tiny_prior(tiny.get('prior_checks',{}),local_factor)
    tiny_phase=validate_phase_firstorder_tiny(tiny.get('phase_firstorder_checks',{}))
    # Synthetic CUDA mask values are bound by its actual numerical reader and
    # frozen source; CPU random draws are not falsely claimed to reproduce CUDA.
    import numpy as np
    tiny_sources={};tiny_masks={}
    import torch
    tree=ast.parse((C/'trainer/paired_gs/trainers/joint_trainer.py').read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='JointTrainer')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_build_freq_weight')
    namespace={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'original_frequency_builder','exec'),namespace)
    for scan in ('pre','post'):
        row=tiny['frequency_evidence']['scans'][scan];shape=tuple(row['source_weight']['shape'])
        stub=types.SimpleNamespace(config=source_config,volume_shape=shape,device=torch.device('cpu'))
        high=source_config['loss'].get('kspace_high_weight_plain') if scan=='pre' else None
        tiny_sources[scan]=namespace['_build_freq_weight'](stub,high_override=high).numpy()
        tiny_masks[scan]=np.load(row['acquired_mask']['path'],allow_pickle=False)
    tiny_frequency=validate_frequency_arrays(tiny['frequency_evidence'],tiny_path.parent,t,tiny_sources,tiny_masks)
    tiny_frequency['mask_verification']='actual CUDA-original-mask reader and source-bound synthetic fixture; CPU RNG not substituted'
    head=metrics['rows'];require(set(head)==set(objective.SCANS),'both whole-head rows required')
    for scan in objective.SCANS:
        report.head_metrics(head[scan]);require(read(run/f'scores_{scan}.json')['rows']==[head[scan]],'head score files disagree')
        check=metrics['audit_checks'][scan];old=zero_metrics['audit_checks'][scan]
        require(0<=check['zf_ratio_relative_error']<=check['zf_ratio_tolerance']==.001
                and all(math.isfinite(check[k]) and check[k]>0 and check[k]==old[k]
                        for k in ('scale_trainer','scale_gauge','factor_scale')),'original export gauge differs')
    bound_paths={str(complete_path):sha(complete_path),str(launch_path):sha(launch_path),str(proof_path):sha(proof_path),**bound,**artifacts}
    for gate_path in (args.cuda_operator_gate,args.zero_audit_gate):
        gate=read(gate_path);bound_paths[str(gate_path)]=sha(gate_path);bound_paths.update(gate['artifacts'])
        bound_paths[gate['post_run_invariants']]=gate['post_run_invariants_sha256']
        bound_paths.update(gate.get('control_bindings',{}))
    for name in ('provenance.json','metrics.json','actual_config.yaml','outer_trace.jsonl','scores_pre.json','scores_post.json',
                 'candidate_pre_complex.npy','candidate_post_complex.npy','density_solver_nonresumable.pth'):
        bound_paths[str(run/name)]=sha(run/name)
    evidence={'completed':str(complete_path),'checkpoint':saved,'actual_work':work,'accepted_outer_steps':5,**protocol,
        'formal_config_check':formal_config,'zero_config_check':zero_config,
        'source_frequency_reconstruction':domains,'frequency_array_check':frequency,
        'zero_frequency_array_check':zero_frequency,'tiny_frequency_array_check':tiny_frequency,
        'tiny_prior_coverage':tiny_prior,
        'tiny_phase_firstorder_check':tiny_phase,
        'bound_artifacts_sha256':bound_paths,'numerical_runtime':runtime,'command_path_check':command_check,
        'budget_disclosure':'5 outer density steps, cap96CG each; full-curvature Rayleigh damping, reduced phase-first-order CG/endpoint, full nonlinear merit/gradient/Armijo. Actual VJP/renderer/wavelet counts and runtime; not500Adam or equal compute, no full-MM majorization.',
        'claim_boundary':provenance['claim_boundary']}
    return run,head,provenance,evidence


def reserve_output(output):
    output=Path(output).resolve()
    require(output.is_relative_to(S) and output!=S,'fresh scoring output must be inside FM/scoring')
    output.mkdir(parents=True,exist_ok=False)
    return output


def scoring_implementations():
    expected={str(C/'objective.py'):OBJECTIVE_SHA,str(C/'report.py'):REPORT_SHA,
              str(D_SCORER):D_SCORER_SHA,str(REFERENCE_FREEZE):REFERENCE_FREEZE_SHA,
              str(ar.SCORER):ar.SCORER_HASH,str(ar.SCRIPTS/'arms.py'):ar.ARMS_HASH}
    require(all(sha(path)==digest for path,digest in expected.items()),'frozen original scoring/reference implementation changed')
    return expected


def observed_failures():
    """Actual owned FM attempts remain failures, never completed trial rows."""
    setup,formal,bindings=[],[],{}
    for path in sorted((FM/'verification').glob('*_failure.json')):
        value=read(path);name=path.name[:-len('_failure.json')];selection_path=path.with_name(name+'_selection.json')
        selection=read(selection_path)
        require(value.get('status')=='failed' and value.get('stage') in ('tiny','zero','solve')
                and value.get('reservation_token') and value.get('reservation_token')==selection.get('reservation_token')
                and value.get('parent_pid')==selection.get('parent_pid') and value.get('stage')==selection.get('stage'),
                'failure is not an actual owned FM attempt')
        row={'name':name,'path':str(path),'sha256':sha(path),'stage':value['stage'],
             'scope':'setup/control; not a completed scientific trial' if value['stage']!='solve' else 'failed formal attempt; not completed',
             'child_started':value.get('worker_pid') is not None,'returncode':value.get('returncode'),
             'error':value.get('error'),'counted_as_completed_training':False}
        (formal if value['stage']=='solve' else setup).append(row)
        bindings.update({str(path):sha(path),str(selection_path):sha(selection_path)})
    return setup,formal,bindings


def score_trial(exp,t,local_factor,recipe_path,output):
    runtime=numerical_runtime()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only scoring requires empty CUDA visibility')
    require(exp not in ('','.','..') and Path(exp).name==exp,'invalid experiment name')
    protocol=profile(t,local_factor);anchor=load_anchor(recipe_path);implementations=scoring_implementations()
    require(local_factor in anchor['recipe']['local_factor_grid'],'score factor is not the single root-selected factor1')
    output=reserve_output(output)
    write(output/'STARTED.json',{'status':'validating','exp':exp,**protocol,'anchor':anchor['bindings']})
    supervisor=load(FM/'run_gpu.py','_fm_scoring_supervisor');invariants=supervisor.snapshot_invariants()
    run,head,provenance,evidence=validate_actual_run(exp,t,local_factor,recipe_path,supervisor,invariants)
    manifest=invariants['input_manifest'];references=report.load_references(ar.TOPIC,manifest)
    require(references==anchor['previous']['references'],'all ten audited opponent/reference metrics must be unchanged')
    freeze=existing_reference_freeze(manifest,references)
    observed={**evidence['bound_artifacts_sha256'],**anchor['bindings'],**implementations,
              **supervisor.source_hashes(),str(Path(__file__).resolve()):sha(__file__),str(ar.BASELINE):sha(ar.BASELINE)}
    baseline=objective.load_baseline(ar.BASELINE);raw=output/'lesion_allmetrics.json'
    command=[str(GSTMI),'-X','utf8','-B',str(ar.SCORER),'--min_core','150','--mcgs_file',
             str(run/'candidate_{scan}_complex.npy'),'--out',str(raw)]
    started=time.perf_counter()
    with (output/'scorer.log').open('x') as stream:
        rc=subprocess.run(command,cwd=C,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','RETRO_TOPIC':str(ar.TOPIC),
            'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','PYTHONPATH':'','OMP_NUM_THREADS':'1'},
            stdout=stream,stderr=subprocess.STDOUT).returncode
    require(rc==0,'original all59 scorer failed; preserve its log and partial output')
    result=objective.score(read(raw),baseline,head)
    result.update(run_dir=str(run),method_variant='phase_firstorder',**protocol,frequency_evidence=evidence,
                  candidate_sha256={s:observed[str(run/f'candidate_{s}_complex.npy')] for s in objective.SCANS},
                  scorer_command=command)
    write(output/'objective.json',result)
    trial={'name':exp,'method_variant':'phase_firstorder',**protocol,'score':result['score'],
           'strict_success':result['strict_success'],'best_plane':result['best_plane'],
           'plane_counts':report.plane_counts(result),'whole_head_metrics':{s:report.head_metrics(head[s]) for s in objective.SCANS},
           'run_dir':str(run),'objective':str(output/'objective.json'),'objective_sha256':sha(output/'objective.json'),
           'metrics_sha256':sha(run/'metrics.json'),'source_checkpoint_sha256':SOURCE_SHA,
           'solver_elapsed_seconds':provenance['total_elapsed_seconds'],'solver_evidence':evidence,
           'candidate':{'name':exp,'local_factor':local_factor,'nlambda':0.,**protocol},
           'anchor_incumbent':anchor['score'],'anchor_bindings':anchor['bindings'],
           'improves_anchor':ar.is_improvement(result['score'],anchor['score']),
           'decision_scope':'single root-selected phase-first-order candidate; global promotion awaits verified aggregate',
           'runtime_scope':'measured child setup/solve/export/head-scoring runtime; not500Adam or equal compute'}
    enhancement=report.enhancement_bundle(ar.TOPIC,ar.ROI,references,[trial],manifest,freeze)
    require(all(sha(p)==h for p,h in observed.items()),'actual candidate/source/control/anchor changed during scoring')
    require(supervisor.snapshot_invariants()==invariants,'full original inputs/source/binary changed during scoring')
    write(output/'enhancement.json',enhancement);write(output/'trial.json',trial)
    write(output/'completed.json',{'status':'verified','exp':exp,**protocol,'returncode':rc,
        'anchor_bindings':anchor['bindings'],'raw_sha256':sha(raw),'trial_sha256':sha(output/'trial.json'),
        'objective_sha256':sha(output/'objective.json'),'enhancement_sha256':sha(output/'enhancement.json'),
        'observed_artifact_sha256_before_and_after':observed,'invariants_before_and_after':invariants,
        'scorer_command':command,'scorer_cwd':str(C),'elapsed_seconds':time.perf_counter()-started,
        'numerical_runtime':runtime,
        'scoring_policy':'All59 head-masked lesion-containing planes, bothscans/all4metrics/all10opponents. Frozen thresholds;8ULP checks recomputation only.'})
    print(json.dumps({'status':'verified','trial':str(output/'trial.json'),'S':result['score'],
        'strict_planes':trial['plane_counts']['all_eight_both_scans'],'improves_anchor':trial['improves_anchor']}),flush=True)


def aggregate(directories,recipe_path,output):
    runtime=numerical_runtime()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only aggregation requires empty CUDA visibility')
    require(len(directories)==1,'aggregate exactly one completed phase-first-order candidate')
    anchor=load_anchor(recipe_path);scoring_implementations();supervisor=load(FM/'run_gpu.py','_fm_aggregate_supervisor')
    invariants=supervisor.snapshot_invariants();baseline=objective.load_baseline(ar.BASELINE)
    trials,enhancements,seen,observed=[],[],set(),dict(anchor['bindings'])
    prior_names={t['name'] for t in anchor['previous']['trials']}
    for directory in directories:
        directory=Path(directory).resolve();require(directory.is_relative_to(S) and directory!=S,'input result must be inside FM/scoring')
        done,trial=read(directory/'completed.json'),read(directory/'trial.json')
        require(done.get('status')=='verified' and done.get('returncode')==0 and done.get('numerical_runtime')==runtime
                and done.get('anchor_bindings')==trial.get('anchor_bindings')==anchor['bindings']
                and done.get('invariants_before_and_after')==invariants,'partial/stale/different-anchor score is inadmissible')
        for name,key in (('trial.json','trial_sha256'),('objective.json','objective_sha256'),
                         ('enhancement.json','enhancement_sha256'),('lesion_allmetrics.json','raw_sha256')):
            require(sha(directory/name)==done[key],'scored output bytes changed')
            observed[str(directory/name)]=done[key]
        require(all(sha(p)==h for p,h in done['observed_artifact_sha256_before_and_after'].items()),
                'actual scored source/candidate/control evidence changed')
        observed.update(done['observed_artifact_sha256_before_and_after'])
        stored=read(directory/'objective.json');recomputed=objective.score(read(directory/'lesion_allmetrics.json'),baseline,stored['whole_head_metrics'])
        require(all(stored[k]==recomputed[k] for k in ('score','strict_success','planes','best_plane')),'objective recomputation differs')
        t=trial['frequency_t'];local_factor=trial['local_factor'];protocol=profile(t,local_factor)
        require(all(trial.get(k)==done.get(k)==stored.get(k)==v for k,v in protocol.items()),
                'scored frequency/strength/phase profile differs')
        for value in (trial,done,stored):validate_phase_metadata(value)
        require(trial.get('method_variant')==stored.get('method_variant')=='phase_firstorder'
                and trial['name']==done['exp'] and trial['name'] not in seen|prior_names
                and trial['score']==recomputed['score'] and trial['strict_success']==recomputed['strict_success']
                and trial['plane_counts']==report.plane_counts(recomputed)
                and trial['anchor_incumbent']==anchor['score']
                and trial['improves_anchor']==ar.is_improvement(trial['score'],anchor['score']),
                'duplicate or mismatched actual trial/anchor decision')
        require(trial['whole_head_metrics']=={s:report.head_metrics(stored['whole_head_metrics'][s]) for s in objective.SCANS},
                'whole-head table differs')
        seen.add(trial['name']);trials.append(trial);enhancements.append(read(directory/'enhancement.json'))
    previous=deepcopy(anchor['previous']);final=apply_acceptance(previous,trials,anchor['score'])
    setup_failures,formal_failures,failure_bindings=observed_failures();observed.update(failure_bindings)
    prior_failure_paths={row.get('path') for row in previous.get('setup_failures',[])}
    previous.setdefault('setup_failures',[]).extend(row for row in setup_failures if row['path'] not in prior_failure_paths)
    previous.setdefault('phase_firstorder_formal_attempt_failures',[]).extend(formal_failures)
    previous.setdefault('failed_trials',[]).extend({**row,'event':'trial_failed','exp':row['name'],
                                                   'method_variant':'phase_firstorder'} for row in formal_failures)
    previous.setdefault('terminal_failed_trials',[]).extend(row['name'] for row in formal_failures)
    previous['status_counts']['failed']=previous['status_counts'].get('failed',0)+len(formal_failures)
    previous['status_counts']['failure_events']=previous['status_counts'].get('failure_events',0)+len(formal_failures)
    for trial,enhancement in zip(trials,enhancements):
        previous['enhancement_fidelity']['rows'].append(next(r for r in enhancement['rows'] if r['method']==trial['name']))
        previous['enhancement_fidelity']['inputs'][trial['name']]=enhancement['inputs'][trial['name']]
    previous.update(status='completed_real_trial_snapshot',created_utc=datetime.now(timezone.utc).isoformat(),
        previous_summary=anchor['recipe']['completed_prior_summary'],artifacts={},numerical_runtime=runtime,
        phase_firstorder_family={'frequency_t':1.,'declared_local_factors':anchor['recipe']['local_factor_grid'],**PHASE,
                          'completed_local_factors':[t['local_factor'] for t in trials],
                          'cg_steps':96,'true_residual_diagnostics':True,'completed_candidates':[t['name'] for t in trials],
                          'initial_incumbent':anchor['score'],'final_incumbent':final,'anchor_bindings':anchor['bindings'],
                          'global_promotion':'root only after this single verified candidate'},
        solver_budget_disclosure=trials[-1]['solver_evidence']['budget_disclosure'])
    previous['claim_status']='Provisional: fresh repeat and matched improved GS-single required' if any(t['strict_success'] for t in trials) else 'Target unmet'
    require(all(sha(p)==h for p,h in observed.items()) and supervisor.snapshot_invariants()==invariants,
            'source/candidate/control/anchor changed during aggregation')
    load_anchor(recipe_path)
    output=reserve_output(output);write(output/'summary.json',previous)
    write(output/'validator_proof.json',{'status':'verified','anchor_bindings':anchor['bindings'],
        'all_observed_artifact_sha256':observed,'invariants_before_and_after':invariants,'numerical_runtime':runtime,
        'training_count_policy':'One real completed scientific trial appended; all prior actual trials/failures preserved. Archive rescoring/setup controls not new training.'})
    print(json.dumps({'summary':str(output/'summary.json'),'completed':previous['status_counts']['completed'],'family_incumbent':final}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);sub=p.add_subparsers(dest='action',required=True)
    score=sub.add_parser('score',allow_abbrev=False)
    score.add_argument('--exp',required=True);score.add_argument('--frequency-t',type=float,choices=(1.,),default=1.)
    score.add_argument('--local-factor',type=float,choices=(1.,),required=True)
    score.add_argument('--selected-recipe',type=Path,required=True);score.add_argument('--output',type=Path,required=True)
    combined=sub.add_parser('aggregate',allow_abbrev=False)
    combined.add_argument('--trial',type=Path,action='append',required=True)
    combined.add_argument('--selected-recipe',type=Path,required=True);combined.add_argument('--output',type=Path,required=True)
    status=sub.add_parser('status',allow_abbrev=False);status.add_argument('--exp',required=True)
    args=p.parse_args()
    if args.action=='score':score_trial(args.exp,args.frequency_t,args.local_factor,args.selected_recipe,args.output)
    elif args.action=='aggregate':aggregate(args.trial,args.selected_recipe,args.output)
    else:
        require(args.exp not in ('','.','..') and Path(args.exp).name==args.exp,'invalid experiment name')
        path=FM/'outputs'/args.exp/'provenance.json';value=read(path) if path.exists() else {}
        print(json.dumps({'experiment':args.exp,'observed_status':value.get('status','pending'),
            'parent_completion_present':(FM/'verification'/f'{args.exp}_completed.json').exists(),'scored':False,
            'accepted_outer_steps':value.get('accepted_outer_steps'),'frequency_t':value.get('frequency_t'),
            'local_factor':value.get('local_factor')}))


if __name__=='__main__':main()
