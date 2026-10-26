#!/usr/bin/env python3
"""Read-only full-Adam flat-frequency validation and original fixed-anchor scoring."""
from __future__ import annotations

import argparse
import ast
import hashlib
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
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
G = Path(__file__).resolve().parents[1]
R = G.parent
C = R / "nonlocal_autoresearch"
S = G / "scoring"
sys.path.insert(0, str(C))
import autoresearch as ar
import objective
import report

NAMES = ("positions", "scales", "rotations", "density_real", "density_imag",
         "density_real_contrast", "density_imag_contrast")
NOMINAL = (.0003, .002, .0003, .015, .015, .015, .015)
SOURCE_SHA = "39e1632ff07e530fb7a3454e78698363a62a856f301c4159d3b4c9a784a48c9e"
OBJECTIVE_SHA = "cb9fa0c4ae2227947e06d3384559f31138086780aa729540b6ebfd3cc0a0bd41"
REPORT_SHA = "fcce12451ba3983a00b232e4c7c34f927a7c7a6827949d9e0e5f8c657d20b800"
FREQUENCY_UTILITY_SHA = "a150298dd8de4fcaa4ab77d7b60e00c79c8f6b1da8ad5448c3fe3c7182806933"
REFERENCE_FREEZE = C / "verification/report_reference_freeze.json"
REFERENCE_FREEZE_SHA = "6ebe48fa543af773274f7d0c10bc0c21593669b9305784a8382fab7e9896e634"


def require(condition, message):
    if not condition:
        raise ValueError(message)


read, sha, write = objective.read_json, objective.sha256_file, ar.write_new


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def numerical_runtime():
    """Use the reviewed child guard before any Torch check or scoring operation."""
    helper = load(G/"runner.py","_geometry_scoring_runtime_guard")
    return helper.runtime()


def existing_reference_freeze(manifest,references):
    """Read the historical freeze only; never invoke C's create-if-missing API."""
    require(REFERENCE_FREEZE.is_file() and sha(REFERENCE_FREEZE) == REFERENCE_FREEZE_SHA,
            "existing audited preformal reference freeze is missing/changed; recreation is forbidden")
    frozen = read(REFERENCE_FREEZE)
    require(frozen.get("created_before_formal_trials") is True,"historical reference freeze is not preformal")
    report._verify_reference_inputs(frozen,manifest,ar.TOPIC,ar.ROI,references)
    return frozen


def load_anchor(recipe_path):
    """Dynamic root selection only after the completed inner-precision family."""
    recipe=read(recipe_path)
    require(recipe.get('status')=='selected' and recipe.get('selected_by')=='root'
            and recipe.get('frequency_t')==1. and recipe.get('geometry_factors')==[1.,2.]
            and recipe.get('minimum_score_improvement')==1e-4,'root-selected flat-frequency geometry1/2 recipe required')
    history, incumbent=recipe['completed_prior_summary'],recipe['incumbent']
    prior_path,current_path=Path(history['path']),Path(incumbent['path'])
    require(prior_path.is_absolute() and current_path.is_absolute()
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
            and len(previous.get('inner_precision_family',{}).get('completed_candidates',[]))==1,
            'wait for the actual complete inner-precision batch; prior counts are not hardcoded')
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


def base_state(value):
    return {key:item for key,item in value.items() if key not in ("geometry_restart", "optimizer_probe")}


def validate_optimizer(geometry, metrics, provenance, factor):
    require(factor in (1.,2.) and not isinstance(factor, bool), "only full-Adam geometry factors1/2 are declared")
    require(geometry.get("status") == provenance.get("status") == "complete", "partial/failed run is not a completed trial")
    require(provenance.get("geometry_restart") == geometry, "embedded and standalone geometry provenance disagree")
    actual = geometry["restore"]
    require(actual["factor"] == factor and actual["source_iteration"] == 4000 and actual["is_noop"] is (factor==1.)
            and actual["geometry_groups"] == list(NAMES[:3]) and actual["scheduler_clock_preserved"] is True,
            "actual geometry intervention differs")
    require(all(value.get('method_variant')=='frequency_adam' and value.get('frequency_t')==1.
                and value.get('geometry_factor')==factor for value in (geometry,metrics,provenance)),
            'actual full-Adam frequency/geometry profile differs')
    planned = geometry["planned"]
    require(planned["geometry_factor"] == factor and planned["geometry_groups"] == list(NAMES[:3])
            and planned["density_lr_factor"] == 1. and planned["betas"] == [.9,.999]
            and planned["apply_after_strict_restore_iteration"] == 4000, "planned recipe differs")
    source, applied, final = actual["source"], actual["after"], geometry["final"]
    require(geometry["after_restore"] == applied == metrics["before"]["geometry_restart"]
            and final == metrics["after"]["geometry_restart"], "actual scalar snapshots disagree")
    for snapshot, step in ((source,4000),(applied,4000),(final,4500)):
        require([group["name"] for group in snapshot["groups"]] == list(NAMES), "seven actual canonical optimizer groups required")
        scheduler = snapshot["scheduler"]
        require(scheduler["last_epoch"] == step and scheduler["step_count"] == step + 1
                and scheduler["gamma"] == .9995 and len(scheduler["base_lrs"]) == len(scheduler["last_lrs"]) == 7,
                "ordinary Adam/scheduler clocks differ")
        for i,group in enumerate(snapshot["groups"]):
            require(group["step"] == step and group["betas"] == [.9,.999]
                    and group["eps"] == 1e-8 and group["weight_decay"] == 0.
                    and scheduler["base_lrs"][i] == group["initial_lr"]
                    and scheduler["last_lrs"][i] == group["lr"], "optimizer hyperparameters or scheduler rates differ")
    for i, (old, changed, last) in enumerate(zip(source["groups"], applied["groups"], final["groups"])):
        require(old["initial_lr"] == NOMINAL[i] and math.isclose(old["lr"], NOMINAL[i] * .9995 ** 4000, rel_tol=1e-10),
                "original source rates differ")
        multiplier = factor if i < 3 else 1.
        expected = {**old, "lr":old["lr"] * multiplier, "initial_lr":old["initial_lr"] * multiplier}
        require(changed == expected, "only three geometry current/initial rates may change")
        require({k:v for k,v in last.items() if k not in ("step","lr")} ==
                {k:v for k,v in changed.items() if k not in ("step","lr")}
                and math.isclose(last["lr"], changed["lr"] * .9995 ** 500, rel_tol=1e-12),
                "final Adam rates do not match original500-step decay")
    for value in (metrics, provenance):
        require(value["mode"] == "all" and value["source_iteration"] == 4000
                and value["end_iteration"] == 4500 and value["forward_backward_evaluations"] == 500
                and value["source_checkpoint_sha256"] == SOURCE_SHA, "ordinary4000-to4500 budget/source differs")
    require(base_state(provenance["before"]) == base_state(metrics["before"])
            and base_state(provenance["after"]) == base_state(metrics["after"]), "source and metric tensor snapshots disagree")
    require(all(metrics["before"]["adam"][n]["step"] == 4000 and metrics["after"]["adam"][n]["step"] == 4500
                for n in NAMES), "all seven Adam groups must advance4000-to4500")
    return source, applied, final


def original_shifts():
    numerical_runtime()
    import torch
    generator = torch.Generator(device="cpu").manual_seed(7961)
    for _ in range(4000):
        torch.randint(0,2,(3,),generator=generator)
    return [torch.randint(0,2,(3,),generator=generator).tolist() for _ in range(500)]


def validate_frequency_use(value,evidence,dc_norm,step):
    require(set(dc_norm)=={'pre','post'} and all(type(v) in (int,float) and math.isfinite(v) and v>0
                                               for v in dc_norm.values()),'positive unchanged source DC norms required')
    expected={'frequency_t':evidence['frequency_t'],'dc_norm':dc_norm,'loss_call_scans':['pre','post'],'loss_calls_total':2*step}
    for field,name in (('effective_weight_byte_sha256','effective_weight'),('source_weight_byte_sha256','source_weight'),
                       ('acquired_mask_byte_sha256','acquired_mask')):
        expected[field]={scan:evidence['scans'][scan][name]['byte_sha256'] for scan in ('pre','post')}
    require(value.get('installed_before_first_forward') is True and value.get('actual_trainer_weights_match') is True
            and all(value.get(k)==v for k,v in expected.items()),'actual per-step frequency use differs from installed source-bound weights')
    return value


def validate_trace(traces, factor, source, frequency_evidence, dc_norm):
    require(len(traces) == 500, "exactly500 original full-Adam updates required")
    require(frequency_evidence.get('frequency_t')==1.,'formal500 trace requires flat-frequency t1')
    for index, (row,shift) in enumerate(zip(traces, original_shifts())):
        validate_frequency_use(row.get('frequency_use',{}),frequency_evidence,dc_norm,index+1)
        require(row["mode"] == "all" and row["absolute_iteration"] == 4001 + index
                and row["wavelet_shift"] == shift, "original absolute clock/wavelet sequence differs")
        require(row["gradients_finite"] is True and row["full_gradient_group_count"] == 7
                and row["updated_parameter_groups"] == list(NAMES), "all seven full-Adam groups must update")
        require(all(math.isfinite(row[k]) for k in ("loss","unclipped_grad_norm","postclip_gradient_norm")), "nonfinite observed update")
        require(row.get("nonlocal",{}).get("enabled") is False, "unexpected nonlocal prior")
        for i,group in enumerate(source["groups"]):
            lr = group["lr"] * (factor if i < 3 else 1.) * .9995 ** index
            require(math.isclose(row["learning_rates_used"][group["name"]],lr,rel_tol=1e-12)
                    and math.isclose(row["learning_rates_next"][group["name"]],lr*.9995,rel_tol=1e-12),
                    "actual per-step geometry/density LR differs")


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


def frequency_fingerprint(value):
    return {scan:{name:{key:row[name][key] for key in ('byte_sha256','shape','dtype')}
                  for name in ('source_weight','effective_weight','acquired_mask')}
            for scan,row in value['scans'].items()}


def run_profile(factor,t=1.):
    require(type(factor) in (int,float) and factor in (1.,2.)
            and type(t) in (int,float) and t in (0.,1.),'undeclared full-Adam frequency/geometry profile')
    return {'method_variant':'frequency_adam','frequency_t':float(t),'geometry_factor':float(factor)}


def validate_frequency_record(value,embedded,steps,t,factor):
    require(type(steps) is int and (steps,t,factor) in ((0,1.,1.),(2,0.,1.),(500,1.,1.),(500,1.,2.)),
            'frequency control/trial protocol differs')
    require(value.get('status')=='complete' and all(value.get(k)==v for k,v in run_profile(factor,t).items())
            and all(row.get('frequency_adam')==value for row in embedded),'frequency record incomplete or embedding differs')
    require(value.get('installed_before_first_forward') is True and value.get('completed_steps')==steps
            and value.get('loss_calls_total')==2*steps,'actual installed frequency step/loss count differs')
    evidence=value['frequency_evidence'];check=value.get('frequency_array_check',{})
    require(evidence.get('frequency_t')==t and check.get('passed') is True and check.get('frequency_t')==t,
            'independent child frequency-array proof missing or mismatched')
    final=value.get('final_weights_check',{})
    require(all(final.get(k) is True for k in ('passed','actual_trainer_weights_match',
                'source_weights_unchanged','original_masks_unchanged')),'actual final/source frequency weights or masks changed')
    norms=value.get('dc_norm',{})
    require(set(norms)=={'pre','post'} and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in norms.values()),
            'positive original DC normalization required')
    if steps:validate_frequency_use(value.get('final_frequency_use',{}),evidence,norms,steps)
    else:require(value.get('final_frequency_use') is None,'audit0 cannot claim training loss calls')
    return value


def validate_checkpoint_recipe(checkpoint,t,factor,manifest_sha):
    run_profile(factor,t)
    expected=dict(version=1,frequency_t=t,geometry_factor=factor,local_factor=1.,nlambda=0.,
        wavelet_protocol='original_single_phase',frequency_manifest_sha256=manifest_sha,requires_same_recipe_on_resume=True)
    value=checkpoint.get('frequency_adam_recipe',{})
    require(value==expected and value.get('requires_same_recipe_on_resume') is True,
            'ordinary checkpoint requires its actual frequency/geometry resume recipe')
    return value


def validate_actual_config(run,mode,steps,factor,t):
    import yaml
    require((mode,steps,factor,t) in (('audit',0,1.,1.),('all',2,1.,0.),('all',500,1.,1.),('all',500,2.,1.)),
            'actual config is not a declared control or trial')
    run=Path(run);base=load(C/'runner.py','_frequency_adam_scoring_original_C_config')
    expected=base.build_config(mode,steps,run,local_factor=1.,nlambda=0.,sigma_rel=.015)
    backend=expected.setdefault('mode',{})
    expected['campaign_overrides']['runtime_backend']={'mode.strict_cuda':{
        'source_present':'strict_cuda' in backend,'source':backend.get('strict_cuda'),'effective':True}}
    backend['strict_cuda']=True
    planned=dict(geometry_factor=factor,geometry_groups=list(NAMES[:3]),density_lr_factor=1.,betas=[.9,.999],
        apply_after_strict_restore_iteration=4000,moment_policy='retain source Adam moments and step4000; no reset',
        scheduler_policy='retain clock; scale three current/initial/base/last LRs coherently')
    expected['optimizer_overrides']=planned;expected['campaign_overrides']['geometry_restart']=planned
    expected['frequency_overrides']=dict(frequency_t=t,mask_source='original acquired mask_p/mask_c; never fit masks',
        utility_sha256=FREQUENCY_UTILITY_SHA)
    path=run/'actual_config.yaml'
    require(yaml.safe_load(path.read_text())==expected,'actual source/geometry/frequency config differs')
    return {'path':str(path),'sha256':sha(path),**run_profile(factor,t),'matches_original_C_build_config':True}


def validate_control_bindings(launch,complete,proof,bindings):
    require(launch.get('control_bindings')==complete.get('control_bindings')==bindings
            and proof.get('control_bindings_before')==proof.get('control_bindings_after')==bindings
            and all(sha(path)==digest for path,digest in bindings.items()),
            'review/control/quality anchor bindings changed before or after the actual run')
    return bindings


def validate_run(exp, factor, recipe_path, supervisor, invariants):
    observed_runtime=numerical_runtime()
    import torch
    import yaml
    hashes=supervisor.code_hashes();profile=run_profile(factor)
    require(set(hashes)=={str(G/name) for name in ('overrides.py','runner.py','run_gpu.py','frequency_weights.py')}
            and hashes[str(G/'frequency_weights.py')]==FREQUENCY_UTILITY_SHA,'all four reviewed H sources and immutable frequency utility required')
    run=G/'outputs'/exp;complete_path=G/'verification'/f'{exp}_completed.json';complete=read(complete_path)
    require(complete.get('passed') is True and complete.get('status')=='complete' and complete.get('returncode')==0
            and complete.get('stage')=='trial500' and complete.get('code_sha256')==hashes
            and complete.get('invariants')==invariants and all(complete.get(k)==v for k,v in profile.items()),
            'actual full-Adam parent completion/source/profile differs')
    proof_path=Path(complete['post_run_invariants'])
    require(sha(proof_path)==complete['post_run_invariants_sha256'],'parent post-run proof changed')
    proof=read(proof_path);expected={'code_sha256':hashes,'invariants':invariants}
    require(proof.get('status')=='verified' and proof.get('before')==proof.get('after')==expected,
            'source/input/binary before-after proof differs')
    launch_path=G/'verification'/f'{exp}_launch.json';launch=read(launch_path);request=launch['request']
    require(launch['code_sha256']==hashes and launch['invariants']==invariants
            and all(launch.get(k)==v for k,v in profile.items())
            and request['stage']=='trial500' and request['exp']==exp
            and request['geometry_factor']==factor and request['frequency_t']==1.,'launch request/source/profile differs')
    args=argparse.Namespace(**request,preferred_gpu=0,reserved_gpu=None)
    for name in ('review_gate','zero_audit_gate','noop_gate','selected_recipe'):setattr(args,name,Path(getattr(args,name)))
    require(args.selected_recipe.resolve()==Path(recipe_path).resolve(),'scorer anchor is not the actual root execution recipe')
    supervisor.validate_stage_gates(args,hashes,invariants)
    bindings=validate_control_bindings(launch,complete,proof,supervisor.control_bindings(args))
    require(launch['command']==supervisor.child_command(args,launch_path,launch['resource']['gpu'])
            and launch['initial_visibility']==str(launch['resource']['gpu']),'actual reviewed numerical child command differs')
    require(launch['parent_runtime']['torch_imported'] is False
            and Path(launch['parent_runtime']['python']).resolve()==supervisor.CPU_PYTHON.resolve(),'parent was not stdlib-only NVGEN')
    artifacts=supervisor.validate_child_artifacts(args,run,hashes,invariants)
    require(artifacts==complete['artifacts'],'completed actual artifact set changed')
    metrics,provenance,geometry,frequency=[read(run/name) for name in
        ('metrics.json','provenance.json','geometry_provenance.json','frequency_provenance.json')]
    require(geometry['runtime']==observed_runtime and geometry['code_sha256']==hashes and geometry['invariants']==invariants
            and provenance['source_checkpoint']==invariants['input_manifest']['source_checkpoint']
            and provenance['source_config']==invariants['input_manifest']['source_config']
            and provenance['source_config_sha256']==invariants['source_config_sha256'],'actual source/config/runtime bindings differ')
    source,applied,final=validate_optimizer(geometry,metrics,provenance,factor)
    validate_frequency_record(frequency,[metrics,provenance,geometry],500,1.,factor)
    candidate={'name':exp,'local_factor':1.,'nlambda':0.,'sigma_rel':.015}
    head,_=ar.validate_run(run,candidate,invariants['input_manifest'])
    helper=load(G/'runner.py','_frequency_adam_scoring_cpu_helpers')
    formal_config=validate_actual_config(run,'all',500,factor,1.)
    require(geometry['backend']['prebuilt_extension']==invariants['prebuilt_extension']
            and geometry['backend']['backend']=='tile_cuda' and geometry['backend']['strict_cuda'] is True
            and geometry['backend']['use_cuda'] is True and geometry['backend']['compiler_branch_bypassed'] is True,
            'actual strict prebuilt backend differs')
    require(base_state(metrics['before'])==read(R/'block_opt/outputs/all_500/metrics.json')['before'],
            'initial seven tensors/Adam moments differ from original source control')
    source_config=yaml.safe_load(Path(invariants['input_manifest']['source_config']).read_text())
    sources,masks,domains=reconstruct_source_arrays(source_config,invariants['input_manifest'])
    frequency_check=validate_frequency_arrays(frequency['frequency_evidence'],run,1.,sources,masks)
    traces=[json.loads(line) for line in (run/'step_trace.jsonl').read_text().splitlines()]
    validate_trace(traces,factor,source,frequency['frequency_evidence'],frequency['dc_norm'])
    require(frequency['final_frequency_use']==traces[-1]['frequency_use'],'terminal frequency-use record differs from actual last training loss')
    controls={};zero_metrics=zero_provenance=None;control_files={}
    for stage,gate_path,mode,steps,t in (('audit0',args.zero_audit_gate,'audit',0,1.),('noop2',args.noop_gate,'all',2,0.)):
        gate=read(gate_path)
        directory=Path(next(path for path in gate['artifacts'] if Path(path).name=='metrics.json')).parent
        require(directory.resolve().is_relative_to((G/'outputs').resolve()),'actual control is outside owned H runs')
        control_launch_path=G/'verification'/f'{directory.name}_launch.json'
        control_proof_path=Path(gate['post_run_invariants'])
        require(sha(control_proof_path)==gate['post_run_invariants_sha256'],'control post-run proof changed')
        validate_control_bindings(read(control_launch_path),gate,read(control_proof_path),gate['control_bindings'])
        control_files.update(gate['control_bindings'])
        control_files.update({str(path):sha(path) for path in (control_launch_path,control_proof_path)})
        cm,cp,cg,cf=[read(directory/name) for name in ('metrics.json','provenance.json','geometry_provenance.json','frequency_provenance.json')]
        require(all(all(value.get(k)==v for k,v in run_profile(1.,t).items()) for value in (cm,cp,cg)),
                'actual control frequency/geometry/method differs')
        validate_frequency_record(cf,[cm,cp,cg],steps,t,1.)
        config_check=validate_actual_config(directory,mode,steps,1.,t)
        array_check=validate_frequency_arrays(cf['frequency_evidence'],directory,t,sources,masks)
        require(cf['dc_norm']==frequency['dc_norm'],'source DC denominator changed across controls/trial')
        control_trace=[json.loads(line) for line in (directory/'step_trace.jsonl').read_text().splitlines()]
        require(len(control_trace)==steps and cm['source_iteration']==4000 and cm['end_iteration']==4000+steps
                and cm['forward_backward_evaluations']==steps,'control actual Adam budget differs')
        for index,row in enumerate(control_trace,1):validate_frequency_use(row.get('frequency_use',{}),cf['frequency_evidence'],cf['dc_norm'],index)
        if steps:require(cf['final_frequency_use']==control_trace[-1]['frequency_use'],'noop2 final loss-use record differs')
        else:
            require(cm['before']==cm['after'] and frequency_fingerprint(cf['frequency_evidence'])==frequency_fingerprint(frequency['frequency_evidence']),
                    'audit0 changed original state or differs from formal flat-frequency weights')
            zero_metrics,zero_provenance=cm,cp
        checkpoint_path=Path(cm['final_checkpoint'])
        require(checkpoint_path.resolve().is_relative_to(directory.resolve()),'control checkpoint outside owned run')
        control_checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        recipe=validate_checkpoint_recipe(control_checkpoint,t,1.,sha(directory/'frequency_weights/manifest.json'));del control_checkpoint
        controls[stage]={'config_check':config_check,'frequency_array_check':array_check,'checkpoint_recipe':recipe}
        control_files.update(array_check['bound_files_sha256'])
        for name in ('frequency_provenance.json','actual_config.yaml','step_trace.jsonl'):control_files[str(directory/name)]=sha(directory/name)
    require(all(provenance[key]==zero_provenance[key] for key in ('normalization_protocol','export_protocol')),
            'original export/gauge protocol differs from zero control')
    source_checkpoint=torch.load(invariants['input_manifest']['source_checkpoint'],map_location='cpu',weights_only=True)
    source_check=helper.validate_checkpoint(source_checkpoint,helper.SOURCE_SHAPES,4000,metrics['before'],source);del source_checkpoint
    checkpoint_path=Path(metrics['final_checkpoint'])
    require(checkpoint_path.resolve().is_relative_to(run.resolve()),'actual final checkpoint outside owned run')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    final_check=helper.validate_checkpoint(checkpoint,helper.SOURCE_SHAPES,4500,metrics['after'],final)
    final_recipe=validate_checkpoint_recipe(checkpoint,1.,factor,sha(run/'frequency_weights/manifest.json'));del checkpoint
    for scan in objective.SCANS:
        report.head_metrics(head[scan]);require(read(run/f'scores_{scan}.json')['rows']==[head[scan]],'whole-head score files disagree')
        check=metrics['audit_checks'][scan]
        require(0<=check['zf_ratio_relative_error']<=check['zf_ratio_tolerance']==.001,'physical export gauge failed')
        require(all(math.isfinite(check[k]) and check[k]>0 and check[k]==zero_metrics['audit_checks'][scan][k]
                    for k in ('scale_trainer','scale_gauge','factor_scale')),'physical export scale differs from zero control')
    bound={str(path):sha(path) for path in (complete_path,proof_path,launch_path,Path(args.review_gate),Path(args.selected_recipe))}
    bound.update(bindings)
    for path in (args.zero_audit_gate,args.noop_gate):
        gate=read(path);bound[str(path)]=sha(path);bound.update(gate['artifacts'])
        bound[gate['post_run_invariants']]=gate['post_run_invariants_sha256']
    bound.update(artifacts);bound.update(control_files);bound.update(frequency_check['bound_files_sha256'])
    for name in ('frequency_provenance.json','actual_config.yaml','step_trace.jsonl'):bound[str(run/name)]=sha(run/name)
    bound.update({str(run/f'scores_{scan}.json'):sha(run/f'scores_{scan}.json') for scan in objective.SCANS})
    evidence={'completed':str(complete_path),'completed_sha256':sha(complete_path),**profile,
        'actual_Adam_updates':500,'initial_iteration':4000,'final_iteration':4500,
        'source_checkpoint_check':source_check,'final_checkpoint_check':final_check,'checkpoint_recipe':final_recipe,
        'final_checkpoint':str(checkpoint_path),'final_checkpoint_sha256':sha(checkpoint_path),
        'formal_config_check':formal_config,'frequency_array_check':frequency_check,'source_frequency_reconstruction':domains,
        'controls':controls,'all500_original_wavelet_draws_checked':True,'all7_actual_step_rates_checked':True,
        'all500_actual_frequency_use_records_checked':True,'bound_artifacts_sha256':bound,'numerical_runtime':observed_runtime,
        'budget_disclosure':'500 original seven-group Adam updates with original single-phase wavelet RNG; not equal compute to prior density-MM/CG trials.'}
    return run,head,metrics,evidence


def reserve_output(path):
    path = path.resolve()
    require(path.is_relative_to(S) and path != S,"fresh output must be inside H/scoring")
    path.mkdir(parents=True,exist_ok=False)
    return path


def apply_acceptance(previous,trials,anchor_score):
    require([t.get('geometry_factor') for t in trials]==[1.,2.]
            and all(t.get('frequency_t')==1. for t in trials),'aggregate exactly flat-frequency geometry1 then2')
    incumbent=anchor_score
    for trial in trials:
        require(math.isfinite(trial['score']),'nonfinite actual score')
        trial.update(completed_index=len(previous['trials'])+1,anchor_incumbent=anchor_score,
                     improves_anchor=ar.is_improvement(trial['score'],anchor_score),incumbent_before=incumbent,
                     accepted=ar.is_improvement(trial['score'],incumbent))
        trial['decision']='accepted' if trial['accepted'] else 'rejected'
        trial['decision_scope']='ordered full-Adam flat-frequency family; global promotion belongs to root after both candidates'
        previous['trials'].append(trial);previous['status_counts']['completed']+=1
        previous['status_counts']['accepted' if trial['accepted'] else 'rejected']+=1
        if trial['accepted']:incumbent=trial['score']
    return incumbent


def observed_failures():
    """Actual owned H attempts remain failures, never completed trial rows."""
    setup,formal,bindings=[],[],{}
    for path in sorted((G/'verification').glob('*_failure.json')):
        value=read(path);name=path.name[:-len('_failure.json')];selection_path=path.with_name(name+'_selection.json')
        selection=read(selection_path)
        require(value.get('status')=='failed' and value.get('stage') in ('audit0','noop2','trial500')
                and value.get('reservation_token') and value.get('reservation_token')==selection.get('reservation_token')
                and value.get('parent_pid')==selection.get('parent_pid')
                and value.get('stage')==selection.get('request',{}).get('stage')
                and selection.get('request',{}).get('exp')==name,
                'failure is not an actual owned H attempt')
        row={'name':name,'path':str(path),'sha256':sha(path),'stage':value['stage'],
             'scope':'setup/control; not a completed scientific trial' if value['stage']!='trial500' else 'failed formal attempt; not completed',
             'child_started':value.get('worker_pid') is not None,'returncode':value.get('returncode'),
             'error':value.get('error'),'counted_as_completed_training':False}
        (formal if value['stage']=='trial500' else setup).append(row)
        bindings.update({str(path):sha(path),str(selection_path):sha(selection_path)})
    return setup,formal,bindings


def score_trial(exp,factor,recipe_path,output):
    runtime = numerical_runtime()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "","CPU-only scoring requires empty CVD")
    require(Path(exp).name == exp and exp not in ("", ".", ".."),"invalid experiment name")
    selected=load_anchor(recipe_path);profile=run_profile(factor);output=reserve_output(output)
    write(output/"STARTED.json",{"status":"validating","exp":exp,**profile,"anchor_bindings":selected["bindings"]})
    supervisor = load(G/"run_gpu.py","_geometry_scoring_supervisor")
    invariants = supervisor.snapshot_invariants()
    run,head,metrics,evidence = validate_run(exp,factor,recipe_path,supervisor,invariants)
    manifest = invariants["input_manifest"]
    references = report.load_references(ar.TOPIC,manifest)
    require(references==selected["previous"]["references"],"all ten frozen opponents must remain unchanged")
    freeze = existing_reference_freeze(manifest,references)
    observed = {**evidence["bound_artifacts_sha256"],str(__file__):sha(__file__),**selected["bindings"],
                str(C/"objective.py"):OBJECTIVE_SHA,str(C/"report.py"):REPORT_SHA,str(ar.BASELINE):sha(ar.BASELINE),str(ar.SCORER):sha(ar.SCORER),str(ar.SCRIPTS/"arms.py"):sha(ar.SCRIPTS/"arms.py")}
    observed.update(supervisor.code_hashes())
    observed[str(REFERENCE_FREEZE)] = REFERENCE_FREEZE_SHA
    require(observed[str(ar.SCORER)] == ar.SCORER_HASH and observed[str(ar.SCRIPTS/"arms.py")] == ar.ARMS_HASH,"original scorer changed")
    raw = output/"lesion_allmetrics.json"
    command = [str(ar.PYTHON),"-X","utf8","-B",str(ar.SCORER),"--min_core","150","--mcgs_file",
               str(run/"candidate_{scan}_complex.npy"),"--out",str(raw)]
    started = time.perf_counter()
    with (output/"scorer.log").open("x") as stream:
        rc = subprocess.run(command,cwd=C,env={**os.environ,"CUDA_VISIBLE_DEVICES":"","RETRO_TOPIC":str(ar.TOPIC),
            "PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1","PYTHONPATH":"","OMP_NUM_THREADS":"1"},
            stdout=stream,stderr=subprocess.STDOUT).returncode
    require(rc == 0,"original scorer failed; retain raw log")
    result = objective.score(read(raw),objective.load_baseline(ar.BASELINE),head)
    result.update(run_dir=str(run),**profile,geometry_evidence=evidence,
                  candidate_sha256={s:observed[str(run/f"candidate_{s}_complex.npy")] for s in objective.SCANS},scorer_command=command)
    write(output/"objective.json",result)
    trial = {"name":exp,**profile,
             "score":result["score"],"strict_success":result["strict_success"],"best_plane":result["best_plane"],
             "plane_counts":report.plane_counts(result),"whole_head_metrics":{s:report.head_metrics(head[s]) for s in objective.SCANS},
             "run_dir":str(run),"objective":str(output/"objective.json"),"objective_sha256":sha(output/"objective.json"),
             "metrics_sha256":sha(run/"metrics.json"),"training_elapsed_seconds":metrics["training_elapsed_seconds"],
             "geometry_evidence":evidence,"candidate":{"name":exp,"local_factor":1.,"nlambda":0.,**profile},
             "anchor_incumbent":selected["score"],"anchor_bindings":selected["bindings"],
             "improves_anchor":ar.is_improvement(result["score"],selected["score"]),
             "budget_disclosure":evidence["budget_disclosure"],
             "decision_scope":"acceptance against evolving family incumbent is computed only in explicit aggregate"}
    enhancement = report.enhancement_bundle(ar.TOPIC,ar.ROI,references,[trial],manifest,freeze)
    require(observed == {p:sha(p) for p in observed},"actual candidate/source/bound proof changed during scoring")
    require(supervisor.snapshot_invariants() == invariants,"full original inputs/source/binary changed during scoring")
    write(output/"enhancement.json",enhancement)
    write(output/"trial.json",trial)
    write(output/"completed.json",{"status":"verified",**profile,"exp":exp,"returncode":rc,"anchor_bindings":selected["bindings"],
        "raw_sha256":sha(raw),"trial_sha256":sha(output/"trial.json"),"objective_sha256":sha(output/"objective.json"),
        "enhancement_sha256":sha(output/"enhancement.json"),"scorer_command":command,"scorer_cwd":str(C),
        "observed_artifact_sha256_before_and_after":observed,"invariants_before_and_after":invariants,
        "elapsed_seconds":time.perf_counter()-started,"numerical_runtime":runtime})
    print(json.dumps({"status":"verified","trial":str(output/"trial.json"),"S":result["score"],
                      "strict_planes":trial["plane_counts"]["all_eight_both_scans"],"improves_anchor":trial["improves_anchor"]}),flush=True)


def aggregate(directories,recipe_path,output):
    runtime = numerical_runtime()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "","CPU-only aggregation requires empty CVD")
    require(len(directories)==2,"wait for both completed flat-frequency geometry1/2 trials")
    selected=load_anchor(recipe_path);previous=deepcopy(selected["previous"])
    trials, additions, seen = [],[],set()
    prior_names = {t["name"] for t in previous["trials"]}
    supervisor = load(G/"run_gpu.py","_geometry_scoring_aggregate_supervisor")
    invariants = supervisor.snapshot_invariants()
    all_observed=dict(selected["bindings"])
    baseline = objective.load_baseline(ar.BASELINE)
    for directory in directories:
        directory = directory.resolve(); require(directory.is_relative_to(S) and directory!=S,"scored input must be inside H/scoring")
        done,trial = read(directory/"completed.json"),read(directory/"trial.json")
        require(done["status"]=="verified" and done["returncode"]==0
                and done.get("anchor_bindings")==trial.get("anchor_bindings")==selected["bindings"],"partial/stale/different-anchor scoring is inadmissible")
        require(done["numerical_runtime"] == runtime,"completed numerical scorer runtime differs")
        require(done["invariants_before_and_after"] == invariants,"aggregation original/source/binary bindings differ")
        for name,key in (("trial.json","trial_sha256"),("objective.json","objective_sha256"),("enhancement.json","enhancement_sha256"),("lesion_allmetrics.json","raw_sha256")):
            require(sha(directory/name) == done[key],"scored result bytes changed")
            all_observed[str(directory/name)]=done[key]
        require(all(sha(p) == h for p,h in done["observed_artifact_sha256_before_and_after"].items()),"scored source/candidate evidence changed")
        all_observed.update(done["observed_artifact_sha256_before_and_after"])
        stored = read(directory/"objective.json")
        recomputed = objective.score(read(directory/"lesion_allmetrics.json"),baseline,stored["whole_head_metrics"])
        require(all(stored[k] == recomputed[k] for k in ("score","strict_success","planes","best_plane")),"objective reconstruction differs")
        profile=run_profile(trial["geometry_factor"])
        require(all(trial.get(k)==done.get(k)==stored.get(k)==v for k,v in profile.items()),"scored full-Adam profile differs")
        require(trial["name"] == done["exp"] and trial["geometry_factor"] == done["geometry_factor"]
                and trial["anchor_incumbent"]==selected["score"]
                and trial["improves_anchor"]==ar.is_improvement(trial["score"],selected["score"])
                and trial["score"] == recomputed["score"] and trial["strict_success"] == recomputed["strict_success"]
                and trial["name"] not in seen | prior_names,"duplicate or mismatched actual trial")
        require(trial["plane_counts"] == report.plane_counts(recomputed)
                and trial["whole_head_metrics"] == {s:report.head_metrics(stored["whole_head_metrics"][s]) for s in objective.SCANS},
                "trial plane counts or whole-head metrics disagree")
        seen.add(trial["name"]); trials.append(trial)
        additions.append(read(directory/"enhancement.json"))
    factors = [t["geometry_factor"] for t in trials]
    incumbent = apply_acceptance(previous,trials,selected["score"])
    setup_failures,formal_failures,failure_bindings=observed_failures();all_observed.update(failure_bindings)
    prior_failure_paths={row.get("path") for row in previous.get("setup_failures",[])}
    previous.setdefault("setup_failures",[]).extend(row for row in setup_failures if row["path"] not in prior_failure_paths)
    previous.setdefault("frequency_adam_formal_attempt_failures",[]).extend(formal_failures)
    previous.setdefault("failed_trials",[]).extend({**row,"event":"trial_failed","exp":row["name"],"method_variant":"frequency_adam"} for row in formal_failures)
    previous.setdefault("terminal_failed_trials",[]).extend(row["name"] for row in formal_failures)
    for key in ("failed","failure_events"):previous["status_counts"][key]=previous["status_counts"].get(key,0)+len(formal_failures)
    for trial,enhancement in zip(trials,additions):
        previous["enhancement_fidelity"]["rows"].append(next(r for r in enhancement["rows"] if r["method"] == trial["name"]))
        previous["enhancement_fidelity"]["inputs"][trial["name"]] = enhancement["inputs"][trial["name"]]
    previous.update(status="completed_real_trial_snapshot",created_utc=datetime.now(timezone.utc).isoformat(),
                    previous_summary=selected["recipe"]["completed_prior_summary"],
                    frequency_adam_family={"frequency_t":1.,"declared_geometry_factors":[1.,2.],"completed_geometry_factors":factors,
                        "initial_incumbent":selected["score"],"final_incumbent":incumbent,"anchor_bindings":selected["bindings"],
                        "global_promotion":"root only after both candidates","budget_disclosure":trials[-1]["budget_disclosure"]},
                    numerical_runtime=runtime,artifacts={})
    previous["claim_status"] = "Provisional: requires fresh repeat and fair improved GS-single" if any(t["strict_success"] for t in trials) else "Target unmet"
    require(all(sha(p) == h for p,h in all_observed.items()) and supervisor.snapshot_invariants() == invariants,
            "source/candidate/evidence changed during aggregation")
    load_anchor(recipe_path)
    output=reserve_output(output);write(output/"summary.json",previous)
    write(output/"validator_proof.json",{"status":"verified","anchor_bindings":selected["bindings"],
        "all_observed_artifact_sha256":all_observed,"invariants_before_and_after":invariants,"numerical_runtime":runtime,
        "training_count_policy":"Two actual completed Adam trials appended; all prior history/failures preserved. Controls/archive rescoring are not new training."})
    print(json.dumps({"summary":str(output/"summary.json"),"completed":previous["status_counts"]["completed"],"incumbent":incumbent}),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    sub = p.add_subparsers(dest="action",required=True)
    score = sub.add_parser("score",allow_abbrev=False)
    score.add_argument("--exp",required=True);score.add_argument("--geometry-factor",type=float,choices=(1.,2.),required=True)
    score.add_argument("--selected-recipe",type=Path,required=True);score.add_argument("--output",type=Path,required=True)
    combined = sub.add_parser("aggregate",allow_abbrev=False)
    combined.add_argument("--trial",type=Path,action="append",required=True)
    combined.add_argument("--selected-recipe",type=Path,required=True);combined.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    if args.action == "score": score_trial(args.exp,args.geometry_factor,args.selected_recipe,args.output)
    else: aggregate(args.trial,args.selected_recipe,args.output)


if __name__ == "__main__":
    main()
