#!/usr/bin/env python3
"""Restricted H14500 export-only MC-GS+proxDC child; never a training entry."""
from __future__ import annotations
import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
R=ROOT.parent
C=R/'nonlocal_autoresearch'
PF=R/'phase_firstorder'
GSTMI=Path('/home/congcongliu/anaconda3/envs/GSTMI/bin/python')
if Path(sys.executable).resolve()!=GSTMI.resolve():
    raise ValueError('export numerical child requires actual GSTMI interpreter')
import torch
NAMES=('positions','scales','rotations','density_real','density_imag','density_real_contrast','density_imag_contrast')


def numerical_runtime():
    if Path(sys.executable).resolve()!=GSTMI.resolve() or torch.__version__!='2.7.1+cu118' or torch.version.cuda!='11.8':
        raise ValueError('require actual GSTMI Torch2.7.1+cu118/CUDA11.8')
    return dict(python=sys.executable,python_version=list(sys.version_info[:3]),torch_version=str(torch.__version__),cuda_version=torch.version.cuda)


def load(path,name):
    path=Path(path).resolve()
    if name in sys.modules:
        module=sys.modules[name]
        if Path(getattr(module,'__file__','')).resolve()!=path:raise ValueError('foreign source module binding: '+name)
        return module
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def source_paths():
    for folder in (C/'trainer',C/'trainer/paired_gs'):
        if str(folder) not in sys.path:sys.path.insert(0,str(folder))


def source_module(name,relative):
    source_paths();module=importlib.import_module(name)
    if Path(module.__file__).resolve()!=(C/'trainer/paired_gs'/relative).resolve():
        raise ValueError('foreign C source module: '+name)
    return module


def base_runner():return load(C/'runner.py','_proximal_frozen_C_runner')
def supervisor():return load(ROOT/'run_gpu.py','_proximal_supervisor')
def physics():return load(ROOT/'physics.py','_proximal_physics')


CONSTRUCTION_COUNTERS=('fft_batches','ifft_batches','scalar_fft_transforms','scalar_ifft_transforms','coil_fft_transforms','coil_ifft_transforms')
CANONICAL_COUNTERS=('gaussian_renders','canonical_export_calls','canonical_fft_batches','canonical_ifft_batches','canonical_coil_fft_transforms','canonical_coil_ifft_transforms')


def progress_state():
    return dict(active_domain='stage_setup',construction=dict.fromkeys(CONSTRUCTION_COUNTERS,0),
        canonical=dict.fromkeys(CANONICAL_COUNTERS,0),scans={},tiny_identity_physics={},
        gaussian_reference_renders=0,current_scan=None,unassigned_proximal_work=None)


@contextlib.contextmanager
def observe_source_transforms(work=None):
    """Count every normal-constructor FFT alias without changing its operands."""
    work=work if work is not None else dict.fromkeys(CONSTRUCTION_COUNTERS,0)
    modules=[source_module('data.transforms','data/transforms.py'),source_module('data.dataset','data/dataset.py'),
        source_module('trainers.joint_trainer','trainers/joint_trainer.py')]
    saved=[]
    def observe(fn,kind):
        def measured(x,*args,**kwargs):
            if x.ndim not in (3,4):raise ValueError('source FFT needs scalar-volume or coil-volume operands')
            work[kind+'_batches']+=1
            work[('scalar_' if x.ndim==3 else 'coil_')+kind+'_transforms']+=1 if x.ndim==3 else len(x)
            return fn(x,*args,**kwargs)
        return measured
    try:
        for module in modules:
            for kind in ('fft','ifft'):
                name=kind+'3c'
                if hasattr(module,name):
                    original=getattr(module,name);saved.append((module,name,original));setattr(module,name,observe(original,kind))
        yield work
    finally:
        for module,name,original in reversed(saved):setattr(module,name,original)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes(order='C')).hexdigest()


def plain(value):
    if torch.is_tensor(value):return value.item() if value.numel()==1 else value.detach().cpu().tolist()
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,Path):return str(value)
    return value


def snapshot(trainer):
    """Read state without indexing defaultdict Adam state or creating moments."""
    numerical_runtime();named=dict(trainer.model.named_parameters())
    if tuple(named)!=NAMES:raise ValueError('exactly seven source parameters required')
    params={n:tensor_hash(p) for n,p in named.items()};adam={};adam_ids={}
    for n,p in named.items():
        state=trainer.optimizer.state.get(p,{})
        adam[n]={k:float(v) if k=='step' else tensor_hash(v) for k,v in state.items() if k in ('step','exp_avg','exp_avg_sq')}
        if 'exp_avg' in adam[n]:adam[n]['exp_avg_sha256']=adam[n].pop('exp_avg')
        if 'exp_avg_sq' in adam[n]:adam[n]['exp_avg_sq_sha256']=adam[n].pop('exp_avg_sq')
        adam_ids[n]={k:id(v) for k,v in state.items() if torch.is_tensor(v)}
    identities={id(p):n for n,p in named.items()}
    groups=[{**{k:plain(v) for k,v in group.items() if k!='params'},'param_names':[identities[id(p)] for p in group['params']]} for group in trainer.optimizer.param_groups]
    joined=lambda names:hashlib.sha256(''.join(params[n] for n in names).encode()).hexdigest()
    return dict(parameter_sha256=params,parameter_identities={n:id(p) for n,p in named.items()},requires_grad={n:p.requires_grad for n,p in named.items()},
        gradient_sha256={n:tensor_hash(p.grad) if p.grad is not None else None for n,p in named.items()},
        gradient_identities={n:id(p.grad) if p.grad is not None else None for n,p in named.items()},adam=adam,adam_identities=adam_ids,
        optimizer_groups=groups,scheduler_state=plain(trainer.scheduler.state_dict()) if getattr(trainer,'scheduler',None) is not None else None,
        config_sha256=hashlib.sha256(json.dumps(plain(trainer.config),sort_keys=True,allow_nan=False).encode()).hexdigest(),
        model_sha256=joined(NAMES),geometry_sha256=joined(NAMES[:3]),density_sha256=joined(NAMES[3:]))


def restore_export_only(trainer,checkpoint,expected_recipe,checkpoint_descriptor):
    numerical_runtime()
    if checkpoint.get('iter')!=4500 or checkpoint.get('frequency_adam_recipe')!=expected_recipe:
        raise ValueError('restricted export requires exact H14500 recipe and iteration')
    before={n:id(p) for n,p in trainer.model.named_parameters()}
    module=source_module('trainers.resume_protocol','trainers/resume_protocol.py')
    result=module.restore_checkpoint(trainer.model,trainer.optimizer,trainer.scheduler,checkpoint,expected_iteration=4500)
    after={n:id(p) for n,p in trainer.model.named_parameters()}
    if before!=after:raise RuntimeError('public restore changed parameter identities')
    return dict(iteration=4500,checkpoint=checkpoint_descriptor,frequency_adam_recipe=expected_recipe,
        parameter_identities_before=before,parameter_identities_after=after,public_restore=result)


@contextlib.contextmanager
def export_only(trainer):
    calls=dict(trainer_train=0,optimizer_step=0,scheduler_step=0);saved=[]
    def blocked(name):
        def invoke(*args,**kwargs):
            calls[name]+=1;raise RuntimeError('export-only wrapper forbids '+name)
        return invoke
    try:
        for obj,attr,key in ((trainer,'train','trainer_train'),(trainer.optimizer,'step','optimizer_step'),(getattr(trainer,'scheduler',None),'step','scheduler_step')):
            if obj is None:continue
            existed=attr in vars(obj);previous=vars(obj).get(attr);saved.append((obj,attr,existed,previous))
            setattr(obj,attr,blocked(key))
        yield calls
    finally:
        for obj,attr,existed,previous in reversed(saved):
            if existed:setattr(obj,attr,previous)
            else:delattr(obj,attr)


def canonical_control(trainer,*,backend_kind,work=None):
    """Observe the verbatim C export; its same raw render is the proximal anchor."""
    numerical_runtime();base=base_runner();transforms=source_module('data.transforms','data/transforms.py')
    render_existed='_render' in vars(trainer);render_previous=vars(trainer).get('_render')
    original_render=trainer._render;original_fft=transforms.fft3c;original_ifft=transforms.ifft3c
    work=work if work is not None else dict.fromkeys(CANONICAL_COUNTERS,0)
    captured={}
    def render(channel):
        work['gaussian_renders']+=1;image=original_render(channel)
        if channel in captured:raise RuntimeError('duplicate source Gaussian render')
        captured[channel]=image.detach();return image
    def fft(x,*a,**kw):
        work['canonical_fft_batches']+=1;work['canonical_coil_fft_transforms']+=x.shape[0]
        return original_fft(x,*a,**kw)
    def ifft(x,*a,**kw):
        work['canonical_ifft_batches']+=1;work['canonical_coil_ifft_transforms']+=x.shape[0]
        return original_ifft(x,*a,**kw)
    result={}
    try:
        trainer._render=render;transforms.fft3c=fft;transforms.ifft3c=ifft
        for scan,channel in (('pre','plain'),('post','contrast')):
            work['canonical_export_calls']+=1
            canonical,zf=base.export_trainer_volume(trainer,channel)
            result[scan]=dict(raw=captured[channel],canonical=canonical,zf=zf)
    finally:
        if render_existed:trainer._render=render_previous
        else:delattr(trainer,'_render')
        transforms.fft3c=original_fft;transforms.ifft3c=original_ifft
    return result,work


def reviewed_dependencies():
    """Normal-register verified lazy dependencies; keep frozen module globals intact."""
    source_paths()
    old=list(sys.path)
    try:
        sys.path.insert(0,str(PF))
        for name in ('pcg','frequency_weights','coupled_edge','surrogates','cycle_average','density_operator','gpu_preflight'):
            load(PF/(name+'.py'),name)
        driver=load(PF/'runner.py','_proximal_reviewed_strict_factory')
    finally:sys.path[:]=old
    return base_runner(),driver,sys.modules['gpu_preflight']


def validate_acquisition(trainer):
    for tag in ('p','c'):
        acquired=getattr(trainer,'mask_'+tag);fit=getattr(trainer,'mask_fit_'+tag)
        if acquired.ndim!=3 or not torch.equal(acquired,fit) or not bool(((acquired==0)|(acquired==1)).all()):
            raise ValueError('source acquired mask must exactly equal fitting mask')
        for suffix in ('csm','kspace'):
            original=getattr(trainer,suffix+'_'+tag);actual=getattr(trainer,suffix+'_fit_'+tag)
            if original.dtype!=torch.complex64 or not torch.equal(original,actual):raise ValueError('acquired source CSM/data differs from fit')
        if not torch.equal(getattr(trainer,'kspace_dc_'+tag),getattr(trainer,'kspace_'+tag)):
            raise ValueError('canonical DC data differ from acquired source data')
    return True


def save_image(path,image,gauge):
    import numpy as np
    value=(image.detach().cpu().numpy().astype(np.complex64)*np.float32(gauge['factor_scale'])).astype(np.complex64)
    if value.ndim!=3 or not np.isfinite(value).all():raise ValueError('nonfinite/nonvolume exported image')
    path=Path(path)
    with path.open('xb') as stream:np.save(stream,value,allow_pickle=False)
    return dict(path=str(path.resolve()),sha256=sha256(path),shape=list(value.shape),dtype='complex64',gauge='source_topic')


def relative(a,b):
    aa=a.detach().flatten().to(torch.complex128);bb=b.detach().flatten().to(torch.complex128)
    return float(torch.linalg.vector_norm(aa-bb)/torch.linalg.vector_norm(bb).clamp_min(1e-30))


def tiny_physics(operator):
    """Independent source-FFT identity work, excluded from solver budget."""
    numerical_runtime()
    generator=torch.Generator(device=operator.device).manual_seed(1429)
    def random(shape):
        return torch.complex(torch.randn(shape,device=operator.device,dtype=operator.csm.real.dtype,generator=generator),
            torch.randn(shape,device=operator.device,dtype=operator.csm.real.dtype,generator=generator))
    x,z=random(operator.shape),random(operator.shape);k=random(operator.csm.shape)
    ax,ah=operator.forward(x),operator.adjoint(k)
    inner=lambda a,b:complex(torch.vdot(a.flatten().to(torch.complex128),b.flatten().to(torch.complex128)))
    left,right=inner(ax,k),inner(x,ah)
    hx,hz=operator.normal(x),operator.normal(z)
    combined=operator.normal((.3+.4j)*x+.7*z)
    hl,hr=inner(x,hz),inner(hx,z);curvature=inner(x,hx)
    result=dict(forward_adjoint_relative=abs(left-right)/max(abs(left),abs(right),1e-30),
        normal_hermitian_relative=max(abs(hl-hr)/max(abs(hl),abs(hr),1e-30),abs(curvature.imag)/max(abs(curvature.real),1e-30)),
        normal_linearity_relative=relative(combined,(.3+.4j)*hx+.7*hz),normal_curvature_real=curvature.real,work=dict(operator.work))
    if any(not math.isfinite(result[k]) or result[k]>2e-5 for k in ('forward_adjoint_relative','normal_hermitian_relative','normal_linearity_relative')) or curvature.real<=0:
        raise RuntimeError('actual tiny SENSE identities failed')
    return result


def state_checks(before,after,calls,backend):
    fields={'model_parameters_unchanged':('parameter_sha256','model_sha256','geometry_sha256','density_sha256','requires_grad'),
        'parameter_identities_unchanged':('parameter_identities',),'gradients_unchanged':('gradient_sha256','gradient_identities'),
        'adam_unchanged':('adam','optimizer_groups'),'adam_identities_unchanged':('adam_identities',),
        'scheduler_unchanged':('scheduler_state',),'config_unchanged':('config_sha256',)}
    checks={name:all(before[k]==after[k] for k in keys) for name,keys in fields.items()}
    checks['no_training_steps']=calls==dict(trainer_train=0,optimizer_step=0,scheduler_step=0)
    checks['strict_cuda_prebuilt']=all(backend.get(k) is True for k in ('strict_cuda','use_cuda','compiler_branch_bypassed')) and backend.get('backend')=='tile_cuda'
    if not all(checks.values()):raise RuntimeError('source state or strict backend changed during export')
    return checks


def total_work(canonical,scans,identity=None,reference=0,construction=None):
    first={k:v for k,v in next(iter(scans.values()))['work'].items() if k!='roles'}
    summed={k:sum(row['work'][k] for row in scans.values()) for k in first}
    identities={k:sum(row['work'][k] for row in (identity or {}).values()) for k in first}
    result=dict(canonical)
    construction=dict(construction) if construction is not None else dict.fromkeys(CONSTRUCTION_COUNTERS,0)
    result.update(gaussian_renders=canonical['gaussian_renders']+reference,gaussian_cuda_renders=canonical['gaussian_renders'],
        gaussian_reference_renders=reference,gaussian_vjps=0,physics=summed,tiny_identity_physics=identities,
        construction=construction,total_scalar_fft_transforms=construction['scalar_fft_transforms'],total_scalar_ifft_transforms=construction['scalar_ifft_transforms'],
        total_fft_batches=construction['fft_batches']+canonical['canonical_fft_batches']+summed['fft_batches']+identities['fft_batches'],
        total_ifft_batches=construction['ifft_batches']+canonical['canonical_ifft_batches']+summed['ifft_batches']+identities['ifft_batches'],
        total_coil_fft_transforms=construction['coil_fft_transforms']+canonical['canonical_coil_fft_transforms']+summed['coil_fft_transforms']+identities['coil_fft_transforms'],
        total_coil_ifft_transforms=construction['coil_ifft_transforms']+canonical['canonical_coil_ifft_transforms']+summed['coil_ifft_transforms']+identities['coil_ifft_transforms'])
    return result


def proximal_scan(operator,raw,progress,scan):
    progress['active_domain']='proximal';progress['current_scan']=scan
    row=dict(status='started',algorithm_executed=True,record=None,work=dict(operator.work))
    progress['scans'][scan]=row
    try:result=physics().proximal_dc(operator,raw)
    except Exception as error:
        row.update(status='failed',work=getattr(error,'proximal_work',{**operator.work,'roles':copy.deepcopy(operator.roles)}))
        raise
    row.update(status='completed' if result.status in ('converged','stationary') else 'failed',record=result.record,work=result.work)
    return result


def tiny_stage(progress=None):
    numerical_runtime()
    progress=progress if progress is not None else progress_state()
    base,driver,preflight=reviewed_dependencies();operator,reference,metadata=preflight.make_tiny_cuda_operator(frequency_t=1.)
    trainer=operator.trainer
    # The old factory's distinct fit support is not used in this new SENSE test.
    # Canonical and proximal paths both explicitly consume original acquired data.
    for tag in ('p','c'):
        setattr(trainer,'csm_'+tag,getattr(trainer,'csm_fit_'+tag))
        setattr(trainer,'kspace_'+tag,getattr(trainer,'kspace_fit_'+tag))
        setattr(trainer,'kspace_dc_'+tag,getattr(trainer,'kspace_fit_'+tag))
        setattr(trainer,'mask_fit_'+tag,getattr(trainer,'mask_'+tag))
    trainer.train=lambda:None;trainer.scheduler=None
    before=snapshot(trainer);ph=physics();scans={};identity={};raw_errors={}
    backend=dict(prebuilt_extension=dict(path=metadata['extension_path'],sha256=metadata['extension_sha256']),
        normal_tile_source=str(C/'trainer/paired_gs/gaussian/tile_voxelizer.py'),backend='tile_cuda',strict_cuda=True,use_cuda=True,
        compiler_branch_bypassed=True,factory_metadata=metadata)
    with export_only(trainer) as calls,torch.no_grad():
        progress['active_domain']='canonical'
        controls,canonical=canonical_control(trainer,backend_kind='cuda',work=progress['canonical'])
        for scan,tag,channel in (('pre','p','plain'),('post','c','contrast')):
            progress.update(active_domain='tiny_reference',current_scan=scan)
            progress['gaussian_reference_renders']+=1
            expected=reference.trainer._render(channel);raw_errors[scan]=relative(controls[scan]['raw'],expected)
            if raw_errors[scan]>2e-5:raise RuntimeError('source CUDA/reference raw parity failed')
            csm,y,mask=(getattr(trainer,k+'_'+tag) for k in ('csm','kspace','mask'))
            progress['active_domain']='tiny_identity';identity_op=ph.SenseOperator(csm,y,mask)
            progress['tiny_identity_physics'][scan]=identity_op.work
            identity[scan]=tiny_physics(identity_op)
            actual=proximal_scan(ph.SenseOperator(csm,y,mask),controls[scan]['raw'],progress,scan)
            scans[scan]=dict(algorithm_executed=True,record=actual.record,work=actual.work,n_coils=len(csm),
                gauge=dict(scale_trainer=1.,scale_gauge=1.,factor_scale=1.,source='synthetic_tiny'))
    progress.update(active_domain='state_verification',current_scan=None);after=snapshot(trainer)
    checks=state_checks(before,after,calls,backend)
    return dict(backend=backend,source_restore_performed=False,restored_iteration=None,source_state_before=before,source_state_after=after,
        forbidden_calls=calls,checks=checks,scans=scans,arrays={},work_total=total_work(canonical,scans,identity,reference=progress['gaussian_reference_renders'],construction=progress['construction']),
        tiny_checks=dict(tolerance=2e-5,source_raw_reference_relative=raw_errors,physics=identity,parameters_unchanged=True))


def clinical_stage(stage,directory,definition,invariants,progress=None):
    numerical_runtime()
    progress=progress if progress is not None else progress_state()
    import numpy as np
    import yaml
    base,driver,preflight=reviewed_dependencies()
    cfg=yaml.safe_load(Path(definition['actual_config']['path']).read_text())
    changes={}
    for section,key,value in (('output','output_dir',str(directory/'constructor')),('output','use_tensorboard',False),('mode','strict_cuda',True)):
        changes[section+'.'+key]=dict(source_present=key in cfg[section],source=cfg[section].get(key),effective=value);cfg[section][key]=value
    # This is an export-only construction. No clocks or training recipe are relabeled.
    with (directory/'actual_config.yaml').open('x') as stream:yaml.safe_dump(cfg,stream)
    factory=source_module('trainers.joint_trainer','trainers/joint_trainer.py').JointTrainer
    progress['active_domain']='construction'
    with observe_source_transforms(progress['construction']):
        trainer,backend=driver.construct_strict_cuda_trainer(cfg,torch.device('cuda',0),base,{'invariants':invariants},factory)
    try:
        progress['active_domain']='restore'
        checkpoint=torch.load(definition['checkpoint']['path'],map_location='cpu',weights_only=True,mmap=True)
        frozen_h=load(R/'frequency_adam/runner.py','_proximal_H_checkpoint_validator')
        frozen_h.validate_checkpoint(checkpoint,frozen_h.SOURCE_SHAPES,4500,definition['expected_model_state'],definition['expected_optimizer_actual'])
        restore=restore_export_only(trainer,checkpoint,definition['frequency_adam_recipe'],definition['checkpoint']);del checkpoint
        metadata=load(R/'frequency_adam/overrides.py','_proximal_H_optimizer_metadata')._snapshot(trainer.optimizer,trainer.scheduler)
        if metadata!=definition['expected_optimizer_actual']:raise ValueError('restored H1 actual group rates/scheduler clocks differ')
        restore['optimizer_actual']=metadata
        before=snapshot(trainer)
        if any(before[k]!=definition['expected_model_state'][k] for k in ('parameter_sha256','adam','model_sha256','geometry_sha256','density_sha256')):
            raise ValueError('restored source model/moments differ from accepted H1')
        validate_acquisition(trainer);ph=physics();scans={};arrays={}
        with export_only(trainer) as calls,torch.no_grad():
            progress['active_domain']='canonical'
            controls,canonical=canonical_control(trainer,backend_kind='cuda',work=progress['canonical'])
            for scan,tag in (('pre','p'),('post','c')):
                progress['current_scan']=scan
                row=definition['scans'][scan];gauge={k:row[k] for k in ('scale_trainer','scale_gauge','factor_scale')}
                observed=float(getattr(trainer,'ds_'+tag).normalization_scale)
                if observed!=gauge['scale_trainer']:raise ValueError('fresh source dataset normalization differs from H1')
                arrays[scan]={kind:save_image(directory/f'{kind}_{scan}_complex.npy',controls[scan][kind],gauge) for kind in ('raw','canonical')}
                ref=np.load(row['canonical_array']['path'],allow_pickle=False);current=np.load(arrays[scan]['canonical']['path'],allow_pickle=False)
                diff=float(np.linalg.norm((current-ref).astype(np.complex128).ravel())/max(np.linalg.norm(ref.astype(np.complex128).ravel()),1e-30))
                if not math.isfinite(diff) or diff>1e-5:raise RuntimeError('fresh canonical H1 complex control failed')
                csm,y,mask=(getattr(trainer,k+'_'+tag) for k in ('csm','kspace_dc','mask'))
                op=ph.SenseOperator(csm,y,mask)
                record=None;work=dict(op.work)
                if stage=='prox':
                    actual=proximal_scan(op,controls[scan]['raw'],progress,scan);record=actual.record;work=actual.work
                    arrays[scan]['candidate']=save_image(directory/f'candidate_{scan}_complex.npy',actual.image,gauge)
                else:progress['scans'][scan]=dict(status='completed',algorithm_executed=False,record=None,work=work)
                scans[scan]=dict(algorithm_executed=stage=='prox',record=record,work=work,n_coils=len(csm),gauge=gauge,
                    canonical_check=dict(relative_complex_l2=diff,tolerance=1e-5,reference=row['canonical_array']))
        progress.update(active_domain='state_verification',current_scan=None);after=snapshot(trainer)
        checks=state_checks(before,after,calls,backend)
        checks.update(source_checkpoint_matches_H1=True,source_recipe_verified=True,fit_equals_acquired=True,canonical_complex_matches_H1=True)
        return dict(backend=backend,source_restore_performed=True,restored_iteration=4500,restore_proof=restore,
            source_state_before=before,source_state_after=after,checks=checks,forbidden_calls=calls,scans=scans,arrays=arrays,
            work_total=total_work(canonical,scans,construction=progress['construction']),construction_overrides=changes)
    finally:
        if getattr(trainer,'writer',None) is not None:trainer.writer.close()


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--stage',choices=('tiny','canonical','prox'),required=True);p.add_argument('--exp',required=True)
    p.add_argument('--output',type=Path,default=ROOT/'outputs');p.add_argument('--launch-manifest',type=Path,required=True)
    p.add_argument('--review-gate',type=Path,required=True);p.add_argument('--model-definition',type=Path,required=True)
    return p


def main(argv=None):
    runtime=numerical_runtime();args=parser().parse_args(argv);started=time.perf_counter();sup=supervisor()
    hashes=sup.source_hashes();invariants=sup.snapshot_invariants(args.model_definition)
    launch,reserved=sup.validate_launch(args.launch_manifest,hashes,invariants)
    if (args.stage!=reserved.stage or args.exp!=reserved.exp or args.review_gate.resolve()!=reserved.review_gate.resolve()
            or args.model_definition.resolve()!=reserved.model_definition.resolve()
            or launch['command']!=[str(GSTMI),'-X','utf8','-B','-u',str(ROOT/'runner.py'),*sys.argv[1:]]
            or launch['parent_pid']!=os.getppid() or launch['initial_visibility']!=os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise ValueError('numeric child arguments/actual command/direct parent/visibility differ from reserved launch')
    if Path(args.output).resolve()!=(ROOT/'outputs').resolve():raise ValueError('output root differs')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('exactly one reserved CUDA device required')
    directory=(args.output/args.exp).resolve()
    if not directory.is_relative_to((ROOT/'outputs').resolve()) or directory.name!=args.exp:raise ValueError('invalid owned output')
    directory.mkdir(parents=True,exist_ok=False)
    base=dict(status='complete',stage=args.stage,exp=args.exp,**sup.stage_profile(args.stage),code_sha256=hashes,invariants=invariants,
        launch_sha256=sup.sha256(args.launch_manifest),model_definition=invariants['accepted_h1']['definition_manifest'],
        control_bindings=launch['control_bindings'],numerical_runtime=runtime,device='cuda:0')
    progress=progress_state()
    try:
        actual=tiny_stage(progress) if args.stage=='tiny' else clinical_stage(args.stage,directory,invariants['accepted_h1']['definition'],invariants,progress)
        torch.cuda.synchronize()
        success=all(row['record'] is None or row['record']['status'] in ('converged','stationary') for row in actual['scans'].values())
        value={**base,**actual,'status':'complete' if success else 'failed','total_elapsed_seconds':time.perf_counter()-started}
        progress['active_domain']='publication'
        # Recheck all source/model/input/control bytes before publishing any complete artifact.
        if hashes!=sup.source_hashes() or invariants!=sup.snapshot_invariants(args.model_definition) or sup.control_bindings(reserved)!=launch['control_bindings']:
            raise RuntimeError('source/model/input/control bytes changed in child')
        for name in ('provenance.json','metrics.json'):
            with (directory/name).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
        if not success:
            raise RuntimeError('bounded proximal solve did not meet frozen convergence/endpoint/J checks')
    except Exception as error:
        scan=progress['current_scan']
        if hasattr(error,'proximal_work') and not (scan in progress['scans'] and progress['scans'][scan]['status']=='failed'):
            progress['unassigned_proximal_work']=error.proximal_work
        failure={**base,'status':'failed','observation_status':'partial','error':repr(error),'elapsed_seconds':time.perf_counter()-started,'partial_work':progress}
        with (directory/'failure.json').open('x') as stream:json.dump(failure,stream,indent=2,allow_nan=False);stream.write('\n')
        raise


if __name__=='__main__':main()
