#!/usr/bin/env python3
"""Export-only G2 raw Gaussian anchor; exact frozen proximal solver, no scorer."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sys
import time

import runner as g2
import run_proxdc as route

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def helpers():
    return g2.load_module(route.OLD/'proximal_dc/runner.py','_g2_proxdc_frozen_helpers')


@lru_cache(maxsize=None)
def physics():
    return g2.load_module(route.OLD/'proximal_dc/physics.py','_g2_proxdc_exact_physics')


def progress_state():
    return helpers().progress_state()


def write_progress(directory, progress):
    # Failed scalar endpoints may be NaN/Inf; retain their actual work without
    # publishing nonstandard JSON or turning a failed endpoint into a pass.
    def safe(value):
        if isinstance(value,float) and not math.isfinite(value):return str(value)
        if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [safe(v) for v in value]
        return value
    g2.write_json(directory/'progress.json',safe(progress))


def wait_context_ack(directory, launch, *, timeout=90):
    """Keep the reserved CUDA context alive until its parent observes it."""
    if not math.isfinite(timeout) or not 0 < timeout <= 90:
        raise ValueError('context acknowledgment timeout must be in (0,90]')
    parent=launch.get('parent_pid')
    if type(parent) is not int or parent <= 0:
        raise ValueError('invalid reserved parent')
    deadline=time.monotonic()+timeout
    while time.monotonic() < deadline:
        if os.getppid() != parent:
            raise ValueError('reserved parent exited before context acknowledgment')
        try: os.kill(parent,0)
        except ProcessLookupError as error:
            raise ValueError('reserved parent is no longer alive') from error
        path=directory/'childcontext_ack.json'
        if path.exists():
            ack=route.base.read(path)
            expected=dict(schema_version=1,child_pid=os.getpid(),parent_pid=parent,
                gpu_uuid=launch['gpu_uuid'],context_observed=True)
            if ack != expected:
                raise ValueError('context acknowledgment has wrong child/parent/device/observation')
            return ack
        time.sleep(min(.05,max(0,deadline-time.monotonic())))
    raise TimeoutError('parent did not acknowledge the observed CUDA context')


def topic_array(image, gauge):
    import numpy as np
    for key in ('scale_trainer','scale_gauge','factor_scale'):
        if type(gauge.get(key)) not in (int,float) or not math.isfinite(gauge[key]) or gauge[key] <= 0:
            raise ValueError('invalid frozen reporting gauge')
    if not math.isclose(gauge['factor_scale'],gauge['scale_trainer']/gauge['scale_gauge'],rel_tol=1e-15):
        raise ValueError('inconsistent frozen reporting gauge')
    value = (image.detach().cpu().numpy().astype(np.complex64)*np.float32(gauge['factor_scale'])).astype(np.complex64)
    if value.ndim != 3 or not np.isfinite(value).all():
        raise ValueError('export requires a finite complex volume')
    return value


def reconstruct(trainer, references, gauges, directory, stage, progress):
    """Check both canonical exports before any solve; use only captured raw."""
    import numpy as np
    from runtime import zero_parity
    if stage not in ('canonical','prox'):
        raise ValueError('invalid export stage')
    legacy = helpers(); legacy.validate_acquisition(trainer)
    progress['active_domain'] = 'canonical'
    controls, canonical_work = legacy.canonical_control(trainer,backend_kind='cuda',work=progress['canonical'])
    parity = {}
    for scan in ('pre','post'):
        parity[scan] = zero_parity(topic_array(controls[scan]['canonical'],gauges[scan]),references[scan])
        if not parity[scan]['passed']:
            raise ValueError(f'{scan} fresh G2 canonical reproduction failed')
    progress['canonical_parity'] = parity
    scans, arrays = {}, {}
    for scan, tag in (('pre','p'),('post','c')):
        progress['current_scan'] = scan
        arrays[scan] = {kind:legacy.save_image(directory/f'{kind}_{scan}_complex.npy',
            controls[scan][kind],gauges[scan]) for kind in ('raw','canonical')}
        operator = physics().SenseOperator(*(getattr(trainer,key+'_'+tag) for key in ('csm','kspace_dc','mask')))
        if stage == 'prox':
            # No canonical hard-DC array enters this solve, and no hard-DC
            # mapping is applied after the solver's endpoint verification.
            progress['active_domain'] = 'proximal'
            row = dict(status='started',algorithm_executed=True,record=None,work=operator.work)
            progress['scans'][scan] = row
            try:
                actual = physics().proximal_dc(operator,controls[scan]['raw'],config=physics().ProxConfig(max_matvecs=32))
            except Exception as error:
                row.update(status='failed',work=getattr(error,'proximal_work',operator.work))
                raise
            row.update(status=actual.status,record=actual.record,work=actual.work)
            route.validate_scan(row)
            candidate = actual.image
        else:
            row = dict(status='canonical_only',algorithm_executed=False,record=None,
                work={**operator.work,'roles':{k:dict(v) for k,v in operator.roles.items()}})
            progress['scans'][scan] = row
            candidate = controls[scan]['canonical']
        arrays[scan]['candidate'] = legacy.save_image(directory/f'candidate_{scan}_complex.npy',candidate,gauges[scan])
        scans[scan] = {**row,'gauge':gauges[scan]}
    progress.update(active_domain='state_verification',current_scan=None)
    return dict(canonical_parity=parity,scans=scans,arrays=arrays,
        work_total=legacy.total_work(canonical_work,scans,construction=progress['construction']))


def tiny_stage(device, *, progress=None):
    """Synthetic CUDA physics control; canonical stage separately checks G2/render."""
    import torch
    ph = physics(); shape=(4,4,4)
    progress=progress if progress is not None else progress_state()
    generator=torch.Generator(device=device).manual_seed(235)
    csm=torch.stack((torch.ones(shape,dtype=torch.complex64,device=device),
                     torch.full(shape,2j,dtype=torch.complex64,device=device)))
    csm[:,0,0,0]=0
    mask=(torch.arange(4,device=device)[:,None,None]%2).expand(shape).float()
    data=torch.randn((2,*shape),dtype=torch.complex64,device=device,generator=generator)*mask
    raw=torch.randn(shape,dtype=torch.complex64,device=device,generator=generator)
    scans={}; identities={}
    for scan in ('pre','post'):
        progress.update(active_domain='tiny_identity',current_scan=scan)
        identity_operator=ph.SenseOperator(csm,data,mask)
        progress['tiny_identity_physics'][scan]=identity_operator.work
        identities[scan]=helpers().tiny_physics(identity_operator)
        identities[scan]['work']['roles']={k:dict(v) for k,v in identity_operator.roles.items()}
        operator=ph.SenseOperator(csm,data,mask)
        progress['active_domain']='proximal'
        row=dict(status='started',algorithm_executed=True,record=None,work=operator.work)
        progress['scans'][scan]=row
        try:
            actual=ph.proximal_dc(operator,raw,config=ph.ProxConfig(max_matvecs=32))
        except Exception as error:
            row.update(status='failed',work=getattr(error,'proximal_work',operator.work));raise
        row.update(status=actual.status,record=actual.record,work=actual.work)
        route.validate_scan(row); scans[scan]=row
    progress.update(active_domain='complete',current_scan=None)
    return dict(scans=scans,tiny_checks_passed=True,tiny_identity=identities,
        work_total=helpers().total_work(progress['canonical'],scans,identities,construction=progress['construction']))


def validate_launch(path):
    launch=route.base.read(path)
    route.validate_parameters(argparse.Namespace(**{k:launch.get(k) for k in route.PARAMETERS}))
    directory=Path(launch['run_dir']).resolve()
    if directory.parent != ROOT/'outputs' or Path(path).resolve() != directory/'launch.json':
        raise ValueError('child launch/output ownership differs')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != launch.get('gpu_uuid') or launch.get('parent_pid') != os.getppid():
        raise ValueError('child parent/visibility differs from reserved launch')
    if route.code_hashes() != launch.get('code_sha256'):
        raise ValueError('reviewed route source hashes changed')
    recipe_path=route.base.verify_descriptor(launch.get('recipe'),'selected route recipe')
    recipe=route.base.read(recipe_path)
    if recipe.get('route') != route.ROUTE or recipe.get('parameters') != {k:launch[k] for k in route.PARAMETERS} \
            or recipe.get('code_sha256') != launch['code_sha256'] or recipe.get('invariants') != launch['invariants']:
        raise ValueError('selected route recipe/launch binding differs')
    controls=launch.get('control_bindings',{})
    if not controls or any(route.base.sha256(p) != h for p,h in controls.items()):
        raise ValueError('reviewed control bytes changed before child initialization')
    if launch.get('invariants',{}).get('route') != route.ROUTE \
            or launch['invariants'].get('metric_contract') != route.METRIC_CONTRACT:
        raise ValueError('G2 three-metric input binding required')
    return launch,directory


def clinical_stage(launch,directory,progress):
    import numpy as np
    import torch
    import yaml
    if route.base.sha256(g2.CHECKPOINT) != route.base.G2_SHA256:
        raise ValueError('frozen G2 checkpoint bytes changed')
    checkpoint=torch.load(g2.CHECKPOINT,map_location='cpu',weights_only=True)
    g2.validate_g2_recipe(checkpoint); state=checkpoint['model_state']
    expected={'positions':(3600000,3),'scales':(3600000,3),'rotations':(3600000,4),
              **{name:(3600000,) for name in g2.NAMES[3:]}}
    if set(state) != set(expected) or any(tuple(state[k].shape) != shape for k,shape in expected.items()):
        raise ValueError('G2 state schema/point count differs')
    source=route.base.read(g2.G2/'metrics.json')
    source_config=yaml.safe_load((g2.G2/'actual_config.yaml').read_text())
    config=g2.build_config(source_config,directory,0)
    config['output']['use_tensorboard']=False
    (directory/'config.yaml').write_text(yaml.safe_dump(config))
    gauges={scan:{key:source['audit_checks'][scan][key] for key in ('scale_trainer','scale_gauge','factor_scale')}
            for scan in ('pre','post')}
    scales={source_config['data'][channel]['data_path']:gauges[scan]['scale_trainer']
            for channel,scan in (('plain','pre'),('contrast','post'))}
    progress['active_domain']='construction'
    with helpers().observe_source_transforms(progress['construction']):
        trainer,field,_,_=g2.setup_trainer(config,state,scales,directory,
            dict(mode='baseline',bands=4,width=64,gain_bound=.25,lr=.001))
    del checkpoint,state
    try:
        voxelizer=trainer.voxelizer
        if field is not None or voxelizer.backend != 'tile_cuda' or not voxelizer.strict_cuda or not voxelizer.use_cuda:
            raise ValueError('strict original CUDA G2 renderer required')
        for tag,scan in (('p','pre'),('c','post')):
            dataset=getattr(trainer,'ds_'+tag)
            evidence=dataset.acquired_only_evidence
            if float(dataset.normalization_scale) != gauges[scan]['scale_trainer'] or \
                    evidence.get('ground_truth_constructed') is not False or evidence.get('outside_mask_nonzero') != 0 \
                    or evidence.get('coils') != 42:
                raise ValueError('acquired-only dataset/gauge evidence differs')
        before=helpers().snapshot(trainer)
        references={scan:np.load(g2.G2/f'candidate_{scan}_complex.npy',allow_pickle=False) for scan in ('pre','post')}
        import torch
        with helpers().export_only(trainer) as calls,torch.no_grad():
            result=reconstruct(trainer,references,gauges,directory,launch['stage'],progress)
        after=helpers().snapshot(trainer)
        if before != after or any(calls.values()):
            raise ValueError('G2 model/optimizer/gradient/config state changed during export')
        return {**result,'fixed_model_unchanged':True,'source_state_before':before,'source_state_after':after,
                'forbidden_calls':calls,'data_evidence':{scan:getattr(trainer,'ds_'+tag).acquired_only_evidence
                    for scan,tag in (('pre','p'),('post','c'))}}
    finally:
        if getattr(trainer,'writer',None) is not None: trainer.writer.close()


def execute(path):
    if Path(sys.executable).resolve() != g2.GSTMI.resolve():
        raise ValueError('numerical child requires actual GSTMI Python')
    launch,directory=validate_launch(path)
    import torch
    if torch.__version__ != '2.7.1+cu118' or torch.version.cuda != '11.8' or torch.cuda.device_count() != 1:
        raise ValueError('exact GSTMI Torch/CUDA and one reserved visible GPU required')
    hold=torch.zeros(1,device='cuda:0')
    g2.write_json(directory/'childcontext_ready.json',dict(pid=os.getpid(),parent_pid=os.getppid(),gpu_uuid=launch['gpu_uuid']))
    wait_context_ack(directory,launch)
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
    progress=progress_state()
    try:
        result=tiny_stage(torch.device('cuda:0'),progress=progress) if launch['stage']=='tiny' else clinical_stage(launch,directory,progress)
        torch.cuda.synchronize()
        metrics={**{k:launch[k] for k in route.PARAMETERS},**result,'status':'complete',
            'model_schema':route.ROUTE,'metric_contract':route.METRIC_CONTRACT,
            'steps_observed':0,'training_reference_access':False,'no_hard_dc_after_solve':True,
            'g2_restore_performed':launch['stage']!='tiny','source_iteration':None if launch['stage']=='tiny' else 4500,
            'anchor_source':'synthetic' if launch['stage']=='tiny' else 'same_call_raw_Gaussian_render',
            'code_sha256':launch['code_sha256'],'invariants':launch['invariants'],
            'training_seconds':time.perf_counter()-started,'peak_memory_gb':torch.cuda.max_memory_allocated()/1024**3,
            'peak_reserved_gb':torch.cuda.max_memory_reserved()/1024**3}
        g2.write_json(directory/'metrics.json',metrics)
        g2.write_json(directory/'provenance.json',{**metrics,'launch':launch,'method':'MC-GS+proxDC'})
    finally:
        write_progress(directory,progress)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',type=Path,required=True)
    args=parser.parse_args()
    try: execute(args.launch)
    except BaseException as error:
        try:
            directory=Path(route.base.read(args.launch)['run_dir'])
            g2.write_json(directory/'numeric_failure.json',dict(status='failed',error=repr(error),pid=os.getpid()))
        except Exception: pass
        raise
