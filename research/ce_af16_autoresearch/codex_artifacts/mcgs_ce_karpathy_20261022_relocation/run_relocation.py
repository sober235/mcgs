#!/usr/bin/env python3
"""Stdlib supervisor protocol for the fixed-count full-G2 continuation."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

import run_gpu as base

ROOT = Path(__file__).resolve().parent
OLD = base.OLD
ROUTE = 'g2_fixed_count_relocation_v1'
CHILD_RUNNER = 'relocation_runner.py'
CONTEXT_ACK = True
PARAMETERS = ('route', 'stage', 'mode', 'steps', 'requested_k', 'event_step')
NAMES = ('positions', 'scales', 'rotations', 'density_real', 'density_imag',
         'density_real_contrast', 'density_imag_contrast')
BASE_LRS = (.0006, .004, .0006, .015, .015, .015, .015)
POLICY = dict(source_iteration=4500, seed=42, point_count=3600000,
    optimizer='original_seven_group_Adam_full_state', gamma=.9995,
    restoration='original_resume_protocol_no_geometry_reapply',
    frequency='original_flat_t1_4500_observer', selection='acquired_adjoint_energy_mass_proxy_float64_stable_index',
    event='end_callback_after_first_ordinary_Adam_scheduler', requested_k=18000, event_step=1,
    transform=dict(long_axis_factor=.5, other_axis_factor=.85, offset_factor=1.,
        coefficient_factor=.5/(.5*.85*.85), donor_child_sign=-1, victim_child_sign=1,
        scale_storage='log', clamp_new_children=[-1.,1.]))
RESUME_HASH = 'c30f4711c5abd581dda4832ba9d467e91d2158396262827ac13ca9c0907fcf1a'
BINARY = OLD/'nonlocal_autoresearch/trainer/paired_gs/gaussian/cuda_ext/build/tile_voxelize_cuda_0141c525_7aedd3b0.so'
BINARY_HASH = '3973daeef5f55ebfed931c38538a37d83921a9287b96b69a431da238057194b5'
require = base.require


def code_hashes():
    # Bind the complete local Python source closure, including lazy original
    # trainer/loss/data imports. Extra source binding is deliberate and harmless.
    paths = {ROOT/name for name in (*base.PRODUCTION, CHILD_RUNNER, 'run_relocation.py',
        'relocation.py','relocation_runtime.py','proxdc_runner.py','run_proxdc.py')}
    for folder in ('nonlocal_autoresearch', 'frequency_adam', 'proximal_dc', 'phase_firstorder'):
        paths.update((OLD/folder).rglob('*.py'))
    paths.add(BINARY)
    hashes = {str(p.resolve()):base.sha256(p) for p in sorted(paths)}
    require(hashes[str(BINARY)] == BINARY_HASH, 'frozen original CUDA binary changed')
    require(hashes[str(OLD/'nonlocal_autoresearch/trainer/paired_gs/trainers/resume_protocol.py')] == RESUME_HASH,
            'original resume source changed')
    return hashes


def snapshot_invariants():
    return {**base.snapshot_invariants(), 'route':ROUTE, 'policy':POLICY,
            'metric_contract':'AF16_PSNR_SSIM_NMSE', 'decision_metrics':['psnr','ssim','nmse']}


def parameters(args):
    return {key:getattr(args,key) for key in PARAMETERS}


def validate_parameters(args):
    require(args.route == ROUTE and args.stage in ('zero','smoke','trial')
            and args.mode in ('control','relocate'), 'invalid relocation route/stage/mode')
    require(type(args.steps) is int and args.steps == {'zero':0,'smoke':2,'trial':500}[args.stage],
            'fixed zero0/smoke2/trial500 required')
    require(args.stage != 'zero' or args.mode == 'control', 'zero is control only')
    require(type(args.requested_k) is int and args.requested_k == 18000
            and type(args.event_step) is int and args.event_step == 1, 'fixed k18000/eventstep1 required')


def _number(value, label, minimum=None):
    require(type(value) in (int,float) and math.isfinite(value)
            and (minimum is None or value >= minimum), 'invalid '+label)
    return value


def _integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, 'invalid '+label)
    return value


def _sha(value):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value), 'invalid tensor/file digest')


def _tensor(record, shape, device):
    require(isinstance(record,dict) and record.get('shape') == shape and record.get('dtype') == 'torch.float32'
            and record.get('device') == device, 'tensor shape/dtype/device differs')
    _integer(record.get('tensor_id',record.get('id')), 'tensor identity',1)
    if 'data_ptr' in record: _integer(record['data_ptr'],'tensor storage',1)
    _sha(record.get('sha256'))


def _state(state, iteration, n, device):
    require(isinstance(state,dict) and state.get('schema') == 'full_g2_training_state_v1'
            and state.get('parameter_names') == list(NAMES), 'complete seven-state snapshot required')
    require(set(state.get('parameters',{})) == set(NAMES), 'original seven parameters required')
    opt, sched = state.get('optimizer'), state.get('scheduler')
    require(isinstance(opt,dict) and opt.get('type') == 'torch.optim.adam.Adam'
            and isinstance(sched,dict) and sched.get('type') == 'torch.optim.lr_scheduler.ExponentialLR',
            'original Adam/scheduler types required')
    for obj in (opt,sched): _integer(obj.get('id'),'optimizer/scheduler identity',1)
    require(sched.get('optimizer_id') == opt['id'] and len(opt.get('groups',[])) == 7
            and set(opt.get('states',{})) == set(NAMES), 'optimizer membership/state differs')
    clock=sched.get('state',{})
    require(clock.get('last_epoch') == iteration and type(clock.get('last_epoch')) is int
            and clock.get('_step_count') == iteration+1 and type(clock.get('_step_count')) is int
            and clock.get('gamma') == .9995 and clock.get('base_lrs') == list(BASE_LRS),
            'scheduler actual clocks/base rates differ')
    lrs=[]
    for i,(name,lr,group) in enumerate(zip(NAMES,BASE_LRS,opt['groups'])):
        shape=[n,3] if i<2 else [n,4] if i==2 else [n]
        param=state['parameters'][name]; _tensor(param,shape,device)
        require(param.get('requires_grad') is True, 'all seven parameters must train')
        if param.get('gradient') is not None: _tensor(param['gradient'],shape,device)
        require(group.get('name') == name and group.get('parameter_ids') == [param['tensor_id']],
                'canonical identity group differs')
        options=group.get('options',{})
        require(options.get('initial_lr') == lr and options.get('betas') == [.9,.999]
                and options.get('eps') == 1e-8 and options.get('weight_decay') == 0
                and options.get('amsgrad') is False and options.get('maximize') is False
                and options.get('capturable') is False and options.get('differentiable') is False
                and options.get('fused') is None and options.get('foreach') is None
                and options.get('decoupled_weight_decay') is False, 'original Adam options differ')
        actual_lr=_number(options.get('lr'),'actual LR',0)
        require(math.isclose(actual_lr,lr*.9995**iteration,rel_tol=1e-12), 'actual saved LR differs')
        lrs.append(actual_lr)
        row=opt['states'][name]
        require(set(row) == {'step','exp_avg','exp_avg_sq'}, 'complete Adam moments required')
        for key in ('exp_avg','exp_avg_sq'): _tensor(row[key],shape,device)
        # Adam step tensors reside on CPU for this noncapturable optimizer.
        _tensor(row['step'],[], 'cpu')
        require(type(row['step'].get('value')) in (int,float) and row['step']['value'] == iteration,
                'actual seven Adam scalar clocks differ')
        expected=hashlib.sha256(struct.pack('<f',iteration)).hexdigest()
        require(row['step']['sha256'] == expected, 'Adam clock bytes differ')
    require(clock.get('_last_lr') == lrs, 'scheduler rates disagree with Adam')
    config=state.get('config',{})
    training=config.get('training',{})
    require(training.get('seed') == 42 and type(training.get('seed')) is int
            and training.get('resume_offset') == 4500 and training.get('lr_scheduler') ==
            {'type':'exponential','gamma':.9995} and config.get('adaptive_control',{}).get('enable',False) is False
            and config.get('loss',{}).get('edge_tv_kappa_source') == 'zf', 'frozen training policy differs')


def _preserved(before, after):
    require(before['config'] == after['config'], 'training configuration changed')
    for key in ('optimizer','scheduler'):
        require(before[key]['id'] == after[key]['id'], 'optimizer/scheduler object replaced')
    for name in NAMES:
        for key in ('tensor_id','data_ptr','shape','dtype','device','requires_grad'):
            require(before['parameters'][name][key] == after['parameters'][name][key], 'Parameter object/count changed')
        for key in ('exp_avg','exp_avg_sq','step'):
            a,b=before['optimizer']['states'][name][key],after['optimizer']['states'][name][key]
            for field in ('tensor_id','data_ptr','shape','dtype','device'):
                require(a[field] == b[field], 'Adam tensor object changed')
    for a,b in zip(before['optimizer']['groups'],after['optimizer']['groups']):
        require(a['name'] == b['name'] and a['parameter_ids'] == b['parameter_ids']
                and {k:v for k,v in a['options'].items() if k!='lr'} ==
                    {k:v for k,v in b['options'].items() if k!='lr'}, 'Adam group options changed')
    require(before['optimizer']['defaults'] == after['optimizer']['defaults'], 'Adam defaults changed')


def _indices_hash(rows):
    return hashlib.sha256(b''.join(struct.pack('<q',row) for row in rows)).hexdigest()


def _zero_hash(count):
    digest=hashlib.sha256(); block=b'\0' * (1<<20)
    while count:
        amount=min(count,len(block)); digest.update(block[:amount]); count-=amount
    return digest.hexdigest()


def validate_event(event, n, requested_k, device):
    require(event.get('iteration') == 4501 and type(event.get('iteration')) is int
            and event.get('step') == 1 and type(event.get('step')) is int,
            'event must follow first ordinary update')
    selection,tx=event.get('selection',{}),event.get('transaction',{})
    require(selection.get('schema') == 'relocation_selection_v1' and tx.get('schema') == 'relocation_transaction_v1'
            and selection.get('point_count') == tx.get('point_count') == n
            and selection.get('requested_k') == requested_k, 'actual selection/schema/count differs')
    k=_integer(selection.get('k_eff'),'effective k')
    eligible=_integer(selection.get('eligible_donor_count'),'eligible count')
    for key in ('point_count','k_eff','changed_row_count','untouched_row_count'):
        _integer(tx.get(key),'transaction '+key)
    require(k == min(requested_k,eligible) and tx.get('k_eff') == k, 'effective pair count differs')
    for key,count in (('requested_victims',requested_k),('donors',k),('victims',k)):
        rows=selection.get(key)
        require(isinstance(rows,list) and len(rows)==count and all(type(v) is int and 0<=v<n for v in rows)
                and len(set(rows))==count and selection.get(key+'_sha256') == _indices_hash(rows), 'selection indices/hash differ')
    donors,victims=selection['donors'],selection['victims']
    require(not set(donors)&set(selection['requested_victims']) and victims == selection['requested_victims'][:k],
            'requested victim exclusion differs')
    for key in ('donors','victims','donors_sha256','victims_sha256'):
        require(tx.get(key) == selection[key], 'selection/transaction indices disagree')
    changed=sorted(donors+victims)
    require(tx.get('changed_indices') == changed and tx.get('changed_indices_sha256') == _indices_hash(changed)
            and tx.get('changed_row_count') == 2*k and tx.get('untouched_row_count') == n-2*k
            and tx.get('transform') == POLICY['transform'], 'fixed shape/transform evidence differs')
    require(selection.get('proxy_dtype') == 'float64' and selection.get('tie_policy') == 'original_row_index',
            'selection arithmetic policy differs')
    for key in ('scores_sha256','proxy_sha256'): _sha(selection.get(key))
    for key in ('victim_proxy','donor_scores'):
        rows=selection.get(key)
        require(isinstance(rows,list) and len(rows)==k, 'selection numeric evidence missing')
        for number in rows: _number(number,key,0)
    require(all(v>0 for v in selection['donor_scores']), 'donor scores must be strictly positive')
    before,after=tx.get('before',{}),tx.get('after',{})
    stable=('point_count','model_id','optimizer_id','scheduler_id','steps','step_ids','step_sha256',
            'resolved_group_names','group_parameter_ids','group_options','scheduler')
    for key in stable: require(key in before and before[key] == after.get(key), 'event state/identity/clock changed')
    require(before['point_count']==n and before['resolved_group_names']==list(NAMES)
            and set(before['steps'])==set(before['step_ids'])==set(before['step_sha256'])==set(NAMES)
            and all(type(v) in (int,float) and v==4501 for v in before['steps'].values())
            and before['scheduler'].get('last_epoch')==4501 and before['scheduler'].get('_step_count')==4502,
            'event actual clocks incomplete')
    for key in ('model_id','optimizer_id','scheduler_id'): _integer(before[key],key,1)
    for name in NAMES:
        _integer(before['step_ids'][name],'event Adam step identity',1)
        require(before['step_sha256'][name] == hashlib.sha256(struct.pack('<f',4501)).hexdigest(), 'event Adam step hash differs')
    require(len(before['group_parameter_ids'])==len(before['group_options'])==7,'event groups incomplete')
    for i,name in enumerate(NAMES):
        shape=[n,3] if i<2 else [n,4] if i==2 else [n]
        width=3 if i<2 else 4 if i==2 else 1
        require(before['group_parameter_ids'][i] == [before['parameters'][name]['id']], 'event group identity differs')
        options=before['group_options'][i]
        require(options.get('betas')==[.9,.999] and options.get('eps')==1e-8 and options.get('weight_decay')==0
                and options.get('amsgrad') is False and math.isclose(options.get('lr',-1),BASE_LRS[i]*.9995**4501,rel_tol=1e-12),
                'event optimizer options differ')
        triples=[('parameters',None),('gradients',None),('moments','exp_avg'),('moments','exp_avg_sq')]
        for domain,key in triples:
            a,b=before[domain][name],after[domain][name]
            if key: a,b=a[key],b[key]
            require(a is not None and b is not None,'event existing gradient/state missing')
            _tensor(a,shape,device); _tensor(b,shape,device)
            for field in ('id','shape','dtype','device','untouched_sha256'):
                require(a[field] == b[field], 'event replaced object or changed untouched rows')
            for item in (a,b):
                _sha(item['untouched_sha256']); _sha(item.get('changed_sha256'))
            if domain != 'parameters':
                require(b['changed_sha256'] == _zero_hash(2*k*width*4), 'changed rows not numerically zero')
        require(tx.get('changed_rows_zero',{}).get(name) == dict(exp_avg=True,exp_avg_sq=True,gradient=True),
                'changed row zero checks incomplete')
    if not k: require(before == after, 'no-op event changed state')



def _validate_frequency(value,directory,steps,trace,data,device):
    require(isinstance(value,dict) and value.get('status')=='complete' and value.get('frequency_t')==1.
            and value.get('geometry_factor')==2. and value.get('completed_steps')==steps
            and type(value.get('completed_steps')) is int and value.get('loss_calls_total')==2*steps
            and type(value.get('loss_calls_total')) is int and value.get('installed_before_first_forward') is True,
            'actual frozen frequency installation/count differs')
    require(value.get('final_frequency_use')==(trace[-1]['frequency_use'] if steps else None),
            'final frequency clock differs')
    require(value.get('final_weights_check')==dict(passed=True,actual_trainer_weights_match=True,
            source_weights_unchanged=True,original_masks_unchanged=True),'final source frequency checks differ')
    evidence=value.get('frequency_evidence',{})
    require(evidence==base.read(directory/'frequency_weights/manifest.json') and evidence.get('version')==1
            and evidence.get('frequency_t')==1. and evidence.get('mass_relative_tolerance')==2e-6
            and set(evidence.get('scans',{}))=={'pre','post'},'actual frequency manifest differs')
    check=value.get('frequency_array_check',{})
    require(check.get('passed') is True and check.get('frequency_t')==1.
            and check.get('mass_relative_tolerance')==2e-6 and set(check.get('scans',{}))=={'pre','post'},
            'independent saved frequency check missing')
    for scan in ('pre','post'):
        row,actual=evidence['scans'][scan],check['scans'][scan]
        _integer(row.get('acquired_count'),'acquired frequency samples',1)
        c=_number(row.get('c'),'flat frequency weight',0)
        source=_number(row.get('source_mean_squared_weight'),'source frequency mass',0)
        effective=_number(row.get('effective_mean_squared_weight'),'effective frequency mass',0)
        require(c>0 and source>0 and effective>0 and math.isclose(source,c*c,rel_tol=2e-6)
                and abs(effective/source-1.)<=2e-6,'frequency acquired mass differs')
        for key in ('acquired_count','c','source_mean_squared_weight','effective_mean_squared_weight','relative_mass_error'):
            require(actual.get(key)==row.get(key),'independent frequency numbers differ')
        require(actual.get('original_acquired_mask_exact') is True and actual.get('original_source_weight_exact') is True,
                'original frequency inputs changed')
        require(0<=_number(actual.get('relative_mass_error'),'frequency mass error')<=2e-6
                and 0<=_number(actual.get('interpolation_max_relative_error'),'frequency interpolation error')<=
                _number(actual.get('interpolation_relative_tolerance'),'frequency interpolation tolerance'),
                'actual frequency interpolation/mass failed')
        for kind in ('source_weight','effective_weight','acquired_mask'):
            info=row[kind];path=directory/'frequency_weights'/f'{scan}_{kind}.npy'
            require(info.get('path')==str(path) and info.get('file_sha256')==base.sha256(path)
                    and info.get('shape')==data[scan]['shape'] and info.get('dtype')=='torch.float32'
                    and info.get('device')==device,'actual saved frequency array differs')
            _sha(info.get('byte_sha256'))


def _validate_physics(value,steps,events,coils):
    work=value.get('physics_work',{})
    require(set(work)=={'initialization','training','event','export'},'complete FFT role counters required')
    fields=('fft_batches','ifft_batches','scalar_fft_transforms','scalar_ifft_transforms',
            'coil_fft_transforms','coil_ifft_transforms')
    for role,ffts,iffts in (('initialization',0,0),('training',2*steps,0),('event',2*events,2*events),('export',2,4)):
        expected=dict(zip(fields,(ffts,iffts,0,0,coils*ffts,coils*iffts)))
        require(work[role]==expected and all(type(v) is int for v in work[role].values()),'actual FFT role work differs')
    expected=dict(gaussian_renders=2,canonical_export_calls=2,canonical_fft_batches=2,
        canonical_ifft_batches=4,canonical_coil_fft_transforms=2*coils,canonical_coil_ifft_transforms=4*coils)
    require(value.get('canonical_work')==expected
            and all(type(v) is int for v in value['canonical_work'].values()),'actual canonical export work differs')

def _npy_record(path):
    """Read NumPy header and finite complex64 bytes using only stdlib."""
    with Path(path).open('rb') as stream:
        require(stream.read(6)==b'\x93NUMPY','invalid NPY artifact')
        version=stream.read(2)
        require(version in (b'\x01\x00',b'\x02\x00'),'unsupported NPY version')
        size=struct.unpack('<H' if version[0]==1 else '<I',stream.read(2 if version[0]==1 else 4))[0]
        require(0<size<65536,'oversize NPY header')
        header=ast.literal_eval(stream.read(size).decode('latin1'))
        shape=header.get('shape')
        require(header.get('descr')=='<c8' and header.get('fortran_order') is False
                and isinstance(shape,tuple) and len(shape)==3 and all(type(v) is int and v>0 for v in shape),
                'finite C-order complex64 volume required')
        remaining=math.prod(shape)*8
        while remaining:
            chunk=stream.read(min(1<<20,remaining))
            require(chunk and len(chunk)%8==0,'truncated complex export')
            require(all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',chunk)), 'nonfinite export')
            remaining-=len(chunk)
        require(stream.read(1)==b'','extra export bytes')
    return dict(shape=list(shape),dtype='complex64')


def _validate_child_artifacts(args,directory,hashes,invariants,*,point_count=3600000,device='cuda:0',requested_k=18000,backend='tile_cuda',coils=42):
    validate_parameters(args)
    directory=Path(directory).resolve(strict=True)
    launch,metrics,provenance=[base.read(directory/name) for name in ('launch.json','metrics.json','provenance.json')]
    for value in (launch,metrics,provenance):
        require(all(value.get(k)==v for k,v in parameters(args).items())
                and value.get('code_sha256')==hashes and value.get('invariants')==invariants,'child route/source/input differs')
    require(launch.get('run_dir')==str(directory),'child directory differs')
    require(metrics == provenance, 'metrics/provenance evidence disagrees')
    require(metrics.get('status')=='complete' and metrics.get('model_schema')==ROUTE
            and metrics.get('policy')==POLICY and metrics.get('training_reference_access') is False,
            'complete fixed G2 route required')
    ready,ack,started=[base.read(directory/name) for name in ('childcontext_ready.json','childcontext_ack.json','child_started.json')]
    pid=_integer(ready.get('pid'),'child pid',1)
    require(started.get('pid')==pid and ready.get('parent_pid')==launch.get('parent_pid')
            and ready.get('gpu_uuid')==started.get('gpu_uuid')==launch.get('gpu_uuid')
            and ack==dict(schema_version=1,child_pid=pid,parent_pid=launch['parent_pid'],
                          gpu_uuid=launch['gpu_uuid'],context_observed=True), 'observed context ACK differs')
    runtime=metrics.get('runtime',{})
    require(runtime.get('device')==device and runtime.get('torch')=='2.7.1+cu118'
            and runtime.get('cuda')=='11.8' and runtime.get('python')==str(base.PYTHON)
            and runtime.get('renderer')==backend and runtime.get('strict_cuda') is (backend=='tile_cuda'),
            'exact child runtime/backend required')
    require(runtime.get('default_dtype')=='torch.float32' and runtime.get('autocast_enabled') is False
            and type(runtime.get('cudnn_allow_tf32')) is bool and type(runtime.get('matmul_allow_tf32')) is bool
            and isinstance(runtime.get('python_version'),list) and len(runtime['python_version'])==3
            and all(type(v) is int for v in runtime['python_version']), 'actual precision/runtime evidence differs')
    for key in ('training_seconds','stage_seconds','peak_memory_gb','peak_reserved_gb'): _number(metrics.get(key),key,0)
    _integer(metrics.get('model_id'),'live model identity',1)
    before,after=metrics.get('before'),metrics.get('after')
    _state(before,4500,point_count,device); _state(after,4500+args.steps,point_count,device); _preserved(before,after)
    require(before['config']['training']['max_iterations']==4500+args.steps, 'absolute training budget differs')
    restore=metrics.get('restore',{})
    require(restore.get('iteration')==4500 and restore.get('after')==before and restore.get('group_names')==list(NAMES)
            and restore.get('source',{}).get('sha256')==RESUME_HASH
            and restore.get('source',{}).get('function')=='restore_checkpoint'
            and restore.get('base_lrs')==list(BASE_LRS) and restore.get('restored_lrs')==
                [g['options']['lr'] for g in before['optimizer']['groups']], 'actual restore/source evidence differs')
    require(metrics.get('source_checkpoint')==dict(path=invariants['g2_checkpoint'],sha256=invariants['g2_checkpoint_sha256']),
            'exact frozen G2 source checkpoint required')
    trace=[json.loads(line) for line in (directory/'step_trace.jsonl').read_text().splitlines()]
    events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    require(trace==metrics.get('trace') and events==metrics.get('events') and len(trace)==args.steps,
            'actual trace/event artifact differs')
    for i,row in enumerate(trace,1):
        require(row.get('step')==i and row.get('iteration')==4500+i and row.get('gradients_finite') is True,
                'actual update order/gradients differ')
        _number(row.get('loss'),'training loss'); _number(row.get('unclipped_grad_norm'),'clipped gradient norm',0)
        require(set(row.get('gradient_norms',{}))==set(NAMES),'seven gradients missing')
        for norm in row['gradient_norms'].values(): _number(norm,'gradient norm',0)
        require(row.get('frequency_use')==dict(frequency_t=1.,loss_call_scans=['pre','post'],
            loss_calls_total=2*i,absolute_iteration=4500+i,source_offset=4500), 'actual paired frequency use differs')
        require(row.get('adam_steps')=={name:4500+i for name in NAMES}
                and row.get('scheduler_epoch')==4500+i and row.get('scheduler_count')==4501+i,
                'observed intermediate clocks differ')
    expected_event=int(args.mode=='relocate' and args.steps>0)
    require(len(events)==expected_event,'exactly one event in relocation and zero in control required')
    for event in events:
        validate_event(event,point_count,requested_k,device)
        state=event['transaction']['before']
        require(state['model_id']==metrics['model_id'] and state['optimizer_id']==before['optimizer']['id']
                and state['scheduler_id']==before['scheduler']['id'], 'event uses foreign live model/state objects')
        for name in NAMES:
            require(state['parameters'][name]['id']==before['parameters'][name]['tensor_id']
                    and state['step_ids'][name]==before['optimizer']['states'][name]['step']['tensor_id'],
                    'event uses foreign Parameter/clock object')
            for key in ('exp_avg','exp_avg_sq'):
                require(state['moments'][name][key]['id']==before['optimizer']['states'][name][key]['tensor_id'],
                        'event uses foreign Adam moment object')

    counters=metrics.get('counters',{})
    expected=dict(optimizer_steps=args.steps,scheduler_steps=args.steps,loss_calls=2*args.steps,
        event_calls=expected_event,gaussian_forwards=2*args.steps+2*expected_event+2,gaussian_vjps=2*args.steps)
    require(set(counters)==set(expected)|{'roles','render_calls'},'complete actual counters required')
    require(all(type(counters.get(k)) is int and counters[k]==v for k,v in expected.items()),'actual work budget differs')
    roles={'initialization':dict(forwards=0,vjps=0),'training':dict(forwards=2*args.steps,vjps=2*args.steps),
           'event':dict(forwards=2*expected_event,vjps=0),'export':dict(forwards=2,vjps=0)}
    require(counters.get('roles')==roles, 'render/VJP roles differ')
    calls=counters.get('render_calls',[])
    require(len(calls)==expected['gaussian_forwards'],'actual render calls missing')
    observed=[]
    for i in range(1,args.steps+1):
        observed.extend((('training',4500+i,'plain'),('training',4500+i,'contrast')))
        if expected_event and i==1: observed.extend((('event',4501,'plain'),('event',4501,'contrast')))
    observed.extend((('export',4500+args.steps,'plain'),('export',4500+args.steps,'contrast')))
    require([(r.get('role'),r.get('iteration'),r.get('channel')) for r in calls]==observed
            and all(type(r.get('vjps')) is int and r['vjps']==int(r['role']=='training') for r in calls),
            'render/VJP call roles/order differ')
    require(metrics.get('final_checkpoint_evidence',{}).get('iteration')==4500+args.steps
            and metrics['final_checkpoint_evidence'].get('live_model_and_adam_equal') is True,
            'actual final checkpoint validation missing')
    checkpoint=metrics['final_checkpoint_evidence']
    require(checkpoint.get('model_sha256')=={n:after['parameters'][n]['sha256'] for n in NAMES}
            and checkpoint.get('adam_sha256')=={n:{k:r['sha256'] for k,r in after['optimizer']['states'][n].items()} for n in NAMES}
            and checkpoint.get('groups')==after['optimizer']['groups'], 'saved checkpoint/live bytes disagree')
    progress=base.read(directory/'progress.json')
    require(progress.get('status')=='complete' and progress.get('counters')==counters
            and progress.get('final_iteration')==4500+args.steps,'actual progress completion differs')
    require(base.read(directory/'scheduler.json')==after['scheduler']['state']
            and base.read(directory/'config.json')==before['config'],'saved scheduler/config differs')
    data=metrics.get('data_evidence',{})
    require(set(data)=={'pre','post'} and all(row.get('ground_truth_constructed') is False
        and row.get('mask_inferred') is False and row.get('outside_mask_nonzero')==0
        and row.get('coils')==coils for row in data.values()),'acquired-only evidence differs')
    _validate_frequency(metrics.get('frequency'),directory,args.steps,trace,data,device)
    _validate_physics(metrics,args.steps,expected_event,coils)
    if args.stage=='zero':
        require(before==after,'zero changed restored model/moments/clock/config')
        parity=metrics.get('zero_parity',{})
        require(set(parity)=={'pre','post'},'both G2 canonical parity records required')
        for row in parity.values():
            require(row.get('passed') is True and row.get('tolerance')==1e-5
                    and type(row.get('array_equal')) is bool,'zero parity policy missing')
            require(0<=_number(row.get('complex_relative_l2'),'zero parity')<=1e-5,'zero reproduction failed')
    else: require(metrics.get('zero_parity')=={},'nonzero stage reports undeclared zero parity')
    # Every child-created file except live logs must be in the artifact manifest.
    manifest=metrics.get('artifact_manifest',{})
    mandatory={'config.json','config.yaml','scheduler.json','step_trace.jsonl','events.jsonl','progress.json'}
    mandatory.add('frequency_weights/manifest.json')
    mandatory.update(f'frequency_weights/{scan}_{kind}.npy' for scan in ('pre','post')
                     for kind in ('source_weight','effective_weight','acquired_mask'))
    mandatory.update(f'{kind}_{scan}_complex.npy' for kind in ('raw','canonical','candidate') for scan in ('pre','post'))
    final=Path(metrics.get('final_checkpoint','')).resolve(strict=True)
    require(final.is_relative_to(directory) and final.name=='latest.pth','final checkpoint escapes owned output')
    require(checkpoint.get('file')==base.descriptor(final),'checkpoint evidence file bytes differ')
    mandatory.add(str(final.relative_to(directory)))
    require(mandatory<=set(manifest),'complete final artifacts missing')
    for relative,descriptor in manifest.items():
        p=directory/relative
        require(p.resolve(strict=True).is_relative_to(directory) and p.is_file()
                and descriptor==base.descriptor(p), 'bound artifact bytes/path changed')
    for scan in ('pre','post'):
        for kind in ('raw','canonical','candidate'):
            record=_npy_record(directory/f'{kind}_{scan}_complex.npy')
            require(record['shape']==data[scan]['shape'],'export spatial shape differs')
        require(base.sha256(directory/f'canonical_{scan}_complex.npy')==base.sha256(directory/f'candidate_{scan}_complex.npy'),
                'candidate differs from original canonical hard DC')
    paths={directory/name for name in manifest}
    paths.update(directory/name for name in ('launch.json','metrics.json','provenance.json',
            'childcontext_ready.json','childcontext_ack.json','child_started.json'))
    return {str(p):base.sha256(p) for p in sorted(paths)}


def validate_child_artifacts(args,directory,hashes,invariants):
    require(invariants.get('route')==ROUTE and invariants.get('policy')==POLICY
            and invariants.get('g2_checkpoint_sha256')==base.G2_SHA256,'real frozen relocation inputs required')
    return _validate_child_artifacts(args,directory,hashes,invariants)


def validate_observed_gate(path,stage,hashes,invariants,matched=None):
    gate=base.read(path); params=gate.get('parameters',{})
    require(gate.get('route')==ROUTE and gate.get('schema_version')==1 and gate.get('passed') is True
            and gate.get('status')=='complete' and gate.get('stage')==stage
            and isinstance(params,dict) and set(params)==set(PARAMETERS)
            and all(gate.get(k)==v for k,v in params.items()), 'real completed relocation gate required')
    args=argparse.Namespace(**params); validate_parameters(args)
    require(gate.get('code_sha256')==hashes and gate.get('invariants')==invariants,'foreign/stale relocation gate')
    if matched: require(params['mode']==matched['mode'],'matched-mode relocation smoke required')
    require(type(gate.get('worker_pid')) is int and gate['worker_pid']>0
            and type(gate.get('returncode')) is int and gate['returncode']==0
            and gate.get('gpu_claim_policy')==base.GPU_CLAIM_POLICY, 'actual supervised child completion required')
    require(base.read(Path(gate['run_dir'])/'childcontext_ready.json').get('pid')==gate['worker_pid'],
            'completed worker differs from observed child')
    artifacts=gate.get('artifacts',{})
    require(artifacts and all(base.sha256(p)==h for p,h in artifacts.items()), 'control artifact bytes changed')
    require(validate_child_artifacts(args,gate['run_dir'],hashes,invariants)==artifacts,'actual control artifact set differs')
    proof=base.read(base.verify_descriptor(gate.get('post_run_invariants'),'relocation before/after proof'))
    expected=dict(code_sha256=hashes,invariants=invariants)
    require(proof.get('status')=='verified' and proof.get('before')==proof.get('after')==expected,
            'unchanged reviewed source/input proof required')
    controls=gate.get('control_bindings')
    require(isinstance(controls,dict) and controls and proof.get('control_bindings_before')==
            proof.get('control_bindings_after')==controls and all(base.sha256(p)==h for p,h in controls.items()),
            'control bytes changed')
    return gate


def validate_stage_gates(args,hashes,invariants):
    validate_parameters(args)
    require(isinstance(args.exp,str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*',args.exp),'safe unique exp required')
    _integer(args.preferred_gpu,'preferred GPU')
    require(0<_number(args.timeout,'timeout')<=3600,'owned-child timeout at most3600 required')
    review=base.validate_review(args.review_gate,hashes,invariants)
    require(review.get('route')==ROUTE,'review must cover relocation route')
    recipe=base.read(args.recipe)
    require(recipe.get('route')==ROUTE and recipe.get('schema_version')==1 and recipe.get('status')=='selected'
            and recipe.get('parameters')==parameters(args) and recipe.get('policy')==POLICY
            and recipe.get('code_sha256')==hashes and recipe.get('invariants')==invariants,
            'exact selected relocation CLI/policy/source/input recipe required')
    paths=[args.review_gate,args.recipe,Path(review['review_report']['path'])]
    result=dict(review=review,recipe=recipe)
    for stage,path in ([('zero',args.zero_gate)] if args.stage=='smoke' else
            [('zero',args.zero_gate),('smoke',args.smoke_gate)] if args.stage=='trial' else []):
        require(path is not None,f'{args.stage} requires actual {stage} gate')
        result[stage]=validate_observed_gate(path,stage,hashes,invariants,parameters(args) if stage=='smoke' else None)
        paths.append(path)
    result['control_bindings']={str(Path(p).resolve(strict=True)):base.sha256(p) for p in paths}
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.set_defaults(route=ROUTE,requested_k=18000,event_step=1)
    p.add_argument('--stage',choices=('zero','smoke','trial'),required=True)
    p.add_argument('--mode',choices=('control','relocate'),required=True)
    p.add_argument('--steps',type=int,required=True); p.add_argument('--exp',required=True)
    p.add_argument('--review-gate',type=Path,required=True); p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--zero-gate',type=Path); p.add_argument('--smoke-gate',type=Path)
    p.add_argument('--preferred-gpu',type=int,default=0); p.add_argument('--timeout',type=float,default=3600)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    with base.cancellation_signals():
        try: print(base.execute(args,protocol=sys.modules[__name__]),flush=True)
        except BaseException as error:
            print(base.record_failure(args,error),file=sys.stderr,flush=True)
            return 128+error.signum if isinstance(error,base.SupervisorCancelled) else 1
    return 0


if __name__=='__main__': raise SystemExit(main())
