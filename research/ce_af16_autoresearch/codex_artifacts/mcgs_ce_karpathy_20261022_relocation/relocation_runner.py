#!/usr/bin/env python3
"""Original acquired-only MRI child for one fixed-count G2 relocation event."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

import runner as g2
import run_relocation as route
from proxdc_runner import helpers, topic_array, wait_context_ack
from relocation_runtime import PARAMETER_NAMES, restore_training_state, snapshot_training_state


def _finite_parameters(trainer):
    import torch
    if tuple(dict(trainer.model.named_parameters())) != PARAMETER_NAMES:
        raise ValueError('exactly seven original parameters required; networks forbidden')
    n=trainer.model.positions.shape[0]
    if trainer.model.num_points!=n or not all(p.requires_grad and bool(torch.isfinite(p).all())
                                            for p in trainer.model.parameters()):
        raise ValueError('point count, finite parameter or training policy failed')


class WorkObserver:
    """Count successful original module forwards and density-input VJPs.

    The density input hook runs after the CUDA backward produces its Gaussian
    coefficient derivative. An output-image hook alone would not prove that.
    """
    def __init__(self, trainer):
        self.trainer=trainer; self.role='initialization'; self.iteration=4500; self.channel=None
        self.counters=dict(optimizer_steps=0,scheduler_steps=0,loss_calls=0,event_calls=0,
            gaussian_forwards=0,gaussian_vjps=0,roles={name:dict(forwards=0,vjps=0)
                for name in ('initialization','training','event','export')},render_calls=[])
        self.physics={name:dict(fft_batches=0,ifft_batches=0,scalar_fft_transforms=0,
            scalar_ifft_transforms=0,coil_fft_transforms=0,coil_ifft_transforms=0)
            for name in self.counters['roles']}

    @contextmanager
    def observe(self):
        import torch
        trainer=self.trainer
        render=trainer._render; render_owned='_render' in vars(trainer)
        saved_render=vars(trainer).get('_render')
        scheduler_class=type(trainer.scheduler)
        saved_step=scheduler_class.step
        def measured_render(channel):
            previous=self.channel; self.channel=channel
            try: return render(channel)
            finally: self.channel=previous
        def forwarded(module,inputs,kwargs,output):
            if self.channel not in ('plain','contrast') or self.role not in self.counters['roles']:
                raise ValueError('unassigned Gaussian forward')
            row=dict(role=self.role,iteration=self.iteration,channel=self.channel,vjps=0)
            self.counters['render_calls'].append(row)
            self.counters['gaussian_forwards']+=1; self.counters['roles'][self.role]['forwards']+=1
            density=kwargs.get('density',inputs[3] if len(inputs)>3 else None)
            if density is None: raise ValueError('Gaussian density input missing')
            if torch.is_grad_enabled() and density.requires_grad:
                role=self.role
                def density_vjp(gradient):
                    if not bool(torch.isfinite(gradient).all()): raise ValueError('nonfinite Gaussian density VJP')
                    row['vjps']+=1
                    self.counters['gaussian_vjps']+=1; self.counters['roles'][role]['vjps']+=1
                    return gradient
                density.register_hook(density_vjp)
        def optimized(opt,args,kwargs): self.counters['optimizer_steps']+=1
        def scheduled(scheduler,*args,**kwargs):
            result=saved_step(scheduler,*args,**kwargs)
            if scheduler is trainer.scheduler: self.counters['scheduler_steps']+=1
            return result
        trainer._render=measured_render
        scheduler_class.step=scheduled
        forward_hook=trainer.voxelizer.register_forward_hook(forwarded,with_kwargs=True)
        optimizer_hook=trainer.optimizer.register_step_post_hook(optimized)
        try: yield self
        finally:
            forward_hook.remove(); optimizer_hook.remove()
            if render_owned: trainer._render=saved_render
            else: del trainer._render
            scheduler_class.step=saved_step

    @contextmanager
    def observe_physics(self):
        # Preserve all source aliases and operands, partitioning actual FFT work
        # by the same role used for Gaussian forwards. No new FFT is performed.
        modules=[helpers().source_module(name,relative) for name,relative in (
            ('data.transforms','data/transforms.py'),('data.dataset','data/dataset.py'),
            ('trainers.joint_trainer','trainers/joint_trainer.py'))]
        saved=[]
        def measured(fn,kind):
            def call(x,*args,**kwargs):
                result=fn(x,*args,**kwargs)
                work=self.physics[self.role]
                if x.ndim not in (3,4): raise ValueError('unexpected FFT operand rank')
                work[kind+'_batches']+=1
                work[('scalar_' if x.ndim==3 else 'coil_')+kind+'_transforms']+=1 if x.ndim==3 else x.shape[0]
                return result
            return call
        try:
            for module in modules:
                for kind in ('fft','ifft'):
                    key=kind+'3c'
                    if hasattr(module,key):
                        old=getattr(module,key); saved.append((module,key,old))
                        setattr(module,key,measured(old,kind))
            yield
        finally:
            for module,key,old in reversed(saved): setattr(module,key,old)


def _checkpoint_evidence(trainer,path,iteration):
    import torch
    from relocation_runtime import _tensor_record
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if set(saved)!= {'iter','model_state','optim_state'} or type(saved['iter']) is not int or saved['iter']!=iteration:
        raise ValueError('original final checkpoint schema/absolute iteration differs')
    live=trainer.optimizer.state_dict()
    if set(saved['model_state'])!=set(PARAMETER_NAMES) or saved['optim_state']['param_groups']!=live['param_groups']:
        raise ValueError('saved model/groups differ from live original state')
    if set(saved['optim_state']['state'])!=set(live['state']): raise ValueError('saved Adam state membership differs')
    for name,param in trainer.model.named_parameters():
        if not torch.equal(saved['model_state'][name],param.detach().cpu()):
            raise ValueError('saved final parameter differs from live')
    for key,row in live['state'].items():
        if set(saved['optim_state']['state'][key])!=set(row): raise ValueError('saved Adam state fields differ')
        for field,value in row.items():
            if not torch.equal(saved['optim_state']['state'][key][field],value.detach().cpu()):
                raise ValueError('saved final Adam moment/step differs from live')
    snapshot=snapshot_training_state(trainer)
    return dict(iteration=iteration,live_model_and_adam_equal=True,
        model_sha256={name:_tensor_record(saved['model_state'][name])['sha256'] for name in PARAMETER_NAMES},
        adam_sha256={name:{key:_tensor_record(saved['optim_state']['state'][group['params'][0]][key])['sha256']
                     for key in ('step','exp_avg','exp_avg_sq')}
                     for name,group in zip(PARAMETER_NAMES,saved['optim_state']['param_groups'])},
        groups=snapshot['optimizer']['groups'],file=route.base.descriptor(path))


def run_continuation(trainer,frequency,directory,*,mode,steps,rotation_matrix_fn,requested_k=18000):
    """Run the original trainer; injection is internal and absent from public CLI.

    CPU tests supply tiny original trainer instances and the original rotation
    function. Public execution constructs only the reviewed strict MRI trainer.
    """
    import torch
    from relocation import select_relocations,relocate_rows
    from runtime import begin_frequency_step,finish_frequency_step
    if mode not in ('control','relocate') or type(steps) is not int or steps not in (0,2,500) or (steps==0 and mode!='control'):
        raise ValueError('invalid fixed continuation budget/mode')
    if trainer.config.get('adaptive_control',{}).get('enable',False):
        raise ValueError('inherited adaptive control must remain disabled')
    if trainer.config['loss'].get('edge_tv_kappa_source')!='zf':
        raise ValueError('fixed G2 requires zf kappas and zero initialization renders')
    _finite_parameters(trainer)
    directory=Path(directory)
    before=snapshot_training_state(trainer); trace=[]; events=[]; pending=None; start=None
    work=WorkObserver(trainer)
    for name in ('step_trace.jsonl','events.jsonl'): (directory/name).touch(exist_ok=False)
    def sync():
        if trainer.model.positions.device.type=='cuda': torch.cuda.synchronize()
    def append(name,row):
        with (directory/name).open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush()
    def clocks(iteration):
        values={name:int(trainer.optimizer.state[p]['step'].item()) for name,p in trainer.model.named_parameters()}
        if set(values.values())!={iteration} or trainer.scheduler.last_epoch!=iteration or trainer.scheduler._step_count!=iteration+1:
            raise ValueError('actual Adam/scheduler clocks differ from ordinary update')
        return dict(adam_steps=values,scheduler_epoch=trainer.scheduler.last_epoch,scheduler_count=trainer.scheduler._step_count)
    def callback(event,iteration,payload):
        nonlocal pending,start
        if event=='start':
            work.role='training'; work.iteration=iteration
            sync();start=time.perf_counter()
            begin_frequency_step(frequency._frequency_guard(trainer),4500,iteration)
            if work.counters['optimizer_steps']!=iteration-4501: raise ValueError('unexpected ordinary optimizer step count')
        elif event=='pre_step':
            norms={}
            for name,param in trainer.model.named_parameters():
                if param.grad is None or not bool(torch.isfinite(param.grad).all()):
                    raise ValueError('missing/nonfinite original seven gradient')
                norms[name]=float(torch.linalg.vector_norm(param.grad.double()).item())
            pending=dict(step=iteration-4500,iteration=iteration,**payload,gradient_norms=norms)
        elif event=='end':
            if pending is None or pending['iteration']!=iteration: raise ValueError('missing actual pre-step evidence')
            pending.update(frequency_use=finish_frequency_step(frequency._frequency_guard(trainer),4500,iteration),**clocks(iteration))
            _finite_parameters(trainer)
            if mode=='relocate' and iteration==4501:
                if events: raise ValueError('second relocation event forbidden')
                work.role='event';work.counters['event_calls']+=1
                with torch.no_grad():
                    scores=trainer._evidence_energy_score()
                    donors,victims,selection=select_relocations(trainer.model,scores,requested_k)
                    transaction=relocate_rows(trainer.model,trainer.optimizer,trainer.scheduler,donors,victims,rotation_matrix_fn)
                row=dict(iteration=iteration,step=1,selection=selection,transaction=transaction)
                route.validate_event(row,trainer.model.num_points,requested_k,str(trainer.model.positions.device))
                events.append(row);append('events.jsonl',row)
                work.role='training'
            sync();pending['seconds']=time.perf_counter()-start
            trace.append(pending);append('step_trace.jsonl',pending);pending=None
        else: raise ValueError('unknown original step callback')
    with work.observe(),work.observe_physics():
        sync();training_started=time.perf_counter()
        if steps: trainer.train(step_callback=callback)
        else:
            ckpt=trainer.out_dir/'checkpoints/latest.pth';ckpt.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(iter=4500,model_state=trainer.model.state_dict(),optim_state=trainer.optimizer.state_dict()),ckpt)
        sync();training_seconds=time.perf_counter()-training_started if steps else 0.
        if len(trace)!=steps: raise ValueError('actual training update count differs')
        state=frequency._frequency_guard(trainer)
        work.counters['loss_calls']=state['loss_calls_total']
        final=trainer.out_dir/'checkpoints/latest.pth'
        checkpoint=_checkpoint_evidence(trainer,final,4500+steps)
        after=snapshot_training_state(trainer)
        if steps==0 and before!=after: raise ValueError('zero mutated full restored G2 state')
        work.role='export';work.iteration=4500+steps
        with torch.no_grad():
            exports,canonical_work=helpers().canonical_control(trainer,backend_kind='cuda')
        if snapshot_training_state(trainer)!=after: raise ValueError('export mutated model/Adam/scheduler/config')
    expected_event=int(mode=='relocate' and steps>0)
    expected_roles=dict(initialization=dict(forwards=0,vjps=0),training=dict(forwards=2*steps,vjps=2*steps),
                        event=dict(forwards=2*expected_event,vjps=0),export=dict(forwards=2,vjps=0))
    if work.counters['roles']!=expected_roles or work.counters['optimizer_steps']!=steps or work.counters['scheduler_steps']!=steps:
        raise ValueError('actual Gaussian forward/VJP/optimizer work differs')
    g2.write_json(directory/'scheduler.json',after['scheduler']['state'])
    return dict(training_seconds=training_seconds,model_id=id(trainer.model),before=before,after=after,trace=trace,events=events,counters=work.counters,
        physics_work=work.physics,canonical_work=canonical_work,exports=exports,
        frequency=frequency.frequency_record(trainer,2.),
        final_checkpoint=str(final.resolve()),final_checkpoint_evidence=checkpoint)


def save_exports(exports,gauges,directory):
    import numpy as np
    directory=Path(directory);saved={}
    for scan in ('pre','post'):
        saved[scan]={}
        for kind in ('raw','canonical','candidate'):
            image=exports[scan]['canonical' if kind=='candidate' else kind]
            value=topic_array(image,gauges[scan])
            path=directory/f'{kind}_{scan}_complex.npy'
            with path.open('xb') as stream: np.save(stream,value,allow_pickle=False)
            saved[scan][kind]={**route.base.descriptor(path), 'shape':list(value.shape),'dtype':'complex64'}
    return saved


def validate_launch(path):
    launch=route.base.read(path)
    args=argparse.Namespace(**{key:launch.get(key) for key in route.PARAMETERS});route.validate_parameters(args)
    directory=Path(launch['run_dir']).resolve()
    if directory.parent!=g2.ROOT/'outputs' or Path(path).resolve()!=directory/'launch.json':
        raise ValueError('child launch/output ownership differs')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=launch.get('gpu_uuid') or os.getppid()!=launch.get('parent_pid'):
        raise ValueError('child reserved visibility/parent differs')
    if route.code_hashes()!=launch.get('code_sha256') or route.snapshot_invariants()!=launch.get('invariants'):
        raise ValueError('reviewed source/input bytes changed before initialization')
    recipe=route.base.read(route.base.verify_descriptor(launch.get('recipe'),'selected relocation recipe'))
    if recipe.get('schema_version')!=1 or recipe.get('status')!='selected' or recipe.get('route')!=route.ROUTE or recipe.get('policy')!=route.POLICY or recipe.get('parameters')!=route.parameters(args) \
            or recipe.get('code_sha256')!=launch['code_sha256'] or recipe.get('invariants')!=launch['invariants']:
        raise ValueError('exact selected relocation recipe differs')
    controls=launch.get('control_bindings',{})
    if not controls or any(route.base.sha256(p)!=h for p,h in controls.items()):
        raise ValueError('reviewed control bytes changed before initialization')
    return launch,directory


def clinical_stage(launch,directory):
    import torch
    import yaml
    from runtime import zero_parity
    if route.base.sha256(g2.CHECKPOINT)!=route.base.G2_SHA256: raise ValueError('G2 source checkpoint bytes changed')
    checkpoint=torch.load(g2.CHECKPOINT,map_location='cpu',weights_only=True)
    state=checkpoint['model_state']
    shapes={name:(3600000,3) if i<2 else (3600000,4) if i==2 else (3600000,) for i,name in enumerate(PARAMETER_NAMES)}
    if set(state)!=set(shapes) or any(tuple(state[n].shape)!=shape for n,shape in shapes.items()):
        raise ValueError('original3.6M seven tensor checkpoint required')
    source=route.base.read(g2.G2/'metrics.json'); source_config=yaml.safe_load((g2.G2/'actual_config.yaml').read_text())
    config=g2.build_config(source_config,directory,launch['steps'])
    config['output']['use_tensorboard']=False
    config['output']['exp_name']='fixed_count_relocation'
    config['training']['seed']=42
    if config['loss'].get('edge_tv_kappa_source')!='zf' or config.get('adaptive_control',{}).get('enable',False):
        raise ValueError('frozen G2 zf/disabled adaptive policy differs')
    (directory/'config.yaml').write_text(yaml.safe_dump(config))
    g2.write_json(directory/'config.json',config)
    gauges={scan:{key:source['audit_checks'][scan][key] for key in ('scale_trainer','scale_gauge','factor_scale')}
            for scan in ('pre','post')}
    scales={source_config['data'][channel]['data_path']:gauges[scan]['scale_trainer']
            for channel,scan in (('plain','pre'),('contrast','post'))}
    construction=dict.fromkeys(helpers().CONSTRUCTION_COUNTERS,0)
    with helpers().observe_source_transforms(construction):
        trainer,field,frequency,_=g2.setup_trainer(config,state,scales,directory,dict(mode='baseline'))
    try:
        if field is not None or trainer.voxelizer.backend!='tile_cuda' or not trainer.voxelizer.strict_cuda or not trainer.voxelizer.use_cuda:
            raise ValueError('strict original CUDA renderer and no networks required')
        helpers().validate_acquisition(trainer)
        restore=restore_training_state(trainer,checkpoint)
        del checkpoint,state
        from gaussian.tile_voxelizer import _quaternion_to_rotation_matrix
        data={scan:getattr(trainer,'ds_'+tag).acquired_only_evidence for scan,tag in (('pre','p'),('post','c'))}
        if any(row['coils']!=42 or row['outside_mask_nonzero'] or row['ground_truth_constructed'] is not False for row in data.values()):
            raise ValueError('acquired-only source data differs')
        result=run_continuation(trainer,frequency,directory,mode=launch['mode'],steps=launch['steps'],
                                rotation_matrix_fn=_quaternion_to_rotation_matrix)
        exports=result.pop('exports');parity={}
        if launch['stage']=='zero':
            import numpy as np
            for scan in ('pre','post'):
                # Completed G2 reconstructions are zero-control parity anchors,
                # never passed into training or the acquired-evidence function.
                parity[scan]=zero_parity(topic_array(exports[scan]['canonical'],gauges[scan]),
                    np.load(g2.G2/f'candidate_{scan}_complex.npy',allow_pickle=False))
                if not parity[scan]['passed']: raise ValueError('fresh G2 canonical parity failed')
        arrays=save_exports(exports,gauges,directory)
        return {**result,'restore':restore,'data_evidence':data,'zero_parity':parity,
                'arrays':arrays,'construction_work':construction,'frequency':frequency.frequency_record(trainer,2.)}
    finally:
        if getattr(trainer,'writer',None) is not None: trainer.writer.close()



def publish_result(launch,directory,result,runtime,*,source_checkpoint,elapsed,peak_allocated,peak_reserved):
    """Bind actual files and publish immutable matching completion evidence."""
    directory=Path(directory)
    excluded={'launch.json','metrics.json','provenance.json','launch.log','child_started.json',
              'childcontext_ready.json','childcontext_ack.json'}
    manifest={str(p.relative_to(directory)):route.base.descriptor(p) for p in sorted(directory.rglob('*'))
              if p.is_file() and p.name not in excluded}
    metrics={**route.parameters(argparse.Namespace(**launch)),**result,'status':'complete',
        'model_schema':route.ROUTE,'policy':route.POLICY,'training_reference_access':False,
        'source_checkpoint':route.base.descriptor(source_checkpoint),'code_sha256':launch['code_sha256'],
        'invariants':launch['invariants'],'artifact_manifest':manifest,'runtime':runtime,
        'stage_seconds':elapsed,'peak_memory_gb':peak_allocated,'peak_reserved_gb':peak_reserved}
    for name in ('metrics.json','provenance.json'): route.base.write_new(directory/name,metrics)
    return metrics

def execute(path):
    if Path(sys.executable).resolve()!=g2.GSTMI.resolve(): raise ValueError('actual GSTMI child interpreter required')
    launch,directory=validate_launch(path)
    import numpy as np
    import torch
    if torch.__version__!='2.7.1+cu118' or torch.version.cuda!='11.8' or torch.cuda.device_count()!=1:
        raise ValueError('exact GSTMI Torch/CUDA and one reserved visible GPU required')
    hold=torch.zeros(1,device='cuda:0')
    g2.write_json(directory/'childcontext_ready.json',dict(pid=os.getpid(),parent_pid=os.getppid(),gpu_uuid=launch['gpu_uuid']))
    wait_context_ack(directory,launch)
    torch.manual_seed(42);np.random.seed(42)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
    g2.write_json(directory/'progress.json',dict(status='started',source_iteration=4500))
    result=clinical_stage(launch,directory)
    torch.cuda.synchronize()
    g2.write_json(directory/'progress.json',dict(status='complete',counters=result['counters'],
                    final_iteration=4500+launch['steps']))
    runtime=dict(python=str(g2.GSTMI),torch=str(torch.__version__),cuda=torch.version.cuda,
        device='cuda:0',renderer='tile_cuda',strict_cuda=True,python_version=list(sys.version_info[:3]),
        default_dtype=str(torch.get_default_dtype()),autocast_enabled=torch.is_autocast_enabled(),
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    # Final child snapshot complements the unchanged supervisor's before/after
    # proof and catches input mutations even before it publishes completed.json.
    if route.code_hashes()!=launch['code_sha256'] or route.snapshot_invariants()!=launch['invariants']:
        raise ValueError('source/input bytes changed during numerical work')
    publish_result(launch,directory,result,runtime,source_checkpoint=g2.CHECKPOINT,
        elapsed=time.perf_counter()-started,peak_allocated=torch.cuda.max_memory_allocated()/1024**3,
        peak_reserved=torch.cuda.max_memory_reserved()/1024**3)


if __name__=='__main__':
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--launch',type=Path,required=True)
    args=parser.parse_args()
    try: execute(args.launch)
    except BaseException as error:
        try:
            directory=Path(route.base.read(args.launch)['run_dir'])
            g2.write_json(directory/'numeric_failure.json',dict(status='failed',error=repr(error),pid=os.getpid()))
        except Exception: pass
        raise
