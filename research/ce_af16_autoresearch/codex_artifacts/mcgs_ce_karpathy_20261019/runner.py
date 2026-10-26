#!/usr/bin/env python3
"""G2 numerical child: acquired data, original physics, new labeled parameters."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'mcgs_ce_autoresearch_20261017'
G2 = OLD / 'frequency_adam/outputs/adamflat_g2_v1'
TRAINER = OLD / 'nonlocal_autoresearch/trainer'
CHECKPOINT = G2 / 'training/ar_i1_k30p90/checkpoints/latest.pth'
GSTMI = Path('/home/congcongliu/anaconda3/envs/GSTMI/bin/python')
NAMES = ('positions','scales','rotations','density_real','density_imag',
         'density_real_contrast','density_imag_contrast')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
    os.replace(temp,path)


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def build_config(source, run_dir, steps):
    config = copy.deepcopy(source)
    if any(config['data'][scan]['acceleration_factor'] != 16 or config['data'][scan]['presampled']
           for scan in ('plain','contrast')) or config.get('adaptive_control',{}).get('enable',False):
        raise ValueError('only the unchanged fixed G2 AF16 acquisition and point cloud are supported')
    if isinstance(steps,bool) or not isinstance(steps,int) or not 0 <= steps <= 500:
        raise ValueError('new candidate update budget must be an integer from0 through500')
    config['training'].update(resume_offset=4500,max_iterations=4500+steps,log_every=20)
    config['output'].update(output_dir=str(Path(run_dir)/'training'),exp_name='residual_amplitude')
    config.setdefault('mode',{})['strict_cuda'] = True
    return config


def validate_g2_recipe(checkpoint):
    recipe=checkpoint.get('frequency_adam_recipe',{})
    if checkpoint.get('iter') != 4500 or recipe.get('frequency_t') != 1. or recipe.get('geometry_factor') != 2.:
        raise ValueError('expected the actual G2 iteration4500 flat-frequency checkpoint')


def setup_trainer(config, state, scales, run_dir, launch):
    import torch
    sys.path[:0] = [str(TRAINER/'paired_gs'),str(TRAINER)]
    import gaussian.tile_voxelizer as tile
    binary = TRAINER/'paired_gs/gaussian/cuda_ext/build/tile_voxelize_cuda_0141c525_7aedd3b0.so'
    if hashlib.sha256(binary.read_bytes()).hexdigest() != '3973daeef5f55ebfed931c38538a37d83921a9287b96b69a431da238057194b5':
        raise ValueError('frozen prebuilt CUDA binary changed; refusing compilation or fallback')
    tile._cuda_ext = load_module(binary,binary.stem)
    tile._cuda_load_error = None
    import trainers.joint_trainer as jt
    from runtime import acquired_dataset_class
    acquired = acquired_dataset_class(jt.MRIDataset,scales)
    original_loader = jt._load_dataset

    def frozen_loader(data_cfg):
        # Keep the source argument mapping and replace only its GT-building class.
        previous = jt.MRIDataset
        jt.MRIDataset = acquired
        try: return original_loader(data_cfg)
        finally: jt.MRIDataset = previous

    jt._load_dataset = frozen_loader

    class FrozenG2Trainer(jt.JointTrainer):
        def _setup_model(self):
            from gaussian.joint_gaussian_model import JointGaussianModel3D
            from gaussian.tile_voxelizer import TileVoxelizer
            self.model = JointGaussianModel3D(num_points=state['positions'].shape[0],
                volume_shape=self.volume_shape,device=self.device,
                initial_positions=state['positions'].to(self.device),
                initial_scales=state['scales'].exp().to(self.device),
                initial_densities_plain=torch.complex(state['density_real'],state['density_imag']).to(self.device),
                initial_densities_contrast=torch.complex(state['density_real_contrast'],state['density_imag_contrast']).to(self.device))
            self.model.load_state_dict(state,strict=True)
            for value in self.model.parameters(): value.requires_grad_(False)
            vcfg=self.config['voxelizer']
            self.voxelizer=TileVoxelizer(volume_shape=self.volume_shape,tile_size=vcfg['tile_size'],
                max_radius=vcfg['max_radius'],use_cuda=True,strict_cuda=True,device=str(self.device)).to(self.device)
            if self.voxelizer.backend != 'tile_cuda' or not self.voxelizer.strict_cuda:
                raise ValueError('original strict CUDA renderer was not selected')

    trainer=FrozenG2Trainer(config,device=torch.device('cuda:0'))
    jt._load_dataset=original_loader
    for name in NAMES:
        if not torch.equal(getattr(trainer.model,name).detach().cpu(),state[name]):
            raise ValueError(f'G2 restore drift in {name}')
    frequency=load_module(OLD/'frequency_adam/runner.py','_new_campaign_frozen_frequency')
    frequency.install_frequency_weights(trainer,1.,run_dir)
    before={name:tensor_hash(getattr(trainer.model,name)) for name in NAMES}
    field=None
    if launch['mode'] != 'baseline':
        from amplitude_model import ResidualAmplitudeField
        field=ResidualAmplitudeField(trainer.model.positions,
            trainer.model.get_density('plain'),trainer.model.get_density('contrast'),
            mode=launch['mode'],bands=launch['bands'],width=launch['width'],gain_bound=launch['gain_bound'])
        for channel in ('plain','contrast'):
            if not torch.equal(field.get_density(channel),trainer.model.get_density(channel)):
                raise ValueError('zero residual coefficients do not exactly equal G2')
        trainer.model.add_module('residual_field',field)
        trainer.model.get_density=field.get_density
        trainer.optimizer=torch.optim.Adam(field.parameter_groups(lr=launch['lr']),betas=(.9,.999))
        trainer.scheduler=torch.optim.lr_scheduler.ExponentialLR(trainer.optimizer,gamma=.9995)
    return trainer,field,frequency,before


def execute(launch_path):
    if Path(sys.executable).resolve() != GSTMI.resolve():
        raise ValueError('numerical work requires the fixed GSTMI interpreter')
    launch=json.loads(Path(launch_path).read_text())
    run_dir=Path(launch['run_dir']).resolve()
    if run_dir.parent != ROOT/'outputs' or Path(launch_path).resolve() != run_dir/'launch.json':
        raise ValueError('numeric launch must refer to its supervisor-owned output directory')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != launch['gpu_uuid']:
        raise ValueError('initial GPU visibility differs from supervisor selection')
    for path,digest in launch['code_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('reviewed code changed before numeric initialization')
    import torch
    import numpy as np
    import yaml
    if torch.__version__ != '2.7.1+cu118' or torch.version.cuda != '11.8':
        raise ValueError('unexpected Torch/CUDA runtime')
    hold=torch.zeros(1,device='cuda:0')
    write_json(run_dir/'childcontext_ready.json',{'pid':os.getpid(),'gpu_uuid':launch['gpu_uuid'],
        'logical_device':'cuda:0','torch':torch.__version__,'cuda':torch.version.cuda})
    provenance={'status':'initializing','launch':launch,'training_reference_access':False,
        'model_schema':'g2_residual_amplitude_v1','optimizer_policy':'new moments/clock; not Cartesian Adam continuation'}
    write_json(run_dir/'provenance.json',provenance)
    torch.manual_seed(42); np.random.seed(42)
    checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    validate_g2_recipe(checkpoint)
    state=checkpoint['model_state']
    if set(state) != set(NAMES) or state['positions'].shape != (3600000,3):
        raise ValueError('G2 model point count/schema changed')
    source=json.loads((G2/'metrics.json').read_text())
    source_config=yaml.safe_load((G2/'actual_config.yaml').read_text())
    config=build_config(source_config,run_dir,launch['steps'])
    scales={source_config['data'][channel]['data_path']:source['audit_checks'][scan]['scale_trainer']
            for channel,scan in [('plain','pre'),('contrast','post')]}
    (run_dir/'config.yaml').write_text(yaml.safe_dump(config),encoding='utf8')
    trainer,field,frequency,before=setup_trainer(config,state,scales,run_dir,launch)
    del state,checkpoint
    data_evidence={scan:getattr(trainer,'ds_'+tag).acquired_only_evidence
                   for scan,tag in [('pre','p'),('post','c')]}
    if any(v['outside_mask_nonzero'] or v['coils'] != 42 for v in data_evidence.values()):
        raise ValueError('acquired-only data/coil validation failed')
    traces=[]; step_started=None; step_record=None
    def callback(event,iteration,payload):
        nonlocal step_started,step_record
        from runtime import begin_frequency_step,finish_frequency_step
        if event=='start':
            torch.cuda.synchronize(); step_started=time.perf_counter()
            begin_frequency_step(frequency._frequency_guard(trainer),4500,iteration)
        elif event=='pre_step':
            step_record={'step':iteration-4500,'iteration':iteration,**payload}
            for group in trainer.optimizer.param_groups:
                grads=[p.grad for p in group['params'] if p.grad is not None]
                if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                    raise ValueError('missing/nonfinite gradients in new model parameter group')
        elif event=='end':
            torch.cuda.synchronize()
            step_record.update(seconds=time.perf_counter()-step_started,
                frequency_use=finish_frequency_step(frequency._frequency_guard(trainer),4500,iteration))
            traces.append(step_record)
            with (run_dir/'step_trace.jsonl').open('a') as f:
                f.write(json.dumps(step_record,allow_nan=False)+'\n')
    (run_dir/'step_trace.jsonl').touch(exist_ok=False)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
    if launch['steps']:
        if field is None: raise ValueError('baseline is zero-step only')
        trainer.train(step_callback=callback)
    elif trainer.writer is not None:
        trainer.writer.close()
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    if len(traces) != launch['steps'] or any(not r['gradients_finite'] for r in traces):
        raise ValueError('actual steps/gradient evidence differs from the selected protocol')
    after={name:tensor_hash(getattr(trainer.model,name)) for name in NAMES}
    if before != after:
        raise ValueError('fixed G2 geometry/base coefficients changed')
    checkpoint_out=run_dir/'checkpoint.pth'
    torch.save({'schema':'g2_residual_amplitude_v1','source_iteration':4500,
        'updates':len(traces),'source_checkpoint_sha256':launch['invariants']['g2_checkpoint_sha256'],
        'field_metadata':None if field is None else field.metadata(),
        'field_state':None if field is None else field.state_dict(),
        'optimizer':None if field is None else trainer.optimizer.state_dict(),
        'scheduler':None if field is None else trainer.scheduler.state_dict(),
        'source_geometry_and_coefficients_unchanged':True,'recipe':launch['recipe']},checkpoint_out)
    base=load_module(OLD/'nonlocal_autoresearch/runner.py','_new_campaign_original_export')
    from runtime import zero_parity
    parity={}
    for channel,scan in [('plain','pre'),('contrast','post')]:
        dc,zf=base.export_trainer_volume(trainer,channel)
        candidate=dc.detach().cpu().numpy().astype(np.complex64)
        candidate=(candidate*np.float32(source['audit_checks'][scan]['factor_scale'])).astype(np.complex64)
        np.save(run_dir/f'candidate_{scan}_complex.npy',candidate)
        if launch['stage']=='zero':
            parity[scan]=zero_parity(candidate,np.load(G2/f'candidate_{scan}_complex.npy',allow_pickle=False))
            if not parity[scan]['passed']: raise ValueError(f'{scan} zero-step G2 reproduction failed: {parity[scan]}')
        del dc,zf,candidate
    result={'status':'complete','stage':launch['stage'],'mode':launch['mode'],'steps':len(traces),
        **{key:launch[key] for key in ('bands','width','gain_bound','lr')},
        'steps_observed':len(traces),'code_sha256':launch['code_sha256'],'invariants':launch['invariants'],
        'source_iteration':4500,'training_seconds':elapsed,'gradients_finite':True,
        'field_metadata':None if field is None else field.metadata(), 'data_evidence':data_evidence,
        'fixed_model_unchanged':before==after,'zero_parity':parity,
        'peak_memory_gb':torch.cuda.max_memory_allocated()/1024**3,
        'peak_reserved_gb':torch.cuda.max_memory_reserved()/1024**3,
        'checkpoint':str(checkpoint_out),'final_checkpoint':str(checkpoint_out),'first_loss':traces[0]['loss'] if traces else None,
        'last_loss':traces[-1]['loss'] if traces else None}
    result['frequency']=frequency.frequency_record(trainer,2.)
    provenance.update(result,original_tensor_hashes=before,source_config_sha256=hashlib.sha256((G2/'actual_config.yaml').read_bytes()).hexdigest())
    write_json(run_dir/'provenance.json',provenance)
    write_json(run_dir/'metrics.json',result)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--launch',type=Path,required=True)
    args=parser.parse_args()
    try: execute(args.launch)
    except BaseException as exc:
        try:
            run=Path(json.loads(args.launch.read_text())['run_dir'])
            write_json(run/'numeric_failure.json',{'status':'failed','error':repr(exc),'pid':os.getpid()})
        except Exception: pass
        raise
