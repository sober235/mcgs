#!/usr/bin/env python3
"""GSTMI-only geometry restart child around the unchanged C trainer and export."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parent
BASE_ROOT = ROOT.parent / 'nonlocal_autoresearch'
BASE_RUNNER = BASE_ROOT / 'runner.py'
GSTMI = Path('/home/congcongliu/anaconda3/envs/GSTMI/bin/python')
NAMES = ('positions', 'scales', 'rotations', 'density_real', 'density_imag',
         'density_real_contrast', 'density_imag_contrast')
SOURCE_SHAPES = {'positions': [3600000, 3], 'scales': [3600000, 3], 'rotations': [3600000, 4],
                 **{name: [3600000] for name in NAMES[3:]}}


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def supervisor():
    return load_file(ROOT / 'run_gpu.py', '_geometry_restart_supervisor')


def code_hashes():
    return supervisor().code_hashes()


def runtime():
    # Refuse the orchestration environment before importing its older Torch.
    if Path(sys.executable).resolve() != GSTMI.resolve():
        raise ValueError('model/check numerical child must use the explicit GSTMI Python interpreter')
    import torch
    if torch.__version__ != '2.7.1+cu118' or torch.version.cuda != '11.8':
        raise ValueError('require GSTMI Torch 2.7.1+cu118 with CUDA 11.8')
    return {'python': sys.executable, 'python_version': list(sys.version_info[:3]),
            'torch_version': torch.__version__, 'cuda_version': torch.version.cuda}


def load_base_runner():
    return load_file(BASE_RUNNER, '_geometry_restart_original_runner')


def load_resume_protocol():
    sys.path[:0] = [str(BASE_ROOT / 'trainer/paired_gs'), str(BASE_ROOT / 'trainer')]
    module = importlib.import_module('trainers.resume_protocol')
    expected = BASE_ROOT / 'trainer/paired_gs/trainers/resume_protocol.py'
    if Path(module.__file__).resolve() != expected:
        raise ValueError('unexpected trainer resume protocol source')
    return module


def write_record(path, value):
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    os.replace(temporary, path)


def _array(tensor):
    return tensor.detach().cpu().contiguous().numpy()


def _bytes(tensor):
    return hashlib.sha256(_array(tensor).tobytes(order='C')).hexdigest()


def validate_saved_frequency_evidence(run_dir, value, frequency_t, trainer):
    """Independently read all six saved arrays; never infer RMS from metadata."""
    observed_runtime=runtime()
    import numpy as np
    s=supervisor();folder=Path(run_dir).resolve()/'frequency_weights'
    s.require(frequency_t in (0.,1.) and s.read(folder/'manifest.json') == value
              and value.get('frequency_t') == frequency_t and value.get('mass_relative_tolerance') == 2e-6
              and set(value.get('scans',{})) == {'pre','post'},'saved frequency manifest/profile differs')
    state=trainer._frequency_adam_state;scans={}
    for scan,tag in (('pre','p'),('post','c')):
        row=value['scans'][scan];arrays={}
        for name in ('source_weight','effective_weight','acquired_mask'):
            info=row[name];path=folder/f'{scan}_{name}.npy'
            s.require(info.get('path') == str(path) and path.resolve().parent == folder
                      and s.sha256(path) == info.get('file_sha256'),'saved frequency file path/hash differs')
            array=np.load(path,allow_pickle=False)
            s.require(list(array.shape) == info.get('shape') and 'torch.'+str(array.dtype) == info.get('dtype')
                      and hashlib.sha256(array.tobytes(order='C')).hexdigest() == info.get('byte_sha256'),
                      'saved frequency raw dtype/shape/bytes differ')
            arrays[name]=array
        source,effective,mask=(arrays[k] for k in ('source_weight','effective_weight','acquired_mask'))
        original=_array(getattr(trainer,'mask_'+tag))
        original_source=_array(state['original_builder'](high_override=state['plain_override'])
                               if scan == 'pre' else state['original_builder']())
        s.require(source.shape == effective.shape == mask.shape and source.dtype == effective.dtype
                  and source.dtype.kind == 'f' and np.isfinite(source).all() and np.isfinite(effective).all()
                  and (source >= 0).all() and np.isfinite(mask).all() and ((mask == 0)|(mask == 1)).all()
                  and mask.dtype == original.dtype and np.array_equal(mask,original)
                  and source.dtype == original_source.dtype and np.array_equal(source,original_source),
                  'saved source weights/original acquired masks differ')
        acquired=mask.astype(bool)
        s.require(acquired.any() and (source[acquired] > 0).any(),'empty/zero source acquired mass')
        source64=source.astype(np.float64);samples=source64[acquired]
        scale=float(samples.max());c=scale*math.sqrt(float(np.mean((samples/scale)**2,dtype=np.float64)))
        expected=source if frequency_t == 0. else np.full_like(source,c)
        exact=np.array_equal(source,effective);tolerance=8.*float(np.finfo(source.dtype).eps)
        error=0. if np.array_equal(expected,effective) else float(np.max(
            np.abs(effective.astype(np.float64)-expected)/np.maximum(np.abs(expected),np.finfo(source.dtype).tiny)))
        source_mass=float(np.mean(source64[acquired]**2,dtype=np.float64))
        effective_mass=float(np.mean(effective[acquired].astype(np.float64)**2,dtype=np.float64))
        relative=abs(effective_mass/source_mass-1.)
        s.require(math.isfinite(error) and error <= tolerance and (frequency_t != 0. or exact)
                  and relative <= 2e-6 and int(acquired.sum()) == row['acquired_count']
                  and math.isclose(source_mass,row['source_mean_squared_weight'],rel_tol=1e-12)
                  and math.isclose(effective_mass,row['effective_mean_squared_weight'],rel_tol=1e-12)
                  and math.isclose(c,row['c'],rel_tol=1e-12),'independent saved-array interpolation/RMS/count differs')
        scans[scan]={'acquired_count':int(acquired.sum()),'c':c,'source_mean_squared_weight':source_mass,
            'effective_mean_squared_weight':effective_mass,'relative_mass_error':relative,
            'original_acquired_mask_exact':True,'original_source_weight_exact':True,
            'interpolation_max_relative_error':error,'interpolation_relative_tolerance':tolerance,
            'effective_identity':exact,'source_levels':[float(v) for v in np.unique(source)],
            'acquired_source_levels':[float(v) for v in np.unique(source[acquired])]}
    return {'passed':True,'frequency_t':float(frequency_t),'numerical_runtime':observed_runtime,
            'independent_backend':'NumPy float64 full interpolation and saved-array RMS; original acquired masks',
            'mass_relative_tolerance':2e-6,'scans':scans}


def _frequency_guard(trainer):
    state=trainer._frequency_adam_state;s=state['supervisor']
    s.require(trainer.config['loss'] == state['loss_config'],'source loss config/curriculum changed')
    for scan,tag in (('pre','p'),('post','c')):
        weight=state['effective_weights'][scan]
        s.require(weight._version == state['weight_versions'][scan], 'installed frequency weight mutated')
        static=getattr(trainer,'_static_freq_w_plain' if scan == 'pre' else '_static_freq_w',None)
        s.require(static is weight,'actual trainer frequency weight missing/reverted')
        for name in ('csm_fit','kspace_fit','mask_fit','mask'):
            value=getattr(trainer,name+'_'+tag);original,version=state['source_arguments'][scan][name]
            s.require(value is original and value._version == version,'source data/mask tensor changed')
        s.require(getattr(trainer,'_dc_norm_'+tag) == state['dc_norm'][scan],'source DC norm changed')
    return state


def install_frequency_weights(trainer, frequency_t, run_dir):
    """Install once before render and intercept source train's static rebuild.

    Loss calls validate actual tensor identity/version and unchanged source
    arguments. Full source/effective/mask bytes are checked at setup and finish;
    no per-step CPU hashing or additional model forward is introduced.
    """
    runtime()
    import copy
    import numpy as np
    import torch
    flatten_acquired_rms=load_file(ROOT/'frequency_weights.py','_frequency_adam_weight_utility').flatten_acquired_rms
    s=supervisor()
    s.require(type(frequency_t) in (int,float) and frequency_t in (0.,1.)
              and not hasattr(trainer,'_frequency_adam_state'),'frequency install must be fresh t0/t1')
    loss=trainer.config['loss']
    s.require(loss.get('kspace_freq_adaptive') is True and not loss.get('kspace_freq_curriculum',False)
              and loss.get('kspace_high_weight_plain') is not None,'source static per-scan weights/no curriculum required')
    builder=trainer._build_freq_weight;plain_override=loss['kspace_high_weight_plain']
    sources={'pre':builder(high_override=plain_override),'post':builder()}
    effective={scan:flatten_acquired_rms(sources[scan],getattr(trainer,'mask_'+tag),frequency_t)
               for scan,tag in (('pre','p'),('post','c'))}
    folder=Path(run_dir).resolve()/'frequency_weights';folder.mkdir(parents=True,exist_ok=False)
    scans={}
    for scan,tag in (('pre','p'),('post','c')):
        original,effect,mask=sources[scan],effective[scan],getattr(trainer,'mask_'+tag)
        source64=_array(original).astype(np.float64);acquired=_array(mask).astype(bool)
        samples=source64[acquired];scale=float(samples.max())
        c=scale*math.sqrt(float(np.mean((samples/scale)**2,dtype=np.float64)))
        source_mass=float(np.mean(samples**2,dtype=np.float64))
        effective_mass=float(np.mean(_array(effect)[acquired].astype(np.float64)**2,dtype=np.float64))
        row={'acquired_count':int(acquired.sum()),'c':c,'source_mean_squared_weight':source_mass,
             'effective_mean_squared_weight':effective_mass,'relative_mass_error':abs(effective_mass/source_mass-1.),
             't0_exact_equal':True if frequency_t == 0. else None}
        for name,tensor in (('source_weight',original),('effective_weight',effect),('acquired_mask',mask)):
            path=folder/f'{scan}_{name}.npy';np.save(path,_array(tensor),allow_pickle=False)
            row[name]={'path':str(path),'file_sha256':s.sha256(path),'byte_sha256':_bytes(tensor),
                       'shape':list(tensor.shape),'dtype':str(tensor.dtype),'device':str(tensor.device)}
        scans[scan]=row
    evidence={'version':1,'frequency_t':float(frequency_t),'mass_relative_tolerance':2e-6,'scans':scans}
    write_record(folder/'manifest.json',evidence)
    state={'frequency_t':float(frequency_t),'run_dir':str(Path(run_dir).resolve()),'frequency_evidence':evidence,'supervisor':s,
        'source_weights':sources,'effective_weights':effective,'original_builder':builder,'plain_override':plain_override,
        'weight_versions':{scan:value._version for scan,value in effective.items()},'loss_config':copy.deepcopy(loss),
        'dc_norm':{scan:getattr(trainer,'_dc_norm_'+tag) for scan,tag in (('pre','p'),('post','c'))},
        'source_arguments':{scan:{name:(getattr(trainer,name+'_'+tag),getattr(trainer,name+'_'+tag)._version)
                                 for name in ('csm_fit','kspace_fit','mask_fit','mask')}
                            for scan,tag in (('pre','p'),('post','c'))},
        'loss_calls_total':0,'completed_steps':0,'step_scans':[],'current_iteration':None,
        'curriculum_iteration':None,'final_frequency_use':None,'forward_calls':0}
    trainer._frequency_adam_state=state
    trainer._static_freq_w,trainer._static_freq_w_plain=effective['post'],effective['pre']
    state['frequency_array_check']=validate_saved_frequency_evidence(run_dir,evidence,frequency_t,trainer)
    original_loss,original_render,original_curriculum=trainer._kspace_loss,trainer._render,trainer._freq_weight_curriculum
    def build(high_override=None):
        _frequency_guard(trainer)
        s.require(high_override is None or high_override == plain_override,'undeclared source high-band override')
        return effective['post' if high_override is None else 'pre']
    def curriculum(it,max_iter):
        _frequency_guard(trainer)
        s.require(not trainer.config['loss'].get('kspace_freq_curriculum',False),'frequency curriculum forbidden')
        result=original_curriculum(it,max_iter)
        s.require(it == state['current_iteration'] and result is effective['post'],'actual source curriculum weight differs')
        state['curriculum_iteration']=it
        return result
    def render(channel):
        _frequency_guard(trainer);state['forward_calls'] += 1
        return original_render(channel)
    def kspace_loss(vol,csm,ksp_meas,mask,freq_weight,norm=None):
        _frequency_guard(trainer)
        candidates=[(scan,tag) for scan,tag in (('pre','p'),('post','c')) if csm is getattr(trainer,'csm_fit_'+tag)]
        s.require(len(candidates) == 1,'source loss scan cannot be identified')
        scan,tag=candidates[0]
        s.require(state['current_iteration'] == state['curriculum_iteration'] is not None
                  and state['step_scans'] == ([] if scan == 'pre' else ['pre']), 'actual source loss call sequence differs')
        s.require(freq_weight is effective[scan] and ksp_meas is getattr(trainer,'kspace_fit_'+tag)
                  and mask is getattr(trainer,'mask_fit_'+tag) and norm == state['dc_norm'][scan],
                  'actual loss frequency weight/source data/mask/norm differs')
        result=original_loss(vol,csm,ksp_meas,mask,freq_weight,norm=norm)
        state['step_scans'].append(scan);state['loss_calls_total'] += 1
        return result
    trainer._build_freq_weight,trainer._freq_weight_curriculum=build,curriculum
    trainer._render,trainer._kspace_loss=render,kspace_loss
    return {'frequency_evidence':evidence,'frequency_array_check':state['frequency_array_check']}


def begin_frequency_step(trainer, iteration):
    state=_frequency_guard(trainer)
    state['supervisor'].require(iteration == 4001+state['completed_steps'] and
        (state['current_iteration'] is None or state['step_scans'] == ['pre','post']), 'frequency step clock differs')
    state['current_iteration']=iteration;state['step_scans']=[];state['curriculum_iteration']=None


def frequency_step_evidence(trainer, iteration):
    state=_frequency_guard(trainer);s=state['supervisor']
    s.require(iteration == state['current_iteration'] == 4001+state['completed_steps']
              and state['step_scans'] == ['pre','post'] and state['loss_calls_total'] == 2*(state['completed_steps']+1),
              'actual two-scan frequency loss calls are incomplete')
    evidence=state['frequency_evidence']
    result={'frequency_t':state['frequency_t'],'installed_before_first_forward':True,'actual_trainer_weights_match':True,
        'loss_call_scans':list(state['step_scans']),'loss_calls_total':state['loss_calls_total'],'dc_norm':dict(state['dc_norm'])}
    for name,key in (('effective_weight','effective_weight_byte_sha256'),('source_weight','source_weight_byte_sha256'),
                     ('acquired_mask','acquired_mask_byte_sha256')):
        result[key]={scan:evidence['scans'][scan][name]['byte_sha256'] for scan in ('pre','post')}
    state['completed_steps'] += 1;state['final_frequency_use']=result
    return result


def frequency_record(trainer, geometry_factor, status='complete'):
    state=_frequency_guard(trainer);s=supervisor();evidence=state['frequency_evidence']
    checked=validate_saved_frequency_evidence(state['run_dir'],evidence,state['frequency_t'],trainer)
    s.require(checked == state['frequency_array_check'],'frequency independent array check changed')
    for scan in ('pre','post'):
        s.require(_bytes(state['effective_weights'][scan]) == evidence['scans'][scan]['effective_weight']['byte_sha256']
                  and _bytes(state['source_weights'][scan]) == evidence['scans'][scan]['source_weight']['byte_sha256'],
                  'final actual frequency weight bytes changed')
    return {'status':status,'method_variant':'frequency_adam','frequency_t':state['frequency_t'],
        'geometry_factor':float(geometry_factor),'frequency_evidence':evidence,'frequency_array_check':checked,
        'installed_before_first_forward':True,'dc_norm':dict(state['dc_norm']),
        'loss_calls_total':state['loss_calls_total'],'completed_steps':state['completed_steps'],
        'final_frequency_use':state['final_frequency_use'],
        'final_weights_check':{'passed':True,'actual_trainer_weights_match':True,
                              'source_weights_unchanged':True,'original_masks_unchanged':True}}


def checkpoint_recipe(record):
    return {'version':1,'frequency_t':record['frequency_t'],'geometry_factor':record['geometry_factor'],
        'local_factor':1.,'nlambda':0.,'wavelet_protocol':'original_single_phase',
        'frequency_manifest_sha256':supervisor().sha256(Path(record['frequency_evidence']['scans']['pre']['source_weight']['path']).parent/'manifest.json'),
        'requires_same_recipe_on_resume':True}


def attach_checkpoint_recipe(path, record):
    runtime()
    import torch
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    supervisor().require('frequency_adam_recipe' not in checkpoint,'frequency recipe cannot overwrite an existing one')
    checkpoint['frequency_adam_recipe']=checkpoint_recipe(record)
    torch.save(checkpoint,path)


def install_strict_backend(base, invariants, *, frequency_t=None, run_dir=None, frequency_callback=None):
    """Bind the verified .so in the normal Gaussian module before construction."""
    s = supervisor(); binary = Path(invariants['prebuilt_extension']['path']).resolve()
    source = (base.TRAINER / 'paired_gs/gaussian/tile_voxelizer.py').resolve()
    folder = source.parent / 'cuda_ext'
    pieces = [hashlib.md5((folder / name).read_bytes()).hexdigest()[:8]
              for name in ('tile_voxelize_cuda.cpp', 'tile_voxelize_cuda_kernel.cu')]
    expected = folder / 'build' / ('tile_voxelize_cuda_' + '_'.join(pieces) + '.so')
    s.require(binary == expected and s.sha256(binary) == invariants['prebuilt_extension']['sha256'],
              'verified prebuilt binary path/source hash changed; compilation is forbidden')
    tile = importlib.import_module('gaussian.tile_voxelizer')
    s.require(Path(tile.__file__).resolve() == source, 'normal tile voxelizer is not the frozen C source')
    old_ext, old_error = tile._cuda_ext, tile._cuda_load_error
    old_module = sys.modules.get(binary.stem)
    extension = load_file(binary, binary.stem)
    s.require(Path(extension.__file__).resolve() == binary
              and callable(getattr(extension, 'forward', None)) and callable(getattr(extension, 'backward', None)),
              'prebuilt extension does not expose the original source kernels')
    tile._cuda_ext, tile._cuda_load_error = extension, None
    trainer_module = None
    try:
        trainer_module = importlib.import_module('trainers.joint_trainer')
        s.require(Path(trainer_module.__file__).resolve() == base.TRAINER / 'paired_gs/trainers/joint_trainer.py',
                  'unexpected original trainer module')
        original_factory = trainer_module.JointTrainer
    except BaseException:
        tile._cuda_ext, tile._cuda_load_error = old_ext, old_error
        if old_module is None: sys.modules.pop(binary.stem, None)
        else: sys.modules[binary.stem] = old_module
        raise
    backend_record = {'prebuilt_extension': dict(invariants['prebuilt_extension']), 'normal_tile_source': str(source),
                      'compiler_branch_bypassed': True, 'status': 'installed'}

    def factory(cfg, device=None):
        s.require(cfg.get('voxelizer', {}).get('type', 'tile_cuda') == 'tile_cuda'
                  and cfg.get('mode', {}).get('strict_cuda') is True, 'strict source tile_cuda config is required')
        original_render=None
        if frequency_t is not None:
            s.require(isinstance(original_factory,type) and callable(getattr(original_factory,'_render',None)),
                      'source trainer class/render guard required')
            original_render=original_factory._render
            def construction_render_guard(self,*args,**kwargs):
                raise ValueError('source attempted a forward before frequency weights were installed')
            original_factory._render=construction_render_guard
        try:
            trainer = original_factory(cfg, device=device)
        finally:
            if original_render is not None: original_factory._render=original_render
        voxelizer = trainer.voxelizer
        if not (voxelizer.backend == 'tile_cuda' and voxelizer.strict_cuda is True and voxelizer.use_cuda is True
                and not bool(getattr(voxelizer, 'use_cupy', False))):
            if getattr(trainer, 'writer', None) is not None: trainer.writer.close()
            raise RuntimeError('clinical trainer selected a fallback/non-strict CUDA backend')
        backend_record.update(status='constructed', backend=voxelizer.backend, strict_cuda=True, use_cuda=True)
        if frequency_t is not None:
            try:
                install_frequency_weights(trainer,frequency_t,run_dir)
                if frequency_callback is not None: frequency_callback(trainer)
            except BaseException:
                if getattr(trainer,'writer',None) is not None: trainer.writer.close()
                raise
        return trainer

    trainer_module.JointTrainer = factory

    def restore():
        trainer_module.JointTrainer = original_factory
        tile._cuda_ext, tile._cuda_load_error = old_ext, old_error
        if old_module is None: sys.modules.pop(binary.stem, None)
        else: sys.modules[binary.stem] = old_module
    return backend_record, restore


@contextlib.contextmanager
def configure_runner(base, protocol, factor, launch, frequency_t=None):
    """Patch only private module objects; source files stay unchanged."""
    geometry = load_file(ROOT / 'overrides.py', '_geometry_restart_overrides')
    previous = {name: getattr(base, name) for name in
                ('ROOT', 'TRAINER', 'build_config', 'reserve_run', 'state_snapshot', 'source_copy_record','StepTrace')}
    original_restore = protocol.restore_checkpoint
    base.ROOT, base.TRAINER = ROOT, BASE_ROOT / 'trainer'
    planned = {'geometry_factor': float(factor), 'geometry_groups': list(NAMES[:3]), 'density_lr_factor': 1.,
               'betas': [.9, .999], 'apply_after_strict_restore_iteration': 4000,
               'moment_policy': 'retain source Adam moments and step4000; no reset',
               'scheduler_policy': 'retain clock; scale three current/initial/base/last LRs coherently'}
    frequency_t=launch.get('request',{}).get('frequency_t',1.) if frequency_t is None else frequency_t
    if type(frequency_t) not in (int,float) or frequency_t not in (0.,1.):raise ValueError('require declared frequency t0/t1')
    record = {'status': 'planned', 'planned': planned, 'code_sha256': launch['code_sha256'],
              'method_variant':'frequency_adam','frequency_t':float(frequency_t),'geometry_factor':float(factor),
              'invariants': launch['invariants'], 'source_runner': str(BASE_RUNNER),
              'source_runner_sha256': base.sha256_file(BASE_RUNNER), 'entrypoint': str(Path(__file__).resolve()),
              'runtime': runtime(), 'source_checkpoint': str(base.CHECKPOINT), 'source_config': str(base.BASE_CONFIG)}
    run_dir, backend_restore, snapshot_count, restored_once, frequency_trainer = None, None, 0, False, None
    escaped = None

    def write():
        if run_dir is not None: write_record(run_dir / 'geometry_provenance.json', record)

    def build_config(*args, **kwargs):
        cfg = previous['build_config'](*args, **kwargs)
        if (cfg['campaign_overrides']['local_factor'], cfg['campaign_overrides']['nlambda'],
                cfg['campaign_overrides']['sigma_rel']) != (1., 0., .015):
            raise ValueError('geometry route cannot change original regularization or nonlocal settings')
        if cfg.get('voxelizer', {}).get('type', 'tile_cuda') != 'tile_cuda':
            raise ValueError('geometry route requires the source tile_cuda renderer')
        mode = cfg.setdefault('mode', {})
        cfg['campaign_overrides']['runtime_backend'] = {'mode.strict_cuda': {
            'source_present': 'strict_cuda' in mode, 'source': mode.get('strict_cuda'), 'effective': True}}
        mode['strict_cuda'] = True
        cfg['optimizer_overrides'] = dict(planned)
        cfg['campaign_overrides']['geometry_restart'] = dict(planned)
        cfg['frequency_overrides']={'frequency_t':float(frequency_t),
            'mask_source':'original acquired mask_p/mask_c; never fit masks',
            'utility_sha256':supervisor().sha256(ROOT/'frequency_weights.py')}
        return cfg

    def reserve_run(*args, **kwargs):
        nonlocal run_dir
        run_dir = previous['reserve_run'](*args, **kwargs)
        record.update(run_dir=str(run_dir), child_pid=os.getpid())
        write(); return run_dir

    def source_copy_record():
        nonlocal backend_restore
        manifest, delta = previous['source_copy_record']()
        def installed(trainer):
            nonlocal frequency_trainer
            frequency_trainer=trainer
            record['frequency_adam']=frequency_record(trainer,factor,status='installed')
            write_record(run_dir/'frequency_provenance.json',record['frequency_adam'])
            write()
        record['backend'], backend_restore = install_strict_backend(base, launch['invariants'],
            frequency_t=frequency_t,run_dir=run_dir,frequency_callback=installed)
        write(); return manifest, delta

    def restore_checkpoint(model, optimizer, scheduler, checkpoint, *args, **kwargs):
        nonlocal restored_once
        if restored_once: raise ValueError('geometry restart refuses a second restore of the source checkpoint')
        # The authoritative strict source restoration always precedes our scalar change.
        resume = original_restore(model, optimizer, scheduler, checkpoint, *args, **kwargs)
        restored_once = True
        actual = geometry.apply_geometry_lr_factor(model, optimizer, scheduler, factor)
        record.update(status='restored', restore=actual)
        write(); return {**resume, 'geometry_restart': actual}

    def state_snapshot(trainer):
        nonlocal snapshot_count
        state = previous['state_snapshot'](trainer)
        actual = geometry._snapshot(trainer.optimizer, trainer.scheduler)
        record['after_restore' if snapshot_count == 0 else 'final'] = actual
        snapshot_count += 1; state['geometry_restart'] = actual
        write(); return state

    class FrequencyStepTrace(previous['StepTrace']):
        def __call__(self,phase,iteration,data):
            if phase == 'start':begin_frequency_step(self.trainer,iteration)
            elif phase == 'pre_step':data={**data,'frequency_use':frequency_step_evidence(self.trainer,iteration)}
            super().__call__(phase,iteration,data)

    base.build_config, base.reserve_run = build_config, reserve_run
    base.state_snapshot, base.source_copy_record = state_snapshot, source_copy_record
    base.StepTrace=FrequencyStepTrace
    protocol.restore_checkpoint = restore_checkpoint
    try:
        yield record
    except BaseException as exc:
        escaped = exc
        record.update(status='failed', error=repr(exc)); write(); raise
    finally:
        try:
            if run_dir is not None:
                path = run_dir / 'provenance.json'
                if path.is_file():
                    try:
                        value = supervisor().read(path)
                        if not isinstance(value, dict): raise ValueError('source provenance is not an object')
                    except (OSError, ValueError) as exc:
                        record['provenance_read_error'] = repr(exc)
                        if escaped is None:
                            record.update(status='failed', error=repr(exc))
                            write(); raise
                    else:
                        if escaped is None: record['status'] = value.get('status', record['status'])
                        if frequency_trainer is not None:
                            try:
                                record['frequency_adam']=frequency_record(frequency_trainer,factor,status=record['status'])
                                write_record(run_dir/'frequency_provenance.json',record['frequency_adam'])
                                value.update(method_variant='frequency_adam',frequency_t=float(frequency_t),
                                             geometry_factor=float(factor),frequency_adam=record['frequency_adam'])
                                metrics_path=run_dir/'metrics.json'
                                if metrics_path.is_file():
                                    metrics=supervisor().read(metrics_path)
                                    metrics.update(method_variant='frequency_adam',frequency_t=float(frequency_t),
                                                   geometry_factor=float(factor),frequency_adam=record['frequency_adam'])
                                    write_record(metrics_path,metrics)
                            except BaseException as exc:
                                record['frequency_validation_error']=repr(exc)
                                if escaped is None:
                                    record.update(status='failed',error=repr(exc));write();raise
                        value['geometry_restart'] = record
                        if record['status'] == 'failed': value.update(status='failed', error=record['error'])
                        write_record(path, value)
                write()
        finally:
            if backend_restore is not None: backend_restore()
            protocol.restore_checkpoint = original_restore
            for name, value in previous.items(): setattr(base, name, value)


def fail_owned_run(record, error):
    """Never rewrite a stale or unreserved run while reporting a failed child."""
    if not record or record.get('child_pid') != os.getpid() or not record.get('run_dir'): return
    run = Path(record['run_dir']).resolve()
    if not run.is_relative_to(ROOT.resolve()): return
    s = supervisor()
    for name in ('geometry_provenance.json', 'provenance.json','frequency_provenance.json'):
        path = run / name
        if path.is_file():
            try:
                value = s.read(path)
                if not isinstance(value, dict): continue
                value.update(status='failed', error=repr(error)); write_record(path, value)
            except (OSError, ValueError): pass  # Preserve partial JSON for parent failure evidence.


def expected_wavelet_shifts(count, protocol):
    runtime()
    import torch
    generator = protocol.resumed_wavelet_generator(42, 1, 4000)
    return [torch.randint(0, 2, (3,), generator=generator).tolist() for _ in range(count)]


def base_snapshot(value):
    return {key: item for key, item in value.items() if key not in ('geometry_restart', 'optimizer_probe')}


def validate_checkpoint(checkpoint, expected_shapes, end_iteration, snapshot, optimizer_actual):
    """Check an ordinary Adam checkpoint against actual hashes and LR metadata."""
    runtime()
    import torch
    s = supervisor()
    s.require(checkpoint.get('iter') == end_iteration and 'optim_state' in checkpoint,
              'ordinary resumable Adam checkpoint with final absolute iteration is required')
    model = checkpoint['model_state']; saved = checkpoint['optim_state']
    s.require(set(model) == set(NAMES) and len(saved['param_groups']) == 7 and len(saved['state']) == 7,
              'checkpoint must contain exactly seven original model/Adam groups')
    ids = []
    def digest(tensor):
        return hashlib.sha256(memoryview(tensor.detach().cpu().contiguous().numpy()).cast('B')).hexdigest()
    for name, group, observed in zip(NAMES, saved['param_groups'], optimizer_actual['groups']):
        tensor = model[name]
        s.require(list(tensor.shape) == expected_shapes[name] and tensor.dtype == torch.float32
                  and bool(torch.isfinite(tensor).all()), f'checkpoint shape/dtype/nonfinite model at {name}')
        s.require(digest(tensor) == snapshot['parameter_sha256'][name], f'saved model does not match final state at {name}')
        s.require(len(group['params']) == 1, f'checkpoint group mapping differs at {name}')
        ident = group['params'][0]; ids.append(ident); state = saved['state'][ident]
        s.require(float(state['step']) == end_iteration and set(state) == {'step','exp_avg','exp_avg_sq'},
                  f'checkpoint Adam step/fields differ at {name}')
        for key in ('exp_avg', 'exp_avg_sq'):
            value = state[key]
            s.require(value.shape == tensor.shape and value.dtype == tensor.dtype
                      and bool(torch.isfinite(value).all()), f'invalid checkpoint Adam {key} at {name}')
            s.require(digest(value) == snapshot['adam'][name][key + '_sha256'], f'saved Adam does not match final {name}/{key}')
        s.require(bool((state['exp_avg_sq'] >= 0).all()), f'negative Adam variance at {name}')
        for key in ('lr','initial_lr','eps','weight_decay'):
            s.require(group[key] == observed[key], f'saved Adam {key} differs from actual {name}')
        s.require(list(group['betas']) == observed['betas'] == [.9,.999], f'saved Adam betas changed at {name}')
    s.require(len(set(ids)) == 7 and set(saved['state']) == set(ids), 'checkpoint Adam state mapping is not one-to-one')
    return {'iter': end_iteration, 'schema': ['iter','model_state','optim_state'],
            'model_shapes': expected_shapes, 'all_seven_model_and_moment_hashes_match': True}


def validate_trace(traces, steps, factor, source, protocol):
    s = supervisor(); s.require(len(traces) == steps, 'observed trace count differs from stage budget')
    shifts = expected_wavelet_shifts(steps, protocol)
    for index, row in enumerate(traces):
        iteration = 4001 + index
        s.require(row['mode'] == 'all' and row['absolute_iteration'] == iteration and row['wavelet_shift'] == shifts[index],
                  'absolute iteration or original replayed wavelet sequence differs')
        s.require(row['gradients_finite'] is True and row['full_gradient_group_count'] == 7
                  and row['updated_parameter_groups'] == list(NAMES), 'not all seven original Adam groups updated')
        for key in ('loss','unclipped_grad_norm','postclip_gradient_norm'):
            s.require(math.isfinite(row[key]), f'nonfinite observed {key}')
        for j, group in enumerate(source['groups']):
            name = group['name']; multiplier = factor if j < 3 else 1.
            used = group['lr'] * multiplier * .9995 ** index
            following = used * .9995
            s.require(math.isclose(row['learning_rates_used'][name], used, rel_tol=1e-12, abs_tol=0.)
                      and math.isclose(row['learning_rates_next'][name], following, rel_tol=1e-12, abs_tol=0.),
                      f'actual trace learning rate differs at {iteration}/{name}')
        s.require(row.get('nonlocal', {}).get('enabled') is False, 'unexpected active nonlocal prior')
    return {'count': steps, 'first_shifts': shifts[:4], 'all_absolute_iterations_and_rates_checked': True}


def validate_frequency_use(frequency, traces, profile):
    s=supervisor();t=profile['frequency_t'];steps=profile['steps']
    s.require(frequency.get('status') == 'complete' and frequency.get('method_variant') == 'frequency_adam'
              and frequency.get('frequency_t') == t and frequency.get('geometry_factor') == profile['geometry_factor']
              and frequency.get('installed_before_first_forward') is True
              and frequency.get('completed_steps') == steps and frequency.get('loss_calls_total') == 2*steps,
              'actual frequency installation/step counts differ')
    dc=frequency.get('dc_norm',{})
    s.require(set(dc) == {'pre','post'} and all(type(v) in (int,float) and math.isfinite(v) and v > 0 for v in dc.values()),
              'original per-scan DC denominator missing')
    s.require(frequency.get('final_weights_check') == {'passed':True,'actual_trainer_weights_match':True,
        'source_weights_unchanged':True,'original_masks_unchanged':True},'final actual source/installed weight checks failed')
    evidence=frequency['frequency_evidence'];expected={}
    for name,key in (('effective_weight','effective_weight_byte_sha256'),('source_weight','source_weight_byte_sha256'),
                     ('acquired_mask','acquired_mask_byte_sha256')):
        expected[key]={scan:evidence['scans'][scan][name]['byte_sha256'] for scan in ('pre','post')}
    for index,row in enumerate(traces):
        use=row.get('frequency_use',{})
        s.require(use == {'frequency_t':t,'installed_before_first_forward':True,'actual_trainer_weights_match':True,
            'loss_call_scans':['pre','post'],'loss_calls_total':2*(index+1),'dc_norm':dc,**expected},
            'actual per-step loss frequency weights/source denominator differ')
    s.require(frequency.get('final_frequency_use') == (traces[-1]['frequency_use'] if traces else None),
              'terminal actual frequency use differs')


def verify_run(args, launch, protocol):
    runtime()
    import torch
    s = supervisor(); run = ROOT / 'outputs' / args.exp
    metrics, provenance, geometry = [s.read(run / name) for name in
                                    ('metrics.json','provenance.json','geometry_provenance.json')]
    profile = s.stage_profile(args.stage, args.geometry_factor,getattr(args,'frequency_t',None)); end = 4000 + profile['steps']
    result = {'status': 'checking', 'passed': False, 'stage': args.stage, 'geometry_factor': args.geometry_factor,
              'method_variant':'frequency_adam','frequency_t':profile['frequency_t'],
              'code_sha256': launch['code_sha256'], 'invariants': launch['invariants'], 'checks': {}}
    try:
        s.require(provenance['status'] == geometry['status'] == 'complete', 'original/geometry runner did not complete')
        s.require(provenance['source_model_shapes_before_restore'] == SOURCE_SHAPES,
                  'source checkpoint is not the original seven 3.6M parameter shapes')
        s.require(provenance['source_checkpoint_sha256'] == launch['invariants']['source_checkpoint_sha256']
                  and provenance['source_config_sha256'] == launch['invariants']['source_config_sha256']
                  and provenance['source_checkpoint'] == launch['invariants']['input_manifest']['source_checkpoint']
                  and provenance['source_config'] == launch['invariants']['input_manifest']['source_config']
                  and provenance['runner_sha256'] == s.sha256(BASE_RUNNER), 'original source provenance differs')
        actual = geometry['restore']; source = actual['source']; after = actual['after']; final = geometry['final']
        s.require(actual['factor'] == args.geometry_factor and actual['source_iteration'] == 4000,
                  'actual applied geometry controls differ')
        for value, iteration in ((source,4000),(after,4000),(geometry['after_restore'],4000),(final,end)):
            s.require([g['name'] for g in value['groups']] == list(NAMES)
                      and all(g['step'] == iteration and g['betas'] == [.9,.999] for g in value['groups'])
                      and value['scheduler']['last_epoch'] == iteration
                      and value['scheduler']['step_count'] == iteration + 1
                      and value['scheduler']['gamma'] == .9995, 'actual Adam/scheduler clock or betas differ')
            s.require(len(value['scheduler']['base_lrs']) == len(value['scheduler']['last_lrs']) == 7
                      and all(value['scheduler']['base_lrs'][i] == group['initial_lr']
                              and value['scheduler']['last_lrs'][i] == group['lr']
                              for i, group in enumerate(value['groups'])), 'actual scheduler base/last LR differs from group rates')
        s.require(geometry['after_restore'] == after, 'initial actual optimizer observation differs from applied controls')
        for index, (old, applied, last) in enumerate(zip(source['groups'], after['groups'], final['groups'])):
            factor = args.geometry_factor if index < 3 else 1.
            s.require(applied['lr'] == old['lr'] * factor and applied['initial_lr'] == old['initial_lr'] * factor
                      and math.isclose(last['lr'], applied['lr'] * .9995 ** profile['steps'], rel_tol=1e-12)
                      and last['initial_lr'] == applied['initial_lr'], 'actual geometry/density rate intervention differs')
        result['checks']['actual_optimizer_rates'] = True
        s.require(provenance['mode'] == metrics['mode'] == profile['mode']
                  and provenance['end_iteration'] == metrics['end_iteration'] == end
                  and metrics['forward_backward_evaluations'] == profile['steps'], 'actual budget/mode differs')
        traces = [json.loads(line) for line in (run/'step_trace.jsonl').read_text().splitlines()]
        result['trace'] = validate_trace(traces, profile['steps'], args.geometry_factor, source, protocol)
        result['checks']['wavelet_replay'] = True
        frequency=s.read(run/'frequency_provenance.json')
        s.require(all(value.get('method_variant') == 'frequency_adam' and value.get('frequency_t') == profile['frequency_t']
                      and value.get('geometry_factor') == args.geometry_factor and value.get('frequency_adam') == frequency
                      for value in (metrics,provenance,geometry)), 'frequency protocol/records disagree')
        validate_frequency_use(frequency,traces,profile)
        result['frequency_files']=s.validate_frequency_files(run,frequency['frequency_evidence'],profile['frequency_t'])
        s.validate_frequency_check(frequency['frequency_array_check'],frequency['frequency_evidence'],profile['frequency_t'])
        import yaml
        cfg=yaml.safe_load((run/'actual_config.yaml').read_text())
        s.require(cfg.get('frequency_overrides') == {'frequency_t':profile['frequency_t'],
            'mask_source':'original acquired mask_p/mask_c; never fit masks','utility_sha256':s.sha256(ROOT/'frequency_weights.py')},
            'actual frequency configuration differs')
        result['checks']['actual_frequency_weights'] = result['checks']['independent_frequency_arrays'] = True
        reference_metrics = s.read(s.NOOP_REFERENCES[0] / 'metrics.json')
        s.require(base_snapshot(metrics['before']) == base_snapshot(reference_metrics['before']),
                  'initial model/moment tensor hashes differ from strict source control')
        result['checks']['initial_model_and_moments_exact'] = True
        checkpoint_path = Path(metrics['final_checkpoint']).resolve()
        s.require(checkpoint_path.is_relative_to(run.resolve()), 'final checkpoint is outside this fresh run')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        s.require(checkpoint.get('frequency_adam_recipe') == checkpoint_recipe(frequency),
                  'ordinary Adam checkpoint frequency/geometry resume recipe differs')
        result['checkpoint'] = validate_checkpoint(checkpoint, SOURCE_SHAPES, end, metrics['after'], final)
        result['checks']['ordinary_adam_checkpoint'] = True
        del checkpoint
        if args.stage == 'audit0':
            checks = metrics['audit_checks']
            s.require(base_snapshot(metrics['before']) == base_snapshot(metrics['after']) and source == after == final,
                      'zero-step control changed model/Adam/LR/scheduler states')
            s.require(set(checks) == {'pre','post'} and all(
                checks[scan]['passed'] is True and checks[scan]['zf_ratio_relative_error'] <= 1e-3
                and math.isfinite(checks[scan]['source_array_relative_l2'])
                and checks[scan]['source_array_relative_l2'] <= 1e-5 for scan in checks),
                'original zero metric/gauge or predeclared complex reproduction bound failed')
            result['zero_audit'] = {'passed': True, 'reconstruction_relative_l2_tolerance': 1e-5,
                'registered_head_psnr': {'pre':34.690260046170366,'post':36.93978348808641},
                'relative_l2': {scan:checks[scan]['source_array_relative_l2'] for scan in checks},
                'exact_equal': {scan:checks[scan]['source_array_relative_l2'] == 0. for scan in checks}}
        if args.stage == 'noop2':
            s.require(actual['is_noop'] is True and source == after, 'factor-1 application was not a true no-op')
            ar = s.load_campaign(); old_ar = sys.modules.get('autoresearch'); sys.modules['autoresearch'] = ar
            try: preflight = load_file(BASE_ROOT/'run_preflight.py', '_geometry_original_final_drift')
            finally:
                if old_ar is None: sys.modules.pop('autoresearch',None)
                else: sys.modules['autoresearch'] = old_ar
            comparisons = []
            for reference in s.NOOP_REFERENCES:
                prior = s.read(reference/'metrics.json'); old_provenance = s.read(reference/'provenance.json')
                old_traces = [json.loads(line) for line in (reference/'step_trace.jsonl').read_text().splitlines()]
                s.require(base_snapshot(metrics['before']) == base_snapshot(prior['before']) and len(old_traces) == 2,
                          'numerical reference initial state or budget differs')
                differences = []
                for left,right in zip(old_traces,traces):
                    for key in ('absolute_iteration','wavelet_shift','learning_rates_used','learning_rates_next'):
                        s.require(left[key] == right[key], f'no-op reference trace {key} differs')
                    s.require(set(left['weighted_components']) == set(right['weighted_components']), 'loss component names differ')
                    differences.extend(abs(left[k]-right[k]) for k in ('loss','unclipped_grad_norm','postclip_gradient_norm'))
                    differences.extend(abs(left['weighted_components'][k]-right['weighted_components'][k]) for k in left['weighted_components'])
                s.require(all(math.isfinite(v) and v <= 1e-7 for v in differences), 'no-op loss/gradient drift exceeds 1e-7')
                head = {scan: abs(metrics['rows'][scan]['head_psnr'] - prior['rows'][scan]['head_psnr']) for scan in ('pre','post')}
                s.require(all(math.isfinite(v) and v <= .001 for v in head.values()), 'no-op head PSNR drift exceeds .001dB')
                for key in ('normalization_protocol','export_protocol'):
                    s.require(provenance[key] == old_provenance[key], 'original export/scoring gauge differs')
                drift = preflight.final_drift(reference, run, prior, metrics)
                comparisons.append({'reference_run':str(reference), 'maximum_loss_gradient_difference':max(differences),
                                    'head_psnr_difference':head,'final_drift':drift})
            result['noop'] = {'passed': True, 'comparisons':comparisons,
                'tolerances': {'model_relative_l2':1e-6,'adam_relative_l2':1e-4,'reconstruction_relative_l2':1e-5,
                               'scalar_loss_gradient_absolute':1e-7,'head_psnr_absolute_db':.001}}
        s.require(geometry.get('backend',{}).get('status') == 'constructed'
                  and geometry['backend'].get('compiler_branch_bypassed') is True
                  and geometry['backend'].get('prebuilt_extension') == launch['invariants']['prebuilt_extension'],
                  'strict verified prebuilt backend was not observed')
        result.update(status='complete',passed=True)
    except BaseException as exc:
        result.update(status='validation_failed',error=repr(exc)); raise
    finally:
        write_record(run/'validation.json',result)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ['--describe-runtime']:
        print(json.dumps(runtime())); return
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--stage', required=True, choices=('audit0','noop2','trial500'))
    p.add_argument('--exp',required=True);p.add_argument('--geometry-factor',type=float,required=True)
    p.add_argument('--frequency-t',type=float,choices=(0.,1.),required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--launch-manifest',type=Path,required=True)
    args=p.parse_args(argv); observed_runtime=runtime(); s=supervisor()
    profile=s.stage_profile(args.stage,args.geometry_factor,args.frequency_t);launch=s.read(args.launch_manifest)
    s.require(launch['code_sha256'] == code_hashes() and launch['parent_pid'] == os.getppid()
              and launch['request']['stage'] == args.stage and launch['request']['exp'] == args.exp
              and launch['request']['geometry_factor'] == args.geometry_factor
              and launch['request']['frequency_t'] == args.frequency_t,
              'child request/source/parent differs from owned launch manifest')
    s.require(Path('/proc/'+str(os.getppid())+'/exe').resolve() == s.CPU_PYTHON.resolve()
              and launch['parent_runtime']['torch_imported'] is False
              and launch['command'] == [str(GSTMI),'-X','utf8','-B','-u',str(ROOT/'runner.py'),*argv],
              'child was not launched by the reviewed stdlib-only parent and fixed GSTMI command')
    s.require(args.gpu >= 0 and os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu)
              and launch['initial_visibility'] == str(args.gpu), 'physical GPU reservation differs')
    base=load_base_runner();protocol=load_resume_protocol();old_argv=list(sys.argv);old_cwd=Path.cwd()
    record=None
    try:
        with configure_runner(base,protocol,args.geometry_factor,launch,args.frequency_t) as record:
            record['runtime']=observed_runtime;record['stage']=args.stage;record['wrapper_argv']=argv
            sys.argv=[str(ROOT/'runner.py'),'--mode',profile['mode'],'--gpu',str(args.gpu),'--exp',args.exp,
                      '--steps',str(profile['steps']),'--local-factor','1','--nlambda','0','--sigma-rel','.015']
            base.main()
        metrics=supervisor().read(ROOT/'outputs'/args.exp/'metrics.json')
        attach_checkpoint_recipe(Path(metrics['final_checkpoint']),record['frequency_adam'])
        verify_run(args,launch,protocol)
    except BaseException as exc:
        fail_owned_run(record, exc)
        raise
    finally:
        sys.argv=old_argv;os.chdir(old_cwd)


if __name__ == '__main__':
    main()
