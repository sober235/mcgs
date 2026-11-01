#!/usr/bin/env python3
"""Long-budget / LR-schedule continuation of the MC-GS CE AF16 keeper chain (stage 2 onward).

Base checkpoints are the iteration-4500 stage-1 outputs of the two verified keeper-chain controls
(seed bases 42 = i1_ctrl, 7 = b1s7_ctrl in mcgs_ce_densify_20261029/stack, read-only). Stage 2 is the
K fast-TV continuation (profile from recipe) run for recipe['steps'] updates with optional LR edits;
at every milestone the canonical export is written and relaxed with alpha, then scored with the frozen
evaluator. Milestone 500 with the default recipe reproduces the control chain.

Editable surface: recipe.json.  Commands: stage2 (numeric child), run, ledger.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

WS = Path(__file__).resolve().parent
AR = Path('/home/congcongliu/codex_artifacts/mcgs_ce_autoresearch_20261017')
K = Path('/home/congcongliu/codex_artifacts/mcgs_ce_karpathy_20261022_relocation')
DENSIFY = Path('/home/congcongliu/codex_artifacts/mcgs_ce_densify_20261029')
G2 = AR / 'frequency_adam/outputs/adamflat_g2_v1'
PY = Path('/home/congcongliu/anaconda3/envs/GSTMI/bin/python')
TOPIC = Path('/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois')
RUNS = WS / 'runs'
LOCK = RUNS / '.gpu_claim.lock'
CLAIMS = RUNS / '.gpu_claims.json'
NAMES = ('positions', 'scales', 'rotations', 'density_real', 'density_imag',
         'density_real_contrast', 'density_imag_contrast')
BASE_LRS = (.0006, .004, .0006, .015, .015, .015, .015)
GAMMA = .9995
SCANS = (('pre', 'plain', 'p'), ('post', 'contrast', 'c'))
STAGE2_EXP = 'fixed_count_relocation'
SCRUB = ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'CUDA_LAUNCH_BLOCKING', 'PYTHONSTARTUP', 'PYTHONPATH',
         'PYTHONHOME', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS')
BASES = {42: DENSIFY / 'stack/i1_ctrl/stage1/training/ar_i1_k30p90/checkpoints/latest.pth',
         7: DENSIFY / 'stack/b1s7_ctrl/stage1/training/ar_i1_k30p90/checkpoints/latest.pth'}
CONTROLS = {42: WS / 'runs/L6_steps1000_lr_quarter/scores/s42_m0625/objective.json',
            7: WS / 'runs/L6_steps1000_lr_quarter/scores/s7_m0625/objective.json'}
KEEP_MARGIN = 1e-4
MAX_STEPS = 4000
DEFAULT_RECIPE = {'steps': 500, 'milestones': [500], 'gamma': GAMMA, 'lr_scale': 1., 'geo_lr_scale': 1.,
                  'profile': 'tv1125_wav075', 'alpha': .5}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 23), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_recipe(path):
    recipe = {**DEFAULT_RECIPE, **read_json(path)}
    unknown = set(recipe) - set(DEFAULT_RECIPE)
    require(not unknown, f'unknown recipe keys {sorted(unknown)}')
    steps, milestones = recipe['steps'], recipe['milestones']
    require(isinstance(steps, int) and 1 <= steps <= MAX_STEPS, f'steps must be 1..{MAX_STEPS}')
    require(milestones == sorted(set(milestones)) and all(isinstance(m, int) and 1 <= m <= steps for m in milestones)
            and milestones[-1] == steps, 'milestones must be increasing integers ending at steps')
    require(0 < recipe['gamma'] <= 1 and recipe['lr_scale'] > 0 and recipe['geo_lr_scale'] > 0 and 0 <= recipe['alpha'] <= 1,
            'invalid gamma / lr scale / alpha')
    return recipe


def g2_gauges():
    metrics = read_json(G2 / 'metrics.json')
    return {scan: {key: metrics['audit_checks'][scan][key] for key in ('scale_trainer', 'scale_gauge', 'factor_scale')}
            for scan, _, _ in SCANS}


def topic_array(image, gauge):
    import numpy as np
    require(math.isclose(gauge['factor_scale'], gauge['scale_trainer'] / gauge['scale_gauge'], rel_tol=1e-15),
            'inconsistent frozen reporting gauge')
    value = (image.detach().cpu().numpy().astype(np.complex64) * np.float32(gauge['factor_scale'])).astype(np.complex64)
    require(value.ndim == 3 and np.isfinite(value).all(), 'export requires a finite complex volume')
    return value


def runtime_record(torch):
    return {'python': sys.executable, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'device': torch.cuda.get_device_name(0),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'ld_library_path': os.environ.get('LD_LIBRARY_PATH'), 'omp_num_threads': os.environ.get('OMP_NUM_THREADS')}


def numeric_setup(run_dir):
    require(Path(sys.executable).resolve() == PY.resolve(), 'numeric stages require the GSTMI interpreter')
    sys.dont_write_bytecode = True
    sys.path[:] = [entry for entry in sys.path if Path(entry or '.').resolve() != WS]
    for key, relative in (('TMPDIR', 'tmp'), ('TORCH_EXTENSIONS_DIR', 'cache/torch'), ('CUPY_CACHE_DIR', 'cache/cupy'),
                          ('CUDA_CACHE_PATH', 'cache/cuda'), ('MPLCONFIGDIR', 'cache/matplotlib'),
                          ('XDG_CACHE_HOME', 'cache/xdg')):
        path = run_dir / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    import torch
    require(torch.__version__ == '2.7.1+cu118' and torch.version.cuda == '11.8', 'unexpected Torch/CUDA runtime')
    require(torch.cuda.device_count() == 1, 'exactly one visible GPU is required')
    hold = torch.zeros(1, device='cuda:0')
    write_json(run_dir / 'context_ready.json', {'pid': os.getpid(), 'device': torch.cuda.get_device_name(0),
                                                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
    return torch, hold


class Trace:
    def __init__(self, path, begin, end, after=None):
        self.path, self.begin, self.end, self.after = Path(path), begin, end, after
        self.rows, self.row, self.started = [], None, None
        self.path.touch(exist_ok=False)

    def __call__(self, phase, iteration, data):
        import torch
        if phase == 'start':
            torch.cuda.synchronize()
            self.started = time.perf_counter()
            self.begin(iteration)
        elif phase == 'pre_step':
            self.row = {'iteration': iteration, 'loss': data['loss'], 'wavelet_shift': data['wavelet_shift'],
                        'unclipped_grad_norm': data['unclipped_grad_norm'],
                        'weighted_components': data['weighted_components']}
        elif phase == 'end':
            require(self.row is not None and self.row['iteration'] == iteration, 'missing pre-step record')
            self.row['frequency_use'] = self.end(iteration)
            torch.cuda.synchronize()
            self.row['seconds'] = time.perf_counter() - self.started
            self.rows.append(self.row)
            with self.path.open('a') as stream:
                stream.write(json.dumps(self.row, allow_nan=False) + '\n')
            self.row = None
            if self.after is not None:
                self.after(iteration)
        else:
            raise ValueError(f'unknown step phase {phase}')


def validate_checkpoint(trainer, checkpoint, named):
    import torch
    require(isinstance(checkpoint, dict) and checkpoint.get('iter') == 4500, 'stage-2 source iteration must be 4500')
    training = trainer.config['training']
    require(training.get('resume_offset') == 4500 and 4500 <= training.get('max_iterations', -1) <= 4500 + MAX_STEPS
            and training.get('seed') == 42 and training.get('lr_scheduler') == {'type': 'exponential', 'gamma': GAMMA},
            'training configuration must retain seed 42, offset 4500 and the exponential schedule')
    model_state = checkpoint['model_state']
    require(set(model_state) == set(NAMES), 'checkpoint must contain exactly the seven original tensors')
    count = named['positions'].shape[0]
    for index, (name, param) in enumerate(named.items()):
        shape = (count, 3) if index < 2 else (count, 4) if index == 2 else (count,)
        value = model_state[name]
        require(tuple(param.shape) == shape and tuple(value.shape) == shape and value.dtype == param.dtype == torch.float32
                and bool(torch.isfinite(value).all()), f'model tensor differs for {name}')
    saved = checkpoint['optim_state']
    groups, states = saved['param_groups'], saved['state']
    require(len(groups) == 7 and set(states) == set(range(7)), 'Adam must have seven canonical groups')
    for index, (name, base_lr, group) in enumerate(zip(NAMES, BASE_LRS, groups)):
        require(group['params'] == [index], f'Adam group order differs for {name}')
        require(math.isclose(group['initial_lr'], base_lr, rel_tol=1e-12, abs_tol=0)
                and math.isclose(group['lr'], base_lr * GAMMA ** 4500, rel_tol=1e-12, abs_tol=0),
                f'Adam saved learning rate differs for {name}')
        require(float(states[index]['step']) == 4500, f'Adam clock differs for {name}')


def relax(stage_dir, alpha, out_dir):
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=False)
    record = {'alpha': alpha, 'formula': 'raw + float32(alpha) * (canonical - raw)', 'scans': {}}
    for scan, _, _ in SCANS:
        raw = np.load(stage_dir / f'raw_{scan}_complex.npy', allow_pickle=False)
        canonical = np.load(stage_dir / f'canonical_{scan}_complex.npy', allow_pickle=False)
        require(raw.shape == canonical.shape and raw.dtype == canonical.dtype == np.complex64, 'matching complex64 exports')
        candidate = canonical.copy() if alpha == 1 else raw + np.float32(alpha) * (canonical - raw)
        require(candidate.dtype == np.complex64 and np.isfinite(candidate).all(), 'nonfinite relaxed export')
        path = out_dir / f'candidate_{scan}_complex.npy'
        np.save(path, candidate, allow_pickle=False)
        record['scans'][scan] = {'path': str(path), 'sha256': sha256(path)}
    write_json(out_dir / 'relax.json', record)


def stage2(args):
    run_dir = Path(args.out).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    recipe = load_recipe(args.recipe)
    write_json(run_dir / 'recipe.json', recipe)
    torch, hold = numeric_setup(run_dir)
    import numpy as np
    import yaml
    os.chdir(K)
    sys.path.insert(0, str(K))
    kruntime = importlib.import_module('runtime')
    require(Path(kruntime.__file__).resolve() == (K / 'runtime.py').resolve(), 'foreign runtime module')
    g2 = load_module(K / 'runner.py', '_long_k_runner')
    profiles = load_module(K / 'fast_tv_profiles.py', '_long_fast_tv_profiles')
    relocation = load_module(K / 'relocation_runtime.py', '_long_relocation_runtime')
    torch.manual_seed(42)
    np.random.seed(42)
    source = Path(args.ckpt).resolve()
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    state = checkpoint['model_state']
    count = int(state['positions'].shape[0])
    require(checkpoint.get('iter') == 4500 and set(state) == set(NAMES), 'expected a seven-tensor iteration-4500 checkpoint')
    steps = recipe['steps']
    source_config = yaml.safe_load((G2 / 'actual_config.yaml').read_text())
    config = profiles.transform_config(g2.build_config(source_config, run_dir, min(steps, 500)), recipe['profile'])
    config['training']['max_iterations'] = 4500 + steps
    config['training']['save_every'] = 10 ** 9
    config['output']['use_tensorboard'] = False
    config['output']['exp_name'] = STAGE2_EXP
    config['training']['seed'] = 42
    config['gaussian']['initial_num_points'] = count
    require(config['loss'].get('edge_tv_kappa_source') == 'zf' and not config.get('adaptive_control', {}).get('enable', False),
            'frozen G2 zf kappa / disabled adaptive policy differs')
    effective = profiles.validate_effective_loss(config['loss'], recipe['profile'])
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(config))
    gauges = g2_gauges()
    scales = {source_config['data'][channel]['data_path']: gauges[scan]['scale_trainer'] for scan, channel, _ in SCANS}
    trainer, field, frequency, _ = g2.setup_trainer(config, state, scales, run_dir, {'mode': 'baseline'})
    require(field is None and trainer.voxelizer.backend == 'tile_cuda' and trainer.voxelizer.strict_cuda
            and trainer.voxelizer.use_cuda, 'strict original CUDA renderer and no networks required')
    helpers = load_module(AR / 'proximal_dc/runner.py', '_long_proxdc_helpers')
    helpers.validate_acquisition(trainer)
    relocation._validate_checkpoint = validate_checkpoint
    restore = relocation.restore_training_state(trainer, checkpoint)
    del checkpoint, state
    trainer.scheduler.gamma = recipe['gamma']
    for index, group in enumerate(trainer.optimizer.param_groups):
        group['lr'] *= recipe['lr_scale'] * (recipe['geo_lr_scale'] if index < 3 else 1.)
    lr_after = [group['lr'] for group in trainer.optimizer.param_groups]
    milestones = {4500 + m: m for m in recipe['milestones']}
    exported = {}

    def export(iteration):
        if iteration not in milestones:
            return
        out = run_dir / f'm{milestones[iteration]:04d}'
        out.mkdir()
        with torch.no_grad():
            exports, _ = helpers.canonical_control(trainer, backend_kind='cuda')
        arrays = {}
        for scan, _, _ in SCANS:
            for kind in ('raw', 'canonical'):
                path = out / f'{kind}_{scan}_complex.npy'
                with path.open('xb') as stream:
                    np.save(stream, topic_array(exports[scan][kind], gauges[scan]), allow_pickle=False)
                arrays[f'{kind}_{scan}'] = sha256(path)
        del exports
        torch.cuda.empty_cache()
        relax(out, recipe['alpha'], out / f'relax_a{recipe["alpha"]:g}')
        exported[milestones[iteration]] = {'dir': str(out), 'sha256': arrays,
                                           'lr': [group['lr'] for group in trainer.optimizer.param_groups]}
        write_json(run_dir / 'milestones.json', exported)

    guard = frequency._frequency_guard
    trace = Trace(run_dir / 'step_trace.jsonl', lambda it: kruntime.begin_frequency_step(guard(trainer), 4500, it),
                  lambda it: kruntime.finish_frequency_step(guard(trainer), 4500, it), after=export)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    trainer.train(step_callback=trace)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    require(len(trace.rows) == steps and trace.rows[-1]['iteration'] == 4500 + steps, 'observed update count differs')
    require(sorted(exported) == recipe['milestones'], 'missing milestone exports')
    metrics = {'stage': 'stage2', 'recipe': recipe, 'effective_coefficients': effective, 'source_checkpoint': str(source),
               'points': count, 'steps': steps, 'lr_after_edit': lr_after,
               'restore': {key: value for key, value in restore.items() if key not in ('before', 'after')},
               'first_steps': trace.rows[:2], 'last_loss': trace.rows[-1]['loss'], 'training_seconds': elapsed,
               'peak_allocated_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
               'peak_reserved_gb': torch.cuda.max_memory_reserved() / 2 ** 30, 'milestones': exported,
               'runtime': runtime_record(torch)}
    write_json(run_dir / 'metrics.json', metrics)
    print(json.dumps({'stage': 'stage2', 'first_loss': trace.rows[0]['loss'], 'last_loss': trace.rows[-1]['loss'],
                      'seconds': elapsed}), flush=True)


def child_env(gpu):
    env = {key: value for key, value in os.environ.items() if key not in SCRUB}
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu), PYTHONNOUSERSITE='1', PYTHONPATH='',
               PYTHONDONTWRITEBYTECODE='1', CUDA_CACHE_DISABLE='1', OMP_NUM_THREADS='1')
    return env


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def live_claims():
    claims = read_json(CLAIMS) if CLAIMS.is_file() else {}
    return {gpu: claim for gpu, claim in claims.items() if pid_alive(claim['pid'])}


def select_gpu(need_gb, exclude):
    sys.path.insert(0, str(WS))
    import pick_gpu
    chosen = pick_gpu.pick(pick_gpu.snapshot(3), 1, need_gb * 1024, 60., set(exclude))
    return chosen[0] if chosen else None


def launch_numeric(argv, run_dir, log, need_gb=55., wait_limit=14400., ready_timeout=600.):
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    waited = 0.
    while True:
        with LOCK.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                claims = live_claims()
                gpu = select_gpu(need_gb, {int(key) for key in claims})
                if gpu is not None:
                    stream = Path(log).open('x')
                    command = [str(PY), '-X', 'utf8', '-B', '-u', str(Path(__file__).resolve()), 'stage2', *map(str, argv)]
                    process = subprocess.Popen(command, env=child_env(gpu), stdout=stream, stderr=subprocess.STDOUT,
                                               start_new_session=True, cwd=str(WS))
                    claims[str(gpu)] = {'pid': process.pid, 'run_dir': str(run_dir)}
                    write_json(CLAIMS, claims)
                    deadline = time.monotonic() + ready_timeout
                    while (not (Path(run_dir) / 'context_ready.json').exists() and process.poll() is None
                           and time.monotonic() < deadline):
                        time.sleep(1.)
                    # Hold the lock until the child's memory footprint is visible to later pickers.
                    time.sleep(90.)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        if gpu is not None:
            break
        require(waited < wait_limit, 'no GPU qualified under the shared-server rule within the wait limit')
        time.sleep(60.)
        waited += 60.
    returncode = process.wait()
    stream.close()
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            claims = live_claims()
            claims.pop(str(gpu), None) if claims.get(str(gpu), {}).get('pid') == process.pid else None
            write_json(CLAIMS, claims)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return {'gpu': gpu, 'command': command, 'returncode': returncode, 'log': str(log)}


def score(run_dir, out_dir, name, log):
    env = {key: value for key, value in os.environ.items() if key not in SCRUB}
    env.update(PYTHONNOUSERSITE='1', PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
               RETRO_TOPIC=str(TOPIC), RETRO_TRAINER=str(TOPIC / 'trainer/_trainer_e778bf2_ar'))
    with Path(log).open('x') as stream:
        returncode = subprocess.run([str(PY), '-B', str(K / 'evaluate.py'), '--run-dir', str(run_dir),
                                     '--out-dir', str(out_dir), '--name', name],
                                    env=env, stdout=stream, stderr=subprocess.STDOUT, cwd=str(WS)).returncode
    require(returncode == 0, f'evaluator failed for {name}; see {log}')
    return objective_summary(out_dir / 'objective.json')


def objective_summary(path):
    value = read_json(path)
    planes = value['lesion_planes']
    best = planes['best_plane']
    limit = min(best['comparisons'].items(), key=lambda item: item[1]['margin'])
    return {'wh': value['score'], 's6': planes['score_six'], 'plane': [best['axis'], best['index']],
            'limit': limit[0], 'limit_opponents': limit[1]['labels'], 'strict_success': planes['strict_success'],
            'wh_strict_success': value['strict_success'],
            'passing_planes_by_scan': planes['passing_planes_by_scan'], 'objective': str(path)}


def run_seed(root, seed, recipe_path, recipe, record, errors):
    try:
        out = root / f's{seed}'
        if not out.exists():
            result = launch_numeric(['--ckpt', BASES[seed], '--out', out, '--recipe', recipe_path], out, root / f's{seed}.log')
            record['stages'][str(seed)] = result
            require(result['returncode'] == 0, f"stage2 seed {seed} failed; see {result['log']}")
        elif not (out / 'metrics.json').is_file():
            pid = read_json(out / 'context_ready.json')['pid']
            while pid_alive(pid):
                time.sleep(30.)
            require((out / 'metrics.json').is_file(), f'existing stage2 seed {seed} ended without metrics; see {root}/s{seed}.log')
            record['stages'][str(seed)] = {'resumed_pid': pid}
        (root / 'scores').mkdir(exist_ok=True)
        for m in recipe['milestones']:
            target = out / f'm{m:04d}' / f'relax_a{recipe["alpha"]:g}'
            key = f's{seed}_m{m:04d}'
            if (root / 'scores' / key / 'objective.json').is_file():
                record['scores'][key] = objective_summary(root / 'scores' / key / 'objective.json')
                continue
            record['scores'][key] = score(target, root / 'scores' / key, f'{root.name}_{key}', root / 'scores' / f'{key}.log')
    except Exception as error:
        errors.append(f'seed {seed}: {error}')


def run(args):
    recipe_path = Path(args.recipe).resolve()
    recipe = load_recipe(recipe_path)
    root = (RUNS / args.name).resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    seeds = [int(s) for s in args.seeds.split(',')]
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=WS).stdout.strip()
    record = {'name': args.name, 'commit': commit, 'recipe': recipe, 'seeds': seeds, 'started_unix': time.time(),
              'stages': {}, 'scores': {}}
    write_json(root / 'run.json', record)
    errors = []
    threads = []
    for seed in seeds:
        thread = threading.Thread(target=run_seed, args=(root, seed, recipe_path, recipe, record, errors))
        thread.start()
        threads.append(thread)
        time.sleep(5.)
    for thread in threads:
        thread.join()
    control = {seed: objective_summary(CONTROLS[seed]) for seed in seeds}
    control_s6 = sum(control[s]['s6'] for s in seeds) / len(seeds)
    control_wh = sum(control[s]['wh'] for s in seeds) / len(seeds)
    table = {}
    for m in recipe['milestones']:
        keys = [f's{s}_m{m:04d}' for s in seeds]
        if all(k in record['scores'] for k in keys):
            s6 = sum(record['scores'][k]['s6'] for k in keys) / len(keys)
            wh = sum(record['scores'][k]['wh'] for k in keys) / len(keys)
            table[m] = {'mean_s6': s6, 'mean_wh': wh, 'delta_s6': s6 - control_s6, 'delta_wh': wh - control_wh,
                        'per_seed_delta_s6': {s: record['scores'][f's{s}_m{m:04d}']['s6'] - control[s]['s6'] for s in seeds},
                        'strict_success_any_seed': any(record['scores'][k]['strict_success'] for k in keys)}
    best = max(table, key=lambda m: table[m]['mean_s6']) if table else None
    record.update(errors=errors, control={'per_seed': control, 'mean_s6': control_s6, 'mean_wh': control_wh},
                  milestones=table, best_milestone=best,
                  decision={'rule': f'keep if seed-mean relax S6 at some milestone > control mean + {KEEP_MARGIN}',
                            'keep': bool(best is not None and not errors and table[best]['delta_s6'] > KEEP_MARGIN)},
                  finished_unix=time.time())
    record['wall_seconds'] = record['finished_unix'] - record['started_unix']
    write_json(root / 'run.json', record)
    print(json.dumps({'name': args.name, 'errors': errors, 'best_milestone': best,
                      'milestones': {m: {k: round(v, 6) for k, v in t.items() if isinstance(v, float)} for m, t in table.items()},
                      'decision': record['decision']}), flush=True)


def ledger(args):
    root = RUNS / args.name
    record = read_json(root / 'run.json')
    status = args.status or ('keep' if record['decision']['keep'] else 'discard')
    best = record.get('best_milestone')
    with (WS / 'results.tsv').open('a') as table, (WS / 'ledger.jsonl').open('a') as journal:
        if record.get('errors') and not record['milestones']:
            table.write('\t'.join([record['commit'], args.name, '', '', '', '', '', 'crash', args.description]) + '\n')
        for m, row in sorted(record['milestones'].items(), key=lambda item: int(item[0])):
            row_status = status if str(m) == str(best) else 'milestone'
            line = [record['commit'], f'{args.name}_m{int(m):04d}', f"{row['mean_s6']:.6f}", f"{row['delta_s6']:+.6f}",
                    f"{row['mean_wh']:.6f}", f"{row['delta_wh']:+.6f}",
                    ' '.join(f"s{s}:{d:+.5f}" for s, d in row['per_seed_delta_s6'].items()), row_status, args.description]
            table.write('\t'.join(line) + '\n')
        journal.write(json.dumps({'name': args.name, 'status': status, 'description': args.description,
                                  'record': record, 'long_chain_sha256': sha256(Path(__file__).resolve())},
                                 allow_nan=False) + '\n')
    print(f'ledger appended for {args.name} ({status})')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    two = sub.add_parser('stage2')
    two.add_argument('--ckpt', required=True, type=Path)
    two.add_argument('--out', required=True, type=Path)
    two.add_argument('--recipe', required=True, type=Path)
    go = sub.add_parser('run')
    go.add_argument('--name', required=True)
    go.add_argument('--recipe', default=str(WS / 'recipe.json'))
    go.add_argument('--seeds', default='42,7')
    go.add_argument('--resume', action='store_true')
    led = sub.add_parser('ledger')
    led.add_argument('--name', required=True)
    led.add_argument('--description', required=True)
    led.add_argument('--status')
    args = parser.parse_args(argv)
    {'stage2': stage2, 'run': run, 'ledger': ledger}[args.command](args)


if __name__ == '__main__':
    main()
