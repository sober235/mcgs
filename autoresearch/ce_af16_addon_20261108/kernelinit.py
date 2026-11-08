#!/usr/bin/env python3
"""kernelinit.py -- add-on round H2 (2026-11-08): retrain the seed-42 base with voxel-isotropic Gaussian widths.

Finding (diag_gauss_width.py on TO02 s42): the frozen init (gaussian.init_cloud.scales_3nn) sets one isotropic scale
in NORMALISED coordinates (mean 3-NN distance, positions in [-1, 1] per axis). On the 256 x 232 x 176 grid that is
1.02 / 0.93 / 0.70 voxels std along axes 0 / 1 / 2, and the widths are still there after 5125 iterations. The
fully sampled readout (axis 0) gets the widest kernel; diag_direction.py shows MC-GS keeps only 0.72-0.83 of the
true power along it at |f| 0.2-0.5 (ZS-SSDU 0.94).

--seed 7 replays b1_i1_seed7 (identical config except training.seed; its densify trainer copy differs only in
adaptive-control code that this config disables). --width-scale c multiplies the voxel-isotropic width (H3 dose).

H2: in THIS process only, scales_3nn is wrapped so that axis i's normalised scale is multiplied by N_min / N_i,
i.e. every axis gets the voxel width the shortest axis already has (~0.70 voxel). Nothing else changes: the frozen
e778bf2 trainer, its config snapshot (ar_i1_k30p90: official mask, 3.6 M points, seed 42, 4000 iterations), the
frozen stage1 (stack_chain.stage1, 4000 -> 4500) and the threeaxis stage2 of the keeper (enqueued separately with
item "bases"). --width-mode normalised runs the identical chain with the original init: the retrain control.

Commands: chain (driver on ONE launcher-exposed GPU: base child then stage1 child), base, stage1 (GPU children),
check-init (CPU: the wrapped init on a synthetic cloud).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PY = Path('/home/congcongliu/anaconda3/envs/GSTMI/bin/python')
TOPIC = Path('/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois')
MASK = TOPIC / 'sim/mask/mask_af16_3d.h5'
BASE_TRAINER = TOPIC / 'trainer/_trainer_e778bf2_ar'
BASE_CFG = BASE_TRAINER / 'results_arc/ar_i1_k30p90/ar_i1_k30p90/config.yaml'
BASE_SO = BASE_TRAINER / 'paired_gs/gaussian/cuda_ext/build/tile_voxelize_cuda_0141c525_7aedd3b0.so'
BASE_SO_SHA = '3552766d6cec6b7239d1b6dda1e7c36003aa60ab6644548b232271a4853d55dc'
ARTS = Path('/data0/congcong/code/GS/3DGS/outputs/codex_artifacts')
DENSIFY = ARTS / 'mcgs_ce_densify_20261029'
STACK_CHAIN = DENSIFY / 'stack_chain.py'
ROOT = ARTS / 'mcgs_ce_addon_20261108'
EXP = 'ar_i1_k30p90'
SHAPE = (256, 232, 176)
MODES = ('normalised', 'voxel_iso')
SCRUB = ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'CUDA_LAUNCH_BLOCKING', 'PYTHONSTARTUP', 'PYTHONPATH',
         'PYTHONHOME', 'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')


def log(*a):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), *a, flush=True)


def require(ok, msg):
    if not ok:
        raise SystemExit(f'REFUSED: {msg}')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def axis_factors(shape=SHAPE):
    n = min(shape)
    return tuple(n / s for s in shape)


def voxel_std(scales, shape=SHAPE):
    """Axis-aligned init (identity rotation): voxel std along axis i = s_i * N_i / 2."""
    import torch
    return scales * torch.tensor([s / 2 for s in shape], dtype=scales.dtype, device=scales.device)


def install_width(init_cloud, mode, record, width_scale=1.0):
    """Wrap init_cloud.scales_3nn (the trainer imports it from the module at construction time)."""
    require(mode in MODES, f'width mode {mode}')
    require(0.3 <= width_scale <= 1.5 and (mode == 'voxel_iso' or width_scale == 1.0), f'width_scale {width_scale}')
    orig = init_cloud.scales_3nn

    def scales_3nn(positions, shape, scale_mult=1.0, **kw):
        import torch
        base = orig(positions, shape, scale_mult, **kw)
        require(tuple(shape) == SHAPE and base.shape[1] == 3 and torch.equal(base[:, 0], base[:, 1])
                and torch.equal(base[:, 0], base[:, 2]), 'frozen scales_3nn is no longer isotropic per point')
        out = base if mode == 'normalised' else base * torch.tensor(axis_factors(shape), dtype=base.dtype,
                                                                    device=base.device) * width_scale
        vs = voxel_std(out, shape)
        record.update(mode=mode, width_scale=width_scale, factors=axis_factors(shape) if mode == 'voxel_iso' else (1., 1., 1.),
                      points=int(out.shape[0]), median_voxel_std=[float(vs[:, i].median()) for i in range(3)],
                      median_voxel_std_original=[float(voxel_std(base, shape)[:, i].median()) for i in range(3)])
        log(f'[kernelinit] {mode}: median voxel std {record["median_voxel_std"]} '
            f'(original {record["median_voxel_std_original"]})')
        return out
    init_cloud.scales_3nn = scales_3nn


# ---------------------------------------------------------------- base 0 -> 4000 (GPU child)
def cmd_base(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    require(Path(sys.executable).resolve() == PY.resolve(), 'numeric stages require the GSTMI interpreter')
    sys.dont_write_bytecode = True
    for key, rel in (('TMPDIR', 'tmp'), ('TORCH_EXTENSIONS_DIR', 'cache/torch'), ('CUDA_CACHE_PATH', 'cache/cuda'),
                     ('MPLCONFIGDIR', 'cache/matplotlib'), ('XDG_CACHE_HOME', 'cache/xdg')):
        (out / rel).mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(out / rel)
    import torch
    import yaml
    require(torch.cuda.device_count() == 1, 'exactly one visible GPU is required')
    hold = torch.zeros(1, device='cuda:0')  # noqa: F841
    os.chdir(BASE_TRAINER)
    sys.path[:0] = [str(BASE_TRAINER / 'paired_gs'), str(BASE_TRAINER)]
    require(sha256(BASE_SO) == BASE_SO_SHA, 'prebuilt base binary changed; compilation is forbidden')
    import gaussian.tile_voxelizer as tile
    require(Path(tile.__file__).resolve() == (BASE_TRAINER / 'paired_gs/gaussian/tile_voxelizer.py').resolve(),
            'foreign tile voxelizer module')
    tile._cuda_ext = load_module(BASE_SO, BASE_SO.stem)
    tile._cuda_load_error = None
    import gaussian.init_cloud as init_cloud
    require(Path(init_cloud.__file__).resolve() == (BASE_TRAINER / 'paired_gs/gaussian/init_cloud.py').resolve(),
            'foreign init_cloud module')
    record = {}
    install_width(init_cloud, args.width_mode, record, args.width_scale)
    from seed import set_seed
    import trainers.joint_trainer as jt
    require(Path(jt.__file__).resolve() == (BASE_TRAINER / 'paired_gs/trainers/joint_trainer.py').resolve(),
            'foreign joint trainer module')
    cfg = yaml.safe_load(BASE_CFG.read_text())
    require(args.seed in (42, 7), 'seeds 42 (ar_i1_k30p90) and 7 (b1_i1_seed7: same config, seed 7) only')
    require(cfg['gaussian'].get('init_scale_mode') == '3nn' and cfg['training']['seed'] == 42
            and all(cfg['data'][c]['mask_path'] == str(MASK) for c in ('plain', 'contrast')),
            'base config snapshot differs from the seed-42 base (3nn init, seed 42, official mask)')
    cfg['output']['output_dir'] = str(out / 'training')
    cfg['training']['seed'] = args.seed      # the seed-7 base config differs from seed 42's only in seed and output
    cfg['training']['max_iterations'] = args.iters
    if args.iters != 4000:  # smoke only: the trainer requires warmup < iters / 2 (the real run keeps 500)
        cfg['training']['shared_warmup_iters'] = max(1, args.iters // 4)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    set_seed(args.seed, deterministic=False)
    trainer = jt.JointTrainer(cfg, device=torch.device('cuda', 0))
    require(record.get('mode') == args.width_mode, 'the wrapped scales_3nn was not used by the trainer')
    t0 = time.time()
    trainer.train()
    final = out / 'training' / EXP / 'checkpoints/latest.pth'
    ck = torch.load(final, map_location='cpu', weights_only=False)
    require(ck.get('iter') == args.iters, f'final checkpoint at iter {ck.get("iter")}, expected {args.iters}')
    s = torch.exp(ck['model_state']['scales'].double())
    write_json(out / 'metrics.json', dict(stage='base', iters=args.iters, seconds=time.time() - t0, init=record,
                                          final_checkpoint=str(final), final_sha256=sha256(final),
                                          config_source=str(BASE_CFG), driver_sha256=sha256(__file__),
                                          final_median_normalised_scale=[float(s[:, i].median()) for i in range(3)],
                                          peak_allocated_gb=torch.cuda.max_memory_allocated() / 2 ** 30))
    log(f'base done in {time.time() - t0:.0f}s -> {final}')


def cmd_stage1(args):
    sc = load_module(STACK_CHAIN, '_kernelinit_stack_chain')
    sc.stage1(argparse.Namespace(ckpt=Path(args.ckpt), out=Path(args.out), steps=args.steps, no_export=True))


# ---------------------------------------------------------------- chain (driver; children on ONE GPU)
def cmd_chain(args):
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    require(inherited == str(args.gpu), f'launcher exposed CUDA_VISIBLE_DEVICES={inherited}, expected --gpu {args.gpu}')
    run_root = ROOT / args.name
    run_root.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if k not in SCRUB}
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONNOUSERSITE='1',
               PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1', CUDA_CACHE_DISABLE='1')
    me = [str(PY), '-X', 'utf8', '-B', '-u', str(Path(__file__).resolve())]
    iters, steps = (20, 2) if args.smoke else (4000, 500)
    write_json(run_root / 'spec.json', dict(name=args.name, width_mode=args.width_mode, seed=args.seed,
                                            width_scale=args.width_scale, gpu=args.gpu, smoke=args.smoke,
                                            source_digest=os.environ.get('MCGS_SOURCE_DIGEST'),
                                            driver_sha256=sha256(__file__), started=time.strftime('%Y-%m-%d %H:%M:%S')))

    def child(name, argv, cwd):
        t = time.time()
        log(f'[{name}] start')
        with open(run_root / f'{name}.log', 'x') as stream:
            code = subprocess.call([str(a) for a in argv], env=env, stdout=stream, stderr=subprocess.STDOUT, cwd=str(cwd))
        log(f'[{name}] exit {code} after {time.time() - t:.0f}s (log {run_root}/{name}.log)')
        require(code == 0, f'{name} failed')

    child('base', me + ['base', '--out', run_root / 'base', '--iters', iters, '--width-mode', args.width_mode,
                        '--seed', args.seed, '--width-scale', args.width_scale],
          cwd=Path(__file__).resolve().parent)
    base_ckpt = run_root / 'base/training' / EXP / 'checkpoints/latest.pth'
    if args.smoke:      # stage1 needs an iteration-4000 checkpoint; plumbing of the init and base is what is checked
        log('SMOKE OK (base only)')
        return
    child('stage1', me + ['stage1', '--ckpt', base_ckpt, '--out', run_root / 'stage1', '--steps', steps], cwd=DENSIFY)
    final = run_root / 'stage1/training' / EXP / 'checkpoints/latest.pth'
    require(final.is_file(), 'stage1 checkpoint missing')
    write_json(run_root / 'done.json', dict(stage1_checkpoint=str(final), sha256=sha256(final),
                                            finished=time.strftime('%Y-%m-%d %H:%M:%S')))
    log('CHAIN DONE', final)


def cmd_check_init(args):
    """CPU: the wrapper on a synthetic jittered-grid cloud (positions in [-1, 1], frozen scales_3nn)."""
    import torch
    sys.path[:0] = [str(BASE_TRAINER / 'paired_gs')]
    import gaussian.init_cloud as init_cloud
    g = torch.Generator().manual_seed(0)
    vox = torch.stack(torch.meshgrid(*(torch.arange(0, n, 4) for n in SHAPE), indexing='ij'), -1).reshape(-1, 3).float()
    pos = (vox + 0.5) / torch.tensor(SHAPE) * 2 - 1 + 1e-4 * torch.randn(vox.shape, generator=g)
    for mode in MODES:
        rec = {}
        wrapped = __import__('types').ModuleType('w')
        wrapped.scales_3nn = init_cloud.scales_3nn
        install_width(wrapped, mode, rec)
        out = wrapped.scales_3nn(pos, SHAPE)
        vs = rec['median_voxel_std']
        if mode == 'voxel_iso':
            require(max(vs) - min(vs) < 1e-3 * max(vs), f'voxel_iso widths are not equal: {vs}')
            require(abs(vs[2] - rec['median_voxel_std_original'][2]) < 1e-6, 'axis 2 width must be unchanged')
        else:
            require(torch.equal(out, init_cloud.scales_3nn(pos, SHAPE)), 'normalised mode must be the frozen init')
    log('CHECK-INIT OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('base'); b.add_argument('--out', required=True); b.add_argument('--iters', type=int, default=4000)
    b.add_argument('--width-mode', choices=MODES, required=True)
    b.add_argument('--seed', type=int, default=42); b.add_argument('--width-scale', type=float, default=1.0)
    s = sub.add_parser('stage1'); s.add_argument('--ckpt', required=True); s.add_argument('--out', required=True)
    s.add_argument('--steps', type=int, default=500)
    c = sub.add_parser('chain'); c.add_argument('--gpu', type=int, required=True); c.add_argument('--name', required=True)
    c.add_argument('--width-mode', choices=MODES, required=True); c.add_argument('--smoke', action='store_true')
    c.add_argument('--seed', type=int, default=42); c.add_argument('--width-scale', type=float, default=1.0)
    sub.add_parser('check-init')
    args = ap.parse_args()
    {'base': cmd_base, 'stage1': cmd_stage1, 'chain': cmd_chain, 'check-init': cmd_check_init}[args.cmd](args)


if __name__ == '__main__':
    main()
