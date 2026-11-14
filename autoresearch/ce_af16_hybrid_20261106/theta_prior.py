#!/usr/bin/env python3
"""theta_prior.py -- leak-free MC-GS prior for mcgs_zsref.py (CE AF16, 2026-11-07).

Why: full run 1 of mcgs_zsref.py (README.md here) started from FV37, an MC-GS fit to ALL acquired
k-space. Its Lambda (loss) and Gamma (validation) samples were therefore already seen by the prior,
so the self-supervised loss fell by pulling the measured low frequencies toward the data instead of
learning the unacquired high frequencies. Here the MC-GS prior is refitted on the Theta split only.

What: one fixed split of the acquired AF16 mask (constant along readout), identical to the step-0
split of mcgs_zsref.py: Gamma = 10 % (seed 0), then Theta / Lambda = 80 / 20 % of the rest (seed 1234),
i.e. 72 / 18 / 10 % of the acquired samples. The FV37 seed-42 chain is replayed unchanged on Theta:
  base    e778bf2_ar JointTrainer, config snapshot of ar_i1_k30p90, 0 -> 4000 (as arc/cycle.py trains it)
  stage1  densify stack_chain.stage1 (G2 recipe, AR trainer), 4000 -> 4500, no export
  stage2  losstune long_chain.stage2 with the FV37 recipe, 4500 -> 5125, relax alpha 1
  zsref   mcgs_zsref.py --split-file (Theta for DC, Lambda for loss, Gamma for early stop), prior = stage2
Only the mask path changes. Frozen code is imported read-only; the two chain modules read the G2
config through their module global G2, which is pointed at ROOT/g2_theta (a copy of G2's
actual_config.yaml with mask_path -> Theta and G2's metrics.json, the frozen gauges). Normalisation
in the simulated pipeline uses the fully sampled image (mask independent, as in every chain run),
and stage 1 still requires it to equal the frozen G2 gauge. After every stage the mask the trainer
actually used is compared with Theta. No GT / ROI is read by this file.

Commands: split, base, stage1, stage2, chain (sequential children on ONE launcher-exposed GPU).
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
BASE_CKPT4000 = BASE_TRAINER / 'results_arc/ar_i1_k30p90/ar_i1_k30p90/checkpoints/latest.pth'
BASE_SO = BASE_TRAINER / 'paired_gs/gaussian/cuda_ext/build/tile_voxelize_cuda_0141c525_7aedd3b0.so'
BASE_SO_SHA = '3552766d6cec6b7239d1b6dda1e7c36003aa60ab6644548b232271a4853d55dc'
ARTS = Path('/data0/congcong/code/GS/3DGS/outputs/codex_artifacts')
DENSIFY = ARTS / 'mcgs_ce_densify_20261029'
STACK_CHAIN = DENSIFY / 'stack_chain.py'
I1_STAGE1_CKPT = DENSIFY / 'stack/i1_ctrl/stage1/training/ar_i1_k30p90/checkpoints/latest.pth'
WS = ARTS / 'mcgs_ce_losstune_20261101'
LONG_CHAIN = WS / 'long_chain.py'
FV37_RECIPE = WS / 'fixed_views_20261104/recipes/FV37_lambda_wav_0.json'
FV37_S42 = WS / 'runs/FV37_lambda_wav_0/s42/m0625/relax_a1'
G2 = ARTS / 'mcgs_ce_autoresearch_20261017/frequency_adam/outputs/adamflat_g2_v1'
EXP = 'ar_i1_k30p90'
HERE = Path(__file__).resolve().parent
SCRUB = ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'CUDA_LAUNCH_BLOCKING', 'PYTHONSTARTUP', 'PYTHONPATH',
         'PYTHONHOME', 'OMP_NUM_THREADS')


def log(*a):
    print(*a, flush=True)


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


def theta3d(root):
    import numpy as np
    with np.load(Path(root) / 'split.npz') as z:
        theta = z['theta']
    return np.broadcast_to(theta[None], (256,) + theta.shape)


def check_used_mask(run_dir, root):
    """The frozen frequency-weight installer saves the acquired mask the trainer actually holds."""
    import numpy as np
    want = theta3d(root)
    for scan in ('pre', 'post'):
        got = np.load(Path(run_dir) / 'frequency_weights' / f'{scan}_acquired_mask.npy') > 0
        require(got.shape == want.shape and np.array_equal(got, want), f'{scan}: trainer mask is not Theta in {run_dir}')
    log(f'  mask check: {run_dir.name} trained on Theta ({want[0].sum()} of 232x176 phase points)')


# ---------------------------------------------------------------- split (CPU)
def cmd_split(args):
    import h5py
    import numpy as np
    import torch
    import yaml
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    require(not (root / 'split.npz').exists(), f'{root}/split.npz exists; never overwrite a split')
    with h5py.File(MASK, 'r') as f:
        m3 = np.asarray(f['mask'][:])
    require(m3.dtype == np.uint8 and m3.shape == (256, 232, 176) and (m3 == m3[0:1]).all(), 'unexpected official mask')
    m2 = torch.from_numpy(m3[0] > 0)
    idx = torch.nonzero(m2, as_tuple=False); na = idx.shape[0]          # same order and seeds as mcgs_zsref.py
    perm = torch.randperm(na, generator=torch.Generator(device='cpu').manual_seed(0))
    n_val = int(0.10 * na); val_idx, pool = perm[:n_val], perm[n_val:]
    p = torch.randperm(pool.numel(), generator=torch.Generator(device='cpu').manual_seed(1234))
    cut = int(0.80 * p.numel())

    def mk(sel):
        mm = torch.zeros_like(m2); mm[idx[sel, 0], idx[sel, 1]] = True; return mm.numpy()

    theta, lam, gamma = mk(pool[p[:cut]]), mk(pool[p[cut:]]), mk(val_idx)
    require(not (theta & lam).any() and not (theta & gamma).any() and not (lam & gamma).any()
            and np.array_equal(theta | lam | gamma, m3[0] > 0), 'split is not a partition of the acquired mask')
    np.savez(root / 'split.npz', theta=theta, lam=lam, gamma=gamma)
    h5 = root / 'mask_theta_3d.h5'
    with h5py.File(h5, 'w') as f:
        f.create_dataset('mask', data=np.broadcast_to(theta[None].astype(np.uint8), m3.shape).copy())
    g2 = root / 'g2_theta'
    g2.mkdir()
    config = yaml.safe_load((G2 / 'actual_config.yaml').read_text())
    for channel in ('plain', 'contrast'):
        require(config['data'][channel]['mask_path'] == str(MASK), 'G2 config mask differs from the official mask')
        config['data'][channel]['mask_path'] = str(h5)
    (g2 / 'actual_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (g2 / 'metrics.json').write_bytes((G2 / 'metrics.json').read_bytes())
    rec = dict(official_mask=str(MASK), official_mask_sha256=sha256(MASK), acquired=int(na),
               theta=int(theta.sum()), lam=int(lam.sum()), gamma=int(gamma.sum()),
               theta_acceleration=float(theta.size / theta.sum()), seeds=dict(gamma=0, theta_lambda=1234),
               split_sha256=sha256(root / 'split.npz'), theta_mask_sha256=sha256(h5),
               g2_source=str(G2), note='Theta/Lambda/Gamma = 72/18/10 % of acquired samples; no GT read')
    write_json(root / 'split.json', rec)
    log('split', json.dumps(rec))


# ---------------------------------------------------------------- base 0 -> 4000 (GPU child)
def cmd_base(args):
    import numpy as np
    root, out = Path(args.root), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    require(Path(sys.executable).resolve() == PY.resolve(), 'numeric stages require the GSTMI interpreter')
    sys.dont_write_bytecode = True
    for key, rel in (('TMPDIR', 'tmp'), ('TORCH_EXTENSIONS_DIR', 'cache/torch'), ('CUDA_CACHE_PATH', 'cache/cuda'),
                     ('MPLCONFIGDIR', 'cache/matplotlib'), ('XDG_CACHE_HOME', 'cache/xdg')):
        (out / rel).mkdir(parents=True, exist_ok=True); os.environ[key] = str(out / rel)
    import torch
    import yaml
    require(torch.cuda.device_count() == 1, 'exactly one visible GPU is required')
    hold = torch.zeros(1, device='cuda:0')  # noqa: F841  (hold the context, as arc/cycle.py does)
    os.chdir(BASE_TRAINER)
    sys.path[:0] = [str(BASE_TRAINER / 'paired_gs'), str(BASE_TRAINER)]
    require(sha256(BASE_SO) == BASE_SO_SHA, 'prebuilt base binary changed; compilation is forbidden')
    import gaussian.tile_voxelizer as tile
    require(Path(tile.__file__).resolve() == (BASE_TRAINER / 'paired_gs/gaussian/tile_voxelizer.py').resolve(),
            'foreign tile voxelizer module')
    tile._cuda_ext = load_module(BASE_SO, BASE_SO.stem)
    tile._cuda_load_error = None
    from seed import set_seed
    import trainers.joint_trainer as jt
    require(Path(jt.__file__).resolve() == (BASE_TRAINER / 'paired_gs/trainers/joint_trainer.py').resolve(),
            'foreign joint trainer module')
    cfg = yaml.safe_load(BASE_CFG.read_text())
    for channel in ('plain', 'contrast'):
        require(cfg['data'][channel]['mask_path'] == str(MASK), 'base config mask differs from the official mask')
        cfg['data'][channel]['mask_path'] = str(root / 'mask_theta_3d.h5')
    cfg['output']['output_dir'] = str(out / 'training')
    cfg['training']['max_iterations'] = args.iters
    if args.iters != 4000:  # smoke only: the trainer requires warmup < iters / 2 (the real run keeps 500)
        cfg['training']['shared_warmup_iters'] = max(1, args.iters // 4)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    set_seed(42, deterministic=False)
    trainer = jt.JointTrainer(cfg, device=torch.device('cuda', 0))
    want = torch.from_numpy(np.ascontiguousarray(theta3d(root))).float()
    gauges = json.loads((G2 / 'metrics.json').read_text())['audit_checks']
    for scan, tag in (('pre', 'p'), ('post', 'c')):
        mask = getattr(trainer, 'mask_' + tag).detach().float().cpu()
        mask = mask if mask.ndim == 3 else mask[0]
        require(torch.equal(mask > 0, want > 0), f'{scan}: base trainer mask is not Theta')
        scale = float(getattr(trainer, 'ds_' + tag).normalization_scale)
        require(scale == gauges[scan]['scale_trainer'], f'{scan}: normalisation {scale} differs from the frozen gauge')
    log(f'  base: Theta mask and frozen gauges confirmed; training {args.iters} iterations')
    t0 = time.time()
    trainer.train()
    final = out / 'training' / EXP / 'checkpoints/latest.pth'
    ck = torch.load(final, map_location='cpu', weights_only=False)
    require(ck.get('iter') == args.iters, f'final checkpoint at iter {ck.get("iter")}, expected {args.iters}')
    write_json(out / 'metrics.json', dict(stage='base', iters=args.iters, seconds=time.time() - t0,
                                          final_checkpoint=str(final), config_source=str(BASE_CFG),
                                          peak_allocated_gb=torch.cuda.max_memory_allocated() / 2 ** 30))
    log(f'  base done in {time.time()-t0:.0f}s -> {final}')


# ---------------------------------------------------------------- stage1 / stage2 (GPU children)
def cmd_stage1(args):
    sc = load_module(STACK_CHAIN, '_theta_stack_chain')
    sc.G2 = Path(args.root) / 'g2_theta'
    sc.stage1(argparse.Namespace(ckpt=Path(args.ckpt), out=Path(args.out), steps=args.steps, no_export=True))
    check_used_mask(Path(args.out).resolve(), args.root)


def cmd_stage2(args):
    lc = load_module(LONG_CHAIN, '_theta_long_chain')
    lc.G2 = Path(args.root) / 'g2_theta'
    lc.stage2(argparse.Namespace(ckpt=Path(args.ckpt), out=Path(args.out), recipe=Path(args.recipe)))
    check_used_mask(Path(args.out).resolve(), args.root)


# ---------------------------------------------------------------- chain (driver, no GPU of its own)
def cmd_chain(args):
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    require(inherited == str(args.gpu), f'launcher exposed CUDA_VISIBLE_DEVICES={inherited}, expected --gpu {args.gpu}')
    root = Path(args.root)
    if not (root / 'split.json').exists():
        cmd_split(args)
    env = {k: v for k, v in os.environ.items() if k not in SCRUB}
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONNOUSERSITE='1',
               PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1', CUDA_CACHE_DISABLE='1')
    me = [str(PY), '-X', 'utf8', '-B', '-u', str(Path(__file__).resolve())]
    run_root = root / ('smoke' if args.smoke else 'chain')
    run_root.mkdir(parents=True, exist_ok=True)

    def child(name, argv, omp=None, cwd=None):
        out = run_root / name
        if (out / 'metrics.json').exists():
            log(f'[{name}] already complete, reused'); return
        argv = [str(a) for a in argv]
        e = dict(env)
        if omp:
            e['OMP_NUM_THREADS'] = omp
        t = time.time()
        log(f'[{name}] start: {" ".join(map(str, argv))}')
        with open(run_root / f'{name}.log', 'a') as stream:
            code = subprocess.call(argv, env=e, stdout=stream, stderr=subprocess.STDOUT, cwd=str(cwd or HERE))
        log(f'[{name}] exit {code} after {time.time()-t:.0f}s (log {run_root}/{name}.log)')
        require(code == 0, f'{name} failed')

    if args.smoke:          # plumbing only: tiny budgets from the existing keeper checkpoints
        recipe = json.loads(FV37_RECIPE.read_text()); recipe.update(steps=2, milestones=[2])
        (run_root / 'recipe_smoke.json').write_text(json.dumps(recipe, indent=2))
        child('base', me + ['base', '--root', root, '--out', run_root / 'base', '--iters', 20])
        child('stage1', me + ['stage1', '--root', root, '--ckpt', BASE_CKPT4000, '--out', run_root / 'stage1', '--steps', 2],
              cwd=DENSIFY)
        child('stage2', me + ['stage2', '--root', root, '--ckpt', I1_STAGE1_CKPT, '--out', run_root / 'stage2',
                              '--recipe', run_root / 'recipe_smoke.json'], omp='1', cwd=WS)
        e = dict(env); e['ZSREF_SMOKE_EPOCHS'] = '1'
        argv = [str(PY), '-u', str(HERE / 'mcgs_zsref.py'), '--device', 'cuda', '--gpu', str(args.gpu), '--smoke',
                '--init-dir', str(FV37_S42), '--split-file', str(root / 'split.npz'),
                '--infer-init-dir', str(FV37_S42), '--out', str(run_root / 'zsref')]
        with open(run_root / 'zsref.log', 'a') as stream:
            code = subprocess.call(argv, env=e, stdout=stream, stderr=subprocess.STDOUT, cwd=str(HERE))
        log(f'[zsref smoke] exit {code} (log {run_root}/zsref.log)')
        require(code == 0, 'zsref smoke failed')
        log('SMOKE OK'); return
    child('base', me + ['base', '--root', root, '--out', run_root / 'base', '--iters', 4000])
    child('stage1', me + ['stage1', '--root', root, '--ckpt', run_root / 'base/training' / EXP / 'checkpoints/latest.pth',
                          '--out', run_root / 'stage1', '--steps', 500], cwd=DENSIFY)
    child('stage2', me + ['stage2', '--root', root, '--ckpt', run_root / 'stage1/training' / EXP / 'checkpoints/latest.pth',
                          '--out', run_root / 'stage2', '--recipe', FV37_RECIPE], omp='1', cwd=WS)
    prior = run_root / 'stage2/m0625/relax_a1'
    require((prior / 'candidate_pre_complex.npy').exists(), 'stage2 prior export missing')
    if args.no_zsref:
        log('PRIOR DONE', prior); return
    zs = run_root / 'zsref'
    if not (zs / 'fit_log.json').exists():
        e = dict(env); e.pop('OMP_NUM_THREADS', None)
        argv = [str(PY), '-u', str(HERE / 'mcgs_zsref.py'), '--device', 'cuda', '--gpu', str(args.gpu),
                '--init-dir', str(prior), '--split-file', str(root / 'split.npz'),
                '--infer-init-dir', str(FV37_S42), '--out', str(zs)]
        log(f'[zsref] start: {" ".join(argv)}')
        with open(run_root / 'zsref.log', 'a') as stream:
            code = subprocess.call(argv, env=e, stdout=stream, stderr=subprocess.STDOUT, cwd=str(HERE))
        log(f'[zsref] exit {code} (log {run_root}/zsref.log)')
        require(code == 0, 'zsref failed')
    log('CHAIN DONE', zs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('split'); s.add_argument('--root', required=True)
    b = sub.add_parser('base'); b.add_argument('--root', required=True); b.add_argument('--out', required=True)
    b.add_argument('--iters', type=int, default=4000)
    for name in ('stage1', 'stage2'):
        p = sub.add_parser(name); p.add_argument('--root', required=True)
        p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
        if name == 'stage1':
            p.add_argument('--steps', type=int, default=500)
        else:
            p.add_argument('--recipe', required=True)
    c = sub.add_parser('chain'); c.add_argument('--root', required=True); c.add_argument('--gpu', type=int, required=True)
    c.add_argument('--smoke', action='store_true'); c.add_argument('--no-zsref', action='store_true')
    args = ap.parse_args()
    {'split': cmd_split, 'base': cmd_base, 'stage1': cmd_stage1, 'stage2': cmd_stage2, 'chain': cmd_chain}[args.cmd](args)


if __name__ == '__main__':
    main()
