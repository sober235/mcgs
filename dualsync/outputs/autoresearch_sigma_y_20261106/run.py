"""Run the Tuned sampler with exactly one of sigma_y / eta changed (one GPU, deterministic).

Usage (server, from repo root):
  python outputs/autoresearch_sigma_y_20261106/run.py --tag sy0.5 --sigma-y 0.5 \
      --cases tune --gpu 4

The module is stdlib-only until CUDA visibility is bound. Every case is reserved
in ledger.jsonl before inference and counts against protocol max_runs even if it
fails. Existing outputs are never overwritten and nothing outside this directory
is written.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE / 'protocol.json'
LEDGER = HERE / 'ledger.jsonl'
TAG_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,40}')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require_isolated_python():
    """~/.local on the server holds a broken newer torch that shadows the env."""
    if not sys.flags.no_user_site:
        raise RuntimeError('Run with PYTHONNOUSERSITE=1 (or python -s) so ~/.local is ignored')


def load_protocols(protocol_path=PROTOCOL):
    p = json.loads(Path(protocol_path).read_text())
    if digest(p['base_protocol']) != p['base_protocol_sha256']:
        raise ValueError('Base protocol changed')
    return p, json.loads(Path(p['base_protocol']).read_text())


def make_params(base_params, variable, value, protocol):
    """Copy accepted params, changing exactly one approved variable."""
    spec = protocol['variables'].get(variable)
    if spec is None:
        raise ValueError('Unapproved variable: ' + str(variable))
    lo, hi = spec['range']
    value = float(value)
    if not lo <= value <= hi:
        raise ValueError(variable + ' outside approved range')
    if base_params['sigma_y'] != protocol['variables']['sigma_y']['accepted'] or 'eta' in base_params:
        raise ValueError('Base params are not the accepted Tuned params')
    params = dict(base_params)
    if variable == 'eta':
        if value != spec['accepted']:
            params['eta'] = value  # only eta runs carry the key; eta 1 == original sampler
    else:
        params[variable] = value
    return params


def run_variable(params, protocol):
    """(variable, value) actually changed by a params dict; null -> sigma_y accepted."""
    if 'eta' in params:
        return 'eta', float(params['eta'])
    return 'sigma_y', float(params['sigma_y'])


def resolve_cases(spec, protocol, base):
    if spec in ('tune', 'confirmation'):
        cases = list(protocol[spec + '_cases'])
    elif spec == 'null':
        cases = [protocol['null_case']]
    else:
        cases = [c.strip() for c in spec.split(',') if c.strip()]
    if not cases or len(cases) != len(set(cases)) or any(c not in base['cases'] for c in cases):
        raise ValueError('Invalid case list')
    return cases


def read_ledger(path=LEDGER):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_ledger(entry, path=LEDGER):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(entry, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def check_budget(n_new, protocol, ledger):
    reserved = sum(1 for e in ledger if e['event'] == 'reserve')
    if reserved + n_new > protocol['max_runs']:
        raise RuntimeError('GPU run budget exceeded: %d reserved + %d new > %d'
                           % (reserved, n_new, protocol['max_runs']))
    return reserved


def check_no_repeat(tag, variable, value, cases, ledger):
    for e in ledger:
        if e['event'] != 'reserve' or e['case'] not in cases:
            continue
        e_var = e.get('variable', 'sigma_y')
        e_val = float(e.get('value', e.get('sigma_y')))
        if e['tag'] == tag or (e_var == variable and e_val == float(value)):
            raise RuntimeError('Already attempted %s for tag/setting; no retries' % e['case'])


def bind_visibility(gpu, environ=None):
    env = os.environ if environ is None else environ
    gpu = str(gpu)
    if not re.fullmatch(r'\d+', gpu):
        raise ValueError('Exactly one physical GPU index required')
    if 'CUDA_VISIBLE_DEVICES' in env and env['CUDA_VISIBLE_DEVICES'] != gpu:
        raise ValueError('CUDA allocation conflict')
    if environ is None and 'torch' in sys.modules:
        raise RuntimeError('Bind CUDA visibility before importing torch')
    env['CUDA_VISIBLE_DEVICES'] = gpu


def verify_sources(protocol, base):
    for rel, sha in protocol['source_sha256'].items():
        if digest(ROOT / rel) != sha:
            raise ValueError('Frozen source changed: ' + rel)
    inventory = [f.relative_to(base['data_dir']).as_posix()
                 for f in sorted(Path(base['data_dir']).rglob('*.mat'))]
    if inventory != base['input_inventory']:
        raise ValueError('Global input inventory differs')
    cp = base['checkpoint']
    stat = Path(cp['path']).stat()
    if stat.st_size != cp['size'] or stat.st_mtime_ns != cp['mtime_ns'] or digest(cp['path']) != cp['sha256']:
        raise ValueError('Checkpoint differs')


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in value.items()})
    return value


def run(tag, variable, value, cases, gpu):
    require_isolated_python()
    protocol, base = load_protocols()
    params = make_params(base['params'], variable, value, protocol)
    sampler_sha = None
    if 'eta' in params:
        sampler_sha = digest(HERE / 'eta_sampler.py')
        if sampler_sha != protocol['variables']['eta']['sampler_sha256']:
            raise ValueError('eta sampler changed')
    out_dir = HERE / 'runs' / tag
    for case in cases:
        if (out_dir / (case + '.mat')).exists():
            raise RuntimeError('Refusing to overwrite ' + case)
    ledger = read_ledger()
    check_no_repeat(tag, variable, params.get(variable, value), cases, ledger)
    check_budget(len(cases), protocol, ledger)
    bind_visibility(gpu)
    verify_sources(protocol, base)
    for case in cases:
        if digest(Path(base['data_dir']) / (case + '.mat')) != base['input_sha256'][case + '.mat']:
            raise ValueError('Input changed: ' + case)

    sys.path.insert(0, str(ROOT))
    import numpy as np
    import scipy.io as sio
    import sigpy as sp
    import torch
    import yaml
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU required')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from runners.diffusion import Diffusion, get_gaussian_mask
    from runners.diffusion_gate import load_ksp
    if 'eta' in params:
        sys.path.insert(0, str(HERE))
        import eta_sampler
        generalized_steps_tianyuan = eta_sampler.load().generalized_steps_tianyuan
    else:
        from functions.denoising_tianyuan import generalized_steps_tianyuan
    from utils import c2r, r2c, cgSENSE, Abs

    seed = protocol['base_seed']
    cp = base['checkpoint']
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    cfg = yaml.safe_load((ROOT / 'configs/fastMRI_tianyuan.yml').read_text())
    cfg['gpu_id'] = str(gpu)
    cfg['data']['image_size'] = base['image_size']
    cfg['tianyuan'] = dict(params, data_dir=base['data_dir'], output_dir=str(out_dir))
    cfg['sampling']['weight'] = str(Path(cp['path']).parent)
    cfg['sampling']['ckpt_id'] = int(re.fullmatch(r'ckpt_(\d+)\.pth', Path(cp['path']).name).group(1))
    config = _namespace(cfg)
    config.device = torch.device('cuda:0')
    args = SimpleNamespace(use_pretrained=False, fid=True, interpolation=False, sequence=False,
                           seed=seed, timesteps=base['timesteps'])
    done = []

    class SigmaRunner(Diffusion):
        def sample_fid(self, model):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            gmask = get_gaussian_mask(
                beta_schedule=config.diffusion.beta_schedule, beta_start=config.diffusion.beta_start,
                beta_end=config.diffusion.beta_end,
                num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
                Fourier=config.diffusion.Fourier, image_size=config.data.image_size)
            seq = range(0, self.num_timesteps, self.num_timesteps // args.timesteps)
            for case in cases:
                index = base['case_global_index'][case]
                append_ledger(dict(event='reserve', tag=tag, variable=variable, value=float(value),
                                   sigma_y=params['sigma_y'], sampler_sha256=sampler_sha, case=case,
                                   gpu=str(gpu), utc=time.time()))
                start = time.time()
                try:
                    ksp = load_ksp(str(Path(base['data_dir']) / (case + '.mat')))
                    orig_shape = ksp.shape
                    ksp = sp.resize(np.expand_dims(ksp, 0), (1, config.data.image_size, config.data.image_size))
                    k0 = c2r(torch.from_numpy(np.expand_dims(ksp, 0))).type(torch.FloatTensor).to(config.device)
                    mask = torch.where(r2c(k0) != 0, torch.ones_like(r2c(k0)), torch.zeros_like(r2c(k0)))
                    atb = r2c(k0) * mask
                    csm = torch.ones_like(atb)
                    _, cg_sense = cgSENSE(atb, atb, csm, mask, 10, 1e-5)
                    coeff = torch.max(Abs(c2r(cg_sense)))
                    cg_sense = cg_sense / coeff
                    atb = atb / coeff
                    x_init = c2r(cg_sense)
                    gen = torch.Generator(device=x_init.device).manual_seed(seed + index + 7)
                    jac_noise = torch.randn(x_init.shape, device=x_init.device, generator=gen)
                    torch.manual_seed(seed + index); np.random.seed(seed + index)
                    torch.cuda.reset_peak_memory_stats()
                    trace = {}
                    _, x0_preds = generalized_steps_tianyuan(
                        x_init, atb, csm, mask, seq, model, gmask, self.betas, self.sigmas, self.psigmas,
                        self.type, cfg=params, jac_noise=jac_noise, trace=trace)
                    fd = np.array([[r['step'], r['relative_error']] for r in trace.pop('fd')])
                    payload = dict(
                        cg_sense=cg_sense.squeeze().cpu().numpy(), diff=r2c(x0_preds[-1]).squeeze().cpu().numpy(),
                        coeff=float(coeff), orig_shape=np.array(orig_shape), fd=fd,
                        trace={k: np.array(v) for k, v in trace.items()}, seed=seed + index, base_seed=seed,
                        global_index=index, jac_seed=seed + index + 7,
                        params_json=json.dumps(params, sort_keys=True), sampler_sha256=str(sampler_sha),
                        input_sha256=base['input_sha256'][case + '.mat'], checkpoint_sha256=cp['sha256'])
                    dest = out_dir / (case + '.mat')
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_name(dest.name + '.' + uuid.uuid4().hex + '.tmp')
                    try:
                        sio.savemat(str(tmp), payload, appendmat=False)
                        os.link(tmp, dest)  # atomic publish, refuses overwrite
                    finally:
                        if tmp.exists():
                            tmp.unlink()
                except Exception as error:
                    append_ledger(dict(event='failed', tag=tag, case=case, error=repr(error), utc=time.time()))
                    raise
                append_ledger(dict(event='complete', tag=tag, case=case, sha256=digest(dest),
                                   seconds=round(time.time() - start, 1),
                                   peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                   utc=time.time()))
                done.append(str(dest))

    SigmaRunner(args, config, device=config.device).sample()
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sigma-y', type=float)
    group.add_argument('--eta', type=float)
    parser.add_argument('--cases', required=True, help='tune | confirmation | null | comma list')
    parser.add_argument('--gpu', required=True)
    a = parser.parse_args(argv)
    if not TAG_RE.fullmatch(a.tag):
        parser.error('bad tag')
    protocol, base = load_protocols()
    cases = resolve_cases(a.cases, protocol, base)
    variable, value = ('eta', a.eta) if a.eta is not None else ('sigma_y', a.sigma_y)
    for path in run(a.tag, variable, value, cases, a.gpu):
        print('wrote', path)


if __name__ == '__main__':
    main()
