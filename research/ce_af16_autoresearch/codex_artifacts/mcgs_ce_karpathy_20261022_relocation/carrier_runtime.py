"""External carrier state composed with the preserved original continuation.

The public entry requires a verified CUDA artifact. Only the private tiny CPU
fixture entry accepts a native-boundary oracle; no automatic fallback exists.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from types import FunctionType

_ROOT = Path(__file__).resolve().parent
_SCHEMA = 'carrier_continuation_v1'
_CHANNELS = {'plain': 'pre', 'contrast': 'post'}


def _contract():
    return json.loads((_ROOT / 'carrier_mri_contract.json').read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _descriptor(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=_sha(path))


def _tensor(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                finite=bool(torch.isfinite(value).all()), zero=bool((value == 0).all()))


def _equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if type(a) is not type(b): return False
    if isinstance(a, dict): return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _cpu(value):
    import torch
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list): return [_cpu(v) for v in value]
    if isinstance(value, tuple): return tuple(_cpu(v) for v in value)
    return value


def _optimizer(parameters, policy):
    import torch
    return torch.optim.Adam(parameters, lr=policy['lr'], betas=tuple(policy['betas']),
                            eps=policy['eps'], weight_decay=policy['weight_decay'], amsgrad=policy['amsgrad'])


def _state(eta, optimizer, policy):
    import torch
    clocks = {}; moments = {}; frequencies = {}
    for name, parameter in eta.items():
        if not bool(torch.isfinite(parameter).all()): raise ValueError('nonfinite carrier eta')
        state = optimizer.state.get(parameter, {})
        clocks[name] = int(state['step'].item()) if state else 0
        moments[name] = {k: _tensor(v) for k, v in state.items()}
        f = policy['frequency_bound'] * parameter.detach().tanh()
        if not bool(torch.isfinite(f).all()) or not bool((f.abs() <= policy['frequency_bound']).all()):
            raise ValueError('nonfinite or out-of-bounds carrier frequency')
        frequencies[name] = dict(min=float(f.min()), max=float(f.max()), max_abs=float(f.abs().max()))
    return dict(eta={n: dict(_tensor(p), tensor_id=id(p), device=str(p.device)) for n, p in eta.items()},
                adam_steps=clocks, moments=moments, frequency=frequencies,
                independent_parameters=len({p.data_ptr() for p in eta.values()}) == 2,
                optimizer_options=_cpu(optimizer.state_dict()['param_groups']))


def _pair_upper_bound(positions, scales, rotations, shape, *, limit):
    """O(K) sphere bound of actual-M tile pairs, before any pair allocation."""
    import torch
    import carrier_operator as op
    with torch.no_grad():
        centers, m = op._geometry(positions, scales, rotations, shape)
        if not bool(torch.isfinite(m).all()): raise ValueError('nonfinite actual-M geometry')
        eigenvalues = torch.linalg.eigvalsh(m.double())
        reliable = (eigenvalues[:, 0] > 0) & (eigenvalues[:, 0] > eigenvalues[:, -1]*1e-8)
        radius = torch.zeros_like(centers, dtype=torch.float64)
        radius[reliable] = (3/eigenvalues[reliable, 0].sqrt()).unsqueeze(1) * centers.new_tensor(shape).double()/2
        reliable &= torch.isfinite(radius).all(-1)
        edge = centers.new_tensor(shape, dtype=torch.float64)-1
        low = torch.minimum((centers.double()-radius).floor().clamp_min(0), edge).long()//op._TILE
        high = torch.minimum((centers.double()+radius).ceil().clamp_min(0), edge).long()//op._TILE
        low[~reliable] = 0
        high[~reliable] = high.new_tensor(op._tile_shape(shape))-1
        counts = (high-low+1).prod(-1)
        total = int(counts.sum())
        if total > limit: raise ValueError(f'carrier pair resource limit exceeded: {total} > {limit}')
        return dict(upper_pairs=total, max_pairs_per_atom=int(counts.max()),
                    reliable_atoms=int(reliable.sum()), fallback_atoms=int((~reliable).sum()))


class _NativeObserver:
    """Record successful native returns, preserving the original render role."""
    def __init__(self, native, current, work):
        self.native, self.current, self.work = native, current, work
        self.rows = {}

    def forward(self, *args):
        observer = self.current()
        row = dict(role=observer.role, iteration=observer.iteration, channel=observer.channel, backwards=0)
        result = self.native.forward(*args)
        if row['channel'] not in _CHANNELS or row['role'] not in self.work['roles']:
            raise ValueError('unassigned carrier native forward')
        self.work['forwards'] += 1
        self.work['roles'][row['role']]['forwards'] += 1
        self.work['calls'].append(row)
        # The custom autograd context retains the exact center tensor object.
        self.rows[id(args[0])] = row
        return result

    def backward(self, *args):
        row = self.rows[id(args[1])]
        result = self.native.backward(*args)
        row['backwards'] += 1
        self.work['backwards'] += 1
        self.work['roles'][row['role']]['backwards'] += 1
        return result


@contextmanager
def _instance_bindings(trainer, render, train):
    saved = {name: (name in vars(trainer), vars(trainer).get(name)) for name in ('_render', 'train')}
    trainer._render, trainer.train = render, train
    try: yield
    finally:
        for name, (owned, value) in saved.items():
            if owned: setattr(trainer, name, value)
            else: delattr(trainer, name)


def run_carrier_continuation(trainer, frequency, directory, *, mode, steps, rotation_matrix_fn, requested_k=18000):
    """Run the fixed carrier route inside an externally verified artifact scope."""
    return _run_carrier_continuation(trainer, frequency, directory, mode=mode, steps=steps,
        rotation_matrix_fn=rotation_matrix_fn, requested_k=requested_k)


def _run_carrier_continuation(trainer, frequency, directory, *, mode, steps, rotation_matrix_fn,
                              requested_k=18000, _cpu_extension=None, _test_k=None):
    import torch
    import carrier_operator as operator
    import relocation_runner as original
    import carrier_artifact
    contract = _contract(); policy = contract['policy']['carrier']
    if type(requested_k) is not int or requested_k != policy['k']:
        raise ValueError('fixed subset requires requested_k=18000')
    if mode not in ('control', 'carrier') or type(steps) is not int or steps not in (0, 2, 500) or (steps == 0 and mode != 'control'):
        raise ValueError('invalid fixed carrier mode/budget')
    device = trainer.model.positions.device
    n = trainer.model.positions.shape[0]
    if _cpu_extension is None:
        if device.type != 'cuda': raise RuntimeError('public carrier requires verified CUDA artifact')
        extension = carrier_artifact.active_provenance()
        if operator._provenance != extension or operator._extension is None:
            raise RuntimeError('verified active carrier operator binding differs')
        if n != contract['policy']['point_count'] or _test_k is not None:
            raise ValueError('fixed original G2 point count/subset required')
        k = policy['k']; native = operator._extension
    else:
        if device.type != 'cpu' or not (type(_test_k) is int and 0 < _test_k <= n <= 32):
            raise ValueError('private CPU oracle requires explicit tiny fixture')
        k = _test_k; native = _cpu_extension
        extension = dict(module='private_cpu_test_oracle', binary_sha256=contract['artifact']['binary']['sha256'],
                         identity_sha256=contract['artifact']['identity_sha256'], test_only=True)
    directory = Path(directory)
    enabled = mode == 'carrier' or steps == 0
    rng_before = torch.random.get_rng_state().clone()
    ids_cpu = torch.randperm(n, generator=torch.Generator(device='cpu').manual_seed(policy['selection_seed']), device='cpu')[:k].sort().values
    if not torch.equal(rng_before, torch.random.get_rng_state()): raise ValueError('selection mutated caller RNG')
    selection_hash = _tensor(ids_cpu)['sha256']
    ids = ids_cpu.to(device)
    eta = {name: torch.nn.Parameter(torch.zeros(k, 3, dtype=torch.float32, device=device)) for name in policy['contrasts']}
    if any(id(p) == id(q) for p in eta.values() for q in trainer.model.parameters()):
        raise ValueError('carrier parameter overlaps original model')
    optimizer = _optimizer(list(eta.values()), policy)
    initial = _state(eta, optimizer, policy)
    selection_path = directory/'carrier_selection.pt'
    with selection_path.open('xb') as stream: torch.save(ids_cpu, stream)
    selection = dict(k=k, n=n, seed=policy['selection_seed'], method=policy['selection'], dtype='torch.int64',
                     sorted_unique=True, min=int(ids_cpu.min()), max=int(ids_cpu.max()), sha256=selection_hash,
                     rng_before_sha256=_tensor(rng_before)['sha256'], rng_after_sha256=_tensor(torch.random.get_rng_state())['sha256'],
                     file=_descriptor(selection_path))
    work = dict(forwards=0, backwards=0, optimizer_steps=0, scheduler_steps=0,
                roles={r: dict(forwards=0, backwards=0) for r in ('initialization', 'training', 'event', 'export')}, calls=[])
    bounds = dict(policy='actual_M_lambda_min_sphere_or_full_volume', limit=policy['pair_upper_limit'],
                  tile_size=policy['tile_size'], calls=[], max_upper_pairs=0, fallback_atoms_total=0)
    trace = []; observer = None; pending = None; export_before = None; zero_raw_parity = {}
    trace_path = directory/'carrier_trace.jsonl'; trace_path.touch(exist_ok=False)
    saved_render, saved_train = trainer._render, trainer.train
    class Observer(original.WorkObserver):
        def __init__(self, value):
            nonlocal observer
            super().__init__(value); observer = self
    private_globals = dict(original.run_continuation.__globals__, WorkObserver=Observer)
    continuation = FunctionType(original.run_continuation.__code__, private_globals,
                                original.run_continuation.__name__, original.run_continuation.__defaults__,
                                original.run_continuation.__closure__)
    continuation.__kwdefaults__ = original.run_continuation.__kwdefaults__
    proxy = _NativeObserver(native, lambda: observer, work)
    def render(channel):
        nonlocal export_before
        result = saved_render(channel)
        if not enabled: return result
        if observer.role == 'export' and export_before is None:
            export_before = _state(eta, optimizer, policy)
        p, s, q = (getattr(trainer.model, name)[ids] for name in ('positions', 'scales', 'rotations'))
        row = dict(role=observer.role, iteration=observer.iteration, channel=channel,
                   **_pair_upper_bound(p, s, q, trainer.volume_shape, limit=policy['pair_upper_limit']))
        bounds['calls'].append(row)
        bounds['max_upper_pairs'] = max(bounds['max_upper_pairs'], row['upper_pairs'])
        bounds['fallback_atoms_total'] += row['fallback_atoms']
        # A fresh complex node keeps the original voxelizer's density hook pure.
        density = trainer.model.get_density(channel)[ids]
        frequencies = policy['frequency_bound'] * eta[_CHANNELS[channel]].tanh()
        if _cpu_extension is None:
            correction = operator.carrier_correction(p, s, q, density, frequencies, trainer.volume_shape)
        else:
            values = tuple(x.contiguous() for x in (p, s, q, density, frequencies))
            shape = operator._validate(values, trainer.volume_shape)
            centers, m = operator._geometry(*values[:3], shape)
            ranges, pairs = operator._tile_pairs(centers, m, shape)
            correction = operator._Carrier.apply(centers.contiguous(), m.contiguous(), density.contiguous(),
                                                  frequencies.contiguous(), ranges, pairs, shape, proxy)
        combined = result + correction
        if steps == 0:
            numerator = torch.linalg.vector_norm((combined-result).to(torch.complex128))
            denominator = torch.linalg.vector_norm(result.to(torch.complex128)).clamp_min(1e-30)
            relative_l2 = float(numerator/denominator)
            exact_equal = torch.equal(combined, result)
            if not math.isfinite(relative_l2) or relative_l2 > 1e-5:
                raise ValueError('zero carrier raw export parity failed')
            zero_raw_parity[_CHANNELS[channel]] = dict(exact_equal=exact_equal, relative_l2=relative_l2)
        return combined
    def train(*args, step_callback=None, **kwargs):
        def callback(event, iteration, payload):
            nonlocal pending
            if event == 'start':
                step_callback(event, iteration, payload)
                if enabled: optimizer.zero_grad(set_to_none=True)
            elif event == 'pre_step':
                if enabled:
                    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in eta.values()):
                        raise ValueError('missing/nonfinite carrier gradient')
                    norms = {n: float(torch.linalg.vector_norm(p.grad.double())) for n, p in eta.items()}
                    pre = float(torch.nn.utils.clip_grad_norm_(list(eta.values()), policy['clip_norm'], error_if_nonfinite=True))
                    post = math.sqrt(sum(float(p.grad.double().square().sum()) for p in eta.values()))
                    pending = dict(iteration=iteration, step=iteration-contract['policy']['source_iteration'],
                                   gradient_norms=norms, pre_clip_norm=pre, post_clip_norm=post, finite=True,
                                   before=_state(eta, optimizer, policy))
                step_callback(event, iteration, payload)
            elif event == 'end':
                if enabled:
                    if pending is None: raise ValueError('missing carrier pre-step evidence')
                    if any(int(trainer.optimizer.state[p]['step']) != iteration for p in trainer.model.parameters()):
                        raise ValueError('carrier update preceded original Adam')
                    optimizer.step(); work['optimizer_steps'] += 1
                    state = _state(eta, optimizer, policy)
                    if set(state['adam_steps'].values()) != {work['optimizer_steps']}:
                        raise ValueError('actual carrier Adam clocks differ')
                    pending['after'] = state
                    pending['changed_parameters'] = {n: pending['before']['eta'][n]['sha256'] != state['eta'][n]['sha256'] for n in eta}
                    trace.append(pending)
                    with trace_path.open('a') as stream: stream.write(json.dumps(pending, allow_nan=False)+'\n')
                    pending = None
                step_callback(event, iteration, payload)
            else: raise ValueError('unknown original callback')
        return saved_train(*args, step_callback=callback, **kwargs)
    with operator._load_lock:
        previous_extension = operator._extension
        if _cpu_extension is None: operator._extension = proxy
    try:
        with _instance_bindings(trainer, render, train):
            result = continuation(trainer, frequency, directory, mode='control', steps=steps,
                                  rotation_matrix_fn=rotation_matrix_fn, requested_k=requested_k)
    finally:
        with operator._load_lock: operator._extension = previous_extension
    final = _state(eta, optimizer, policy)
    if enabled and export_before != final: raise ValueError('canonical export mutated external carrier state')
    expected_updates = steps if mode == 'carrier' else 0
    expected_forward = 2*steps+2 if mode == 'carrier' else (2 if steps == 0 else 0)
    expected_backward = 2*expected_updates
    if (work['forwards'], work['backwards'], work['optimizer_steps']) != (expected_forward, expected_backward, expected_updates):
        raise ValueError('actual separate carrier native/optimizer work differs')
    if _tensor(ids.cpu())['sha256'] != selection_hash: raise ValueError('carrier selection mutated')
    iteration = contract['policy']['source_iteration']+steps
    checkpoint_path = directory/'carrier_checkpoint.pt'
    saved = dict(schema=_SCHEMA, iteration=iteration, policy=policy, n=n, ids=ids_cpu.clone(),
                 eta=_cpu(eta), optimizer=_cpu(optimizer.state_dict()), local_updates=work['optimizer_steps'],
                 enabled=enabled, binary_sha256=extension['binary_sha256'],
                 identity_sha256=extension['identity_sha256'], gaussian=result['final_checkpoint_evidence']['file'],
                 selection_sha256=selection_hash, schedule='constant')
    with checkpoint_path.open('xb') as stream: torch.save(saved, stream)
    checkpoint = _descriptor(checkpoint_path)
    paired_path = directory/'paired_carrier_checkpoint.json'
    paired = dict(schema='carrier_checkpoint_pair_v1', iteration=iteration, gaussian=saved['gaussian'], carrier=checkpoint,
                  policy=policy, selection_sha256=selection_hash, binary_sha256=extension['binary_sha256'])
    with paired_path.open('x') as stream: json.dump(paired, stream, indent=2, sort_keys=True, allow_nan=False)
    read = _validate_carrier_checkpoint(checkpoint_path, expected_iteration=iteration, expected_policy=policy,
        expected_selection_sha256=selection_hash, expected_binary_sha256=extension['binary_sha256'],
        _test_geometry=(n, k) if _cpu_extension is not None else None)
    if not _equal(saved, read): raise ValueError('saved carrier tensor/moment/option differs from actual live state')
    result['carrier'] = dict(schema=_SCHEMA, enabled=enabled, policy=policy, selection=selection,
        initial_state=initial, final_state=final, trace=trace, work=work, extension=extension, bounds=bounds,
        checkpoint=checkpoint, readback=dict(weights_only=True, map_location='cpu', live_state_equal=True,
                                            iteration=iteration, trace_file=_descriptor(trace_path)),
        paired_checkpoint_manifest=_descriptor(paired_path),
        export_state=dict(before=export_before if enabled else final, after=final, unchanged=True,
                          calls=work['roles']['export']['forwards'], zero_raw_parity=zero_raw_parity))
    return result


def validate_carrier_checkpoint(path, *, expected_iteration, expected_policy,
                                expected_selection_sha256, expected_binary_sha256):
    """Safe CPU paired-component validation; no implicit zero-state resume."""
    return _validate_carrier_checkpoint(path, expected_iteration=expected_iteration,
        expected_policy=expected_policy, expected_selection_sha256=expected_selection_sha256,
        expected_binary_sha256=expected_binary_sha256)


def _validate_carrier_checkpoint(path, *, expected_iteration, expected_policy,
                                 expected_selection_sha256, expected_binary_sha256, _test_geometry=None):
    import torch
    contract = _contract()
    if (not _equal(expected_policy, contract['policy']['carrier'])
            or expected_binary_sha256 != contract['artifact']['binary']['sha256']
            or type(expected_iteration) is not int or expected_iteration not in (4500, 4502, 5000)):
        raise ValueError('checkpoint expectations differ from fixed carrier contract')
    path = Path(path)
    value = torch.load(path, map_location='cpu', weights_only=True)
    required = {'schema', 'iteration', 'policy', 'n', 'ids', 'eta', 'optimizer', 'local_updates', 'enabled',
                'binary_sha256', 'identity_sha256', 'gaussian', 'selection_sha256', 'schedule'}
    if (type(value) is not dict or set(value) != required or value['schema'] != _SCHEMA
            or type(value['iteration']) is not int or value['iteration'] != expected_iteration
            or not _equal(value['policy'], expected_policy) or value['schedule'] != 'constant'
            or value['binary_sha256'] != expected_binary_sha256
            or value['identity_sha256'] != _contract()['artifact']['identity_sha256']
            or value['selection_sha256'] != expected_selection_sha256):
        raise ValueError('carrier checkpoint schema/iteration/policy/artifact differs')
    ids = value['ids']; n = value['n']
    if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or ids.ndim != 1 or ids.numel() == 0
            or type(n) is not int or n < len(ids) or not bool(((ids >= 0) & (ids < n)).all())
            or not bool((ids[1:] > ids[:-1]).all()) or _tensor(ids)['sha256'] != expected_selection_sha256):
        raise ValueError('carrier checkpoint selection differs')
    geometry = (contract['policy']['point_count'], expected_policy['k']) if _test_geometry is None else _test_geometry
    if (n, len(ids)) != geometry:
        raise ValueError('carrier checkpoint fixed selection shape/count differs')
    eta = value['eta']
    if not isinstance(eta, dict) or set(eta) != {'pre', 'post'}:
        raise ValueError('carrier contrast state differs')
    for p in eta.values():
        if (not isinstance(p, torch.Tensor) or p.dtype != torch.float32 or tuple(p.shape) != (len(ids), 3)
                or not bool(torch.isfinite(p).all())):
            raise ValueError('carrier eta shape/dtype/finiteness differs')
    updates = value['local_updates']
    if (type(updates) is not int or type(value['enabled']) is not bool
            or updates != (expected_iteration-_contract()['policy']['source_iteration'] if value['enabled'] else 0)
            or updates < 0):
        raise ValueError('carrier local update clock differs')
    if not updates and any(not bool((p == 0).all()) for p in eta.values()):
        raise ValueError('untrained carrier state must retain zero initialization')
    dummy = [torch.nn.Parameter(torch.zeros_like(p)) for p in eta.values()]
    expected_groups = _optimizer(dummy, expected_policy).state_dict()['param_groups']
    opt = value['optimizer']
    if not isinstance(opt, dict) or set(opt) != {'state', 'param_groups'} or not _equal(opt['param_groups'], expected_groups):
        raise ValueError('carrier Adam options differ')
    if set(opt['state']) != ({0, 1} if updates else set()): raise ValueError('carrier Adam membership differs')
    for row in opt['state'].values():
        if set(row) != {'step', 'exp_avg', 'exp_avg_sq'}: raise ValueError('carrier Adam fields differ')
        for name, tensor in row.items():
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                    or tuple(tensor.shape) != (() if name == 'step' else (len(ids), 3))
                    or not bool(torch.isfinite(tensor).all())):
                raise ValueError('carrier Adam shape/dtype/finite differs')
        if float(row['step']) != updates or bool((row['exp_avg_sq'] < 0).any()):
            raise ValueError('carrier Adam clock/moment differs')
    pair_path = path.parent/'paired_carrier_checkpoint.json'
    paired = json.loads(pair_path.read_text())
    expected_pair = dict(schema='carrier_checkpoint_pair_v1', iteration=expected_iteration, gaussian=value['gaussian'],
        carrier=_descriptor(path), policy=expected_policy, selection_sha256=expected_selection_sha256,
        binary_sha256=expected_binary_sha256)
    if not _equal(paired, expected_pair) or _descriptor(value['gaussian']['path']) != value['gaussian']:
        raise ValueError('carrier/Gaussian paired checkpoint incomplete or different')
    gaussian = torch.load(value['gaussian']['path'], map_location='cpu', weights_only=True)
    if type(gaussian.get('iter')) is not int or gaussian['iter'] != expected_iteration:
        raise ValueError('paired Gaussian checkpoint iteration differs')
    return value
