"""Pure fixed-budget Gaussian selection and in-place Adam row transaction.

This module neither imports the renderer nor loads data. The caller supplies the
original quaternion function; indices are paired in returned ranking order.
"""
from __future__ import annotations

import hashlib
import json
import math

import torch


NAMES = ('positions', 'scales', 'rotations', 'density_real', 'density_imag',
         'density_real_contrast', 'density_imag_contrast')
_WIDTHS = (3, 3, 4, None, None, None, None)
_AMPLITUDE_FACTOR = .5 / (.5 * .85 * .85)


def _hash(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _json(value):
    """Copy optimizer/scheduler metadata into strict JSON, before any writes."""
    if torch.is_tensor(value):
        return _json(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError('optimizer/scheduler metadata must be finite JSON values')


def _parameters(model):
    if tuple(name for name, _ in model.named_parameters()) != NAMES:
        raise ValueError('expected exactly the seven canonical model parameters')
    parameters = {name: getattr(model, name) for name in NAMES}
    first = parameters['positions']
    if not isinstance(first, torch.nn.Parameter) or first.ndim != 2:
        raise ValueError('positions must be an N by 3 Parameter')
    n = first.shape[0]
    if n < 1 or getattr(model, 'num_points', None) != n:
        raise ValueError('model num_points must match its nonempty row count')
    if first.dtype not in (torch.float32, torch.float64) or first.layout != torch.strided:
        raise ValueError('model parameters must use dense float32 or float64')
    for (name, parameter), width in zip(parameters.items(), _WIDTHS):
        expected = (n,) if width is None else (n, width)
        if (not isinstance(parameter, torch.nn.Parameter) or tuple(parameter.shape) != expected
                or parameter.dtype != first.dtype or parameter.device != first.device
                or parameter.layout != torch.strided):
            raise ValueError(f'invalid shape, device or dtype for {name}')
    if len({id(parameter) for parameter in parameters.values()}) != 7:
        raise ValueError('canonical parameters must be distinct objects')
    return parameters, n


def select_relocations(model, scores, requested_k):
    """Rank low mass victims and positive acquired-evidence donors stably.

    All initially requested victims stay excluded from donor eligibility even
    when positive evidence reduces the effective pair count.
    """
    parameters, n = _parameters(model)
    device = parameters['positions'].device
    if isinstance(requested_k, bool) or not isinstance(requested_k, int) or not 0 <= requested_k <= n:
        raise ValueError('requested_k must be an integer between zero and point count')
    if (not torch.is_tensor(scores) or scores.shape != (n,) or scores.device != device
            or scores.dtype not in (torch.float32, torch.float64) or scores.layout != torch.strided):
        raise ValueError('scores must be a dense floating vector on the model device')
    with torch.no_grad():
        if not torch.isfinite(scores).all() or (scores < 0).any():
            raise ValueError('evidence scores must be finite and nonnegative')
        mass_inputs = [parameters[name].detach().to(torch.float64) for name in NAMES[3:]]
        logs = parameters['scales'].detach().to(torch.float64)
        if not all(torch.isfinite(value).all() for value in [logs, *mass_inputs]):
            raise ValueError('mass proxy inputs must be finite')
        plain = torch.hypot(mass_inputs[0], mass_inputs[1])
        contrast = torch.hypot(mass_inputs[2], mass_inputs[3])
        proxy = torch.maximum(plain, contrast) * logs.exp().prod(dim=1)
        if not torch.isfinite(proxy).all():
            raise ValueError('nonfinite float64 mass proxy')
        requested_victims = torch.argsort(proxy, stable=True)[:requested_k]
        eligible = scores > 0
        eligible[requested_victims] = False
        eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
        ranked = torch.argsort(scores[eligible_indices], descending=True, stable=True)
        k_eff = min(requested_k, eligible_indices.numel())
        donors = eligible_indices[ranked[:k_eff]].clone()
        victims = requested_victims[:k_eff].clone()
        record = {
            'schema': 'relocation_selection_v1', 'point_count': n,
            'requested_k': requested_k, 'k_eff': k_eff,
            'eligible_donor_count': eligible_indices.numel(),
            'requested_victims': requested_victims.cpu().tolist(),
            'requested_victims_sha256': _hash(requested_victims),
            'donors': donors.cpu().tolist(), 'victims': victims.cpu().tolist(),
            'donors_sha256': _hash(donors), 'victims_sha256': _hash(victims),
            'victim_proxy': proxy[victims].cpu().tolist(),
            'donor_scores': scores[donors].detach().cpu().tolist(),
            'scores_sha256': _hash(scores), 'proxy_sha256': _hash(proxy),
            'proxy_dtype': 'float64', 'tie_policy': 'original_row_index',
        }
    return donors, victims, record


def _validate_indices(donors, victims, device, n):
    for value in (donors, victims):
        if (not torch.is_tensor(value) or value.dtype != torch.int64 or value.ndim != 1
                or value.device != device or value.layout != torch.strided):
            raise ValueError('indices must be dense int64 vectors on the model device')
        if value.numel() and ((value < 0).any() or (value >= n).any()):
            raise ValueError('row index out of range')
    if donors.numel() != victims.numel():
        raise ValueError('donors and victims must have matching lengths')
    joined = torch.cat((donors, victims))
    if torch.unique(joined).numel() != joined.numel():
        raise ValueError('donor and victim indices must be unique and disjoint')
    return donors.detach().clone(), victims.detach().clone(), joined.sort().values


def _progressions_overlap(left, right):
    """Exact intersection of two finite arithmetic address progressions."""
    a, stride_a, count_a = left
    b, stride_b, count_b = right
    low = max(a, b)
    high = min(a + stride_a * (count_a - 1), b + stride_b * (count_b - 1))
    if low > high:
        return False
    if stride_a == 0:
        return a == b if stride_b == 0 else (a - b) % stride_b == 0
    if stride_b == 0:
        return (b - a) % stride_a == 0
    divisor = math.gcd(stride_a, stride_b)
    if (b - a) % divisor:
        return False
    modulus = stride_b // divisor
    row = ((b - a) // divisor * pow(stride_a // divisor, -1, modulus)) % modulus
    first = a + stride_a * row
    period = stride_a * modulus
    first += ((low - first + period - 1) // period) * period
    return first <= high


def _byte_regions_overlap(left, left_width, right, right_width):
    """Compare element byte intervals, including differently aligned wrappers."""
    left_end = left[0] + left[1] * (left[2] - 1) + left_width
    right_end = right[0] + right[1] * (right[2] - 1) + right_width
    if left_end <= right[0] or right_end <= left[0]:
        return False
    # Two element intervals overlap when their start difference lies in this
    # finite range. This also handles clocks whose element width differs.
    return any(_progressions_overlap(left, (right[0] + shift, right[1], right[2]))
               for shift in range(1 - left_width, right_width))


def _validate_writable_storage(tensors, steps):
    """Reject actual address overlap while admitting noncontiguous row layouts.

    Original tensors have rank one or two and at most four columns. Each column
    is one arithmetic byte-address progression, so exact checks need no
    tensor-sized enumeration and admit interleaved complex real/imag views.
    Storage wrappers can have different bases into the same physical allocation;
    comparisons therefore use absolute addresses on the same device.
    """
    regions = {}
    for tensor in tensors:
        width = tensor.element_size()
        columns = 1 if tensor.ndim == 1 else tensor.shape[1]
        row_stride = tensor.stride(0) * width
        if tensor.shape[0] > 1 and row_stride == 0:
            raise ValueError('writable tensors must not have internal overlap')
        addresses = []
        for column in range(columns):
            offset = tensor.data_ptr() + (0 if tensor.ndim == 1 else column * tensor.stride(1) * width)
            progression = (offset, row_stride, tensor.shape[0])
            if any(_progressions_overlap(progression, old) for old in addresses):
                raise ValueError('writable tensors must not have internal overlap')
            addresses.append(progression)
        previous = regions.setdefault(tensor.device, [])
        if any(_byte_regions_overlap(new, width, old, old_width)
               for new in addresses for old, old_width in previous):
            raise ValueError('writable tensors must not overlap each other')
        previous.extend((address, width) for address in addresses)
    for step in steps:
        if any(_byte_regions_overlap((step.data_ptr(), 0, 1), step.element_size(), old, old_width)
               for old, old_width in regions.get(step.device, [])):
            raise ValueError('Adam scalar steps must not overlap writable tensor bytes')


def _validate_optimizer(parameters, optimizer, scheduler):
    if not isinstance(optimizer, torch.optim.Adam) or len(optimizer.param_groups) != 7:
        raise ValueError('expected the original seven-group Adam optimizer')
    if (not isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
            or scheduler.optimizer is not optimizer or scheduler.gamma != .9995):
        raise ValueError('expected the original ExponentialLR attached to Adam')
    writable = list(parameters.values())
    steps = []
    for (name, parameter), group in zip(parameters.items(), optimizer.param_groups):
        if (len(group['params']) != 1 or group['params'][0] is not parameter
                or ('name' in group and group['name'] != name)):
            raise ValueError('Adam groups must follow canonical parameter identity and order')
        if (group.get('betas') != (.9, .999) or group.get('eps') != 1e-8
                or group.get('weight_decay') != 0 or group.get('amsgrad') is not False
                or group.get('maximize', False) or group.get('differentiable', False)):
            raise ValueError('unsupported Adam continuation options')
        if not isinstance(group['lr'], (float, int)) or not math.isfinite(group['lr']) or group['lr'] < 0:
            raise ValueError('Adam learning rates must be finite and nonnegative')
        state = optimizer.state.get(parameter)
        if not state or set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('expected populated non-AMSGrad Adam state')
        step = state['step']
        if (not torch.is_tensor(step) or step.ndim != 0 or not torch.isfinite(step)
                or step.item() < 0 or step.item() != int(step.item())):
            raise ValueError('Adam step must be a finite nonnegative scalar integer')
        steps.append(step)
        for key in ('exp_avg', 'exp_avg_sq'):
            moment = state[key]
            if (not torch.is_tensor(moment) or moment.shape != parameter.shape
                    or moment.dtype != parameter.dtype or moment.device != parameter.device
                    or moment.layout != torch.strided):
                raise ValueError(f'invalid Adam {key} for {name}')
            writable.append(moment)
        gradient = parameter.grad
        if gradient is not None and (gradient.shape != parameter.shape or gradient.dtype != parameter.dtype
                                     or gradient.device != parameter.device or gradient.layout != torch.strided):
            raise ValueError(f'invalid gradient for {name}')
        if gradient is not None:
            writable.append(gradient)
    _validate_writable_storage(writable, steps)
    # Fail unsupported/nonfinite metadata before the transaction, not at export.
    _json(scheduler.state_dict())
    _json([{key: value for key, value in group.items() if key != 'params'} for group in optimizer.param_groups])


def _tensor_record(tensor, untouched, changed):
    if tensor is None:
        return None
    return {'id': id(tensor), 'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'device': str(tensor.device), 'sha256': _hash(tensor),
            'untouched_sha256': _hash(tensor[untouched]),
            'changed_sha256': _hash(tensor[changed])}


def _snapshot(model, parameters, optimizer, scheduler, untouched, changed):
    return {
        'point_count': model.num_points, 'model_id': id(model),
        'optimizer_id': id(optimizer), 'scheduler_id': id(scheduler),
        'parameters': {name: _tensor_record(p, untouched, changed) for name, p in parameters.items()},
        'gradients': {name: _tensor_record(p.grad, untouched, changed) for name, p in parameters.items()},
        'moments': {name: {key: _tensor_record(optimizer.state[p][key], untouched, changed)
                           for key in ('exp_avg', 'exp_avg_sq')} for name, p in parameters.items()},
        'steps': {name: optimizer.state[p]['step'].item() for name, p in parameters.items()},
        'step_ids': {name: id(optimizer.state[p]['step']) for name, p in parameters.items()},
        'step_sha256': {name: _hash(optimizer.state[p]['step']) for name, p in parameters.items()},
        'resolved_group_names': list(parameters),
        'group_parameter_ids': [[id(p) for p in group['params']] for group in optimizer.param_groups],
        'group_options': _json([{key: value for key, value in group.items() if key != 'params'}
                                for group in optimizer.param_groups]),
        'scheduler': _json(scheduler.state_dict()),
    }


def relocate_rows(model, optimizer, scheduler, donors, victims, rotation_matrix_fn):
    """Replace each donor/victim pair with two children without changing shape.

    All validation and both children are computed before the first write. Adam's
    scalar age survives; only affected moment and existing gradient rows reset.
    """
    parameters, n = _parameters(model)
    device = parameters['positions'].device
    donors, victims, changed = _validate_indices(donors, victims, device, n)
    _validate_optimizer(parameters, optimizer, scheduler)
    if not callable(rotation_matrix_fn):
        raise ValueError('rotation_matrix_fn must be callable')
    k_eff = donors.numel()
    with torch.no_grad():
        parents = {name: parameter[donors].detach().clone() for name, parameter in parameters.items()}
        if not all(torch.isfinite(value).all() for value in parents.values()):
            raise ValueError('donor values must all be finite')
        children = {}
        if k_eff:
            scales = parents['scales'].exp()
            axes = scales.argmax(dim=1)
            factors = torch.full_like(scales, .85)
            factors.scatter_(1, axes[:, None], .5)
            child_scales = scales * factors
            if not torch.isfinite(child_scales).all() or (child_scales <= 0).any():
                raise ValueError('child scales must be positive and finite')
            local_shift = torch.zeros_like(scales)
            local_shift.scatter_(1, axes[:, None], child_scales.gather(1, axes[:, None]))
            rotation = rotation_matrix_fn(parents['rotations'].clone())
            if (not torch.is_tensor(rotation) or rotation.shape != (k_eff, 3, 3)
                    or rotation.dtype != scales.dtype or rotation.device != device
                    or rotation.layout != torch.strided or not torch.isfinite(rotation).all()):
                raise ValueError('rotation function returned invalid matrices')
            shift = torch.bmm(rotation, local_shift.unsqueeze(-1)).squeeze(-1)
            # Validate the unclamped sums too: clamp must not conceal overflow.
            minus = parents['positions'] - shift
            plus = parents['positions'] + shift
            if not torch.isfinite(minus).all() or not torch.isfinite(plus).all():
                raise ValueError('child positions overflowed')
            children['positions'] = (minus.clamp(-1., 1.), plus.clamp(-1., 1.))
            logs = child_scales.log()
            children['scales'] = (logs, logs)
            children['rotations'] = (parents['rotations'], parents['rotations'])
            for name in NAMES[3:]:
                coefficient = parents[name] * _AMPLITUDE_FACTOR
                children[name] = (coefficient, coefficient)
            if not all(torch.isfinite(value).all() for pair in children.values() for value in pair):
                raise ValueError('nonfinite derived child parameter')
        untouched = torch.ones(n, dtype=torch.bool, device=device)
        untouched[changed] = False
        before = _snapshot(model, parameters, optimizer, scheduler, untouched, changed)
        if k_eff:
            for name, parameter in parameters.items():
                parameter.index_copy_(0, donors, children[name][0])
                parameter.index_copy_(0, victims, children[name][1])
                for key in ('exp_avg', 'exp_avg_sq'):
                    optimizer.state[parameter][key].index_fill_(0, changed, 0.)
                if parameter.grad is not None:
                    parameter.grad.index_fill_(0, changed, 0.)
        after = _snapshot(model, parameters, optimizer, scheduler, untouched, changed)
        zero_checks = {
            name: {**{key: bool(torch.count_nonzero(optimizer.state[p][key][changed]) == 0)
                       for key in ('exp_avg', 'exp_avg_sq')},
                   'gradient': p.grad is None or bool(torch.count_nonzero(p.grad[changed]) == 0)}
            for name, p in parameters.items()
        }
        record = {
            'schema': 'relocation_transaction_v1', 'point_count': n, 'k_eff': k_eff,
            'changed_row_count': changed.numel(), 'untouched_row_count': n - changed.numel(),
            'donors': donors.cpu().tolist(), 'victims': victims.cpu().tolist(),
            'donors_sha256': _hash(donors), 'victims_sha256': _hash(victims),
            'changed_indices': changed.cpu().tolist(), 'changed_indices_sha256': _hash(changed),
            'before': before, 'after': after, 'changed_rows_zero': zero_checks,
            'transform': {'long_axis_factor': .5, 'other_axis_factor': .85,
                          'offset_factor': 1., 'coefficient_factor': _AMPLITUDE_FACTOR,
                          'donor_child_sign': -1, 'victim_child_sign': 1,
                          'scale_storage': 'log', 'clamp_new_children': [-1., 1.]},
        }
        json.dumps(record, allow_nan=False)
    return record
