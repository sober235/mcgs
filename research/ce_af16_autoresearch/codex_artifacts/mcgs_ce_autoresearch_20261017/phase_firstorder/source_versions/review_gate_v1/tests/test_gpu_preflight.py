"""Exercise the preflight mathematics with CPU source-physics fixtures."""
import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from density_operator import DensityLinearOperator
from gpu_preflight import check_physics
from test_density_operator import TinyTrainer, assert_nested_equal


def test_physics_checks_measure_actual_operator_without_parameter_updates():
    trainer = TinyTrainer(dtype=torch.float32)
    before = copy.deepcopy(trainer.optimizer.state_dict())
    references = dict(trainer.model._parameters)
    values = {name: p.detach().clone() for name, p in references.items()}
    result = check_physics(DensityLinearOperator(trainer))
    assert result['passed'] is True
    assert set(result['errors']) == {'superposition', 'adjoint', 'normal_linearity', 'normal_hermitian'}
    assert all(value < result['tolerance'] for value in result['errors'].values())
    assert all(trainer.model._parameters[name] is p for name, p in references.items())
    assert all(torch.equal(values[name], p) for name, p in references.items())
    assert_nested_equal(trainer.optimizer.state_dict(), before)


def test_bad_adjoint_is_rejected_without_changing_tolerance():
    operator = DensityLinearOperator(TinyTrainer(dtype=torch.float32))
    actual = operator.adjoint
    operator.adjoint = lambda a, b: actual(a, b) * 1.1
    with pytest.raises(RuntimeError, match='adjoint'):
        check_physics(operator)


@pytest.mark.parametrize('tolerance', [0., -1., float('inf'), float('nan'), .1])
def test_preflight_tolerance_cannot_be_disabled(tolerance):
    with pytest.raises(ValueError, match='tolerance'):
        check_physics(DensityLinearOperator(TinyTrainer()), tolerance=tolerance)


def test_actual_source_ast_loads_under_ascii_without_cuda_or_compilation():
    script = """
import locale, types
assert locale.getpreferredencoding(False).lower() in ('ansi_x3.4-1968', 'ascii')
import gpu_preflight as gp
def forbidden(*args, **kwargs):
    raise AssertionError('CUDA/build loading is forbidden in this CPU replay')
gp.torch.cuda.init = forbidden
gp.torch.cuda.is_available = forbidden
gp._load_module = forbidden
gp.make_tiny_cuda_operator = forbidden
trainer = types.SimpleNamespace()
gp._load_source_functions(trainer)
assert all(callable(getattr(trainer, name)) for name in ('_render', '_kspace_loss', '_build_freq_weight'))
assert not gp.torch.cuda.is_initialized()
print('actual-source ASCII replay passed; no CUDA or extension loading')
"""
    environment = {**os.environ, "LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0",
                   "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", script],
                            cwd=Path(__file__).resolve().parents[1], env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "actual-source ASCII replay passed" in result.stdout


@pytest.mark.parametrize("mode,batches", [("single", 5), ("uniform8", 40)])
def test_new_prior_mode_matches_independent_source_wavelet_value_and_gradient(mode, batches):
    from gpu_preflight import check_prior_mode
    torch.manual_seed(18)
    x = torch.randn(2, 6, 6, 6, dtype=torch.complex128)
    original = x.clone()
    result = check_prior_mode(x, wavelet_mode=mode)
    assert result["passed"] is True and result["wavelet_mode"] == mode
    assert result["wavelet_transform_batches"] == batches
    assert result["reference_wavelet_transform_batches"] == (1 if mode == "single" else 8)
    assert all(value < 1e-12 for value in result["errors"].values())
    torch.testing.assert_close(x, original, rtol=0, atol=0)
