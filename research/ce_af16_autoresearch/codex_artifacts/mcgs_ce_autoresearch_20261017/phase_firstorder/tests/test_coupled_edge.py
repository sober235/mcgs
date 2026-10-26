import ast
from pathlib import Path

import pytest
import torch

from coupled_edge import build_frozen_edge


def source_method():
    path = Path(__file__).resolve().parents[2] / "nonlocal_autoresearch/trainer/paired_gs/trainers/joint_trainer.py"
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_edge_align_loss")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_edge_align_loss"]


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_value_and_both_gradients_match_actual_trainer_at_anchor(dtype):
    torch.manual_seed(12)
    x = (torch.randn(2, 4, 5, 6, dtype=dtype) + 1).requires_grad_()
    frozen = build_frozen_edge(x, p=.9, kappa=30.)
    actual = source_method()(None, x[0], x[1], .9, 30.)
    expected_grad, = torch.autograd.grad(actual, x)
    result = frozen.merit(x)
    actual_grad, = torch.autograd.grad(result, x)
    torch.testing.assert_close(result, actual, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_gates_stay_frozen_away_from_anchor_and_info_is_private():
    torch.manual_seed(1)
    x = torch.randn(2, 4, 5, 6, dtype=torch.complex128)
    frozen = build_frozen_edge(x)
    y = x.clone()
    y[0] *= 5
    assert abs(float(frozen.merit(y) - source_method()(None, y[0], y[1], .9, 30))) > 1e-5
    frozen.info["p"] = 0
    assert frozen.info["p"] == .9
    assert frozen.merit(x).grad_fn is None


def test_invalid_shape_or_nonfinite_is_rejected():
    with pytest.raises(ValueError):
        build_frozen_edge(torch.ones(3, 2, 2, 2, dtype=torch.complex64))
    with pytest.raises(ValueError):
        build_frozen_edge(torch.full((2, 2, 2, 2), complex(float("nan")), dtype=torch.complex64))
