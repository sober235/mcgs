import pytest
import torch

from pcg import NumericalBreakdown, solve_pcg


def hermitian_spd(dtype):
    matrix = torch.tensor([[2 + 1j, .5 - .2j, -.3j], [1j, 3., -.4], [.2, .1j, 1.5]], dtype=dtype)
    return matrix.mH @ matrix + .2 * torch.eye(3, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complex_spd_matches_direct_solution_with_real_diagonal_preconditioning(dtype):
    matrix = hermitian_spd(dtype)
    rhs = torch.tensor([1 + 2j, -.5 + .3j, 2 - 1j], dtype=dtype, requires_grad=True)
    x0 = torch.full_like(rhs, .1 - .2j)
    rhs_before, x_before = rhs.detach().clone(), x0.clone()
    preconditioner = matrix.diagonal().real.reciprocal().detach().requires_grad_()
    pre_before = preconditioner.detach().clone()
    calls = []
    events = []

    def apply(vector):
        calls.append(vector.detach().clone())
        return matrix @ vector

    result = solve_pcg(apply, rhs, x0=x0, preconditioner=preconditioner,
                       maxiter=12, rtol=2e-6 if dtype == torch.complex64 else 1e-12,
                       callback=events.append)
    assert result.converged and result.status == "converged"
    assert result.matvecs == len(calls)
    assert torch.allclose(result.x, torch.linalg.solve(matrix, rhs), rtol=2e-5, atol=2e-6)
    assert torch.linalg.vector_norm(matrix @ result.x - rhs) / torch.linalg.vector_norm(rhs) < 3e-6
    assert not result.x.requires_grad and result.x.grad_fn is None
    assert torch.equal(rhs.detach(), rhs_before) and torch.equal(x0, x_before)
    assert torch.equal(preconditioner.detach(), pre_before)
    assert result.residual_history[-1] <= result.residual_history[0]
    assert len(events) == result.iterations + 1  # Initial residual and accepted CG steps.
    assert all(isinstance(value, (str, bool, int, float, type(None)))
               for event in events for value in event.values())
    assert events[-1]["matvecs"] == result.matvecs


def test_damped_rank_deficient_normal_system():
    factor = torch.tensor([[1., 1j, 0], [0, 0, 1.]], dtype=torch.complex128)
    matrix = factor.mH @ factor + .01 * torch.eye(3, dtype=torch.complex128)
    rhs = torch.tensor([1 + 2j, -3 + .2j, .5j], dtype=torch.complex128)
    result = solve_pcg(lambda x: matrix @ x, rhs, rtol=1e-11, maxiter=10)
    assert result.converged
    assert torch.allclose(result.x, torch.linalg.solve(matrix, rhs), atol=1e-9)


def test_zero_rhs_no_operator_calls_and_budgets_count_actual_matvecs():
    calls = []

    def apply(vector):
        calls.append(1)
        return torch.tensor([2., 3.], dtype=torch.float64) * vector

    zero = torch.zeros(2, dtype=torch.complex128)
    result = solve_pcg(apply, zero, max_matvecs=0)
    assert result.converged and result.matvecs == 0 and calls == []
    rhs = torch.ones_like(zero)
    result = solve_pcg(apply, rhs, max_matvecs=0)
    assert not result.converged and result.status == "matvec_budget"
    assert result.iterations == result.matvecs == 0
    result = solve_pcg(apply, rhs, max_matvecs=1, maxiter=20, rtol=1e-12)
    assert not result.converged and result.status == "matvec_budget"
    assert result.iterations == result.matvecs == 1 and len(calls) == 1
    assert torch.linalg.vector_norm(apply(result.x) - rhs) > 1e-3
    calls.clear()
    # Nonzero x0 consumes one call to measure its residual before any CG step.
    result = solve_pcg(apply, rhs, x0=torch.ones_like(rhs), max_matvecs=1)
    assert result.matvecs == 1 and result.iterations == 0 and len(calls) == 1
    assert not result.converged and result.status == "matvec_budget"


def test_zero_budget_does_not_claim_exact_nonzero_initial_guess():
    rhs = torch.ones(2, dtype=torch.complex128)
    result = solve_pcg(lambda x: x, rhs, x0=rhs, max_matvecs=0)
    assert result.matvecs == 0 and not result.converged
    assert result.status == "matvec_budget"


@pytest.mark.parametrize("kind", ["negative", "zero", "nan", "nonhermitian"])
def test_curvature_and_nonfinite_breakdowns_are_explicit(kind):
    rhs = torch.tensor([1 + 1j, 2.], dtype=torch.complex128)
    multipliers = {"negative": -1., "zero": 0., "nan": float("nan"), "nonhermitian": 1 + 1j}
    with pytest.raises(NumericalBreakdown):
        solve_pcg(lambda x: multipliers[kind] * x, rhs, maxiter=5)


@pytest.mark.parametrize("preconditioner", [torch.tensor([-1., 1.]), torch.tensor([0., 1.]),
                                           torch.tensor([float("nan"), 1.]), torch.ones(2, dtype=torch.complex128)])
def test_nonpositive_nonfinite_or_complex_preconditioner_is_rejected(preconditioner):
    with pytest.raises((ValueError, TypeError, NumericalBreakdown)):
        solve_pcg(lambda x: x, torch.ones(2, dtype=torch.complex128), preconditioner=preconditioner)


def test_maxiter_is_nonconverged_and_history_is_bounded():
    rhs = torch.ones(3, dtype=torch.complex128)
    result = solve_pcg(lambda x: hermitian_spd(x.dtype) @ x, rhs, maxiter=1, rtol=1e-14)
    assert not result.converged and result.status == "maxiter"
    assert result.iterations == 1 and result.matvecs == 1
    assert len(result.residual_history) <= 257


def test_long_nonconverged_solve_bounds_scalar_history_and_detaches_apply_graphs():
    diagonal = torch.logspace(0, 5, 400, dtype=torch.float64).requires_grad_()
    rhs = torch.ones(400, dtype=torch.complex128, requires_grad=True)
    seen = []
    result = solve_pcg(lambda x: diagonal * x, rhs, rtol=0, atol=0, maxiter=300,
                       callback=lambda stats: seen.append(stats))
    assert result.iterations == 300 and result.matvecs == 300
    assert not result.converged and result.status == "maxiter"
    assert len(result.residual_history) == 257 and len(seen) == 301
    assert all(isinstance(value, float) for value in result.residual_history)
    assert result.x.grad_fn is None and not result.x.requires_grad
    assert diagonal.grad is None and rhs.grad is None


def test_apply_cannot_mutate_solver_inputs_or_original_rhs():
    rhs = torch.tensor([1 + 2j, 3 - 1j], dtype=torch.complex128)
    before = rhs.clone()

    def deliberately_inplace(vector):
        vector.mul_(2)
        return vector

    result = solve_pcg(deliberately_inplace, rhs, rtol=1e-12)
    assert result.converged and torch.equal(result.x, before / 2)
    assert torch.equal(rhs, before)
