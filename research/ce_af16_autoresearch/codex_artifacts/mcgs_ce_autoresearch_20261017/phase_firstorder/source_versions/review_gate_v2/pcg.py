"""Bounded complex Hermitian PCG with detached vectors and scalar diagnostics."""
from collections import deque
from dataclasses import dataclass
import math
import numbers

import torch


class NumericalBreakdown(RuntimeError):
    """Nonfinite arithmetic or a violated positive Hermitian normal-system assumption."""


@dataclass
class PCGResult:
    x: torch.Tensor
    converged: bool
    iterations: int
    matvecs: int
    residual_history: list
    status: str


def _real_dot(left, right):
    """Real Hermitian inner product with products and accumulation in float64."""
    return ((left.real.double() * right.real.double()).sum()
            + (left.imag.double() * right.imag.double()).sum()).item()


def _norm(vector):
    square = _real_dot(vector, vector)
    if not math.isfinite(square) or square < 0:
        raise NumericalBreakdown("nonfinite residual/vector norm")
    return math.sqrt(square)


def _integer(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def solve_pcg(apply, rhs, x0=None, preconditioner=None, maxiter=50, rtol=1e-3,
              atol=0, max_matvecs=None, callback=None):
    """Solve Hx=rhs with complex Hermitian positive-definite PCG.

    ``preconditioner`` is a finite positive *real inverse diagonal* multiplying
    residuals; each complex entry uses the same scale for its real and imaginary
    parts. It may broadcast to rhs. All supplied tensors are detached and never
    mutated. ``apply`` receives a private detached clone, must return shape/dtype/
    device matching rhs, and every actual call consumes one matvec, including an
    initial nonzero x0 residual. An omitted or all-zero x0 needs no initial call.

    Callbacks receive a new dict of Python scalars at initialization and after
    each accepted CG iteration. Residual history retains at most 257 scalars.
    Residuals are the CG recurrence residuals; no uncounted verification matvec
    is made. Budget exhaustion returns a nonconverged result.
    """
    if not callable(apply):
        raise TypeError("apply must be callable")
    if not isinstance(rhs, torch.Tensor) or rhs.dtype not in (torch.complex64, torch.complex128):
        raise TypeError("rhs must be a complex64 or complex128 tensor")
    if rhs.numel() == 0:
        raise ValueError("rhs must not be empty")
    maxiter = _integer("maxiter", maxiter)
    budget = None if max_matvecs is None else _integer("max_matvecs", max_matvecs)
    if not math.isfinite(float(rtol)) or not math.isfinite(float(atol)) or rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be finite and nonnegative")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")
    b = rhs.detach().clone()
    if not bool(torch.isfinite(b).all()):
        raise NumericalBreakdown("rhs contains nonfinite values")
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        if not isinstance(x0, torch.Tensor) or x0.shape != b.shape or x0.dtype != b.dtype or x0.device != b.device:
            raise ValueError("x0 must match rhs shape, dtype, and device")
        x = x0.detach().clone()
        if not bool(torch.isfinite(x).all()):
            raise NumericalBreakdown("x0 contains nonfinite values")
    inverse = None
    if preconditioner is not None:
        if not isinstance(preconditioner, torch.Tensor) or preconditioner.is_complex() or not preconditioner.is_floating_point():
            raise TypeError("preconditioner must be a positive real floating tensor")
        if preconditioner.device != b.device:
            raise ValueError("preconditioner and rhs must share a device")
        try:
            inverse = torch.broadcast_to(preconditioner.detach(), b.shape).clone()
        except RuntimeError as error:
            raise ValueError("preconditioner must broadcast to rhs") from error
        if not bool(torch.isfinite(inverse).all()) or not bool((inverse > 0).all()):
            raise ValueError("preconditioner must be finite and strictly positive")
        inverse = inverse.to(dtype=b.real.dtype)
        if not bool(torch.isfinite(inverse).all()) or not bool((inverse > 0).all()):
            raise ValueError("preconditioner must remain finite and positive in rhs precision")
    history = deque(maxlen=257)
    iterations, matvecs = 0, 0
    threshold = max(float(atol), float(rtol) * _norm(b))
    residual_norm = None

    def emit(alpha=None, beta=None, curvature=None):
        if callback is not None:
            callback({"iteration": iterations, "matvecs": matvecs, "residual_norm": residual_norm,
                      "threshold": threshold, "alpha": alpha, "beta": beta, "curvature": curvature})

    def finish(converged, status):
        return PCGResult(x.detach().clone(), converged, iterations, matvecs, list(history), status)

    def has_budget():
        return budget is None or matvecs < budget

    def matvec(vector):
        nonlocal matvecs
        # enable_grad permits callers to obtain first-order VJPs while solver
        # vector arithmetic below remains under no_grad.
        matvecs += 1
        with torch.enable_grad():
            value = apply(vector.detach().clone())
        if not isinstance(value, torch.Tensor) or value.shape != b.shape or value.dtype != b.dtype or value.device != b.device:
            raise ValueError("apply must return a tensor matching rhs shape, dtype, and device")
        value = value.detach().clone()
        if not bool(torch.isfinite(value).all()):
            raise NumericalBreakdown("normal application returned nonfinite values")
        return value

    with torch.no_grad():
        if bool((x != 0).any()):
            if not has_budget():
                emit()
                return finish(False, "matvec_budget")
            residual = b - matvec(x)
        else:
            residual = b.clone()
        residual_norm = _norm(residual)
        history.append(residual_norm)
        emit()
        if residual_norm <= threshold:
            return finish(True, "converged")
        z = residual.clone() if inverse is None else inverse * residual
        rz = _real_dot(residual, z)
        if not math.isfinite(rz) or rz <= 0:
            raise NumericalBreakdown("preconditioned residual inner product is not finite and positive")
        direction = z.clone()
        for _ in range(maxiter):
            if not has_budget():
                return finish(False, "matvec_budget")
            product = matvec(direction)
            curvature = _real_dot(direction, product)
            imag = ((direction.real.double() * product.imag.double()).sum()
                    - (direction.imag.double() * product.real.double()).sum()).item()
            tolerance = 5e-5 if b.dtype == torch.complex64 else 1e-11
            scale = _norm(direction) * _norm(product)
            if not math.isfinite(curvature) or not math.isfinite(imag):
                raise NumericalBreakdown("normal curvature is nonfinite")
            if abs(imag) > tolerance * max(scale, abs(curvature)):
                raise NumericalBreakdown("normal application has seriously non-Hermitian curvature")
            if curvature <= 0:
                raise NumericalBreakdown("normal curvature is not strictly positive")
            alpha = rz / curvature
            if not math.isfinite(alpha):
                raise NumericalBreakdown("PCG step length is nonfinite")
            x = x + alpha * direction
            residual = residual - alpha * product
            iterations += 1
            residual_norm = _norm(residual)
            if not bool(torch.isfinite(x).all()):
                raise NumericalBreakdown("PCG iterate is nonfinite")
            history.append(residual_norm)
            if residual_norm <= threshold:
                emit(alpha=alpha, curvature=curvature)
                return finish(True, "converged")
            z = residual.clone() if inverse is None else inverse * residual
            next_rz = _real_dot(residual, z)
            if not math.isfinite(next_rz) or next_rz <= 0:
                raise NumericalBreakdown("preconditioned residual inner product is not finite and positive")
            beta = next_rz / rz
            if not math.isfinite(beta):
                raise NumericalBreakdown("PCG recurrence coefficient is nonfinite")
            emit(alpha=alpha, beta=beta, curvature=curvature)
            direction = z + beta * direction
            rz = next_rz
        return finish(False, "maxiter")
