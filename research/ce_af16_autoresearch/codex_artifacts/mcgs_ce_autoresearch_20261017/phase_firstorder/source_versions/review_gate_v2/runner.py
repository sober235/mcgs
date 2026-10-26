#!/usr/bin/env python3
"""Bounded fixed-geometry density directions and true frozen-merit acceptance.

This is a research solver, not an Adam continuation. It never calls train() or
custom CUDA double backward. The clinical CLI requires independently reviewed
source hashes and separate CUDA-operator / zero-step audit gates.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

GSTMI = Path("/home/congcongliu/anaconda3/envs/GSTMI/bin/python")
if Path(sys.executable).resolve() != GSTMI.resolve():
    raise ValueError("frequency numerical work requires the actual GSTMI interpreter")
import torch

from coupled_edge import build_frozen_edge
from pcg import NumericalBreakdown, solve_pcg
from surrogates import build_frozen_prior

ROOT = Path(__file__).resolve().parent
SOURCE_ITERATION = 4000
PRODUCTION = ("runner.py", "coupled_edge.py", "density_operator.py", "surrogates.py", "pcg.py", "cycle_average.py",
              "gpu_preflight.py", "run_gpu.py", "frequency_weights.py")


def numerical_runtime():
    if Path(sys.executable).resolve() != GSTMI.resolve():
        raise ValueError("frequency numerical work requires the actual GSTMI interpreter")
    if torch.__version__ != "2.7.1+cu118" or torch.version.cuda != "11.8":
        raise ValueError("frequency numerical work requires GSTMI Torch2.7.1+cu118/CUDA11.8")
    return {"python":sys.executable,"python_version":list(sys.version_info[:3]),
            "torch_version":str(torch.__version__),"cuda_version":torch.version.cuda}


def phase_protocol(mode="first_order"):
    if mode not in ("full", "first_order"):
        raise ValueError("phase curvature mode must be full or first_order")
    return {"phase_curvature_mode":mode,"damping_curvature_mode":"full",
            "update_class":"phase_first_order_quasi_newton" if mode == "first_order" else "full_frozen_prior_curvature",
            "full_MM_majorization":False}


def frequency_protocol(t,mode="uniform8",shift=None):
    if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or t not in (0.,.5,1.):
        raise ValueError("frequency_t must be exactly0/.5/1")
    if mode != "uniform8" or shift is not None:
        raise ValueError("frequency route requires uniform8 with no individual wavelet shift")
    return {"frequency_t":float(t),**wavelet_protocol(mode,shift)}


def strength_protocol(frequency_t,local_factor,mode="uniform8",shift=None):
    if (isinstance(local_factor,bool) or not isinstance(local_factor,(int,float))
            or not math.isfinite(local_factor) or local_factor != 1. or frequency_t != 1.):
        raise ValueError("phase-first-order route requires fixed t1 and local factor1")
    return {**frequency_protocol(frequency_t,mode,shift),"local_factor":float(local_factor),
            "cg_steps":96,"true_residual_diagnostics":True,**phase_protocol()}


def factor_trace(record, local_factor):
    """Bind CLI factor metadata without altering measured numerical fields."""
    strength_protocol(1.,local_factor)
    if (any(record.get(k) != v for k,v in phase_protocol().items())
            or record.get("full_MM_majorization") is not False):
        raise ValueError("actual phase trace must already prove first_order; never relabel a full update")
    return {**record,"local_factor":float(local_factor)}


def validate_saved_frequency_evidence(run_dir,value,frequency_t,operator):
    """Read six saved arrays and independently recompute the full transform.

    This numerical child check uses NumPy float64 rather than the frequency
    utility. It checks actual original acquired masks and source tensors too.
    The stdlib-only parent verifies this proof and all file hashes separately.
    """
    runtime=numerical_runtime()
    import numpy as np
    frequency_protocol(frequency_t)
    folder=Path(run_dir).resolve()/"frequency_weights"
    if (json.loads((folder/"manifest.json").read_text()) != value or value.get("frequency_t") != frequency_t
            or value.get("mass_relative_tolerance") != 2e-6 or set(value.get("scans",{})) != {"pre","post"}):
        raise ValueError("saved frequency manifest/profile mismatch")
    scans={}
    for channel,scan,tag in ((0,"pre","p"),(1,"post","c")):
        row=value["scans"][scan]; arrays={}
        for name in ("source_weight","effective_weight","acquired_mask"):
            info=row[name];path=folder/f"{scan}_{name}.npy"
            if str(path) != info.get("path") or path.resolve().parent != folder or sha256_file(path) != info.get("file_sha256"):
                raise ValueError("saved frequency file path/hash differs")
            array=np.load(path,allow_pickle=False)
            if (list(array.shape) != info.get("shape") or "torch."+str(array.dtype) != info.get("dtype")
                    or hashlib.sha256(array.tobytes(order="C")).hexdigest() != info.get("byte_sha256")):
                raise ValueError("saved frequency raw dtype/shape/bytes differ")
            arrays[name]=array
        source,effective,mask=(arrays[k] for k in ("source_weight","effective_weight","acquired_mask"))
        original=getattr(operator.trainer,"mask_"+tag).detach().cpu().contiguous().numpy()
        original_source=operator._source_frequency_weights[channel].detach().cpu().contiguous().numpy()
        if (source.shape != effective.shape or source.shape != mask.shape or source.dtype != effective.dtype
                or source.dtype.kind != "f" or not np.isfinite(source).all() or not np.isfinite(effective).all()
                or (source < 0).any() or not np.isfinite(mask).all() or not ((mask == 0)|(mask == 1)).all()
                or mask.dtype != original.dtype or not np.array_equal(mask,original)
                or source.dtype != original_source.dtype or not np.array_equal(source,original_source)):
            raise ValueError("saved source weights/original acquired masks differ or are invalid")
        acquired=mask.astype(bool)
        if not acquired.any() or not (source[acquired] > 0).any():
            raise ValueError("saved original acquired mask has no positive source mass")
        original64=source.astype(np.float64)
        def rms(samples):
            scale=float(np.max(samples))
            return scale*math.sqrt(float(np.mean((samples/scale)**2,dtype=np.float64)))
        c=rms(original64[acquired])
        if frequency_t == 0.:
            expected=source
        elif frequency_t == 1.:
            expected=np.full_like(source,c)
        else:
            mixed=(1.-frequency_t)*original64+frequency_t*c
            expected=(mixed*(c/rms(mixed[acquired]))).astype(source.dtype)
        exact=np.array_equal(source,effective)
        tolerance=8.*float(np.finfo(source.dtype).eps)
        error=0. if np.array_equal(expected,effective) else float(np.max(np.abs(effective.astype(np.float64)-expected)/np.maximum(np.abs(expected),np.finfo(source.dtype).tiny)))
        source_mass=float(np.mean(original64[acquired]**2,dtype=np.float64))
        effective_mass=float(np.mean(effective[acquired].astype(np.float64)**2,dtype=np.float64))
        relative=abs(effective_mass/source_mass-1.)
        if (not math.isfinite(error) or error > tolerance or (frequency_t == 0. and not exact)
                or relative > 2e-6 or int(acquired.sum()) != row["acquired_count"]
                or not math.isclose(source_mass,row["source_mean_squared_weight"],rel_tol=1e-12)
                or not math.isclose(effective_mass,row["effective_mean_squared_weight"],rel_tol=1e-12)
                or not math.isclose(c,row["c"],rel_tol=1e-12)):
            raise ValueError("independent saved-array interpolation/RMS/count check failed")
        scans[scan]={"acquired_count":int(acquired.sum()),"source_mean_squared_weight":source_mass,
            "effective_mean_squared_weight":effective_mass,"c":c,"relative_mass_error":relative,
            "original_acquired_mask_exact":True,"original_source_weight_exact":True,
            "interpolation_max_relative_error":error,"interpolation_relative_tolerance":tolerance,
            "effective_identity":exact,"source_levels":[float(v) for v in np.unique(source)],
            "acquired_source_levels":[float(v) for v in np.unique(source[acquired])]}
    return {"passed":True,"frequency_t":float(frequency_t),"numerical_runtime":runtime,
            "independent_backend":"NumPy float64 full interpolation and saved-array RMS; original acquired masks",
            "mass_relative_tolerance":2e-6,"scans":scans}


def wavelet_protocol(mode="single", shift=None):
    """Make averaging explicit; an individual phase cannot be silently ignored."""
    if mode not in ("single", "uniform8"):
        raise ValueError("wavelet mode must be single or uniform8")
    if mode == "uniform8" and shift is not None:
        raise ValueError("wavelet_shift must be omitted for uniform8")
    if mode == "single":
        shift = (1, 1, 1) if shift is None else tuple(shift)
        if len(shift) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 15 for v in shift):
            raise ValueError("wavelet shift must be three integers in [0,15]")
    return {"wavelet_mode": mode, "fixed_wavelet_shift": list(shift) if mode == "single" else None,
            "wavelet_phase_count": 1 if mode == "single" else 8}


@dataclasses.dataclass(frozen=True)
class SolverConfig:
    outer_steps: int = 1
    cg_steps: int = 50
    cg_rtol: float = 1e-3
    damping_rel: float = 1e-3
    max_vjp_pairs: int = 500
    max_render_pairs: int = 700
    max_merit_forwards: int = 100
    backtracks: int = 12
    backtrack_factor: float = .5
    armijo: float = 1e-4
    true_residual_diagnostics: bool = False

    def __post_init__(self):
        if type(self.true_residual_diagnostics) is not bool:
            raise ValueError("true_residual_diagnostics must be boolean")
        for name in ("outer_steps", "cg_steps", "max_vjp_pairs", "max_render_pairs", "max_merit_forwards", "backtracks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "outer_steps" else 1):
                raise ValueError(f"invalid nonnegative/positive integer {name}")
        for name in ("cg_rtol", "damping_rel", "backtrack_factor", "armijo"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.backtrack_factor >= 1 or self.armijo >= 1:
            raise ValueError("backtrack_factor and armijo must be below 1")


@dataclasses.dataclass
class SolverResult:
    density: torch.Tensor
    status: str
    accepted_steps: int
    work: dict
    trace: list


class _BudgetStop(Exception):
    pass


def _dot(a, b):
    return float((a.real.double() * b.real.double()).sum() + (a.imag.double() * b.imag.double()).sum())


def optimize_density(operator, *, prior_kwargs, edge_weight=.001, edge_p=.9,
                     edge_kappa=30., config=None, callback=None, prior_factory=None,
                     wavelet_mode="single", phase_curvature_mode="full"):
    """Compute bounded density directions without changing the supplied model.

    Every normal evaluation uses one fresh render and one first-order VJP.
    The damping probe consumes the same budget as a CG matvec. The full merit
    gradient includes edge alignment; that term's Hessian is omitted from H.
    All acceptance evaluations use identical frozen prior weights and gates.
    We refresh these only after acceptance and do not claim descent across
    refreshes or equivalence to the original stochastic cycle-spin objective.
    """
    config = config or SolverConfig()
    phase = phase_protocol(phase_curvature_mode)
    first_order = phase_curvature_mode == "first_order"
    if not isinstance(config, SolverConfig):
        raise TypeError("config must be SolverConfig")
    if not math.isfinite(edge_weight) or edge_weight < 0:
        raise ValueError("edge_weight must be finite and nonnegative")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")
    wavelet_protocol(wavelet_mode, prior_kwargs.get("wavelet_shift"))
    rho = operator.initial_density.detach().clone()
    if rho.ndim != 2 or rho.shape[0] != 2 or not rho.is_complex() or not bool(torch.isfinite(rho).all()):
        raise ValueError("initial density must be finite complex (2,N)")
    work = dict(render_pairs=0, renderer_calls=0, vjp_pairs=0, gradient_evaluations=0,
                normal_evaluations=0, merit_forward_evaluations=0, support_forward_pairs=0,
                wavelet_transform_batches=0, wavelet_mode=wavelet_mode)
    if first_order:
        work.update(phase, full_curvature_normal_evaluations=0, phase_first_order_normal_evaluations=0)
    if config.true_residual_diagnostics:
        work.update(true_residual_evaluations=0, true_residual_elapsed_seconds=0.)
    trace, accepted = [], 0
    if config.outer_steps == 0:
        return SolverResult(rho, "complete", 0, work, trace)
    if prior_factory is None:
        if wavelet_mode == "single":
            prior_factory = build_frozen_prior
        else:
            from cycle_average import build_cycle_average_prior
            prior_factory = build_cycle_average_prior
    if not callable(prior_factory):
        raise TypeError("prior_factory must be callable")
    prior, previous_wavelets, current_record = None, 0, None

    def sync_wavelets():
        current = 0 if prior is None else prior.wavelet_transform_batches
        work["wavelet_transform_batches"] = previous_wavelets + current
        if current_record is not None:
            current_record["wavelet_transform_batches"] = current

    def finish(status):
        sync_wavelets()
        return SolverResult(rho, status, accepted, work, trace)

    def render(value, *, vjp=False, category=None):
        if work["render_pairs"] >= config.max_render_pairs:
            raise _BudgetStop("render_budget")
        if vjp and work["vjp_pairs"] >= config.max_vjp_pairs:
            raise _BudgetStop("vjp_budget")
        if category == "merit_forward_evaluations" and work[category] >= config.max_merit_forwards:
            raise _BudgetStop("merit_forward_budget")
        work["render_pairs"] += 1
        work["renderer_calls"] += 2
        if category is not None:
            work[category] += 1
        return torch.stack(operator.render(value))

    def gradient(value, loss):
        # Mark an attempted actual VJP immediately before calling autograd.
        work["vjp_pairs"] += 1
        result, = torch.autograd.grad(loss, value, create_graph=False, retain_graph=False)
        if not bool(torch.isfinite(result).all()):
            raise NumericalBreakdown("density VJP is nonfinite")
        return result.detach()

    try:
        # structural_support() internally renders B1 once (or uses its cache).
        # The API's renderer counter makes that real setup work visible.
        previous = operator.counters["render_pairs"]
        if config.max_render_pairs < 1:
            raise _BudgetStop("render_budget")
        support = torch.stack(operator.structural_support())
        setup = operator.counters["render_pairs"] - previous
        if setup not in (0, 1):
            raise RuntimeError("structural_support exceeded its one-render contract")
        work["render_pairs"] += setup
        work["renderer_calls"] += 2 * setup
        work["support_forward_pairs"] = setup
        for outer in range(config.outer_steps):
            previous_wavelets = work["wavelet_transform_batches"]
            prior = None
            current_record = None
            value = rho.detach().clone().requires_grad_(True)
            images = render(value, vjp=True, category="gradient_evaluations")
            prior = prior_factory(images.detach(), structural_support=support, **prior_kwargs)
            edge = build_frozen_edge(images.detach(), p=edge_p, kappa=edge_kappa) if edge_weight else None

            def merit(paired):
                result = operator.data_loss(paired[0], paired[1]) + prior.merit(paired)
                return result + edge_weight * edge.merit(paired) if edge is not None else result

            initial_loss = merit(images)
            before = float(initial_loss.detach())
            if not math.isfinite(before):
                raise NumericalBreakdown("frozen merit is nonfinite at anchor")
            g = gradient(value, initial_loss)
            del images, initial_loss, value
            gradient_square = _dot(g, g)
            record = {"outer_step": outer + 1, "merit_before": before, "prior": prior.info,
                      "wavelet_mode": wavelet_mode,
                      "edge": edge.info if edge is not None else {"enabled": False},
                      "gradient_norm": math.sqrt(gradient_square), "accepted": False}
            if first_order:
                record.update(phase)
                record["curvature_evaluations"] = {"full_rayleigh":0,"reduced_cg":0,"reduced_endpoint":0}
            trace.append(record)
            current_record = record
            if gradient_square == 0:
                record["status"] = "stationary"
                return finish("stationary")

            def homogeneous_normal(direction, *, include_phase=True):
                with torch.enable_grad():
                    p = direction.detach().clone().requires_grad_(True)
                    paired = render(p, vjp=True, category="normal_evaluations")
                    if first_order:
                        key = "full_curvature_normal_evaluations" if include_phase else "phase_first_order_normal_evaluations"
                        work[key] += 1
                    if include_phase:
                        # Keep the original full path's graph/operation order.
                        quadratic = .5 * (operator.data_loss(paired[0], paired[1], zero_target=True)
                                          + prior.quadratic(paired))
                    else:
                        quadratic = .5 * (operator.data_loss(paired[0], paired[1], zero_target=True)
                                          + prior.quadratic(paired, include_phase=False))
                    result = gradient(p, quadratic)
                    return result

            # The normalized current full-gradient direction supplies one
            # deterministic Rayleigh scale, not a spectral bound. Count the
            # probe as a normal/VJP evaluation, including its full render.
            probe = g / math.sqrt(gradient_square)
            hp = homogeneous_normal(probe)
            if first_order:
                record["curvature_evaluations"]["full_rayleigh"] += 1
            rayleigh = _dot(probe, hp)
            if not math.isfinite(rayleigh) or rayleigh < 0:
                raise NumericalBreakdown("damping Rayleigh probe is negative/nonfinite")
            eta = config.damping_rel * max(rayleigh, 1e-12)
            record.update(damping=eta, damping_rayleigh=rayleigh)
            del hp, probe

            def apply(direction):
                return homogeneous_normal(direction, include_phase=not first_order) + eta * direction

            remaining = min(config.max_vjp_pairs - work["vjp_pairs"],
                            config.max_render_pairs - work["render_pairs"])
            if remaining <= 0:
                raise _BudgetStop("vjp_budget" if work["vjp_pairs"] >= config.max_vjp_pairs else "render_budget")
            cg = solve_pcg(apply, -.5 * g, maxiter=config.cg_steps, rtol=config.cg_rtol,
                           max_matvecs=remaining)
            record.update(cg_status=cg.status, cg_converged=cg.converged, cg_iterations=cg.iterations,
                          cg_matvecs=cg.matvecs, cg_residual_history=cg.residual_history)
            if first_order:
                record["curvature_evaluations"]["reduced_cg"] = cg.matvecs
            direction = cg.x
            if config.true_residual_diagnostics:
                # Independently evaluate the same damped, frozen H at the
                # endpoint. This extra normal call is diagnostics only: it
                # never replaces the recurrence residual or alters direction.
                if direction.is_cuda:
                    torch.cuda.synchronize(direction.device)
                diagnostic_start = time.perf_counter()
                try:
                    residual = -.5 * g - apply(direction)
                    residual_norm = math.sqrt(_dot(residual, residual))
                    if not math.isfinite(residual_norm):
                        raise NumericalBreakdown("true endpoint residual is nonfinite")
                    work["true_residual_evaluations"] += 1
                    rhs_norm = .5 * math.sqrt(gradient_square)
                    recursive_norm = cg.residual_history[-1]
                    record["true_residual"] = {
                        "diagnostic_only":True,"rhs_norm":rhs_norm,"norm":residual_norm,
                        "relative":residual_norm/rhs_norm,"recursive_norm":recursive_norm,
                        "recursive_relative":recursive_norm/rhs_norm,
                        "relative_norm_discrepancy":abs(residual_norm-recursive_norm)/rhs_norm,
                        "normal_evaluations":1}
                    if first_order:
                        record["curvature_evaluations"]["reduced_endpoint"] += 1
                        record["true_residual"]["phase_curvature_mode"] = "first_order"
                    del residual
                finally:
                    if direction.is_cuda:
                        torch.cuda.synchronize(direction.device)
                    diagnostic_elapsed = time.perf_counter()-diagnostic_start
                    work["true_residual_elapsed_seconds"] += diagnostic_elapsed
                record["true_residual"]["elapsed_seconds"] = diagnostic_elapsed
            slope = _dot(g, direction)
            if not math.isfinite(slope) or slope >= 0:
                raise NumericalBreakdown("computed density direction is not a finite strict descent direction")
            record["directional_derivative"] = slope
            trials = []
            for backtrack in range(config.backtracks):
                alpha = config.backtrack_factor ** backtrack
                candidate = rho + alpha * direction
                with torch.no_grad():
                    paired = render(candidate, category="merit_forward_evaluations")
                    after = float(merit(paired))
                if not math.isfinite(after):
                    raise NumericalBreakdown("backtracking frozen merit is nonfinite")
                trials.append({"alpha": alpha, "merit": after})
                if after < before and after <= before + config.armijo * alpha * slope:
                    rho = candidate.detach().clone()
                    accepted += 1
                    record.update(accepted=True, alpha=alpha, merit_after=after, status="accepted")
                    break
            record["backtracking"] = trials
            sync_wavelets()
            if callback is not None:
                # JSON roundtrip keeps callbacks from mutating solver diagnostics.
                callback(json.loads(json.dumps({**record, "work": work}, allow_nan=False)))
            if not record["accepted"]:
                record["status"] = "backtracking_failed"
                return finish("backtracking_failed")
        return finish("complete")
    except _BudgetStop as error:
        if trace:
            trace[-1].setdefault("status", str(error))
        return finish(str(error))
    except Exception as error:
        # Caller can persist the diagnostic even when physics/prior construction
        # fails. The source model has never been updated by this solver.
        sync_wavelets()
        error.solver_work = dict(work)
        error.solver_trace = trace
        raise


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reserve_run(output, exp):
    output = Path(output).expanduser().resolve()
    if not output.is_relative_to(ROOT) or not exp or Path(exp).name != exp or exp in (".", ".."):
        raise ValueError("fresh output must be inside density_mm and exp must be one directory name")
    output.mkdir(parents=True, exist_ok=True)
    result = output / exp
    result.mkdir(exist_ok=False)
    return result


def validate_audit(checks):
    if set(checks) != {"pre", "post"} or not all(checks[scan].get("passed") is True for scan in checks):
        raise RuntimeError("zero-step registered-output audit failed; do not widen tolerances")


def write_solver_checkpoint(path, model, result, *, source_checkpoint, source_checkpoint_sha256,
                            frequency_t=0.,frequency_evidence=None,execution=None,local_factor=1.):
    """Deliberately omit the source restore protocol's iter/optim_state keys."""
    if frequency_evidence is not None and frequency_evidence.get("frequency_t") != frequency_t:
        raise ValueError("solver checkpoint frequency evidence/t mismatch")
    torch.save({"variant": "phase_firstorder", "source_iteration": SOURCE_ITERATION,
                "resumable_by_original_adam": False, "source_checkpoint": str(source_checkpoint),
                "source_checkpoint_sha256": source_checkpoint_sha256,
                "source_optimizer_state": "unchanged in source checkpoint; deliberately not embedded",
                "frequency_t":float(frequency_t),"frequency_evidence":frequency_evidence,
                "local_factor":float(local_factor),
                **phase_protocol(result.work.get("phase_curvature_mode","full")),
                "cg_steps":96,"true_residual_diagnostics":True,
                "execution":execution,
                **wavelet_protocol(result.work["wavelet_mode"]),
                "solver_counters": result.work, "accepted_outer_steps": result.accepted_steps,
                "model_state": model.state_dict()}, path)


def validate_saved_solver(path,after,result,frequency_t,frequency_evidence,execution,local_factor=1.):
    """Verify the real saved artifact in the numerical child; no Adam resume keys."""
    checkpoint = torch.load(path,map_location="cpu",weights_only=True)
    phase=phase_protocol(result.work.get("phase_curvature_mode","full"))
    if (checkpoint.get("variant") != "phase_firstorder" or checkpoint.get("resumable_by_original_adam") is not False
            or {"iter","optim_state"} & set(checkpoint) or checkpoint.get("source_iteration") != 4000
            or checkpoint.get("frequency_t") != frequency_t or checkpoint.get("frequency_evidence") != frequency_evidence
            or checkpoint.get("local_factor") != local_factor
            or checkpoint.get("cg_steps") != 96 or checkpoint.get("true_residual_diagnostics") is not True
            or checkpoint.get("execution") != execution
            or checkpoint.get("solver_counters") != result.work or checkpoint.get("accepted_outer_steps") != result.accepted_steps
            or checkpoint.get("wavelet_mode") != "uniform8"
            or any(checkpoint.get(k) != v for k,v in phase.items())):
        raise ValueError("actual nonresumable frequency checkpoint schema/parameters differ")
    observed = {}
    for name,tensor in checkpoint["model_state"].items():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("nonfinite saved solver model")
        array=tensor.detach().cpu().contiguous().numpy()
        observed[name]=hashlib.sha256(memoryview(array).cast("B")).hexdigest()
    if observed != after["parameter_sha256"]:
        raise ValueError("actual saved solver model does not match final seven-tensor state")
    return {"path":str(path),"sha256":sha256_file(path),"passed":True,"all_seven_model_hashes_match":True,
            "resumable_by_original_adam":False,"frequency_t":float(frequency_t),"local_factor":float(local_factor),
            "cg_steps":96,"true_residual_diagnostics":True,**phase}


def validate_zero_exports(checks):
    validate_audit(checks)
    errors={scan:checks[scan].get("source_array_relative_l2",float("nan")) for scan in ("pre","post")}
    if not all(math.isfinite(v) and 0 <= v <= 1e-5 for v in errors.values()):
        raise RuntimeError("zero-source complex reproduction exceeded fixed1e-5 tolerance")
    return {"passed":True,"relative_l2_tolerance":1e-5,"relative_l2":errors,
            "exact_equal":{scan:value == 0. for scan,value in errors.items()}}


def _load_base():
    path = ROOT.parent / "nonlocal_autoresearch/runner.py"
    spec = importlib.util.spec_from_file_location("_density_mm_frozen_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_prior_kwargs(trainer, shift=None, *, wavelet_mode="single"):
    """Resolve the source's active strengths/modes without calling train()."""
    from losses.edge_tv import resolve_edge_kappa
    loss = trainer.config["loss"]
    unsupported = ("lambda_private_sp", "lambda_dtv", "lambda_vtv", "lambda_sub",
                   "lambda_nl", "lambda_nl_plain", "lambda_nl_contrast")
    if any(float(loss.get(key, 0)) != 0 for key in unsupported):
        raise NotImplementedError("driver does not silently omit another active source prior")
    if not loss.get("edge_tv", False) or loss.get("edge_tv_kappa_source") != "zf" or int(loss.get("edge_tv_kappa_refresh_iter", 0)):
        raise NotImplementedError("require fixed ZF-derived source edge-TV kappas")
    geo = float(loss["tv_weight"] if "tv_weight" in loss and "lambda_geo" not in loss else loss.get("lambda_geo", 5e-4))
    tv = [float(loss.get("lambda_geo_plain", geo)), float(loss.get("lambda_geo_contrast", geo))]
    wavelet = float(loss.get("lambda_wav", 0))
    wave = [float(loss.get("lambda_wav_plain", wavelet)), float(loss.get("lambda_wav_contrast", wavelet))]
    q = float(loss.get("edge_tv_kappa_q", .75))
    kappas = [resolve_edge_kappa(trainer.zf_p, float(loss.get("edge_tv_kappa_q_plain", q))),
              resolve_edge_kappa(trainer.zf_c, float(loss.get("edge_tv_kappa_q_contrast", q)))]
    protocol = wavelet_protocol(wavelet_mode, shift)
    result = dict(lambda_tv=tv, lambda_phase=float(loss.get("lambda_ptv", 0)), lambda_wavelet=wave,
                kappa=kappas, tv_eps=float(loss.get("tv_eps", 0)), phase_eps=float(loss.get("tv_eps", 0)),
                w_floor=float(loss.get("edge_tv_wfloor", .1)),
                tv_vector=bool(loss.get("tv_vector", False)), wavelet=str(loss.get("wav_name", "db2")),
                wavelet_levels=int(loss.get("wav_levels", 3)), wavelet_on=str(loss.get("wav_on", "complex")),
                wavelet_eps=float(loss.get("wav_eps", 0)))
    if wavelet_mode == "single":
        result["wavelet_shift"] = tuple(protocol["fixed_wavelet_shift"])
    return result


def _validate_review(path):
    gate = json.loads(Path(path).read_text())
    if gate.get("status") != "pass" or gate.get("unresolved_findings") != 0:
        raise ValueError("independent clean review gate is required")
    for name in PRODUCTION:
        file = ROOT / name
        if gate.get("code_sha256", {}).get(str(file)) != sha256_file(file):
            raise ValueError(f"reviewed source hash is missing/stale: {file}")
    for file, digest in gate["code_sha256"].items():
        if sha256_file(file) != digest:
            raise ValueError(f"additional reviewed source hash is stale: {file}")
    return gate


def prepare_runtime_cuda(cfg):
    """Require the source CUDA renderer and record the sole runtime override."""
    if cfg.get("voxelizer", {}).get("type", "tile_cuda") != "tile_cuda":
        raise ValueError("clinical density runs require voxelizer.type=tile_cuda")
    mode = cfg.setdefault("mode", {})
    override = {"mode.strict_cuda": {"source_present": "strict_cuda" in mode,
                                     "source": mode.get("strict_cuda"), "effective": True}}
    mode["strict_cuda"] = True
    return override


def construct_strict_cuda_trainer(cfg, device, base, cuda_gate, trainer_factory):
    """Install the gated prebuilt binary before the normal trainer constructor.

    The source TileVoxelizer normally calls cpp_extension.load with its own
    source build directory. A cache environment variable cannot prevent those
    writes. Populating the *normal* module's process-local extension reference
    bypasses that branch; strict mode then forbids fallback to another backend.
    """
    from gpu_preflight import extension_path, _load_module
    if cfg.get("voxelizer", {}).get("type", "tile_cuda") != "tile_cuda" or cfg.get("mode", {}).get("strict_cuda") is not True:
        raise ValueError("strict tile_cuda backend must be configured before construction")
    recorded = cuda_gate.get("invariants", {}).get("prebuilt_extension")
    if not isinstance(recorded, dict) or not recorded.get("path") or not recorded.get("sha256"):
        raise ValueError("CUDA gate has no verified prebuilt extension")
    binary = extension_path().resolve()
    expected_source = (base.TRAINER / "paired_gs/gaussian/tile_voxelizer.py").resolve()
    expected_build = expected_source.parent / "cuda_ext/build"
    if binary.parent != expected_build or Path(recorded["path"]).resolve() != binary or sha256_file(binary) != recorded["sha256"]:
        raise ValueError("prebuilt extension path/source hash does not match the CUDA gate")
    tile = importlib.import_module("gaussian.tile_voxelizer")
    if not getattr(tile, "__file__", None) or Path(tile.__file__).resolve() != expected_source:
        raise ValueError("normal Gaussian tile module is not the exact frozen C source")
    extension = _load_module(binary.stem, binary)
    old_extension, old_error = tile._cuda_ext, tile._cuda_load_error
    tile._cuda_ext, tile._cuda_load_error = extension, None
    trainer = None
    try:
        trainer = trainer_factory(cfg, device=device)
        voxelizer = trainer.voxelizer
        if (voxelizer.use_cuda is not True or voxelizer.strict_cuda is not True
                or voxelizer.backend != "tile_cuda" or bool(getattr(voxelizer, "use_cupy", False))):
            raise RuntimeError("clinical trainer selected a fallback/non-strict CUDA backend")
        return trainer, {"prebuilt_extension": {"path": str(binary), "sha256": recorded["sha256"]},
                         "normal_tile_source": str(expected_source),
                         "backend": voxelizer.backend, "strict_cuda": voxelizer.strict_cuda,
                         "use_cuda": voxelizer.use_cuda, "compiler_branch_bypassed": True}
    except Exception:
        tile._cuda_ext, tile._cuda_load_error = old_extension, old_error
        if trainer is not None and getattr(trainer, "writer", None) is not None:
            trainer.writer.close()
        raise


def main(argv=None):
    runtime=numerical_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("zero", "frequency_mm"), required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--review-gate", type=Path, required=True)
    parser.add_argument("--cuda-operator-gate", type=Path, required=True)
    parser.add_argument("--zero-audit-gate", type=Path)
    parser.add_argument("--outer-steps", type=int, default=5)
    parser.add_argument("--cg-steps", type=int, default=96)
    parser.add_argument("--true-residual-diagnostics",action="store_true",required=True)
    parser.add_argument("--max-vjp-pairs", type=int, default=500)
    parser.add_argument("--max-render-pairs", type=int, default=700)
    parser.add_argument("--damping-rel", type=float, default=1e-3)
    parser.add_argument("--local-factor", type=float, default=1.)
    parser.add_argument("--wavelet-mode", choices=("single", "uniform8"), default="uniform8")
    parser.add_argument("--wavelet-shift", type=int, nargs=3, default=None)
    parser.add_argument("--frequency-t",type=float,choices=(0.,.5,1.),default=1.)
    parser.add_argument("--phase-curvature-mode",choices=("first_order",),default="first_order")
    args = parser.parse_args(argv)
    # A separate CPU reservation supervisor must select an idle/lightly-loaded
    # compatible physical GPU and set its initial CVD before starting this CLI.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible == "-1":
        parser.error("launch through a GPU reservation supervisor with exactly one visible device")
    try:
        wavelet = strength_protocol(args.frequency_t,args.local_factor,args.wavelet_mode,args.wavelet_shift)
        if args.cg_steps != 96:
            raise ValueError("precision controls and solve require cg_steps96")
        if args.mode == "frequency_mm" and (args.local_factor != 1. or args.outer_steps != 5
                or args.max_vjp_pairs != 500 or args.max_render_pairs != 700 or args.damping_rel != .001):
            raise ValueError("formal phase-first-order solve requires t1/factor1 and fixed5x96 budget")
        review = _validate_review(args.review_gate)
        cuda_gate = json.loads(args.cuda_operator_gate.read_text())
        if (cuda_gate.get("passed") is not True or cuda_gate.get("code_sha256") != review["code_sha256"]
                or any(cuda_gate.get(k) != v for k,v in wavelet.items())):
            raise ValueError("fresh tiny-CUDA adjoint/superposition gate is required")
        if args.mode == "frequency_mm":
            if args.zero_audit_gate is None:
                raise ValueError("optimization requires a successful zero-step clinical audit gate")
            zero_gate = json.loads(args.zero_audit_gate.read_text())
            if (zero_gate.get("passed") is not True or zero_gate.get("code_sha256") != review["code_sha256"]
                    or any(zero_gate.get(k) != v for k,v in wavelet.items())):
                raise ValueError("clinical zero-step audit gate is stale/missing")
        config = SolverConfig(outer_steps=0 if args.mode == "zero" else args.outer_steps,
                              cg_steps=args.cg_steps, max_vjp_pairs=args.max_vjp_pairs,
                              max_render_pairs=args.max_render_pairs, damping_rel=args.damping_rel,
                              true_residual_diagnostics=True)
        base = _load_base()
        cfg = base.build_config("audit", 0, args.output.resolve() / args.exp, local_factor=args.local_factor, nlambda=0.)
        runtime_cuda_override = prepare_runtime_cuda(cfg)
        cfg["frequency_overrides"]={"frequency_t":args.frequency_t,"mask_source":"original acquired mask_p/mask_c; never fit masks",
                                    "utility_sha256":sha256_file(ROOT/"frequency_weights.py")}
        cfg["inner_precision_overrides"]={"cg_steps":96,"true_residual_diagnostics":True}
        cfg["phase_firstorder_overrides"]=phase_protocol()
        run_dir = reserve_run(args.output, args.exp)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
    execution={"declared_parameters":{**wavelet,"solver_config":dataclasses.asdict(config),"local_factor":args.local_factor,"nlambda":0.},
               "child_command":list(sys.orig_argv),"source_code_sha256":review["code_sha256"]}
    provenance = {"status": "starting", "mode": args.mode, "method_variant": "phase_firstorder", "exp": args.exp,
                  "source_iteration": SOURCE_ITERATION, "solver_config": dataclasses.asdict(config),
                  "code_sha256": review["code_sha256"], "source_checkpoint": str(base.CHECKPOINT),
                  "source_checkpoint_sha256": sha256_file(base.CHECKPOINT),
                  "source_config_sha256": sha256_file(base.BASE_CONFIG),
                  "runtime_overrides": runtime_cuda_override,
                  **wavelet, "visible_device": visible,"numerical_runtime":runtime,"execution":execution,
                  "claim_boundary": "phase first-order/quasi-Newton; true frozen-merit Armijo descent only; no full MM or cross-refresh/stochastic-objective claim",
                  "normalization_protocol": "original retrospective scalar normalization retained; GT and lesion masks are scoring only"}
    destination = run_dir / "provenance.json"
    def save():
        destination.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    save()
    start = time.perf_counter()
    trainer = None
    try:
        import yaml
        (run_dir / "actual_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        os.environ.update(RETRO_TOPIC=str(base.TOPIC), RETRO_TRAINER=str(base.TRAINER), PYTHONDONTWRITEBYTECODE="1")
        sys.dont_write_bytecode = True
        # All generated caches stay within the isolated run.
        for key in ("TMPDIR", "TORCH_EXTENSIONS_DIR", "CUPY_CACHE_DIR", "CUDA_CACHE_PATH", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
            directory = run_dir / "cache" / key.lower()
            directory.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(directory)
        sys.path[:0] = [str(base.TRAINER / "paired_gs"), str(base.TRAINER)]
        from seed import set_seed
        from trainers.joint_trainer import JointTrainer
        from trainers.resume_protocol import restore_checkpoint
        from density_operator import DensityLinearOperator
        set_seed(42, deterministic=False)
        checkpoint = torch.load(base.CHECKPOINT, map_location="cpu", weights_only=True)
        if checkpoint.get("iter") != SOURCE_ITERATION:
            raise ValueError("source checkpoint is not iteration 4000")
        trainer, provenance["runtime_cuda"] = construct_strict_cuda_trainer(
            cfg, torch.device("cuda", 0), base, cuda_gate, JointTrainer)
        provenance["resume"] = restore_checkpoint(trainer.model, trainer.optimizer, trainer.scheduler,
                                                   checkpoint, expected_iteration=SOURCE_ITERATION)
        del checkpoint
        before = base.state_snapshot(trainer)
        provenance["before"] = before
        operator = DensityLinearOperator(trainer,frequency_t=args.frequency_t)
        provenance["frequency_evidence"]=operator.save_frequency_evidence(run_dir)
        provenance["frequency_array_check"]=validate_saved_frequency_evidence(run_dir,provenance["frequency_evidence"],args.frequency_t,operator)
        save()
        prior_kwargs = source_prior_kwargs(trainer, args.wavelet_shift, wavelet_mode=args.wavelet_mode) if args.mode != "zero" else {}
        with (run_dir / "outer_trace.jsonl").open("w", buffering=1) as stream:
            def trace(record):
                stream.write(json.dumps(factor_trace(record,args.local_factor), allow_nan=False) + "\n")
            result = optimize_density(operator, prior_kwargs=prior_kwargs, config=config, wavelet_mode=args.wavelet_mode,
                                       phase_curvature_mode=args.phase_curvature_mode,
                                       edge_weight=float(cfg["loss"].get("lambda_edge", 0)),
                                       edge_p=float(cfg["loss"].get("edge_p", .75)),
                                       edge_kappa=float(cfg["loss"].get("edge_kappa", 10)), callback=trace)
        result.trace=[factor_trace(row,args.local_factor) for row in result.trace]
        # Install only the accepted density; original Parameter objects and all
        # optimizer states remain intact. Geometry is never written.
        with torch.no_grad():
            for channel, suffix in enumerate(("", "_contrast")):
                getattr(trainer.model, "density_real" + suffix).copy_(result.density[channel].real)
                getattr(trainer.model, "density_imag" + suffix).copy_(result.density[channel].imag)
        after = base.state_snapshot(trainer)
        if before["geometry_sha256"] != after["geometry_sha256"] or before["adam"] != after["adam"]:
            raise RuntimeError("solver changed geometry or original Adam state")
        if args.mode == "zero" and before != after:
            raise RuntimeError("zero-step audit changed source model/state")
        artifact = run_dir / "density_solver_nonresumable.pth"
        write_solver_checkpoint(artifact, trainer.model, result, source_checkpoint=base.CHECKPOINT,
                                source_checkpoint_sha256=provenance["source_checkpoint_sha256"],frequency_t=args.frequency_t,
                                frequency_evidence=provenance["frequency_evidence"],execution=execution,local_factor=args.local_factor)
        provenance["solver_checkpoint"]=validate_saved_solver(artifact,after,result,args.frequency_t,provenance["frequency_evidence"],execution,args.local_factor)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        rows, checks = base.export_and_score(trainer, run_dir, "audit" if args.mode == "zero" else args.mode, elapsed)
        if args.mode == "zero":
            provenance["zero_export_check"]=validate_zero_exports(checks)
        if sha256_file(base.CHECKPOINT) != provenance["source_checkpoint_sha256"] or sha256_file(base.BASE_CONFIG) != provenance["source_config_sha256"]:
            raise RuntimeError("source checkpoint/config changed during run")
        execution_status = ("complete" if args.mode == "zero" or (result.accepted_steps > 0 and result.status == "complete")
                            else "no_update" if result.accepted_steps == 0 else "partial")
        provenance.update(status=execution_status, solver_status=result.status, after=after,
                          solver_counters=result.work, operator_counters=operator.counters,
                          accepted_outer_steps=result.accepted_steps, trace=result.trace,
                          original_adam_unchanged=True, geometry_unchanged=True,
                          total_elapsed_seconds=time.perf_counter()-start,
                          cuda_memory_bytes={"peak_allocated": torch.cuda.max_memory_allocated(),
                                             "peak_reserved": torch.cuda.max_memory_reserved()})
        (run_dir / "metrics.json").write_text(json.dumps({"method_variant": "phase_firstorder", "mode": args.mode,
              **wavelet,
              "solver_config":dataclasses.asdict(config),
              "numerical_runtime":runtime,"execution":execution,"frequency_evidence":provenance["frequency_evidence"],
              "frequency_array_check":provenance["frequency_array_check"],
              "solver_checkpoint":provenance["solver_checkpoint"],"zero_export_check":provenance.get("zero_export_check"),
              "source_iteration": SOURCE_ITERATION, "rows": rows, "audit_checks": checks,
              "solver_status": result.status, "solver_counters": result.work,
              "accepted_outer_steps": result.accepted_steps, "provenance": str(destination)}, indent=2, allow_nan=False)+"\n")
        save()
    except Exception as error:
        provenance.update(status="failed", error=repr(error), total_elapsed_seconds=time.perf_counter()-start,
                          failure_work=getattr(error, "solver_work", None), failure_trace=getattr(error, "solver_trace", None))
        save()
        raise
    finally:
        if trainer is not None and trainer.writer is not None:
            trainer.writer.close()
    print(str(run_dir))


if __name__ == "__main__":
    main()
