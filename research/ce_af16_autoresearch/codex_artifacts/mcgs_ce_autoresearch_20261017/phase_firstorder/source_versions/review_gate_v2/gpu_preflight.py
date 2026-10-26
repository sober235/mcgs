#!/usr/bin/env python3
"""Tiny source-CUDA physics measurements, never a reconstruction quality claim."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import torch

from density_operator import DensityLinearOperator

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "nonlocal_autoresearch"
PAIRED = BASE / "trainer/paired_gs"
DEFAULT_TOLERANCE = 2e-5
MAX_TOLERANCE = 1e-4
TINY_SHAPE = (36, 36, 36)  # Even layout; source center_fraction=.05 contains8 central voxels.


def _norm(value):
    if isinstance(value, (tuple, list)):
        return math.sqrt(sum(_norm(part) ** 2 for part in value))
    value = value.detach()
    square = value.real.double().square().sum()
    if value.is_complex():
        square = square + value.imag.double().square().sum()
    return math.sqrt(float(square))


def _inner(left, right):
    if isinstance(left, (tuple, list)):
        return sum(_inner(a, b) for a, b in zip(left, right))
    return complex(torch.vdot(left.detach().flatten().to(torch.complex128),
                              right.detach().flatten().to(torch.complex128)).item())


def _finite(label, value):
    parts = value if isinstance(value, (tuple, list)) else (value,)
    if not all(bool(torch.isfinite(part).all()) for part in parts):
        raise RuntimeError(f"{label} contains nonfinite values")


def _relative(actual, expected):
    if isinstance(actual, (tuple, list)):
        difference = tuple(a - b for a, b in zip(actual, expected))
    else:
        difference = actual - expected
    error = _norm(difference) / max(_norm(actual), _norm(expected), 1e-30)
    if not math.isfinite(error):
        raise RuntimeError("physics comparison is nonfinite")
    return error


def check_physics(operator, *, tolerance=DEFAULT_TOLERANCE, seed=1729):
    """Measure complex B and DC-normal identities using first-order VJPs only.

    This generic routine is CPU-testable. Actual CUDA is only selected by the
    separately gated tiny factory. The tolerance cannot be disabled or raised
    above 1e-4. Work is measured from actual operator counters plus normal VJPs.
    """
    if not math.isfinite(tolerance) or not 0 < tolerance <= MAX_TOLERANCE:
        raise ValueError("tolerance must be finite, positive, and at most 1e-4")
    template = operator.initial_density
    generator = torch.Generator(device=template.device).manual_seed(seed)

    def random_like(value):
        return torch.complex(torch.randn(value.shape, device=value.device, dtype=value.real.dtype, generator=generator),
                             torch.randn(value.shape, device=value.device, dtype=value.real.dtype, generator=generator))

    before = dict(operator.counters)
    a, b = 1.2 - .7j, -.4 + .9j
    rho, sigma = random_like(template), random_like(template)
    with torch.no_grad():
        first, second = operator.render(rho), operator.render(sigma)
        combined = operator.render(a * rho + b * sigma)
        _finite("render", (*first, *second, *combined))
        superposition = _relative(combined, tuple(a * x + b * y for x, y in zip(first, second)))
        images = tuple(random_like(image) for image in first)
    adjoint = operator.adjoint(*images)
    _finite("adjoint", adjoint)
    left, right = _inner(first, images), _inner(rho, adjoint)
    adjoint_error = abs(left - right) / max(abs(left), abs(right), 1e-30)
    normal_calls = 0

    def normal(value):
        nonlocal normal_calls
        with torch.enable_grad():
            external = value.detach().clone().requires_grad_(True)
            pre, post = operator.render(external)
            loss = operator.data_loss(pre, post, zero_target=True) * .5
            _finite("zero-target DC", loss)
            normal_calls += 1
            product, = torch.autograd.grad(loss, external, create_graph=False, retain_graph=False)
        _finite("normal", product)
        return product.detach()

    h_rho, h_sigma = normal(rho), normal(sigma)
    h_combined = normal(a * rho + b * sigma)
    normal_linearity = _relative(h_combined, a * h_rho + b * h_sigma)
    cross_left, cross_right = _inner(rho, h_sigma), _inner(h_rho, sigma)
    hermitian = abs(cross_left - cross_right) / max(abs(cross_left), abs(cross_right), 1e-30)
    curvature = _inner(rho, h_rho)
    if not math.isfinite(curvature.real) or not math.isfinite(curvature.imag) or curvature.real <= 0:
        raise RuntimeError("normal positive-curvature check failed")
    hermitian = max(hermitian, abs(curvature.imag) / max(abs(curvature.real), 1e-30))
    errors = dict(superposition=superposition, adjoint=adjoint_error,
                  normal_linearity=normal_linearity, normal_hermitian=hermitian)
    for name, error in errors.items():
        if not math.isfinite(error) or error > tolerance:
            raise RuntimeError(f"{name} relative error {error!r} exceeds tolerance {tolerance}")
    counters = {name: operator.counters[name] - count for name, count in before.items()}
    counters.update(normal_applications=normal_calls, normal_vjp_calls=normal_calls,
                    total_vjp_calls=normal_calls + counters["vjp_calls"])
    return {"passed": True, "tolerance": tolerance, "seed": seed, "errors": errors,
            "normal_curvature": {"real": curvature.real, "imag": curvature.imag},
            "dtype": str(template.dtype), "device": str(template.device), "work": counters}


def _load_source_functions(trainer):
    namespace = {"torch": torch, "fft": torch.fft}
    for file, class_name, names in ((PAIRED / "data/transforms.py", None, {"fft3c"}),
                                     (PAIRED / "trainers/joint_trainer.py", "JointTrainer",
                                      {"_render", "_kspace_loss", "_build_freq_weight"})):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        body = tree.body if class_name is None else next(node.body for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == class_name)
        selected = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
        if len(selected) != len(names):
            raise RuntimeError(f"source method set changed in {file}")
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(file), "exec"), namespace)
    for name in ("_render", "_kspace_loss", "_build_freq_weight"):
        setattr(trainer, name, types.MethodType(namespace[name], trainer))


def check_prior_mode(images, *, wavelet_mode="single", wavelet_shift=None,local_factor=1.):
    """Exercise the selected prior against independent original-source phases.

    This routine is CPU-testable and runs on actual toy CUDA images in main().
    TV/phase strengths are zero here to isolate the new wavelet averaging;
    full TV/phase counting and normal-system checks have separate CPU fixtures.
    """
    from runner import wavelet_protocol
    from surrogates import build_frozen_prior
    protocol = wavelet_protocol(wavelet_mode, wavelet_shift)
    if isinstance(local_factor,bool) or local_factor not in (1.,.75,.5):
        raise ValueError("tiny local_factor must be1/.75/.5")
    strengths=(.12*local_factor,.1*local_factor)
    if wavelet_mode == "uniform8":
        from cycle_average import build_cycle_average_prior
        factory, options = build_cycle_average_prior, {}
        shifts = tuple(itertools.product((0, 1), repeat=3))
    else:
        factory = build_frozen_prior
        options = {"wavelet_shift": tuple(protocol["fixed_wavelet_shift"])}
        shifts = (tuple(protocol["fixed_wavelet_shift"]),)
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] != 2 or not images.is_complex():
        raise ValueError("prior check requires paired complex images")
    x = images.detach().clone().requires_grad_(True)
    prior = factory(x.detach(), lambda_tv=0., lambda_phase=0., lambda_wavelet=strengths, kappa=1., **options)
    actual = prior.merit(x)
    actual_gradient, = torch.autograd.grad(actual, x)
    quadratic = prior.quadratic(x)
    quadratic_gradient, = torch.autograd.grad(quadratic, x)
    source = _load_module("_density_mm_independent_source_wavelet", PAIRED / "losses/wavelet_l1.py")
    reference_x = images.detach().clone().requires_grad_(True)
    expected = sum(source.wavelet_l1([reference_x[0], reference_x[1]], wavelet="db4", levels=1,
                                     eps=0., shifts=shift, on="complex", weights=strengths)
                   for shift in shifts) / len(shifts)
    expected_gradient, = torch.autograd.grad(expected, reference_x)
    errors = {"value_relative": abs(float(actual.detach()-expected.detach())) / max(abs(float(expected.detach())), 1e-30),
              "gradient_relative": _relative(actual_gradient, expected_gradient),
              "quadratic_gradient_relative": _relative(quadratic_gradient, expected_gradient)}
    if not all(math.isfinite(value) and 0 <= value <= DEFAULT_TOLERANCE for value in errors.values()):
        raise RuntimeError(f"selected wavelet prior failed independent source checks: {errors}")
    return {"passed": True, **protocol, "tolerance": DEFAULT_TOLERANCE, "errors": errors,
            "local_factor":float(local_factor),"lambda_tv":[0.,0.],"lambda_phase":[0.,0.],"lambda_wavelet":list(strengths),
            "tested_terms":["wavelet_mean8"] if wavelet_mode == "uniform8" else ["wavelet_single"],"nonzero_TV_phase_tested":False,
            "value": float(actual.detach()), "source_expected_value": float(expected.detach()),
            "wavelet_transform_batches": prior.wavelet_transform_batches,
            "reference_wavelet_transform_batches": len(shifts), "device": str(images.device)}


def check_phase_firstorder(operator, *, local_factor=1.):
    """Actual source phase derivative plus physical reduced-CG execution.

    Main calls this on the strict CUDA toy operator. CPU tests exercise the
    identical function with explicit linear physics, without replacing losses.
    The original full helper is used independently to check the same anchor eta.
    """
    import dataclasses
    import importlib
    from runner import numerical_runtime, phase_protocol, SolverConfig, optimize_density
    from surrogates import build_frozen_prior
    numerical_runtime()
    if isinstance(local_factor,bool) or local_factor != 1.:
        raise ValueError("phase tiny requires selected factor1")
    initial=operator.initial_density.detach().clone()
    before=dict(operator.counters)
    images=torch.stack(operator.render(initial)).detach()
    support=torch.stack(operator.structural_support())
    phase=build_frozen_prior(images,lambda_tv=0.,lambda_phase=.03,lambda_wavelet=0.,
                             kappa=(.6,.8),tv_eps=.04,phase_eps=.04,structural_support=support)
    package_name="_phase_firstorder_tiny_source_losses"
    if package_name not in sys.modules:
        package=types.ModuleType(package_name);package.__path__=[str(PAIRED/"losses")]
        sys.modules[package_name]=package
    source=importlib.import_module(package_name+".phase_tv")
    point=images.clone().requires_grad_(True)
    actual=phase.merit(point)
    actual_gradient,=torch.autograd.grad(actual,point)
    reference=images.clone().requires_grad_(True)
    expected=sum(.03*source.phase_tv(v,k,0.,.04) for v,k in zip(reference,(.6,.8)))
    expected_gradient,=torch.autograd.grad(expected,reference)
    error=_relative(actual_gradient,expected_gradient)
    value_error=abs(float(actual-expected))/max(abs(float(expected)),1e-30)
    if error>DEFAULT_TOLERANCE or not math.isfinite(value_error) or value_error>DEFAULT_TOLERANCE or _norm(actual_gradient)<=0:
        raise RuntimeError("actual nonzero phase gradient/source parity failed")
    # A deterministic supported image direction directly exercises CUDA tensor
    # full/reduced quadratic branches, independently of role metadata.
    index=torch.arange(images.numel(),dtype=images.real.dtype,device=images.device).reshape_as(images.real)
    direction=torch.complex(torch.sin(index*.13),torch.cos(index*.17))*support
    full_point=direction.detach().requires_grad_(True)
    full_gradient,=torch.autograd.grad(phase.quadratic(full_point),full_point)
    reduced_point=direction.detach().requires_grad_(True)
    reduced_gradient,=torch.autograd.grad(phase.quadratic(reduced_point,include_phase=False),reduced_point)
    if _norm(full_gradient)<=0 or torch.count_nonzero(reduced_gradient)!=0:
        raise RuntimeError("phase curvature did not change from full positive form to zero")
    source_delta={k:operator.counters[k]-before.get(k,0) for k in operator.counters}
    settings=dict(lambda_tv=(.1,.1),lambda_phase=.03,lambda_wavelet=(.12,.1),kappa=(.6,.8),
                  tv_eps=.04,phase_eps=.04,w_floor=0.)
    cfg=SolverConfig(outer_steps=2,cg_steps=4,true_residual_diagnostics=True,
                     max_vjp_pairs=40,max_render_pairs=100,max_merit_forwards=24)
    result=optimize_density(operator,prior_kwargs=settings,config=cfg,wavelet_mode="uniform8",
                            phase_curvature_mode="first_order")
    control_cfg=dataclasses.replace(cfg,outer_steps=1,cg_steps=1)
    control=optimize_density(operator,prior_kwargs=settings,config=control_cfg,wavelet_mode="uniform8")
    if result.status!="complete" or result.accepted_steps!=2 or control.status!="complete" or control.accepted_steps!=1:
        raise RuntimeError("actual tiny phase-first-order/control solver did not complete")
    eta=result.trace[0]["damping"];reference_eta=control.trace[0]["damping"]
    eta_error=abs(eta-reference_eta)/max(abs(eta),abs(reference_eta),1e-30)
    if eta_error>DEFAULT_TOLERANCE or not torch.equal(operator.initial_density,initial):
        raise RuntimeError("phase first-order changed original full-probe eta or source density")
    after=dict(operator.counters)
    if after["render_pairs"]-before["render_pairs"] != source_delta["render_pairs"]+result.work["render_pairs"]+control.work["render_pairs"]:
        raise RuntimeError("tiny phase solver hid actual renderer work")
    return {"passed":True,**phase_protocol(),"local_factor":1.,"tolerance":DEFAULT_TOLERANCE,
            "lambda_tv":[.1,.1],"lambda_phase":[.03,.03],"lambda_wavelet":[.12,.1],
            "phase_gradient_norm":_norm(actual_gradient),"source_phase_gradient_relative_error":error,
            "source_phase_value_relative_error":value_error,
            "phase_full_probe_gradient_norm":_norm(full_gradient),"phase_reduced_probe_gradient_norm":_norm(reduced_gradient),
            "full_control_eta_relative_error":eta_error,"phase_image_vjp_evaluations":4,
            "source_check_operator_delta":source_delta,"operator_counters_before":before,"operator_counters_after":after,
            "solver_config":dataclasses.asdict(cfg),"solver_status":result.status,
            "accepted_outer_steps":result.accepted_steps,"solver_counters":result.work,"trace":result.trace,
            "full_control_solver_config":dataclasses.asdict(control_cfg),"full_control_solver_status":control.status,
            "full_control_solver_counters":control.work,"full_control_trace":control.trace,
            "source_density_unchanged":True,"device":str(images.device)}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def extension_path():
    """Resolve the already-built source-hash-named extension; never build it."""
    folder = PAIRED / "gaussian/cuda_ext"
    pieces = [hashlib.md5((folder / name).read_bytes()).hexdigest()[:8]
              for name in ("tile_voxelize_cuda.cpp", "tile_voxelize_cuda_kernel.cu")]
    name = "tile_voxelize_cuda_" + "_".join(pieces)
    path = folder / "build" / (name + ".so")
    if not path.is_file():
        raise RuntimeError(f"required prebuilt extension does not exist: {path}; compilation is forbidden")
    return path


def check_t0_source_parity(operator):
    """Compare the exact t0 branch with verbatim original source data physics.

    Frequency source tensors are byte-identical. CUDA VJP atomics use the
    existing2e-5 tiny physics bound, without requiring bitwise repeated VJPs.
    """
    control = DensityLinearOperator(operator.trainer, frequency_t=0.)
    rho = control.initial_density.detach().clone().requires_grad_(True)
    images = control.render(rho)
    def source_loss(zero):
        total = 0.
        for image, (csm, measured, fit_mask, norm), weight in zip(images, control._dc,
                                                                control._source_frequency_weights):
            target = measured.new_zeros(()) if zero else measured
            total = total + control.trainer._kspace_loss(image, csm, target, fit_mask, weight, norm=norm)
        return total
    source, actual = source_loss(False), control.data_loss(*images)
    zero_source, zero_actual = source_loss(True), control.data_loss(*images, zero_target=True)
    actual_normal, = torch.autograd.grad(zero_actual / 2, rho, retain_graph=True)
    source_normal, = torch.autograd.grad(zero_source / 2, rho)
    normal_difference = _relative(actual_normal, source_normal)
    exact = all(a is b and torch.equal(a, b) for a, b in
                zip(control._source_frequency_weights, control._frequency_weights))
    result = {"weights_exact_equal": exact,
              "source_loss_absolute_difference": abs(float((source-actual).detach())),
              "zero_loss_absolute_difference": abs(float((zero_source-zero_actual).detach())),
              "normal_relative_difference": normal_difference,
              "normal_relative_tolerance": DEFAULT_TOLERANCE,
              "frequency_summary": control.frequency_summary()}
    if (not exact or result["source_loss_absolute_difference"] != 0.
            or result["zero_loss_absolute_difference"] != 0. or normal_difference > DEFAULT_TOLERANCE):
        raise RuntimeError(f"t0 source-weight/data/normal parity failed: {result}")
    return {"passed": True, **result}


def tiny_acquisition_masks(shape, *, device, generator):
    """Keep all three original frequency bands and deterministic fit-only support."""
    acquired = (torch.rand(shape, device=device, generator=generator) > .3).float()
    known_zero = (torch.rand(shape, device=device, generator=generator) > .95).float()
    center = tuple(size // 2 for size in shape)
    acquired[center] = 1.
    acquired[center[0], center[1], center[2] + 3] = 1.  # Source middle band.
    acquired[0, 0, 0] = 1.  # Source high band.
    acquired[0, 0, 1] = 0.; known_zero[0, 0, 1] = 1.  # Definitely fit-only.
    return acquired, torch.maximum(acquired, known_zero)


def tiny_frequency_bands(operator):
    """Actual per-scan source/acquired/fit band counts, not nominal labels."""
    result = {}
    for scan, tag, source, effective, acquired in zip(("pre", "post"), ("p", "c"),
            operator._source_frequency_weights, operator._frequency_weights, operator._acquired_masks):
        levels = torch.unique(source)
        if levels.numel() != 3:
            raise RuntimeError("tiny fixture must exercise actual source center/middle/high weights")
        fit = getattr(operator.trainer, "mask_fit_" + tag).bool()
        bands = [{"weight": float(level), "volume_count": int((source == level).sum()),
                  "acquired_count": int(((source == level) & acquired.bool()).sum()),
                  "fit_count": int(((source == level) & fit).sum())} for level in levels]
        exact = torch.equal(source, effective)
        if (any(row["acquired_count"] <= 0 for row in bands)
                or not bool((fit & ~acquired.bool()).any()) or exact != (operator.frequency_t == 0.)):
            raise RuntimeError("tiny fixture lacks an acquired band/fit-only sample or a nonidentity flattening")
        result[scan] = {"bands": bands, "effective_identity": exact,
                        "fit_only_count": int((fit & ~acquired.bool()).sum())}
    return result


def make_tiny_cuda_operator(*, seed=1729, frequency_t=0.):
    """96 signed-complex Gaussians on 36^3, actual C model and strict tile CUDA.

    Call only after the CPU supervisor's current review/invariant gates. No
    original data/checkpoint is loaded and the extension compiler is never used.
    Returns the source-CUDA operator and a reference-backend operator sharing
    immutable toy geometry, plus metadata.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("tiny CUDA preflight requires exactly one reserved visible CUDA device")
    device = torch.device("cuda", 0)
    package_name = "_density_mm_tiny_gaussian"
    package = types.ModuleType(package_name)
    package.__path__ = [str(PAIRED / "gaussian")]
    sys.modules[package_name] = package
    _load_module(package_name + ".gaussian_model", PAIRED / "gaussian/gaussian_model.py")
    joint = _load_module(package_name + ".joint_gaussian_model", PAIRED / "gaussian/joint_gaussian_model.py")
    tile = _load_module(package_name + ".tile_voxelizer", PAIRED / "gaussian/tile_voxelizer.py")
    binary = extension_path()
    extension = _load_module(binary.stem, binary)
    # This bypasses _load_cuda_ext's build branch entirely. strict_cuda prevents
    # a CuPy or reference fallback from silently masquerading as the CUDA check.
    tile._cuda_ext = extension
    tile._cuda_load_error = None
    generator = torch.Generator(device=device).manual_seed(seed)
    shape, n_points = TINY_SHAPE, 96
    positions = torch.rand((n_points, 3), device=device, generator=generator) * 1.6 - .8
    scales = torch.rand((n_points, 3), device=device, generator=generator) * .08 + .05

    def random_complex(shape):
        return torch.complex(torch.randn(shape, device=device, generator=generator),
                             torch.randn(shape, device=device, generator=generator))

    model = joint.JointGaussianModel3D(n_points, shape, device,
                                      initial_positions=positions,
                                      initial_densities_plain=random_complex((n_points,)),
                                      initial_densities_contrast=random_complex((n_points,)),
                                      initial_scales=scales)
    rotations = torch.randn((n_points, 4), device=device, generator=generator)
    model.rotations = torch.nn.Parameter(rotations / rotations.norm(dim=-1, keepdim=True))
    trainer = types.SimpleNamespace(model=model, volume_shape=shape, device=device,
            config={"adaptive_control": {"enable": False}, "gaussian": {"num_private_points": 0},
                    "loss": {"kspace_freq_adaptive": True, "kspace_high_weight": 2.8,
                             "kspace_high_weight_plain": 2.2, "kspace_center_fraction": .05,
                             "kspace_center_weight": .3, "kspace_mid_weight": 1.}})
    trainer.optimizer = torch.optim.Adam(model.parameters())
    trainer.voxelizer = tile.TileVoxelizer(shape, device=device, use_cuda=True, strict_cuda=True)
    if not trainer.voxelizer.use_cuda or trainer.voxelizer.use_cupy:
        raise RuntimeError("strict source CUDA extension was not selected")
    coordinates = torch.meshgrid(*(torch.linspace(-1, 1, size, device=device) for size in shape), indexing="ij")
    coil = torch.exp(1j * (coordinates[0] + 1.3 * coordinates[1] - .7 * coordinates[2])) * .65
    for tag in ("p", "c"):
        acquired, fit = tiny_acquisition_masks(shape, device=device, generator=generator)
        setattr(trainer, f"mask_{tag}", acquired)
        setattr(trainer, f"mask_fit_{tag}", fit)
        setattr(trainer, f"_dc_norm_{tag}", max(float(acquired.sum()), 1.))
        setattr(trainer, f"csm_fit_{tag}", torch.stack((torch.ones_like(coil), coil)))
        setattr(trainer, f"kspace_fit_{tag}", random_complex((2, *shape)))
    _load_source_functions(trainer)
    reference = copy.copy(trainer)
    reference.voxelizer = tile.TileVoxelizer(shape, device=device, use_cuda=False, strict_cuda=False)
    _load_source_functions(reference)
    metadata = {"num_points": n_points, "volume_shape": list(shape), "coils": 2,
                "extension_path": str(binary), "extension_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "source_root": str(BASE), "seed": seed, "cuda_backend": "prebuilt C tile extension",
                "frequency_t": float(frequency_t),
                "original_acquired_counts": {tag: int(getattr(trainer, f"mask_{tag}").sum()) for tag in ("p", "c")},
                "fit_counts": {tag: int(getattr(trainer, f"mask_fit_{tag}").sum()) for tag in ("p", "c")}}
    operator = DensityLinearOperator(trainer, frequency_t=frequency_t)
    reference_operator = DensityLinearOperator(reference, frequency_t=frequency_t)
    metadata["frequency_bands"] = tiny_frequency_bands(operator)
    return operator, reference_operator, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-gate", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--wavelet-mode", choices=("uniform8",), default="uniform8")
    parser.add_argument("--wavelet-shift", type=int, nargs=3, default=None)
    parser.add_argument("--frequency-t", type=float, choices=(0., .5, 1.), default=1.)
    parser.add_argument("--local-factor",type=float,choices=(1.,.75,.5),default=1.)
    parser.add_argument("--cg-steps",type=int,choices=(96,),default=96)
    parser.add_argument("--true-residual-diagnostics",action="store_true",required=True)
    parser.add_argument("--phase-curvature-mode",choices=("first_order",),default="first_order")
    args = parser.parse_args(argv)
    from runner import numerical_runtime
    observed_runtime = numerical_runtime()
    # Imports only the CPU supervisor; all checks precede CUDA initialization.
    import run_gpu
    hashes = run_gpu.source_hashes()
    review = run_gpu.validate_review(args.review_gate, hashes)
    launch = run_gpu.read(args.launch_manifest)
    protocol = run_gpu.strength_protocol(args.frequency_t,args.local_factor,args.wavelet_mode, args.wavelet_shift)
    run_gpu.require(launch.get("stage") == "tiny" and launch.get("parent_pid") == os.getppid(),
                    "preflight must be a direct child of its reserved CPU supervisor")
    run_gpu.require(launch.get("code_sha256") == hashes
                    and launch.get("initial_visibility") == os.environ.get("CUDA_VISIBLE_DEVICES")
                    == str(launch.get("resource", {}).get("gpu")), "invalid/stale preflight launch")
    run_gpu.require(all(launch.get(k) == v for k,v in protocol.items()), "preflight wavelet protocol differs from reserved request")
    run_gpu.require(launch.get("frequency_t") == args.frequency_t, "preflight frequency_t differs from reserved request")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"passed": False, "status": "starting", "code_sha256": hashes,
              **protocol,
              "frequency_t": args.frequency_t, "runtime": observed_runtime,
              "invariants": launch["invariants"], "review_gate": str(args.review_gate)}
    try:
        operator, reference, metadata = make_tiny_cuda_operator(frequency_t=args.frequency_t)
        result["frequency_evidence"] = operator.save_frequency_evidence(args.output)
        from runner import validate_saved_frequency_evidence
        result["frequency_array_check"] = validate_saved_frequency_evidence(
            args.output, result["frequency_evidence"], args.frequency_t, operator)
        result["reference_frequency_summary"] = reference.frequency_summary()
        if operator.frequency_summary() != reference.frequency_summary():
            raise RuntimeError("tiny CUDA/reference frozen frequency arrays differ")
        result["t0_source_parity"] = check_t0_source_parity(operator)
        original = {name: parameter.detach().clone() for name, parameter in operator.model._parameters.items()}
        identities = dict(operator.model._parameters)
        result.update(check_physics(operator))
        with torch.no_grad():
            actual, expected = operator.render(operator.initial_density), reference.render(reference.initial_density)
            result["reference_forward_error"] = _relative(actual, expected)
        result["prior_checks"] = check_prior_mode(torch.stack(actual), wavelet_mode=args.wavelet_mode,
                                                  wavelet_shift=args.wavelet_shift,local_factor=args.local_factor)
        result["phase_firstorder_checks"] = check_phase_firstorder(operator,local_factor=args.local_factor)
        gradients = tuple(torch.ones_like(image) * (.3 + .7j) for image in actual)
        result["reference_adjoint_error"] = _relative(operator.adjoint(*gradients), reference.adjoint(*gradients))
        for key in ("reference_forward_error", "reference_adjoint_error"):
            if result[key] > DEFAULT_TOLERANCE:
                raise RuntimeError(f"{key} exceeds predeclared {DEFAULT_TOLERANCE}")
        if not all(operator.model._parameters[name] is parameter and torch.equal(parameter, original[name])
                   for name, parameter in identities.items()):
            raise RuntimeError("tiny preflight changed source model parameters")
        result.update(passed=True, status="pass", factory=metadata, parameters_unchanged=True,
                      total_operator_counters=dict(operator.counters))
    except BaseException as error:
        result.update(passed=False, status="failed", error=repr(error))
        raise
    finally:
        run_gpu.write_new(args.output / "preflight.json", result)


if __name__ == "__main__":
    main()
