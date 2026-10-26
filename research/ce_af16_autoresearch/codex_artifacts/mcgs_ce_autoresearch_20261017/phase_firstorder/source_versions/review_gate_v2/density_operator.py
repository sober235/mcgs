"""Fixed-geometry density interface to the existing paired-GS renderer.

This module imports no source trainer, CUDA extension, acquisition, or GT. The
caller supplies an already constructed trainer. Only its existing ``_render``
and ``_kspace_loss`` methods implement rendering and measured-data physics.
Temporary parameter substitution is process-local and restores object identity.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import threading

import torch
from frequency_weights import flatten_acquired_rms


PARAMETER_NAMES = ("positions", "scales", "rotations", "density_real", "density_imag",
                   "density_real_contrast", "density_imag_contrast")
GEOMETRY_NAMES = PARAMETER_NAMES[:3]


class DensityLinearOperator:
    """Complex-linear ``(2,N)`` density rendering with first-order VJP adjoints.

    Channel order is plain, contrast (called pre, post by the prototype API).
    ``adjoint`` uses the Hermitian convention ``<B rho,g> = <rho,B^H g>``.
    The source DC is a sum of squared, masked, frequency-weighted residuals;
    consequently ``grad(data_loss(render(rho),zero_target=True)/2)`` is
    ``A^H A rho``. Original acquisition denominators remain fixed even when the
    fit mask includes known zeros. Frequency weights are frozen at construction.

    This context serializes this operator's calls, but is not a mechanism for
    sharing the supplied model with another training thread.
    """

    def __init__(self, trainer, *, frequency_t=0.):
        if (isinstance(frequency_t, bool) or not isinstance(frequency_t, Real)
                or not math.isfinite(frequency_t) or frequency_t not in (0., .5, 1.)):
            raise ValueError("frequency_t must be exactly 0, .5 or 1")
        self.frequency_t = float(frequency_t)
        self.trainer = trainer
        self.model = trainer.model
        self._lock = threading.RLock()
        self._support = None
        self.counters = {"render_pairs": 0, "renderer_calls": 0,
                         "adjoint_calls": 0, "vjp_calls": 0,
                         "data_loss_calls": 0, "kspace_loss_calls": 0}
        config = trainer.config
        if bool(config.get("adaptive_control", {}).get("enable", False)):
            raise NotImplementedError("adaptive geometry/control is unsupported for a fixed density operator")
        if bool(getattr(self.model, "has_private", False)) or int(config.get("gaussian", {}).get("num_private_points", 0)):
            raise NotImplementedError("private Gaussian clouds are unsupported")
        loss_config = config.get("loss", {})
        if bool(loss_config.get("kspace_freq_curriculum", False)):
            raise NotImplementedError("frequency curriculum must be disabled; weights must stay frozen")
        top = self.model._parameters
        if set(top) != set(PARAMETER_NAMES) or set(dict(self.model.named_parameters())) != set(PARAMETER_NAMES):
            raise ValueError("expected exactly seven top-level source parameters: " + ", ".join(PARAMETER_NAMES))
        first = top["density_real"]
        if first.ndim != 1 or first.numel() == 0 or first.dtype not in (torch.float32, torch.float64):
            raise ValueError("source densities must be nonempty float32/64 vectors of shape (N,)")
        self.n_points = first.numel()
        self.device, self.real_dtype = first.device, first.dtype
        self.dtype = torch.complex64 if first.dtype == torch.float32 else torch.complex128
        expected_shapes = ((self.n_points, 3), (self.n_points, 3), (self.n_points, 4)) + ((self.n_points,),) * 4
        for name, shape in zip(PARAMETER_NAMES, expected_shapes):
            value = top[name]
            if not isinstance(value, torch.nn.Parameter) or tuple(value.shape) != shape:
                raise ValueError(f"unsupported source parameter {name}: expected shape {shape}")
            if value.device != self.device or value.dtype != self.real_dtype:
                raise ValueError("all seven parameters must share floating dtype and device")
            if not bool(torch.isfinite(value.detach()).all()):
                raise ValueError(f"source parameter {name} must be finite")
        self.volume_shape = tuple(trainer.volume_shape)
        if len(self.volume_shape) != 3 or any(int(size) <= 0 for size in self.volume_shape):
            raise ValueError("expected a nonempty three-dimensional volume")
        self.initial_density = torch.stack((torch.complex(top["density_real"].detach(), top["density_imag"].detach()),
                                            torch.complex(top["density_real_contrast"].detach(), top["density_imag_contrast"].detach()))).clone()
        if not callable(getattr(trainer, "_render", None)) or not callable(getattr(trainer, "_kspace_loss", None)):
            raise TypeError("trainer must expose existing _render and _kspace_loss methods")
        # These are constructed by source train(), not by the trainer constructor.
        # Use the source builder directly and never call train() to initialize them.
        shared = trainer._build_freq_weight()
        override = loss_config.get("kspace_high_weight_plain")
        plain = trainer._build_freq_weight(high_override=override) if override is not None else shared
        self._source_frequency_weights = tuple(self._freeze_weight(weight) for weight in (plain, shared))
        acquired = []
        for tag in ("p", "c"):
            mask = getattr(trainer, f"mask_{tag}", None)
            if (not isinstance(mask, torch.Tensor) or mask.is_complex()
                    or tuple(mask.shape) != self.volume_shape or mask.device != self.device):
                raise ValueError("original acquired mask_p/mask_c must match volume/device; fit masks are not a fallback")
            acquired.append(mask.detach().clone())
        self._acquired_masks = tuple(acquired)
        # One insertion point for merit, homogeneous normals and backtracking.
        # The reviewed utility returns the exact source Tensor at t0.
        self._frequency_weights = tuple(flatten_acquired_rms(weight, mask, self.frequency_t)
                                        for weight, mask in zip(self._source_frequency_weights, self._acquired_masks))
        self.frequency_summary()  # Both scans must meet the fixed measured mass bound.
        self._dc = []
        for tag in ("p", "c"):
            csm, measured, mask = (getattr(trainer, f"{prefix}_{tag}")
                                   for prefix in ("csm_fit", "kspace_fit", "mask_fit"))
            norm = float(getattr(trainer, f"_dc_norm_{tag}"))
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError("original acquisition DC normalizer must be finite and positive")
            if tuple(mask.shape) != self.volume_shape or tuple(csm.shape[1:]) != self.volume_shape:
                raise ValueError("fit mask and coil maps must match volume_shape")
            if tuple(measured.shape) != tuple(csm.shape):
                raise ValueError("fit measurements and coil maps must have identical shapes")
            if any(value.device != self.device for value in (csm, measured, mask)):
                raise ValueError("fit tensors and densities must share a device")
            self._dc.append((csm.detach(), measured.detach(), mask.detach(), norm))

    def _freeze_weight(self, weight):
        if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
            raise ValueError("source frequency weight must exist as a real floating Tensor; no fallback")
        if tuple(weight.shape) != self.volume_shape or weight.device != self.device:
            raise ValueError("frequency weight must match the volume and device")
        if weight.is_complex() or not bool(torch.isfinite(weight).all()):
            raise ValueError("frequency weight must be finite and real")
        return weight.detach().clone()

    @staticmethod
    def _weight_descriptor(value):
        array = value.detach().cpu().contiguous().numpy()
        return {"byte_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}

    def frequency_summary(self):
        """Actual frozen arrays' scalar/byte evidence; no input/scoring data enter."""
        scans = {}
        for scan, source, effective, mask in zip(("pre", "post"), self._source_frequency_weights,
                                                 self._frequency_weights, self._acquired_masks):
            acquired = mask.bool()
            source_mass = float(source[acquired].double().square().mean())
            effective_mass = float(effective[acquired].double().square().mean())
            if not math.isfinite(source_mass) or source_mass <= 0. or not math.isfinite(effective_mass):
                raise ValueError("frequency evidence requires finite positive acquired mean squared weights")
            relative = abs(effective_mass / source_mass - 1.)
            exact = torch.equal(source, effective)
            if relative > 2e-6 or (self.frequency_t == 0. and not exact):
                raise ValueError("frequency weights violate fixed acquired-mass/t0 identity bounds")
            scans[scan] = {"source_weight": self._weight_descriptor(source),
                           "effective_weight": self._weight_descriptor(effective),
                           "acquired_mask": self._weight_descriptor(mask),
                           "acquired_count": int(acquired.sum()), "c": math.sqrt(source_mass),
                           "source_mean_squared_weight": source_mass,
                           "effective_mean_squared_weight": effective_mass,
                           "relative_mass_error": relative,
                           "t0_exact_equal": exact if self.frequency_t == 0. else None}
        return {"version": 1, "frequency_t": self.frequency_t,
                "mass_relative_tolerance": 2e-6, "scans": scans}

    def save_frequency_evidence(self, run_dir):
        """Save six exact arrays and independently checkable JSON in a fresh leaf."""
        import numpy as np
        result = self.frequency_summary()
        directory = Path(run_dir).resolve() / "frequency_weights"
        directory.mkdir(exist_ok=False)
        for scan, source, effective, mask in zip(("pre", "post"), self._source_frequency_weights,
                                                 self._frequency_weights, self._acquired_masks):
            for name, value in (("source_weight", source), ("effective_weight", effective), ("acquired_mask", mask)):
                path = directory / f"{scan}_{name}.npy"
                with path.open("xb") as stream:
                    np.save(stream, value.detach().cpu().contiguous().numpy(), allow_pickle=False)
                result["scans"][scan][name].update(path=str(path), file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        with (directory / "manifest.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return result

    def _validate_density(self, rho):
        if not isinstance(rho, torch.Tensor) or not rho.is_complex():
            raise TypeError("rho must be a complex tensor")
        if tuple(rho.shape) != (2, self.n_points) or rho.dtype != self.dtype or rho.device != self.device:
            raise ValueError(f"rho must have shape (2,{self.n_points}), dtype {self.dtype}, device {self.device}")
        if not bool(torch.isfinite(rho.detach()).all()):
            raise ValueError("rho must be finite")

    @contextmanager
    def _external_density(self, rho):
        self._validate_density(rho)
        with self._lock:
            original = dict(self.model._parameters)
            if set(original) != set(PARAMETER_NAMES):
                raise ValueError("model parameter structure changed after operator construction")
            try:
                for name in GEOMETRY_NAMES:
                    self.model._parameters[name] = original[name].detach()
                for name, replacement in zip(PARAMETER_NAMES[3:],
                                              (rho[0].real, rho[0].imag, rho[1].real, rho[1].imag)):
                    self.model._parameters[name] = replacement
                yield
            finally:
                self.model._parameters.clear()
                self.model._parameters.update(original)

    def render(self, rho):
        """Render signed complex external densities, preserving their autograd graph."""
        with self._external_density(rho):
            self.counters["render_pairs"] += 1
            result = []
            for channel in ("plain", "contrast"):
                self.counters["renderer_calls"] += 1
                image = self.trainer._render(channel)
                if not isinstance(image, torch.Tensor) or not image.is_complex() or tuple(image.shape) != self.volume_shape:
                    raise ValueError("existing renderer must return a complex volume of volume_shape")
                result.append(image)
            return tuple(result)

    def adjoint(self, pre_grad, post_grad):
        """Apply B^H by one first-order VJP; never request custom double backward."""
        gradients = (pre_grad, post_grad)
        for gradient in gradients:
            if not isinstance(gradient, torch.Tensor) or tuple(gradient.shape) != self.volume_shape:
                raise ValueError("adjoint image gradients must match volume_shape")
            if gradient.device != self.device or gradient.dtype != self.dtype:
                raise ValueError("adjoint image gradients must match density dtype and device")
            if not bool(torch.isfinite(gradient.detach()).all()):
                raise ValueError("adjoint image gradients must be finite")
        self.counters["adjoint_calls"] += 1
        with torch.enable_grad():
            rho = torch.zeros_like(self.initial_density, requires_grad=True)
            images = self.render(rho)
            self.counters["vjp_calls"] += 1
            result, = torch.autograd.grad(images, rho, grad_outputs=tuple(g.detach() for g in gradients),
                                          create_graph=False, retain_graph=False)
        return result.detach()

    def structural_support(self):
        """Cache B1>0; B has the source renderer's real nonnegative Gaussian basis."""
        if self._support is None:
            with torch.no_grad():
                images = self.render(torch.ones_like(self.initial_density))
                support = []
                for image in images:
                    if not bool(torch.isfinite(image).all()):
                        raise ValueError("B1 must be finite before structural support is inferred")
                    if bool((image.imag != 0).any()) or bool((image.real < 0).any()):
                        raise ValueError("structural support requires a real nonnegative Gaussian basis")
                    support.append((image.real > 0).detach().clone())
                self._support = tuple(support)
        return tuple(mask.clone() for mask in self._support)

    def data_loss(self, pre, post, zero_target=False):
        """Call source DC exactly, retaining its original norm and squared weights."""
        self.counters["data_loss_calls"] += 1
        values = []
        for image, data, weight in zip((pre, post), self._dc, self._frequency_weights):
            if tuple(image.shape) != self.volume_shape:
                raise ValueError("data_loss images must match volume_shape")
            csm, measured, mask, norm = data
            # Scalar zero broadcasts through the source subtraction: no full coil
            # allocation and, crucially, no denominator recomputation from zero Y.
            target = measured.new_zeros(()) if zero_target else measured
            self.counters["kspace_loss_calls"] += 1
            values.append(self.trainer._kspace_loss(image, csm, target, mask, weight, norm=norm))
        return values[0] + values[1]
