"""Fit a fresh residual magnitude CNN using only one supplied crop and noise.

Returns independent float32 arrays. Identity means an exact copy of the
canonical float32 conversion; input descriptors retain original dtype/bytes.
No input tensor, optimizer, model, or graph is retained across invocations.
"""

import hashlib
import itertools
import json
import math
from numbers import Integral, Real
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _real(name, value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real")
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _array_description(array):
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def _parameter_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode() + b"\0")
        digest.update(array.dtype.str.encode() + b"\0")
        digest.update(json.dumps(list(array.shape)).encode() + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _descriptor(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class MagnitudeNoiseCNN(nn.Module):
    """Five radius-one convolutions; forward returns input minus predicted noise."""

    def __init__(self, width=32):
        super().__init__()
        width = _positive_int("width", width)
        layers = []
        channels = [1, width, width, width, width, 1]
        for i in range(5):
            layers.append(nn.Conv3d(channels[i], channels[i + 1], 3,
                                    padding=1, dtype=torch.float32, device="cpu"))
            if i < 4:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x):
        return x - self.network(x)


@torch.no_grad()
def tiled_inference(model, x, *, tile_size=32):
    """Infer 3D interiors with radius-five context and true global boundaries.

    ``x`` is [N,1,D,H,W]. Global edges are clipped, allowing each convolution's
    native zero padding to reproduce full-volume inference there.
    """
    tile_size = _positive_int("tile_size", tile_size)
    if x.ndim != 5 or x.shape[1] != 1 or any(n < 1 for n in x.shape):
        raise ValueError("x must be a nonempty [N,1,D,H,W] tensor")
    model.eval()
    out = torch.empty_like(x)
    shape = x.shape[-3:]
    with torch.autocast(device_type=x.device.type, enabled=False):
        for start in itertools.product(*(range(0, n, tile_size) for n in shape)):
            end = tuple(min(a + tile_size, n) for a, n in zip(start, shape))
            lo = tuple(max(0, a - 5) for a in start)
            hi = tuple(min(n, b + 5) for b, n in zip(end, shape))
            halo = tuple(slice(a, b) for a, b in zip(lo, hi))
            interior = tuple(slice(a, b) for a, b in zip(start, end))
            local = tuple(slice(a - l, b - l) for a, b, l in zip(start, end, lo))
            out[(...,) + interior] = model(x[(...,) + halo])[(...,) + local]
    return out


class CaseMagnitudeDenoiser:
    """Independent per-invocation case-only fits with immutable artifact paths.

    Input validation occurs before directory/RNG/record changes. Failed active
    calls preserve their reserved directory for diagnosis and do not append a
    success record; a retry refuses that destination rather than overwriting it.
    This callable is used serially by per-scan refreshes, not concurrently.
    """

    def __init__(self, output_dir, *, device, seed=42, updates=128,
                 patch_size=32, batch_size=4, width=32, lr=1e-3, tile_size=32):
        self.updates = _positive_int("updates", updates)
        self.patch_size = _positive_int("patch_size", patch_size)
        self.batch_size = _positive_int("batch_size", batch_size)
        self.width = _positive_int("width", width)
        self.tile_size = _positive_int("tile_size", tile_size)
        self.lr = _real("lr", lr, positive=True)
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        self.seed = int(seed)
        try:
            self.device = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError("device must be cpu or an available cuda device") from exc
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.device.type == "cpu" and self.device.index is not None:
            raise ValueError("cpu device must not have an index")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("cuda is unavailable")
            index = self.device.index
            if index is None:
                index = torch.cuda.current_device()
            if index >= torch.cuda.device_count():
                raise ValueError("cuda device index is unavailable")
            self.device = torch.device("cuda", index)
        self.output_dir = Path(output_dir).absolute()
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError("output_dir must be a directory")
        self.last_stats = None
        self.records = []

    def _validate(self, z, sigma):
        if not isinstance(z, np.ndarray) or z.ndim != 3:
            raise ValueError("input must be a real 3D numpy array")
        if z.dtype.kind not in "uif" or any(n < self.patch_size for n in z.shape):
            raise ValueError("input must be real numeric data with dimensions >= patch_size")
        if not np.isfinite(z).all() or (z < 0).any():
            raise ValueError("input must be finite and nonnegative")
        sigma = _real("sigma", sigma)
        with np.errstate(over="ignore", invalid="ignore"):
            canonical = np.array(z, dtype=np.float32, order="C", copy=True)
        if not np.isfinite(canonical).all():
            raise ValueError("input must remain finite in float32")
        normalization = max(float(np.quantile(z, 0.99)), 1e-6)
        normalized_sigma = sigma / normalization
        if not math.isfinite(normalized_sigma) or normalized_sigma > np.finfo(np.float32).max:
            raise ValueError("normalized sigma must be representable in float32")
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = canonical / normalization
        if not np.isfinite(normalized).all():
            raise ValueError("normalized input must be finite in float32")
        return canonical, sigma, normalization, normalized_sigma

    def _fit(self, canonical, normalization, normalized_sigma, seed, record):
        # Only CPU default RNG is used by ordinary layer initialization. CUDA
        # noise uses its own generator; no visible CUDA default RNG is touched.
        # Adam also creates a default-device step scalar. Scope default
        # allocations to CPU without changing the caller's default device;
        # noise and model execution still use the explicitly selected device.
        # fork_rng restores CPU initialization RNG even if fitting raises.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(seed)
            model = MagnitudeNoiseCNN(width=self.width).to(self.device)
            initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            record["initial_parameter_sha256"] = _parameter_hash(initial)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                         betas=(0.9, 0.999), eps=1e-8,
                                         weight_decay=0)
            noise_rng = torch.Generator(device=self.device).manual_seed(seed)
            crop_rng = random.Random(seed)
            image = torch.from_numpy(canonical / normalization).to(self.device)[None, None]
            losses = []
            max_gradient = 0.0
            steps_with_nonzero = 0
            p = self.patch_size
            with torch.enable_grad(), torch.autocast(device_type=self.device.type, enabled=False):
                model.train()
                for _ in range(self.updates):
                    patches = []
                    for _ in range(self.batch_size):
                        start = [crop_rng.randrange(n - p + 1) for n in canonical.shape]
                        slices = tuple(slice(a, a + p) for a in start)
                        patches.append(image[(...,) + slices])
                    target = torch.cat(patches, dim=0).detach()
                    noise = torch.randn(target.shape, dtype=torch.float32,
                                        device=self.device, generator=noise_rng)
                    noisy = target + noise * normalized_sigma
                    optimizer.zero_grad(set_to_none=True)
                    loss = (model(noisy) - target).square().mean()
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("nonfinite denoiser training loss")
                    loss.backward()
                    gradients = [parameter.grad for parameter in model.parameters()]
                    if any(g is None for g in gradients):
                        raise FloatingPointError("missing denoiser parameter gradient")
                    maxima = torch.stack([g.abs().max() for g in gradients])
                    if not torch.isfinite(maxima).all().item():
                        raise FloatingPointError("nonfinite denoiser parameter gradient")
                    maximum = float(maxima.max().item())
                    max_gradient = max(max_gradient, maximum)
                    steps_with_nonzero += int(maximum > 0)
                    optimizer.step()
                    losses.append(float(loss.item()))
            final = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if not all(torch.isfinite(v).all().item() for v in final.values()):
                raise FloatingPointError("nonfinite trained denoiser parameters")
            record.update(
                updates=len(losses), losses={"first": losses[0], "last": losses[-1]},
                gradients={"all_finite": True, "any_nonzero": max_gradient > 0,
                           "max_abs": max_gradient, "steps_with_nonzero": steps_with_nonzero},
                changed_parameters=[k for k in initial if not torch.equal(initial[k], final[k])],
                final_parameter_sha256=_parameter_hash(final),
            )
            with torch.no_grad():
                inferred = tiled_inference(model, image, tile_size=self.tile_size)[0, 0]
                raw = (inferred * normalization).cpu().numpy()
            if not np.isfinite(raw).all():
                raise FloatingPointError("nonfinite denoiser inference")
            clipped_count = int((raw < 0).sum())
            return np.maximum(raw, 0).astype(np.float32, copy=True), clipped_count, final

    def __call__(self, z, sigma):
        canonical, sigma, normalization, normalized_sigma = self._validate(z, sigma)
        call_index = len(self.records)
        seed = self.seed + 1009 * call_index
        if seed >= 2**63:
            raise ValueError("invocation seed exceeds supported range")
        call_dir = self.output_dir / f"call_{call_index:04d}"
        # mkdir(exist_ok=False) atomically reserves this invocation.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        call_dir.mkdir(exist_ok=False)
        started = time.perf_counter()
        status = "identity_zero_sigma" if sigma == 0 else (
            "identity_zero_input" if not canonical.any() else "trained")
        record = dict(
            schema="case_magnitude_denoiser_call_v1", call_index=call_index,
            seed=seed, status=status, input=_array_description(z), sigma=sigma,
            normalization=normalization, normalized_sigma=normalized_sigma,
            updates=0, patch_size=self.patch_size, batch_size=self.batch_size,
            width=self.width, lr=self.lr, tile_size=self.tile_size,
            losses=None, gradients=None,
            changed_parameters=[], initial_parameter_sha256=None,
            final_parameter_sha256=None, elapsed_seconds=0.0,
            output=None, checkpoint=None, target=None,
        )
        final_state = None
        if status == "trained":
            output, clipped, final_state = self._fit(
                canonical, normalization, normalized_sigma, seed, record)
        else:
            output, clipped = canonical.copy(), 0
        record["output"] = {**_array_description(output), "clipped_count": clipped}
        target_path = call_dir / "target.npy"
        with target_path.open("xb") as stream:
            np.save(stream, output, allow_pickle=False)
        record["target"] = _descriptor(target_path)
        if final_state is not None:
            checkpoint = {key: record[key] for key in (
                "call_index", "seed", "input", "output", "updates", "sigma",
                "normalization", "normalized_sigma", "initial_parameter_sha256",
                "final_parameter_sha256", "width", "patch_size", "batch_size", "lr",
                "tile_size")}
            checkpoint.update(schema="case_magnitude_denoiser_checkpoint_v1",
                              model_state_dict=final_state)
            checkpoint_path = call_dir / "checkpoint.pt"
            with checkpoint_path.open("xb") as stream:
                torch.save(checkpoint, stream)
            record["checkpoint"] = _descriptor(checkpoint_path)
        record["elapsed_seconds"] = time.perf_counter() - started
        manifest_path = call_dir / "manifest.json"
        with manifest_path.open("x") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        record["manifest"] = _descriptor(manifest_path)
        self.records.append(record)
        self.last_stats = record
        return output
