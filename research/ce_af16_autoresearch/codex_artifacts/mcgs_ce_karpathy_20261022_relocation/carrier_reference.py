"""Dense CPU double-precision oracle for selected Gaussian carrier corrections.

This intentionally allocates voxel-by-selected-atom tensors for tiny numerical
fixtures. It imports no renderer or CUDA extension and is not a production path.
"""

import math
from numbers import Integral

import torch


def _correction_from_geometry(centers, inverse_covariance, density, frequencies, volume_shape):
    """Evaluate in voxel coordinates; caller supplies validated double tensors.

    The private geometry boundary also permits direct local-VJP acceptance tests.
    Support membership is fixed for differentiation, as in the historical kernel.
    """
    half = centers.new_tensor(volume_shape) / 2
    voxels = torch.stack(torch.meshgrid(
        *(torch.arange(n, dtype=torch.float64, device="cpu") for n in volume_shape),
        indexing="ij"), dim=-1).reshape(-1, 3)
    displacement = voxels.unsqueeze(0) - centers.unsqueeze(1)
    normalized = displacement / half
    mahal = torch.einsum("kvi,kij,kvj->kv", normalized, inverse_covariance, normalized)
    weight = torch.where(mahal.detach() <= 9, torch.exp(-mahal / 2), 0)
    theta = 2 * math.pi * (displacement * frequencies.unsqueeze(1)).sum(-1)
    # Avoid cancellation in cos(theta)-1 while retaining the frequency derivative
    # at theta=0 through sin(theta). Never short-circuit zero frequencies.
    coefficient = torch.complex(-2 * torch.sin(theta / 2).square(), torch.sin(theta))
    return (weight * density.unsqueeze(1) * coefficient).sum(0).reshape(volume_shape)


def carrier_correction_reference(positions, log_scales, rotations, density, frequencies, volume_shape):
    """Return sum G*rho*(exp(i*2*pi*f.(v-C))-1), as CPU complex128.

    Inputs are CPU float64 [K,3] positions, log scales, and frequencies; CPU
    float64 [K,4] wxyz quaternions; and CPU complex128 [K] densities. Frequencies
    are cycles per voxel, strictly inside (-0.5, 0.5) on each D,H,W axis.
    Positions are normalized coordinates and scales are normalized lengths.
    Every input remains connected to autograd, including for an empty subset.
    """
    if not isinstance(volume_shape, (tuple, list, torch.Size)):
        raise TypeError("volume_shape must contain three positive integers")
    if len(volume_shape) != 3:
        raise ValueError("volume_shape must contain three positive integers")
    if any(isinstance(n, bool) or not isinstance(n, Integral) for n in volume_shape):
        raise TypeError("volume_shape must contain three positive integers")
    if any(n <= 0 for n in volume_shape):
        raise ValueError("volume_shape must contain three positive integers")
    volume_shape = tuple(int(n) for n in volume_shape)

    inputs = (positions, log_scales, rotations, density, frequencies)
    names = ("positions", "log_scales", "rotations", "density", "frequencies")
    for name, value in zip(names, inputs):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU")
        expected_dtype = torch.complex128 if name == "density" else torch.float64
        if value.dtype != expected_dtype:
            raise TypeError(f"{name} must have dtype {expected_dtype}")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [K,3]")
    count = positions.shape[0]
    shapes = ((count, 3), (count, 3), (count, 4), (count,), (count, 3))
    for name, value, shape in zip(names, inputs, shapes):
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain only finite values")
    if not ((frequencies > -0.5) & (frequencies < 0.5)).all().item():
        raise ValueError("frequencies must be strictly inside (-0.5, 0.5) cycles per voxel")

    half = positions.new_tensor(volume_shape) / 2
    centers = (positions + 1) * half - 0.5
    q = rotations / (torch.linalg.vector_norm(rotations, dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(-1)
    r0 = torch.stack((1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w,
                      2 * x * z + 2 * y * w), dim=-1)
    r1 = torch.stack((2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z,
                      2 * y * z - 2 * x * w), dim=-1)
    r2 = torch.stack((2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w,
                      1 - 2 * x * x - 2 * y * y), dim=-1)
    rotation_matrix = torch.stack((r0, r1, r2), dim=-2)
    inverse_variance = (log_scales.exp() + 1e-8).pow(-2)
    inverse_covariance = (rotation_matrix * inverse_variance.unsqueeze(1)) @ rotation_matrix.transpose(-1, -2)
    return _correction_from_geometry(centers, inverse_covariance, density, frequencies, volume_shape)
