"""
Direct k-space rendering from 3D Gaussians.

Key insight: Fourier transform of a 3D Gaussian is also a Gaussian.
For a Gaussian g(x) = exp(-½(x-μ)ᵀΣ⁻¹(x-μ)), its Fourier transform is:
G(k) = exp(-2πik·μ) * exp(-2π²kᵀΣk)

This allows us to render k-space directly without voxelization.
"""

import torch
import torch.nn as nn
import numpy as np


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 3x3 rotation matrix.

    Args:
        q: (N, 4) quaternions [w, x, y, z]

    Returns:
        R: (N, 3, 3) rotation matrices
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2*(y**2 + z**2)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)

    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x**2 + z**2)
    R[:, 1, 2] = 2*(y*z - w*x)

    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x**2 + y**2)

    return R


def gaussian_kspace_transform(
    positions: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    densities: torch.Tensor,
    kspace_coords: torch.Tensor
) -> torch.Tensor:
    """
    Compute k-space values from 3D Gaussians at specified coordinates.

    For each Gaussian n with center μ_n, covariance Σ_n, and density ρ_n:
    G_n(k) = ρ_n * exp(-2πik·μ_n) * exp(-2π²kᵀΣ_nk)

    Args:
        positions: (N, 3) Gaussian centers in spatial domain
        scales: (N, 3) Gaussian scales (diagonal of Σ before rotation)
        rotations: (N, 4) quaternions defining rotation
        densities: (N, 2) complex density [real, imag]
        kspace_coords: (M, 3) k-space coordinates to evaluate

    Returns:
        kspace_values: (M, 2) complex k-space values [real, imag]
    """
    N = positions.shape[0]  # Number of Gaussians
    M = kspace_coords.shape[0]  # Number of k-space samples

    # Convert quaternions to rotation matrices
    R = quaternion_to_rotation_matrix(rotations)  # (N, 3, 3)

    # Build covariance matrices: Σ = R * diag(scales²) * Rᵀ
    scales_sq = scales ** 2  # (N, 3)
    S_diag = torch.diag_embed(scales_sq)  # (N, 3, 3)
    Sigma = torch.bmm(torch.bmm(R, S_diag), R.transpose(1, 2))  # (N, 3, 3)

    # Phase term: exp(-2πik·μ)
    # k·μ shape: (M, N)
    k_dot_mu = torch.matmul(kspace_coords, positions.T)  # (M, N)
    phase = -2 * np.pi * k_dot_mu  # (M, N)

    # Amplitude term: exp(-2π²kᵀΣk)
    # For each Gaussian n and k-space point m:
    # kᵀΣk = Σ_ij k_i Σ_ij k_j

    # Expand for batch computation
    k_expanded = kspace_coords.unsqueeze(1)  # (M, 1, 3)
    Sigma_expanded = Sigma.unsqueeze(0)  # (1, N, 3, 3)

    # k^T Sigma k for all combinations
    # (M, N, 1, 3) @ (M, N, 3, 3) @ (M, N, 3, 1) -> (M, N, 1, 1)
    k_Sigma_k = torch.matmul(
        torch.matmul(k_expanded.unsqueeze(2), Sigma_expanded),
        k_expanded.unsqueeze(3)
    ).squeeze(-1).squeeze(-1)  # (M, N)

    amplitude = torch.exp(-2 * np.pi**2 * k_Sigma_k)  # (M, N)

    # Combine phase and amplitude
    # exp(-2πik·μ) = cos(-2πk·μ) + i*sin(-2πk·μ)
    cos_phase = torch.cos(phase)  # (M, N)
    sin_phase = torch.sin(phase)  # (M, N)

    # Complex multiplication: (amplitude * exp(i*phase)) * density
    # density = ρ_real + i*ρ_imag
    rho_real = densities[:, 0]  # (N,)
    rho_imag = densities[:, 1]  # (N,)

    # Real part: amplitude * (cos*ρ_real - sin*ρ_imag)
    real_part = amplitude * (cos_phase * rho_real - sin_phase * rho_imag)  # (M, N)

    # Imag part: amplitude * (sin*ρ_real + cos*ρ_imag)
    imag_part = amplitude * (sin_phase * rho_real + cos_phase * rho_imag)  # (M, N)

    # Sum over all Gaussians
    kspace_real = real_part.sum(dim=1)  # (M,)
    kspace_imag = imag_part.sum(dim=1)  # (M,)

    # Stack to complex representation
    kspace_values = torch.stack([kspace_real, kspace_imag], dim=1)  # (M, 2)

    return kspace_values


class DirectKSpaceRenderer(nn.Module):
    """
    Renders k-space directly from 3D Gaussians without voxelization.
    """

    def __init__(self, volume_shape, coil_maps=None):
        super().__init__()
        self.volume_shape = volume_shape
        self.coil_maps = coil_maps

    def forward(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        densities: torch.Tensor,
        kspace_coords: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Render k-space from Gaussians.

        Args:
            positions: (N, 3) Gaussian centers
            scales: (N, 3) Gaussian scales
            rotations: (N, 4) quaternions
            densities: (N, 2) complex density
            kspace_coords: (D, H, W, 3) k-space coordinate grid
            mask: (D, H, W) sampling mask (optional)

        Returns:
            kspace: (D, H, W, 2) complex k-space values
        """
        D, H, W = kspace_coords.shape[:3]

        # Flatten k-space coordinates
        if mask is not None:
            # Only compute at sampled points
            sampled_coords = kspace_coords[mask]  # (M, 3)
            kspace_sampled = gaussian_kspace_transform(
                positions, scales, rotations, densities, sampled_coords
            )  # (M, 2)

            # Scatter back to full grid
            kspace = torch.zeros(D, H, W, 2, device=positions.device, dtype=positions.dtype)
            kspace[mask] = kspace_sampled
        else:
            # Compute at all points
            coords_flat = kspace_coords.reshape(-1, 3)  # (D*H*W, 3)
            kspace_flat = gaussian_kspace_transform(
                positions, scales, rotations, densities, coords_flat
            )  # (D*H*W, 2)
            kspace = kspace_flat.reshape(D, H, W, 2)

        return kspace


def create_kspace_coordinates(volume_shape, device='cuda'):
    """
    Create k-space coordinate grid.

    K-space coordinates range from -0.5 to 0.5 in normalized units.

    Args:
        volume_shape: (D, H, W) spatial dimensions

    Returns:
        kspace_coords: (D, H, W, 3) k-space coordinates
    """
    D, H, W = volume_shape

    # Create frequency grids (fftshift convention)
    kz = torch.fft.fftfreq(D, d=1.0, device=device)
    ky = torch.fft.fftfreq(H, d=1.0, device=device)
    kx = torch.fft.fftfreq(W, d=1.0, device=device)

    # Create meshgrid
    kz_grid, ky_grid, kx_grid = torch.meshgrid(kz, ky, kx, indexing='ij')

    # Stack into coordinate array
    kspace_coords = torch.stack([kz_grid, ky_grid, kx_grid], dim=-1)  # (D, H, W, 3)

    return kspace_coords
