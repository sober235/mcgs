"""Tile-based voxelization for paper-faithful 3DGSMR."""

import hashlib
import math
import os
from typing import Tuple

import torch
import torch.nn as nn

_cuda_ext = None
_cuda_load_error = None


def _extension_sources() -> Tuple[str, str]:
    base_dir = os.path.join(os.path.dirname(__file__), "cuda_ext")
    return (
        os.path.join(base_dir, "tile_voxelize_cuda.cpp"),
        os.path.join(base_dir, "tile_voxelize_cuda_kernel.cu"),
    )


def _load_cuda_ext(strict: bool = False):
    global _cuda_ext, _cuda_load_error
    if _cuda_ext is not None:
        return _cuda_ext
    if _cuda_load_error is not None:
        if strict:
            raise RuntimeError(_cuda_load_error)
        return None
    try:
        from torch.utils.cpp_extension import load

        cpp_src, cuda_src = _extension_sources()
        with open(cpp_src, "rb") as handle:
            cpp_hash = hashlib.md5(handle.read()).hexdigest()[:8]
        with open(cuda_src, "rb") as handle:
            cu_hash = hashlib.md5(handle.read()).hexdigest()[:8]
        build_name = f"tile_voxelize_cuda_{cpp_hash}_{cu_hash}"
        build_dir = os.path.join(os.path.dirname(cuda_src), "build")
        os.makedirs(build_dir, exist_ok=True)

        # Prefer conda env's libcudart.so.12 (CUDA 12.x, PyTorch compat) over
        # the system CUDA 13.x to avoid "driver version insufficient" at runtime.
        # Use explicit full-path linking to ensure the right version is loaded.
        extra_ldflags = []
        search_roots = []
        for env_name in ("CONDA_PREFIX", "CUDA_HOME", "CUDA_PATH"):
            value = os.environ.get(env_name)
            if value:
                search_roots.append(value)
        search_roots.append("/usr/local/cuda")
        cudart_lib_path = None
        for root_dir in search_roots:
            for lib_dir in (os.path.join(root_dir, "lib64"), os.path.join(root_dir, "lib")):
                # Prefer versioned libcudart.so.12 (PyTorch compat) over libcudart.so.13
                for lib_name in ("libcudart.so.12", "libcudart.so"):
                    full_path = os.path.join(lib_dir, lib_name)
                    if os.path.isfile(full_path):
                        cudart_lib_path = full_path
                        break
                if cudart_lib_path:
                    break
            if cudart_lib_path:
                break
        if cudart_lib_path:
            # Use explicit full-path to bypass linker search-path precedence
            lib_dir = os.path.dirname(cudart_lib_path)
            extra_ldflags = [cudart_lib_path, f"-Wl,-rpath,{lib_dir}"]

        _cuda_ext = load(
            name=build_name,
            sources=[cpp_src, cuda_src],
            build_directory=build_dir,
            verbose=False,
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_ldflags=extra_ldflags,
        )
        # Runtime validation: verify the extension actually executes on GPU.
        # A stale .so compiled against a different CUDA version loads fine but
        # fails at kernel launch with "CUDA driver version is insufficient".
        _N = 1
        _dev = torch.device("cuda", 0)
        _center = torch.zeros(_N, 3, device=_dev, dtype=torch.float32)
        _cov_inv = torch.eye(3, device=_dev, dtype=torch.float32).unsqueeze(0).expand(_N, 3, 3).contiguous()
        _den_r = torch.ones(_N, device=_dev, dtype=torch.float32)
        _den_i = torch.zeros(_N, device=_dev, dtype=torch.float32)
        _tile_ranges = torch.zeros(1, 2, dtype=torch.int32, device=_dev)
        _pair_ids = torch.zeros(0, dtype=torch.int32, device=_dev)
        _half_shape = torch.tensor([4.0, 4.0, 4.0], dtype=torch.float32, device=_dev)
        _cuda_ext.forward(
            _center, _cov_inv, _den_r, _den_i, _tile_ranges, _pair_ids, _half_shape,
            8, 8, 8, 8,
        )
        return _cuda_ext
    except Exception as exc:  # pragma: no cover
        _cuda_load_error = f"Failed to build tile_cuda extension: {exc}"
        if strict:
            raise RuntimeError(_cuda_load_error) from exc
        print(f"[TileVoxelizer] {_cuda_load_error}")
        print("[TileVoxelizer] Falling back to the PyTorch tile reference backend.")
        return None


# ---------------------------------------------------------------------------
# CuPy-based fallback: compiles the same CUDA kernel via NVRTC (CUDA 12.x).
# Used when torch.utils.cpp_extension.load fails due to CUDA version mismatch.
# ---------------------------------------------------------------------------

_cupy_module = None
_cupy_load_error = None

_TILE_KERNEL_SRC = r"""
#define kThreads 256
#define kStageGaussians 32

__device__ inline void decode_tile_cp(
    int tile_id, int tile_size,
    int d_size, int h_size, int w_size,
    int n_th, int n_tw,
    int* d0, int* d1, int* h0, int* h1, int* w0, int* w1,
    int* rh, int* rw, int* total_voxels)
{
    const int td = tile_id / (n_th * n_tw);
    const int rem = tile_id % (n_th * n_tw);
    const int th = rem / n_tw;
    const int tw = rem % n_tw;
    *d0 = td * tile_size;
    *d1 = min(*d0 + tile_size, d_size);
    *h0 = th * tile_size;
    *h1 = min(*h0 + tile_size, h_size);
    *w0 = tw * tile_size;
    *w1 = min(*w0 + tile_size, w_size);
    const int rd = *d1 - *d0;
    *rh = *h1 - *h0;
    *rw = *w1 - *w0;
    *total_voxels = rd * (*rh) * (*rw);
}

extern "C" __global__ void tile_voxelize_forward_cupy(
    const float* __restrict__ centers,
    const float* __restrict__ cov_inv,
    const float* __restrict__ density_r,
    const float* __restrict__ density_i,
    const int* __restrict__ tile_ranges,
    const int* __restrict__ pair_ids,
    const float* __restrict__ half_shape,
    float* __restrict__ vol_r,
    float* __restrict__ vol_i,
    int tile_size, int d_size, int h_size, int w_size, int num_tiles)
{
    const int tile_id = blockIdx.x;
    if (tile_id >= num_tiles) return;
    const int pair_start = tile_ranges[tile_id * 2 + 0];
    const int pair_end   = tile_ranges[tile_id * 2 + 1];
    if (pair_start >= pair_end) return;

    const int n_th = (h_size + tile_size - 1) / tile_size;
    const int n_tw = (w_size + tile_size - 1) / tile_size;
    int d0, d1, h0, h1, w0, w1, rh, rw, total_voxels;
    decode_tile_cp(tile_id, tile_size, d_size, h_size, w_size, n_th, n_tw,
                   &d0, &d1, &h0, &h1, &w0, &w1, &rh, &rw, &total_voxels);

    __shared__ int   sh_gid[kStageGaussians];
    __shared__ float sh_cx[kStageGaussians];
    __shared__ float sh_cy[kStageGaussians];
    __shared__ float sh_cz[kStageGaussians];
    __shared__ float sh_cov[kStageGaussians * 9];
    __shared__ float sh_dr[kStageGaussians];
    __shared__ float sh_di[kStageGaussians];

    const float hs0 = half_shape[0], hs1 = half_shape[1], hs2 = half_shape[2];

    for (int vid = threadIdx.x; vid < total_voxels; vid += kThreads) {
        const int ld = vid / (rh * rw);
        const int lr = vid % (rh * rw);
        const int lh = lr / rw, lw = lr % rw;
        const int gd = d0 + ld, gh = h0 + lh, gw = w0 + lw;
        float acc_r = 0.f, acc_i = 0.f;

        for (int bs = pair_start; bs < pair_end; bs += kStageGaussians) {
            const int sc = min(kStageGaussians, pair_end - bs);
            if (threadIdx.x < sc) {
                const int gid = pair_ids[bs + threadIdx.x];
                sh_gid[threadIdx.x] = gid;
                sh_cx[threadIdx.x] = centers[gid * 3 + 0];
                sh_cy[threadIdx.x] = centers[gid * 3 + 1];
                sh_cz[threadIdx.x] = centers[gid * 3 + 2];
                sh_dr[threadIdx.x] = density_r[gid];
                sh_di[threadIdx.x] = density_i[gid];
                for (int m = 0; m < 9; ++m)
                    sh_cov[threadIdx.x * 9 + m] = cov_inv[gid * 9 + m];
            }
            __syncthreads();
            for (int lg = 0; lg < sc; ++lg) {
                const float dx = ((float)gd - sh_cx[lg]) / hs0;
                const float dy = ((float)gh - sh_cy[lg]) / hs1;
                const float dz = ((float)gw - sh_cz[lg]) / hs2;
                const float* M = &sh_cov[lg * 9];
                const float md0 = M[0]*dx + M[1]*dy + M[2]*dz;
                const float md1 = M[3]*dx + M[4]*dy + M[5]*dz;
                const float md2 = M[6]*dx + M[7]*dy + M[8]*dz;
                const float mahal = dx*md0 + dy*md1 + dz*md2;
                if (mahal <= 9.f) {
                    const float w = expf(-0.5f * mahal);
                    acc_r += w * sh_dr[lg];
                    acc_i += w * sh_di[lg];
                }
            }
            __syncthreads();
        }
        const int flat = gd * h_size * w_size + gh * w_size + gw;
        vol_r[flat] = acc_r;
        vol_i[flat] = acc_i;
    }
}

// Phase 11 Plan 11-01 — masked forward (NVRTC parity with the C++/CUDA path).
// Identical to tile_voxelize_forward_cupy plus per-Gaussian gate z[N].
extern "C" __global__ void tile_voxelize_forward_cupy_masked(
    const float* __restrict__ centers,
    const float* __restrict__ cov_inv,
    const float* __restrict__ density_r,
    const float* __restrict__ density_i,
    const signed char* __restrict__ z,
    const int* __restrict__ tile_ranges,
    const int* __restrict__ pair_ids,
    const float* __restrict__ half_shape,
    float* __restrict__ vol_r,
    float* __restrict__ vol_i,
    int tile_size, int d_size, int h_size, int w_size, int num_tiles)
{
    const int tile_id = blockIdx.x;
    if (tile_id >= num_tiles) return;
    const int pair_start = tile_ranges[tile_id * 2 + 0];
    const int pair_end   = tile_ranges[tile_id * 2 + 1];
    if (pair_start >= pair_end) return;

    const int n_th = (h_size + tile_size - 1) / tile_size;
    const int n_tw = (w_size + tile_size - 1) / tile_size;
    int d0, d1, h0, h1, w0, w1, rh, rw, total_voxels;
    decode_tile_cp(tile_id, tile_size, d_size, h_size, w_size, n_th, n_tw,
                   &d0, &d1, &h0, &h1, &w0, &w1, &rh, &rw, &total_voxels);

    __shared__ int   sh_gid[kStageGaussians];
    __shared__ float sh_cx[kStageGaussians];
    __shared__ float sh_cy[kStageGaussians];
    __shared__ float sh_cz[kStageGaussians];
    __shared__ float sh_cov[kStageGaussians * 9];
    __shared__ float sh_dr[kStageGaussians];
    __shared__ float sh_di[kStageGaussians];

    const float hs0 = half_shape[0], hs1 = half_shape[1], hs2 = half_shape[2];

    for (int vid = threadIdx.x; vid < total_voxels; vid += kThreads) {
        const int ld = vid / (rh * rw);
        const int lr = vid % (rh * rw);
        const int lh = lr / rw, lw = lr % rw;
        const int gd = d0 + ld, gh = h0 + lh, gw = w0 + lw;
        float acc_r = 0.f, acc_i = 0.f;

        for (int bs = pair_start; bs < pair_end; bs += kStageGaussians) {
            const int sc = min(kStageGaussians, pair_end - bs);
            if (threadIdx.x < sc) {
                const int gid = pair_ids[bs + threadIdx.x];
                sh_gid[threadIdx.x] = gid;
                sh_cx[threadIdx.x] = centers[gid * 3 + 0];
                sh_cy[threadIdx.x] = centers[gid * 3 + 1];
                sh_cz[threadIdx.x] = centers[gid * 3 + 2];
                // CF-01 binary gate — z[gid] ∈ {0, 1}.
                const float gate = (float)((int)z[gid]);
                sh_dr[threadIdx.x] = density_r[gid] * gate;
                sh_di[threadIdx.x] = density_i[gid] * gate;
                for (int m = 0; m < 9; ++m)
                    sh_cov[threadIdx.x * 9 + m] = cov_inv[gid * 9 + m];
            }
            __syncthreads();
            for (int lg = 0; lg < sc; ++lg) {
                const float dx = ((float)gd - sh_cx[lg]) / hs0;
                const float dy = ((float)gh - sh_cy[lg]) / hs1;
                const float dz = ((float)gw - sh_cz[lg]) / hs2;
                const float* M = &sh_cov[lg * 9];
                const float md0 = M[0]*dx + M[1]*dy + M[2]*dz;
                const float md1 = M[3]*dx + M[4]*dy + M[5]*dz;
                const float md2 = M[6]*dx + M[7]*dy + M[8]*dz;
                const float mahal = dx*md0 + dy*md1 + dz*md2;
                if (mahal <= 9.f) {
                    const float w = expf(-0.5f * mahal);
                    acc_r += w * sh_dr[lg];
                    acc_i += w * sh_di[lg];
                }
            }
            __syncthreads();
        }
        const int flat = gd * h_size * w_size + gh * w_size + gw;
        vol_r[flat] = acc_r;
        vol_i[flat] = acc_i;
    }
}

extern "C" __global__ void tile_voxelize_backward_cupy(
    const float* __restrict__ centers,
    const float* __restrict__ cov_inv,
    const float* __restrict__ density_r,
    const float* __restrict__ density_i,
    const int*   __restrict__ tile_ranges,
    const int*   __restrict__ pair_ids,
    const float* __restrict__ half_shape,
    const float* __restrict__ grad_vol_r,
    const float* __restrict__ grad_vol_i,
    float* __restrict__ grad_centers,
    float* __restrict__ grad_cov_inv,
    float* __restrict__ grad_den_r,
    float* __restrict__ grad_den_i,
    int tile_size, int d_size, int h_size, int w_size, int num_tiles)
{
    const int tile_id = blockIdx.x;
    if (tile_id >= num_tiles) return;
    const int pair_start = tile_ranges[tile_id * 2 + 0];
    const int pair_end   = tile_ranges[tile_id * 2 + 1];
    if (pair_start >= pair_end) return;

    const int n_th = (h_size + tile_size - 1) / tile_size;
    const int n_tw = (w_size + tile_size - 1) / tile_size;
    int d0, d1, h0, h1, w0, w1, rh, rw, total_voxels;
    decode_tile_cp(tile_id, tile_size, d_size, h_size, w_size, n_th, n_tw,
                   &d0, &d1, &h0, &h1, &w0, &w1, &rh, &rw, &total_voxels);

    __shared__ int   sh_gid[kStageGaussians];
    __shared__ float sh_cx[kStageGaussians];
    __shared__ float sh_cy[kStageGaussians];
    __shared__ float sh_cz[kStageGaussians];
    __shared__ float sh_cov[kStageGaussians * 9];
    __shared__ float sh_dr[kStageGaussians];
    __shared__ float sh_di[kStageGaussians];

    const float hs0 = half_shape[0], hs1 = half_shape[1], hs2 = half_shape[2];

    for (int bs = pair_start; bs < pair_end; bs += kStageGaussians) {
        const int sc = min(kStageGaussians, pair_end - bs);
        if (threadIdx.x < sc) {
            const int gid = pair_ids[bs + threadIdx.x];
            sh_gid[threadIdx.x] = gid;
            sh_cx[threadIdx.x] = centers[gid * 3 + 0];
            sh_cy[threadIdx.x] = centers[gid * 3 + 1];
            sh_cz[threadIdx.x] = centers[gid * 3 + 2];
            sh_dr[threadIdx.x] = density_r[gid];
            sh_di[threadIdx.x] = density_i[gid];
            for (int m = 0; m < 9; ++m)
                sh_cov[threadIdx.x * 9 + m] = cov_inv[gid * 9 + m];
        }
        __syncthreads();

        for (int vid = threadIdx.x; vid < total_voxels; vid += kThreads) {
            const int ld = vid / (rh * rw);
            const int lr = vid % (rh * rw);
            const int lh = lr / rw, lw = lr % rw;
            const int gd = d0 + ld, gh = h0 + lh, gw = w0 + lw;
            const int flat = gd * h_size * w_size + gh * w_size + gw;
            const float gvr = grad_vol_r[flat], gvi = grad_vol_i[flat];

            for (int lg = 0; lg < sc; ++lg) {
                const int gid = sh_gid[lg];
                const float dx = ((float)gd - sh_cx[lg]) / hs0;
                const float dy = ((float)gh - sh_cy[lg]) / hs1;
                const float dz = ((float)gw - sh_cz[lg]) / hs2;
                const float* M = &sh_cov[lg * 9];
                const float md0 = M[0]*dx + M[1]*dy + M[2]*dz;
                const float md1 = M[3]*dx + M[4]*dy + M[5]*dz;
                const float md2 = M[6]*dx + M[7]*dy + M[8]*dz;
                const float mahal = dx*md0 + dy*md1 + dz*md2;
                if (mahal <= 9.f) {
                    const float w = expf(-0.5f * mahal);
                    const float dL_dw = sh_dr[lg]*gvr + sh_di[lg]*gvi;
                    const float dL_dm = dL_dw * (-0.5f) * w;
                    atomicAdd(&grad_den_r[gid], w * gvr);
                    atomicAdd(&grad_den_i[gid], w * gvi);
                    atomicAdd(&grad_centers[gid*3+0], dL_dm * (-2.f * md0 / hs0));
                    atomicAdd(&grad_centers[gid*3+1], dL_dm * (-2.f * md1 / hs1));
                    atomicAdd(&grad_centers[gid*3+2], dL_dm * (-2.f * md2 / hs2));
                    atomicAdd(&grad_cov_inv[gid*9+0], dL_dm * dx * dx);
                    atomicAdd(&grad_cov_inv[gid*9+1], dL_dm * dx * dy);
                    atomicAdd(&grad_cov_inv[gid*9+2], dL_dm * dx * dz);
                    atomicAdd(&grad_cov_inv[gid*9+3], dL_dm * dy * dx);
                    atomicAdd(&grad_cov_inv[gid*9+4], dL_dm * dy * dy);
                    atomicAdd(&grad_cov_inv[gid*9+5], dL_dm * dy * dz);
                    atomicAdd(&grad_cov_inv[gid*9+6], dL_dm * dz * dx);
                    atomicAdd(&grad_cov_inv[gid*9+7], dL_dm * dz * dy);
                    atomicAdd(&grad_cov_inv[gid*9+8], dL_dm * dz * dz);
                }
            }
        }
        __syncthreads();
    }
}
"""


def _load_cupy_ext(strict: bool = False):
    """Load tile voxelizer via CuPy NVRTC (CUDA 12.x runtime compilation).

    Falls back to this when torch.utils.cpp_extension fails due to CUDA
    version mismatch between system nvcc (13.x) and runtime (12.x).
    """
    global _cupy_module, _cupy_load_error
    if _cupy_module is not None:
        return _cupy_module
    if _cupy_load_error is not None:
        if strict:
            raise RuntimeError(_cupy_load_error)
        return None
    try:
        import cupy as cp  # type: ignore

        _cupy_module = cp.RawModule(
            code=_TILE_KERNEL_SRC,
            options=("--use_fast_math",),
            name_expressions=[
                "tile_voxelize_forward_cupy",
                "tile_voxelize_backward_cupy",
                # Phase 11 Plan 11-01 — masked forward for CF Δ-render parity.
                "tile_voxelize_forward_cupy_masked",
            ],
        )
        # Validate: trigger compilation + small test
        fwd_fn = _cupy_module.get_function("tile_voxelize_forward_cupy")
        print(f"[TileVoxelizer] CuPy NVRTC compilation succeeded (CUDA 12.x compatible).")
        return _cupy_module
    except Exception as exc:  # pragma: no cover
        _cupy_load_error = f"CuPy NVRTC compilation failed: {exc}"
        if strict:
            raise RuntimeError(_cupy_load_error) from exc
        print(f"[TileVoxelizer] {_cupy_load_error}")
        return None


def _tensor_to_cupy(t: "torch.Tensor"):
    """Zero-copy conversion from PyTorch CUDA tensor to CuPy array.

    Uses torch.as_tensor for round-trip compatibility via the CUDA data pointer.
    Detaches grad-tracked tensors since CuPy operates on raw data.
    """
    import cupy as cp  # type: ignore

    if t.requires_grad:
        t = t.detach()
    # Use __cuda_array_interface__ for robust zero-copy transfer
    return cp.ndarray(
        shape=t.shape,
        dtype=cp.float32,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(t.data_ptr(), t.numel() * t.element_size(), t),
            0,
        ),
    )


def _cupy_to_tensor(a, dtype=None) -> "torch.Tensor":
    """Zero-copy conversion from CuPy array to PyTorch tensor."""
    import torch  # noqa: F811 — already imported at module level

    return torch.as_tensor(a, device="cuda")


def _tile_forward_cupy(center_vox, cov_inv, density, half_shape, tile_ranges, pair_ids, d_size, h_size, w_size, tile_size):
    """Forward pass using CuPy NVRTC kernel (CUDA 12.x compatible)."""
    import cupy as cp  # type: ignore

    mod = _load_cupy_ext(strict=False)
    if mod is None:
        return None  # caller falls back to Python

    fwd_fn = mod.get_function("tile_voxelize_forward_cupy")
    num_tiles = tile_ranges.shape[0]

    centers_cp = _tensor_to_cupy(center_vox.contiguous().float())
    cov_cp = _tensor_to_cupy(cov_inv.reshape(-1, 9).contiguous().float())
    den_r_cp = _tensor_to_cupy(density.real.contiguous().float())
    den_i_cp = _tensor_to_cupy(density.imag.contiguous().float())
    tr_cp = _tensor_to_cupy(tile_ranges.contiguous())
    pi_cp = _tensor_to_cupy(pair_ids.contiguous())
    hs_cp = _tensor_to_cupy(half_shape.contiguous().float())

    vol_r_cp = cp.zeros(d_size * h_size * w_size, dtype=cp.float32)
    vol_i_cp = cp.zeros(d_size * h_size * w_size, dtype=cp.float32)

    fwd_fn(
        (num_tiles,), (256,),
        (centers_cp, cov_cp, den_r_cp, den_i_cp, tr_cp, pi_cp, hs_cp,
         vol_r_cp, vol_i_cp,
         cp.int32(tile_size), cp.int32(d_size), cp.int32(h_size), cp.int32(w_size), cp.int32(num_tiles)),
    )

    vol_r = _cupy_to_tensor(vol_r_cp).view(d_size, h_size, w_size)
    vol_i = _cupy_to_tensor(vol_i_cp).view(d_size, h_size, w_size)
    return vol_r, vol_i


class _TileVoxelizeCuPy(torch.autograd.Function):
    """CuPy-based autograd function for tile voxelization (CUDA 12.x compat)."""

    @staticmethod
    def forward(ctx, center_vox, cov_inv, density_r, density_i, tile_ranges, pair_ids, half_shape, tile_size, d_size, h_size, w_size):
        import cupy as cp  # type: ignore

        mod = _load_cupy_ext(strict=True)
        fwd_fn = mod.get_function("tile_voxelize_forward_cupy")
        num_tiles = int(tile_ranges.shape[0])

        cov_flat = cov_inv.reshape(-1, 9).contiguous().float()
        centers_cp = _tensor_to_cupy(center_vox.contiguous().float())
        cov_cp = _tensor_to_cupy(cov_flat)
        den_r_cp = _tensor_to_cupy(density_r.contiguous().float())
        den_i_cp = _tensor_to_cupy(density_i.contiguous().float())
        tr_cp = _tensor_to_cupy(tile_ranges.contiguous())
        pi_cp = _tensor_to_cupy(pair_ids.contiguous())
        hs_cp = _tensor_to_cupy(half_shape.contiguous().float())

        vol_r_cp = cp.zeros(d_size * h_size * w_size, dtype=cp.float32)
        vol_i_cp = cp.zeros(d_size * h_size * w_size, dtype=cp.float32)

        fwd_fn(
            (num_tiles,), (256,),
            (centers_cp, cov_cp, den_r_cp, den_i_cp, tr_cp, pi_cp, hs_cp,
             vol_r_cp, vol_i_cp,
             cp.int32(tile_size), cp.int32(d_size), cp.int32(h_size), cp.int32(w_size), cp.int32(num_tiles)),
        )

        ctx.save_for_backward(center_vox, cov_inv, density_r, density_i, tile_ranges, pair_ids, half_shape)
        ctx.tile_size = tile_size
        ctx.dims = (d_size, h_size, w_size)
        return _cupy_to_tensor(vol_r_cp), _cupy_to_tensor(vol_i_cp)

    @staticmethod
    def backward(ctx, grad_vol_r, grad_vol_i):
        import cupy as cp  # type: ignore

        center_vox, cov_inv, density_r, density_i, tile_ranges, pair_ids, half_shape = ctx.saved_tensors
        d_size, h_size, w_size = ctx.dims
        tile_size = ctx.tile_size
        num_tiles = int(tile_ranges.shape[0])
        N = center_vox.shape[0]

        mod = _load_cupy_ext(strict=True)
        bwd_fn = mod.get_function("tile_voxelize_backward_cupy")

        cov_flat = cov_inv.reshape(-1, 9).contiguous().float()
        centers_cp = _tensor_to_cupy(center_vox.contiguous().float())
        cov_cp = _tensor_to_cupy(cov_flat)
        den_r_cp = _tensor_to_cupy(density_r.contiguous().float())
        den_i_cp = _tensor_to_cupy(density_i.contiguous().float())
        tr_cp = _tensor_to_cupy(tile_ranges.contiguous())
        pi_cp = _tensor_to_cupy(pair_ids.contiguous())
        hs_cp = _tensor_to_cupy(half_shape.contiguous().float())
        gvr_cp = _tensor_to_cupy(grad_vol_r.contiguous().float())
        gvi_cp = _tensor_to_cupy(grad_vol_i.contiguous().float())

        g_centers = cp.zeros(N * 3, dtype=cp.float32)
        g_cov = cp.zeros(N * 9, dtype=cp.float32)
        g_den_r = cp.zeros(N, dtype=cp.float32)
        g_den_i = cp.zeros(N, dtype=cp.float32)

        bwd_fn(
            (num_tiles,), (256,),
            (centers_cp, cov_cp, den_r_cp, den_i_cp, tr_cp, pi_cp, hs_cp,
             gvr_cp, gvi_cp, g_centers, g_cov, g_den_r, g_den_i,
             cp.int32(tile_size), cp.int32(d_size), cp.int32(h_size), cp.int32(w_size), cp.int32(num_tiles)),
        )

        grad_c = _cupy_to_tensor(g_centers).view(N, 3)
        grad_cov = _cupy_to_tensor(g_cov).view(N, 3, 3)
        grad_dr = _cupy_to_tensor(g_den_r)
        grad_di = _cupy_to_tensor(g_den_i)
        return grad_c, grad_cov, grad_dr, grad_di, None, None, None, None, None, None, None


def _quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    norm = q.norm(dim=-1, keepdim=True)
    q = q / (norm + 1e-8)
    w, x, y, z = q.unbind(-1)
    r0 = torch.stack([1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w], dim=-1)
    r1 = torch.stack([2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w], dim=-1)
    r2 = torch.stack([2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y], dim=-1)
    return torch.stack([r0, r1, r2], dim=-2)


def _bbox_radius_vox(rotation_matrix, scales, half_shape):
    """Per-axis half-extent, in voxels, of the region the kernel actually integrates.

    The kernel contributes wherever mahal <= 9, with mahal measured in normalised coordinates
    (d = (voxel - centre) / half_shape). That ellipsoid's bounding box has half-extent
    3*sqrt(diag(Sigma)) per normalised axis, for Sigma = R diag(scales^2) R^T, hence
    3*sqrt(diag(Sigma))*half_shape in voxels.

    This bound must never be shrunk. build_tile_pairs assigns a gaussian to the tiles its bbox
    touches and the kernel applies no radius test of its own, so a bbox smaller than the
    support truncates the gaussian at whichever tile edge it happens to reach — which makes the
    render depend on tile_size. The earlier isotropic form 3*max(scales*half_shape) broke that
    twice: it was clamped to max_radius, and being isotropic it under-covered the long axes
    whenever the largest scale sat on a short one.
    """
    sigma_diag = (rotation_matrix ** 2 * (scales ** 2).unsqueeze(1)).sum(-1)
    return 3.0 * sigma_diag.sqrt() * half_shape


def build_tile_pairs(center_vox, radius_vox, volume_shape, tile_size):
    """radius_vox is (N,3) per-axis, or (N,) if the caller means an isotropic radius."""
    d_size, h_size, w_size = volume_shape
    device = center_vox.device
    n_points = center_vox.shape[0]

    n_td = math.ceil(d_size / tile_size)
    n_th = math.ceil(h_size / tile_size)
    n_tw = math.ceil(w_size / tile_size)
    num_tiles = n_td * n_th * n_tw

    r3 = radius_vox if radius_vox.dim() == 2 else radius_vox.unsqueeze(1).expand(-1, 3)
    bbox_min_vox = (center_vox - r3).floor().int()
    bbox_max_vox = (center_vox + r3).ceil().int()
    bbox_min_vox[:, 0].clamp_(0, d_size - 1)
    bbox_min_vox[:, 1].clamp_(0, h_size - 1)
    bbox_min_vox[:, 2].clamp_(0, w_size - 1)
    bbox_max_vox[:, 0].clamp_(0, d_size - 1)
    bbox_max_vox[:, 1].clamp_(0, h_size - 1)
    bbox_max_vox[:, 2].clamp_(0, w_size - 1)

    tile_min = bbox_min_vox // tile_size
    tile_max = bbox_max_vox // tile_size
    tile_min[:, 0].clamp_(0, n_td - 1)
    tile_min[:, 1].clamp_(0, n_th - 1)
    tile_min[:, 2].clamp_(0, n_tw - 1)
    tile_max[:, 0].clamp_(0, n_td - 1)
    tile_max[:, 1].clamp_(0, n_th - 1)
    tile_max[:, 2].clamp_(0, n_tw - 1)

    n_per = tile_max - tile_min + 1
    n_tiles_per = n_per[:, 0] * n_per[:, 1] * n_per[:, 2]
    total_pairs = int(n_tiles_per.sum().item())

    if total_pairs == 0:
        return (
            torch.zeros(num_tiles, 2, dtype=torch.int32, device=device),
            torch.zeros(0, dtype=torch.int32, device=device),
        )

    gauss_ids = torch.repeat_interleave(torch.arange(n_points, device=device, dtype=torch.int32), n_tiles_per)
    cumulative = torch.cat(
        [torch.zeros(1, device=device, dtype=torch.int64), n_tiles_per.cumsum(0)[:-1]],
        dim=0,
    )
    local_idx = torch.arange(total_pairs, device=device, dtype=torch.int64) - torch.repeat_interleave(cumulative, n_tiles_per)
    rng_h = torch.repeat_interleave(n_per[:, 1], n_tiles_per)
    rng_w = torch.repeat_interleave(n_per[:, 2], n_tiles_per)
    ld = local_idx // (rng_h * rng_w)
    lr = local_idx % (rng_h * rng_w)
    lh = lr // rng_w
    lw = lr % rng_w

    tmin_d = torch.repeat_interleave(tile_min[:, 0], n_tiles_per)
    tmin_h = torch.repeat_interleave(tile_min[:, 1], n_tiles_per)
    tmin_w = torch.repeat_interleave(tile_min[:, 2], n_tiles_per)
    pair_tile_d = tmin_d + ld.int()
    pair_tile_h = tmin_h + lh.int()
    pair_tile_w = tmin_w + lw.int()
    pair_tile_ids = (pair_tile_d * n_th * n_tw + pair_tile_h * n_tw + pair_tile_w).int()

    sort_idx = torch.argsort(pair_tile_ids, stable=True)
    pair_tile_ids_sorted = pair_tile_ids[sort_idx]
    pair_ids_sorted = gauss_ids[sort_idx]

    tile_ranges = torch.zeros(num_tiles, 2, dtype=torch.int32, device=device)
    changes = torch.cat(
        [torch.tensor([True], device=device), pair_tile_ids_sorted[1:] != pair_tile_ids_sorted[:-1]],
        dim=0,
    )
    boundary_pos = torch.nonzero(changes, as_tuple=True)[0]
    boundary_tiles = pair_tile_ids_sorted[boundary_pos]
    starts = boundary_pos
    ends = torch.cat([boundary_pos[1:], torch.tensor([total_pairs], device=device, dtype=torch.int64)], dim=0)
    tile_ranges[boundary_tiles.long(), 0] = starts.int()
    tile_ranges[boundary_tiles.long(), 1] = ends.int()
    return tile_ranges, pair_ids_sorted


class _TileVoxelizeCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, center_vox, cov_inv, density_r, density_i, tile_ranges, pair_ids, half_shape, tile_size, d_size, h_size, w_size):
        cuda = _load_cuda_ext(strict=True)
        vol_r, vol_i = cuda.forward(
            center_vox.contiguous().float(),
            cov_inv.contiguous().float(),
            density_r.contiguous().float(),
            density_i.contiguous().float(),
            tile_ranges.contiguous().int(),
            pair_ids.contiguous().int(),
            half_shape.contiguous().float(),
            tile_size,
            d_size,
            h_size,
            w_size,
        )
        ctx.save_for_backward(center_vox, cov_inv, density_r, density_i, tile_ranges, pair_ids, half_shape)
        ctx.tile_size = tile_size
        ctx.dims = (d_size, h_size, w_size)
        return vol_r, vol_i

    @staticmethod
    def backward(ctx, grad_vol_r, grad_vol_i):
        center_vox, cov_inv, density_r, density_i, tile_ranges, pair_ids, half_shape = ctx.saved_tensors
        d_size, h_size, w_size = ctx.dims
        cuda = _load_cuda_ext(strict=True)
        grad_centers, grad_cov_inv, grad_den_r, grad_den_i = cuda.backward(
            center_vox.contiguous().float(),
            cov_inv.contiguous().float(),
            density_r.contiguous().float(),
            density_i.contiguous().float(),
            tile_ranges.contiguous().int(),
            pair_ids.contiguous().int(),
            half_shape.contiguous().float(),
            grad_vol_r.contiguous().float(),
            grad_vol_i.contiguous().float(),
            ctx.tile_size,
            d_size,
            h_size,
            w_size,
        )
        return grad_centers, grad_cov_inv, grad_den_r, grad_den_i, None, None, None, None, None, None, None


class _TileVoxelizeCUDAWithMask(torch.autograd.Function):
    """Phase 11 Plan 11-01 — masked tile forward (no backward).

    Per-Gaussian binary gate ``z[N]`` multiplies each staged density value at
    kernel-load time. Used by ``TileVoxelizer.render_with_mask`` for the
    counterfactual LOO Δ-render primitive (CF-01 / CF-02). CF is no-grad —
    autograd is intentionally NOT registered. Callers must wrap the dispatch
    in ``torch.no_grad()`` to avoid an autograd-graph attachment surprise.
    """

    @staticmethod
    def forward(
        ctx,
        center_vox,
        cov_inv,
        density_r,
        density_i,
        z,
        tile_ranges,
        pair_ids,
        half_shape,
        tile_size,
        d_size,
        h_size,
        w_size,
    ):
        cuda = _load_cuda_ext(strict=True)
        vol_r, vol_i = cuda.forward_with_mask(
            center_vox.contiguous().float(),
            cov_inv.contiguous().float(),
            density_r.contiguous().float(),
            density_i.contiguous().float(),
            z.contiguous().to(torch.int8),
            tile_ranges.contiguous().int(),
            pair_ids.contiguous().int(),
            half_shape.contiguous().float(),
            tile_size,
            d_size,
            h_size,
            w_size,
        )
        return vol_r, vol_i

    @staticmethod
    def backward(ctx, grad_vol_r, grad_vol_i):  # pragma: no cover — no-grad path
        raise RuntimeError(
            "_TileVoxelizeCUDAWithMask has no backward; CF Δ-render is no-grad. "
            "Wrap render_with_mask() calls in torch.no_grad()."
        )


def _tile_forward_with_mask_cupy(
    center_vox,
    cov_inv,
    density_r,
    density_i,
    z,
    tile_ranges,
    pair_ids,
    half_shape,
    d_size,
    h_size,
    w_size,
    tile_size,
):
    """CuPy NVRTC fallback for the masked forward (parity with C++/CUDA).

    Returns ``(vol_r, vol_i)`` as torch tensors of shape ``(d_size * h_size * w_size,)``
    matching the C++ extension's contract. Used only when the PyTorch JIT path
    fails to load and CuPy NVRTC compilation succeeds.
    """
    import cupy as cp  # type: ignore

    mod = _load_cupy_ext(strict=True)
    fwd_fn = mod.get_function("tile_voxelize_forward_cupy_masked")
    num_tiles = int(tile_ranges.shape[0])

    cov_flat = cov_inv.reshape(-1, 9).contiguous().float()
    centers_cp = _tensor_to_cupy(center_vox.contiguous().float())
    cov_cp = _tensor_to_cupy(cov_flat)
    den_r_cp = _tensor_to_cupy(density_r.contiguous().float())
    den_i_cp = _tensor_to_cupy(density_i.contiguous().float())
    z_int8 = z.contiguous().to(torch.int8)
    z_cp = cp.ndarray(
        shape=z_int8.shape,
        dtype=cp.int8,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(z_int8.data_ptr(), z_int8.numel() * z_int8.element_size(), z_int8),
            0,
        ),
    )
    tr_cp = _tensor_to_cupy(tile_ranges.contiguous())
    pi_cp = _tensor_to_cupy(pair_ids.contiguous())
    hs_cp = _tensor_to_cupy(half_shape.contiguous().float())

    vol_r_cp = cp.zeros(d_size * h_size * w_size, dtype=cp.float32)
    vol_i_cp = cp.zeros(d_size * h_size * w_size, dtype=cp.float32)

    fwd_fn(
        (num_tiles,), (256,),
        (centers_cp, cov_cp, den_r_cp, den_i_cp, z_cp, tr_cp, pi_cp, hs_cp,
         vol_r_cp, vol_i_cp,
         cp.int32(tile_size), cp.int32(d_size), cp.int32(h_size), cp.int32(w_size), cp.int32(num_tiles)),
    )

    return _cupy_to_tensor(vol_r_cp), _cupy_to_tensor(vol_i_cp)


def _tile_forward_pytorch(center_vox, cov_inv, density, half_shape, tile_ranges, pair_ids, d_size, h_size, w_size, tile_size):
    num_tiles = tile_ranges.shape[0]
    n_th = math.ceil(h_size / tile_size)
    n_tw = math.ceil(w_size / tile_size)
    volume_flat = torch.zeros(d_size * h_size * w_size, dtype=torch.complex64, device=center_vox.device)

    for tile_id in range(num_tiles):
        g_start = int(tile_ranges[tile_id, 0].item())
        g_end = int(tile_ranges[tile_id, 1].item())
        if g_start >= g_end:
            continue
        gids = pair_ids[g_start:g_end].long()
        td = tile_id // (n_th * n_tw)
        rem = tile_id % (n_th * n_tw)
        th = rem // n_tw
        tw = rem % n_tw
        vd_s, vd_e = td * tile_size, min(td * tile_size + tile_size, d_size)
        vh_s, vh_e = th * tile_size, min(th * tile_size + tile_size, h_size)
        vw_s, vw_e = tw * tile_size, min(tw * tile_size + tile_size, w_size)
        vd = torch.arange(vd_s, vd_e, device=center_vox.device, dtype=torch.float32)
        vh = torch.arange(vh_s, vh_e, device=center_vox.device, dtype=torch.float32)
        vw = torch.arange(vw_s, vw_e, device=center_vox.device, dtype=torch.float32)
        gd, gh, gw = torch.meshgrid(vd, vh, vw, indexing="ij")
        voxels = torch.stack([gd, gh, gw], dim=-1).reshape(-1, 3)
        centers = center_vox[gids]
        mats = cov_inv[gids]
        rho = density[gids]
        diff_vox = voxels.unsqueeze(1) - centers.unsqueeze(0)
        diff_norm = diff_vox / half_shape.unsqueeze(0).unsqueeze(0)
        mahal = torch.einsum("vgd,gde,vge->vg", diff_norm, mats, diff_norm)
        weights = torch.exp(-0.5 * mahal) * (mahal <= 9.0).float()
        contrib = (weights * rho.unsqueeze(0)).sum(dim=1)
        flat_idx = gd.reshape(-1).long() * h_size * w_size + gh.reshape(-1).long() * w_size + gw.reshape(-1).long()
        volume_flat[flat_idx] = contrib
    return volume_flat.view(d_size, h_size, w_size)


class TileVoxelizer(nn.Module):
    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        tile_size: int = 8,
        max_radius: int = 20,
        use_cuda: bool = True,
        strict_cuda: bool = False,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.volume_shape = volume_shape
        self.tile_size = tile_size
        self.max_radius = max_radius
        self.device = device
        self.strict_cuda = strict_cuda
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.use_cupy = False
        self.backend = "tile"
        if self.use_cuda:
            ext = _load_cuda_ext(strict=False)  # validation inside; error re-raised if strict
            if ext is None:
                self.use_cuda = False
                # Try CuPy NVRTC fallback (CUDA 12.x compatible, no nvcc needed)
                cupy_mod = _load_cupy_ext(strict=False)
                if cupy_mod is not None:
                    self.use_cupy = True
                    self.backend = "tile_cupy"
                elif strict_cuda:
                    raise RuntimeError("tile_cuda was requested in strict mode, but both the CUDA extension and CuPy NVRTC are unavailable.")
            else:
                self.backend = "tile_cuda"
        if not self.use_cuda and not self.use_cupy:
            if strict_cuda:
                raise RuntimeError("tile_cuda was requested in strict mode, but the CUDA extension is unavailable.")
            self.backend = "tile"
        print(
            f"[TileVoxelizer] volume={volume_shape}, tile_size={tile_size}, "
            f"max_radius={max_radius} (accepted for config compatibility; the bbox is the exact "
            f"mahal<=9 extent and is never clamped), backend={self.backend}"
        )

    def forward(self, positions: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor, density: torch.Tensor, chunk_size: int = 4096):
        del chunk_size
        d_size, h_size, w_size = self.volume_shape
        shape_tensor = torch.tensor([d_size, h_size, w_size], device=positions.device, dtype=torch.float32)
        half_shape = shape_tensor * 0.5
        center_vox = (positions + 1.0) * 0.5 * shape_tensor - 0.5
        rotation_matrix = _quaternion_to_rotation_matrix(rotations)
        radius_vox = _bbox_radius_vox(rotation_matrix, scales, half_shape)
        s_inv = torch.reciprocal(scales + 1e-8)
        transform = rotation_matrix * s_inv.unsqueeze(1)
        cov_inv = transform @ transform.transpose(1, 2)

        with torch.no_grad():
            tile_ranges, pair_ids = build_tile_pairs(center_vox.detach(), radius_vox.detach(), self.volume_shape, self.tile_size)

        if self.use_cuda:
            vol_r, vol_i = _TileVoxelizeCUDA.apply(
                center_vox,
                cov_inv,
                density.real.contiguous(),
                density.imag.contiguous(),
                tile_ranges,
                pair_ids,
                half_shape,
                self.tile_size,
                d_size,
                h_size,
                w_size,
            )
            return vol_r.view(d_size, h_size, w_size), vol_i.view(d_size, h_size, w_size)

        if self.use_cupy:
            vol_r, vol_i = _TileVoxelizeCuPy.apply(
                center_vox,
                cov_inv,
                density.real.contiguous(),
                density.imag.contiguous(),
                tile_ranges,
                pair_ids,
                half_shape,
                self.tile_size,
                d_size,
                h_size,
                w_size,
            )
            return vol_r.view(d_size, h_size, w_size), vol_i.view(d_size, h_size, w_size)

        return _tile_forward_pytorch(
            center_vox,
            cov_inv,
            density,
            half_shape,
            tile_ranges,
            pair_ids,
            d_size,
            h_size,
            w_size,
            self.tile_size,
        )

    @torch.no_grad()
    def render_with_mask(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        density: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Phase 11 Plan 11-01 — forward render with per-Gaussian binary gate.

        z ∈ {0, 1}^N gates each Gaussian's contribution before tile accumulation.
        z=ones is the identity vs ``forward()`` (within fp32 roundoff); z[idx]=0
        is the ``do(z_idx = 0)`` counterfactual intervention that drops that
        Gaussian from the rendered volume — the LOO Δ-render primitive used by
        ``CounterfactualAnalyzer`` (CF-01 / CF-02).

        Parameters
        ----------
        positions : (N, 3) float
            Normalized ``[-1, 1]`` Gaussian centres.
        scales : (N, 3) float
            Per-axis scales (positive, not log-parameterized).
        rotations : (N, 4) float
            Unit quaternions.
        density : (N,) complex64
            Per-Gaussian amplitudes.
        z : (N,) int8 (or castable)
            Binary gate; non-binary values are also valid as a soft
            attenuation, but Phase 11 only passes 0/1.

        Returns
        -------
        volume : (D, H, W) complex64
            Rendered volume with gated contributions.

        Notes
        -----
        * Backward is intentionally disabled — CF is no-grad.
        * Returns a single complex tensor, NOT a (vol_r, vol_i) tuple.
        * Backend dispatch matches ``forward()``: CUDA JIT (primary), CuPy
          NVRTC (fallback), pure-PyTorch (last resort).
        """
        if z.shape[0] != positions.shape[0]:
            raise ValueError(
                f"z shape {tuple(z.shape)} incompatible with N={positions.shape[0]}; "
                "expected (N,) binary gate"
            )

        d_size, h_size, w_size = self.volume_shape
        shape_tensor = torch.tensor(
            [d_size, h_size, w_size], device=positions.device, dtype=torch.float32
        )
        half_shape = shape_tensor * 0.5
        center_vox = (positions + 1.0) * 0.5 * shape_tensor - 0.5
        rotation_matrix = _quaternion_to_rotation_matrix(rotations)
        radius_vox = _bbox_radius_vox(rotation_matrix, scales, half_shape)
        s_inv = torch.reciprocal(scales + 1e-8)
        transform = rotation_matrix * s_inv.unsqueeze(1)
        cov_inv = transform @ transform.transpose(1, 2)

        tile_ranges, pair_ids = build_tile_pairs(
            center_vox, radius_vox, self.volume_shape, self.tile_size
        )

        if self.use_cuda:
            vol_r, vol_i = _TileVoxelizeCUDAWithMask.apply(
                center_vox,
                cov_inv,
                density.real.contiguous(),
                density.imag.contiguous(),
                z,
                tile_ranges,
                pair_ids,
                half_shape,
                self.tile_size,
                d_size,
                h_size,
                w_size,
            )
            return torch.complex(
                vol_r.view(d_size, h_size, w_size),
                vol_i.view(d_size, h_size, w_size),
            )

        if self.use_cupy:
            vol_r, vol_i = _tile_forward_with_mask_cupy(
                center_vox,
                cov_inv,
                density.real.contiguous(),
                density.imag.contiguous(),
                z,
                tile_ranges,
                pair_ids,
                half_shape,
                d_size,
                h_size,
                w_size,
                self.tile_size,
            )
            return torch.complex(
                vol_r.view(d_size, h_size, w_size),
                vol_i.view(d_size, h_size, w_size),
            )

        # Pure-PyTorch fallback: pre-multiply density by gate then dispatch
        # to the existing _tile_forward_pytorch — preserves the contract
        # without needing a separate masked Python kernel.
        z_complex = z.to(density.real.dtype).to(density.device)
        gated_density = density * z_complex
        vol = _tile_forward_pytorch(
            center_vox,
            cov_inv,
            gated_density,
            half_shape,
            tile_ranges,
            pair_ids,
            d_size,
            h_size,
            w_size,
            self.tile_size,
        )
        return vol
