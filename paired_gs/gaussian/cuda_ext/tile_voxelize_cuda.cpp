#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> tile_voxelize_forward_cuda(
    torch::Tensor centers,
    torch::Tensor cov_inv,
    torch::Tensor density_r,
    torch::Tensor density_i,
    torch::Tensor tile_ranges,
    torch::Tensor pair_ids,
    torch::Tensor half_shape,
    int tile_size,
    int d_size,
    int h_size,
    int w_size);

std::vector<torch::Tensor> tile_voxelize_backward_cuda(
    torch::Tensor centers,
    torch::Tensor cov_inv,
    torch::Tensor density_r,
    torch::Tensor density_i,
    torch::Tensor tile_ranges,
    torch::Tensor pair_ids,
    torch::Tensor half_shape,
    torch::Tensor grad_vol_r,
    torch::Tensor grad_vol_i,
    int tile_size,
    int d_size,
    int h_size,
    int w_size);

// Phase 11 Plan 11-01 — masked forward (CF-01 / CF-02 LOO Δ-render).
// Distinct entry point so the existing `forward` symbol stays
// binary-identical for all Phase 7-10 callers (RESEARCH Pitfall 3).
std::vector<torch::Tensor> tile_voxelize_forward_with_mask_cuda(
    torch::Tensor centers,
    torch::Tensor cov_inv,
    torch::Tensor density_r,
    torch::Tensor density_i,
    torch::Tensor z,
    torch::Tensor tile_ranges,
    torch::Tensor pair_ids,
    torch::Tensor half_shape,
    int tile_size,
    int d_size,
    int h_size,
    int w_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &tile_voxelize_forward_cuda, "Tile voxelization forward (CUDA)");
    m.def("backward", &tile_voxelize_backward_cuda, "Tile voxelization backward (CUDA)");
    m.def("forward_with_mask", &tile_voxelize_forward_with_mask_cuda,
          "Tile voxelization forward with per-Gaussian binary gate z[N] (CUDA, no-grad)");
}
