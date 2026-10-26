#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <limits>

using torch::Tensor;
Tensor carrier_forward_cuda(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, std::vector<int64_t>);
std::vector<Tensor> carrier_backward_cuda(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, std::vector<int64_t>);

namespace {
void tensor_check(const Tensor& t, const Tensor& centers, at::ScalarType dtype, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.device() == centers.device(), name, " must share the CUDA device");
    TORCH_CHECK(t.scalar_type() == dtype && t.is_contiguous(), name, " has wrong dtype or layout");
}

void validate(Tensor centers, Tensor m, Tensor density, Tensor f, Tensor ranges, Tensor ids,
              const std::vector<int64_t>& shape) {
    TORCH_CHECK(centers.is_cuda(), "centers must be CUDA");
    TORCH_CHECK(shape.size() == 3, "shape must have 3 axes");
    int64_t voxels = 1, tiles = 1;
    for (auto n : shape) {
        TORCH_CHECK(n > 0 && n <= std::numeric_limits<int32_t>::max(), "invalid shape");
        TORCH_CHECK(voxels <= std::numeric_limits<int32_t>::max()/n, "voxel count exceeds int32");
        voxels *= n;
        tiles *= 1+(n-1)/8;
    }
    tensor_check(centers, centers, at::kFloat, "centers");
    tensor_check(m, centers, at::kFloat, "M");
    tensor_check(density, centers, at::kComplexFloat, "density");
    tensor_check(f, centers, at::kFloat, "frequencies");
    tensor_check(ranges, centers, at::kInt, "ranges");
    tensor_check(ids, centers, at::kInt, "ids");
    TORCH_CHECK(centers.dim() == 2 && centers.size(1) == 3, "centers must be [K,3]");
    const auto k = centers.size(0);
    TORCH_CHECK(k <= std::numeric_limits<int32_t>::max(), "K exceeds int32");
    TORCH_CHECK(m.sizes() == torch::IntArrayRef({k,3,3}), "M must be [K,3,3]");
    TORCH_CHECK(density.sizes() == torch::IntArrayRef({k}), "density must be [K]");
    TORCH_CHECK(f.sizes() == torch::IntArrayRef({k,3}), "frequencies must be [K,3]");
    TORCH_CHECK(ranges.sizes() == torch::IntArrayRef({tiles,2}), "invalid tile range shape");
    TORCH_CHECK(ids.dim() == 1 && ids.numel() <= std::numeric_limits<int32_t>::max(), "invalid ids shape/count");
    for (const auto& t : {centers,m,density,f})
        TORCH_CHECK(torch::isfinite(t).all().item<bool>(), "nonfinite geometry/density/frequency");
    TORCH_CHECK(((f > -0.5) & (f < 0.5)).all().item<bool>(), "frequency out of bounds");
    TORCH_CHECK(torch::equal(m, m.transpose(1,2)), "M must be symmetric");
    const auto p = ids.numel();
    auto starts = ranges.select(1,0).to(at::kLong);
    auto ends = ranges.select(1,1).to(at::kLong);
    TORCH_CHECK(((starts >= 0) & (ends >= starts) & (ends <= p)).all().item<bool>(), "unsafe tile ranges");
    TORCH_CHECK(((ids >= 0) & (ids < k)).all().item<bool>(), "unsafe atom index");
    auto active = ends > starts;
    auto s = starts.masked_select(active), e = ends.masked_select(active);
    if (p == 0) {
        TORCH_CHECK(s.numel() == 0, "nonempty ranges without pairs");
    } else {
        TORCH_CHECK(s.numel() > 0 && s[0].item<int64_t>() == 0 && e[-1].item<int64_t>() == p,
                    "tile ranges must cover exactly all pairs");
        TORCH_CHECK(torch::equal(s.slice(0,1), e.slice(0,0,-1)), "tile ranges overlap or leave gaps");
        // Stable construction assigns each atom at most once per tile. Enforce
        // this even for direct bindings before any kernel pointer access.
        auto boundary = torch::zeros({p}, ids.options().dtype(at::kBool));
        boundary.index_fill_(0, s, true);
        auto bad_order = ids.slice(0,1) <= ids.slice(0,0,-1);
        TORCH_CHECK(!(bad_order & ~boundary.slice(0,1)).any().item<bool>(), "duplicate or unsorted tile atom IDs");
    }
}
}

Tensor forward(Tensor centers, Tensor m, Tensor density, Tensor f, Tensor ranges, Tensor ids,
               std::vector<int64_t> shape) {
    TORCH_CHECK(centers.is_cuda(), "centers must be CUDA");
    const c10::cuda::CUDAGuard guard(centers.device());
    validate(centers,m,density,f,ranges,ids,shape);
    return carrier_forward_cuda(centers,m,density,f,ranges,ids,shape);
}

std::vector<Tensor> backward(Tensor grad, Tensor centers, Tensor m, Tensor density, Tensor f,
                             Tensor ranges, Tensor ids, std::vector<int64_t> shape) {
    TORCH_CHECK(centers.is_cuda(), "centers must be CUDA");
    const c10::cuda::CUDAGuard guard(centers.device());
    validate(centers,m,density,f,ranges,ids,shape);
    tensor_check(grad,centers,at::kComplexFloat,"grad_output");
    TORCH_CHECK(grad.sizes() == torch::IntArrayRef(shape), "invalid grad_output shape");
    TORCH_CHECK(torch::isfinite(grad).all().item<bool>(), "nonfinite grad_output");
    return carrier_backward_cuda(grad,centers,m,density,f,ranges,ids,shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward);
    m.def("backward", &backward);
}
