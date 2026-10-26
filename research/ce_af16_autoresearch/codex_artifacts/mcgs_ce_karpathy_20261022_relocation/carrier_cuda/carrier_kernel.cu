#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <vector>

using torch::Tensor;
using Complex = c10::complex<float>;
constexpr int TILE = 8;
constexpr int THREADS = 256;
constexpr float TWO_PI = 6.2831853071795864769f;

// Each block owns a tile. Threads own disjoint voxels and loop only through
// that tile's selected atoms. No block barriers exist, including ragged tiles.
template <bool BACKWARD>
__global__ void carrier_kernel(const float* centers, const float* matrices, const Complex* density,
    const float* frequency, const int* ranges, const int* ids, int D, int H, int W,
    Complex* output, const Complex* grad_output, float* gc, float* gm, float* gd, float* gf) {
    const int64_t nH = 1+(static_cast<int64_t>(H)-1)/8;
    const int64_t nW = 1+(static_cast<int64_t>(W)-1)/8;
    const int64_t tile = blockIdx.x;
    const int64_t td = tile/(nH*nW), th = (tile/nW)%nH, tw = tile%nW;
    for (int local = threadIdx.x; local < TILE*TILE*TILE; local += blockDim.x) {
        const int64_t vd = td*TILE + local/64;
        const int64_t vh = th*TILE + (local/8)%8;
        const int64_t vw = tw*TILE + local%8;
        if (vd >= D || vh >= H || vw >= W) continue;
        const int64_t voxel = (vd*H+vh)*W+vw;
        float result_r = 0, result_i = 0;
        float gr = 0, gi = 0;
        if (BACKWARD) { gr = grad_output[voxel].real(); gi = grad_output[voxel].imag(); }
        for (int pair = ranges[2*tile]; pair < ranges[2*tile+1]; ++pair) {
            const int64_t k = ids[pair];
            const float u[3] = {vd-centers[3*k], vh-centers[3*k+1], vw-centers[3*k+2]};
            const float h[3] = {D*.5f, H*.5f, W*.5f};
            const float d[3] = {u[0]/h[0], u[1]/h[1], u[2]/h[2]};
            float md[3] = {0,0,0};
            for (int a=0; a<3; ++a)
                for (int b=0; b<3; ++b) md[a] += matrices[9*k+3*a+b]*d[b];
            const float mahal = d[0]*md[0]+d[1]*md[1]+d[2]*md[2];
            if (mahal > 9.f) continue;
            const float weight = expf(-.5f*mahal);
            const float theta = TWO_PI*(frequency[3*k]*u[0]+frequency[3*k+1]*u[1]+frequency[3*k+2]*u[2]);
            float sine, cosine;
            sincosf(theta,&sine,&cosine);
            const float sh = sinf(.5f*theta);
            const float cm1 = -2.f*sh*sh;
            const float a = density[k].real(), b = density[k].imag();
            const float zr = a*cm1-b*sine, zi = a*sine+b*cm1;
            if (!BACKWARD) { result_r += weight*zr; result_i += weight*zi; }
            else {
                const float Q = gr*zr+gi*zi;
                const float T = -gr*(a*sine+b*cosine)+gi*(a*cosine-b*sine);
                atomicAdd(gd+2*k, weight*(cm1*gr+sine*gi));
                atomicAdd(gd+2*k+1, weight*(-sine*gr+cm1*gi));
                for (int x=0; x<3; ++x) {
                    atomicAdd(gc+3*k+x, weight*Q*md[x]/h[x]-TWO_PI*weight*T*frequency[3*k+x]);
                    atomicAdd(gf+3*k+x, TWO_PI*weight*T*u[x]);
                    for (int y=0; y<3; ++y)
                        atomicAdd(gm+9*k+3*x+y, -.5f*weight*Q*d[x]*d[y]);
                }
            }
        }
        if (!BACKWARD) output[voxel] = Complex(result_r,result_i);
    }
}

Tensor carrier_forward_cuda(Tensor c, Tensor m, Tensor density, Tensor f, Tensor ranges,
                             Tensor ids, std::vector<int64_t> shape) {
    const c10::cuda::CUDAGuard guard(c.device());
    auto output = torch::zeros(shape,density.options());
    if (c.size(0) == 0) return output;
    auto stream = at::cuda::getCurrentCUDAStream(c.get_device());
    carrier_kernel<false><<<ranges.size(0), THREADS, 0, stream>>>(
        c.data_ptr<float>(),m.data_ptr<float>(),density.data_ptr<Complex>(),f.data_ptr<float>(),
        ranges.data_ptr<int>(),ids.data_ptr<int>(),shape[0],shape[1],shape[2],output.data_ptr<Complex>(),
        nullptr,nullptr,nullptr,nullptr,nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<Tensor> carrier_backward_cuda(Tensor grad, Tensor c, Tensor m, Tensor density,
    Tensor f, Tensor ranges, Tensor ids, std::vector<int64_t> shape) {
    const c10::cuda::CUDAGuard guard(c.device());
    auto gc = torch::zeros_like(c), gm = torch::zeros_like(m);
    auto gd = torch::zeros_like(density), gf = torch::zeros_like(f);
    if (c.size(0) == 0) return {gc,gm,gd,gf};
    auto stream = at::cuda::getCurrentCUDAStream(c.get_device());
    carrier_kernel<true><<<ranges.size(0), THREADS, 0, stream>>>(
        c.data_ptr<float>(),m.data_ptr<float>(),density.data_ptr<Complex>(),f.data_ptr<float>(),
        ranges.data_ptr<int>(),ids.data_ptr<int>(),shape[0],shape[1],shape[2],nullptr,
        grad.data_ptr<Complex>(),gc.data_ptr<float>(),gm.data_ptr<float>(),
        reinterpret_cast<float*>(gd.data_ptr<Complex>()),gf.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gc,gm,gd,gf};
}
