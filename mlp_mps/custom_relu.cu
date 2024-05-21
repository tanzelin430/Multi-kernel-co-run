#include <cuda_runtime.h>
#include <torch/extension.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAStream.h>

__global__ void relu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

torch::Tensor relu_cuda_launcher(torch::Tensor input) {
    int size = input.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    auto output = torch::empty_like(input);
    // at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    cudaDeviceSynchronize();
    return output;
}