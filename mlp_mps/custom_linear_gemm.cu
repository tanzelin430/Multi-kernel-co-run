#include <cuda_runtime.h>
#include <torch/extension.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAStream.h>

__global__ void matmul_kernel(float* input, float* weight, float* bias, float* output, int M, int input_size, int output_size) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x < output_size && idx_y < M) {
        float sum = bias[idx_x];
        for (int i = 0; i < input_size; ++i) {
            sum += input[idx_y * input_size + i] * weight[idx_x * input_size + i];
        }
        output[idx_y * output_size + idx_x] = sum;
    }
}

torch::Tensor linear_gemm_cuda_launcher(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {

    int M = input.size(0);
    int input_size = input.size(1);
    int output_size = weight.size(0);

    const int threads_x = 256;
    const int threads_y = 1;
    const int blocks_x = (output_size + threads_x - 1) / threads_x;
    const int blocks_y = (M + threads_y - 1) / threads_y;

    dim3 threads(threads_x, threads_y);
    dim3 blocks(blocks_x, blocks_y);

    auto output = torch::empty({M, output_size}, input.options());
    // at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    matmul_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M,
        input_size,
        output_size
    );

    // cudaDeviceSynchronize();
    return output;
}