#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void linear_kernel_optimized(float* input, float* weight, float* bias, float* output, int input_size, int output_size, int elements_per_thread) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start_idx = global_idx * elements_per_thread;
    int end_idx = min(output_size, start_idx + elements_per_thread);

    for (int idx = start_idx; idx < end_idx; ++idx) {
        float sum = bias[idx];
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weight[idx * input_size + i];
        }
        output[idx] = sum;
    }
}

torch::Tensor linear_cuda_launcher(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int input_size = input.size(1);
    int output_size = weight.size(0);

    const int threads = 256;
    // Adjust the elements_per_thread based on your specific workload and hardware
    const int elements_per_thread = 4;
    const int blocks = (output_size + elements_per_thread * threads - 1) / (elements_per_thread * threads);

    auto output = torch::empty({input.size(0), output_size}, input.options());
    linear_kernel_optimized<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        input_size,
        output_size,
        elements_per_thread
    );
    return output;
}

