#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor linear_cuda_launcher(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);

torch::Tensor linear_cuda_wrapper(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    return linear_cuda_launcher(input, weight, bias);
}

torch::Tensor relu_cuda_launcher(torch::Tensor input);

torch::Tensor relu_cuda_wrapper(torch::Tensor input) {
    return relu_cuda_launcher(input);
}

torch::Tensor linear_gemm_cuda_launcher(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);

torch::Tensor linear_gemm_cuda_wrapper(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    return linear_gemm_cuda_launcher(input, weight, bias);
}

PYBIND11_MODULE(ops, m) {
    m.def("linear_cuda", &linear_cuda_wrapper, "Linear CUDA Wrapper");
    m.def("relu_cuda", &relu_cuda_wrapper, "ReLU CUDA Wrapper");
    m.def("linear_gemm_cuda", &linear_gemm_cuda_wrapper, "Linear GEMM CUDA Wrapper");
}