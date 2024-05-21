#include <stdio.h>
#include <cuda_runtime.h>

__global__ void gemv(float *A, float *x, float *y, int m, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("called")
    if (idx < m) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            sum += A[idx * n + j] * x[j];
        }
        y[idx] = sum;
    }
}

extern "C" {
void launch_gemv(float *A, float *x, float *y, int m, int n) {
    dim3 blockDim(256);
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x);
    gemv<<<gridDim, blockDim>>>(A, x, y, m, n);
    cudaDeviceSynchronize();
}
}