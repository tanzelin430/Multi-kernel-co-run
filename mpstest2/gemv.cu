#include <stdio.h>
#include <cuda_runtime.h>

extern "C" {
    __global__ void gemv(float *A, float *x, float *y, int m, int n) {
        printf("kernel called");
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < m) {
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                sum += A[idx * n + j] * x[j];
            }
            y[idx] = sum;
        }
    }
}


}