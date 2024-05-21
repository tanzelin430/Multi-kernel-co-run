extern "C" {
    __global__ void gemv_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N) {
            float value = 0;
            for (int k = 0; k < K; k++) {
                value += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = value;
        }
    }



    __global__ void gemm_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N) {
            float value = 0;
            for (int k = 0; k < K; k++) {
                value += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = value;
        }
    }
}