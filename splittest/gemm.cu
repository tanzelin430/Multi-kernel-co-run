#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>
__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
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

void matmul_host(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float value = 0;
            for (int k = 0; k < K; k++) {
                value += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = value;
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <M>\n", argv[0]);
        return 1;
    }
    int M = atoi(argv[1]);
    int N = 4096;
    int K = 4096;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *h_A, *h_B, *h_C, *h_C_ref;
    float *d_A, *d_B, *d_C;

    // Allocate host memory
    h_A = (float *)malloc(sizeA);
    h_B = (float *)malloc(sizeB);
    h_C = (float *)malloc(sizeC);
    h_C_ref = (float *)malloc(sizeC);
    srand(time(NULL));
    // Initialize host arrays
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] = 1.0f + rand()%5; // Example value
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B[i * N + j] = 1.0f + rand()%5; // Example value
        }
    }

    // Allocate device memory
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    // Copy data to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    for (size_t i = 0; i < 5; i++)
    {
        matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("%f,", elapsedTime);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Verify the result on the host
    // matmul_host(h_A, h_B, h_C_ref, M, N, K);
    // for (int i = 0; i < M * N; i++) {
    //     if (fabs(h_C[i] - h_C_ref[i]) > 1e-6) {
    //         printf("Result verification failed at element %d! device: %f, host: %f\n", i, h_C[i], h_C_ref[i]);
    //         break;
    //     }
    // }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}