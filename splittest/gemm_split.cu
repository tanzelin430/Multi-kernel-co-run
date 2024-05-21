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
    int M = 1;
    int N = 4096;
    int K = 4096;
    if (argc != 2){
        printf("Usage: %s <num_kernels>\n", argv[0]);
        return 1;
    }
    int num_kernels = atoi(argv[1]);

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *h_A, *h_B, *h_C, *h_C_ref;
    float *d_A, *d_B, *d_C;
    srand(time(NULL));
    // Allocate host memory
    h_A = (float *)malloc(num_kernels * sizeA);
    h_B = (float *)malloc(sizeB);
    h_C = (float *)malloc(num_kernels * sizeC);
    h_C_ref = (float *)malloc(num_kernels * sizeC);

    // Initialize host arrays
    for (int i = 0; i < num_kernels * M; i++) {
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
    cudaMalloc((void **)&d_A, num_kernels * sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, num_kernels * sizeC);

    // Copy data to device
    cudaMemcpy(d_A, h_A, num_kernels * sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Warm-up
    dim3 blockDim (32, 1);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    for (size_t j = 0; j < 5; j++)
    {
        for (int i = 0; i < num_kernels; i++) {
            matmul_kernel<<<gridDim, blockDim>>>(d_A + i * M * K, d_B, d_C + i * M * N, M, N, K);
        }
    }


    // Create CUDA streams
    cudaStream_t streams[num_kernels];
    for (int i = 0; i < num_kernels; i++) {
        cudaStreamCreate(&streams[i]);
    }
    // Start timing
    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch 40 kernels
    for (int i = 0; i < num_kernels; i++) {
        matmul_kernel<<<gridDim, blockDim, 0, streams[i]>>>(d_A + i * M * K, d_B, d_C + i * M * N, M, N, K);
    }

    // Stop timing
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, num_kernels * sizeC, cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_kernels; i++) {
        cudaStreamDestroy(streams[i]);
    }
    // Compare the results
    // matmul_host(h_A, h_B, h_C_ref, num_kernels, N, K);
    // for (int i = 0; i < num_kernels * N; i++) {
    //     if (fabs(h_C[i] - h_C_ref[i]) > 1e-6) {
    //         printf("Result verification failed at element %d! device: %f, host: %f\n", i, h_C[i], h_C_ref[i]);
    //         break;
    //     }
    // }

    // Print elapsed time
    printf("%f,", elapsedTime);

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