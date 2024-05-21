#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>
unsigned int max_SM_V100 = 80;
void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}
unsigned int * _SMC_initiativeArray(unsigned int max_SM_V100){
    unsigned int * _SMC_workerCount = (unsigned int *)malloc(max_SM_V100*sizeof(unsigned int));
    for (int idx = 0; idx < max_SM_V100; idx++)
    {
        _SMC_workerCount[idx] = 0;
    }
    return _SMC_workerCount;
}
unsigned int * _SMC_buildChunkSeq(unsigned int _SMC_chunksPerSM, unsigned int max_SM_V100){
    unsigned int total_chunk = _SMC_chunksPerSM*max_SM_V100;
    unsigned int * _SMC_newChunkSeq = (unsigned int *)malloc(total_chunk*sizeof(unsigned int));
    for (int idx = 0; idx < total_chunk; idx++)
    {
        _SMC_newChunkSeq[idx] = idx;
    }
    return _SMC_newChunkSeq;
}
#define _SMC_getSMid unsigned int _SMC_smid; asm("mov.u32 %0, %smid;" : "=r"(_SMC_smid))
#define _SMC_init \
unsigned int _SMC_workersNeeded = 15; /*每个SM需要的blk数，直接打满，filling策略*/ \
unsigned int _SMC_chunksPerSM = 60; \
unsigned int * h_SMC_newChunkSeq = _SMC_buildChunkSeq(_SMC_chunksPerSM, max_SM_V100); /*新的chunk序列*/ \
unsigned int * h_SMC_workerCount = _SMC_initiativeArray(max_SM_V100); /*每个SM的工作数*/ \
unsigned int *d_SMC_workerCount, *d_SMC_newChunkSeq;\
checkCudaError(cudaMalloc((void **)&d_SMC_workerCount, max_SM_V100 * sizeof(unsigned int)));\
checkCudaError(cudaMalloc((void **)&d_SMC_newChunkSeq, _SMC_chunksPerSM * max_SM_V100 * sizeof(unsigned int)));\
checkCudaError(cudaMemcpy(d_SMC_workerCount, h_SMC_workerCount, max_SM_V100 * sizeof(unsigned int), cudaMemcpyHostToDevice));\
checkCudaError(cudaMemcpy(d_SMC_newChunkSeq, h_SMC_newChunkSeq, _SMC_chunksPerSM * max_SM_V100 * sizeof(unsigned int), cudaMemcpyHostToDevice));\
const int max_thread_num_per_SM_V100 = 2048;\
const int max_block_per_SM_V100 = 32;\
const int host_sm_v100 = 80;\
int blockSize = max_thread_num_per_SM_V100 / max_block_per_SM_V100; /*一个block里面有64个线程*/  \
int gridSize = host_sm_v100*max_block_per_SM_V100*100; \

#define _SMC_Begin \
    __shared__ int _SMC_workingCTAs; \
    _SMC_getSMid; \
    if(threadIdx.x == 0) \
        _SMC_workingCTAs = \
        atomicInc(&_SMC_workerCount[_SMC_smid], INT_MAX);  /*create a workercount for this SM */ \
    __syncthreads(); \
    if(_SMC_workingCTAs >= _SMC_workersNeeded) return;  /*exit*/ \
    int _SMC_chunksPerCTA = \
        (_SMC_chunksPerSM / _SMC_workersNeeded); /*这个应该是一个常数，在这份代码中等于2*/\ 
    int _SMC_startChunkIdx = _SMC_smid * _SMC_chunksPerSM + \
        _SMC_workingCTAs * _SMC_chunksPerCTA; /*发起chunk的线性坐标*/\ 
    int _SMC_chunkID; \
    for(int _SMC_chunkIdx = _SMC_startChunkIdx; \
        _SMC_chunkIdx < _SMC_startChunkIdx + _SMC_chunksPerCTA; \
        _SMC_chunkIdx++){ /*persistent thread的思路*/\
            _SMC_chunkID = _SMC_newChunkSeq[_SMC_chunkIdx];  /*从新的序列中获取工作索引,_SMC_chunkidx是一个全局线性的坐标*/
#define _SMC_End }

__global__ void gemv_kernel(const float *A, const float *B, float *C, int N, unsigned int * _SMC_workerCount, unsigned int * _SMC_newChunkSeq, unsigned int _SMC_chunksPerSM, unsigned int _SMC_workersNeeded) {
    _SMC_Begin
    // if(_SMC_smid > 1) return;
    int col = _SMC_chunkID * blockDim.x + threadIdx.x;
    // if (threadIdx.x == 0){
    //     printf("smid: %d, workingCTAs: %d, chunkID: %d\n", _SMC_smid, _SMC_workingCTAs, _SMC_chunkID);
    // }
    // __syncthreads();
    if (col < N) {
        // __syncthreads();
        // printf("realsmid: %d, workingCTAs: %d, chunkID: %d, col: %d\n", _SMC_smid, _SMC_workingCTAs, _SMC_chunkID, col);
        // printf("threadid:%d", threadIdx.x);
        float value = 0;
        for (int k = 0; k < N; k++) {
            value += A[k] * B[k * N + col];
        }
        C[col] = value;
    }
    else {
        //wating all thread work to finish
        __syncthreads();
    }
    // __syncthreads();
    _SMC_End
}

void gemv_host(const float *A, const float *B, float *C, int N) {
    printf("testing...");
    for (int col = 0; col < N; col++) {
        float value = 0;
        for (int k = 0; k < N; k++) {
            value += A[k] * B[k * N + col];
        }
        C[col] = value;
    }
}

int main() {
    int N = 4096;
    size_t sizeA = N * sizeof(float);
    size_t sizeB = N * N * sizeof(float);
    size_t sizeC = N * sizeof(float);

    float *h_A, *h_B, *h_C, *h_C_ref;
    float *d_A, *d_B, *d_C;

    // Allocate host memory
    h_A = (float *)malloc(sizeA);
    h_B = (float *)malloc(sizeB);
    h_C = (float *)malloc(sizeC);
    h_C_ref = (float *)malloc(sizeC);

    // Initialize host arrays
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f; // Example value
        for (int j = 0; j < N; j++) {
            h_B[i * N + j] = 1 + rand()%5; // Example value
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
    // int blockSize = 256;
    // int gridSize = (N + blockSize - 1) / blockSize;
    _SMC_init
    gemv_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N,d_SMC_workerCount,d_SMC_newChunkSeq, _SMC_chunksPerSM, _SMC_workersNeeded);

    checkCudaError(cudaMemcpy(h_SMC_workerCount, d_SMC_workerCount, max_SM_V100 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // for(int i=0;i<max_SM_V100;i++){
    //     printf("mysmid: %d, workerCount: %d\n", i, h_SMC_workerCount[i]);
    // }

    // Copy result back to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Verify the result on the host
    gemv_host(h_A, h_B, h_C_ref, N);
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-6) {
            printf("Result verification failed at element %d! device: %f, host: %f\n", i, h_C[i], h_C_ref[i]);
            break;
        }
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // 释放设备端内存
    cudaFree(d_SMC_workerCount);
    cudaFree(d_SMC_newChunkSeq);

    // 释放主机端内存
    free(h_SMC_workerCount);
    free(h_SMC_newChunkSeq);
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}