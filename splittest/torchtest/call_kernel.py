import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import sys
import time
import os

# Set the desired device ID
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load the custom GEMV kernel
mod = drv.module_from_file("gemm.cubin")
gemm_kernel = mod.get_function("gemm_kernel")


M = 1
N = 4096
K = 4096

def linear(vector, matrix):
    stream = drv.Stream()
    block_size = (32, 1, 1)
    grid_size = (int(np.ceil(N / block_size[0])), int(np.ceil(M / block_size[1])), 1)
    gemm_result = gpuarray.zeros((M, N), np.float32)
    gemm_kernel(vector, matrix, gemm_result, np.int32(M), np.int32(N), np.int32(K), block=block_size, grid=grid_size, stream=stream)
    ret_gemm_result = gemm_result.get()
    return ret_gemm_result


# Generate input matrices and vectors (values between 1 and 5)
matrices = np.random.uniform(1, 5, size=(K,N)).astype(np.float32)
vectors = [np.random.uniform(1, 5, size=(M,K)).astype(np.float32) for _ in range(num_kernels)]

# Generate input matrices for GEMM kernel
matrices_A = np.random.uniform(1, 5, size=(40, K)).astype(np.float32)
matrices_B = np.random.uniform(1, 5, size=(K, N)).astype(np.float32)

# Allocate GPU memory for results
results_gpu = [gpuarray.zeros((M, N), np.float32) for _ in range(num_kernels)]
results_gemm_gpu = gpuarray.zeros((40, N), np.float32)

# transfer data to the GPU asynchrounously with in the stream
matrices_gpu = gpuarray.to_gpu(matrices)
vectors_gpu = [gpuarray.to_gpu(vector) for vector in vectors]
matrices_A_gpu = gpuarray.to_gpu(matrices_A)
# matrices_B_gpu = gpuarray.to_gpu(matrices_B)

block_size_gemm = (32, 8, 1)
grid_size_gemm = (int(np.ceil(N / block_size_gemm[0])), int(np.ceil(40 / block_size_gemm[1])), 1)
# Warm-up
block_size = (32, 1, 1)
grid_size = (int(np.ceil(N / block_size[0])), int(np.ceil(M / block_size[1])), 1)
for _ in range(5):
    stream = drv.Stream()
    gemv_kernel(vectors_gpu[0], matrices_gpu, results_gpu[0], np.int32(M), np.int32(N), np.int32(K),
                block=block_size, grid=grid_size, stream=stream)
    stream = drv.Stream()
    gemm_kernel(matrices_A_gpu, matrices_gpu, results_gemm_gpu, np.int32(40), np.int32(N), np.int32(K),
                block=block_size_gemm, grid=grid_size_gemm, stream=stream)
# drv.Context.synchronize()
# time.sleep(10)
# Start timing
# Launch 20 GEMV kernels and 5 GEMM kernels
# Start timing
start_event = drv.Event()
end_event = drv.Event()

# Launch 20 GEMV kernels and 5 GEMM kernels
start_event.record()

for i in range(num_kernels):
    stream = drv.Stream()
    gemv_kernel(vectors_gpu[i], matrices_gpu, results_gpu[i], np.int32(M), np.int32(N), np.int32(K),
                block=block_size, grid=grid_size, stream=stream)

for i in range(5):
    stream = drv.Stream()
    gemm_kernel(matrices_A_gpu, matrices_gpu, results_gemm_gpu[i], np.int32(40), np.int32(N), np.int32(K),
                block=block_size_gemm, grid=grid_size_gemm, stream=stream)

# Stop timing
drv.Context.synchronize()
end_event.record()
end_event.synchronize()

# Calculate elapsed time
elapsed_time = end_event.time_since(start_event)

# Retrieve results
results = [result_gpu.get() for result_gpu in results_gpu]
results_gemm = [result_gpu.get() for result_gpu in results_gemm_gpu]

# Print elapsed time
print(f"{elapsed_time}")
# for i, result in enumerate(results):
#     numpy_result = np.matmul(vectors[i],matrices)
#     error = np.abs(result - numpy_result)
#     if np.allclose(result, numpy_result, atol=1e-6):
#         print(f"Result {i + 1} is correct.")
#     else:
#         print(f"Result {i + 1} is incorrect. Maximum error: {np.max(error)}")