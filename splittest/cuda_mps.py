import numpy as np
import sys
import time
import os
from multiprocessing import Pool
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

drv.init()

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <num_kernels>")
    sys.exit(1)

num_kernels = int(sys.argv[1])

M = 1
N = 4096
K = 4096

# Generate input matrices and vectors (values between 1 and 5)
matrices = np.random.uniform(1, 5, size=(K, N)).astype(np.float32)
vectors = [np.random.uniform(1, 5, size=(M, K)).astype(np.float32) for _ in range(num_kernels)]

# Generate input matrices for GEMM kernel
matrices_A = np.random.uniform(1, 5, size=(40, K)).astype(np.float32)
matrices_B = np.random.uniform(1, 5, size=(K, N)).astype(np.float32)

# Allocate GPU memory for results
results_gpu = [gpuarray.zeros((M, N), np.float32) for _ in range(num_kernels)]
results_gemm_gpu = gpuarray.zeros((40, N), np.float32)

# Transfer data to the GPU
matrices_gpu = gpuarray.to_gpu(matrices)
vectors_gpu = [gpuarray.to_gpu(vector) for vector in vectors]
matrices_A_gpu = gpuarray.to_gpu(matrices_A)

block_size_gemm = (32, 8, 1)
grid_size_gemm = (int(np.ceil(N / block_size_gemm[0])), int(np.ceil(40 / block_size_gemm[1])), 1)

# Define the functions to be executed in parallel
def launch_gemv_kernel(i):
    ctx = drv.Device(0).make_context()
    
    mod = drv.module_from_file("gemv_kernel.cubin")
    gemv_kernel = mod.get_function("gemv_kernel")

    block_size = (32, 1, 1)
    grid_size = (int(np.ceil(N / block_size[0])), int(np.ceil(M / block_size[1])), 1)
    stream = drv.Stream()
    gemv_kernel(vectors_gpu[i], matrices_gpu, results_gpu[i], np.int32(M), np.int32(N), np.int32(K),
                block=block_size, grid=grid_size, stream=stream)
    ctx.pop()

def launch_gemm_kernel(i):
    ctx = drv.Device(0).make_context()
    
    mod = drv.module_from_file("gemv_kernel.cubin")
    gemm_kernel = mod.get_function("gemm_kernel")

    block_size_gemm = (32, 8, 1)
    grid_size_gemm = (int(np.ceil(N / block_size_gemm[0])), int(np.ceil(40 / block_size_gemm[1])), 1)
    stream = drv.Stream()
    gemm_kernel(matrices_A_gpu, matrices_gpu, results_gemm_gpu, np.int32(40), np.int32(N), np.int32(K),
                block=block_size_gemm, grid=grid_size_gemm, stream=stream)
    ctx.pop()

# Create a multiprocessing.Pool with a sufficient number of workers
with Pool(processes=num_kernels + 5) as pool:
    pool.starmap(launch_gemv_kernel, [(i,) for i in range(num_kernels)])
    pool.starmap(launch_gemm_kernel, [(i,) for i in range(5)])

# Retrieve results
results = [result_gpu.get() for result_gpu in results_gpu]
results_gemm = [result_gpu.get() for result_gpu in results_gemm_gpu]

# Print results (optional)
print("Results:")
print(results)
print(results_gemm)