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
mod = drv.module_from_file("gemv_kernel.cubin")
gemv_kernel = mod.get_function("gemv_kernel")

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <num_kernels>")
    sys.exit(1)

num_kernels = int(sys.argv[1])

M = 1
N = 4096
K = 4096

# Generate input matrices and vectors (values between 1 and 5)
matrices = np.random.uniform(1, 5, size=(K,N)).astype(np.float32)
vectors = [np.random.uniform(1, 5, size=(M,K)).astype(np.float32) for _ in range(num_kernels)]


# Allocate GPU memory for results
results_gpu = [gpuarray.zeros((M, N), np.float32) for _ in range(num_kernels)]


# transfer data to the GPU asynchrounously with in the stream
matrices_gpu = gpuarray.to_gpu(matrices)
vectors_gpu = [gpuarray.to_gpu(vector) for vector in vectors]
# Warm-up
block_size = (32, 1, 1)
grid_size = (int(np.ceil(N / block_size[0])), int(np.ceil(M / block_size[1])), 1)

for i in range(3):
    gemv_kernel(vectors_gpu[i], matrices_gpu, results_gpu[i], np.int32(M), np.int32(N), np.int32(K),
                block=block_size, grid=grid_size)
        
# drv.Context.synchronize()
# time.sleep(10)
# Start timing
start_event = drv.Event()
end_event = drv.Event()

# Launch 40 kernels
start_event.record()
for i in range(num_kernels):
    gemv_kernel(vectors_gpu[i], matrices_gpu, results_gpu[i], np.int32(M), np.int32(N), np.int32(K),
                block=block_size, grid=grid_size)

# Stop timing
end_event.record()
end_event.synchronize()

# Calculate elapsed time
elapsed_time = end_event.time_since(start_event)

# Retrieve results
results = [result_gpu.get() for result_gpu in results_gpu]

# Print elapsed time
print(f"{elapsed_time}")

# for i, result in enumerate(results):
#     numpy_result = np.matmul(vectors[i],matrices)
#     error = np.abs(result - numpy_result)
#     if np.allclose(result, numpy_result, atol=1e-6):
#         print(f"Result {i + 1} is correct.")
#     else:
#         print(f"Result {i + 1} is incorrect. Maximum error: {np.max(error)}")