import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

# Load the custom GEMV kernel
mod = drv.module_from_file("gemv_kernel.cubin")
gemv_kernel = mod.get_function("gemv_kernel")

# Number of GEMV kernels to launch
num_kernels = 40
hidden_dim = 4096
# Create streams
streams = [drv.Stream() for _ in range(num_kernels)]

# Generate input matrices and vectors
matrices = [np.random.uniform(1, 5, size=(hidden_dim,hidden_dim)).astype(np.float32) for _ in range(num_kernels)]
vectors = [np.random.uniform(1, 5, size=(1,hidden_dim)).astype(np.float32) for _ in range(num_kernels)]

# Allocate GPU memory and transfer data
matrices_gpu = [gpuarray.to_gpu(matrix) for matrix in matrices]
vectors_gpu = [gpuarray.to_gpu(vector) for vector in vectors]

# Allocate GPU memory for results
results_gpu = [gpuarray.zeros((1, hidden_dim), np.float32) for _ in range(num_kernels)]

# Launch custom GEMV kernels
block_size = (32, 1, 1)
grid_size = (128, 1, 1)
for i in range(num_kernels):
    # print("doing this")
    gemv_kernel(vectors_gpu[i], matrices_gpu[i], results_gpu[i], np.int32(1), np.int32(4096), np.int32(4096),
                block=block_size, grid=grid_size, stream=streams[i])

# Synchronize streams and retrieve results
results = [result_gpu.get(stream=stream) for result_gpu, stream in zip(results_gpu, streams)]

# Print results
# for i, result in enumerate(results):
#     print(f"Result {i + 1}:\n{result}")
# for i, result in enumerate(results):
#     numpy_result = np.matmul(vectors[i],matrices[i])
#     error = np.abs(result - numpy_result)
#     if np.allclose(result, numpy_result, atol=1e-6):
#         print(f"Result {i + 1} is correct.")
#     else:
#         print(f"Result {i + 1} is incorrect. Maximum error: {np.max(error)}")