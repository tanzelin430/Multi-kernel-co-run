import ctypes
import numpy as np
from multiprocessing import Process
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import sys
import time

# Load the custom GEMV kernel
mod = drv.module_from_file("gemv.cubin")
launch_gemv = mod.get_function("launch_gemv")

def run_gemv(A, x, y):
    m, n = A.shape
    launch_gemv(A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                m, n)

def main():
    # Initialize the input matrices and vector
    A = np.random.rand(40, 4096).astype(np.float32)
    x = np.random.rand(4096).astype(np.float32)
    y = np.zeros((40, 1), dtype=np.float32)

    # Launch 40 GEMV kernels in parallel using multiprocessing
    processes = []
    for i in range(40):
        Ai = A[i:i+1, :]
        yi = y[i:i+1, :]
        p = Process(target=run_gemv, args=(Ai, x, yi))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("Result:")
    print(y)

if __name__ == "__main__":
    main()