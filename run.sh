#!/bin/bash

# Compile the program
nvcc gemm.cu -o matmul
nvcc gemm_split.cu -o matmul_split
output_file="output_80.csv"
# Run the program with num_kernels from 1 to 20
echo "multikernel:" > $output_file
for i in {1..80}; do
    echo "Running with num_kernels = $i"
    ./matmul_split $i >> $output_file
    sleep 5
done
echo "singlekernel:" >> $output_file
for i in {1..80}; do
    echo "Running with num_kernels = $i"
    ./matmul $i >> $output_file
    sleep 5
done