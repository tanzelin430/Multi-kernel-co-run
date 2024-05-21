import torch
import time
from tqdm import tqdm

def measure_time(matrix1, matrix2):
    """ Function to measure the time taken for matrix multiplication. """
    #warm up
    for _ in range(5):
        torch.matmul(matrix1, matrix2)
        torch.cuda.synchronize()  # Wait for all kernels to finish
    start_time = time.time()
    for _ in range(100):
        torch.matmul(matrix1, matrix2)
        torch.cuda.synchronize()  # Ensure accurate timing by waiting for all kernels to finish
    time_used = time.time() - start_time
    return time_used/100

def save_times_to_file(filename, batch_sizes, times):
    with open(filename, 'w') as file:
        for batch_size, time in zip(batch_sizes, times):
            file.write(f"{batch_size},{time}\n")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 4096
    max_batch_size = 40
    matrix1 = torch.randn(hidden_dim, hidden_dim, device=device)
    matrix2_max = torch.randn(max_batch_size, hidden_dim, device=device)
    batch_sizes = range(1, max_batch_size+1)
    times_batch = []
    for batch_size in tqdm(batch_sizes, desc="batching"):
        matrix2 = matrix2_max[:batch_size, :]
        elapsed_time = measure_time(matrix2, matrix1)
        times_batch.append(elapsed_time)

    save_times_to_file(f'batch_gem_times_{max_batch_size}.txt', batch_sizes, times_batch)