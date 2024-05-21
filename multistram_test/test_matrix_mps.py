import torch
import time
from tqdm import tqdm

def measure_time(matrix1, matrix2, num_matrices):
    """ Function to measure the time taken for matrix multiplication. """
    streams = [torch.cuda.Stream() for _ in range(num_matrices)]

    #warm up
    for _ in range(5):
        for i in range(num_matrices):
            with torch.cuda.stream(streams[i]):
                torch.matmul(matrix1, matrix2)
        torch.cuda.synchronize()  # Wait for all kernels to finish

    start_time = time.time()
    for _ in range(100):
        for i in range(num_matrices):
            with torch.cuda.stream(streams[i]):
                torch.matmul(matrix1, matrix2)
        torch.cuda.synchronize()  # Ensure accurate timing by waiting for all kernels to finish
    time_used = time.time() - start_time
    return time_used

def save_times_to_file(filename, batch_sizes, times):
    with open(filename, 'w') as file:
        for batch_size, time in zip(batch_sizes, times):
            file.write(f"{batch_size},{time}\n")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 4096
    max_batch_size = 40
    matrix1 = torch.randn(hidden_dim, hidden_dim, device=device)
    matrix2 = torch.randn(hidden_dim, hidden_dim, device=device)
    batch_sizes = range(1, max_batch_size+1)
    times_batch = []

    for batch_size in tqdm(batch_sizes, desc="batching"):
        elapsed_time = measure_time(matrix1, matrix2, batch_size)
        times_batch.append(elapsed_time)

    save_times_to_file(f'batch_gem_times_mps_{max_batch_size}.txt', batch_sizes, times_batch)