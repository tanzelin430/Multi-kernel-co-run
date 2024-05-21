import torch
import torch.nn as nn
from torch.multiprocessing import Pool, set_start_method
import time
import ops
import argparse

class CustomLinearMLP(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super(CustomLinearMLP, self).__init__()
        self.linear1_weight = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.linear1_bias = torch.nn.Parameter(torch.randn(hidden_size))
        self.linear2_weight = torch.nn.Parameter(torch.randn(output_size, hidden_size))
        self.linear2_bias = torch.nn.Parameter(torch.randn(output_size))

    def forward(self, x):
        layer1 = ops.linear_cuda(x, self.linear1_weight, self.linear1_bias)
        relu_output = torch.relu(layer1)
        for _ in range(1000):
            layer1 = ops.linear_cuda(relu_output, self.linear1_weight, self.linear1_bias)
            relu_output = torch.relu(layer1)
        layer2 = ops.linear_cuda(relu_output, self.linear2_weight, self.linear2_bias)
        relu_output = torch.relu(layer2)
        result = ops.linear_cuda(relu_output, self.linear2_weight, self.linear2_bias)
        return result

def inference_wrapper(args):
    input_data, model = args
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(input_data)

    output = model(input_data)
    return output

def main(num_of_requests):
    model = CustomLinearMLP(4096, 10).cuda()
    model.share_memory()  # Prepare model for shared memory usage

    inputs = [torch.randn(1, 4096).cuda() for _ in range(num_of_requests)]

    set_start_method('spawn', force=True)

    start_time = time.time()

    with Pool(processes=8) as pool:  # Adjust the number of processes based on your system's capabilities
        results = pool.map(inference_wrapper, [(input_data, model) for input_data in inputs])

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_of_requests", type=int, help="Number of requests to process")
    args = parser.parse_args()
    
    main(args.num_of_requests)

