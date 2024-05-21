import torch
import torch.nn as nn
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
        layer1 = ops.linear_gemm_cuda(x, self.linear1_weight, self.linear1_bias)
        relu_output = ops.relu_cuda(layer1)
        for _ in range(1000):
            layer1 = ops.linear_gemm_cuda(relu_output, self.linear1_weight, self.linear1_bias)
            relu_output = ops.relu_cuda(layer1)
        result = ops.linear_cuda(relu_output, self.linear2_weight, self.linear2_bias)
        return result

def inference(input_data):
    # print("inference_"f'{i+1}'" started")
    output = model(input_data)
    # print("Inference"f'{i+1}' "result:", output)

model = CustomLinearMLP(4096, 10).cuda()
model.share_memory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_size", type=int, help="Number of requests to process")
    args = parser.parse_args()



    input_batch = torch.randn(args.batch_size, 4096).cuda()
    # Warm up
    for _ in range(10):
        inference(input_batch)

    #synchonize
    torch.cuda.synchronize()
    # Measure inference time
    start_time = time.time()
    inference(input_batch)
    end_time = time.time()

    print(end_time - start_time)