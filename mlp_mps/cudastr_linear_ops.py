import torch
import time
import ops
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

hidden_size = 4096
input_data = torch.randn(1, hidden_size).cuda()
weight = torch.randn(10, hidden_size).cuda()
bias = torch.randn(10).cuda()

with torch.cuda.device(0):
    with torch.cuda.stream(stream1):
        # print(time.time(),"time in process 1")
        for _ in range(10000):
            output = ops.linear_cuda(input_data, weight, bias)
    with torch.cuda.stream(stream2):
        # print(time.time(),"time in process 2")
        for _ in range(1000):
            output = ops.linear_cuda(input_data, weight, bias)