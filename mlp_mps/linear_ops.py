from torch.multiprocessing import Process, set_start_method
import torch
import time
import ops

stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()
stream3 = torch.cuda.Stream()
torch.cuda.synchronize()
# Create some random input data
hidden_size = 10000
input_data = torch.randn(1, hidden_size).cuda()
weight = torch.randn(10, hidden_size).cuda()
bias = torch.randn(10).cuda()
def process1():
    # global stream1, stream2, stream3
    with torch.cuda.stream(stream1):
        print(time.time(),"time in process 1")
        for _ in range(10000):
            output = ops.linear_cuda(input_data, weight, bias)
def process2():
    # global stream1, stream2, stream3
    with torch.cuda.stream(stream2):
        print(time.time(),"time in process 2")
        for _ in range(10000):
            output = ops.linear_cuda(input_data, weight, bias)

def process3():
    # global stream1, stream2, stream3
    with torch.cuda.stream(stream3):
        print(time.time(),"time in process 3")
        for _ in range(10000):
            output = ops.linear_cuda(input_data, weight, bias)
if __name__ == "__main__":
    set_start_method('spawn',force = True)
    start = time.time()
    p1 = Process(target = process1)
    p2 = Process(target = process2)
    p3 = Process(target = process3)
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    torch.cuda.synchronize()