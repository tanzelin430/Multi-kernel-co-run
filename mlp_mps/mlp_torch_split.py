import torch
import torch.nn as nn
from torch.multiprocessing import Process, set_start_method, Barrier
import time
import argparse

hidden_size = 4096
# class MLP(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(MLP, self).__init__()
#         self.weight1 = nn.Parameter(torch.randn(hidden_size, hidden_size))
#         self.weight2 = nn.Parameter(torch.randn(hidden_size, output_size))

#     def forward(self, x):  # cuda stream
#         x = torch.relu(torch.mm(x, self.weight1))
#         for _ in range(1000):
#             x = torch.relu(torch.mm(x, self.weight1))
#         x = torch.mm(x, self.weight2)
#         return x

class MLP(nn.Module):
    def __init__(self,  hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):#cuda stream
        x = torch.relu(self.fc1(x))
        for _ in range(500):
            x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MLP(hidden_size, 10).cuda()
model.share_memory()

def inference(input_data, i, barrier):
    # Warmup
    # with torch.no_grad():
    for _ in range(10):
        model(input_data)
    # torch.cuda.synchronize()
    barrier.wait()

    if i == 0:
        global start_time
        start_time = time.time()

    output = model(input_data)

    barrier.wait()

    if i == 0:
        global end_time
        end_time = time.time()
        print(end_time - start_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_of_requests", type=int, help="Number of requests to process")
    args = parser.parse_args()

    set_start_method('spawn', force=True)
    processes = []
    inputs = [torch.randn(1, hidden_size).cuda() for _ in range(args.num_of_requests)]
    barrier = Barrier(args.num_of_requests)
    # for _ in range(10):
    #     inference(inputs[0],0,barrier)
    
    for i in range(args.num_of_requests):
        p = Process(target=inference, args=(inputs[i], i, barrier))
        processes.append(p)
        p.start()

    # for _ in range(10):
    #     #launch request->dispatch->cudastream
    for p in processes:
        p.join()