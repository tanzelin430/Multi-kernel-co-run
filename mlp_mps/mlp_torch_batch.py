import torch
import torch.nn as nn
import time
import argparse
hidden_size = 4096
class MLP(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        for _ in range(1000):
            x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MLP(hidden_size, 10).cuda()

def inference(input_data):
    # print("Starting inference")
    output = model(input_data)
    # print("Inference result:", output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_size", type=int, help="Number of requests to process")
    args = parser.parse_args()

    input_batch = torch.randn(args.batch_size, hidden_size).cuda()

    # Warm up
    for _ in range(10):
        inference(input_batch)
    #synchonize
    torch.cuda.synchronize()
    # Measure inference time

    start_time = time.time()
    inference(input_batch)
    torch.cuda.synchronize()
    end_time = time.time()

    print(end_time - start_time)