import time
import torch

# 方法一：使用40个单独的矩阵乘法，每个算子分配一个CUDA stream。
def method1():
    a = torch.randn(1, 4096).cuda()
    b = torch.randn(4096).cuda()


    start = time.time()

    streams = []
    for _ in range(40):
        stream = torch.cuda.Stream(non_blocking=True)
        with torch.cuda.stream(stream):
            torch.matmul(a, b)
            #torch.cuda.synchronize()
        streams.append(stream)

    # 等待所有stream完成
    for stream in streams:
        stream.synchronize()

    end = time.time()

    return end - start

# 方法二：直接计算[40, 4096] * [4096, 4096]矩阵乘法。
def method2():
    a = torch.randn(40, 4096).cuda()
    b = torch.randn(4096, 4096).cuda()

    start = time.time()

    torch.matmul(a, b)

    torch.cuda.synchronize()
    end = time.time()

    return end - start

time1 = method1()
time2 = method2()

print(f"方法一运行时间：{time1:.6f}秒")
print(f"方法二运行时间：{time2:.6f}秒")