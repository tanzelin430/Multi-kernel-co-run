
# Persistent block policy for SM-dispatch
- gemm_test
  - mlp_mps
  - mpstest
  - mpstest2
  - multistream_test
  - SM-dispatch
  - splittest
- README.md
- run.sh
- split

该文件夹是针对论文**Enabling and Exploiting Flexible Task Assignment on GPU through SM-Centric Program Transformations**做的对SM分配推理系统的一系列尝试


针对MLP模型 我做了一系列的技术基础验证
## SM-dispatch 代码复现
SM-dispatch是基于论文复现的源码，可以灵活分配给单个kernel的计算资源（SM Level）


## MPS 并行
mlp_mps文件夹下面的代码，做了一个多进程推理的demo，一个进程负责一个推理请求，同时，因为当下的SM分配系统需要深入到cuda kernel的源码层进行修改，所以在其中也完成了从cuda kernel源码到pytorch进行cuda kernel调用的全过程

mpstest(2)两个文件夹下面的代码，展示了将一个batch gemm分割成为batch个gemv的具体实现。

## Cuda stream并行
multistream_test以及split test是用cuda stream来进行多核并行的尝试，split test文件夹下面的代码更加全面，涵盖了从cuda kernel到pytorch调用的全部流程
可以使用nvcc -arch=compute_<> -code=sm_<> -cubin gemv_kernel.cu来编译

可以使用run.sh进行批量测试


