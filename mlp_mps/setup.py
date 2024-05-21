# setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ops',
    ext_modules=[
        CUDAExtension('ops', [
            'ops.cpp',
            'custom_linear_gemv.cu',
            'custom_linear_gemm.cu',
            'custom_relu.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

