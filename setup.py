import os
os.environ['CUDA_HOME'] = '/usr/local/cuda-12'


from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension



setup(
    name='cuLegKan',
    packages=find_packages(),
    version='0.0.0',
    author='Yuxue Yang',
    ext_modules=[
        CUDAExtension(
            'legendre_ops', # operator name
            ['./cpp/legendre.cpp',
             './cpp/legendre_cuda.cu',]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)