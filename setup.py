from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='Quantize',
      description='A C++/CUDA PyTorch extension to quantize tensors.',
      ext_modules=[
          CUDAExtension('quantization', [
          'quant_cuda.cpp',
          'quant.cu',
          ]),
      ],
      cmdclass={
          'build_ext': BuildExtension
      },
      packages=find_packages(),
      install_requires=[
          'numpy>=1.14.2',
          'torch>=1.2.0',
      ],
      include_package_data=True
 )
