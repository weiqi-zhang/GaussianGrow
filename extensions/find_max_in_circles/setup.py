from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='find_max_in_circles',
    version='1.0',
    description='Parallel circle-region maximum scan used by GaussianGrow overlap detection.',
    ext_modules=[
        CUDAExtension(
            name='find_max_in_circles',
            sources=['find_max_in_circles.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)