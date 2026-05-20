from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='depthid_render',
    version='1.0',
    description='Per-pixel depth and Gaussian-ID renderer used by GaussianGrow.',
    ext_modules=[
        CUDAExtension(
            name='depthid_render',
            sources=['wake_up.cpp', 'render.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)