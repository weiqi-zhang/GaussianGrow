from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='depthid_render_mask_control_v2',
    version='2.0',
    description='Mask-controlled depth/ID renderer (variant 2) used by GaussianGrow.',
    ext_modules=[
        CUDAExtension(
            name='depthid_render_mask_control_v2',
            sources=['wake_up.cpp', 'render.cu'],
            extra_compile_args={
                'cxx': [],
                'nvcc': ['--extended-lambda'],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)