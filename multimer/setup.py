import os
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
version_dependent_macros = [
    '-DVERSION_GE_1_1',
    '-DVERSION_GE_1_3',
    '-DVERSION_GE_1_5',
]

extra_cuda_flags = [
    '-std=c++14',
    '-maxrregcount=50',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda'
]
setup(
        name='trainable_folding',
        ext_modules=[CUDAExtension(
        name="attn_core_inplace_cuda",
        sources=[
            "utils/kernel/csrc/softmax_cuda.cpp",
            "utils/kernel/csrc/softmax_cuda_kernel.cu",
        ],
        include_dirs=[
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'utils/kernel/csrc/'
            )
        ],
        extra_compile_args={
            'cxx': ['-O3'] + version_dependent_macros,
            'nvcc': (
                ['-O3', '--use_fast_math'] +
                version_dependent_macros +
                extra_cuda_flags
            ),
        }
    )],
        cmdclass={'build_ext': BuildExtension},
        install_requires=[
        'torch',
        'deepspeed',
        'biopython',
        'ml-collections',
        'numpy',
        'scipy',
        'pytorch_lightning',
        'dm-tree',
    ]
)
