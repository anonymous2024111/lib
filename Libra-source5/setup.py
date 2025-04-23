from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='Libra5',
    ext_modules=[
        CppExtension(
            name='Libra5Block', 
            sources=[
            './Block/block.cpp'
            ],
            extra_compile_args=['-O3', '-fopenmp', '-mcx16'],
            libraries=["numa", "tcmalloc_minimal"]
         ),
        CUDAExtension(
            name='Libra5BenchmarkGCN', 
            sources=[
            './benchmarkGCN/mGCNkernel.cu',
            './benchmarkGCN/mGCN.cpp',
            ]
         ),
        CUDAExtension(
            name='Libra5BenchmarkGAT', 
            sources=[
            './benchmarkGAT/mGATkernel.cu',
            './benchmarkGAT/mGAT.cpp',
            ]
         ),
        # CUDAExtension(
        #     name='Libra5SpMM', 
        #     sources=[
        #     './SpMM/mGCNkernel.cu',
        #     './SpMM/mGCN.cpp',
        #     ]
        #  ),
        # CUDAExtension(
        #     name='LibraGCN_new', 
        #     sources=[
        #     './GCN/GCNkernel.cu',
        #     './GCN/GCN.cpp',
        #     ]
        #  ),
        # CUDAExtension(
        #     name='LibraAGNN_new', 
        #     sources=[
        #     './GAT/mGATkernel.cu',
        #     './GAT/mGAT.cpp',
        #     ]
        #  ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


