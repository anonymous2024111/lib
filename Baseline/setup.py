from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='LBaseline',
    ext_modules=[
       CUDAExtension(
            name='GNNAdvisor', 
            sources=[
            './GNNAdvisor/GNNAdvisor_kernel.cu',
            './GNNAdvisor/GNNAdvisor.cpp',
            ]
         ),
       CUDAExtension(
            #v2 without gemm in GNNAdisor
            name='GNNAdvisor_v2', 
            sources=[
            './GNNAdvisor_v2/GNNAdvisor_kernel.cu',
            './GNNAdvisor_v2/GNNAdvisor.cpp',
            ]
         ),
       CUDAExtension(
            name='TCGNN', 
            sources=[
            './TCGNN/TCGNN_kernel.cu',
            './TCGNN/TCGNN.cpp',
            ]
         ) ,
       CUDAExtension(
            name='GESpMM', 
            sources=[
            './GESpMM/gespmmkernel.cu', 
            './GESpMM/gespmm.cpp',
            ]
         ) ,
       CUDAExtension(
            name='cuSPARSE', 
            sources=[
            './cuSPARSE/spmm_csr_kernel.cu',
            './cuSPARSE/spmm_csr.cpp',
            ]
         ) ,
       CppExtension(
            name='Rabbit', 
            sources=[
            './Rabbit/reorder.cpp',
            ],
            extra_compile_args=['-O3', '-fopenmp', '-mcx16'],
            libraries=["numa", "tcmalloc_minimal"]
         )  
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


