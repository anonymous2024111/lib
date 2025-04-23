#include <stdio.h>
#include <mma.h>
#include <cstdint>
#include <iostream>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// half automic_add(half a, half b){
//      float a1 = __half2float(a);
//      float a2 = __half2float(b);
//      return __float2half(atomicAdd(&a1, a2));
// }
__device__ void automic_add(__half* address, __half val)
{
    unsigned int* address_as_uint = reinterpret_cast<unsigned int*>(address);
    unsigned int old = *address_as_uint, assumed;
    do {
        assumed = old;
        __half2 old_val = *reinterpret_cast<__half2*>(&assumed);
        __half2 new_val = __halves2half2(val, val); // 将一个半精度浮点数复制到两个半精度浮点数
        old = atomicCAS(address_as_uint, assumed, *reinterpret_cast<unsigned int*>(&new_val));
    } while (assumed != old);
}


struct cudaComputeUtils_fp16_trans{
    const float2 * c_rhs_matrix_;
    const int *c_col_indices_;
    const half *c_values_;
    float * output_matrix_;
    int nOri_;
    int warpin_id_;
    __device__ __forceinline__ cudaComputeUtils_fp16_trans(
        const float2 * rhs_matrix,
        int *c_col_indices,
        half *c_values,
        float * output_matrix,
        int nOri,
        int warpin_id
    ):
    c_rhs_matrix_(rhs_matrix),
    c_col_indices_(c_col_indices),
    c_values_(c_values),
    output_matrix_(output_matrix),
    nOri_(nOri),
    warpin_id_(warpin_id)
    // warp_id_(warp_id),
    // c_row_offsets_(c_row_offsets)
    {}
    
    // CUDA Compute
    __device__ __forceinline__ void cudaCompute( int nonzeros,int atomic){


        at::Half res_[4] = {0.0, 0.0, 0.0, 0.0};
        half * res = reinterpret_cast<half *>(res_);
        float2 b[1] = {make_float2(0.0f, 0.0f)}; 
        half *bb = reinterpret_cast<half *>(b);
        half a = __float2half(0.0);
        int col = 0;
        #pragma unroll
        for(int i=0; i<nonzeros; i++)
        {
            a = *(c_values_ + i);
            col = *(c_col_indices_ + i);
            b[0] = __ldg(c_rhs_matrix_ + col*nOri_/4 + warpin_id_); 
            for(int j=0;j<4;j++)
                res[j] = __hadd(res[j], __hmul(a , bb[j]));
            // res[1] = __hadd(res[0], __hmul(a , b.y));
            // res[2] = __hadd(res[0], __hmul(a , b.z));
            // res[3] = __hadd(res[0], __hmul(a , b.w));

        }
        for(int i=0; i<4; i++)
        {
            if(atomic==0)
            *(output_matrix_ + warpin_id_*4 + i) += __half2float(res[i]);
            else  {
                    atomicAdd((output_matrix_ + warpin_id_*4 + i), __half2float(res[i]));
                }
        }
    }
};



//仅用于CUDA计算
struct cudaComputeUtils_fp16_trans_short{
    const half * c_rhs_matrix_;
    const int *c_col_indices_;
    const half *c_values_;
    float * output_matrix_;
    int nOri_;
    int warpin_id_;
    // int warp_id_;
    // const int *c_row_offsets_;
    // Constructor
    __device__ __forceinline__ cudaComputeUtils_fp16_trans_short(
        const half * rhs_matrix,
        const int *c_col_indices,
        const half *c_values,
        float * output_matrix,
        int nOri,
        int warpin_id
        // int warp_id,
        // const int * c_row_offsets
    ):
    c_rhs_matrix_(rhs_matrix),
    c_col_indices_(c_col_indices),
    c_values_(c_values),
    output_matrix_(output_matrix),
    nOri_(nOri),
    warpin_id_(warpin_id)
    // warp_id_(warp_id),
    // c_row_offsets_(c_row_offsets)
    {}
    
    // CUDA Compute
    __device__ __forceinline__ void cudaCompute( int nonzeros, int dimN_index, int row, int atomic){

        at::Half res_[2] = {0.0, 0.0};
        half * res = reinterpret_cast<half *>(res_);
        int c_col_offset = dimN_index + warpin_id_;

        #pragma unroll
        for(int i=0; i<nonzeros; i++)
        {
            half a = __ldg(c_values_ + i);
            int col = __ldg(c_col_indices_ + i);
            for(int j=0; j<2; j++)
            {
                if((c_col_offset + j*32)<nOri_){
                half b = __ldg(c_rhs_matrix_ + col*nOri_ + c_col_offset + j*32);  
                res[j] =__hadd(res[j], __hmul(a , b));
                // if(row==0 && threadIdx.x==0)
                // printf("%f, %f, %f\n", a, b, res[1]);
                }
            }
        }
        // if(row==0 && threadIdx.x==0)
        // printf("%d, %f, %f\n", nonzeros, res[0], res[1]);
        for(int i=0; i<2; i++)
        {
            if((c_col_offset + i*32)<nOri_){
                if(atomic==0)
                *(output_matrix_ + row*nOri_ + c_col_offset + i*32) += __half2float(res[i]);
                else  {
                     atomicAdd((output_matrix_ + row*nOri_ + c_col_offset + i*32), __half2float(res[i]));
                    }
            }  
        }
    }
};
