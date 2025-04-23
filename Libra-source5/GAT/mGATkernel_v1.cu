#include <stdio.h>
#include <mma.h>
#include <cstdint>
#include <iostream>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include "./sddmm_utils/dense_tile.h"
#include "./sddmm_utils/compute.h"
#include "./sddmm_utils/output_tile.h"
#include <cuda_runtime.h>

//spmm_forward_tf32_csr_kernel, two kernel
__global__ void sddmm_forward_tf32_csr_v2_kernel_cuda(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_column,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int parts,
    int nOri,
    int mOri,
    int splitk)
{
    //假如每个block中1个warp, 每个warp负责计算稀疏矩阵的一行中的非零元
    int c_part_offset_vec = ((blockIdx.z*splitk)+blockIdx.y);
    if(c_part_offset_vec>=parts) return;
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        extern __shared__  float res_share[];
        float * res_share_ = res_share + warpin_id*nonzeros;
        int * sparse_col = (int *) & res_share[32*nonzeros];
        const float * rhs_matrix_a = rhs_matrix + out_row_offset*nOri;

        int iters = (nOri/32) + 1;
        if((nOri%32)==0) iters-=1;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            //32个线程一起搬运
            if(warpin_id<nonzeros){
                *(sparse_col + threadIdx.x) = __ldg(c_column + c_row_offset_vec + threadIdx.x);               
            }
            for(int j=0;j<nonzeros;j++)
            *(res_share_ + j) = 0.0;
            
            __syncthreads();

            for(int i=0; i<iters; i++){
                if((i*32+warpin_id) < nOri){
                    float a = __ldg(rhs_matrix_a + i*32 + warpin_id);
                    for(int j=0;j<nonzeros;j++){
                        *(res_share_ + j) +=  a * __ldg(rhs_matrix + sparse_col[j]*nOri + i*32 + warpin_id);
                // if(c_part_offset_vec==0 and threadIdx.x==0)
                // {
                //     printf("%f %d %f %f\n", a, sparse_col[j],__ldg(rhs_matrix + sparse_col[j]*nOri + i*32 + warpin_id) ,*(res_share + j));
                // }
                    }
                }
            }
            __syncthreads();
            if(warpin_id<nonzeros){
                //每个线程负责处理一个nnz的结果
                float res = 0.0;
                float * temp = res_share+warpin_id;
                for(int i=0; i<32; i++)
                {
                    res+= *(temp);
                    temp+=nonzeros;
                }
                *(output_matrix + c_row_offset_vec + warpin_id) = res;
                // if(c_part_offset_vec==0)
                // {
                //     printf("%d %f\n", nonzeros,res);
                // }
            }
        }
    } 
}

__global__ void sddmm_forward_tf32_csr_v2_kernel_cuda_v2(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_column,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int parts,
    int nOri,
    int mOri,
    int splitk)
{
    //假如每个block中1个warp, 每个warp负责计算稀疏矩阵的一行中的非零元
    int c_part_offset_vec = ((blockIdx.z*splitk)+blockIdx.y);
    if(c_part_offset_vec>=parts) return;
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        const float * rhs_matrix_a = rhs_matrix + out_row_offset*nOri;
        int col = __ldg(c_column + c_row_offset_vec + warpin_id);               
        const float * rhs_matrix_b = rhs_matrix + col*nOri;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            if(warpin_id<nonzeros){
                float res = 0.0;
                for(int i=0;i<nOri;i++)
                {
                    res+= __ldg(rhs_matrix_a+i) * __ldg(rhs_matrix_b+i);
                }
                *(output_matrix + c_row_offset_vec + warpin_id) = res;
            }
        }
    } 
}

__global__ void sddmm_forward_fp16_csr_v2_kernel_cuda_v2(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_column,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int parts,
    int nOri,
    int mOri,
    int splitk)
{
    //假如每个block中1个warp, 每个warp负责计算稀疏矩阵的一行中的非零元
    int c_part_offset_vec = ((blockIdx.z*splitk)+blockIdx.y);
    if(c_part_offset_vec>=parts) return;
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        const half * rhs_matrix_a = rhs_matrix + out_row_offset*nOri;
        int col = __ldg(c_column + c_row_offset_vec + warpin_id);               
        const half * rhs_matrix_b = rhs_matrix + col*nOri;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            if(warpin_id<nonzeros){
                half res = __float2half(0.0);
                for(int i=0;i<nOri;i++)
                {
                    res =  __hadd(res, __hmul(__ldg(rhs_matrix_a+i) , __ldg(rhs_matrix_b+i)));
                }
                *(output_matrix + c_row_offset_vec + warpin_id) = res;
            }
        }
    } 
}

// fp16
__global__ void sddmm_forward_fp16_csr_v2_kernel_cuda(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_column,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int parts,
    int nOri,
    int mOri,
    int splitk)
{
    //假如每个block中1个warp, 每个warp负责计算稀疏矩阵的一行中的非零元
    int c_part_offset_vec = ((blockIdx.z*splitk)+blockIdx.y);
    if(c_part_offset_vec>=parts) return;
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        extern __shared__ half res_share_fp16[];
        half * res_share_ = res_share_fp16 + warpin_id*nonzeros;
        int * sparse_col = (int *) & res_share_fp16[32*nonzeros];
        const half * rhs_matrix_a = rhs_matrix + out_row_offset*nOri;
        int iters = (nOri/32) + 1;
        if((nOri%32)>0) iters-=1;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            //32个线程一起搬运
            if(warpin_id<nonzeros){
                *(sparse_col + threadIdx.x) = __ldg(c_column + c_row_offset_vec + threadIdx.x);               
            }
            for(int j=0;j<nonzeros;j++)
            *(res_share_ + j) = __float2half(0.0);
            
            __syncthreads();

            for(int i=0; i<iters; i++){
                if((i*32+warpin_id) < nOri){
                    half a = __ldg(rhs_matrix_a + i*32 + warpin_id);
                    for(int j=0;j<nonzeros;j++){
                        half b = __ldg(rhs_matrix + sparse_col[j]*nOri + i*32 + warpin_id);
                        float c = (__half2float(a) * __half2float(b));
                        *(res_share_ + j) =__hadd(*(res_share_ + j), __float2half(c));
                    }
                }
            }
            __syncthreads();
            
            if(warpin_id<nonzeros){
                //每个线程负责处理一个nnz的结果
                half res = __float2half(0.0);
                half * temp = res_share_fp16+warpin_id;
                for(int i=0; i<32; i++)
                {
                    res = __hadd(res, *(temp));
                    temp+=nonzeros;
                }
                *(output_matrix + c_row_offset_vec + warpin_id) = res;
            }
        }
    } 
    
    
}


//cuda 长短行
void sddmm_forward_tf32_cuda_kernel_split(

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    float * rhs_matrix,
    float * output_matrix,

    const int nOri,
    const int mOri,
    int parts)
{
    int splitk = 0;
    if(parts<500000) splitk=8;
    else splitk=((parts/1250000)+1)*20;

    dim3 grid_dim(1, splitk ,((parts/splitk)+1));
    dim3 block_dim(32, 1, 1);

    int sharedmemory = 32*32*sizeof(float)+ 32*sizeof(int);


  
        sddmm_forward_tf32_csr_v2_kernel_cuda<<<grid_dim, block_dim, sharedmemory>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix,
            parts, nOri, mOri, splitk);

}

float sddmm_forward_tf32_cuda_kernel_split_navie(

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    float * rhs_matrix,
    float * output_matrix,

    const int nOri,
    const int mOri,
    int epoches,
    int parts)
{
    int splitk = 0;
    if(parts<500000) splitk=8;
    else splitk=((parts/1250000)+1)*20;

    dim3 grid_dim(1, splitk ,((parts/splitk)+1));
    dim3 block_dim(32, 1, 1);

    int sharedmemory = 32*32*sizeof(float)+ 32*sizeof(int);
    for(int iter=0; iter<10; ++iter){
        sddmm_forward_tf32_csr_v2_kernel_cuda_v2<<<grid_dim, block_dim>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix,
            parts, nOri, mOri, splitk);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        sddmm_forward_tf32_csr_v2_kernel_cuda_v2<<<grid_dim, block_dim>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix,
            parts, nOri, mOri, splitk);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}

float sddmm_forward_fp16_cuda_kernel_split_navie(

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    half * rhs_matrix,
    half * output_matrix,

    const int nOri,
    const int mOri,
    int epoches,
    int parts)
{
    int splitk = 0;
    if(parts<500000) splitk=8;
    else splitk=((parts/1250000)+1)*20;

    dim3 grid_dim(1, splitk ,((parts/splitk)+1));
    dim3 block_dim(32, 1, 1);

    int sharedmemory = 32*32*sizeof(half)+ 32*sizeof(int);
    for(int iter=0; iter<10; ++iter){
        sddmm_forward_fp16_csr_v2_kernel_cuda_v2<<<grid_dim, block_dim>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix,
            parts, nOri, mOri, splitk);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        sddmm_forward_fp16_csr_v2_kernel_cuda_v2<<<grid_dim, block_dim>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix,
            parts, nOri, mOri, splitk);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}
//cuda 长短行 fp16
void sddmm_forward_fp16_cuda_kernel_split(

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    half * rhs_matrix,
    half * output_matrix,

    const int nOri,
    const int mOri,
    int parts)
{
    int splitk = 0;
    if(parts<500000) splitk=8;
    else splitk=((parts/1250000)+1)*20;

    dim3 grid_dim(1, splitk ,((parts/splitk)+1));
    dim3 block_dim(32, 1, 1);

    int sharedmemory = 32*32*sizeof(half)+ 32*sizeof(int);


        sddmm_forward_fp16_csr_v2_kernel_cuda<<<grid_dim, block_dim, sharedmemory>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix,
            parts, nOri, mOri, splitk);

}



// sddmm tcu tf32
__global__ void sddmm_forward_tf32_binary_v2_kernel_tcu(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const int* __restrict__ t_column,
    const long* __restrict__ t_binary,
    const int* t_window_row,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int warps)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;

    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;
    int id=blockIdx.x*warps+(threadIdx.x/32);
    if(id>= tcu_blocks) return;
    int warp_id = threadIdx.x>>5;
    int warpin_id = threadIdx.x%32;
    float output_fragment[4] = {0,0,0,0};
    long col=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4));
    long col1=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4) +8);
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    long row=cur_m_index_vec*8 + (warpin_id/4);
    const float *r_matrix;
    if(col>=0) r_matrix = rhs_matrix + col*nOri;
    const float *l_matrix = rhs_matrix + row*nOri;
    int steps = nOri/8;
    int residue = nOri%8;
    mmaComputeUtils_tf32_gen computer(
        l_matrix,
        r_matrix, 
        output_fragment, 
        threadIdx.x);

    if(steps>0)
    for(int i = 0; i < steps; i++){
        computer.TileMAC(nOri, col, col1, mOri, row);
    }
    if(residue>0)
        computer.TileMACResidue(residue, nOri, col, col1, mOri, row);

    // if(row<mOri)
    // {
        int offset = ((warpin_id%4)*2*8) + (warpin_id/4);
        int offset1 = offset+8;
        long binary_1 = __ldg(t_binary + t_win_offset*2 + id*2);
        long binary_2 = __ldg(t_binary + t_win_offset*2 + id*2 + 1);
        long temp_0 = (binary_1 >> offset);
        long temp_1 = (binary_1 >> offset1);
        long temp_2 = (binary_2 >> offset);
        long temp_3 = (binary_2 >> offset1);
        long a= 1;
        long mask = (a << offset);
        long mask1 = (a << offset1);

        int fifthBit_0 = ((temp_0) & 1);
        int fifthBit_1 = ((temp_1) & 1);
        int fifthBit_2 = ((temp_2) & 1);
        int fifthBit_3 = ((temp_3) & 1);


        int value_offset = __ldg(t_block_offset + t_win_offset + id);
        float * output_matrix_1 = output_matrix + value_offset;
        float * output_matrix_2 = output_matrix_1 + __popcll(binary_1);
        if(fifthBit_0 == 1)
        {
            int block_offset_1 = __popcll(binary_1 & (mask-1));
            *(output_matrix_1 + block_offset_1) = output_fragment[0];
        }
        if(fifthBit_1 == 1)
        {
            int block_offset_1 = __popcll(binary_1 & (mask1-1));
            *(output_matrix_1 + block_offset_1 ) = output_fragment[1];
        }
        if(fifthBit_2 == 1)
        {
            int block_offset_2 = __popcll(binary_2 & (mask-1));
            *(output_matrix_2 + block_offset_2) = output_fragment[2];
        }
        if(fifthBit_3 == 1)
        {
            int block_offset_2 = __popcll(binary_2 & (mask1-1));
            *(output_matrix_2 + block_offset_2 ) = output_fragment[3];
        }
        // long binary_1 = __ldg(t_binary + t_win_offset*2 + id*2);
        // long binary_2 = __ldg(t_binary + t_win_offset*2 + id*2 + 1);
        // long temp_1 = (binary_1 >> (warpin_id*2));
        // long temp_2 = (binary_2 >> (warpin_id*2));
        // long a= 1;
        // long mask = (a << (warpin_id*2));
        // long mask1 = (a << ((warpin_id*2)+1));
        // int fifthBit_1 = ((temp_1) & 1);
        // int fifthBit_2 = ((temp_2) & 1);
        // int block_offset_1 = -1;
        // int block_offset_2 = -1;
        // int value_offset = __ldg(t_block_offset + t_win_offset + id);
        // float * output_matrix_1 = output_matrix + value_offset;
        // float * output_matrix_2 = output_matrix_1 + __popcll(binary_1);
        // if(fifthBit_1 == 1)
        // {
        //     block_offset_1 = __popcll(binary_1 & (mask-1));
        //     *(output_matrix_1 + block_offset_1) = output_fragment[0];
        // }
        // if(fifthBit_2 == 1)
        // {
        //     block_offset_2 = __popcll(binary_2 & (mask-1));
        //     *(output_matrix_2 + block_offset_2 ) = output_fragment[2];
        // }

        // if((row+1)<mOri)
        // {
        //     fifthBit_1 = ((temp_1>>1) & 1);
        //     fifthBit_2 = ((temp_2>>1) & 1);
        //     if(fifthBit_1 == 1)
        //     {
        //         if(block_offset_1==-1)
        //         {
        //             block_offset_1 = __popcll(binary_1 & (mask1-1));
        //             *(output_matrix_1 + block_offset_1) = output_fragment[1];
        //         }
        //         else
        //         *(output_matrix_1 + block_offset_1 + 1) = output_fragment[1];
        //     }
        //     if(fifthBit_2 == 1)
        //     {
        //         if(block_offset_2==-1)
        //         {
        //             block_offset_2 = __popcll(binary_2 & (mask1-1));
        //             *(output_matrix_2 + block_offset_2) = output_fragment[3];
        //         }
        //         else *(output_matrix_2 + block_offset_2 + 1) = output_fragment[3];
        //     }
        // }
    // }
    /*
    if(row<mOri)
    {
        long binary_1 = __ldg(t_binary + t_win_offset*2 + id*2 + warpin_id/16);
        long temp_1 = (binary_1 >> ((warpin_id%16)*4));
        long a= 1;
        long mask = (a << ((warpin_id%16)*4));
        int fifthBit_1 = ((temp_1) & 1);
        int block_offset_1 = -1;
        int value_offset = __ldg(t_block_offset + t_win_offset + id);
        float * output_matrix_1 = output_matrix + value_offset;
        if(warpin_id>15){
            long binary_2 = __ldg(t_binary + t_win_offset*2 + id*2);
            output_matrix_1 += __popcll(binary_2);
        }
        if(fifthBit_1 == 1)
        {
            block_offset_1 = __popcll(binary_1 & (mask-1));
            *(output_matrix_1 + block_offset_1) = output_fragment[0];
        }
        for(int q=1;q<4;q++){
            temp_1 = temp_1>>1;
            fifthBit_1 = (temp_1 & 1);
            mask = mask<<1;
            if(fifthBit_1 == 1)
            {
                if(block_offset_1==-1)
                {
                    block_offset_1 = __popcll(binary_1 & (mask-1));
                    *(output_matrix_1 + block_offset_1) = output_fragment[q];
                }
                else{
                *(output_matrix_1 + block_offset_1 + 1) = output_fragment[q];
                block_offset_1++;}
            }
        }
    }
    */
}



//tcu cuda
void sddmm_forward_tf32_tcu_part_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * d_t_binary,

    float * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int maxPart,
    const int dimN,
    const int mOri)
{
    int splitk = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk=8;
    else splitk=((parts_t/1250000)+1)*20;
    dim3 grid_dim(((maxPart/warps)+1), splitk ,(parts_t/splitk+1));
    dim3 block_dim(warps*32, 1, 1);


        sddmm_forward_tf32_binary_v2_kernel_tcu<<<grid_dim, block_dim>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_binary, 
            t_window_row,
            rhs_matrix,  
            output_matrix,
            parts_t, dimN, mOri, splitk, warps);

}


// sddmm tcu fp16
__global__ void sddmm_forward_fp16_binary_v2_kernel_tcu(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const int* __restrict__ t_column,
    const long* __restrict__ t_binary,
    const int* t_window_row,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int warps)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;

    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;
    int id=blockIdx.x*warps+(threadIdx.x/32);
    if(id>= tcu_blocks) return;
    int warp_id = threadIdx.x>>5;
    int warpin_id = threadIdx.x%32;
    at::Half output_fragment_[4] = {0,0,0,0};
    half * output_fragment = reinterpret_cast< half *>(output_fragment_);
    long col=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4));
    long col1=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4) +8);
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    long row=cur_m_index_vec*8 + (warpin_id/4);
    const half *r_matrix;
    if(col>=0) r_matrix = rhs_matrix + col*nOri;
    const half *l_matrix = rhs_matrix + row*nOri;
    int steps = nOri/16;
    int residue = nOri%16;
    mmaComputeUtils_fp16_gen computer(
        reinterpret_cast<const at::Half *>(l_matrix),
        reinterpret_cast<const at::Half *>(r_matrix), 
        reinterpret_cast<uint32_t *>(output_fragment), 
        threadIdx.x);

    if(steps>0)
    for(int i = 0; i < steps; i++){
        computer.TileMAC(nOri, col, col1, mOri, row);
    }
    if(residue>0)
        computer.TileMACResidue(residue, nOri, col, col1, mOri, row);


        int offset = ((warpin_id%4)*2*8) + (warpin_id/4);
        int offset1 = offset+8;
        long binary_1 = __ldg(t_binary + t_win_offset*2 + id*2);
        long binary_2 = __ldg(t_binary + t_win_offset*2 + id*2 + 1);
        long temp_0 = (binary_1 >> offset);
        long temp_1 = (binary_1 >> offset1);
        long temp_2 = (binary_2 >> offset);
        long temp_3 = (binary_2 >> offset1);
        long a= 1;
        long mask = (a << offset);
        long mask1 = (a << offset1);

        int fifthBit_0 = ((temp_0) & 1);
        int fifthBit_1 = ((temp_1) & 1);
        int fifthBit_2 = ((temp_2) & 1);
        int fifthBit_3 = ((temp_3) & 1);


        int value_offset = __ldg(t_block_offset + t_win_offset + id);
        half * output_matrix_1 = output_matrix + value_offset;
        half * output_matrix_2 = output_matrix_1 + __popcll(binary_1);
        if(fifthBit_0 == 1)
        {
            int block_offset_1 = __popcll(binary_1 & (mask-1));
            *(output_matrix_1 + block_offset_1) = output_fragment[0];
        }
        if(fifthBit_1 == 1)
        {
            int block_offset_1 = __popcll(binary_1 & (mask1-1));
            *(output_matrix_1 + block_offset_1 ) = output_fragment[1];
        }
        if(fifthBit_2 == 1)
        {
            int block_offset_2 = __popcll(binary_2 & (mask-1));
            *(output_matrix_2 + block_offset_2) = output_fragment[2];
        }
        if(fifthBit_3 == 1)
        {
            int block_offset_2 = __popcll(binary_2 & (mask1-1));
            *(output_matrix_2 + block_offset_2 ) = output_fragment[3];
        }
    // if(row<mOri)
    // {
    //     long binary_1 = __ldg(t_binary + t_win_offset*2 + id*2);
    //     long binary_2 = __ldg(t_binary + t_win_offset*2 + id*2 + 1);
    //     long temp_1 = (binary_1 >> (warpin_id*2));
    //     long temp_2 = (binary_2 >> (warpin_id*2));
    //     long a= 1;
    //     long mask = (a << (warpin_id*2));
    //     long mask1 = (a << ((warpin_id*2)+1));
    //     int fifthBit_1 = ((temp_1) & 1);
    //     int fifthBit_2 = ((temp_2) & 1);
    //     int block_offset_1 = -1;
    //     int block_offset_2 = -1;
    //     int value_offset = __ldg(t_block_offset + t_win_offset + id);
    //     half * output_matrix_1 = output_matrix + value_offset;
    //     half * output_matrix_2 = output_matrix_1 + __popcll(binary_1);
    //     if(fifthBit_1 == 1)
    //     {
    //         block_offset_1 = __popcll(binary_1 & (mask-1));
    //         *(output_matrix_1 + block_offset_1) = output_fragment[0];
    //     }
    //     if(fifthBit_2 == 1)
    //     {
    //         block_offset_2 = __popcll(binary_2 & (mask-1));
    //         *(output_matrix_2 + block_offset_2 ) = output_fragment[2];
    //     }

    //     if((row+1)<mOri)
    //     {
    //         fifthBit_1 = ((temp_1>>1) & 1);
    //         fifthBit_2 = ((temp_2>>1) & 1);
    //         if(fifthBit_1 == 1)
    //         {
    //             if(block_offset_1==-1)
    //             {
    //                 block_offset_1 = __popcll(binary_1 & (mask1-1));
    //                 *(output_matrix_1 + block_offset_1) = output_fragment[1];
    //             }
    //             else
    //             *(output_matrix_1 + block_offset_1 + 1) = output_fragment[1];
    //         }
    //         if(fifthBit_2 == 1)
    //         {
    //             if(block_offset_2==-1)
    //             {
    //                 block_offset_2 = __popcll(binary_2 & (mask1-1));
    //                 *(output_matrix_2 + block_offset_2) = output_fragment[3];
    //             }
    //             else *(output_matrix_2 + block_offset_2 + 1) = output_fragment[3];
    //         }
    //     }
    // }
    /*
    if(row<mOri)
    {
        long binary_1 = __ldg(t_binary + t_win_offset*2 + id*2 + warpin_id/16);
        long temp_1 = (binary_1 >> ((warpin_id%16)*4));
        long a= 1;
        long mask = (a << ((warpin_id%16)*4));
        int fifthBit_1 = ((temp_1) & 1);
        int block_offset_1 = -1;
        int value_offset = __ldg(t_block_offset + t_win_offset + id);
        float * output_matrix_1 = output_matrix + value_offset;
        if(warpin_id>15){
            long binary_2 = __ldg(t_binary + t_win_offset*2 + id*2);
            output_matrix_1 += __popcll(binary_2);
        }
        if(fifthBit_1 == 1)
        {
            block_offset_1 = __popcll(binary_1 & (mask-1));
            *(output_matrix_1 + block_offset_1) = output_fragment[0];
        }
        for(int q=1;q<4;q++){
            temp_1 = temp_1>>1;
            fifthBit_1 = (temp_1 & 1);
            mask = mask<<1;
            if(fifthBit_1 == 1)
            {
                if(block_offset_1==-1)
                {
                    block_offset_1 = __popcll(binary_1 & (mask-1));
                    *(output_matrix_1 + block_offset_1) = output_fragment[q];
                }
                else{
                *(output_matrix_1 + block_offset_1 + 1) = output_fragment[q];
                block_offset_1++;}
            }
        }
    }
    */
}



//tcu cuda
void sddmm_forward_fp16_tcu_part_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * d_t_binary,

    half * rhs_matrix,
    half * output_matrix,

    const int parts_t,
    const int maxPart,
    const int dimN,
    const int mOri)
{
    int splitk = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk=8;
    else splitk=((parts_t/1250000)+1)*20;
    dim3 grid_dim(((maxPart/warps)+1), splitk ,(parts_t/splitk+1));
    dim3 block_dim(warps*32, 1, 1);


        sddmm_forward_fp16_binary_v2_kernel_tcu<<<grid_dim, block_dim>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_binary, 
            t_window_row,
            rhs_matrix,  
            output_matrix,
            parts_t, dimN, mOri, splitk, warps);

}
// sddmm tcu tf32 csr
__global__ void sddmm_forward_tf32_csr_v2_kernel_tcu(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const int* __restrict__ t_column,
    const int* __restrict__ t_row,
    const int* t_window_row,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int warps)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;

    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;
    int id=blockIdx.x*warps+(threadIdx.x/32);
    if(id>= tcu_blocks) return;
    int warp_id = threadIdx.x>>5;
    int warpin_id = threadIdx.x%32;
    float output_fragment[4] = {0,0,0,0};
    long col=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4));
    long col1=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4) +8);
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    long row=cur_m_index_vec*8 + (warpin_id/4);
    const float *r_matrix;
    if(col>=0) r_matrix = rhs_matrix + col*nOri;
    const float *l_matrix = rhs_matrix + row*nOri;
    int steps = nOri/8;
    int residue = nOri%8;
    mmaComputeUtils_tf32_gen computer(
        l_matrix,
        r_matrix, 
        output_fragment, 
        threadIdx.x);

    if(steps>0)
    for(int i = 0; i < steps; i++){
        computer.TileMAC(nOri, col, col1, mOri, row);
    }
    if(residue>0)
        computer.TileMACResidue(residue, nOri, col, col1, mOri, row);

    //output
    __shared__ float sparse[128*4];
    float * sparse_ = sparse + warp_id*128;
    //把结果转置写入sparse
    float* sparse_1 = sparse_ + ((warpin_id%4)*2*16) + (warpin_id/4);
    *(sparse_1) = output_fragment[0];
    *(sparse_1+8) = output_fragment[2];
    *(sparse_1+16) = output_fragment[1];
    *(sparse_1+24) = output_fragment[3];
    __syncwarp();
    // if(m_index_vec==0 && threadIdx.x==0)
    // {
    //     for(int q=0;q<128;q++)
    //     printf("%f ", sparse_[q]);
    //     printf("\n");
    // }
    // for(int q=0;q<4;q++)
    // sparse_[warpin_id*4+q] = -1;
    //__syncwarp();
    int value_offset = __ldg(t_block_offset + t_win_offset + id);
    int nnz_block = __ldg(t_block_offset + t_win_offset + id + 1) - value_offset;
    for(int q=0;q<4;q++)
    {
        if((warpin_id*4 + q) < nnz_block){
            int temp = __ldg(t_row + value_offset + warpin_id*4 + q);
            *(output_matrix + value_offset + warpin_id*4 + q) = *(sparse_ + temp);
        }
    }
    // __syncwarp();
    // if(m_index_vec==0 && threadIdx.x==0)
    // {
    //     for(int q=0;q<128;q++)
    //     printf("%d ", sparse_[q]);
    //     printf("\n");
    // }
    // if(row<mOri)
    // {
    //     float* sparse_1 = sparse_ + ((warpin_id%4)*2*16) + (warpin_id/4);
    //     if( *(sparse_1) > 0){
    //         *(output_matrix + value_offset + *(sparse_1)) = output_fragment[0];
    //     }
    //     if( *(sparse_1+8) > 0){
    //         *(output_matrix + value_offset + *(sparse_1+8)) = output_fragment[2];
    //     }

    //     if((row+1)<mOri)
    //     {
    //         if( *(sparse_1+16) > 0){
    //             *(output_matrix + value_offset + *(sparse_1+16)) = output_fragment[1];
    //         }
    //         if( *(sparse_1+24) > 0){
    //             *(output_matrix + value_offset + *(sparse_1+24)) = output_fragment[3];
    //         }
    //     }
    // }
}

//tcu cuda
float sddmm_forward_tf32_tcu_part_kernel_csr(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    int * d_t_row,

    float * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int maxPart,
    const int dimN,
    const int mOri,
    int epoches)
{
    int splitk = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk=8;
    else splitk=((parts_t/1250000)+1)*20;
    dim3 grid_dim(((maxPart/warps)+1), splitk ,(parts_t/splitk+1));
    dim3 block_dim(warps*32, 1, 1);

    for(int iter=0; iter<0; ++iter){
    sddmm_forward_tf32_csr_v2_kernel_tcu<<<grid_dim, block_dim>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_row, 
            t_window_row,
            rhs_matrix,  
            output_matrix,
            parts_t, dimN, mOri, splitk, warps);
    }
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        sddmm_forward_tf32_csr_v2_kernel_tcu<<<grid_dim, block_dim>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_row, 
            t_window_row,
            rhs_matrix,  
            output_matrix,
            parts_t, dimN, mOri, splitk, warps);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}


// sddmm tcu tf32 csr
__global__ void sddmm_forward_fp16_csr_v2_kernel_tcu(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const int* __restrict__ t_column,
    const int* __restrict__ t_row,
    const int* t_window_row,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int warps)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;

    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;
    int id=blockIdx.x*warps+(threadIdx.x/32);
    if(id>= tcu_blocks) return;
    int warp_id = threadIdx.x>>5;
    int warpin_id = threadIdx.x%32;
    at::Half output_fragment_[4] = {0,0,0,0};
    half * output_fragment = reinterpret_cast< half *>(output_fragment_);
    long col=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4));
    long col1=*(t_column + t_win_offset*16 + (id*16) + (warpin_id/4) +8);
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    long row=cur_m_index_vec*8 + (warpin_id/4);
    const half *r_matrix;
    if(col>=0) r_matrix = rhs_matrix + col*nOri;
    const half *l_matrix = rhs_matrix + row*nOri;
    int steps = nOri/16;
    int residue = nOri%16;
    mmaComputeUtils_fp16_gen computer(
        reinterpret_cast<const at::Half *>(l_matrix),
        reinterpret_cast<const at::Half *>(r_matrix), 
        reinterpret_cast<uint32_t *>(output_fragment), 
        threadIdx.x);

    if(steps>0)
    for(int i = 0; i < steps; i++){
        computer.TileMAC(nOri, col, col1, mOri, row);
    }
    if(residue>0)
        computer.TileMACResidue(residue, nOri, col, col1, mOri, row);

    //output
    __shared__ half sparse[128*4];
    half * sparse_ = sparse + warp_id*128;
    //把结果转置写入sparse
    half* sparse_1 = sparse_ + ((warpin_id%4)*2*16) + (warpin_id/4);
    *(sparse_1) = output_fragment[0];
    *(sparse_1+8) = output_fragment[2];
    *(sparse_1+16) = output_fragment[1];
    *(sparse_1+24) = output_fragment[3];
    __syncwarp();
    // if(m_index_vec==0 && threadIdx.x==0)
    // {
    //     for(int q=0;q<128;q++)
    //     printf("%f ", sparse_[q]);
    //     printf("\n");
    // }
    // for(int q=0;q<4;q++)
    // sparse_[warpin_id*4+q] = -1;
    //__syncwarp();
    int value_offset = __ldg(t_block_offset + t_win_offset + id);
    int nnz_block = __ldg(t_block_offset + t_win_offset + id + 1) - value_offset;
    for(int q=0;q<4;q++)
    {
        if((warpin_id*4 + q) < nnz_block){
            int temp = __ldg(t_row + value_offset + warpin_id*4 + q);
            *(output_matrix + value_offset + warpin_id*4 + q) = *(sparse_ + temp);
        }
    }
    // __syncwarp();
    // if(m_index_vec==0 && threadIdx.x==0)
    // {
    //     for(int q=0;q<128;q++)
    //     printf("%d ", sparse_[q]);
    //     printf("\n");
    // }
    // if(row<mOri)
    // {
    //     float* sparse_1 = sparse_ + ((warpin_id%4)*2*16) + (warpin_id/4);
    //     if( *(sparse_1) > 0){
    //         *(output_matrix + value_offset + *(sparse_1)) = output_fragment[0];
    //     }
    //     if( *(sparse_1+8) > 0){
    //         *(output_matrix + value_offset + *(sparse_1+8)) = output_fragment[2];
    //     }

    //     if((row+1)<mOri)
    //     {
    //         if( *(sparse_1+16) > 0){
    //             *(output_matrix + value_offset + *(sparse_1+16)) = output_fragment[1];
    //         }
    //         if( *(sparse_1+24) > 0){
    //             *(output_matrix + value_offset + *(sparse_1+24)) = output_fragment[3];
    //         }
    //     }
    // }
}

//tcu cuda
float sddmm_forward_fp16_tcu_part_kernel_csr(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    int * d_t_row,

    half * rhs_matrix,
    half * output_matrix,

    const int parts_t,
    const int maxPart,
    const int dimN,
    const int mOri,
    int epoches)
{
    int splitk = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk=8;
    else splitk=((parts_t/1250000)+1)*20;
    dim3 grid_dim(((maxPart/warps)+1), splitk ,(parts_t/splitk+1));
    dim3 block_dim(warps*32, 1, 1);

    for(int iter=0; iter<10; ++iter){
    sddmm_forward_fp16_csr_v2_kernel_tcu<<<grid_dim, block_dim>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_row, 
            t_window_row,
            rhs_matrix,  
            output_matrix,
            parts_t, dimN, mOri, splitk, warps);
    }
    cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        sddmm_forward_fp16_csr_v2_kernel_tcu<<<grid_dim, block_dim>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_row, 
            t_window_row,
            rhs_matrix,  
            output_matrix,
            parts_t, dimN, mOri, splitk, warps);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}

//tcu+cuda
//tcu cuda
void spmm_forward_tf32_tcu_cuda_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * d_t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    float * rhs_matrix,
    float * output_matrix_t,
    float * output_matrix_c,

    const int parts_t,
    const int maxPart,
    const int nOri,
    const int mOri,
    int parts)
{
    //tcu
    int splitk_t = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim_t(((maxPart/warps)+1), splitk_t ,(parts_t/splitk_t+1));
    dim3 block_dim_t(warps*32, 1, 1);

    //cuda
    int splitk_c = 0;
    if(parts<500000) splitk_c=8;
    else splitk_c=((parts/1250000)+1)*20;
    dim3 grid_dim_c(1, splitk_c ,((parts/splitk_c)+1));
    dim3 block_dim_c(32, 1, 1);
    int sharedmemory = 32*32*sizeof(float)+ 32*sizeof(int);
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);


        sddmm_forward_tf32_binary_v2_kernel_tcu<<<grid_dim_t, block_dim_t, 0, stream1>>>(
                t_row_offset, 
                t_blockNew_offset,
                t_column, 
                d_t_binary, 
                t_window_row,
                rhs_matrix,  
                output_matrix_t,
                parts_t, nOri, mOri, splitk_t, warps);

        sddmm_forward_tf32_csr_v2_kernel_cuda<<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix_c,
            parts, nOri, mOri, splitk_c);

}


float spmm_forward_tf32_tcu_cuda_kernel_navie(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * d_t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    float * rhs_matrix,
    float * output_matrix_t,
    float * output_matrix_c,

    const int parts_t,
    const int maxPart,
    const int nOri,
    const int mOri,
    int epoches,
    int parts)
{
    //tcu
    int splitk_t = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim_t(((maxPart/warps)+1), splitk_t ,(parts_t/splitk_t+1));
    dim3 block_dim_t(warps*32, 1, 1);

    //cuda
    int splitk_c = 0;
    if(parts<500000) splitk_c=8;
    else splitk_c=((parts/1250000)+1)*20;
    dim3 grid_dim_c(1, splitk_c ,((parts/splitk_c)+1));
    dim3 block_dim_c(32, 1, 1);
    int sharedmemory = 32*32*sizeof(float)+ 32*sizeof(int);
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    for(int iter=0; iter<10; ++iter){
    sddmm_forward_tf32_binary_v2_kernel_tcu<<<grid_dim_t, block_dim_t, 0, stream1>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_binary, 
            t_window_row,
            rhs_matrix,  
            output_matrix_t,
            parts_t, nOri, mOri, splitk_t, warps);

    sddmm_forward_tf32_csr_v2_kernel_cuda_v2<<<grid_dim_c, block_dim_c, 0 , stream2>>>(
        c_row_offset, 
        c_row,
        c_column,
        rhs_matrix,  
        output_matrix_c,
        parts, nOri, mOri, splitk_c);
    }
        cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        sddmm_forward_tf32_binary_v2_kernel_tcu<<<grid_dim_t, block_dim_t, 0, stream1>>>(
                t_row_offset, 
                t_blockNew_offset,
                t_column, 
                d_t_binary, 
                t_window_row,
                rhs_matrix,  
                output_matrix_t,
                parts_t, nOri, mOri, splitk_t, warps);

        sddmm_forward_tf32_csr_v2_kernel_cuda_v2<<<grid_dim_c, block_dim_c, 0, stream2>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix_c,
            parts, nOri, mOri, splitk_c);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}

void spmm_forward_fp16_tcu_cuda_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * d_t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    half * rhs_matrix,
    half * output_matrix_t,
    half * output_matrix_c,

    const int parts_t,
    const int maxPart,
    const int nOri,
    const int mOri,
    int parts)
{
    //tcu
    int splitk_t = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim_t(((maxPart/warps)+1), splitk_t ,(parts_t/splitk_t+1));
    dim3 block_dim_t(warps*32, 1, 1);

    //cuda
    int splitk_c = 0;
    if(parts<500000) splitk_c=8;
    else splitk_c=((parts/1250000)+1)*20;
    dim3 grid_dim_c(1, splitk_c ,((parts/splitk_c)+1));
    dim3 block_dim_c(32, 1, 1);
    int sharedmemory = 32*32*sizeof(float)+ 32*sizeof(int);
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

   
        sddmm_forward_fp16_binary_v2_kernel_tcu<<<grid_dim_t, block_dim_t, 0, stream1>>>(
                t_row_offset, 
                t_blockNew_offset,
                t_column, 
                d_t_binary, 
                t_window_row,
                rhs_matrix,  
                output_matrix_t,
                parts_t, nOri, mOri, splitk_t, warps);

        sddmm_forward_fp16_csr_v2_kernel_cuda<<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix_c,
            parts, nOri, mOri, splitk_c);

}
// fp16
template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_cuda(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_atomic,
    const int* __restrict__ c_column,
    const half* __restrict__ c_value,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int parts,
    int grid_x)
{
    int c_part_offset_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(c_part_offset_vec>=parts) return;

    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    int dimN_index = blockIdx.x * Tile_N + warp_id*64;
    if((warp_id+1)*64 > (nOri + 32))
    return;
    // int c_part_offset_vec = m_index_vec; 
    // if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        // int dimN_index = blockIdx.x * Tile_N;

        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        // __shared__ float c_part_result[dimN];
        extern __shared__ at::Half sparse_2[];
        //half * sparse = sparse_;
        half * sparse =reinterpret_cast<half *>(sparse_2);
        int * sparse_col = (int *) & sparse_2[32];
        // *(c_part_result) = 0.0;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            
            // __shared__ float sparse[16*4];      
            //32个线程把数据搬运到shared, 前提是partSize小于等于32
            if(threadIdx.x<nonzeros){
                *(sparse + threadIdx.x) = __ldg(c_value + c_row_offset_vec + threadIdx.x);
                *(sparse_col + threadIdx.x) = __ldg(c_column + c_row_offset_vec + threadIdx.x);               
            }
            __syncthreads();
            // if(threadIdx.x==0 && c_part_offset_vec==0)
            // {printf("%f, %f, %f\n",sparse[0], sparse[1] , sparse[2]);
            // printf("%d, %d, %d\n",sparse_col[0], sparse_col[1] , sparse_col[2]);}
            cudaComputeUtils_fp16_trans compute(
            rhs_matrix,
            sparse_col,
            // c_column + c_row_offset_vec,
            sparse,
            // c_value + c_row_offset_vec,
            output_matrix,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros, dimN_index,
            out_row_offset,
            __ldg(c_atomic + c_part_offset_vec));    

        }
    } 
}

template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_tcu(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const half* __restrict__ t_value,
    const int* __restrict__ t_column,
    const long* __restrict__ t_binary,
    const int* t_window_row,
    const int* t_atomic,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;
    //判断执行tcu还是cuda

    //tcu
    // 需要计算的TCU block个数tcu_blocks
     int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;

    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;
    //用于TCU计算的结果
    uint32_t output_fragment_[2] = {0,0};
    half * output_fragment = reinterpret_cast<half *>(output_fragment_);
    //稀疏的块, 16*8
    // __shared__ at::Half sparse_[64];
    // half * sparse = reinterpret_cast<half *>(sparse_);
    // __shared__ int sparse_to_col[8];

    float sparse_fragment[1] = {0.0};
    at::Half dense_fragment1_[4] = {0.0, 0.0, 0.0, 0.0};
    half * sparse_fragment1 = reinterpret_cast<half *>(sparse_fragment);
    half * dense_fragment = reinterpret_cast<half *>(dense_fragment1_);
    uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
    uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
    
    const int * t_column_ = t_column + t_win_offset*8;
    // uint32_t * t_output_fragment_ = reinterpret_cast<uint32_t*>(t_output_fragment);
    //读取稠密矩阵的行偏移
    int col_offset = dimN_index + (warp_id<<4) + (warpin_id/4);
    const half * matrix_base_ = rhs_matrix + col_offset;
    //循环遍历每个block
    for(int i=0; i<tcu_blocks; i++)
    {
        // __syncthreads();
        //block内非零元的数量
        int value_offset = __ldg(t_block_offset + t_win_offset + i);
        long binary_1 = __ldg(t_binary + t_win_offset*2 + 2*i);
        long binary_2 = __ldg(t_binary + t_win_offset*2 + 2*i + 1);
        long a= 1;
        long mask = (a << (warpin_id*2));

        long temp = (binary_1 >> (warpin_id*2));
        int fifthBit = ((temp) & 1);
        int block_offset = -1;
        if(fifthBit == 1){
            block_offset = __popcll(binary_1 & (mask-1));
            sparse_fragment1[0] = __ldg(t_value + value_offset + block_offset);
        }else{
            sparse_fragment1[0]=__float2half(0.0);
        }
        fifthBit = ((temp>>1) & 1);
        if(fifthBit == 1){
            if(block_offset==-1)
            {
                mask = (a << ((warpin_id*2)+1));
                block_offset = __popcll(binary_1 & (mask-1));
            }
            sparse_fragment1[1] = __ldg(t_value + value_offset + block_offset + 1);
        }else{
            sparse_fragment1[1]=__float2half(0.0);
        }
        //搬运dense数据
        int col =  __ldg(t_column_ + (threadIdx.x%4)*2);
        int col1 =  __ldg(t_column_ + (threadIdx.x%4)*2 + 1);
        t_column_ += 8;
        for(int d=0;d<2;d++)
        {
            if((col_offset + d*8) < nOri)
            { 
                    if(col != -1)
                    {
                        *(dense_fragment + 2*d) = __ldg(matrix_base_ + (col*nOri) + d*8);
                    }

                    if(col1 != -1)
                    {
                        *(dense_fragment + 2*d + 1) = __ldg(matrix_base_ + (col1*nOri) + d*8);
                    }
            }

        } 
        __syncwarp();

            //MMA计算
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
                "{%0,%1}, \t"
                "{%2,%3}, \t"
                "{%4}, \t"
                "{%0,%1}; ":
                "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                "r"(dense_fragment_[0]),  "r"(dense_fragment_[1]),
                "r"(sparse_fragment_[0])
            );  
    //第二次
            int nnz = __popcll(binary_1);
         temp = (binary_2 >> (warpin_id*2));
         fifthBit = ((temp) & 1);
         block_offset = -1;
        if(fifthBit == 1){
            block_offset = __popcll(binary_2 & (mask-1));
            sparse_fragment1[0] = __ldg(t_value + value_offset + nnz + block_offset);
        }else{
            sparse_fragment1[0]=__float2half(0.0);
        }
        fifthBit = ((temp>>1) & 1);
        if(fifthBit == 1){
            if(block_offset==-1)
            {
                mask = (a << ((warpin_id*2)+1));
                block_offset = __popcll(binary_2 & (mask-1));
            }
            sparse_fragment1[1] = __ldg(t_value + value_offset + nnz + block_offset + 1);
        }else{
            sparse_fragment1[1]=__float2half(0.0);
        }
        //搬运dense数据
         col =  __ldg(t_column_ + (threadIdx.x%4)*2);
         col1 =  __ldg(t_column_ + (threadIdx.x%4)*2 + 1);
        t_column_ += 8;
        for(int d=0;d<2;d++)
        {
            if((col_offset + d*8) < nOri)
            { 
                    if(col != -1)
                    {
                        *(dense_fragment + 2*d) = __ldg(matrix_base_ + (col*nOri) + d*8);
                    }

                    if(col1 != -1)
                    {
                        *(dense_fragment + 2*d + 1) = __ldg(matrix_base_ + (col1*nOri) + d*8);
                    }
            }

        } 
        __syncwarp();

            //MMA计算
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 \t"
                "{%0,%1}, \t"
                "{%2,%3}, \t"
                "{%4}, \t"
                "{%0,%1}; ":
                "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                "r"(dense_fragment_[0]),  "r"(dense_fragment_[1]),
                "r"(sparse_fragment_[0])
            );  
            
    }
        int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
        int cur_t_atomic = __ldg(t_atomic + m_index_vec);
        int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
        int col=dimN_index + warp_id*16 + + (warpin_id/4);

        if(row<mOri)
        {
            float * output_matrix_ = output_matrix +(row*nOri)+col;
            if(cur_t_atomic==0)
            {
                if(col<nOri)
                *(output_matrix_ ) = __half2float(output_fragment[0]);
                if((col+8)<nOri)
                *(output_matrix_+8) =  __half2float(output_fragment[2]);
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    *(output_matrix_) = __half2float(output_fragment[1]);
                    if((col+8)<nOri)
                    *(output_matrix_+8) = __half2float( output_fragment[3]);
                }
            }else{
                if(col<nOri)
                atomicAdd(output_matrix_ ,__half2float(output_fragment[0]));
                if((col+8)<nOri)
                atomicAdd(output_matrix_+8, __half2float(output_fragment[2]));
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    atomicAdd(output_matrix_ , __half2float(output_fragment[1]));
                    if((col+8)<nOri)
                    atomicAdd(output_matrix_+8 , __half2float(output_fragment[3]));
                }
            }
        }
}



void spmm_forward_fp16_tcu_cuda_kernel_spmm(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * t_binary,
    int * t_atomic,
    half * t_value, 

    int * c_row_offset,
    int * c_row,
    int * c_column, 
    int * c_atomic,
    half * c_value, 

    half * rhs_matrix,
    float * output_matrix,


    const int parts_t,
    const int parts_c,
    const int nOri,
    const int mOri)
{
    //tcu
    int n1_t=nOri;
    if((nOri%16)!=0) n1_t=((nOri/16)+1)*16;
    int grid_x_t = (n1_t/64)+1;
    if(n1_t%64==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(128, 1, 1);

    //cuda
    int n1_c=nOri;
    if((nOri%64)!=0) n1_c=((nOri/64)+1)*64;
    int grid_x_c = (n1_c/128)+1;
    if(n1_c%128==0) grid_x_c-=1;

    int windows =  parts_c;
    int splitk_c = 0;
    if(windows<500000) splitk_c=8;
    else splitk_c=((windows/1250000)+1)*20;

    dim3 grid_dim_c(grid_x_c, splitk_c ,((windows/splitk_c)+1));
    dim3 block_dim(64, 1, 1);

    int sharedmemory = 32*(sizeof(half)+ sizeof(int));
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

   
        spmm_forward_fp16_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
                    t_row_offset, 
                    t_blockNew_offset,
                    t_value, 
                    t_column, 
                    t_binary,
                    t_window_row,
                    t_atomic,
                    rhs_matrix,  
                    output_matrix,
                    n1_t, parts_t, nOri, mOri, splitk_t, grid_x_t);


        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim, sharedmemory,stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, windows, nOri, mOri, splitk_c, parts_c, grid_x_c);

}
float spmm_forward_fp16_tcu_cuda_kernel_navie(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * d_t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    half * rhs_matrix,
    half * output_matrix_t,
    half * output_matrix_c,

    const int parts_t,
    const int maxPart,
    const int nOri,
    const int mOri,
    int epoches,
    int parts)
{
    //tcu
    int splitk_t = 0;
    //一个block中4个warp
    int warps = 4;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim_t(((maxPart/warps)+1), splitk_t ,(parts_t/splitk_t+1));
    dim3 block_dim_t(warps*32, 1, 1);

    //cuda
    int splitk_c = 0;
    if(parts<500000) splitk_c=8;
    else splitk_c=((parts/1250000)+1)*20;
    dim3 grid_dim_c(1, splitk_c ,((parts/splitk_c)+1));
    dim3 block_dim_c(32, 1, 1);
    int sharedmemory = 32*32*sizeof(float)+ 32*sizeof(int);
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    for(int iter=0; iter<10; ++iter){
    sddmm_forward_fp16_binary_v2_kernel_tcu<<<grid_dim_t, block_dim_t, 0, stream1>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_column, 
            d_t_binary, 
            t_window_row,
            rhs_matrix,  
            output_matrix_t,
            parts_t, nOri, mOri, splitk_t, warps);

    sddmm_forward_fp16_csr_v2_kernel_cuda_v2<<<grid_dim_c, block_dim_c, 0 , stream2>>>(
        c_row_offset, 
        c_row,
        c_column,
        rhs_matrix,  
        output_matrix_c,
        parts, nOri, mOri, splitk_c);
    }
        cudaDeviceSynchronize();
    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end); 
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        sddmm_forward_fp16_binary_v2_kernel_tcu<<<grid_dim_t, block_dim_t, 0, stream1>>>(
                t_row_offset, 
                t_blockNew_offset,
                t_column, 
                d_t_binary, 
                t_window_row,
                rhs_matrix,  
                output_matrix_t,
                parts_t, nOri, mOri, splitk_t, warps);

        sddmm_forward_fp16_csr_v2_kernel_cuda_v2<<<grid_dim_c, block_dim_c, 0, stream2>>>(
            c_row_offset, 
            c_row,
            c_column,
            rhs_matrix,  
            output_matrix_c,
            parts, nOri, mOri, splitk_c);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}


template <int Tile_N>
__global__ void spmm_forward_tf32_csr_v2_kernel_cuda(
    const int* __restrict__ c_row_offset,
    const int* __restrict__ c_row,
    const int* __restrict__ c_atomic,
    const int* __restrict__ c_column,
    const float* __restrict__ c_value,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int parts,
    int grid_x)
{
    int c_part_offset_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(c_part_offset_vec>=parts) return;
    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    int dimN_index = blockIdx.x * Tile_N + warp_id*64;
    if((warp_id+1)*64 > (nOri + 32))
    return;
    // int c_part_offset_vec = m_index_vec; 
    // if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        // int dimN_index = blockIdx.x * Tile_N;

        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        // __shared__ float c_part_result[dimN];
        extern __shared__ float sparse_[];
        float * sparse =sparse_;
        int * sparse_col = (int *) & sparse_[32];
        // *(c_part_result) = 0.0;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            // printf("11111");
            // __shared__ float sparse[16*4];      
            //32个线程把数据搬运到shared, 前提是partSize小于等于32
            if(threadIdx.x<nonzeros){
                *(sparse + threadIdx.x) = __ldg(c_value + c_row_offset_vec + threadIdx.x);
                *(sparse_col + threadIdx.x) = __ldg(c_column + c_row_offset_vec + threadIdx.x);               
            }
            __syncthreads();
            // if(threadIdx.x==0 && c_part_offset_vec==0)
            // {printf("%f, %f, %f\n",sparse[0], sparse[1] , sparse[2]);
            // printf("%d, %d, %d\n",sparse_col[0], sparse_col[1] , sparse_col[2]);}
            cudaComputeUtils_tf32_trans compute(
            rhs_matrix,
            sparse_col,
            // c_column + c_row_offset_vec,
            sparse,
            // c_value + c_row_offset_vec,
            output_matrix,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros, dimN_index,
            out_row_offset,
            __ldg(c_atomic + c_part_offset_vec));    

        }
    } 
    
    
}

template <int Tile_N>
__global__ void spmm_forward_tf32_csr_v2_kernel_tcu(
    const int* __restrict__ t_window_offset,
    const int* __restrict__ t_block_offset,
    const float* __restrict__ t_value,
    const int* __restrict__ t_column,
    const long* __restrict__ t_binary,
    const int* t_window_row,
    const int* t_atomic,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int windows,
    int nOri,
    int mOri,
    int splitk,
    int grid_x)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;
    int dimN_index = blockIdx.x * Tile_N;
    //判断执行tcu还是cuda

    //tcu
    // 需要计算的TCU block个数tcu_blocks
    int t_win_offset = __ldg(t_window_offset + m_index_vec);
    int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
    if(tcu_blocks==0) return;

        int warp_id = threadIdx.x>>5;
        if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
        int warpin_id = threadIdx.x%32;
        //用于TCU计算的结果
        float t_output_fragment[4] = {0.0, 0.0, 0.0, 0.0}; 
        //稀疏的块, 16*8
        // __shared__ float sparse[32];
        // __shared__ int sparse_to_col[4];
        float sparse_fragment[2] = {0.0, 0.0};
        float dense_fragment[2] = {0.0, 0.0};
        uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
        uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
        const int * t_column_ = t_column + t_win_offset*8;
        //读取稠密矩阵的行偏移
        int col_offset = dimN_index + (warp_id<<4) + (warpin_id/4);
        const float * matrix_base_ = rhs_matrix + col_offset;
        //循环遍历每个block
        for(int i=0; i<tcu_blocks; i++)
        {
            //每个block需要进行两次8x8的mma，先取第一块，再取第二块；取第一块后需要计算第一块中的非零元个数
            long binary_1 = __ldg(t_binary + t_win_offset*2 + i*2);
            long binary_2 = __ldg(t_binary + t_win_offset*2 + i*2 + 1);
            long a= 1;
            int mask1 = (a << (warpin_id));
            int mask2 = (a << (warpin_id + 32));
            int value_offset = __ldg(t_block_offset + t_win_offset + i);

            //第一次
            //block内非零元的数量
            long temp = (binary_1 >> (warpin_id));
            int fifthBit = ((temp) & 1);
            if(fifthBit == 1){
                int block_offset = __popcll(binary_1 & (mask1-1));
                sparse_fragment[0] = __ldg(t_value + value_offset + block_offset);
            }else{
                sparse_fragment[0]= 0.0;
            }
            temp = temp >> 32;
            fifthBit = ((temp) & 1);
            if(fifthBit == 1){
                int block_offset = __popcll(binary_1 & (mask2-1));
                sparse_fragment[1] = __ldg(t_value + value_offset + block_offset);
            }else{
                sparse_fragment[1]= 0.0;
            }

            //搬运dense数据
            int col =  __ldg(t_column_ + (threadIdx.x%4));
            int col1 =  __ldg(t_column_ + (threadIdx.x%4) + 4);
            t_column_ += 8;
            for(int d=0;d<2;d++)
            {
                if((col_offset + d*8) < nOri)
                { 
                    if(col != -1)
                    {
                        *(dense_fragment + d) = __ldg(matrix_base_ + (col*nOri) +  d*8);
                    }
                    if(col1 != -1)
                    {
                        *(dense_fragment + d + 2) = __ldg(matrix_base_ + (col1*nOri) +  d*8);
                    }
                }
            }

            __syncwarp();
            //MMA计算
            asm volatile(
            "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                : "=f"(t_output_fragment[0]), "=f"(t_output_fragment[1]), "=f"(t_output_fragment[2]), "=f"(t_output_fragment[3])
                : "r"(dense_fragment_[0]), "r"(dense_fragment_[1]), "r"(sparse_fragment_[0]), "f"(t_output_fragment[0]), "f"(t_output_fragment[1]), "f"(t_output_fragment[2]), "f"(t_output_fragment[3]));
        

            //第二次开始
            //获取第一次中有多少非零元
            int nnz = __popcll(binary_1);
            temp = (binary_2 >> (warpin_id));
            fifthBit = ((temp) & 1);
            if(fifthBit == 1){
                int block_offset = __popcll(binary_2 & (mask1-1));
                sparse_fragment[0] = __ldg(t_value + value_offset + nnz + block_offset);
            }else{
                sparse_fragment[0]= 0.0;
            }
            temp = temp >> 32;
            fifthBit = ((temp) & 1);
            if(fifthBit == 1){
                int block_offset = __popcll(binary_2 & (mask2-1));
                sparse_fragment[1] = __ldg(t_value + value_offset + + nnz + block_offset);
            }else{
                sparse_fragment[1]= 0.0;
            }

            //搬运稠密数据
             col =  __ldg(t_column_ + (threadIdx.x%4));
             col1 =  __ldg(t_column_ + (threadIdx.x%4) + 4);
            for(int d=0;d<2;d++)
            {
                if((col_offset + d*8) < nOri)
                { 
                    if(col != -1)
                    {
                        *(dense_fragment + d) = __ldg(matrix_base_ + (col*nOri) +  d*8);
                    }
                    if(col1 != -1)
                    {
                        *(dense_fragment + d + 2) = __ldg(matrix_base_ + (col1*nOri) +  d*8);
                    }
                }
            }
            __syncwarp();
            //MMA计算
            asm volatile(
            "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                : "=f"(t_output_fragment[0]), "=f"(t_output_fragment[1]), "=f"(t_output_fragment[2]), "=f"(t_output_fragment[3])
                : "r"(dense_fragment_[0]), "r"(dense_fragment_[1]), "r"(sparse_fragment_[0]), "f"(t_output_fragment[0]), "f"(t_output_fragment[1]), "f"(t_output_fragment[2]), "f"(t_output_fragment[3]));
        }

        //原子写入gloabl
        int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
        int cur_t_atomic = __ldg(t_atomic + m_index_vec);
        int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
        int col=dimN_index + warp_id*16 + + (warpin_id/4);

        if(row<mOri)
        {
            float * output_matrix_ = output_matrix +(row*nOri)+col;
            if(cur_t_atomic==0)
            {
                if(col<nOri)
                *(output_matrix_ ) = t_output_fragment[0];
                if((col+8)<nOri)
                *(output_matrix_+8) =  t_output_fragment[2];
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    *(output_matrix_) = t_output_fragment[1];
                    if((col+8)<nOri)
                    *(output_matrix_+8) =  t_output_fragment[3];
                }
            }else{
                if(col<nOri)
                atomicAdd(output_matrix_ , t_output_fragment[0]);
                if((col+8)<nOri)
                atomicAdd(output_matrix_+8, t_output_fragment[2]);
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    atomicAdd(output_matrix_ , t_output_fragment[1]);
                    if((col+8)<nOri)
                    atomicAdd(output_matrix_+8 , t_output_fragment[3]);
                }
            }
        }
}

//spmm back
void spmm_forward_tf32_tcu_cuda_kernel_spmm(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * t_binary,
    int * t_atomic,
    float * t_value, 

    int * c_row_offset,
    int * c_row,
    int * c_column, 
    int * c_atomic,
    float * c_value, 

    float * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int parts_c,
    const int nOri,
    const int mOri)
{
    //tcu
    int n1_t=nOri;
    if((nOri%16)!=0) n1_t=((nOri/16)+1)*16;
    int grid_x_t = (n1_t/64)+1;
    if(n1_t%64==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(128, 1, 1);

    //cuda
    int n1_c=nOri;
    if((nOri%64)!=0) n1_c=((nOri/64)+1)*64;
    int grid_x_c = (n1_c/128)+1;
    if(n1_c%128==0) grid_x_c-=1;

    int windows =  parts_c;
    int splitk_c = 0;
    if(windows<500000) splitk_c=8;
    else splitk_c=((windows/1250000)+1)*20;

    dim3 grid_dim_c(grid_x_c, splitk_c ,((windows/splitk_c)+1));
    dim3 block_dim(64, 1, 1);

    int sharedmemory = 32*sizeof(float)+ 32*sizeof(int);
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);


        spmm_forward_tf32_csr_v2_kernel_tcu<64><<<grid_dim_t, block_dim_t, 0, stream1>>>(
            t_row_offset, 
            t_blockNew_offset,
            t_value, 
            t_column, 
            t_binary,
            t_window_row,
            t_atomic,
            rhs_matrix,  
            output_matrix,
            n1_t, parts_t, nOri, mOri, splitk_t, grid_x_t);

        spmm_forward_tf32_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, windows, nOri, mOri, splitk_c, parts_c, grid_x_c);

}