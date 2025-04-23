#include <stdio.h>
#include <mma.h>
#include <cstdint>
#include <iostream>
#include <torch/extension.h>
#include "./spmm_utils/dense_tile.h"
#include "./spmm_utils/compute.h"
#include "./spmm_utils/output_tile.h"
#define mma_k = 8
#include <cuda_fp16.h>
#include <cuda_runtime.h>


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
    
    const int * t_column_ = t_column + t_win_offset*8 + ((warpin_id %4)*2);
    // uint32_t * t_output_fragment_ = reinterpret_cast<uint32_t*>(t_output_fragment);
    //读取稠密矩阵的行偏移

    const half2 * matrix_base_ = reinterpret_cast<const half2 *>(rhs_matrix + dimN_index);
    //循环遍历每个block
    for(int i=0; i<tcu_blocks; i++)
    {
        // __syncthreads();
        //block内非零元的数量
        int value_offset = __ldg(t_block_offset + t_win_offset + i);
        // int nnz_block = __ldg(t_block_offset + t_win_offset + i + 1) - value_offset;
        long binary = __ldg(t_binary + t_win_offset + i);

        long temp = (binary >> (warpin_id*2));
        long a= 1;
        long mask = (a << (warpin_id*2));
        int fifthBit = ((temp) & 1);
        int block_offset = -1;
        if(fifthBit == 1){
            block_offset = __popcll(binary & (mask-1));
            sparse_fragment1[0] = __ldg(t_value + value_offset + block_offset);
        }else{
            sparse_fragment1[0]=__float2half(0.0);
        }
        fifthBit = ((temp>>1) & 1);
        if(fifthBit == 1){
            if(block_offset==-1)
            {
                mask = (a << ((warpin_id*2)+1));
                block_offset = __popcll(binary & (mask-1));
            }
            sparse_fragment1[1] = __ldg(t_value + value_offset + block_offset + 1);
        }else{
            sparse_fragment1[1]=__float2half(0.0);
        }
        //搬运稀疏数据
        // if(threadIdx.x<nnz_block)
        // {
        //     half v = __ldg(t_value + value_offset + threadIdx.x);
        //     int row = __ldg(t_row + value_offset + threadIdx.x);
        //     int col = __ldg(t_col + value_offset + threadIdx.x);
        //     *(sparse + row*8 + col) = v;
        // }
        //  __syncthreads();
        //搬运dense数据
        // int col =  __ldg(t_column_ + (threadIdx.x%4)*2);
        // int col1 =  __ldg(t_column_ + (threadIdx.x%4)*2 + 1);
        long col_temp[2];
        for(int k=0; k<2; k++)
            col_temp[k] = __ldg(t_column_ + k);
        t_column_ += 8;
        int col_offset =  (warp_id<<3) + (warpin_id/4);
        for(int i=0;i<2;i++)
        {
            if(col_temp[i] != -1){
                const long offset = (col_temp[i]*(nOri/2));
                half2 temp = __ldg(matrix_base_ + offset + col_offset);
                dense_fragment[i]= temp.x;
                dense_fragment[i + 2]= temp.y;
            }else{
                dense_fragment[i]= __float2half(0.0);
                dense_fragment[i + 2]= __float2half(0.0);
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
        int col=dimN_index + warp_id*16 + + (warpin_id/4)*2;

        if(row<mOri)
        {
            float * output_matrix_ = output_matrix +(row*nOri)+col;
            if(cur_t_atomic==0)
            {
                if(col<nOri)
                *(output_matrix_ ) = __half2float(output_fragment[0]);
                if((col+1)<nOri)
                *(output_matrix_+1) =  __half2float(output_fragment[2]);
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    *(output_matrix_) = __half2float(output_fragment[1]);
                    if((col+1)<nOri)
                    *(output_matrix_+1) = __half2float( output_fragment[3]);
                }
            }else{
                if(col<nOri)
                atomicAdd(output_matrix_ ,__half2float(output_fragment[0]));
                if((col+1)<nOri)
                atomicAdd(output_matrix_+1, __half2float(output_fragment[2]));
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    atomicAdd(output_matrix_ , __half2float(output_fragment[1]));
                    if((col+1)<nOri)
                    atomicAdd(output_matrix_+1 , __half2float(output_fragment[3]));
                }
            }
        }
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
    int grid_x,
    int partsize)
{
    int c_part_offset_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(c_part_offset_vec>=parts) return;

    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    // if((warp_id+1)*64 > (nOri + 32))
    // return;
    // int c_part_offset_vec = m_index_vec; 
    // if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        // __shared__ float c_part_result[dimN];
        extern __shared__ at::Half sparse_2[];
        //half * sparse = sparse_;
        half * sparse =reinterpret_cast<half *>(sparse_2);
        int * sparse_col = (int *) & sparse_2[partsize];
        // *(c_part_result) = 0.0;
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            
            // __shared__ float sparse[16*4];      
            if(threadIdx.x<nonzeros){
                *(sparse + threadIdx.x) = __ldg(c_value + c_row_offset_vec + threadIdx.x);
                *(sparse_col + threadIdx.x) = __ldg(c_column + c_row_offset_vec + threadIdx.x);               
            }
            __syncthreads();

            cudaComputeUtils_fp16_trans compute(
            reinterpret_cast<half4 *>(rhs_matrix),
            sparse_col,
            sparse,
            output_matrix + out_row_offset*nOri,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros,
            __ldg(c_atomic + c_part_offset_vec));    

        }
    } 
}

template <int Tile_N>
__global__ void spmm_forward_fp16_csr_v2_kernel_cuda_short(
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
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=windows)
    return;

    //判断执行tcu还是cuda

    //cuda
    int warpin_id = threadIdx.x%32;
    int warp_id = threadIdx.x/32;
    int dimN_index = blockIdx.x * Tile_N + warp_id*64;
    if((warp_id+1)*64 > (nOri + 32))
    return;
    int c_part_offset_vec = m_index_vec; 
    // if(swizzle)
    // c_part_offset_vec = __ldg(c_part_offset + c_part_offset_vec);
    if(c_part_offset_vec>=parts) return;
    //当前cuda需要计算的行
    int out_row_offset = __ldg(c_row + c_part_offset_vec);
    if(out_row_offset<mOri)
    {
        // int dimN_index = blockIdx.x * Tile_N;

        int c_row_offset_vec = __ldg(c_row_offset + c_part_offset_vec); 
        int nonzeros = __ldg(c_row_offset + c_part_offset_vec + 1) -  c_row_offset_vec;
        // if(m_index_vec==0 && threadIdx.x==0)
        // printf("%d\n", nonzeros);
        //进行cuda计算
        if(nonzeros!=0) 
        {  
            cudaComputeUtils_fp16_trans_short compute(
            rhs_matrix,
            c_column + c_row_offset_vec,
            c_value + c_row_offset_vec,
            output_matrix,
            nOri,
            warpin_id);           

            compute.cudaCompute(nonzeros, dimN_index,
            out_row_offset,
            __ldg(c_atomic + c_part_offset_vec));    

        }
    } 
    
    
}

//tcu cuda 
float spmm_forward_fp16_tcu_cuda_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    half * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    long *  t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    half * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    half * c_value_short, 

    half * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    int epoches)
{
    //tcu
    int n1_t=dimN;
    if((dimN%16)!=0) n1_t=((dimN/16)+1)*16;
    int grid_x_t = (n1_t/64)+1;
    if(n1_t%64==0) grid_x_t-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    // 4是每个block中的warp数量
    dim3 grid_dim_t(grid_x_t, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim_t(128, 1, 1);

    //cuda
    int n1_c=dimN;
    if((dimN%64)!=0) n1_c=((dimN/64)+1)*64;
    //每个block默认处理128
    int grid_x_c = (n1_c/128)+1;
    if(n1_c%128==0) grid_x_c-=1;

    int windows =  parts_c;
    int splitk_c = 0;
    if(windows<500000) splitk_c=8;
    else splitk_c=((windows/1250000)+1)*20;

    int windows_short =  parts_c_short;
    int splitk_c_short = 0;
    if(windows_short<500000) splitk_c_short=8;
    else splitk_c_short=((windows_short/1250000)+1)*20;

    dim3 grid_dim_c(grid_x_c, splitk_c ,((windows/splitk_c)+1));
    dim3 grid_dim_c_short(grid_x_c, splitk_c_short ,((windows_short/splitk_c_short)+1));
    dim3 block_dim_c(32, 1, 1);
    dim3 block_dim(64, 1, 1);
    int sharedmemory = partsize_c*(sizeof(half)+ sizeof(int));
    // half2 * rhs_matrix_c = reinterpret_cast<half2 *>(rhs_matrix);
    cudaStream_t stream1,stream2,stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    for(int iter=0; iter<10; ++iter){
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
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory, stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, windows, dimN, mOri, splitk_c, parts_c, grid_x_c,partsize_c);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim, 0, stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, windows_short, dimN, mOri, splitk_c_short, parts_c_short, grid_x_c);
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
                    n1_t, parts_t, dimN, mOri, splitk_t, grid_x_t);

        spmm_forward_fp16_csr_v2_kernel_cuda<128><<<grid_dim_c, block_dim_c, sharedmemory,stream2>>>(
            c_row_offset, 
            c_row,
            c_atomic,
            c_column,
            c_value, 
            rhs_matrix,  
            output_matrix,
            n1_c, windows, dimN, mOri, splitk_c, parts_c, grid_x_c,partsize_c);
        
        spmm_forward_fp16_csr_v2_kernel_cuda_short<128><<<grid_dim_c_short, block_dim,0,stream3>>>(
            c_row_offset_short, 
            c_row_short,
            c_atomic_short,
            c_column_short,
            c_value_short, 
            rhs_matrix,  
            output_matrix,
            n1_c, windows_short, dimN, mOri, splitk_c_short, parts_c_short, grid_x_c);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}
