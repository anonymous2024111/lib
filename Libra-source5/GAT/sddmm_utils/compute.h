#include <mma.h>
#include <cstdint>
#include <stdio.h>
#include <cuda_fp16.h>
#include <torch/extension.h>


/*
TF32
*/
struct mmaComputeUtils_tf32_gen{
    const float* lhs_tile_;
    const float* rhs_tile_;
    float* output_fragment_;
    const int lane_id_;
    const int warpin_id;
    const int warp_id;
    int dimN_index_;
    int col_;
    int row_;
    int dimW_;

    // Constructor
    __device__ __forceinline__ mmaComputeUtils_tf32_gen(
        const float *lhs_tile,
        const float *rhs_tile,
        float* output_fragment,
        int lane_id):
            lhs_tile_(lhs_tile),
            rhs_tile_(rhs_tile),
            output_fragment_(output_fragment),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5){}
    
    // Compute
    __device__ __forceinline__ void TileMAC(int dimW, long col, long col1,int dimMori, long row){
        //load lhs_matrix
        float lhs_fragment_tf32[2]={0.0, 0.0};
        uint32_t *lhs_fragment = reinterpret_cast<uint32_t *>(lhs_fragment_tf32);
        if(row < dimMori){
        lhs_fragment_tf32[0]=*(lhs_tile_+((warpin_id%4)));
        lhs_fragment_tf32[1]=*(lhs_tile_+((warpin_id%4))+4);
        lhs_tile_=lhs_tile_+8;}

        //load rhs_matrix
        float rhs_fragment_tf32[4]={0.0, 0.0, 0.0, 0.0};
        uint32_t *rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment_tf32);
        if(col>=0){
        rhs_fragment_tf32[0]=*(rhs_tile_ + (warpin_id%4));
        rhs_fragment_tf32[2]=*(rhs_tile_ + (warpin_id%4) + 4);
        }
        if((col1)>=0){
        rhs_fragment_tf32[1]=*(rhs_tile_ + (warpin_id%4) + ((col1-col)*dimW));
        rhs_fragment_tf32[3]=*(rhs_tile_ + (warpin_id%4) + 4 + ((col1-col)*dimW));
        }
        //更新rhs_tile
        rhs_tile_=rhs_tile_+8;
            // if(warpin_id==4&warp_id==1)
            // {
            //     half *p=reinterpret_cast<half *>(rhs_fragment);
            //     for(int i=0;i<8;i++)
            //     printf("thread_id:%d, value is:%f\n", warpin_id,__half2float(*(p+i)));
            // }
     
       
        asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
            : "r"(rhs_fragment[0]), "r"(rhs_fragment[1]), "r"(rhs_fragment[2]), "r"(rhs_fragment[3]),
             "r"(lhs_fragment[0]), "r"(lhs_fragment[1]), 
             "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));      
    } 

    
    

    __device__ __forceinline__ void TileMACResidue(int residue,int dimW, long col, long col1, int dimMori,  long row){
        //load lhs_matrix
        float lhs_fragment_tf32[2]={0.0, 0.0};
        uint32_t *lhs_fragment = reinterpret_cast<uint32_t *>(lhs_fragment_tf32);
        if((warpin_id%4)<residue)
        lhs_fragment_tf32[0]=*(lhs_tile_+(warpin_id%4));
        if(((warpin_id%4) + 4)<residue)
        lhs_fragment_tf32[1]=*(lhs_tile_+(warpin_id%4) + 4);

        //load rhs_matrix
        float rhs_fragment_tf32[4]={0.0, 0.0, 0.0, 0.0};
        uint32_t *rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment_tf32);
        if(col>=0){
            if((warpin_id%4)<residue)
            rhs_fragment_tf32[0]=*(rhs_tile_ +(warpin_id%4));
            if(((warpin_id%4) +4)<residue)
            rhs_fragment_tf32[2]=*(rhs_tile_ +((warpin_id%4)*2) +8);
        }
        if((col1)>=0){
            if((warpin_id%4)<residue)
            rhs_fragment_tf32[1]=*(rhs_tile_ + (warpin_id%4) + ((col1-col)*dimW));
            if(((warpin_id%4) +4)<residue)
            rhs_fragment_tf32[3]=*(rhs_tile_ + (warpin_id%4) + 4 + ((col1-col)*dimW));
        }

        //MMA 
        asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
            : "r"(rhs_fragment[0]), "r"(rhs_fragment[1]), "r"(rhs_fragment[2]), "r"(rhs_fragment[3]),
             "r"(lhs_fragment[0]), "r"(lhs_fragment[1]), 
             "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));      
        }

};

/*
FP16
*/

struct mmaComputeUtils_fp16_gen{
    const half2* lhs_tile_;
    const half2* rhs_tile_;
    uint32_t* output_fragment_;
    const int lane_id_;
    const int warpin_id;
    const int warp_id;
    int dimN_index_;
    int col_;
    int row_;
    int dimW_;

    // Constructor
    __device__ __forceinline__ mmaComputeUtils_fp16_gen(
        const half2 *lhs_tile,
        const half2*rhs_tile,
        uint32_t* output_fragment,
        int lane_id):
            lhs_tile_(lhs_tile),
            rhs_tile_(rhs_tile),
            output_fragment_(output_fragment),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5){}
    
    // Compute
    __device__ __forceinline__ void TileMAC(int dimW, long col, long col1,int dimMori, long row){
        //load lhs_matrix
        at::Half lhs_fragment_half[4]={0.0,0.0,0.0,0.0};
        half2 *lhs_fragment_half2 = reinterpret_cast<half2*>(lhs_fragment_half);
        uint32_t *lhs_fragment=reinterpret_cast<uint32_t*>(lhs_fragment_half);
        if(row < dimMori){
        lhs_fragment_half2[0] = *(lhs_tile_+(warpin_id%4));
        lhs_fragment_half2[1] = *(lhs_tile_+(warpin_id%4)+4);

        // lhs_fragment_half[0]=*(lhs_tile_+((warpin_id%4)*2));
        // lhs_fragment_half[1]=*(lhs_tile_+((warpin_id%4)*2)+1);
        // lhs_fragment_half[2]=*(lhs_tile_+((warpin_id%4)*2)+8);
        // lhs_fragment_half[3]=*(lhs_tile_+((warpin_id%4)*2)+9);
        lhs_tile_=lhs_tile_+8;}

        //load rhs_matrix
        at::Half rhs_fragment_half[8]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        half2 *rhs_fragment_half2 = reinterpret_cast<half2*>(rhs_fragment_half);
        uint32_t *rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment_half);
        if(col>=0){
        // rhs_fragment_half[0]=*(rhs_tile_ + ((warpin_id%4)*2));
        // rhs_fragment_half[1]=*(rhs_tile_ + ((warpin_id%4)*2) + 1);
        // rhs_fragment_half[4]=*(rhs_tile_ + ((warpin_id%4)*2) + 8);
        // rhs_fragment_half[5]=*(rhs_tile_ + ((warpin_id%4)*2) + 9);
        rhs_fragment_half2[0] = *(rhs_tile_+(warpin_id%4));
        rhs_fragment_half2[2] = *(rhs_tile_+(warpin_id%4)+4);
        }
        if((col1)>=0){
        int temp = ((col1-col)*dimW)/2;
        rhs_fragment_half2[1] = *(rhs_tile_+(warpin_id%4) + temp);
        rhs_fragment_half2[3] = *(rhs_tile_+(warpin_id%4)+4 + temp);
        // rhs_fragment_half[2]=*(rhs_tile_ + ((warpin_id%4)*2) + ((col1-col)*dimW));
        // rhs_fragment_half[3]=*(rhs_tile_ + ((warpin_id%4)*2) + 1 + ((col1-col)*dimW));
        // rhs_fragment_half[6]=*(rhs_tile_ + ((warpin_id%4)*2) + 8 + ((col1-col)*dimW));
        // rhs_fragment_half[7]=*(rhs_tile_ + ((warpin_id%4)*2) + 9 + ((col1-col)*dimW));
        }
        //更新rhs_tile
        rhs_tile_=rhs_tile_+8;
            // if(warpin_id==4&warp_id==1)
            // {
            //     half *p=reinterpret_cast<half *>(rhs_fragment);
            //     for(int i=0;i<8;i++)
            //     printf("thread_id:%d, value is:%f\n", warpin_id,__half2float(*(p+i)));
            // }
     
        //MMA 
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \t"
            "{%0,%1}, \t"
            "{%2,%3,%4,%5}, \t"
            "{%6,%7}, \t"
            "{%0,%1}; ":
            "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
            "r"(rhs_fragment[0]),  "r"(rhs_fragment[1]),  "r"(rhs_fragment[2]),  "r"(rhs_fragment[3]),
            "r"(lhs_fragment[0]), "r"(lhs_fragment[1])
        ); 

    }
    
};

//仅用于CUDA计算
struct cudaComputeUtils_tf32_trans{
    const float * c_rhs_matrix_;
    const int *c_col_indices_;
    const float *c_values_;
    float * output_matrix_;
    int nOri_;
    int warpin_id_;
    // int warp_id_;
    // const int *c_row_offsets_;
    // Constructor
    __device__ __forceinline__ cudaComputeUtils_tf32_trans(
        const float * rhs_matrix,
        const int *c_col_indices,
        const float *c_values,
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


        // float res = 0.0;
        // int c_col_offset = dimN_index + warpin_id_;
        // if(c_col_offset<nOri_)
        // { 
        //     #pragma unroll
        //     for(int i=0; i<nonzeros; i++)
        //     {
        //         int col = *(c_col_indices_ + i);
        //         float b = __ldg(c_rhs_matrix_ + col*nOri_ + c_col_offset);  
        //         res +=  *(c_values_ + i) * b;
        //         // res +=  *(c_values_ + i) * b;
        //     }
        //     if(atomic==0)
        //     *(output_matrix_ + row*nOri_ + c_col_offset) = res;
        //     else  atomicAdd((output_matrix_ + row*nOri_ + c_col_offset) , res);
        //     // *(output_matrix_ + row*nOri_ + c_col_offset) = res;
        // }   

        float res[2] = {0.0, 0.0};
        int c_col_offset = dimN_index + warpin_id_;

        #pragma unroll
        for(int i=0; i<nonzeros; i++)
        {
            float a = *(c_values_ + i);
            int col = *(c_col_indices_ + i);
            for(int j=0; j<2; j++)
            {
                if((c_col_offset + j*32)<nOri_){
                float b = __ldg(c_rhs_matrix_ + col*nOri_ + c_col_offset + j*32);  
                res[j]+= a * b;
                }
            }

        }
        for(int i=0; i<2; i++)
        {
            if((c_col_offset + i*32)<nOri_){
                if(atomic==0)
                *(output_matrix_ + row*nOri_ + c_col_offset + i*32) += res[i];
                else  atomicAdd((output_matrix_ + row*nOri_ + c_col_offset + i*32) , res[i]);
                // if(threadIdx.x==0)
                // printf("%f\n",  *(output_matrix_ + row*nOri_ + c_col_offset + i*32));
            }  
        }
    }
};


struct cudaComputeUtils_fp16_trans{
    const half * c_rhs_matrix_;
    const int *c_col_indices_;
    const half *c_values_;
    float * output_matrix_;
    int nOri_;
    int warpin_id_;
    // int warp_id_;
    // const int *c_row_offsets_;
    // Constructor
    __device__ __forceinline__ cudaComputeUtils_fp16_trans(
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


        // float res = 0.0;
        // int c_col_offset = dimN_index + warpin_id_;
        // if(c_col_offset<nOri_)
        // { 
        //     #pragma unroll
        //     for(int i=0; i<nonzeros; i++)
        //     {
        //         int col = *(c_col_indices_ + i);
        //         float b = __ldg(c_rhs_matrix_ + col*nOri_ + c_col_offset);  
        //         res +=  *(c_values_ + i) * b;
        //         // res +=  *(c_values_ + i) * b;
        //     }
        //     if(atomic==0)
        //     *(output_matrix_ + row*nOri_ + c_col_offset) = res;
        //     else  atomicAdd((output_matrix_ + row*nOri_ + c_col_offset) , res);
        //     // *(output_matrix_ + row*nOri_ + c_col_offset) = res;
        // }   

        at::Half res_[2] = {0.0, 0.0};
        half * res = reinterpret_cast<half *>(res_);
        int c_col_offset = dimN_index + warpin_id_;

        #pragma unroll
        for(int i=0; i<nonzeros; i++)
        {
            half a = *(c_values_ + i);
            int col = *(c_col_indices_ + i);
            for(int j=0; j<2; j++)
            {
                if((c_col_offset + j*32)<nOri_){
                half b = __ldg(c_rhs_matrix_ + col*nOri_ + c_col_offset + j*32);  
                res[j] = __hadd(res[j], __hmul(a , b));
                }
            }

        }
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
