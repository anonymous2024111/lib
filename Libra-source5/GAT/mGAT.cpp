#include <torch/extension.h>
#include <cuda_fp16.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

inline
cudaError_t checkCuda(cudaError_t result){
    if (result != cudaSuccess){
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}






// // tcu cuda
void spmm_forward_tf32_tcu_cuda_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * d_t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_column, 

    float * lhs_matrix,
    float * rhs_matrix,
    float * output_matrix_t,
    float * output_matrix_c,

    const int parts_t,
    const int maxPart,
    const int nOri,
    const int mOri,
    int parts
    );

std::vector<torch::Tensor> spmm_forward_tf32_tcu_cuda(
    torch::Tensor t_row_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_column, 
    torch::Tensor t_window_row,
    torch::Tensor t_binary, 

    torch::Tensor c_row_offsets,
    torch::Tensor c_row, 
    torch::Tensor c_column, 

    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,

    const int nnz,
    const int maxPart,
    const int parts_t,
    const int nOri,
    const int mOri,
    const int kOri,

    const int parts)
{
    auto output_matrix = torch::zeros({c_column.size(0) + nnz }, torch::kCUDA).to(torch::kFloat32);
    CHECK_CUDA(t_row_offset);
    CHECK_CUDA(t_blockNew_offset);
    CHECK_CUDA(t_column);
    CHECK_CUDA(t_window_row);
    CHECK_CUDA(t_binary);

    CHECK_CUDA(c_row_offsets);
    CHECK_CUDA(c_row);
    CHECK_CUDA(c_column);
    CHECK_CUDA(lhs_matrix);
    CHECK_CUDA(rhs_matrix);
    // std::cout << c_column.size(0)+nnz <<std::endl;

    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    long * t_binary_ = t_binary.data<long>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_column_ = c_column.data<int>();

    float * lhs_matrix_ = lhs_matrix.data<float>(); 
    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

   spmm_forward_tf32_tcu_cuda_kernel(
      t_row_offset_,
      t_blockNew_offset_,
      t_column_,
      t_window_row_,
      t_binary_,
      
      c_row_offsets_, 
      c_row_, 
      c_column_,

      lhs_matrix_,
      rhs_matrix_,
      output_matrix_,
      output_matrix_ + nnz,

      parts_t,
      maxPart,
      nOri,
      mOri,
      parts); 

    return {output_matrix,};
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
    half * lhs_matrix,
    half * rhs_matrix,
    half * output_matrix_t,
    half * output_matrix_c,

    const int parts_t,
    const int maxPart,
    const int nOri,
    const int mOri,
    int parts
    );
std::vector<torch::Tensor> spmm_forward_fp16_tcu_cuda(
    torch::Tensor t_row_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_column, 
    torch::Tensor t_window_row,
    torch::Tensor t_binary, 

    torch::Tensor c_row_offsets,
    torch::Tensor c_row, 
    torch::Tensor c_column, 

    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,

    const int nnz,
    const int maxPart,
    const int parts_t,
    const int nOri,
    const int mOri,
    const int kOri,

    const int parts)
{
    auto output_matrix = torch::zeros({c_column.size(0) + nnz }, torch::kCUDA).to(torch::kFloat16);
    CHECK_CUDA(t_row_offset);
    CHECK_CUDA(t_blockNew_offset);
    CHECK_CUDA(t_column);
    CHECK_CUDA(t_window_row);
    CHECK_CUDA(t_binary);

    CHECK_CUDA(c_row_offsets);
    CHECK_CUDA(c_row);
    CHECK_CUDA(c_column);

    CHECK_CUDA(lhs_matrix);
    CHECK_CUDA(rhs_matrix);
    // std::cout << c_column.size(0)+nnz <<std::endl;

    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    long * t_binary_ = t_binary.data<long>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_column_ = c_column.data<int>();


    half * lhs_matrix_ = reinterpret_cast<half *>(lhs_matrix.data<at::Half>()); 
    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>());


  spmm_forward_fp16_tcu_cuda_kernel(
      t_row_offset_,
      t_blockNew_offset_,
      t_column_,
      t_window_row_,
      t_binary_,
      
      c_row_offsets_, 
      c_row_, 
      c_column_,

      lhs_matrix_,
      rhs_matrix_,
      output_matrix_,
      output_matrix_ + nnz,

      parts_t,
      maxPart,
      nOri,
      mOri,
      parts);

    return {output_matrix,};
}


// // tcu cuda
void spmm_forward_tf32_tcu_cuda_kernel_spmm(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * d_t_binary,
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
    const int mOri
    );
std::vector<torch::Tensor> spmm_forward_tf32_tcu_cuda_spmm(
    torch::Tensor t_row_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_column, 
    torch::Tensor t_window_row,
    torch::Tensor t_binary, 
    torch::Tensor t_atomic,
    torch::Tensor t_value, 

    torch::Tensor c_row_offsets,
    torch::Tensor c_row, 
    torch::Tensor c_column, 
    torch::Tensor c_atomic,    
    torch::Tensor c_value, 

    torch::Tensor rhs_matrix,


    const int parts_t,
    const int parts_c,
    const int nOri,
    const int mOri,
    const int kOri
    )
{
    auto output_matrix = torch::zeros({mOri,nOri}, torch::kCUDA).to(torch::kFloat32);
    CHECK_CUDA(t_row_offset);
    CHECK_CUDA(t_blockNew_offset);
    CHECK_CUDA(t_column);
    CHECK_CUDA(t_window_row);
    CHECK_CUDA(t_binary);
    CHECK_CUDA(t_atomic);
    CHECK_CUDA(t_value);

    CHECK_CUDA(c_row_offsets);
    CHECK_CUDA(c_row);
    CHECK_CUDA(c_column);
    CHECK_CUDA(c_atomic);
    CHECK_CUDA(c_value);

    // std::cout << c_column.size(0)+nnz <<std::endl;

    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    long * t_binary_ = t_binary.data<long>();
    int * t_atomic_ = t_atomic.data<int>();
    float * t_value_ = t_value.data<float>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_column_ = c_column.data<int>();
    int * c_atomic_ = c_atomic.data<int>();
    float * c_value_ = c_value.data<float>(); 


    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

   spmm_forward_tf32_tcu_cuda_kernel_spmm(
      t_row_offset_,
      t_blockNew_offset_,
      t_column_,
      t_window_row_,
      t_binary_,
      t_atomic_,
      t_value_,

      c_row_offsets_, 
      c_row_, 
      c_column_,
      c_atomic_,
      c_value_,

      rhs_matrix_,
      output_matrix_,

      parts_t,
      parts_c,
      nOri,
      mOri); 

    return {output_matrix,};
}


void spmm_forward_fp16_tcu_cuda_kernel_spmm(
    int * t_row_offset,
    int * t_blockNew_offset,
    int * t_column, 
    int* t_window_row,
    long * d_t_binary,
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
    const int mOri
    );

std::vector<torch::Tensor> spmm_forward_fp16_tcu_cuda_spmm(
    torch::Tensor t_row_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_column, 
    torch::Tensor t_window_row,
    torch::Tensor t_binary, 
    torch::Tensor t_atomic,
    torch::Tensor t_value, 

    torch::Tensor c_row_offsets,
    torch::Tensor c_row, 
    torch::Tensor c_column, 
    torch::Tensor c_atomic,    
    torch::Tensor c_value, 

    torch::Tensor rhs_matrix,

    const int parts_t,
    const int parts_c,
    const int nOri,
    const int mOri,
    const int kOri)
{
    auto output_matrix = torch::zeros({mOri,nOri}, torch::kCUDA).to(torch::kFloat32);
    CHECK_CUDA(t_row_offset);
    CHECK_CUDA(t_blockNew_offset);
    CHECK_CUDA(t_column);
    CHECK_CUDA(t_window_row);
    CHECK_CUDA(t_binary);
    CHECK_CUDA(t_atomic);
    CHECK_CUDA(t_value);

    CHECK_CUDA(c_row_offsets);
    CHECK_CUDA(c_row);
    CHECK_CUDA(c_column);
    CHECK_CUDA(c_atomic);
    CHECK_CUDA(c_value);
    // std::cout << c_column.size(0)+nnz <<std::endl;

    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    long * t_binary_ = t_binary.data<long>();
    half * t_value_ = reinterpret_cast<half *>(t_value.data<at::Half>());
    int * t_atomic_ = t_atomic.data<int>();

    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_column_ = c_column.data<int>();
    half * c_value_ = reinterpret_cast<half *>(c_value.data<at::Half>());
    int * c_atomic_ = c_atomic.data<int>();

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    float * output_matrix_ = output_matrix.data<float>();


  spmm_forward_fp16_tcu_cuda_kernel_spmm(
      t_row_offset_,
      t_blockNew_offset_,
      t_column_,
      t_window_row_,
      t_binary_,
      t_atomic_,
      t_value_,

      c_row_offsets_, 
      c_row_, 
      c_column_,
      c_atomic_,
      c_value_,

      rhs_matrix_,
      output_matrix_,


      parts_t,
      parts_c,
      nOri,
      mOri); 

    return {output_matrix,};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    // cuda + tcu 
    m.def("forward_tf32", &spmm_forward_tf32_tcu_cuda, "one kernel");
    m.def("forward_fp16", &spmm_forward_fp16_tcu_cuda, "one kernel");

    m.def("forward_tf32_spmm", &spmm_forward_tf32_tcu_cuda_spmm, "one kernel");
    m.def("forward_fp16_spmm", &spmm_forward_fp16_tcu_cuda_spmm, "one kernel");

  }