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


// tcu cuda
void spmm_forward_tf32_tcu_cuda_kernel(
    int * t_row_offset,
    int * t_blockNew_offset,
    float * t_value, 
    int * t_column, 
    int* t_window_row,
    int * t_atomic,
    int *  t_binary,

    int * c_row_offset,
    int * c_row,
    int * c_atomic,
    int * c_column, 
    float * c_value, 

    int * c_row_offset_short,
    int * c_row_short,
    int * c_atomic_short,
    int * c_column_short, 
    float * c_value_short, 

    float * rhs_matrix,
    float * output_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri);

std::vector<torch::Tensor> spmm_forward_tf32_tcu_cuda(
    torch::Tensor t_row_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_column, 
    torch::Tensor t_value, 
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,
    torch::Tensor t_binary, 

    torch::Tensor c_row_offsets,
    torch::Tensor c_row, 
    torch::Tensor c_atomic,    
    torch::Tensor c_column, 
    torch::Tensor c_value, 

    torch::Tensor c_row_offsets_short,
    torch::Tensor c_row_short, 
    torch::Tensor c_atomic_short,    
    torch::Tensor c_column_short, 
    torch::Tensor c_value_short, 

    torch::Tensor rhs_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    const int kOri)
{
    auto output_matrix = torch::zeros({mOri,dimN}, torch::kCUDA).to(torch::kFloat32);
    CHECK_CUDA(t_row_offset);
    CHECK_CUDA(t_blockNew_offset);
    CHECK_CUDA(t_column);
    CHECK_CUDA(t_value);
    CHECK_CUDA(t_window_row);
    CHECK_CUDA(t_atomic);
    CHECK_CUDA(t_binary);

    CHECK_CUDA(c_row_offsets);
    CHECK_CUDA(c_row);
    CHECK_CUDA(c_atomic);
    CHECK_CUDA(c_column);
    CHECK_CUDA(c_value);

    CHECK_CUDA(c_row_offsets_short);
    CHECK_CUDA(c_row_short);
    CHECK_CUDA(c_atomic_short);
    CHECK_CUDA(c_column_short);
    CHECK_CUDA(c_value_short);
    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    float * t_value_ = t_value.data<float>(); 
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();
    int * t_binary_ = t_binary.data<int>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_atomic_ = c_atomic.data<int>();
    int * c_column_ = c_column.data<int>();
    float * c_value_ = c_value.data<float>(); 

    int * c_row_offsets_short_ = c_row_offsets_short.data<int>();
    int * c_row_short_ = c_row_short.data<int>();
    int * c_atomic_short_ = c_atomic_short.data<int>();
    int * c_column_short_ = c_column_short.data<int>();
    float * c_value_short_ = c_value_short.data<float>(); 

    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();

   
    spmm_forward_tf32_tcu_cuda_kernel(
    t_row_offset_,
    t_blockNew_offset_,
    t_value_, 
    t_column_,
    t_window_row_,
    t_atomic_,
    t_binary_,
    
      c_row_offsets_, 
      c_row_, 
      c_atomic_,
      c_column_,
      c_value_,

      c_row_offsets_short_, 
      c_row_short_, 
      c_atomic_short_,
      c_column_short_,
      c_value_short_,

      rhs_matrix_,
      output_matrix_,

    parts_t,
    parts_c,
    partsize_c,
    parts_c_short,
    dimN,
    mOri);

    return {output_matrix};
}

// tcu cuda
void spmm_forward_fp16_tcu_cuda_kernel(
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
    const int mOri);


std::vector<torch::Tensor> spmm_forward_fp16_tcu_cuda(
    torch::Tensor t_row_offset,
    torch::Tensor t_blockNew_offset, 
    torch::Tensor t_column, 
    torch::Tensor t_value, 
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,
    torch::Tensor t_binary, 

    torch::Tensor c_row_offsets,
    torch::Tensor c_row, 
    torch::Tensor c_atomic,    
    torch::Tensor c_column, 
    torch::Tensor c_value, 

    torch::Tensor c_row_offsets_short,
    torch::Tensor c_row_short, 
    torch::Tensor c_atomic_short,    
    torch::Tensor c_column_short, 
    torch::Tensor c_value_short, 

    torch::Tensor rhs_matrix,

    const int parts_t,
    const int parts_c,
    const int partsize_c,
    const int parts_c_short,
    const int dimN,
    const int mOri,
    const int kOri)
{
    auto output_matrix = torch::zeros({mOri,dimN}, torch::kCUDA).to(torch::kFloat32);
    CHECK_CUDA(t_row_offset);
    CHECK_CUDA(t_blockNew_offset);
    CHECK_CUDA(t_column);
    CHECK_CUDA(t_value);
    CHECK_CUDA(t_window_row);
    CHECK_CUDA(t_atomic);
    CHECK_CUDA(t_binary);

    CHECK_CUDA(c_row_offsets);
    CHECK_CUDA(c_row);
    CHECK_CUDA(c_atomic);
    CHECK_CUDA(c_column);
    CHECK_CUDA(c_value);

    CHECK_CUDA(c_row_offsets_short);
    CHECK_CUDA(c_row_short);
    CHECK_CUDA(c_atomic_short);
    CHECK_CUDA(c_column_short);
    CHECK_CUDA(c_value_short);

    //把CPU端的tensor转成C++的数据结构
    int * t_row_offset_ = t_row_offset.data<int>();
    int * t_blockNew_offset_ = t_blockNew_offset.data<int>();
    half * t_value_ = reinterpret_cast<half *>(t_value.data<at::Half>()); 
    int * t_column_ = t_column.data<int>();
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();
    long * t_binary_ = t_binary.data<long>();
    
    int * c_row_offsets_ = c_row_offsets.data<int>();
    int * c_row_ = c_row.data<int>();
    int * c_atomic_ = c_atomic.data<int>();
    int * c_column_ = c_column.data<int>();
    half * c_value_ = reinterpret_cast<half *>(c_value.data<at::Half>()); 

    int * c_row_offsets_short_ = c_row_offsets_short.data<int>();
    int * c_row_short_ = c_row_short.data<int>();
    int * c_atomic_short_ = c_atomic_short.data<int>();
    int * c_column_short_ = c_column_short.data<int>();
    half * c_value_short_ = reinterpret_cast<half *>(c_value_short.data<at::Half>()); 

    half * rhs_matrix_ = reinterpret_cast<half *>(rhs_matrix.data<at::Half>()); 
    float * output_matrix_ = output_matrix.data<float>();

     spmm_forward_fp16_tcu_cuda_kernel(
      t_row_offset_,
      t_blockNew_offset_,
      t_value_, 
      t_column_,
      t_window_row_,
      t_atomic_,
      t_binary_,
      
        c_row_offsets_, 
        c_row_, 
        c_atomic_,
        c_column_,
        c_value_,

        c_row_offsets_short_, 
        c_row_short_, 
        c_atomic_short_,
        c_column_short_,
        c_value_short_,

        rhs_matrix_,
        output_matrix_,

      parts_t,
      parts_c,
      partsize_c,
      parts_c_short,
      dimN,
      mOri);
 
    return {output_matrix};
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("forward_tf32", &spmm_forward_tf32_tcu_cuda, "one kernel");
  m.def("forward_fp16", &spmm_forward_fp16_tcu_cuda, "one kernel");
  // m.def("forward_tf32", &spmm_forward_tf32_tcu, "one kernel");
    // m.def("my_test", &block_sgt_tcu_kernel, "纯tcu");
  }