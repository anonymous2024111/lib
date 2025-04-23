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
    int epoches);


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
    const int kOri,
    int epoches)
{
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);
    
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

    // Device
    int *d_t_row_offset, *d_t_blockNew_offset, *d_t_column;
    int *d_t_window_row, *d_t_atomic;
    long *d_t_binary;
    half *d_t_value; 
    int *d_c_row_offsets, *d_c_row, *d_c_atomic, *d_c_column;
    int *d_c_row_offsets_short, *d_c_row_short, *d_c_atomic_short, *d_c_column_short;
    half *d_c_value, *d_c_value_short; 
	half *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_t_row_offset, (t_row_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_blockNew_offset, (t_blockNew_offset.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_binary, (t_binary.size(0)) * sizeof(long)));

    checkCuda(cudaMalloc(&d_c_row_offsets, (c_row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row, (c_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic, (c_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column, (c_column.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value, (c_value.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_c_row_offsets_short, (c_row_offsets_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_row_short, (c_row_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_atomic_short, (c_atomic_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_column_short, (c_column_short.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_c_value_short, (c_value_short.size(0)) * sizeof(half)));

    checkCuda(cudaMalloc(&d_rhs_matrix, (kOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_t_row_offset, t_row_offset_ , (t_row_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (t_blockNew_offset.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_binary, t_binary_, (t_binary.size(0)) * sizeof(long), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets, c_row_offsets_ , (c_row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row, c_row_, (c_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic, c_atomic_, (c_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column, c_column_, (c_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value, c_value_, (c_value.size(0)) * sizeof(half), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_c_row_offsets_short, c_row_offsets_short_ , (c_row_offsets_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_row_short, c_row_short_, (c_row_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_atomic_short, c_atomic_short_, (c_atomic_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_column_short, c_column_short_, (c_column_short.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_value_short, c_value_short_, (c_value_short.size(0)) * sizeof(half), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(kOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));
    float spmm_ms_avg = 0.0;
    
      spmm_ms_avg =  spmm_forward_fp16_tcu_cuda_kernel(
      d_t_row_offset,
      d_t_blockNew_offset,
      d_t_value, 
      d_t_column,
      d_t_window_row,
      d_t_atomic,
      d_t_binary,
      
      d_c_row_offsets, 
      d_c_row, 
      d_c_atomic,
      d_c_column,
      d_c_value,

      d_c_row_offsets_short, 
      d_c_row_short, 
      d_c_atomic_short,
      d_c_column_short,
      d_c_value_short,

      d_rhs_matrix,
      d_output_matrix,

      parts_t,
      parts_c,
      partsize_c,
      parts_c_short,
      dimN,
      mOri,
      epoches);
      
    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_t_row_offset);
    cudaFree(d_t_blockNew_offset);
    cudaFree(d_t_value);
    cudaFree(d_t_column),
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);
    cudaFree(d_t_binary);

    cudaFree(d_c_row_offsets);
    cudaFree(d_c_row);
    cudaFree(d_c_atomic);
    cudaFree(d_c_column);
    cudaFree(d_c_value);

    cudaFree(d_c_row_offsets_short);
    cudaFree(d_c_row_short);
    cudaFree(d_c_atomic_short);
    cudaFree(d_c_column_short);
    cudaFree(d_c_value_short);

    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaDeviceSynchronize(); 
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("forward_fp16_tcu_cuda", &spmm_forward_fp16_tcu_cuda, "one kernel");

  }