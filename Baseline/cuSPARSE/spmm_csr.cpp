#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

torch::Tensor cuSPARSE_spmm_csr(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM,
    const long dimN,
    const long nnz);


torch::Tensor cuSPARSE_spmm_csr_(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM,
    const long dimN,
    const long nnz){
    
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(rhs_matrix);
    return cuSPARSE_spmm_csr(row_offsets, col_indices, values, rhs_matrix, dimM, dimN, nnz);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuSPARSE_SPMM_CSR", &cuSPARSE_spmm_csr_, "cuSPARSE_SPMM_CSR");
}