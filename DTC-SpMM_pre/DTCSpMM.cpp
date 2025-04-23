#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#define min(x, y) (((x) < (y))? (x) : (y))

void fill_edgeToRow_cuda(int *edgeToRow, int *nodePointer, int num_nodes);

void fill_window_cuda(int *edgeToColumn, int *blockPartition, int *nodePointer,
                      int *edgeList, int blockSize_h, int blockSize_w,
                      int num_nodes);

void fill_segment_cuda(int *nodePointer, int *seg_out, int blockSize_h,
                       int blockSize_w, int num_nodes);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, torch::Tensor>
seg_sort_dequ(int *seg, int *edgeLists, int *nodepointer, int *edgetocol,
              int *edgetorow, int *blockPartition, int *blocknum,
              int *row_window_offset, int blockSize_h, int blockSize_w,
              int num_nodes, int num_edges, int rowwindow_num);

std::tuple<torch::Tensor, int, torch::Tensor>
seg_sort_dequ_fs(int *seg, int *edgeLists, int *nodepointer, int *edgetocol,
              int *edgetorow, int *blockPartition, int *blocknum, int * vectornum,
              int *row_window_offset, int blockSize_h, int blockSize_w,
              int num_nodes, int num_edges, int rowwindow_num);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
torch::Tensor, torch::Tensor, torch::Tensor,
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
seg_sort_dequ_libra_spmm(int *seg, int *edgeLists, int *nodepointer, int *edgetocol, int *edgetorow,
              int *blockpartition, int * groupPartition, int * valuePartition, int *vectorPartition, int *sizePartition,
              int *block_offset, int * group_offset, int * value_offset, 
              int *cudaLongPartition, int *cudaShortPartition, int *cudaLong_offset, int *cudaShort_offset,
              int * cudaLongGroupPartition, int * cudaShortGroupPartition, int *cudaLongGroup_offset, int *cudaShortGroup_offset,
              int *tc_group, int * vector_num, int * vector_nnz,
              int *cuda_long_group, int *cuda_long, int *cuda_short_group, int *cuda_short,
              int blockSize_h, int blockSize_w,
              int num_nodes, int num_edges, int rowwindow_num, int threshold, int Short_len, int t_s, int c_s);
              
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
// torch::Tensor, torch::Tensor, torch::Tensor>
// seg_sort_dequ_libra_sddmm(int *seg, int *edgeLists, int *nodepointer, int *edgetocol, int *edgetorow,
//               int *blockpartition, int * groupPartition, int * valuePartition, int *vectorPartition, int *sizePartition,
//               int *block_offset, int * group_offset, 
//               int *cudaLongPartition, int *cudaLong_offset,
//               int * cudaLongGroupPartition, int *cudaLongGroup_offset,
//               int *tc_group, int * vector_num, int * vector_nnz,
//               int *cuda_long_group, int *cuda_long,
//               int blockSize_h, int blockSize_w,
//               int num_nodes, int num_edges, int rowwindow_num, int threshold, int t_s, int c_s);


std::vector<torch::Tensor>
spmm_forward_cuda(torch::Tensor nodePointer, torch::Tensor edgeList,
                  torch::Tensor blockPartition, torch::Tensor edgeToColumn,
                  torch::Tensor edgeToRow, int num_nodes, int num_edges,
                  int embedding_dim, torch::Tensor input);

std::vector<torch::Tensor> spmm_forward_improved_ptx_uint8_cuda(
    torch::Tensor Rowwindow_offset, torch::Tensor TCblocktile_id,
    torch::Tensor TCblock_offset, torch::Tensor sparse_AToX_idx, int num_nodes,
    int num_edges, int embedding_dim, torch::Tensor input, std::string exeplan);

std::vector<torch::Tensor> spmm_forward_cuda_origin_clock(
    torch::Tensor Rowwindow_offset, torch::Tensor TCblocktile_id,
    torch::Tensor TCblock_offset, torch::Tensor sparse_AToX_idx, int num_nodes,
    int num_edges, int embedding_dim, torch::Tensor input);

std::vector<torch::Tensor> spmm_forward_improved_ptx_uint8_cuda_dtc_for_gcn(
    torch::Tensor Rowwindow_offset, torch::Tensor TCblocktile_id,
    torch::Tensor TCblock_offset, torch::Tensor sparse_AToX_idx, int num_nodes,
    int num_edges, int embedding_dim, torch::Tensor input);

std::vector<torch::Tensor>
spmm_forward_improved_ptx_uint8_prefetch_balance_sort_cuda(
    torch::Tensor Rowwindow_offset, torch::Tensor TCblocktile_id,
    torch::Tensor TCblock_offset, torch::Tensor sparse_AToX_idx,
    torch::Tensor sort, int num_nodes, int num_edges, int embedding_dim,
    torch::Tensor input);

std::vector<torch::Tensor> spmm_balance_forward_cuda_ptx_unit8_prefetch(
    torch::Tensor TCblock_rowid, torch::Tensor TCblocktile_id,
    torch::Tensor TCblock_offset, torch::Tensor sparse_AToX_idx, int tc_count,
    int num_nodes, int num_edges, int embedding_dim, torch::Tensor input,
    std::string exeplan);

std::vector<torch::Tensor>
spmm_balance_clock(torch::Tensor TCblock_rowid, torch::Tensor TCblocktile_id,
                   torch::Tensor TCblock_offset, torch::Tensor sparse_AToX_idx,
                   int tc_count, int num_nodes, int num_edges,
                   int embedding_dim, torch::Tensor input);

std::vector<torch::Tensor> spmm_forward_cusparse(torch::Tensor rowoffset,
                                                 torch::Tensor colind,
                                                 torch::Tensor input,
                                                 int num_nodes, int num_edges,
                                                 int embedding_dim, int algid);

std::vector<torch::Tensor> spmm_forward_cusparse_blocked_ellpack(
    torch::Tensor ell_colind, torch::Tensor input, int num_nodes,
    int block_size, int ell_columns, int embedding_dim);

    
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float>
preprocess_gpu_libra_spmm(torch::Tensor edgeList_tensor, torch::Tensor nodePointer_tensor,
                int num_nodes, int blockSize_h, int blockSize_w,
                int threshold, int Short_len, int t_s, int c_s, int num_windows, int num_nnz) {
    // input tensors.
    auto options_gpu =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    auto edgeList = edgeList_tensor.data<int>();
    //声明需要的数据结构
    //TCU:
    auto blockPartition_tensor =
        torch::zeros({num_windows}, options_gpu);
    auto groupPartition_tensor =
        torch::zeros({num_windows}, options_gpu);
    auto valuePartition_tensor =
        torch::zeros({num_windows}, options_gpu);
    auto vectorPartition_tensor =
        torch::zeros({num_windows}, options_gpu);
    auto sizePartition_tensor =
        torch::zeros({num_windows}, options_gpu);
    auto cudaLongPartition_tensor =
        torch::zeros({(num_windows*blockSize_h)}, options_gpu);
    auto cudaShortPartition_tensor =
        torch::zeros({(num_windows*blockSize_h)}, options_gpu);
    auto cudaLongGroupPartition_tensor =
        torch::zeros({(num_windows)}, options_gpu);
    auto cudaShortGroupPartition_tensor =
        torch::zeros({(num_windows)}, options_gpu);
        
    auto block_offset_tensor =
        torch::zeros({num_windows+1}, options_gpu);
    auto group_offset_tensor =
        torch::zeros({num_windows+1}, options_gpu);
    auto value_offset_tensor =
        torch::zeros({num_windows+1}, options_gpu);
    auto edgeToColumn_tensor =
        torch::zeros({num_nnz}, options_gpu);
    auto edgeToRow_tensor =
        torch::zeros({num_nnz}, options_gpu);
    auto cudaLong_offset_tensor =
        torch::zeros({(num_windows*blockSize_h+1)}, options_gpu);
    auto cudaShort_offset_tensor =
        torch::zeros({(num_windows*blockSize_h+1)}, options_gpu);
    auto cudaLongGroup_offset_tensor =
        torch::zeros({(num_windows+1)}, options_gpu);
    auto cudaShortGroup_offset_tensor =
        torch::zeros({(num_windows+1)}, options_gpu);


    //转换为c的数据结构
    auto blockPartition = blockPartition_tensor.data<int>();
    auto groupPartition = groupPartition_tensor.data<int>();
    auto valuePartition = valuePartition_tensor.data<int>();
    auto vectorPartition = vectorPartition_tensor.data<int>();
    auto sizePartition = sizePartition_tensor.data<int>();
    auto cudaLongPartition = cudaLongPartition_tensor.data<int>();
    auto cudaShortPartition = cudaShortPartition_tensor.data<int>();
    auto cudaLongGroupPartition = cudaLongGroupPartition_tensor.data<int>();
    auto cudaShortGroupPartition = cudaShortGroupPartition_tensor.data<int>();

    auto block_offset = block_offset_tensor.data<int>();
    auto group_offset = group_offset_tensor.data<int>();
    auto value_offset = value_offset_tensor.data<int>();
    auto edgeToColumn = edgeToColumn_tensor.data<int>();
    auto cudaLong_offset = cudaLong_offset_tensor.data<int>();
    auto cudaShort_offset = cudaShort_offset_tensor.data<int>();
    auto cudaLongGroup_offset = cudaLongGroup_offset_tensor.data<int>();
    auto cudaShortGroup_offset = cudaShortGroup_offset_tensor.data<int>();

    //#NNZ
    auto seg_out_tensor = torch::zeros({edgeList_tensor.size(0)}, options_gpu);
    //TCU:
    auto tcgroup = torch::zeros({1}, options_gpu); //TC group个数
    auto vectornum = torch::zeros({1}, options_gpu); //vector个数
    auto vectornnz = torch::zeros({1}, options_gpu); //非零元个数
    //CUDA:
    auto cudalonggroup = torch::zeros({1}, options_gpu); // group个数
    auto cudalong = torch::zeros({1}, options_gpu); // 非零元个数
    auto cudashortgroup = torch::zeros({1}, options_gpu);
    auto cudashort = torch::zeros({1}, options_gpu);

    auto tc_group = tcgroup.data<int>();
    auto vector_num = vectornum.data<int>();
    auto vector_nnz = vectornnz.data<int>();


    auto cuda_long_group = cudalonggroup.data<int>();
    auto cuda_long = cudalong.data<int>();
    auto cuda_short_group = cudashortgroup.data<int>();
    auto cuda_short = cudashort.data<int>();
    

    auto edgeToRow = edgeToRow_tensor.data<int>();
    auto nodePointer = nodePointer_tensor.data<int>();
    auto seg_out = seg_out_tensor.data<int>();
    auto start = std::chrono::high_resolution_clock::now();
    fill_edgeToRow_cuda(edgeToRow, nodePointer, num_nodes);
    int block_counter = 0;


    //Step1. 求出seg_out，即每个非零元自己所属于的window
    fill_segment_cuda(nodePointer, seg_out, blockSize_h, blockSize_w, num_nodes);
    //Step2. 找到每个元素在window中的new column 
    auto tuple_tensor_blockcnt = seg_sort_dequ_libra_spmm(
        seg_out, edgeList, nodePointer, edgeToColumn, edgeToRow, 
        blockPartition, groupPartition, valuePartition, vectorPartition, sizePartition,
        block_offset, group_offset, value_offset, 
        cudaLongPartition, cudaShortPartition, cudaLong_offset, cudaShort_offset,
        cudaLongGroupPartition, cudaShortGroupPartition, cudaLongGroup_offset, cudaShortGroup_offset,
        tc_group, vector_num, vector_nnz,
        cuda_long_group, cuda_long, cuda_short_group, cuda_short, 
        blockSize_h, blockSize_w, num_nodes,
        edgeList_tensor.size(0), blockPartition_tensor.size(0),threshold, Short_len, t_s, c_s);
        
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "\t GPU Preprocess time: " << elapsed_seconds.count()
                << " seconds\n";
                
    auto WindowOffset_tensor = std::get<0>(tuple_tensor_blockcnt);
    auto Curwindow_tensor = std::get<1>(tuple_tensor_blockcnt);
    auto t_Atomic_tensor = std::get<2>(tuple_tensor_blockcnt);
    auto ColumnIndice_tensor = std::get<3>(tuple_tensor_blockcnt);
    auto BlockOffset_tensor = std::get<4>(tuple_tensor_blockcnt);
    auto Binary_tensor = std::get<5>(tuple_tensor_blockcnt);
    auto t_Value_tensor = std::get<6>(tuple_tensor_blockcnt);

    auto cuda_long_group_tensor = std::get<7>(tuple_tensor_blockcnt);
    auto cuda_long_row_tensor = std::get<8>(tuple_tensor_blockcnt);
    auto cuda_long_atomic_tensor = std::get<9>(tuple_tensor_blockcnt);
    auto cuda_long_column_tensor = std::get<10>(tuple_tensor_blockcnt);
    auto cuda_long_value_tensor = std::get<11>(tuple_tensor_blockcnt);

    auto cuda_short_group_tensor = std::get<12>(tuple_tensor_blockcnt);
    auto cuda_short_row_tensor = std::get<13>(tuple_tensor_blockcnt);
    auto cuda_short_atomic_tensor = std::get<14>(tuple_tensor_blockcnt);
    auto cuda_short_column_tensor = std::get<15>(tuple_tensor_blockcnt);
    auto cuda_short_value_tensor = std::get<16>(tuple_tensor_blockcnt);
    // 1. Window内有多少block
    // 2. 当前block对应的window --- 负载均衡
    // 3. 每个block内的偏移
    // 4. 每个block中非零元个数
    // 5. block列索引
    float elapsed_time = static_cast<float>(elapsed_seconds.count());
    return std::make_tuple(WindowOffset_tensor, Curwindow_tensor, t_Atomic_tensor, ColumnIndice_tensor, 
                           BlockOffset_tensor, Binary_tensor, t_Value_tensor, 
                           cuda_long_group_tensor, cuda_long_row_tensor, cuda_long_atomic_tensor, cuda_long_column_tensor, cuda_long_value_tensor,
                           cuda_short_group_tensor, cuda_short_row_tensor, cuda_short_atomic_tensor, cuda_short_column_tensor, cuda_short_value_tensor,
                           elapsed_time);
}



// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
// torch::Tensor, torch::Tensor, torch::Tensor, float>
// preprocess_gpu_libra_sddmm(torch::Tensor edgeList_tensor, torch::Tensor nodePointer_tensor,
//                 int num_nodes, int blockSize_h, int blockSize_w,
//                 int threshold, int t_s, int c_s, int num_windows, int num_nnz) {
//     // input tensors.
//     auto options_gpu =
//       torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

//     auto edgeList = edgeList_tensor.data<int>();
//     //声明需要的数据结构
//     //TCU:
//     auto blockPartition_tensor =
//         torch::zeros({num_windows}, options_gpu);
//     auto groupPartition_tensor =
//         torch::zeros({num_windows}, options_gpu);
//     auto valuePartition_tensor =
//         torch::zeros({num_windows}, options_gpu);
//     auto vectorPartition_tensor =
//         torch::zeros({num_windows}, options_gpu);
//     auto sizePartition_tensor =
//         torch::zeros({num_windows}, options_gpu);
//     auto cudaLongPartition_tensor =
//         torch::zeros({(num_windows*blockSize_h)}, options_gpu);
//     auto cudaLongGroupPartition_tensor =
//         torch::zeros({(num_windows)}, options_gpu);

//     auto block_offset_tensor =
//         torch::zeros({num_windows+1}, options_gpu);
//     auto group_offset_tensor =
//         torch::zeros({num_windows+1}, options_gpu);
//     auto edgeToColumn_tensor =
//         torch::zeros({num_nnz}, options_gpu);
//     auto edgeToRow_tensor =
//         torch::zeros({num_nnz}, options_gpu);
//     auto cudaLong_offset_tensor =
//         torch::zeros({(num_windows*blockSize_h+1)}, options_gpu);
//     auto cudaLongGroup_offset_tensor =
//         torch::zeros({(num_windows+1)}, options_gpu);


//     //转换为c的数据结构
//     auto blockPartition = blockPartition_tensor.data<int>();
//     auto groupPartition = groupPartition_tensor.data<int>();
//     auto valuePartition = valuePartition_tensor.data<int>();
//     auto vectorPartition = vectorPartition_tensor.data<int>();
//     auto sizePartition = sizePartition_tensor.data<int>();
//     auto cudaLongPartition = cudaLongPartition_tensor.data<int>();
//     auto cudaLongGroupPartition = cudaLongGroupPartition_tensor.data<int>();

//     auto block_offset = block_offset_tensor.data<int>();
//     auto group_offset = group_offset_tensor.data<int>();
//     auto edgeToColumn = edgeToColumn_tensor.data<int>();
//     auto cudaLong_offset = cudaLong_offset_tensor.data<int>();
//     auto cudaLongGroup_offset = cudaLongGroup_offset_tensor.data<int>();
    
//     //#NNZ
//     auto seg_out_tensor = torch::zeros({edgeList_tensor.size(0)}, options_gpu);
//     //TCU:
//     auto tcgroup = torch::zeros({1}, options_gpu); //TC group个数
//     auto vectornum = torch::zeros({1}, options_gpu); //vector个数
//     auto vectornnz = torch::zeros({1}, options_gpu); //非零元个数
//     //CUDA:
//     auto cudalonggroup = torch::zeros({1}, options_gpu); // group个数
//     auto cudalong = torch::zeros({1}, options_gpu); // 非零元个数

//     auto tc_group = tcgroup.data<int>();
//     auto vector_num = vectornum.data<int>();
//     auto vector_nnz = vectornnz.data<int>();

//     auto cuda_long_group = cudalonggroup.data<int>();
//     auto cuda_long = cudalong.data<int>();
    

//     auto edgeToRow = edgeToRow_tensor.data<int>();
//     auto nodePointer = nodePointer_tensor.data<int>();
//     auto seg_out = seg_out_tensor.data<int>();
//     auto start = std::chrono::high_resolution_clock::now();
//     fill_edgeToRow_cuda(edgeToRow, nodePointer, num_nodes);
//     int block_counter = 0;


//     //Step1. 求出seg_out，即每个非零元自己所属于的window
//     fill_segment_cuda(nodePointer, seg_out, blockSize_h, blockSize_w, num_nodes);
//     //Step2. 找到每个元素在window中的new column 
//     auto tuple_tensor_blockcnt = seg_sort_dequ_libra_sddmm(
//         seg_out, edgeList, nodePointer, edgeToColumn, edgeToRow, 
//         blockPartition, groupPartition, valuePartition, vectorPartition, sizePartition,
//         block_offset, group_offset, value_offset, 
//         cudaLongPartition, cudaLong_offset,
//         cudaLongGroupPartition, cudaLongGroup_offset,
//         tc_group, vector_num, vector_nnz,
//         cuda_long_group, cuda_long, 
//         blockSize_h, blockSize_w, num_nodes,
//         edgeList_tensor.size(0), blockPartition_tensor.size(0),threshold, t_s, c_s);
        
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed_seconds = end - start;
//     std::cout << "\t GPU Preprocess time: " << elapsed_seconds.count()
//                 << " seconds\n";
                
//     auto WindowOffset_tensor = std::get<0>(tuple_tensor_blockcnt);
//     auto Curwindow_tensor = std::get<1>(tuple_tensor_blockcnt);
//     auto ColumnIndice_tensor = std::get<2>(tuple_tensor_blockcnt);
//     auto BlockOffset_tensor = std::get<3>(tuple_tensor_blockcnt);
//     auto Binary_tensor = std::get<4>(tuple_tensor_blockcnt);

//     auto cuda_long_group_tensor = std::get<5>(tuple_tensor_blockcnt);
//     auto cuda_long_row_tensor = std::get<6>(tuple_tensor_blockcnt);
//     auto cuda_long_column_tensor = std::get<7>(tuple_tensor_blockcnt);


//     float elapsed_time = static_cast<float>(elapsed_seconds.count());
//     return std::make_tuple(WindowOffset_tensor, Curwindow_tensor, ColumnIndice_tensor, 
//                            BlockOffset_tensor, Binary_tensor, 
//                            cuda_long_group_tensor, cuda_long_row_tensor, cuda_long_column_tensor,
//                            elapsed_time*1000);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess_gpu_libra_spmm", &preprocess_gpu_libra_spmm, "Preprocess Step on (CUDA)");
//   m.def("preprocess_gpu_libra_sddmm", &preprocess_gpu_libra_sddmm, "Preprocess Step on (CUDA)");

}
