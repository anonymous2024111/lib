#include "config.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <fstream>
#include <mma.h>
// #include <sputnik/spmm/cuda_spmm.h>
// #include <sputnik/sputnik.h>
#include <sstream>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <torch/extension.h>
#include <vector>
#include <iostream>
#define WPB 8
#define EXE_TIME 10
#define NUM_SM_GPU 128 // 4090
#define USE_SPUTNIK
using namespace nvcuda;

struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;
  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() { cudaEventRecord(start); }

  void Stop() { cudaEventRecord(stop); }

  float Elapsed() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

// From (https://github.com/xxcclong/GNN-Computing)
typedef uint64_t clocktype;
struct Dur {
  clocktype begin;
  clocktype end;
  int smid = -1;
  Dur(clocktype x, clocktype y, int outsm) {
    begin = x;
    end = y;
    smid = outsm;
  }
};

bool cmp(Dur x, Dur y) { return (x.end > y.end); }
static __device__ inline uint64_t GlobalTimer64(void) {
  volatile uint64_t first_reading;
  volatile uint32_t second_reading;
  uint32_t high_bits_first;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(first_reading));
  high_bits_first = first_reading >> 32;
  asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(second_reading));
  if (high_bits_first == second_reading) {
    return first_reading;
  }
  // Return the value with the updated high bits, but the low bits set to 0.
  return ((uint64_t)second_reading) << 32;
}
__device__ inline uint getSMId() {
  uint smid;
  asm("mov.u32 %0, %smid;" : "=r"(smid));
  return smid;
}

//////////////////////////////////////////////////////////////////////
/// Preprocessing
//////////////////////////////////////////////////////////////////////
__global__ void roundup_to_multiple_of_eight(int *input, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    int rounded_value = ((input[tid] + 7) / 8) * 8;
    input[tid] = rounded_value;
  }
}

__global__ void get_padding_tileid_kernel(int *ori_offset, uint8_t *ori_tileid,
                                          int *padded_offset,
                                          uint8_t *padded_tileid, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    int s = ori_offset[tid];
    int e = ori_offset[tid + 1];
    int s1 = padded_offset[tid];
    for (int i = 0; i < e - s; i++) {
      padded_tileid[s1 + i] = ori_tileid[s + i];
    }
  }
}

__global__ void fill_edgeToRow(int *edgeToRow, int *nodePointer,
                               int num_nodes) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  //属于第几个window
  int nid = tid / 32;
  int laneid = tid % 32;
  // check a valid node range.
  if (nid < num_nodes) {
#pragma unroll
    for (int eid = nodePointer[nid] + laneid; eid < nodePointer[nid + 1];
         eid += 32) {
      edgeToRow[eid] = nid;
    }
  }
}
/*Generate segment*/
__global__ void fill_segment(int *nodePointer, int *seg_out, int blockSize_h,
                             int blockSize_w, int num_nodes) {
  int tid = threadIdx.x;
  int winId = blockIdx.x; // each block one window
  //Window开始的行
  unsigned block_start = nodePointer[winId * blockSize_h];
  //Window结束的行
  unsigned block_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  //window内非零元个数
  unsigned num_window_edges = block_end - block_start;
//   if(winId==0 && threadIdx.x==0){
// 	printf("%d\n", num_window_edges);
//   }
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  for (unsigned idx = tid; idx < num_window_edges; idx += threadPerBlock) {
    seg_out[block_start + idx] = winId;
  }
}

void fill_segment_cuda(int *nodePointer, int *seg_out, int blockSize_h,
                       int blockSize_w, int num_nodes) {
  // 每个window由512个线程负责
  int block_size = 512;
  int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
  fill_segment<<<window_count, block_size>>>(nodePointer, seg_out, blockSize_h,
                                             blockSize_w, num_nodes);
  cudaDeviceSynchronize(); // 等待 kernel 完成
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

/*Generate TCblock_rowid*/
__global__ void generate_tcblock_rowid(int *rowwindow_offset,
                                       int *tcblock_rowid,
                                       int num_row_windows) {
  int tid = threadIdx.x;
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = rowwindow_offset[winId];
  unsigned block_end = rowwindow_offset[min(winId + 1, num_row_windows)];
  unsigned num_blocks = block_end - block_start;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  for (unsigned idx = tid; idx < num_blocks; idx += threadPerBlock) {
    tcblock_rowid[block_start + idx] = winId;
  }
}
void generate_tcblock_rowid_cuda(int *rowwindow_offset, int *tcblock_rowid,
                                 int num_row_windows) {
  int block_size = 512;
  int window_count = num_row_windows;
  generate_tcblock_rowid<<<window_count, block_size>>>(
      rowwindow_offset, tcblock_rowid, num_row_windows);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

/* Generate edge2column*/
__device__ __forceinline__ int binarysearch(int *arr, int size, int target) {
  int left = 0;
  int right = size - 1;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (arr[mid] == target) {
      while (mid > 0 && arr[mid - 1] == target) {
        mid--;
      }
      return mid;
    } else if (arr[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return -1;
}
__device__ __forceinline__ void inplace_deduplication(int *array, int length,
                                                      int *loc) {
  int cur = 1;
  while (cur < length) {
    if (array[cur] != array[cur - 1]) {
      (*loc)++;
      array[(*loc)] = array[cur];
    }
    cur++;
  }

  (*loc)++;
}

__device__ __forceinline__ void inplace_deduplication_libra_spmm(int *array, int *counts, int length, int *loc) {
  int count = 1; // 记录当前元素的计数
  for (int cur = 1; cur < length; cur++) {
    if (array[cur] != array[cur - 1]) {
      counts[*loc] = count; // 保存上一个元素的计数
      (*loc)++;             // 更新位置
      array[*loc] = array[cur]; // 将当前元素写入去重数组
      count = 1;            // 重置计数器
    } else {
      count++; // 若相同则增加当前元素的计数
    }
  }

  counts[*loc] = count; // 保存最后一个元素的计数
  (*loc)++;             // 更新位置，表示最终去重后元素个数
}

//去重，以及求vector_num
__device__ __forceinline__ void distribute_libra_spmm(int *array, int *counts, int length, int *loc,
			int threshold, int *vector_num, int *vector_nnz) {
	int count = 1; // 记录当前元素的计数
	for (int cur = 1; cur < length; cur++) {
		if (array[cur] != array[cur - 1]) {
			counts[*loc] = count; // 保存上一个元素的计数
  		(*loc)++;             // 更新位置
			array[*loc] = array[cur]; // 将当前元素写入去重数组
			//判断是否超过阈值
			if(count>=threshold){
				(*vector_num)++;
				(*vector_nnz)+=count;
			}
			count = 1;            // 重置计数器
		} else {
			count++; // 若相同则增加当前元素的计数
		}
	}
	counts[*loc] = count; // 保存最后一个元素的计数
	(*loc)++;             // 更新位置，表示最终去重后元素个数
  if(count>=threshold){
    (*vector_num)++;
    (*vector_nnz)+=count;
  }
}

__device__ __forceinline__ void distribute_cuda_tile_libra_spmm(
	int *counts_cur, int *edgetocol, int start_row, int num_nodes,
	int *nodePointer, int threshold, int Short_len, int c_s, int *cuda_long, int* cuda_short,
	int *cuda_long_group, int * cuda_short_group) {

	int cur = 0;
	//遍历每一行,统计每行cuda tile的元素个数
	for (int cur_row = start_row; cur_row < min(start_row+8, num_nodes); cur_row++) {
		//遍历当前行的所有元素
		for(int m=nodePointer[cur_row]; m<nodePointer[cur_row+1]; m++){
			//如果当前元素的newcol的值小于threshold,则交由CUDA tile
			int col_density = counts_cur[edgetocol[m]];
			if(col_density < threshold){
				cuda_long[cur]++;
			}
		}
		cur++;
	}

	//拆分cuda_long
	for(int i=0; i<8; i++){
    if(cuda_long[i]>0){
      //如果是短行
      if(cuda_long[i]<= Short_len)
      {
        cuda_short[i] = cuda_long[i];
        cuda_long[i] = 0;
        (*cuda_short_group)++;
      }else{
        //如果是长行, 是否需要差分
        if(cuda_long[i]<=c_s){
          //不需要拆分
          (*cuda_long_group)++;
        }else{
          //需要拆分
          (*cuda_long_group) += cuda_long[i]/c_s;
          //判断residue是否存在
          int residue = (cuda_long[i]%c_s);
          if(residue> 0)
          {
            //residue是短行
            if(residue<= Short_len)
            {				
              cuda_short[i] = residue;
              cuda_long[i] -= residue;
              (*cuda_short_group)++;
            }else{
              (*cuda_long_group)++;
            }
          }
        }
      }
	  }
  }

}

/*
//去重，以及求vector_num
__device__ __forceinline__ void distribute_libra_sddmm(int *array, int *counts, int length, int *loc,
			int threshold, int *vector_num, int *vector_nnz) {
	int count = 1; // 记录当前元素的计数
	for (int cur = 1; cur < length; cur++) {
		if (array[cur] != array[cur - 1]) {
			counts[*loc] = count; // 保存上一个元素的计数
  		(*loc)++;             // 更新位置
			array[*loc] = array[cur]; // 将当前元素写入去重数组
			//判断是否超过阈值
			if(count>=threshold){
				(*vector_num)++;
				(*vector_nnz)+=count;
			}
			count = 1;            // 重置计数器
		} else {
			count++; // 若相同则增加当前元素的计数
		}
	}
	counts[*loc] = count; // 保存最后一个元素的计数
	(*loc)++;             // 更新位置，表示最终去重后元素个数
  if(count>=threshold){
    (*vector_num)++;
    (*vector_nnz)+=count;
  }
}

__device__ __forceinline__ void distribute_cuda_tile_libra_sddmm(
	int *counts_cur, int *edgetocol, int start_row, int num_nodes,
	int *nodePointer, int threshold, int Short_len, int c_s, int *cuda_long, int* cuda_short,
	int *cuda_long_group, int * cuda_short_group) {

	int cur = 0;
	//遍历每一行,统计每行cuda tile的元素个数
	for (int cur_row = start_row; cur_row < min(start_row+8, num_nodes); cur_row++) {
		//遍历当前行的所有元素
		for(int m=nodePointer[cur_row]; m<nodePointer[cur_row+1]; m++){
			//如果当前元素的newcol的值小于threshold,则交由CUDA tile
			int col_density = counts_cur[edgetocol[m]];
			if(col_density < threshold){
				cuda_long[cur]++;
			}
		}
		cur++;
	}

	//拆分cuda_long
	for(int i=0; i<8; i++){
    if(cuda_long[i]>0){
      //如果是短行
      if(cuda_long[i]<= Short_len)
      {
        cuda_short[i] = cuda_long[i];
        cuda_long[i] = 0;
        (*cuda_short_group)++;
      }else{
        //如果是长行, 是否需要差分
        if(cuda_long[i]<=c_s){
          //不需要拆分
          (*cuda_long_group)++;
        }else{
          //需要拆分
          (*cuda_long_group) += cuda_long[i]/c_s;
          //判断residue是否存在
          int residue = (cuda_long[i]%c_s);
          if(residue> 0)
          {
            //residue是短行
            if(residue<= Short_len)
            {				
              cuda_short[i] = residue;
              cuda_long[i] -= residue;
              (*cuda_short_group)++;
            }else{
              (*cuda_long_group)++;
            }
          }
        }
      }
	  }
  }

}
*/
__global__ void generate_edgetocolumn(int *nodePointer, int *edgelist,
                                      int *edgelist_sort, int *edgetocol,
                                      int *blockpartition, int *blocknum,
                                      int blockSize_h, int blockSize_w,
                                      int num_nodes) {
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = nodePointer[winId * blockSize_h];
  unsigned block_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = block_end - block_start;
  if (num_window_edges == 0)
    return;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  int *start = edgelist_sort + block_start;
  int size = 0;
  //去重
  inplace_deduplication(start, num_window_edges, &size);
  //num是每个窗口有多少个block
  int num = (size + blockSize_w) / blockSize_w;
  atomicAdd(blocknum, num);
  blockpartition[winId] = num;
  for (unsigned idx = block_start; idx < block_end; idx += 1) {
    int index = binarysearch(start, size + 1, edgelist[idx]);
    edgetocol[idx] = index;
  }
}
void generate_edgetocolumn_cuda(int *nodePointer, int *edgelist,
                                int *edgelist_sort, int *edgetocol,
                                int *blockpartition, int *blocknum,
                                int blockSize_h, int blockSize_w,
                                int num_nodes) {
  int block_size = 1;
  int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
  //每个block负责一个window, 每个block中只有一个线程
//   int block_size1 = 128;
//   int block_count1 = (window_count + 127) / 128;
  generate_edgetocolumn<<<window_count, block_size>>>(
      nodePointer, edgelist, edgelist_sort, edgetocol, blockpartition, blocknum,
      blockSize_h, blockSize_w, num_nodes);
  // generate_edgetocolumn_v1<<< window_count, block_size >>> (nodePointer,
  // edgelist, edgelist_sort, edgetocol, blockpartition, blocknum, blockSize_h,
  // blockSize_w, num_nodes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

__global__ void generate_edgetocolumn_fs(int *nodePointer, int *edgelist,
                                      int *edgelist_sort, int *edgetocol,
                                      int *blockpartition, int *blocknum, int *vectornum,
                                      int blockSize_h, int blockSize_w,
                                      int num_nodes) {
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = nodePointer[winId * blockSize_h];
  unsigned block_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = block_end - block_start;
  if (num_window_edges == 0)
    return;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  int *start = edgelist_sort + block_start;
  int size = 0;
  //去重
  inplace_deduplication(start, num_window_edges, &size);
	if(winId==0)
	printf("%d\n", size);

  //num是每个窗口有多少个block
  int num = (size + blockSize_w) / blockSize_w;
  if((size%blockSize_w)==0) num-=1;
  atomicAdd(blocknum, num);
  atomicAdd(vectornum, size);
  //vector个数
  blockpartition[winId] = size;
  for (unsigned idx = block_start; idx < block_end; idx += 1) {
    int index = binarysearch(start, size + 1, edgelist[idx]);
    edgetocol[idx] = index;
  }
}
void generate_edgetocolumn_cuda_fs(int *nodePointer, int *edgelist,
                                int *edgelist_sort, int *edgetocol,
                                int *blockpartition, int *blocknum, int * vectornum,
                                int blockSize_h, int blockSize_w,
                                int num_nodes) {
  int block_size = 1;
  int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
  //每个block负责一个window, 每个block中只有一个线程
//   int block_size1 = 128;
//   int block_count1 = (window_count + 127) / 128;
  generate_edgetocolumn_fs<<<window_count, block_size>>>(
      nodePointer, edgelist, edgelist_sort, edgetocol, blockpartition, blocknum, vectornum,
      blockSize_h, blockSize_w, num_nodes);
  // generate_edgetocolumn_v1<<< window_count, block_size >>> (nodePointer,
  // edgelist, edgelist_sort, edgetocol, blockpartition, blocknum, blockSize_h,
  // blockSize_w, num_nodes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}


__global__ void generate_partiton_information_libra_spmm(int *nodePointer, int *edgelist,
          int *edgelist_sort, int *counts, int *edgetocol, 
          int *blockPartition, int *groupPartition, int *valuePartition, int *vectorPartition, int *sizePartition, 
          int *cudaLongPartition, int *cudaShortPartition, int * cudaLongGroupPartition, int * cudaShortGroupPartition,
          int *tcgroup, int * vectornum, int * vectornnz,
          int *cudalonggroup, int *cudalong, int *cudashortgroup, int *cudashort,
          int blockSize_h, int blockSize_w,
          int num_nodes, int threshold, int Short_len, int t_s, int c_s) {

	int winId = blockIdx.x; // each warp one window
	unsigned block_start = nodePointer[winId * blockSize_h];
	unsigned block_end =
		nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
	unsigned num_window_edges = block_end - block_start;

	if (num_window_edges == 0)
	return;
  if (num_window_edges == 1){
		atomicAdd(cudashortgroup, 1);
	  atomicAdd(cudashort, 1);
    return;
  }
	const unsigned threadPerBlock = blockDim.x * blockDim.y;
	int *start = edgelist_sort + block_start;
	int *counts_cur = counts + block_start;
	int size = 0;
	//window中vector稠密度超过threshold的数量
	int tc_group = 0;
	int vector_num = 0;
	int vector_nnz = 0;

	//Setp1. 去重，以及求vector_num
	distribute_libra_spmm(start, counts_cur, num_window_edges, &size, threshold, &vector_num, &vector_nnz);
	//将vector_num按照blockSize_w对齐
  int vector_num_ori = vector_num;
	vector_num = ((vector_num + blockSize_w - 1) / blockSize_w) * blockSize_w;
	//划分tc block
	int blocks = vector_num / blockSize_w;
	if(blocks<=t_s && blocks>0) tc_group=1;
	else{
		tc_group = (blocks +  t_s - 1 )/ t_s;
	}
  if(vector_num>0){
    atomicAdd(tcgroup, tc_group);
    atomicAdd(vectornum, vector_num);
    atomicAdd(vectornnz, vector_nnz);
  }
	//vector个数
	blockPartition[winId] = blocks;
	groupPartition[winId] = tc_group;
	valuePartition[winId] = vector_nnz;
  vectorPartition[winId] = vector_num_ori;
  sizePartition[winId] = size;
  // if(winId==50403){
  //   printf("%d, %d, %d\n", tc_group, *tcgroup, blockPartition[winId]);
  // }
	//Setp2. 查找每个元素对应的新的colum id
	for (unsigned idx = block_start; idx < block_end; idx += 1) {
		int index = binarysearch(start, size + 1, edgelist[idx]);
		edgetocol[idx] = index;
	}

	//Setp3. 拆分每行的CUDA tile
	//window中每行的CUDA long tile的元素个数
	int cuda_long_array[8] = {0}; 
	//window中每行的CUDA short tile的元素个数
	int cuda_short_array[8] = {0}; 
	int cuda_long_group=0;
	int cuda_short_group=0;
	distribute_cuda_tile_libra_spmm(counts_cur, edgetocol, (winId * blockSize_h), num_nodes, nodePointer, threshold,
	Short_len, c_s, cuda_long_array, cuda_short_array, &cuda_long_group, &cuda_short_group);
	int cuda_long_sum=0;
	int cuda_short_sum=0;
	for(int i=0;i<8;i++){
		cuda_long_sum+= cuda_long_array[i];
		cuda_short_sum+= cuda_short_array[i];
    cudaLongPartition[winId*blockSize_h + i] = cuda_long_array[i];
    cudaShortPartition[winId*blockSize_h + i] = cuda_short_array[i];
	}
  cudaLongGroupPartition[winId] = cuda_long_group;
  cudaShortGroupPartition[winId] = cuda_short_group;

	atomicAdd(cudalonggroup, cuda_long_group);
	atomicAdd(cudalong, cuda_long_sum);
	atomicAdd(cudashortgroup, cuda_short_group);
	atomicAdd(cudashort, cuda_short_sum);

    // for(int i=0; i<size; i++)
    // { 
    //       if(winId==0)
    //       {
    //         printf("%d, %d\n", start[i], counts_cur[i]);
    //       }
    // }
}
void generate_partiton_information_cuda_libra_spmm(int *nodePointer, int *edgelist,
  int *edgelist_sort, int *counts, int *edgetocol, 
  int *blockpartition, int *groupPartition, int *valuePartition, int * vectorPartition, int *sizePartition,
  int *cudaLongPartition, int *cudaShortPartition, int * cudaLongGroupPartition, int * cudaShortGroupPartition,
  int *tc_group, int * vector_num, int * vector_nnz,
  int *cuda_long_group, int *cuda_long, int *cuda_short_group, int *cuda_short,
  int blockSize_h, int blockSize_w,
  int num_nodes, int threshold, int Short_len, int t_s, int c_s) {
	
    int block_size = 1;
    int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
    //每个block负责一个window, 每个block中只有一个线程
    generate_partiton_information_libra_spmm<<<window_count, block_size>>>(
      nodePointer, edgelist, edgelist_sort, counts, edgetocol, 
      blockpartition, groupPartition, valuePartition, vectorPartition, sizePartition,
      cudaLongPartition, cudaShortPartition, cudaLongGroupPartition, cudaShortGroupPartition,
      tc_group, vector_num, vector_nnz, 
      cuda_long_group, cuda_long, cuda_short_group, cuda_short,
      blockSize_h, blockSize_w, num_nodes, threshold, Short_len,t_s,c_s);
	  cudaDeviceSynchronize(); // 等待 kernel 完成
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
}


/*
__global__ void generate_partiton_information_libra_sddmm(int *nodePointer, int *edgelist,
          int *edgelist_sort, int *counts, int *edgetocol, 
          int *blockPartition, int *groupPartition, int *valuePartition, int *vectorPartition, int *sizePartition, 
          int *cudaLongPartition, int * cudaLongGroupPartition,
          int *tcgroup, int * vectornum, int * vectornnz,
          int *cudalonggroup, int *cudalong,
          int blockSize_h, int blockSize_w,
          int num_nodes, int threshold, int t_s, int c_s) {

	int winId = blockIdx.x; // each warp one window
	unsigned block_start = nodePointer[winId * blockSize_h];
	unsigned block_end =
		nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
	unsigned num_window_edges = block_end - block_start;

	if (num_window_edges == 0)
	return;
  if (num_window_edges == 1){
		atomicAdd(cudalonggroup, 1);
	  atomicAdd(cudalong, 1);
    return;
  }
	const unsigned threadPerBlock = blockDim.x * blockDim.y;
	int *start = edgelist_sort + block_start;
	int *counts_cur = counts + block_start;
	int size = 0;
	//window中vector稠密度超过threshold的数量
	int tc_group = 0;
	int vector_num = 0;
	int vector_nnz = 0;

	//Setp1. 去重，以及求vector_num
	distribute_libra_sddmm(start, counts_cur, num_window_edges, &size, threshold, &vector_num, &vector_nnz);
	//将vector_num按照blockSize_w对齐
  int vector_num_ori = vector_num;
	vector_num = ((vector_num + blockSize_w - 1) / blockSize_w) * blockSize_w;
	//划分tc block
	int blocks = vector_num / blockSize_w;
	if(blocks<=t_s && blocks>0) tc_group=1;
	else{
		tc_group = (blocks +  t_s - 1 )/ t_s;
	}
  if(vector_num>0){
    atomicAdd(tcgroup, tc_group);
    atomicAdd(vectornum, vector_num);
    atomicAdd(vectornnz, vector_nnz);
  }
	//vector个数
	blockPartition[winId] = blocks;
	groupPartition[winId] = tc_group;
	valuePartition[winId] = vector_nnz;
  vectorPartition[winId] = vector_num_ori;
  sizePartition[winId] = size;
  // if(winId==50403){
  //   printf("%d, %d, %d\n", tc_group, *tcgroup, blockPartition[winId]);
  // }
	//Setp2. 查找每个元素对应的新的colum id
	for (unsigned idx = block_start; idx < block_end; idx += 1) {
		int index = binarysearch(start, size + 1, edgelist[idx]);
		edgetocol[idx] = index;
	}

	//Setp3. 拆分每行的CUDA tile
	//window中每行的CUDA long tile的元素个数
	int cuda_long_array[8] = {0}; 
	//window中每行的CUDA short tile的元素个数
	int cuda_short_array[8] = {0}; 
	int cuda_long_group=0;
	int cuda_short_group=0;
	distribute_cuda_tile_libra_sddmm(counts_cur, edgetocol, (winId * blockSize_h), num_nodes, nodePointer, threshold,
	Short_len, c_s, cuda_long_array, cuda_short_array, &cuda_long_group, &cuda_short_group);
	int cuda_long_sum=0;
	int cuda_short_sum=0;
	for(int i=0;i<8;i++){
		cuda_long_sum+= cuda_long_array[i];
		cuda_short_sum+= cuda_short_array[i];
    cudaLongPartition[winId*blockSize_h + i] = cuda_long_array[i];
    cudaShortPartition[winId*blockSize_h + i] = cuda_short_array[i];
	}
  cudaLongGroupPartition[winId] = cuda_long_group;
  cudaShortGroupPartition[winId] = cuda_short_group;

	atomicAdd(cudalonggroup, cuda_long_group);
	atomicAdd(cudalong, cuda_long_sum);
	atomicAdd(cudashortgroup, cuda_short_group);
	atomicAdd(cudashort, cuda_short_sum);

    // for(int i=0; i<size; i++)
    // { 
    //       if(winId==0)
    //       {
    //         printf("%d, %d\n", start[i], counts_cur[i]);
    //       }
    // }
}
void generate_partiton_information_cuda_libra_sddmm(int *nodePointer, int *edgelist,
  int *edgelist_sort, int *counts, int *edgetocol, 
  int *blockpartition, int *groupPartition, int *valuePartition, int * vectorPartition, int *sizePartition,
  int *cudaLongPartition, int * cudaLongGroupPartition,
  int *tc_group, int * vector_num, int * vector_nnz,
  int *cuda_long_group, int *cuda_long, 
  int blockSize_h, int blockSize_w,
  int num_nodes, int threshold, int t_s, int c_s) {
	
    int block_size = 1;
    int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
    //每个block负责一个window, 每个block中只有一个线程
    generate_partiton_information_libra_sddmm<<<window_count, block_size>>>(
      nodePointer, edgelist, edgelist_sort, counts, edgetocol, 
      blockpartition, groupPartition, valuePartition, vectorPartition, sizePartition,
      cudaLongPartition, cudaLongGroupPartition,
      tc_group, vector_num, vector_nnz, 
      cuda_long_group, cuda_long, 
      blockSize_h, blockSize_w, num_nodes, threshold,t_s,c_s);
	  cudaDeviceSynchronize(); // 等待 kernel 完成
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
}
*/

__device__ int find_new_col(const int* ColumnIndice_, unsigned col, int size) {
    int left = 0;
    int right = size - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (ColumnIndice_[mid] == col) {
            return mid; // 找到 col，返回其索引
        } else if (ColumnIndice_[mid] < col) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1; // 没找到，返回 -1
}

__global__ void generate_libra_spmm(int *nodePointer, int *edgelist,
          int *edgelist_sort, int *counts, int *edgetocol, int *edgetorow,
          int *blockPartition, int *groupPartition, int *valuePartition, int *vectorPartition, int *sizePartition,
          int *blockOffset, int *groupOffset, int *valueOffset,
          int *cudaLongPartition, int *cudaShortPartition, 
          int *cudaLong_offset, int *cudaShort_offset, 
          int *cudaLongGroupPartition, int *cudaShortGroupPartition, 
          int *cudaLongGroup_offset, int *cudaShortGroup_offset, 
          int *WindowOffset, int *Curwindow, int *t_Atomic, int *BlockOffset, long *Binary, int *ColumnIndice, float* t_Value,
          int *cuda_long_group, int *cuda_long_row, int *cuda_long_atomic, int * cuda_long_column, float *cuda_long_value, 
          int *cuda_short_group, int *cuda_short_row, int *cuda_short_atomic, int * cuda_short_column, float *cuda_short_value,
          int blockSize_h, int blockSize_w,
          int num_nodes, int threshold, int Short_len, int t_s, int c_s) {

	int winId = blockIdx.x; // each warp one window
	unsigned block_start = nodePointer[winId * blockSize_h];
	unsigned block_end =
		nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
	unsigned num_window_edges = block_end - block_start;
	if (num_window_edges == 0)
	return;
  if (num_window_edges == 1){
    //cuda short 
    //填补CUDA
    //当前window中的group个数
    cuda_short_group += cudaShortGroup_offset[winId];
    cuda_short_row += cudaShortGroup_offset[winId];
    cuda_short_atomic += cudaShortGroup_offset[winId];
  
    for(int j=winId*blockSize_h;j<(winId+1)*blockSize_h;j++)
    {
      int row_short_offset = cudaShort_offset[j];
      int row_short_nnz = cudaShortPartition[j];
      if(row_short_nnz>0){
        int *cuda_short_column_temp =  cuda_short_column + row_short_offset;
        float *cuda_short_value_temp =  cuda_short_value + row_short_offset;
        cuda_short_group[0] = row_short_nnz;
        cuda_short_row[0] = j;
        cuda_short_atomic[0] = 0;
        cuda_short_column_temp[0] = edgelist[nodePointer[j]];
        cuda_short_value_temp[0] = 1.0;
      }
    }
    return;
  }
	const unsigned threadPerBlock = blockDim.x * blockDim.y;
	int *start = edgelist_sort + block_start;
	int *counts_cur = counts + block_start;
	//window中vector稠密度超过threshold的数量
	int tc_group = groupPartition[winId];
	int blocks = blockPartition[winId];
	int vector_nnz = valuePartition[winId];
  int vector_num_ori = vectorPartition[winId];
  int size = sizePartition[winId];

  // 填补 WindowOffset， Curwindow
  if(tc_group>0){
    int start_group = groupOffset[winId];
    int tc_group_ori = blocks/t_s;
    auto WindowOffset_ = WindowOffset + start_group;
    auto Curwindow_ = Curwindow + start_group;
    auto t_Atomic_ = t_Atomic + start_group;

    for(int i=0; i<tc_group_ori; i++)
    {
      WindowOffset_[i] = t_s;
      Curwindow_[i] = winId;
    }
    if((blocks%t_s) > 0){
      WindowOffset_[tc_group_ori] = (blocks%t_s);
      Curwindow_[tc_group_ori] = winId;
    }
    // if(winId == 50403 || winId == 50404)
    // {
    //   printf("%d, %d\n", winId, blocks);
    // }
    // 填补 t_atomic
    if(tc_group == 1 && vector_nnz == num_window_edges){
      t_Atomic_[0] = 0;
    }else{
      for(int i=0; i<tc_group; i++)
      {
        t_Atomic_[i] = 1;
      }
    }
  }

  int start_block = blockOffset[winId];
  auto ColumnIndice_ = ColumnIndice + start_block*blockSize_w;
  //填补 ColumnIndice
  if(vector_num_ori > 0){
    int temp = 0;
    for(int i=0; i<size; i++)
    { 
        if(counts_cur[i] >= threshold){
          ColumnIndice_[temp] = start[i];
          temp++;
        }
    }
  }

  //填补 BlockOffset, t_Value, Binary
  extern __shared__ int pos_ptr[];
  //记录每个block块内非零元个数
  int *tcblock_nnz_ptr = pos_ptr + blocks + 1;
  //记录前一个block块内非零元个数
  int *tcblock_nnz_ptr_pre = pos_ptr + blocks;
  for (int i = 0; i < 2 * blocks + 1; i++) {
    pos_ptr[i] = 0;
  }
  
  for (unsigned e_index = block_start; e_index < block_end; e_index++) {
    unsigned col = edgetocol[e_index]; // new col
    //只有超过阈值的vector中的元素才可以
    if(counts_cur[col] >= threshold){
      //所在的块中的非零元个数+1
      //根据col在ColumnIndice_查找偏移
      int new_col = find_new_col(ColumnIndice_, start[col], vector_num_ori);
      tcblock_nnz_ptr[new_col / blockSize_w]++;
    }

  }
  auto BlockOffset_ = BlockOffset + start_block;
  //填补 BlockOffset
  for (int i = 0; i < blocks; i++) {
    BlockOffset_[i] = tcblock_nnz_ptr[i];
  }

  // // //前缀和
  for (int i = 0; i < blocks; i++) {
    tcblock_nnz_ptr[i] += tcblock_nnz_ptr[i - 1];
  }
  
  int start_value = valueOffset[winId];
  //开始看每个非零元在block内的偏移了
  auto Binary_ = Binary + start_block;
  auto values_ = t_Value + start_value;
  for (unsigned e_index = block_start; e_index < block_end; e_index++) {
    unsigned col = edgetocol[e_index]; // new col
    unsigned row_local = edgetorow[e_index] % blockSize_h;
    //只有超过阈值的vector中的元素才可以
    if(counts_cur[col] >= threshold){
      int new_col = find_new_col(ColumnIndice_, start[col], vector_num_ori);
      unsigned tcblock_id = new_col / blockSize_w;
      unsigned col_local = new_col % blockSize_w;
      //当前非零元素在tileid的位置 = 前面块内非零元个数 + 当前块内的偏移
      long a=1;
      Binary_[tcblock_id] |= a<<(row_local * blockSize_w + col_local);
      //填补t_Value = window内前面块的非零元个数 + 当前块内偏移
      values_[tcblock_nnz_ptr_pre[tcblock_id] + pos_ptr[tcblock_id]] = 1.0;
      pos_ptr[tcblock_id]++;
    }
  }

  //填补CUDA
  //当前window中的group个数
  cuda_long_group += cudaLongGroup_offset[winId];
  cuda_short_group += cudaShortGroup_offset[winId];
  cuda_long_row += cudaLongGroup_offset[winId];
  cuda_short_row += cudaShortGroup_offset[winId];
  cuda_long_atomic += cudaLongGroup_offset[winId];
  cuda_short_atomic += cudaShortGroup_offset[winId];
    
    //开始遍历每行
    for(int j=winId*blockSize_h;j<(winId+1)*blockSize_h;j++)
    {
      int row_long_offset = cudaLong_offset[j];
      int row_short_offset = cudaShort_offset[j];
      int row_long_nnz = cudaLongPartition[j];
      int row_short_nnz = cudaShortPartition[j];

      int *cuda_long_column_temp =  cuda_long_column + row_long_offset;
      int *cuda_short_column_temp =  cuda_short_column + row_short_offset;
      float *cuda_long_value_temp =  cuda_long_value + row_long_offset;
      float *cuda_short_value_temp =  cuda_short_value + row_short_offset;

      //填补cuda_long_group
      if(row_long_nnz>0){
        int temp = row_long_nnz;
         while (temp >= c_s) {
          cuda_long_group[0] = c_s;
          temp -= c_s;
          cuda_long_group++;

          cuda_long_row[0] = j;
          cuda_long_row++;

          cuda_long_atomic[0] = 1;
          cuda_long_atomic++;
        }
        if(temp>0)
        {      
          cuda_long_group[0] = temp;
          cuda_long_group++;

          cuda_long_row[0] = j;
          cuda_long_row++;

          if(temp == row_long_nnz && tc_group==0 && row_short_nnz==0)
          {
            cuda_long_atomic[0] = 0;
            cuda_long_atomic++;
          }else{
            cuda_long_atomic[0] = 1;
            cuda_long_atomic++;
          }

        }
      }
      //填补cuda_short_group
      if(row_short_nnz>0){
          cuda_short_group[0] = row_short_nnz;
          cuda_short_group++;

          cuda_short_row[0] = j;
          cuda_short_row++;

          if(tc_group==0 && row_long_nnz==0)
          {
            cuda_short_atomic[0] = 0;
            cuda_short_atomic++;
          }else{
            cuda_short_atomic[0] = 1;
            cuda_short_atomic++;
          }
      }

      int temp_long = 0;
      //遍历当前行的每个元素
      for(int m=nodePointer[j];m<nodePointer[j+1];m++)
        {
          //如果当前元素不满足threshold
          unsigned col = edgetocol[m];
          if(counts_cur[col] < threshold){
            if(temp_long<row_long_nnz){
              //CUDA long
              cuda_long_column_temp[0] = edgelist[m];
              cuda_long_value_temp[0] = 1.0;
              cuda_long_column_temp++;
              cuda_long_value_temp++;
              temp_long++;
            }else{
              cuda_short_column_temp[0] = edgelist[m];
              cuda_short_value_temp[0] = 1.0;
              cuda_short_column_temp++;
              cuda_short_value_temp++;
              //CUDA short
            }
          }
        }
    }
}
void generate_cuda_libra_spmm(int *nodePointer, int *edgelist, 
  int *edgelist_sort, int *counts, int *edgetocol, int *edgetorow,
  int *blockPartition, int *groupPartition, int *valuePartition, int *vectorPartition, int *sizePartition,
  int *blockOffset, int *groupOffset, int *valueOffset,
  int *cudaLongPartition, int *cudaShortPartition, 
  int *cudaLong_offset, int *cudaShort_offset, 
  int *cudaLongGroupPartition, int *cudaShortGroupPartition, 
  int *cudaLongGroup_offset, int *cudaShortGroup_offset, 
  int *WindowOffset, int *Curwindow, int *t_Atomic, int *BlockOffset, long *Binary, int *ColumnIndice, float *t_Value,
  int *cuda_long_group, int *cuda_long_row, int *cuda_long_atomic, int * cuda_long_column, float *cuda_long_value, 
  int *cuda_short_group, int *cuda_short_row, int *cuda_short_atomic, int * cuda_short_column, float *cuda_short_value,
  int blockSize_h, int blockSize_w,
  int num_nodes, int threshold, int Short_len, int t_s, int c_s, int max_blocks) {
	
    int block_size = 1;
    int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
    const int dynamic_shared_size = (2 * max_blocks + 1) * sizeof(int);
    std::cout << "dynamic_shared_size: " << dynamic_shared_size << std::endl;
    if (dynamic_shared_size > 98304) {
      int maxbytes = 131072; // 96 KB
      cudaFuncSetAttribute(generate_libra_spmm,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    } else if (dynamic_shared_size > 65536) {
      int maxbytes = 98304; // 96 KB
      cudaFuncSetAttribute(generate_libra_spmm,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    } else if (dynamic_shared_size > 32768) {
      int maxbytes = 65536; // 128 KB
      cudaFuncSetAttribute(generate_libra_spmm,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    }
    //每个block负责一个window, 每个block中只有一个线程
    generate_libra_spmm<<<window_count, block_size, dynamic_shared_size>>>(
      nodePointer, edgelist, edgelist_sort, counts, edgetocol, edgetorow,
      blockPartition, groupPartition, valuePartition, vectorPartition, sizePartition,
      blockOffset, groupOffset, valueOffset,
      cudaLongPartition, cudaShortPartition, 
      cudaLong_offset, cudaShort_offset, 
      cudaLongGroupPartition, cudaShortGroupPartition, 
      cudaLongGroup_offset, cudaShortGroup_offset, 
      WindowOffset, Curwindow, t_Atomic, BlockOffset, Binary, ColumnIndice, t_Value,
      cuda_long_group, cuda_long_row, cuda_long_atomic, cuda_long_column, cuda_long_value,
      cuda_short_group, cuda_short_row, cuda_short_atomic, cuda_short_column, cuda_short_value,
      blockSize_h, blockSize_w, num_nodes, threshold, Short_len,t_s,c_s);
	  cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
}


__global__ void generate_edgetocolumn_libra_sddmm(int *nodePointer, int *edgelist,
                                      int *edgelist_sort, int *counts, int *edgetocol,
                                      int *blockpartition, int *blocknum, int *vectornum,
                                      int blockSize_h, int blockSize_w,
                                      int num_nodes) {
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = nodePointer[winId * blockSize_h];
  unsigned block_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = block_end - block_start;
  if (num_window_edges == 0)
    return;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  int *start = edgelist_sort + block_start;
  int *counts_cur = counts + block_start;
  int size = 0;
  //去重
  inplace_deduplication_libra_spmm(start, counts_cur, num_window_edges, &size);
	if(winId==0)
	printf("%d\n", size);

if (winId == 0) {  // 仅在 winId == 0 的线程块中打印，避免冗余输出
  printf("Array start contents:\n");
  for (int i = 0; i < size; i++) {
    printf("%d ", start[i]);
  }
  printf("\n");
  printf("Counts start contents:\n");
  for (int i = 0; i < size; i++) {
    printf("%d ", counts[i]);
  }
  printf("\n");
}

  //num是每个窗口有多少个block
  int num = (size + blockSize_w) / blockSize_w;
  if((size%blockSize_w)==0) num-=1;
  atomicAdd(blocknum, num);
  atomicAdd(vectornum, size);
  //vector个数
  blockpartition[winId] = size;
  for (unsigned idx = block_start; idx < block_end; idx += 1) {
    int index = binarysearch(start, size + 1, edgelist[idx]);
    edgetocol[idx] = index;
  }
}
void generate_edgetocolumn_cuda_libra_sddmm(int *nodePointer, int *edgelist,
                                int *edgelist_sort, int *counts, int *edgetocol,
                                int *blockpartition, int *blocknum, int * vectornum,
                                int blockSize_h, int blockSize_w,
                                int num_nodes) {
  int block_size = 1;
  int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
  //每个block负责一个window, 每个block中只有一个线程
//   int block_size1 = 128;
//   int block_count1 = (window_count + 127) / 128;
  generate_edgetocolumn_libra_sddmm<<<window_count, block_size>>>(
      nodePointer, edgelist, edgelist_sort, counts, edgetocol, blockpartition, blocknum, vectornum,
      blockSize_h, blockSize_w, num_nodes);
  // generate_edgetocolumn_v1<<< window_count, block_size >>> (nodePointer,
  // edgelist, edgelist_sort, edgetocol, blockpartition, blocknum, blockSize_h,
  // blockSize_w, num_nodes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

/*Generate TC offset, tileid and AtoB*/
__global__ void generate_tcoffset_id_atob(
    int *nodePointer, int *rowwindow_offset, int *edgeToColumn, int *edgeToRow,
    int *edgeList, int *tcblock_offset, uint8_t *tcblocktile_id, float *values,
    int *sparseatob, int max_block, int num_nodes, int blockSize_h,
    int blockSize_w, int num_row_windows) {
  extern __shared__ int pos_ptr[];
  int tid = threadIdx.x;
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = rowwindow_offset[winId];
  unsigned block_end = rowwindow_offset[min(winId + 1, num_row_windows)];
  unsigned num_blocks = block_end - block_start;
  if (num_blocks == 0) {
    return;
  }
  int *tcblock_offset_ptr = pos_ptr + num_blocks;
  int *tcblock_offset_global_ptr = tcblock_offset + block_start;
  int *tcblock_nnz_ptr = pos_ptr + num_blocks + 1;
  unsigned element_start = nodePointer[winId * blockSize_h];
  unsigned element_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = element_end - element_start;
  if (num_window_edges == 0) {
    return;
  }
  for (int i = 0; i < 2 * num_blocks + 1; i++) {
    pos_ptr[i] = 0;
  }

  for (unsigned e_index = element_start; e_index < element_end; e_index++) {
    unsigned col = edgeToColumn[e_index]; // new col
	//所在的块中的非零元个数+1
    tcblock_nnz_ptr[col / blockSize_w]++;
  }
  //每个块的非零元个数
  for (int i = 0; i < num_blocks; i++) {
    tcblock_offset_global_ptr[i] = tcblock_nnz_ptr[i];
  }
  for (int i = 0; i < num_blocks; i++) {
    tcblock_nnz_ptr[i] += tcblock_nnz_ptr[i - 1];
  }

  //开始看每个非零元在block内的偏移了
  auto tileid = tcblocktile_id + element_start;
  auto values_ = values + element_start;
  auto sparse_AToB = sparseatob + block_start * blockSize_w;
  for (unsigned e_index = element_start; e_index < element_end; e_index++) {
    unsigned col = edgeToColumn[e_index]; // new col
    unsigned tcblock_id = col / blockSize_w;
    unsigned row_local = edgeToRow[e_index] % blockSize_h;
    unsigned col_local = col % blockSize_w;
	//当前非零元素在tileid的位置 = 前面块内非零元个数 + 当前块内的偏移
    tileid[tcblock_offset_ptr[tcblock_id] + pos_ptr[tcblock_id]] =
        (uint8_t)(row_local * blockSize_w + col_local);
    values_[tcblock_offset_ptr[tcblock_id] + pos_ptr[tcblock_id]] = 1.0;
	//可能存在一个vector中的元素多次赋值，但不影响
    sparse_AToB[tcblock_id * blockSize_w + col_local] = edgeList[e_index];
    pos_ptr[tcblock_id]++;
  }
}

/*Generate TC offset, tileid and AtoB*/
__global__ void generate_tcoffset_id_atob_fs(
    int *nodePointer, int *rowwindow_offset, int *edgeToColumn, int *edgeToRow,
    int *edgeList, float *values,
    int *sparseatob, int max_block, int num_nodes, int blockSize_h,
    int blockSize_w, int num_row_windows) {
//   extern __shared__ int pos_ptr[];
  int tid = threadIdx.x;
  int winId = blockIdx.x; // each warp one window
  unsigned vector_start = rowwindow_offset[winId];
  unsigned vector_end = rowwindow_offset[min(winId + 1, num_row_windows)];
  unsigned num_vector = vector_end - vector_start;
  if (num_vector == 0) {
    return;
  }
//   int *tcblock_offset_ptr = pos_ptr + num_blocks;
//   int *tcblock_offset_global_ptr = tcblock_offset + block_start;
//   int *tcblock_nnz_ptr = pos_ptr + num_blocks + 1;
  unsigned element_start = nodePointer[winId * blockSize_h];
  unsigned element_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = element_end - element_start;
  if (num_window_edges == 0) {
    return;
  }
//   for (int i = 0; i < 2 * num_blocks + 1; i++) {
//     pos_ptr[i] = 0;
//   }

//   for (unsigned e_index = element_start; e_index < element_end; e_index++) {
//     unsigned col = edgeToColumn[e_index]; // new col
// 	//所在的块中的非零元个数+1
//     tcblock_nnz_ptr[col / blockSize_w]++;
//   }
//   //每个块的非零元个数
//   for (int i = 0; i < num_blocks; i++) {
//     tcblock_offset_global_ptr[i] = tcblock_nnz_ptr[i];
//   }
//   for (int i = 0; i < num_blocks; i++) {
//     tcblock_nnz_ptr[i] += tcblock_nnz_ptr[i - 1];
//   }

  //开始看每个非零元在block内的偏移了
//   auto tileid = tcblocktile_id + element_start;
  auto values_ = values + vector_start*blockSize_h;
  auto sparse_AToB = sparseatob + vector_start;
  for (unsigned e_index = element_start; e_index < element_end; e_index++) {
    unsigned col = edgeToColumn[e_index]; // new col
    unsigned tcblock_id = col / blockSize_w;
    unsigned row_local = edgeToRow[e_index] % blockSize_h;
    unsigned col_local = col % blockSize_w;

	//如果存在， 且元素在residue里，需要按每行residue偏移
	int residue = num_vector % blockSize_w;
	if(residue>0 & col>=(num_vector-residue)){
		values_[tcblock_id*blockSize_h*blockSize_w + row_local*residue] = 1.0;
	}else{
		values_[tcblock_id*blockSize_h*blockSize_w + row_local*blockSize_w] = 1.0;
	}
	sparse_AToB[tcblock_id * blockSize_w + col_local] = edgeList[e_index];
    // pos_ptr[tcblock_id]++;
  }
}

void generate_tcoffset_id_atob_cuda(int *nodePointer, int *rowwindow_offset,
                                    int *edgeToColumn, int *edgeToRow,
                                    int *edgeList, int *tcblock_offset,
                                    uint8_t *tcblock_tileid, float *values, int *sparseatob,
                                    int max_block, int num_nodes,
                                    int blockSize_h, int blockSize_w,
                                    int num_row_windows) {
  int block_size = 1;
  int window_count = num_row_windows;
  const int dynamic_shared_size = (2 * max_block + 1) * sizeof(int);
  std::cout << "dynamic_shared_size: " << dynamic_shared_size << std::endl;
  if (dynamic_shared_size > 98304) {
    int maxbytes = 131072; // 96 KB
    cudaFuncSetAttribute(generate_tcoffset_id_atob,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  } else if (dynamic_shared_size > 65536) {
    int maxbytes = 98304; // 96 KB
    cudaFuncSetAttribute(generate_tcoffset_id_atob,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  } else if (dynamic_shared_size > 32768) {
    int maxbytes = 65536; // 128 KB
    cudaFuncSetAttribute(generate_tcoffset_id_atob,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  }
  generate_tcoffset_id_atob<<<window_count, block_size, dynamic_shared_size>>>(
      nodePointer, rowwindow_offset, edgeToColumn, edgeToRow, edgeList,
      tcblock_offset, tcblock_tileid, values, sparseatob, max_block, num_nodes,
      blockSize_h, blockSize_w, num_row_windows);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

void generate_tcoffset_id_atob_cuda_fs(int *nodePointer, int *rowwindow_offset,
                                    int *edgeToColumn, int *edgeToRow,
                                    int *edgeList, float *values, int *sparseatob,
                                    int max_block, int num_nodes,
                                    int blockSize_h, int blockSize_w,
                                    int num_row_windows) {
  int block_size = 1;
  int window_count = num_row_windows;
//   const int dynamic_shared_size = (2 * max_block + 1) * sizeof(int);
//   std::cout << "dynamic_shared_size: " << dynamic_shared_size << std::endl;
//   if (dynamic_shared_size > 98304) {
//     int maxbytes = 131072; // 96 KB
//     cudaFuncSetAttribute(generate_tcoffset_id_atob_fs,
//                          cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
//   } else if (dynamic_shared_size > 65536) {
//     int maxbytes = 98304; // 96 KB
//     cudaFuncSetAttribute(generate_tcoffset_id_atob_fs,
//                          cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
//   } else if (dynamic_shared_size > 32768) {
//     int maxbytes = 65536; // 128 KB
//     cudaFuncSetAttribute(generate_tcoffset_id_atob_fs,
//                          cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
//   }
  generate_tcoffset_id_atob_fs<<<window_count, block_size>>>(
      nodePointer, rowwindow_offset, edgeToColumn, edgeToRow, edgeList,
    values, sparseatob, max_block, num_nodes,
      blockSize_h, blockSize_w, num_row_windows);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

void padding_up_8(int *input, int size) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  roundup_to_multiple_of_eight<<<blocksPerGrid, threadsPerBlock>>>(input, size);
}
void get_padding_tileid(int *ori_offset, uint8_t *ori_tileid,
                        int *padded_offset, uint8_t *padded_tileid, int size) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  get_padding_tileid_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      ori_offset, ori_tileid, padded_offset, padded_tileid, size);
}

void print_first_20(const thrust::device_vector<int>& seg, const thrust::device_vector<int>& el, const std::string& label) {
    std::cout << label << "前 20 个值:" << std::endl;
    std::cout << "Seg: ";
    thrust::host_vector<int> host_seg(seg.begin(), seg.begin() + 176);
    for (int i = 0; i < 176; i++) {
        std::cout << host_seg[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "EL: ";
    thrust::host_vector<int> host_el(el.begin(), el.begin() + 176);
    for (int i = 0; i < 176; i++) {
        std::cout << host_el[i] << " ";
    }
    std::cout << std::endl;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
torch::Tensor, torch::Tensor, torch::Tensor,
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
seg_sort_dequ_libra_spmm(int *seg, int *edgeLists, int *nodepointer, int *edgetocol, int *edgetorow, 
              int *blockPartition, int * groupPartition, int * valuePartition, int *vectorPartition, int *sizePartition,
              int *block_offset, int * group_offset,  int * value_offset,
              int *cudaLongPartition, int *cudaShortPartition, int *cudaLong_offset, int *cudaShort_offset,
              int * cudaLongGroupPartition, int * cudaShortGroupPartition, int *cudaLongGroup_offset, int *cudaShortGroup_offset,
              int *tc_group, int * vector_num, int * vector_nnz,
              int *cuda_long_group, int *cuda_long, int *cuda_short_group, int *cuda_short,
              int blockSize_h, int blockSize_w,
              int num_nodes, int num_edges, int rowwindow_num, int threshold, int Short_len, int t_s, int c_s) {
	thrust::device_ptr<int> Seg = thrust::device_pointer_cast(seg);
	thrust::device_vector<int> deviceSeg(Seg, Seg + num_edges);
	thrust::device_ptr<int> EL = thrust::device_pointer_cast(edgeLists);
	thrust::device_vector<int> deviceEL(EL, EL + num_edges);
	auto begin = thrust::make_zip_iterator(
		thrust::make_tuple(deviceSeg.begin(), deviceEL.begin()));
	auto end = thrust::make_zip_iterator(
		thrust::make_tuple(deviceSeg.end(), deviceEL.end()));
	// 1. 对所有非零元素按照所属于的window id排序， 更重要的是window内按照col排序
	thrust::sort(thrust::device, begin, end);

  // 2. 统计每个window的block和group偏移
	thrust::device_ptr<int> Counts = thrust::device_pointer_cast(edgeLists);
	thrust::device_vector<int> deviceCounts(Counts, Counts + num_edges);
	generate_partiton_information_cuda_libra_spmm(
		nodepointer, edgeLists, thrust::raw_pointer_cast(&deviceEL[0]), thrust::raw_pointer_cast(&deviceCounts[0]),
		edgetocol, blockPartition, groupPartition, valuePartition, vectorPartition, sizePartition,
    cudaLongPartition, cudaShortPartition, cudaLongGroupPartition, cudaShortGroupPartition,
		tc_group, vector_num, vector_nnz,
		cuda_long_group, cuda_long, cuda_short_group, cuda_short, 
		blockSize_h, blockSize_w, num_nodes,threshold,Short_len,t_s,c_s);

  //整合blockpartition和groupPartition
	thrust::device_ptr<int> blockPartition_ptr =
		thrust::device_pointer_cast(blockPartition);
	thrust::device_vector<int> blockPartition_vector(
		blockPartition_ptr, blockPartition_ptr + rowwindow_num);
	thrust::device_ptr<int> block_offset_ptr =
		thrust::device_pointer_cast(block_offset + 1);
	thrust::inclusive_scan(blockPartition_vector.begin(),
							blockPartition_vector.end(), block_offset_ptr);


	thrust::device_ptr<int> groupPartition_ptr =
		thrust::device_pointer_cast(groupPartition);
	thrust::device_vector<int> groupPartition_vector(
		groupPartition_ptr, groupPartition_ptr + rowwindow_num);
	thrust::device_ptr<int> group_offset_ptr =
		thrust::device_pointer_cast(group_offset +1);
	thrust::inclusive_scan(groupPartition_vector.begin(),
							groupPartition_vector.end(), group_offset_ptr);


	thrust::device_ptr<int> valuePartition_ptr =
		thrust::device_pointer_cast(valuePartition);
	thrust::device_vector<int> valuePartition_vector(
		valuePartition_ptr, valuePartition_ptr + rowwindow_num);
	thrust::device_ptr<int> value_offset_ptr =
		thrust::device_pointer_cast(value_offset + 1);
	thrust::inclusive_scan(valuePartition_vector.begin(),
							valuePartition_vector.end(), value_offset_ptr);

	thrust::device_ptr<int> cudaLongPartition_ptr =
		thrust::device_pointer_cast(cudaLongPartition);
	thrust::device_vector<int> vcudaLongPartition_vector(
		cudaLongPartition_ptr, cudaLongPartition_ptr + rowwindow_num*blockSize_h);
	thrust::device_ptr<int> cudaLong_offset_ptr =
		thrust::device_pointer_cast(cudaLong_offset + 1);
	thrust::inclusive_scan(vcudaLongPartition_vector.begin(),
							vcudaLongPartition_vector.end(), cudaLong_offset_ptr);

	thrust::device_ptr<int> cudaShortPartition_ptr =
		thrust::device_pointer_cast(cudaShortPartition);
	thrust::device_vector<int> vcudaShortPartition_vector(
		cudaShortPartition_ptr, cudaShortPartition_ptr + rowwindow_num*blockSize_h);
	thrust::device_ptr<int> cudaShort_offset_ptr =
		thrust::device_pointer_cast(cudaShort_offset + 1);
	thrust::inclusive_scan(vcudaShortPartition_vector.begin(),
							vcudaShortPartition_vector.end(), cudaShort_offset_ptr);

	thrust::device_ptr<int> cudaLongGroupPartition_ptr =
		thrust::device_pointer_cast(cudaLongGroupPartition);
	thrust::device_vector<int> vcudaLongGroupPartition_vector(
		cudaLongGroupPartition_ptr, cudaLongGroupPartition_ptr + rowwindow_num);
	thrust::device_ptr<int> cudaLongGroup_offset_ptr =
		thrust::device_pointer_cast(cudaLongGroup_offset + 1);
	thrust::inclusive_scan(vcudaLongGroupPartition_vector.begin(),
							vcudaLongGroupPartition_vector.end(), cudaLongGroup_offset_ptr);
              
	thrust::device_ptr<int> cudaShortGroupPartition_ptr =
		thrust::device_pointer_cast(cudaShortGroupPartition);
	thrust::device_vector<int> vcudaShortGroupPartition_vector(
		cudaShortGroupPartition_ptr, cudaShortGroupPartition_ptr + rowwindow_num);
	thrust::device_ptr<int> cudaShortGroup_offset_ptr =
		thrust::device_pointer_cast(cudaShortGroup_offset + 1);
	thrust::inclusive_scan(vcudaShortGroupPartition_vector.begin(),
							vcudaShortGroupPartition_vector.end(), cudaShortGroup_offset_ptr);

	//从设备端拷贝到主机端的
	thrust::device_ptr<int> tc_group_ptr = thrust::device_pointer_cast(tc_group);
	thrust::host_vector<int> tc_group_vector(tc_group_ptr, tc_group_ptr + 1);
	int h_tc_group = tc_group_vector[0];

	thrust::device_ptr<int> vector_num_ptr = thrust::device_pointer_cast(vector_num);
	thrust::host_vector<int> vector_num_vector(vector_num_ptr, vector_num_ptr + 1);
	int h_vector_num = vector_num_vector[0];
  int h_blocks = h_vector_num/blockSize_w;

	thrust::device_ptr<int> vector_nnz_ptr = thrust::device_pointer_cast(vector_nnz);
	thrust::host_vector<int> vector_nnz_vector(vector_nnz_ptr, vector_nnz_ptr + 1);
	int h_vector_nnz = vector_nnz_vector[0];

	thrust::device_ptr<int> cuda_long_group_ptr = thrust::device_pointer_cast(cuda_long_group);
	thrust::host_vector<int> cuda_long_group_vector(cuda_long_group_ptr, cuda_long_group_ptr + 1);
	int h_cuda_long_group = cuda_long_group_vector[0];

	thrust::device_ptr<int> cuda_long_ptr = thrust::device_pointer_cast(cuda_long);
	thrust::host_vector<int> cuda_long_vector(cuda_long_ptr, cuda_long_ptr + 1);
	int h_cuda_long = cuda_long_vector[0];

	thrust::device_ptr<int> cuda_short_group_ptr = thrust::device_pointer_cast(cuda_short_group);
	thrust::host_vector<int> cuda_short_group_vector(cuda_short_group_ptr, cuda_short_group_ptr + 1);
	int h_cuda_short_group = cuda_short_group_vector[0];

	thrust::device_ptr<int> cuda_short_ptr = thrust::device_pointer_cast(cuda_short);
	thrust::host_vector<int> cuda_short_vector(cuda_short_ptr, cuda_short_ptr + 1);
	int h_cuda_short = cuda_short_vector[0];

	// printf("%d\n", h_cuda_long_group);
	// printf("%d\n", h_cuda_long);
	// printf("%d\n", h_cuda_short_group);
  // printf("%d\n", h_cuda_short);
  // 根据h_tc_group构造 WindowOffset, Curwindow, t_Atomic
  // 根据h_vector_num构造 BlockOffset, Binary, ColumnIndice, 
  // 根据h_vector_nnz构造 t_Value
	auto options_gpu =
		torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	auto options_gpu_long =
		torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  auto options_gpu_fp32 = 
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

  //TCU
  auto WindowOffset_tensor = torch::zeros({h_tc_group}, options_gpu);
  auto WindowOffset_out_tensor = torch::zeros({(h_tc_group+1)}, options_gpu);
  auto Curwindow_tensor = torch::zeros({h_tc_group}, options_gpu).clone();
  auto t_Atomic_tensor = torch::zeros({h_tc_group}, options_gpu).clone();
  auto BlockOffset_tensor = torch::zeros({h_blocks}, options_gpu);
  auto BlockOffset_out_tensor = torch::zeros({(h_blocks+1)}, options_gpu);
  auto Binary_tensor = torch::zeros({h_blocks}, options_gpu_long).clone();
  auto ColumnIndice_tensor = torch::full({h_vector_num}, -1, options_gpu).clone();
  auto t_Value_tensor = torch::zeros({h_vector_nnz}, options_gpu_fp32).clone();
  //CUDA
  auto cuda_long_offset_tensor = torch::zeros({h_cuda_long_group}, options_gpu);
  auto cuda_long_offset_out_tensor = torch::zeros({(h_cuda_long_group+1)}, options_gpu);
  auto cuda_long_row_tensor = torch::zeros({h_cuda_long_group}, options_gpu).clone();
  auto cuda_long_atomic_tensor = torch::zeros({h_cuda_long_group}, options_gpu).clone();
  auto cuda_long_column_tensor = torch::full({h_cuda_long}, -1, options_gpu).clone();
  auto cuda_long_value_tensor = torch::zeros({h_cuda_long}, options_gpu_fp32).clone();

  auto cuda_short_offset_tensor = torch::zeros({h_cuda_short_group}, options_gpu);
  auto cuda_short_offset_out_tensor = torch::zeros({(h_cuda_short_group+1)}, options_gpu);
  auto cuda_short_row_tensor = torch::zeros({h_cuda_short_group}, options_gpu).clone();
  auto cuda_short_atomic_tensor = torch::zeros({h_cuda_short_group}, options_gpu).clone();
  auto cuda_short_column_tensor = torch::full({h_cuda_short}, -1, options_gpu).clone();
  auto cuda_short_value_tensor = torch::zeros({h_cuda_short}, options_gpu_fp32).clone();

  //TCU
  auto WindowOffset = WindowOffset_tensor.data<int>();
  auto WindowOffset_out = WindowOffset_out_tensor.data<int>();
  auto Curwindow = Curwindow_tensor.data<int>();
  auto t_Atomic = t_Atomic_tensor.data<int>();
  auto BlockOffset = BlockOffset_tensor.data<int>();
  auto Binary = Binary_tensor.data<long>();
  auto ColumnIndice = ColumnIndice_tensor.data<int>();
  auto t_Value = t_Value_tensor.data<float>();
  //CUDA
  auto cuda_long_offset = cuda_long_offset_tensor.data<int>();
  auto cuda_long_row = cuda_long_row_tensor.data<int>();
  auto cuda_long_atomic = cuda_long_atomic_tensor.data<int>();
  auto cuda_long_column = cuda_long_column_tensor.data<int>();
  auto cuda_long_value = cuda_long_value_tensor.data<float>();

  auto cuda_short_offset = cuda_short_offset_tensor.data<int>();
  auto cuda_short_row = cuda_short_row_tensor.data<int>();
  auto cuda_short_atomic = cuda_short_atomic_tensor.data<int>();
  auto cuda_short_column = cuda_short_column_tensor.data<int>();
  auto cuda_short_value = cuda_short_value_tensor.data<float>();

  auto max_element =
      thrust::max_element(thrust::device, blockPartition_vector.begin(),
                          blockPartition_vector.end());
  int max_blocks = *max_element;

	generate_cuda_libra_spmm(
		nodepointer, edgeLists, thrust::raw_pointer_cast(&deviceEL[0]), thrust::raw_pointer_cast(&deviceCounts[0]),edgetocol, edgetorow,
    blockPartition, groupPartition, valuePartition, vectorPartition, sizePartition,
    block_offset, group_offset, value_offset,
    cudaLongPartition, cudaShortPartition, 
    cudaLong_offset, cudaShort_offset, 
    cudaLongGroupPartition, cudaShortGroupPartition, 
    cudaLongGroup_offset, cudaShortGroup_offset, 
    WindowOffset, Curwindow, t_Atomic, BlockOffset, Binary, ColumnIndice, t_Value,
    cuda_long_offset, cuda_long_row, cuda_long_atomic, cuda_long_column, cuda_long_value,
    cuda_short_offset, cuda_short_row, cuda_short_atomic, cuda_short_column, cuda_short_value,
		blockSize_h, blockSize_w, num_nodes, threshold, Short_len, t_s, c_s, max_blocks);

  //累加WindowOffset_tensor
	thrust::device_ptr<int> WindowOffset_ptr =
		thrust::device_pointer_cast(WindowOffset);
	thrust::device_vector<int> WindowOffset_vector(
		WindowOffset_ptr, WindowOffset_ptr + h_tc_group);
	thrust::device_ptr<int> WindowOffset_out_ptr =
	thrust::device_pointer_cast(WindowOffset_out_tensor.data<int>()+1);
	thrust::inclusive_scan(WindowOffset_vector.begin(), WindowOffset_vector.end(), WindowOffset_out_ptr);

    //   thrust::host_vector<int> host_data(WindowOffset_ptr, WindowOffset_ptr + h_tc_group);
    // // 打印主机数据
    // std::cout << "Contents of device pointer:" << std::endl;
    // // for (int i = 0; i < 20; ++i) {
    //     std::cout << host_data[50403] << " ";
    //     std::cout << host_data[50404] << " ";
    // // }
    // std::cout << std::endl;


  //累加BlockOffset_tensor
	thrust::device_ptr<int> BlockOffset_ptr =
		thrust::device_pointer_cast(BlockOffset);
	thrust::device_vector<int> BlockOffset_vector(
		BlockOffset_ptr, BlockOffset_ptr + h_blocks);
	thrust::device_ptr<int> BlockOffset_out_ptr =
	thrust::device_pointer_cast(BlockOffset_out_tensor.data<int>()+1);
	thrust::inclusive_scan(BlockOffset_vector.begin(), BlockOffset_vector.end(), BlockOffset_out_ptr);

  //累加cuda_long_offset_tensor
	thrust::device_ptr<int> cuda_long_offset_ptr =
		thrust::device_pointer_cast(cuda_long_offset);
	thrust::device_vector<int> cuda_long_offset_vector(
		cuda_long_offset_ptr, cuda_long_offset_ptr + h_cuda_long_group);
	thrust::device_ptr<int> cuda_long_offset_out_ptr =
	thrust::device_pointer_cast(cuda_long_offset_out_tensor.data<int>()+1);
	thrust::inclusive_scan(cuda_long_offset_vector.begin(), cuda_long_offset_vector.end(), cuda_long_offset_out_ptr);

  //累加cuda_short_offset_tensor
	thrust::device_ptr<int> cuda_short_offset_ptr =
		thrust::device_pointer_cast(cuda_short_offset);
	thrust::device_vector<int> cuda_short_offset_vector(
		cuda_short_offset_ptr, cuda_short_offset_ptr + h_cuda_short_group);
	thrust::device_ptr<int> cuda_short_offset_out_ptr =
	thrust::device_pointer_cast(cuda_short_offset_out_tensor.data<int>()+1);
	thrust::inclusive_scan(cuda_short_offset_vector.begin(), cuda_short_offset_vector.end(), cuda_short_offset_out_ptr);

	return std::make_tuple(
  WindowOffset_out_tensor, Curwindow_tensor, t_Atomic_tensor, ColumnIndice_tensor, 
  BlockOffset_out_tensor, Binary_tensor, t_Value_tensor,
  cuda_long_offset_out_tensor, cuda_long_row_tensor, cuda_long_atomic_tensor, cuda_long_column_tensor, cuda_long_value_tensor,
  cuda_short_offset_out_tensor, cuda_short_row_tensor, cuda_short_atomic_tensor, cuda_short_column_tensor, cuda_short_value_tensor);
}

/*
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
torch::Tensor, torch::Tensor, torch::Tensor>
seg_sort_dequ_libra_sddmm(int *seg, int *edgeLists, int *nodepointer, int *edgetocol, int *edgetorow,
              int *blockpartition, int * groupPartition, int * valuePartition, int *vectorPartition, int *sizePartition,
              int *block_offset, int * group_offset, 
              int *cudaLongPartition, int *cudaLong_offset,
              int * cudaLongGroupPartition, int *cudaLongGroup_offset,
              int *tc_group, int * vector_num, int * vector_nnz,
              int *cuda_long_group, int *cuda_long,
              int blockSize_h, int blockSize_w,
              int num_nodes, int num_edges, int rowwindow_num, int threshold, int t_s, int c_s) {
	thrust::device_ptr<int> Seg = thrust::device_pointer_cast(seg);
	thrust::device_vector<int> deviceSeg(Seg, Seg + num_edges);
	thrust::device_ptr<int> EL = thrust::device_pointer_cast(edgeLists);
	thrust::device_vector<int> deviceEL(EL, EL + num_edges);
	auto begin = thrust::make_zip_iterator(
		thrust::make_tuple(deviceSeg.begin(), deviceEL.begin()));
	auto end = thrust::make_zip_iterator(
		thrust::make_tuple(deviceSeg.end(), deviceEL.end()));
	// 1. 对所有非零元素按照所属于的window id排序， 更重要的是window内按照col排序
	thrust::sort(thrust::device, begin, end);

  // 2. 统计每个window的block和group偏移
	thrust::device_ptr<int> Counts = thrust::device_pointer_cast(edgeLists);
	thrust::device_vector<int> deviceCounts(Counts, Counts + num_edges);
	generate_partiton_information_cuda_libra_sddmm(
		nodepointer, edgeLists, thrust::raw_pointer_cast(&deviceEL[0]), thrust::raw_pointer_cast(&deviceCounts[0]),
		edgetocol, blockPartition, groupPartition, valuePartition, vectorPartition, sizePartition,
    cudaLongPartition, cudaLongGroupPartition,
		tc_group, vector_num, vector_nnz,
		cuda_long_group, cuda_long,
		blockSize_h, blockSize_w, num_nodes,threshold,t_s,c_s);

  //整合blockpartition和groupPartition
	thrust::device_ptr<int> blockPartition_ptr =
		thrust::device_pointer_cast(blockPartition);
	thrust::device_vector<int> blockPartition_vector(
		blockPartition_ptr, blockPartition_ptr + rowwindow_num);
	thrust::device_ptr<int> block_offset_ptr =
		thrust::device_pointer_cast(block_offset + 1);
	thrust::inclusive_scan(blockPartition_vector.begin(),
							blockPartition_vector.end(), block_offset_ptr);


	thrust::device_ptr<int> groupPartition_ptr =
		thrust::device_pointer_cast(groupPartition);
	thrust::device_vector<int> groupPartition_vector(
		groupPartition_ptr, groupPartition_ptr + rowwindow_num);
	thrust::device_ptr<int> group_offset_ptr =
		thrust::device_pointer_cast(group_offset +1);
	thrust::inclusive_scan(groupPartition_vector.begin(),
							groupPartition_vector.end(), group_offset_ptr);


	thrust::device_ptr<int> valuePartition_ptr =
		thrust::device_pointer_cast(valuePartition);
	thrust::device_vector<int> valuePartition_vector(
		valuePartition_ptr, valuePartition_ptr + rowwindow_num);
	thrust::device_ptr<int> value_offset_ptr =
		thrust::device_pointer_cast(value_offset + 1);
	thrust::inclusive_scan(valuePartition_vector.begin(),
							valuePartition_vector.end(), value_offset_ptr);

	thrust::device_ptr<int> cudaLongPartition_ptr =
		thrust::device_pointer_cast(cudaLongPartition);
	thrust::device_vector<int> vcudaLongPartition_vector(
		cudaLongPartition_ptr, cudaLongPartition_ptr + rowwindow_num*blockSize_h);
	thrust::device_ptr<int> cudaLong_offset_ptr =
		thrust::device_pointer_cast(cudaLong_offset + 1);
	thrust::inclusive_scan(vcudaLongPartition_vector.begin(),
							vcudaLongPartition_vector.end(), cudaLong_offset_ptr);

	thrust::device_ptr<int> cudaShortPartition_ptr =
		thrust::device_pointer_cast(cudaShortPartition);
	thrust::device_vector<int> vcudaShortPartition_vector(
		cudaShortPartition_ptr, cudaShortPartition_ptr + rowwindow_num*blockSize_h);
	thrust::device_ptr<int> cudaShort_offset_ptr =
		thrust::device_pointer_cast(cudaShort_offset + 1);
	thrust::inclusive_scan(vcudaShortPartition_vector.begin(),
							vcudaShortPartition_vector.end(), cudaShort_offset_ptr);

	thrust::device_ptr<int> cudaLongGroupPartition_ptr =
		thrust::device_pointer_cast(cudaLongGroupPartition);
	thrust::device_vector<int> vcudaLongGroupPartition_vector(
		cudaLongGroupPartition_ptr, cudaLongGroupPartition_ptr + rowwindow_num);
	thrust::device_ptr<int> cudaLongGroup_offset_ptr =
		thrust::device_pointer_cast(cudaLongGroup_offset + 1);
	thrust::inclusive_scan(vcudaLongGroupPartition_vector.begin(),
							vcudaLongGroupPartition_vector.end(), cudaLongGroup_offset_ptr);
              
	thrust::device_ptr<int> cudaShortGroupPartition_ptr =
		thrust::device_pointer_cast(cudaShortGroupPartition);
	thrust::device_vector<int> vcudaShortGroupPartition_vector(
		cudaShortGroupPartition_ptr, cudaShortGroupPartition_ptr + rowwindow_num);
	thrust::device_ptr<int> cudaShortGroup_offset_ptr =
		thrust::device_pointer_cast(cudaShortGroup_offset + 1);
	thrust::inclusive_scan(vcudaShortGroupPartition_vector.begin(),
							vcudaShortGroupPartition_vector.end(), cudaShortGroup_offset_ptr);

	//从设备端拷贝到主机端的
	thrust::device_ptr<int> tc_group_ptr = thrust::device_pointer_cast(tc_group);
	thrust::host_vector<int> tc_group_vector(tc_group_ptr, tc_group_ptr + 1);
	int h_tc_group = tc_group_vector[0];

	thrust::device_ptr<int> vector_num_ptr = thrust::device_pointer_cast(vector_num);
	thrust::host_vector<int> vector_num_vector(vector_num_ptr, vector_num_ptr + 1);
	int h_vector_num = vector_num_vector[0];
  int h_blocks = h_vector_num/blockSize_w;

	thrust::device_ptr<int> vector_nnz_ptr = thrust::device_pointer_cast(vector_nnz);
	thrust::host_vector<int> vector_nnz_vector(vector_nnz_ptr, vector_nnz_ptr + 1);
	int h_vector_nnz = vector_nnz_vector[0];

	thrust::device_ptr<int> cuda_long_group_ptr = thrust::device_pointer_cast(cuda_long_group);
	thrust::host_vector<int> cuda_long_group_vector(cuda_long_group_ptr, cuda_long_group_ptr + 1);
	int h_cuda_long_group = cuda_long_group_vector[0];

	thrust::device_ptr<int> cuda_long_ptr = thrust::device_pointer_cast(cuda_long);
	thrust::host_vector<int> cuda_long_vector(cuda_long_ptr, cuda_long_ptr + 1);
	int h_cuda_long = cuda_long_vector[0];

	thrust::device_ptr<int> cuda_short_group_ptr = thrust::device_pointer_cast(cuda_short_group);
	thrust::host_vector<int> cuda_short_group_vector(cuda_short_group_ptr, cuda_short_group_ptr + 1);
	int h_cuda_short_group = cuda_short_group_vector[0];

	thrust::device_ptr<int> cuda_short_ptr = thrust::device_pointer_cast(cuda_short);
	thrust::host_vector<int> cuda_short_vector(cuda_short_ptr, cuda_short_ptr + 1);
	int h_cuda_short = cuda_short_vector[0];

	// printf("%d\n", h_cuda_long_group);
	// printf("%d\n", h_cuda_long);
	// printf("%d\n", h_cuda_short_group);
  // printf("%d\n", h_cuda_short);
  // 根据h_tc_group构造 WindowOffset, Curwindow, t_Atomic
  // 根据h_vector_num构造 BlockOffset, Binary, ColumnIndice, 
  // 根据h_vector_nnz构造 t_Value
	auto options_gpu =
		torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	auto options_gpu_long =
		torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  auto options_gpu_fp32 = 
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

  //TCU
  auto WindowOffset_tensor = torch::zeros({h_tc_group}, options_gpu);
  auto WindowOffset_out_tensor = torch::zeros({(h_tc_group+1)}, options_gpu);
  auto Curwindow_tensor = torch::zeros({h_tc_group}, options_gpu).clone();
  auto t_Atomic_tensor = torch::zeros({h_tc_group}, options_gpu).clone();
  auto BlockOffset_tensor = torch::zeros({h_blocks}, options_gpu);
  auto BlockOffset_out_tensor = torch::zeros({(h_blocks+1)}, options_gpu);
  auto Binary_tensor = torch::zeros({h_blocks}, options_gpu_long).clone();
  auto ColumnIndice_tensor = torch::full({h_vector_num}, -1, options_gpu).clone();
  auto t_Value_tensor = torch::zeros({h_vector_nnz}, options_gpu_fp32).clone();
  //CUDA
  auto cuda_long_offset_tensor = torch::zeros({h_cuda_long_group}, options_gpu);
  auto cuda_long_offset_out_tensor = torch::zeros({(h_cuda_long_group+1)}, options_gpu);
  auto cuda_long_row_tensor = torch::zeros({h_cuda_long_group}, options_gpu).clone();
  auto cuda_long_atomic_tensor = torch::zeros({h_cuda_long_group}, options_gpu).clone();
  auto cuda_long_column_tensor = torch::full({h_cuda_long}, -1, options_gpu).clone();
  auto cuda_long_value_tensor = torch::zeros({h_cuda_long}, options_gpu_fp32).clone();

  auto cuda_short_offset_tensor = torch::zeros({h_cuda_short_group}, options_gpu);
  auto cuda_short_offset_out_tensor = torch::zeros({(h_cuda_short_group+1)}, options_gpu);
  auto cuda_short_row_tensor = torch::zeros({h_cuda_short_group}, options_gpu).clone();
  auto cuda_short_atomic_tensor = torch::zeros({h_cuda_short_group}, options_gpu).clone();
  auto cuda_short_column_tensor = torch::full({h_cuda_short}, -1, options_gpu).clone();
  auto cuda_short_value_tensor = torch::zeros({h_cuda_short}, options_gpu_fp32).clone();

  //TCU
  auto WindowOffset = WindowOffset_tensor.data<int>();
  auto WindowOffset_out = WindowOffset_out_tensor.data<int>();
  auto Curwindow = Curwindow_tensor.data<int>();
  auto t_Atomic = t_Atomic_tensor.data<int>();
  auto BlockOffset = BlockOffset_tensor.data<int>();
  auto Binary = Binary_tensor.data<long>();
  auto ColumnIndice = ColumnIndice_tensor.data<int>();
  auto t_Value = t_Value_tensor.data<float>();
  //CUDA
  auto cuda_long_offset = cuda_long_offset_tensor.data<int>();
  auto cuda_long_row = cuda_long_row_tensor.data<int>();
  auto cuda_long_atomic = cuda_long_atomic_tensor.data<int>();
  auto cuda_long_column = cuda_long_column_tensor.data<int>();
  auto cuda_long_value = cuda_long_value_tensor.data<float>();

  auto cuda_short_offset = cuda_short_offset_tensor.data<int>();
  auto cuda_short_row = cuda_short_row_tensor.data<int>();
  auto cuda_short_atomic = cuda_short_atomic_tensor.data<int>();
  auto cuda_short_column = cuda_short_column_tensor.data<int>();
  auto cuda_short_value = cuda_short_value_tensor.data<float>();

  auto max_element =
      thrust::max_element(thrust::device, blockPartition_vector.begin(),
                          blockPartition_vector.end());
  int max_blocks = *max_element;

	generate_cuda_libra_spmm(
		nodepointer, edgeLists, thrust::raw_pointer_cast(&deviceEL[0]), thrust::raw_pointer_cast(&deviceCounts[0]),edgetocol, edgetorow,
    blockPartition, groupPartition, valuePartition, vectorPartition, sizePartition,
    block_offset, group_offset, value_offset,
    cudaLongPartition, cudaShortPartition, 
    cudaLong_offset, cudaShort_offset, 
    cudaLongGroupPartition, cudaShortGroupPartition, 
    cudaLongGroup_offset, cudaShortGroup_offset, 
    WindowOffset, Curwindow, t_Atomic, BlockOffset, Binary, ColumnIndice, t_Value,
    cuda_long_offset, cuda_long_row, cuda_long_atomic, cuda_long_column, cuda_long_value,
    cuda_short_offset, cuda_short_row, cuda_short_atomic, cuda_short_column, cuda_short_value,
		blockSize_h, blockSize_w, num_nodes, threshold, Short_len, t_s, c_s, max_blocks);

  //累加WindowOffset_tensor
	thrust::device_ptr<int> WindowOffset_ptr =
		thrust::device_pointer_cast(WindowOffset);
	thrust::device_vector<int> WindowOffset_vector(
		WindowOffset_ptr, WindowOffset_ptr + h_tc_group);
	thrust::device_ptr<int> WindowOffset_out_ptr =
	thrust::device_pointer_cast(WindowOffset_out_tensor.data<int>()+1);
	thrust::inclusive_scan(WindowOffset_vector.begin(), WindowOffset_vector.end(), WindowOffset_out_ptr);

    //   thrust::host_vector<int> host_data(WindowOffset_ptr, WindowOffset_ptr + h_tc_group);
    // // 打印主机数据
    // std::cout << "Contents of device pointer:" << std::endl;
    // // for (int i = 0; i < 20; ++i) {
    //     std::cout << host_data[50403] << " ";
    //     std::cout << host_data[50404] << " ";
    // // }
    // std::cout << std::endl;


  //累加BlockOffset_tensor
	thrust::device_ptr<int> BlockOffset_ptr =
		thrust::device_pointer_cast(BlockOffset);
	thrust::device_vector<int> BlockOffset_vector(
		BlockOffset_ptr, BlockOffset_ptr + h_blocks);
	thrust::device_ptr<int> BlockOffset_out_ptr =
	thrust::device_pointer_cast(BlockOffset_out_tensor.data<int>()+1);
	thrust::inclusive_scan(BlockOffset_vector.begin(), BlockOffset_vector.end(), BlockOffset_out_ptr);

  //累加cuda_long_offset_tensor
	thrust::device_ptr<int> cuda_long_offset_ptr =
		thrust::device_pointer_cast(cuda_long_offset);
	thrust::device_vector<int> cuda_long_offset_vector(
		cuda_long_offset_ptr, cuda_long_offset_ptr + h_cuda_long_group);
	thrust::device_ptr<int> cuda_long_offset_out_ptr =
	thrust::device_pointer_cast(cuda_long_offset_out_tensor.data<int>()+1);
	thrust::inclusive_scan(cuda_long_offset_vector.begin(), cuda_long_offset_vector.end(), cuda_long_offset_out_ptr);

  //累加cuda_short_offset_tensor
	thrust::device_ptr<int> cuda_short_offset_ptr =
		thrust::device_pointer_cast(cuda_short_offset);
	thrust::device_vector<int> cuda_short_offset_vector(
		cuda_short_offset_ptr, cuda_short_offset_ptr + h_cuda_short_group);
	thrust::device_ptr<int> cuda_short_offset_out_ptr =
	thrust::device_pointer_cast(cuda_short_offset_out_tensor.data<int>()+1);
	thrust::inclusive_scan(cuda_short_offset_vector.begin(), cuda_short_offset_vector.end(), cuda_short_offset_out_ptr);

	return std::make_tuple(
  WindowOffset_out_tensor, Curwindow_tensor, ColumnIndice_tensor, 
  BlockOffset_out_tensor, Binary_tensor,
  cuda_long_offset_out_tensor, cuda_long_row_tensor, cuda_long_column_tensor;
}
*/
void fill_edgeToRow_cuda(int *edgeToRow, int *nodePointer, int num_nodes) {
  int wrap_size = 32;
  int block_size = 1024;
  //每个warp负责一行的非零元素
  int grid_size = (num_nodes * wrap_size + block_size - 1) / block_size;
  fill_edgeToRow<<<grid_size, block_size>>>(edgeToRow, nodePointer, num_nodes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

