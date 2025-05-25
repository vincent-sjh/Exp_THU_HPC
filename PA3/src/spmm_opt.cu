#include "spmm_opt.h"
#include <vector>
#include <cassert>

const int SLICE_SIZE = 256; // 固定切片大小
const int num_streams = 1; // cuda流数目

// 辅助函数：加载 CSR 数据到共享内存
__device__ void csr_sharer(int thread_id, int begin_idx, int slice_len, int* col_ids, float* weights, int* col_cache, float* weight_cache, int block_size) {
    for (int idx = thread_id; idx < slice_len; idx += block_size) {
        if (idx < slice_len) {
            col_cache[idx] = col_ids[begin_idx + idx];
            weight_cache[idx] = weights[begin_idx + idx];
        }
    }
}


// 辅助函数：执行矩阵乘法并原子更新
__device__ void spmm_computer(int thread_id, int row_idx, int slice_len, int* col_cache, float* weight_cache, float* input_vec, float* output_vec, int dim) {
    float acc = 0.00f;
    for (int i = 0; i < slice_len; ++i) {
        int index = col_cache[i] * dim + thread_id;
        // 计算累加和
        acc += input_vec[index] * weight_cache[i];
    }
    int row_index = row_idx * dim + thread_id;
    atomicAdd(&output_vec[ row_index], acc);
    return;
}

// CUDA 内核：计算稀疏矩阵乘法
__global__ void spmm_kernel_opt(int* row_indices, int* slice_begins, int* slice_ends, int total_slices, int* col_ids, float* weights, float* input_vec, float* output_vec, int dim) {
    int slice_idx = blockIdx.x;
    int tid = threadIdx.x;

    // 检查索引有效性
    if (slice_idx >= total_slices || tid >= dim) return;

    // 获取切片信息
    int row_idx = row_indices[slice_idx];
    int begin_idx = slice_begins[slice_idx];
    int end_idx = slice_ends[slice_idx];
    int slice_len = end_idx - begin_idx;

    // 分配共享内存
    extern __shared__ float shared_data[];
    float* weight_cache = shared_data;
    int* col_cache = (int*)(weight_cache + slice_len);

    // 加载 CSR 数据
    csr_sharer(tid, begin_idx, slice_len, col_ids, weights, col_cache, weight_cache, blockDim.x);
    __syncthreads();
    // 执行计算
    spmm_computer(tid, row_idx, slice_len, col_cache, weight_cache, input_vec, output_vec, dim);
}

void SpMMOpt::preprocess(float* input_vec, float* output_vec) {
    std::vector<int> row_indices, slice_begins, slice_ends, slice_offsets;
    std::vector<int> row_ptr(num_v + 1);
    checkCudaErrors(cudaMemcpy(row_ptr.data(), d_ptr, sizeof(int) * (num_v + 1), cudaMemcpyDeviceToHost));

    // 动态切片大小
    int avg_nnz_per_row = num_e / num_v;
    int dynamic_slice_size = max(256, min(SLICE_SIZE, avg_nnz_per_row));

    for (int r = 0; r < num_v; r++) {
        int start = row_ptr[r], end = row_ptr[r + 1];
        while (start < end) {
            row_indices.push_back(r);
            slice_begins.push_back(start);
            slice_ends.push_back(std::min(start + dynamic_slice_size, end));
            slice_offsets.push_back(start);
            start += dynamic_slice_size;
        }
    }
    slice_offsets.push_back(num_e);
    slice_total = row_indices.size();

    // 分配和拷贝设备内存（不变）
    checkCudaErrors(cudaMalloc(&d_row_indices, sizeof(int) * slice_total));
    checkCudaErrors(cudaMalloc(&d_slice_begins, sizeof(int) * slice_total));
    checkCudaErrors(cudaMalloc(&d_slice_ends, sizeof(int) * slice_total));
    checkCudaErrors(cudaMalloc(&d_slice_offsets, sizeof(int) * (slice_total + 1)));
    checkCudaErrors(cudaMemcpy(d_row_indices, row_indices.data(), sizeof(int) * slice_total, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_slice_begins, slice_begins.data(), sizeof(int) * slice_total, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_slice_ends, slice_ends.data(), sizeof(int) * slice_total, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_slice_offsets, slice_offsets.data(), sizeof(int) * (slice_total + 1), cudaMemcpyHostToDevice));

    grid = dim3(slice_total, 1, 1);
    block = dim3(feat_in, 1, 1);
}


void SpMMOpt::run(float* input_vec, float* output_vec) {
    cudaStream_t streams[num_streams];
    int slices_per_stream = (slice_total + num_streams - 1) / num_streams;

    // 创建流
    for (int i = 0; i < num_streams; ++i) {
        checkCudaErrors(cudaStreamCreate(&streams[i]));
    }

    // 分配共享内存大小
    size_t shared_size = SLICE_SIZE * (sizeof(float) + sizeof(int));

    // 启动内核
    for (int i = 0; i < num_streams; ++i) {
        int start_slice = i * slices_per_stream;
        int end_slice = min(start_slice + slices_per_stream, slice_total);
        if (start_slice < slice_total) {
            dim3 grid(end_slice - start_slice, 1, 1);
            spmm_kernel_opt<<<grid, block, shared_size, streams[i]>>>(
                d_row_indices + start_slice, 
                d_slice_begins + start_slice, 
                d_slice_ends + start_slice, 
                end_slice - start_slice, 
                d_idx, d_val, input_vec, output_vec, feat_in
            );
        }
    }

    // 同步流
    for (int i = 0; i < num_streams; ++i) {
        checkCudaErrors(cudaStreamSynchronize(streams[i]));
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }
}

// // 执行：启动 SpMM
// void SpMMOpt::run(float* input_vec, float* output_vec) {
//     // 计算共享内存大小并启动内核
//     spmm_kernel_opt<<<grid, block, SLICE_SIZE * (sizeof(float) + sizeof(int))>>>(d_row_indices, d_slice_begins, d_slice_ends, slice_total, d_idx, d_val, input_vec, output_vec, feat_in);
//     checkCudaErrors(cudaDeviceSynchronize());
// }

