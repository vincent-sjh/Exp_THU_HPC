#include "apsp.h"

#define TILE_SIZE 32
#define MATRIX_SIZE (TILE_SIZE * 2) // 矩阵块大小为 64
#define UNREACHABLE 0x3fffffff      // 表示不可达的距离值

// 设备函数：加载数据到共享内存
__device__ void load_to_shared(int dim, int global_row, int global_col, int local_row, int local_col,
                               int *matrix, int shared[MATRIX_SIZE][MATRIX_SIZE]) {
    bool valid = global_row < dim && global_col < dim;
    shared[local_row][local_col] = valid ? matrix[global_row * dim + global_col] : UNREACHABLE;
}

// 设备函数：计算 Floyd-Warshall 更新
__device__ void floyd_warshall_update(int &val, int row, int col, int pivot,
                                     int shared1[MATRIX_SIZE][MATRIX_SIZE], int shared2[MATRIX_SIZE][MATRIX_SIZE]) {
    val = min(val, shared1[row][pivot] + shared2[pivot][col]);
}

// 设备函数：写回全局内存
__device__ void write_to_global(int dim, int global_row, int global_col, int val, int *matrix) {
    if (global_row < dim && global_col < dim) {
        matrix[global_row * dim + global_col] = val;
    }
}

// 内核 1：处理中心（枢轴）块
__global__ void kernel_center(int dim, int step, int *matrix) {
    __shared__ int shared_tile[MATRIX_SIZE][MATRIX_SIZE];

    // 线程索引
    const int row0 = threadIdx.y;
    const int col0 = threadIdx.x;
    const int row1 = row0 + TILE_SIZE;
    const int col1 = col0 + TILE_SIZE;

    const int abs_basic = step * MATRIX_SIZE;
    const int abs_row0 = abs_basic + row0;
    const int abs_col0 = abs_basic + col0;
    const int abs_row1 = abs_basic + row1;
    const int abs_col1 = abs_basic + col1;

    // 加载数据到共享内存
    load_to_shared(dim, abs_row0, abs_col0, row0, col0, matrix, shared_tile);
    load_to_shared(dim, abs_row0, abs_col1, row0, col1, matrix, shared_tile);
    load_to_shared(dim, abs_row1, abs_col0, row1, col0, matrix, shared_tile);
    load_to_shared(dim, abs_row1, abs_col1, row1, col1, matrix, shared_tile);

    __syncthreads();

    // Floyd-Warshall 更新
    int val_00 = shared_tile[row0][col0];
    int val_01 = shared_tile[row0][col1];
    int val_10 = shared_tile[row1][col0];
    int val_11 = shared_tile[row1][col1];

    #pragma unroll 64
    for (int pivot = 0; pivot < MATRIX_SIZE; ++pivot) {
        floyd_warshall_update(val_00, row0, col0, pivot, shared_tile, shared_tile);
        floyd_warshall_update(val_01, row0, col1, pivot, shared_tile, shared_tile);
        floyd_warshall_update(val_10, row1, col0, pivot, shared_tile, shared_tile);
        floyd_warshall_update(val_11, row1, col1, pivot, shared_tile, shared_tile);
    }

    // 写回全局内存
    write_to_global(dim, abs_row0, abs_col0, val_00, matrix);
    write_to_global(dim, abs_row0, abs_col1, val_01, matrix);
    write_to_global(dim, abs_row1, abs_col0, val_10, matrix);
    write_to_global(dim, abs_row1, abs_col1, val_11, matrix);
}

// 内核 2：处理行和列块
__global__ void kernel_row_col(int dim, int step, int *matrix) {
    __shared__ int pivot_block[MATRIX_SIZE][MATRIX_SIZE];
    __shared__ int active_block[MATRIX_SIZE][MATRIX_SIZE];

    // 线程索引
    const int r0 = threadIdx.y;
    const int c0 = threadIdx.x;
    const int r1 = r0 + TILE_SIZE;
    const int c1 = c0 + TILE_SIZE;

    const int pivot_basic = step * MATRIX_SIZE;
    const int pivot_r0 = pivot_basic + r0;
    const int pivot_c0 = pivot_basic + c0;
    const int pivot_r1 = pivot_basic + r1;
    const int pivot_c1 = pivot_basic + c1;

    // 加载枢轴块数据
    load_to_shared(dim, pivot_r0, pivot_c0, r0, c0, matrix, pivot_block);
    load_to_shared(dim, pivot_r0, pivot_c1, r0, c1, matrix, pivot_block);
    load_to_shared(dim, pivot_r1, pivot_c0, r1, c0, matrix, pivot_block);
    load_to_shared(dim, pivot_r1, pivot_c1, r1, c1, matrix, pivot_block);

    // 计算当前块索引
    int active_r0, active_c0, active_r1, active_c1;
    if (blockIdx.y) { // 行块
        active_r0 = pivot_r0;
        active_c0 = blockIdx.x * MATRIX_SIZE + c0;
        active_r1 = pivot_r1;
        active_c1 = blockIdx.x * MATRIX_SIZE + c1;
    } else { // 列块
        active_r0 = blockIdx.x * MATRIX_SIZE + r0;
        active_c0 = pivot_c0;
        active_r1 = blockIdx.x * MATRIX_SIZE + r1;
        active_c1 = pivot_c1;
    }

    // 加载当前块数据
    load_to_shared(dim, active_r0, active_c0, r0, c0, matrix, active_block);
    load_to_shared(dim, active_r0, active_c1, r0, c1, matrix, active_block);
    load_to_shared(dim, active_r1, active_c0, r1, c0, matrix, active_block);
    load_to_shared(dim, active_r1, active_c1, r1, c1, matrix, active_block);

    __syncthreads();

    // Floyd-Warshall 更新
    int dist_00 = active_block[r0][c0];
    int dist_01 = active_block[r0][c1];
    int dist_10 = active_block[r1][c0];
    int dist_11 = active_block[r1][c1];

    #pragma unroll 64
    for (int pivot = 0; pivot < MATRIX_SIZE; pivot++) {
        if (blockIdx.y == 1) { // 行块
            floyd_warshall_update(dist_00, r0, c0, pivot, pivot_block, active_block);
            floyd_warshall_update(dist_01, r0, c1, pivot, pivot_block, active_block);
            floyd_warshall_update(dist_10, r1, c0, pivot, pivot_block, active_block);
            floyd_warshall_update(dist_11, r1, c1, pivot, pivot_block, active_block);
        } else { // 列块
            floyd_warshall_update(dist_00, r0, c0, pivot, active_block, pivot_block);
            floyd_warshall_update(dist_01, r0, c1, pivot, active_block, pivot_block);
            floyd_warshall_update(dist_10, r1, c0, pivot, active_block, pivot_block);
            floyd_warshall_update(dist_11, r1, c1, pivot, active_block, pivot_block);
        }
    }

    // 写回全局内存
    write_to_global(dim, active_r0, active_c0, dist_00, matrix);
    write_to_global(dim, active_r0, active_c1, dist_01, matrix);
    write_to_global(dim, active_r1, active_c0, dist_10, matrix);
    write_to_global(dim, active_r1, active_c1, dist_11, matrix);
}

// 内核 3：处理外围块
__global__ void kernel_peripheral(int dim, int step, int *matrix) {
    __shared__ int row_tile[MATRIX_SIZE][MATRIX_SIZE];
    __shared__ int col_tile[MATRIX_SIZE][MATRIX_SIZE];

    // 线程索引
    const int r0 = threadIdx.y;
    const int c0 = threadIdx.x;
    const int r1 = r0 + TILE_SIZE;
    const int c1 = c0 + TILE_SIZE;

    // 当前块全局索引
    const int curr_r0 = blockIdx.y * MATRIX_SIZE + r0;
    const int curr_c0 = blockIdx.x * MATRIX_SIZE + c0;
    const int curr_r1 = blockIdx.y * MATRIX_SIZE + r1;
    const int curr_c1 = blockIdx.x * MATRIX_SIZE + c1;

    // 枢轴行和列索引
    const int pivot_basic = step * MATRIX_SIZE;
    const int pivot_row0 = pivot_basic + r0;
    const int pivot_col0 = pivot_basic + c0;
    const int pivot_row1 = pivot_basic + r1;
    const int pivot_col1 = pivot_basic + c1;

    // 加载枢轴行和列数据
    load_to_shared(dim, pivot_row0, curr_c0, r0, c0, matrix, row_tile);
    load_to_shared(dim, pivot_row0, curr_c1, r0, c1, matrix, row_tile);
    load_to_shared(dim, pivot_row1, curr_c0, r1, c0, matrix, row_tile);
    load_to_shared(dim, pivot_row1, curr_c1, r1, c1, matrix, row_tile);

    load_to_shared(dim, curr_r0, pivot_col0, r0, c0, matrix, col_tile);
    load_to_shared(dim, curr_r0, pivot_col1, r0, c1, matrix, col_tile);
    load_to_shared(dim, curr_r1, pivot_col0, r1, c0, matrix, col_tile);
    load_to_shared(dim, curr_r1, pivot_col1, r1, c1, matrix, col_tile);

    __syncthreads();

    // 加载当前块数据
    int path_00 = (curr_r0 < dim && curr_c0 < dim) ? matrix[curr_r0 * dim + curr_c0] : UNREACHABLE;
    int path_01 = (curr_r0 < dim && curr_c1 < dim) ? matrix[curr_r0 * dim + curr_c1] : UNREACHABLE;
    int path_10 = (curr_r1 < dim && curr_c0 < dim) ? matrix[curr_r1 * dim + curr_c0] : UNREACHABLE;
    int path_11 = (curr_r1 < dim && curr_c1 < dim) ? matrix[curr_r1 * dim + curr_c1] : UNREACHABLE;

    // Floyd-Warshall 更新
    #pragma unroll 64
    for (int pivot = 0; pivot < MATRIX_SIZE; pivot++) {
        floyd_warshall_update(path_00, r0, c0, pivot, col_tile, row_tile);
        floyd_warshall_update(path_01, r0, c1, pivot, col_tile, row_tile);
        floyd_warshall_update(path_10, r1, c0, pivot, col_tile, row_tile);
        floyd_warshall_update(path_11, r1, c1, pivot, col_tile, row_tile);
    }

    // 写回全局内存
    write_to_global(dim, curr_r0, curr_c0, path_00, matrix);
    write_to_global(dim, curr_r0, curr_c1, path_01, matrix);
    write_to_global(dim, curr_r1, curr_c0, path_10, matrix);
    write_to_global(dim, curr_r1, curr_c1, path_11, matrix);
}

// 主机函数：执行内核以计算全源最短路径
void apsp(int n, int *graph) {
    const int num_blocks = (n + MATRIX_SIZE - 1) / MATRIX_SIZE;
    dim3 thread_block(TILE_SIZE, TILE_SIZE);
    dim3 grid_center(1, 1);
    dim3 grid_row_col(num_blocks, 2);
    dim3 grid_peripheral(num_blocks, num_blocks);

    for (int k = 0; k < num_blocks; k++) {
        kernel_center<<<grid_center, thread_block>>>(n, k, graph);
        kernel_row_col<<<grid_row_col, thread_block>>>(n, k, graph);
        kernel_peripheral<<<grid_peripheral, thread_block>>>(n, k, graph);
    }
}