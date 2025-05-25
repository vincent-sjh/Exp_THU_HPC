#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include "worker.h"
// 与前一个进程比较并排序
inline int sort_prev(float*& data, float*& recv_buf, float*& return_buf, int block_len, int prev_len, int rank) {
    if(prev_len == 0) return 0;  // 如果没有前一块数据，直接返回
    
    float prev_max;
    // 与前一个进程交换最大值，检查是否需要排序
    MPI_Sendrecv(&data[0], 1, MPI_FLOAT, rank - 1, 0, 
                 &prev_max, 1, MPI_FLOAT, rank - 1, 0, 
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    if(prev_max <= data[0]) return 0;  // 如果已经有序，无需继续
    
    // 非阻塞发送当前块到前一个进程
    MPI_Request req;
    MPI_Isend(data, block_len, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, &req);
    // 接收前一个进程的块
    MPI_Recv(recv_buf, prev_len, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // 使用指针优化合并排序，并在 MPI_Wait 前开始计算以重叠通信
    float* ptr_data = data + block_len - 1;
    float* ptr_recv = recv_buf + prev_len - 1;
    float* ptr_return = return_buf + block_len - 1;
    while(ptr_return >= return_buf) {
        if(ptr_data >= data && (ptr_recv < recv_buf || *ptr_data > *ptr_recv)) {
            *ptr_return-- = *ptr_data--;
        } else {
            *ptr_return-- = *ptr_recv--;
        }
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);  // 确保发送完成，此时计算已完成一部分或全部
    return 1;
}

// 与后一个进程比较并排序
inline int sort_next(float*& data, float*& recv_buf, float*& return_buf, int block_len, int next_len, int rank) {
    if(next_len == 0) return 0;  // 如果没有后一块数据，直接返回
    
    float next_min;
    // 与后一个进程交换最小值，检查是否需要排序
    MPI_Sendrecv(&data[block_len - 1], 1, MPI_FLOAT, rank + 1, 0,
                 &next_min, 1, MPI_FLOAT, rank + 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    if(data[block_len - 1] <= next_min) return 0;  // 如果已经有序，无需继续
    
    // 非阻塞发送当前块到后一个进程
    MPI_Request req;
    MPI_Isend(data, block_len, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, &req);
    // 接收后一个进程的块
    MPI_Recv(recv_buf, next_len, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // 使用指针优化合并排序，并在 MPI_Wait 前开始计算以重叠通信
    float* ptr_data = data;
    float* ptr_recv = recv_buf;
    float* ptr_return = return_buf;
    float* end_return = return_buf + block_len;
    while(ptr_return < end_return) {
        if(ptr_recv >= recv_buf + next_len) {
            while(ptr_return < end_return) {
                *ptr_return++ = *ptr_data++;
            }
        } else {
            *ptr_return++ = (*ptr_data < *ptr_recv) ? *ptr_data++ : *ptr_recv++;
        }
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);  // 确保发送完成，此时计算已完成一部分或全部
    return 1;
}

// 奇偶排序实现（保持不变，仅展示上下文）
void Worker::sort() {
    // 对本地块进行初始排序
    std::sort(data, data + block_len);
    if(nprocs == 1) return;  // 只有一个进程，无需并行排序
    if(out_of_range) return; // 当前进程没有数据需要排序
    
    const int block_size = ceiling(n, nprocs);  // 计算块大小
    const int worker_num = ceiling(n, block_size);  // 计算活跃工作进程数
    const int prev_size = rank > 0 ? block_size : 0;  // 前一块的大小
    const int next_size = (rank >= worker_num - 1) ? 0 : 
                         (rank == worker_num - 2 ? n - (worker_num - 1) * block_size : block_size);
    bool is_even = !(rank & 1);  // 判断当前进程是奇数还是偶数
    
    if(block_len == 0) return;
    // 为合并操作分配临时缓冲区
    float* return_buf = new float[block_size * 2];
    float* recv_buf = return_buf + block_size;

    // 奇偶排序阶段
    for(int i = 0; i < worker_num; i += 2) {
        if(is_even) {
            if(sort_next(data, recv_buf, return_buf, block_len, next_size, rank)) 
                std::swap(data, return_buf);
            if(sort_prev(data, recv_buf, return_buf, block_len, prev_size, rank)) 
                std::swap(data, return_buf);
        } else {
            if(sort_prev(data, recv_buf, return_buf, block_len, prev_size, rank)) 
                std::swap(data, return_buf);
            if(sort_next(data, recv_buf, return_buf, block_len, next_size, rank)) 
                std::swap(data, return_buf);
        }
    }
    
    delete[] return_buf;  // 释放内存
}