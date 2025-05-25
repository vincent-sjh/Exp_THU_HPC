# 小作业二：MPI Allreduce 实验报告 

宋建昊 2022010853

## 算法目标
每个节点的初始数据存放在`sendbuf`中，希望计算出每个进程的`sendbuf`向量之和。

## 算法实现
```cpp
int previous_rank = (my_rank - 1 + comm_size) % comm_size;
int next_rank = (my_rank + 1) % comm_size;
int chunk_size = count / comm_size;
```
- 计算环形拓扑中的邻居秩
- 使用模运算实现环形连接（例如，进程0的前一个是comm_size-1）
- 将数据分成每个进程处理的相等块

```cpp
auto get_buffer_chunk = [&](int offset_rank) -> float* {
    return send_buffer + (offset_rank * chunk_size);
};
```
### 算法阶段

1. **阶段1：Reduce-Scatter（分散归约）**
```cpp
for (int step = 0; step < comm_size - 1; ++step) {
    int send_rank = (my_rank - step + comm_size) % comm_size;
    int recv_rank = (my_rank - step - 1 + comm_size) % comm_size;
    // ...
```
- 执行comm_size-1步
- 每个进程向next_rank发送数据块，从previous_rank接收数据块
- 通过累加接收到的值来归约结果

循环中的关键操作：
- `MPI_Sendrecv`：组合发送和接收操作
- 累加操作：`recv_chunk[idx] += recv_buffer[idx]`
- 数据在环中传递，逐步累加部分和

2. **阶段2：All-Gather（全收集）**
```cpp
for (int step = 0; step < comm_size - 1; ++step) {
    int send_rank = (my_rank + 1 - step + comm_size) % comm_size;
    int recv_rank = (my_rank - step + comm_size) % comm_size;
    // ...
```
- 同样执行comm_size-1步
- 将归约后的结果分发给所有进程
- 每个进程将其数据块转发给邻居

关键操作：
- `MPI_Sendrecv`：发送和接收数据块
- 复制操作：`recv_chunk[idx] = recv_buffer[idx]`
- 将最终值在环中传播

## 测试

```
Correct.
MPI_Allreduce:   2518 ms.
Naive_Allreduce: 4712.43 ms.
Ring_Allreduce:  2575.95 ms.
```

