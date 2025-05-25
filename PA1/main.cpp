// 包含标准库和MPI库，用于算法、断言、输入输出、时间测量等功能
#include <algorithm>    // 提供排序等算法函数
#include <cassert>      // 提供断言功能，用于调试
#include <cstdio>       // 提供C风格的输入输出函数，如printf
#include <cstdlib>      // 提供通用工具函数，如atoi
#include <mpi.h>        // MPI并行计算库，用于多进程通信
#include <sys/time.h>   // 提供时间测量函数，如gettimeofday
#include "worker.h"     // 自定义头文件，包含Worker类的定义

// 主函数，程序入口，接受命令行参数
int main(int argc, char **argv) {
  // 初始化MPI环境，必须在MPI程序开始时调用
  MPI_Init(&argc, &argv);

  // 定义进程数量和当前进程的排名
  int nprocs, rank;
  // 获取MPI通信组中的进程总数，存储到nprocs
  CHKERR(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
  // 获取当前进程的排名（从0开始），存储到rank
  CHKERR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  // 检查命令行参数数量是否正确（需要3个：程序名、数字个数、输入文件名）
  if (argc != 3) {
    // 如果参数数量不对，且当前进程是rank 0（主进程），打印使用说明
    if (!rank)
      printf("Usage: ./odd_even_sort <number_count> <input_file>\n");
    // 结束MPI环境并退出程序
    MPI_Finalize();
    return 1;  // 返回1表示错误退出
  }

  // 从命令行参数获取要排序的数字个数（n）和输入文件名
  const int n = atoi(argv[1]);         // 将第一个参数转换为整数，表示数字个数
  const char *input_name = argv[2];    // 第二个参数是输入文件名

  // 如果数字个数小于进程数，认为任务太小，不值得并行处理，直接退出
  if (n < nprocs) {
    MPI_Finalize();  // 清理MPI环境
    return 0;        // 正常退出
  }

  // 创建Worker对象，负责排序任务，传入数字个数、进程数和当前进程排名
  Worker *worker = new Worker(n, nprocs, rank);

  /** 从输入文件读取数据 */
  worker->input(input_name);  // 调用Worker的input方法读取输入文件中的数据

  /** 对列表（输入数据）进行排序 */
  timeval start, end;         // 定义时间变量，用于测量排序耗时
  unsigned long time;         // 存储时间差（微秒）
  // 在所有进程间同步，确保计时起点一致
  MPI_Barrier(MPI_COMM_WORLD);
  // 获取排序开始前的当前时间
  gettimeofday(&start, NULL);

  // 执行排序操作，调用Worker类的sort方法（核心并行排序逻辑）
  worker->sort();

  // 在所有进程间再次同步，确保所有进程完成排序后再计时
  MPI_Barrier(MPI_COMM_WORLD);
  // 获取排序完成后的当前时间
  gettimeofday(&end, NULL);
  // 计算排序耗时（单位：微秒），转换为毫秒后存储
  time = 1000000.0 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;

  /** 检查排序结果是否正确 */
  int ret = worker->check();  // 调用Worker的check方法验证排序结果
  if (ret > 0) {
    // 如果检查通过，当前进程打印“pass”
    printf("Rank %d: pass\n", rank);
  } else {
    // 如果检查失败，当前进程打印“failed”
    printf("Rank %d: failed\n", rank);
  }

  // 在所有进程间同步，确保所有检查结果输出后再继续
  MPI_Barrier(MPI_COMM_WORLD);
  // 如果是rank 0（主进程），打印排序的总执行时间（单位：毫秒）
  if (rank == 0) {
    printf("Execution time of function sort is %lf ms.\n", time / 1000.0);
  }

#ifndef NDEBUG
  // 如果未定义NDEBUG（调试模式），每个进程打印“finalize”信息
  //printf("Process %d: finalize\n", rank);
#endif

  // 清理MPI环境，结束并行计算
  MPI_Finalize();
  return 0;  // 正常退出程序
}