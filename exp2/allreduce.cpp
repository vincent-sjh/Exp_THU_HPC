#include <chrono>
#include <iostream>
#include <mpi.h>
#include <time.h>
#include <cstring>
#include <cmath>
#include <algorithm>

#define EPS 1e-5

namespace ch = std::chrono;

void Ring_Allreduce(void* sendbuf, void* recvbuf, int count, MPI_Comm communicator, int comm_size, int my_rank) {
    // Type casting buffers to float pointers
    auto* send_buffer = static_cast<float*>(sendbuf);
    auto* recv_buffer = static_cast<float*>(recvbuf);
    
    // Calculate neighboring ranks and block size
    int previous_rank = (my_rank - 1 + comm_size) % comm_size;
    int next_rank = (my_rank + 1) % comm_size;
    int chunk_size = count / comm_size;

    // Helper lambda to calculate buffer offset
    auto get_buffer_chunk = [&](int offset_rank) -> float* {
        return send_buffer + (offset_rank * chunk_size);
    };

    // Phase 1: Reduce-scatter
    for (int step = 0; step < comm_size - 1; ++step) {
        int send_rank = (my_rank - step + comm_size) % comm_size;
        int recv_rank = (my_rank - step - 1 + comm_size) % comm_size;
        
        float* send_chunk = get_buffer_chunk(send_rank);
        float* recv_chunk = get_buffer_chunk(recv_rank);
        
        // Send to next, receive from previous, and accumulate
        MPI_Sendrecv(send_chunk, chunk_size, MPI_FLOAT, next_rank, 0,
                    recv_buffer, chunk_size, MPI_FLOAT, previous_rank, 0,
                    communicator, MPI_STATUS_IGNORE);
                    
        for (int idx = 0; idx < chunk_size; ++idx) {
            recv_chunk[idx] += recv_buffer[idx];
        }
    }

    // Phase 2: All-gather
    for (int step = 0; step < comm_size - 1; ++step) {
        int send_rank = (my_rank + 1 - step + comm_size) % comm_size;
        int recv_rank = (my_rank - step + comm_size) % comm_size;
        
        float* send_chunk = get_buffer_chunk(send_rank);
        float* recv_chunk = get_buffer_chunk(recv_rank);
        
        // Send to next, receive from previous, and copy
        MPI_Sendrecv(send_chunk, chunk_size, MPI_FLOAT, next_rank, 0,
                    recv_buffer, chunk_size, MPI_FLOAT, previous_rank, 0,
                    communicator, MPI_STATUS_IGNORE);
                    
        for (int idx = 0; idx < chunk_size; ++idx) {
            recv_chunk[idx] = recv_buffer[idx];
        }
    }

    // Final copy to output buffer
    std::memcpy(recv_buffer, send_buffer, count * sizeof(float));
}

// reduce + bcast
void Naive_Allreduce(void* sendbuf, void* recvbuf, int n, MPI_Comm comm, int comm_sz, int my_rank)
{
    MPI_Reduce(sendbuf, recvbuf, n, MPI_FLOAT, MPI_SUM, 0, comm);
    MPI_Bcast(recvbuf, n, MPI_FLOAT, 0, comm);
}

int main(int argc, char *argv[])
{
    int ITER = atoi(argv[1]);
    int n = atoi(argv[2]);
    float* mpi_sendbuf = new float[n];
    float* mpi_recvbuf = new float[n];
    float* naive_sendbuf = new float[n];
    float* naive_recvbuf = new float[n];
    float* ring_sendbuf = new float[n];
    float* ring_recvbuf = new float[n];

    MPI_Init(nullptr, nullptr);
    int comm_sz;
    int my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    srand(time(NULL) + my_rank);
    for (int i = 0; i < n; ++i)
        mpi_sendbuf[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    memcpy(naive_sendbuf, mpi_sendbuf, n * sizeof(float));
    memcpy(ring_sendbuf, mpi_sendbuf, n * sizeof(float));

    //warmup and check
    MPI_Allreduce(mpi_sendbuf, mpi_recvbuf, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    Naive_Allreduce(naive_sendbuf, naive_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
    Ring_Allreduce(ring_sendbuf, ring_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
    bool correct = true;
    for (int i = 0; i < n; ++i)
        if (abs(mpi_recvbuf[i] - ring_recvbuf[i]) > EPS)
        {
            correct = false;
            break;
        }

    if (correct)
    {
        auto beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            MPI_Allreduce(mpi_sendbuf, mpi_recvbuf, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        auto end = ch::high_resolution_clock::now();
        double mpi_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms

        beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            Naive_Allreduce(naive_sendbuf, naive_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
        end = ch::high_resolution_clock::now();
        double naive_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms

        beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            Ring_Allreduce(ring_sendbuf, ring_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
        end = ch::high_resolution_clock::now();
        double ring_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms
        
        if (my_rank == 0)
        {
            std::cout << "Correct." << std::endl;
            std::cout << "MPI_Allreduce:   " << mpi_dur << " ms." << std::endl;
            std::cout << "Naive_Allreduce: " << naive_dur << " ms." << std::endl;
            std::cout << "Ring_Allreduce:  " << ring_dur << " ms." << std::endl;
        }
    }
    else
        if (my_rank == 0)
            std::cout << "Wrong!" << std::endl;

    delete[] mpi_sendbuf;
    delete[] mpi_recvbuf;
    delete[] naive_sendbuf;
    delete[] naive_recvbuf;
    delete[] ring_sendbuf;
    delete[] ring_recvbuf;
    MPI_Finalize();
    return 0;
}
