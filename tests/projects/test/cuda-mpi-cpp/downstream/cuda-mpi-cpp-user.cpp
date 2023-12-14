#include <iostream>
#include <mpi.h>

#include "cuda-mpi-cpp.cuh"

int main() {
    MPI_Init(nullptr, nullptr);

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len = 0;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    std::cout << "Hello world from processor " << processor_name
              << ", rank "  << world_rank
              << " out of " << world_size
              << " processors" << std::endl;

    run_kernel();

    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}
