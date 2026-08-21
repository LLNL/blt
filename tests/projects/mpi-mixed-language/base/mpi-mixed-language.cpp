#include <mpi.h>

#ifndef BLT_FAKE_MPI_CXX_HEADER
#error "Expected the fake CXX MPI include directory"
#endif

#ifndef BLT_FAKE_MPI_CXX_COMPILE_OPTION
#error "Expected the fake CXX MPI compile option"
#endif

#ifndef BLT_FAKE_MPI_CXX_COMPILE_DEFINITION
#error "Expected the fake CXX MPI compile definition"
#endif

#ifndef BLT_FAKE_MPI_CXX_WRAPPED_OPTION
#error "Expected the CMake FindMPI-style CXX/CUDA compile option"
#endif

#if defined(BLT_FAKE_MPI_C_COMPILE_OPTION) ||                                  \
    defined(BLT_FAKE_MPI_C_COMPILE_DEFINITION)
#error "Unexpected fake C MPI compile property in CXX compilation"
#endif

void blt_mpi_mixed_language() {
  MPI_Init(nullptr, nullptr);
  MPI_Finalize();
}
