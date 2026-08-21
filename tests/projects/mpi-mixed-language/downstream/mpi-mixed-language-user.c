#include <mpi.h>

#ifndef BLT_FAKE_MPI_C_HEADER
#error "Expected the fake C MPI include directory"
#endif

#ifndef BLT_FAKE_MPI_C_COMPILE_OPTION
#error "Expected the fake C MPI compile option"
#endif

#ifndef BLT_FAKE_MPI_C_COMPILE_DEFINITION
#error "Expected the fake C MPI compile definition"
#endif

#if defined(BLT_FAKE_MPI_CXX_COMPILE_OPTION) ||                                \
    defined(BLT_FAKE_MPI_CXX_COMPILE_DEFINITION)
#error "Unexpected fake CXX MPI compile property in C compilation"
#endif

void blt_fake_mpi_c(void) {}
