#
#
#

cmake_minimum_required(VERSION 3.3)

project( pi_playground )

include(blt/SetupBLT.cmake)

#
# Example 1: Creating a simple executable.
#
blt_add_executable( NAME example_1
                    SOURCES example_1.cpp )



#
# Example 2: Creating a library and an executable using our library.
#
blt_add_library( NAME    calc_pi
                 HEADERS calc_pi.hpp
                 SOURCES calc_pi.cpp )


blt_add_executable( NAME example_2
                    SOURCES example_2.cpp 
                    DEPENDS_ON calc_pi)

#
# Test 1: Creating an executable using gtest, using the executable via ctest.
#
blt_add_executable( NAME test_1
                    SOURCES test_1.cpp 
                    DEPENDS_ON calc_pi gtest)

blt_add_test( NAME test_1 
              COMMAND test_1)


if(MPI_FOUND)
    #
    # Test 2: Add mpi version of calc_pi, and expand test 1 to also test 
    # the mpi version.
    #

    blt_add_library( NAME calc_pi_mpi
                     HEADERS calc_pi_mpi.hpp
                     SOURCES calc_pi_mpi.cpp 
                     DEPENDS_ON mpi)


    blt_add_executable( NAME test_2
                        SOURCES test_2.cpp 
                        DEPENDS_ON calc_pi calc_pi_mpi gtest)

    blt_add_test( NAME test_2 
                  COMMAND test_2
                  NUM_PROCS 2) # number of mpi tasks to use
endif()


if(CUDA_FOUND)
    #
    # Test 3: Add cuda version of calc_pi, and expand test 1 to also test 
    # the cuda version.
    #
 
    # atomicAdd for doubles requires compute_60 or greater
    # we may need to take another path ...
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_60,code=sm_60)
	
    blt_add_library( NAME calc_pi_cuda
                     HEADERS calc_pi_cuda.hpp
                     SOURCES calc_pi_cuda.cpp 
                     DEPENDS_ON cuda)


    blt_add_executable( NAME test_3
                        SOURCES test_3.cpp 
                        DEPENDS_ON calc_pi calc_pi_cuda gtest cuda_runtime)

    blt_add_test( NAME test_3
                  COMMAND test_3)
endif()

