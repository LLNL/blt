#
#
#

cmake_minimum_required(VERSION 3.3)

project( pi_playground )

include(blt/SetupBLT.cmake)

#
# Example 1: Creating a simple executable.
#
blt_add_executable( NAME calc_pi_example_1
                    SOURCES calc_pi_example_1.cpp )



#
# Example 2: Creating a library and an executable using our library.
#
blt_add_library( NAME calc_pi_serial_lib
                 HEADERS calc_pi_serial_lib.hpp
                 SOURCES calc_pi_serial_lib.cpp )


blt_add_executable( NAME calc_pi_example_2
                    SOURCES calc_pi_example_2.cpp 
                    DEPENDS_ON calc_pi_serial_lib)

#
# Example 3: Creating an executable using gtest, using the executable via ctest.
#
blt_add_executable( NAME calc_pi_example_3_gtest
                    SOURCES calc_pi_example_3_gtest.cpp 
                    DEPENDS_ON calc_pi_serial_lib gtest)

blt_add_test( NAME calc_pi_example_3_gtest 
              COMMAND calc_pi_example_3_gtest)


if(MPI_FOUND)
    #
    # Example 4: Add mpi version of calc_pi, and expand example 3 to also test 
    # the mpi version.
    #

    blt_add_library( NAME calc_pi_mpi_lib
                     HEADERS calc_pi_mpi_lib.hpp
                     SOURCES calc_pi_mpi_lib.cpp 
                     DEPENDS_ON mpi)


    blt_add_executable( NAME calc_pi_example_4_gtest
                        SOURCES calc_pi_example_4_gtest.cpp 
                        DEPENDS_ON calc_pi_serial_lib calc_pi_mpi_lib gtest)

    blt_add_test( NAME calc_pi_example_4_gtest 
                  COMMAND calc_pi_example_4_gtest
                  NUM_PROCS 2) # number of mpi tasks to use
endif()
