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
# Example 3: Creating an executable using gtest, using the executable via ctest.
#
blt_add_executable( NAME test_1
                    SOURCES test_1.cpp 
                    DEPENDS_ON calc_pi gtest)

blt_add_test( NAME test_1 
              COMMAND test_1)


if(MPI_FOUND)
    #
    # Example 4: Add mpi version of calc_pi, and expand example 3 to also test 
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
