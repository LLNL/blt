///////////////////////////////////////////////////////////////////////////////
//
// file: calc_pi_example_3_gtest.cpp
// 
// Simple example that calculates pi via a monte carlo method.
//
// Adapted from:
// https://www.mcs.anl.gov/research/projects/mpi/usingmpi/examples-usingmpi/simplempi/cpi_c.html
///////////////////////////////////////////////////////////////////////////////

#include <gtest/gtest.h>

#include "mpi.h"

#include "calc_pi_serial_lib.hpp"
#include "calc_pi_mpi_lib.hpp"

// test serial lib
TEST(calc_pi_tests, serial_example)
{
    double PI_REF = 3.141592653589793238462643;
    ASSERT_NEAR(calc_pi_serial(1000),PI_REF,1e-1);
}


// test mpi lib
TEST(calc_pi_tests, mpi_example)
{
    double PI_REF = 3.141592653589793238462643;
    ASSERT_NEAR(calc_pi_mpi(10000),PI_REF,1e-2);
}

// main driver
int main(int argc, char * argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    MPI_Init(&argc, &argv);

    result = RUN_ALL_TESTS();

    MPI_Finalize();

    return result;
}