///////////////////////////////////////////////////////////////////////////////
//
// file: test_2.cpp
// 
// Simple example that calculates pi via simple integration.
//
// Adapted from:
// https://www.mcs.anl.gov/research/projects/mpi/usingmpi/examples-usingmpi/simplempi/cpi_c.html
///////////////////////////////////////////////////////////////////////////////

#include <gtest/gtest.h>

#include "mpi.h"

#include "calc_pi.hpp"
#include "calc_pi_mpi.hpp"

// test serial lib
TEST(calc_pi_tests, serial_example)
{
    double PI_REF = 3.141592653589793238462643;
    ASSERT_NEAR(calc_pi(1000),PI_REF,1e-1);
}


// test mpi lib
TEST(calc_pi_tests, mpi_example)
{
    int num_tasks = 0;
    int task_id   = 0;
    
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);

    double PI_REF = 3.141592653589793238462643;
    
    double res = calc_pi_mpi(10000);

    if(task_id == 0)
    {
      ASSERT_NEAR(res,PI_REF,1e-2);
    }
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
