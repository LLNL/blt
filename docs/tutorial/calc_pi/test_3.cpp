///////////////////////////////////////////////////////////////////////////////
//
// file: test_3.cpp
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

const double PI_REF = 3.141592653589793238462643;

// test serial lib
TEST(calc_pi_tests, serial_example)
{
    ASSERT_NEAR(calc_pi(1000),PI_REF,1e-1);
}


// test cuda lib
TEST(calc_pi_tests, cuda_example)
{
    ASSERT_NEAR(calc_pi_cuda(1000),PI_REF,1e-1);
}
