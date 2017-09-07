///////////////////////////////////////////////////////////////////////////////
//
// file: calc_pi_example_3_gtest.cpp
// 
// Simple example that calculates pi via a monte carlo method.
//
// Adapted from:
// https://www.mcs.anl.gov/research/projects/mpi/usingmpi/examples-usingmpi/simplempi/cpi_c.html
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cstdlib>
#include <math.h>

#include <gtest/gtest.h>

#include "calc_pi_serial_lib.hpp"


TEST(calc_pi_tests, serial_example)
{
    double PI_REF = 3.141592653589793238462643;
    ASSERT_NEAR(calc_pi_serial(1000),PI_REF,1e-1);
}
