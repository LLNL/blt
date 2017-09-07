///////////////////////////////////////////////////////////////////////////////
//
// file: calc_pi_exe.cpp
// 
// Simple example that calculates pi via a monte carlo method.
//
// Adapted from:
// https://www.mcs.anl.gov/research/projects/mpi/usingmpi/examples-usingmpi/simplempi/cpi_c.html
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cstdlib>
#include <math.h>

#include "calc_pi_serial_lib.hpp"

// -- main driver -- //
// -- main driver -- //
int main(int argc, char * argv[] )
{
    double PI_REF = 3.141592653589793238462643;

    int num_itrs = 100;
    if(argc >1)
    {
        num_itrs = std::atoi(argv[1]);
    }
    
    std::cout << "calculating pi using " 
              << num_itrs 
              << " monte carlo iterations."
              << std::endl;

    double pi = calc_pi_serial(num_itrs);

    std::cout.precision(16);

    std::cout << "pi is approximately "
              << pi 
              << " , Error is "
              << fabs(pi - PI_REF)
              << std::endl;

}