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

// -- calculate pi via monte carlo methods  -- //
double calc_pi(int num_itrs)
{
    int i =0;
    
    double h   = 1.0 / (double) num_itrs;
    double sum = 0.0;
    
    while(i != num_itrs)
    {
        double x = h * ((double)i - 0.5);
        sum += (4.0 / (1.0 + x*x));
        i++;
    }
    
    double pi = h * sum;
    
    return pi;
}

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

    double pi = calc_pi(num_itrs);

    std::cout.precision(16);

    std::cout << "pi is approximately "
              << pi 
              << " , Error is "
              << fabs(pi - PI_REF)
              << std::endl;

}