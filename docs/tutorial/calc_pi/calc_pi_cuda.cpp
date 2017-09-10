///////////////////////////////////////////////////////////////////////////////
//
// file: calc_pi_cuda.cpp
// 
// Source file for calc_pi_cuda library example.
//
// Adapted from:
// https://www.mcs.anl.gov/research/projects/mpi/usingmpi/examples-usingmpi/simplempi/cpi_c.html
///////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
///
/// file: calc_pi_cuda.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include "gtest/gtest.h"
#include <stdio.h>

__device__ const char *STR = "HELLO WORLD!";
const char STR_LENGTH = 12;

__global__ void hello()
{
    printf("%c\n", STR[threadIdx.x % STR_LENGTH]);
}


// -- calculate pi via simple integration  -- //
double calc_pi(int num_intervals)
{
    int i =0;
    
    double h   = 1.0 / (double) num_intervals;
    double sum = 0.0;
    
    while(i != num_intervals)
    {
        double x = h * ((double)i - 0.5);
        sum += (4.0 / (1.0 + x*x));
        i++;
    }
    
    double pi = h * sum;
    
    int num_threads = STR_LENGTH;
    int num_blocks = 1;
    hello<<<num_blocks,num_threads>>>();
    cudaDeviceSynchronize();
    
    
    return pi;
}



