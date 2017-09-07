///////////////////////////////////////////////////////////////////////////////
//
// file: calc_pi_mpi_lib.cpp
// 
// Source file for calc_pi_serial library example.
//
// Adapted from:
// https://www.mcs.anl.gov/research/projects/mpi/usingmpi/examples-usingmpi/simplempi/cpi_c.html
///////////////////////////////////////////////////////////////////////////////

#include "mpi.h"

// -- calculate pi via monte carlo methods  -- //
double calc_pi_mpi(int num_itrs)
{
    int num_tasks = 0;
    int task_id   = 0;
    
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
    
    double h   = 1.0 / (double) num_itrs;
    double sum = 0.0;
    
    // TODO: Fix to distribute num_itrs across num_tasks 
    for(int i = task_id + 1; i <= num_itrs; i+= num_tasks) 
    {
        double x = h * ((double)i - 0.5);
        sum += (4.0 / (1.0 + x*x));
    }
    
    double pi_local = h * sum;
    double pi = 0;
    
    MPI_Reduce(&pi_local,
               &pi,
               1,
               MPI_DOUBLE,
               MPI_SUM,
               0,
               MPI_COMM_WORLD);

    
    return pi;
}
