///////////////////////////////////////////////////////////////////////////////
//
// file: calc_pi_serial_lib.cpp
// 
// Source file for calc_pi_serial library example.
//
// Adapted from:
// https://www.mcs.anl.gov/research/projects/mpi/usingmpi/examples-usingmpi/simplempi/cpi_c.html
///////////////////////////////////////////////////////////////////////////////


// -- calculate pi via monte carlo methods  -- //
double calc_pi_serial(int num_itrs)
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
