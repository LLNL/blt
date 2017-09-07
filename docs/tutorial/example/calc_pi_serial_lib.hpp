///////////////////////////////////////////////////////////////////////////////
//
// file: calc_pi_serial_lib.hpp
// 
// Header file for calc_pi_serial library example.
//
// Adapted from:
// https://www.mcs.anl.gov/research/projects/mpi/usingmpi/examples-usingmpi/simplempi/cpi_c.html
///////////////////////////////////////////////////////////////////////////////


#ifndef CALC_PI_SERIAL_HPP
#define CALC_PI_SERIAL_HPP
//#include "mpi.h"
#include <math.h>


// -- calculate pi via monte carlo methods  -- //
double calc_pi_serial(int num_itrs);

#endif
