///////////////////////////////////////////////////////////////////////////////
///
/// \file calc_pi_cuda.hpp
/// 
/// Header file for calc_pi_cuda library example.
///
/// Adapted from:
///  https://www.mcs.anl.gov/research/projects/mpi/usingmpi/examples-usingmpi/simplempi/cpi_c.html
///////////////////////////////////////////////////////////////////////////////

#ifndef CALC_PI_CUDA_HPP
#define CALC_PI_CUDA_HPP

///
/// \brief calculate pi using cuda 
///
///  Estimate pi by integrating f(x) = 4/(1+x^2) from 0 to 1 using 
///  numerical integration over a given number of intervals.
///
double calc_pi_cuda(int num_intervals);

#endif
