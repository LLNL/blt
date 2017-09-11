///////////////////////////////////////////////////////////////////////////////
///
/// \file calc_pi.hpp
/// 
/// Header file for calc_pi library example.
///
/// Adapted from:
///  https://www.mcs.anl.gov/research/projects/mpi/usingmpi/examples-usingmpi/simplempi/cpi_c.html
///////////////////////////////////////////////////////////////////////////////

#ifndef CALC_PI_HPP
#define CALC_PI_HPP

///
/// \brief calculate pi 
///
///  Estimate pi by integrating f(x) = 4/(1+x^2) from 0 to 1 using 
///  the trapezoidal rule with a given number of intervals.
/// 
double calc_pi(int num_intervals);

#endif
