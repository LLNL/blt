#include <iostream>
#include <omp.h>

#include "openmp-mpi-cpp.hpp"

void test_func()
{
  #pragma omp parallel
  {
    int thId = omp_get_thread_num();
    int thNum = omp_get_num_threads();
    int thMax = omp_get_max_threads();

    #pragma omp critical
    std::cout << "My thread id is: " << thId << std::endl
              << "Num threads is: " << thNum << std::endl
              << "Max threads is: " << thMax << std::endl;
  }
}
