#include <omp.h>
#include <iostream>

void test_func()
{

  #pragma omp parallel
  {
    int thId = omp_get_thread_num();
    int thNum = omp_get_num_threads();
    int thMax = omp_get_max_threads();

    #pragma omp critical
    std::cout <<"\nMy thread id is: " << thId
              <<"\nNum threads is: " << thNum
              <<"\nMax threads is: " << thMax
              << std::endl;
  }
}

int main() {
  test_func();
  return 0;
}
