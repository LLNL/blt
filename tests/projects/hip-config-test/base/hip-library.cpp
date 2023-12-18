#include <iostream>
#include <hip/hip_runtime.h>
#include "hip-library.hpp"

__global__ void kernel() {
  int tidx = hipThreadIdx_x;
  int bidx = hipBlockIdx_x;
  printf("Hello from block %u, thread %u\n", bidx, tidx);
}

void run_kernel() {
  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  std::cout << " System minor " << devProp.minor << std::endl;
  std::cout << " System major " << devProp.major << std::endl;
  std::cout << " agent prop name " << devProp.name << std::endl;

  hipLaunchKernelGGL(kernel, dim3(4), dim3(4), 0, 0);
}
