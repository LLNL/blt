#include <stdio.h>
#include <cuda.h>

__global__ void kernel() {
  int tidx = threadIdx.x;
  int bidx = blockIdx.x;
  printf("Hello from block %u, thread %u\n", bidx, tidx);
}

void run_kernel() {
  kernel<<<4, 4>>>();
  cudaDeviceSynchronize();
}
