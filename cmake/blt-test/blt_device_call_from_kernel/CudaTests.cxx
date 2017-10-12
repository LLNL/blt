#include <stdio.h>
#include "Parent.h"
#include "Child.h"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) {
         exit(code);
      }
   }
}
__global__ void kernelApply(Parent** myGpuParent) 
{
   double *input = new double[4];
   input[0] = 1.0;
   input[1] = 2.0;
   input[2] = 3.0;
   input[3] = 4.0;
   (*myGpuParent)->Evaluate(input);
}

int main(void)
{
    Child *c = new Child(0.0, 0.0, 0.0, 0.0);
    kernelApply<<<1, 1>>>(c->m_gpuParent);
    gpuAssert(cudaDeviceSynchronize(),__FILE__,__LINE__);
    return 0;
}
