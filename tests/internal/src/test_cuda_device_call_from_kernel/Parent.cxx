#include "Parent.h"
#include <string.h>

__host__ __device__ Parent::Parent(const char *id, int order)
   :
   m_gpuParent(NULL),
   m_gpuExtractedParents(NULL)
{}

__global__ void kernelDelete(Parent** myGpuParent) {}
__global__ void kernelDeleteExtracted(Parent*** gpuExtractedParents) {}

