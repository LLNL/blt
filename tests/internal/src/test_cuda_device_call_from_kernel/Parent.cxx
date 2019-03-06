// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other BLT Project Developers. See the top-level COPYRIGHT file for details
// 
// SPDX-License-Identifier: (BSD-3-Clause)

#include "Parent.h"
#include <string.h>

__host__ __device__ Parent::Parent(const char *id, int order)
   :
   m_gpuParent(NULL),
   m_gpuExtractedParents(NULL)
{}

__global__ void kernelDelete(Parent** myGpuParent) {}
__global__ void kernelDeleteExtracted(Parent*** gpuExtractedParents) {}

