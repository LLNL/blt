// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other BLT Project Developers. See the top-level LICENSE file for details
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <mpi.h>

#include "hip/hip_runtime.h"

__global__ void blt_hip_mpi_kernel() {}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  hipLaunchKernelGGL(blt_hip_mpi_kernel, dim3(1), dim3(1), 0, 0);
  MPI_Finalize();
  return 0;
}
