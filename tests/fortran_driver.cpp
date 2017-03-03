/*
 * Copyright (c) 2015, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */


//----------------------------------------------------------------------
extern "C" int fortran_test();

int main()
{
  int result = 0;

  // finalized when exiting main scope
  result = fortran_test();

  return result;
}
