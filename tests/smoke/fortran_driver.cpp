// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other BLT Project Developers. See the top-level COPYRIGHT file for details
// 
// SPDX-License-Identifier: (BSD-3-Clause)

//----------------------------------------------------------------------
extern "C" int fortran_test();

int main()
{
  int result = 0;

  // finalized when exiting main scope
  result = fortran_test();

  return result;
}
