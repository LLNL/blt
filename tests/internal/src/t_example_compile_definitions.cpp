// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other BLT Project Developers. See the top-level COPYRIGHT file for details
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

//------------------------------------------------------------------------------

// Simple test that expects symbol A to be defined as a non-zero number
TEST(blt_compile_definitions,check_A_defined)
{
  #if A
  SUCCEED();
  #else
  FAIL() << "Compiler define A was not defined as a non-zero number";
  #endif
}

// Simple test that expects symbol B to be defined
TEST(blt_compile_definitions,check_B_defined)
{
  #ifdef B
  SUCCEED();
  #else
  FAIL() << "Compiler define B was not defined";
  #endif
}
