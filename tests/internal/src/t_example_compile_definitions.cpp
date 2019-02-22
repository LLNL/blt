//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-725085
//
// All rights reserved.
//
// This file is part of BLT.
//
// For additional details, please also read BLT/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
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
