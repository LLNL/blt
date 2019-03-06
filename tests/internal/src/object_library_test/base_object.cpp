// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other BLT Project Developers. See the top-level COPYRIGHT file for details
// 
// SPDX-License-Identifier: (BSD-3-Clause)

#include "base_object.hpp"
#include "inherited_base.hpp"

int base_number()
{
    return inherited_number() + 2;
}
