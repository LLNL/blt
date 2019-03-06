// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other BLT Project Developers. See the top-level COPYRIGHT file for details
// 
// SPDX-License-Identifier: (BSD-3-Clause)

#include "object.hpp"

#include <iostream>

int main()
{
    int number = object_number();
    if(number == 3) {
        std::cout << number
                  << " was correctly returned from object and base library."
                  << std::endl;
        return 0;
    }
    std::cerr << "Error:"
              << number
              << " was returned from object and base library."
              << std::endl
              << "3 was the correct number."
              << std::endl;
    return 1;
}
