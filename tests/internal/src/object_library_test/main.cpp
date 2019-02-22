//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
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
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
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
