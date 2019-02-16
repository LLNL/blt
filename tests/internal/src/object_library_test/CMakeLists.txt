###############################################################################
# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-725085
#
# All rights reserved.
#
# This file is part of BLT.
#
# For additional details, please also read BLT/LICENSE.
###############################################################################
#
# Simple test that uses the blt_add_library macro with object libraries
#
###############################################################################

blt_add_library( NAME    base_object
                 SOURCES base_object.cpp
                 HEADERS base_object.hpp
                 OBJECT  TRUE)

blt_add_library( NAME       object
                 SOURCES    object.cpp
                 HEADERS    object.hpp
                 DEPENDS_ON base_object
                 OBJECT     TRUE)

blt_add_executable( NAME       blt_test_object_libraries
                    SOURCES    main.cpp
                    DEPENDS_ON object)
blt_add_test( NAME    blt_test_object_libraries
              COMMAND blt_test_object_libraries)
