###############################################################################
# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

############################
# Setup compiler options
############################

#################################################
# OpenMP
# (OpenMP support is provided by the compiler)
#################################################
message(STATUS "OpenMP Support is ${ENABLE_OPENMP}")
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    message(STATUS "OpenMP CXX Flags: ${OpenMP_CXX_FLAGS}")
    
    # register openmp with blt
    blt_register_library(NAME openmp
                         COMPILE_FLAGS ${OpenMP_CXX_FLAGS} 
                         LINK_FLAGS  ${OpenMP_CXX_FLAGS} 
                         DEFINES USE_OPENMP)
endif()

#####################################################
# Set some variables to simplify determining compiler
# Compiler string list from: 
#   https://cmake.org/cmake/help/v3.0/variable/CMAKE_LANG_COMPILER_ID.html
####################################################3

# use CMAKE_BUILD_TOOL to identify visual studio
# and CMAKE_CXX_COMPILER_ID for all other cases

if("${CMAKE_BUILD_TOOL}" MATCHES "(msdev|devenv|nmake|MSBuild)")
    set(COMPILER_FAMILY_IS_MSVC 1)
    message(STATUS "Compiler family is MSVC")

elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(COMPILER_FAMILY_IS_GNU 1)
    message(STATUS "Compiler family is GNU")

elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang") # For Clang or AppleClang
    set(COMPILER_FAMILY_IS_CLANG 1)
    message(STATUS "Compiler family is Clang")
    
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "XL")
    set(COMPILER_FAMILY_IS_XL 1)
    message(STATUS "Compiler family is XL")

elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(COMPILER_FAMILY_IS_INTEL 1)
    message(STATUS "Compiler family is Intel")

endif()


################################################
# Support for extra compiler flags and defines
################################################

message(STATUS "Adding optional BLT definitions and compiler flags")

####################################################
# create relocatable static libs by default
####################################################
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

##############################################
# Support Extra Definitions for all targets
##############################################
if(BLT_DEFINES)
    add_definitions(${BLT_DEFINES})
    message(STATUS "Added \"${BLT_DEFINES}\" to definitions")
endif()

##########################################
# If set, BLT_<LANG>_FLAGS are added to 
# all targets that use <LANG>-Compiler
##########################################

##########################################
# Support Extra Flags for the C compiler.
##########################################
if(BLT_C_FLAGS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${BLT_C_FLAGS}")
    message(STATUS "Updated CMAKE_C_FLAGS to \"${CMAKE_C_FLAGS}\"")
endif()

#############################################
# Support Extra Flags for the C++ compiler.
#############################################
if(BLT_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BLT_CXX_FLAGS}")
     message(STATUS "Updated CMAKE_CXX_FLAGS to \"${CMAKE_CXX_FLAGS}\"")
endif()

################################################
# Support Extra Flags for the Fortran compiler.
################################################
if(ENABLE_FORTRAN AND BLT_FORTRAN_FLAGS)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${BLT_FORTRAN_FLAGS}")
     message(STATUS "Updated CMAKE_Fortran_FLAGS to \"${CMAKE_Fortran_FLAGS}\"")
endif()

################################################
# Support Extra Flags for the CUDA compiler.
# N.B. must be ; delimited.
################################################
if(ENABLE_CUDA AND BLT_CUDA_FLAGS)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${BLT_CUDA_FLAGS}")
    # Replace spaces with semicolons in the CUDA flags
    foreach(flag_var CUDA_NVCC_FLAGS)
      if(${flag_var} MATCHES " ")
        string(REGEX REPLACE " " ";" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES " ")
    endforeach(flag_var)
endif()


###############################################################
# Support Extra Flags based on CMake Config Type
###############################################################
#
# We guard this approach to avoid issues with CMake generators
# that support multiple configurations, like Visual Studio.
#
###############################################################
if(NOT CMAKE_CONFIGURATION_TYPES)

    set(cfg_types DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)

    foreach(cfg_type in ${cfg_types})
        # flags for the C compiler
        if(BLT_C_FLAGS_${cfg_type})
            set(CMAKE_C_FLAGS_${cfg_type}
                "${CMAKE_C_FLAGS_${cfg_type}} ${BLT_C_FLAGS_${cfg_type}}")
            message(STATUS "Updated CMAKE_C_FLAGS_${cfg_type} to \"${CMAKE_C_FLAGS_${cfg_type}}\"")
        endif()

        # flags for the C++ compiler
        if(BLT_CXX_FLAGS_${cfg_type})
            set(CMAKE_CXX_FLAGS_${cfg_type}
                "${CMAKE_CXX_FLAGS_${cfg_type}} ${BLT_CXX_FLAGS_${cfg_type}}")
            message(STATUS "Updated CMAKE_CXX_FLAGS_${cfg_type} to \"${CMAKE_CXX_FLAGS_${cfg_type}}\"")
        endif()

        # flags for the Fortran compiler
        if(ENABLE_FORTRAN AND BLT_FORTRAN_FLAGS_${cfg_type})
            set(CMAKE_Fortran_FLAGS_${cfg_type}
                "${CMAKE_Fortran_FLAGS_${cfg_type}} ${BLT_FORTRAN_FLAGS_${cfg_type}}")
            message(STATUS "Updated CMAKE_Fortran_FLAGS_${cfg_type} to \"${CMAKE_Fortran_FLAGS_${cfg_type}}\"")
        endif()

    endforeach()

endif()



################################
# RPath Settings
################################

# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

################################
# Enable C++11/14
################################

SET( CMAKE_CXX_EXTENSIONS OFF )
SET( CMAKE_CXX_STANDARD_REQUIRED ON )

if( BLT_CXX_STD STREQUAL c++98 ) 
    set(CMAKE_CXX_STANDARD 98)
elseif( BLT_CXX_STD STREQUAL c++11 )
    set(CMAKE_CXX_STANDARD 11)
elseif( BLT_CXX_STD STREQUAL c++14 )
    set(CMAKE_CXX_STANDARD 14)
else()
    message(FATAL_ERROR "${BLT_CXX_STD} is an invalid entry for BLT_CXX_STD.
    Valid Options are ( c++98, c++11, c++14 )")
endif()

message(STATUS "Standard C++${CMAKE_CXX_STANDARD} selected") 


##################################################################
# Additional compiler warnings and treatment of warnings as errors
##################################################################

blt_append_custom_compiler_flag(
    FLAGS_VAR BLT_ENABLE_ALL_WARNINGS_FLAG
     DEFAULT "-Wall -Wextra"
     CLANG   "-Wall -Wextra" 
                    # Additional  possibilities for clang include: 
                    #       "-Wdocumentation -Wdeprecated -Weverything"
     MSVC    "/W4"
                    # Additional  possibilities for visual studio include:
                    # "/Wall /wd4619 /wd4668 /wd4820 /wd4571 /wd4710"
     XL      ""     # qinfo=<grp> produces additional messages on XL
                    # qflag=<x>:<x> defines min severity level to produce messages on XL
                    #     where x is i info, w warning, e error, s severe; default is: 
                    # (default is  qflag=i:i)
     )

blt_append_custom_compiler_flag(
    FLAGS_VAR BLT_WARNINGS_AS_ERRORS_FLAG
     DEFAULT  "-Werror"
     MSVC     "/WX"
     XL       "qhalt=w"       # i info, w warning, e error, s severe (default)
     )

#
# Modify flags to avoid static linking runtime issues on windows.
# (adapted from RAJA)
#

if ( COMPILER_FAMILY_IS_MSVC AND NOT BUILD_SHARED_LIBS )
    foreach(flag_var
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${flag_var} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endif()

set(langFlags "CMAKE_C_FLAGS" "CMAKE_CXX_FLAGS")

if (ENABLE_ALL_WARNINGS)
    message(STATUS  "Enabling all compiler warnings on all targets.")

    foreach(flagVar ${langFlags})
        set(${flagVar} "${${flagVar}} ${BLT_ENABLE_ALL_WARNINGS_FLAG}") 
    endforeach()
endif()

if (ENABLE_WARNINGS_AS_ERRORS)
    message(STATUS  "Enabling treatment of warnings as errors on all targets.")

    foreach(flagVar ${langFlags})   
        set(${flagVar} "${${flagVar}} ${BLT_WARNINGS_AS_ERRORS_FLAG}") 
    endforeach()
endif()


foreach(flagVar ${langFlags})   
    message(STATUS "${flagVar} flags are:  ${${flagVar}}")
endforeach()


################################
# Enable Fortran
################################
if(ENABLE_FORTRAN)
    # if enabled but no fortran compiler, halt the configure
    if(CMAKE_Fortran_COMPILER)
        message(STATUS  "Fortran support enabled.")
    else()
        message(FATAL_ERROR "Fortran support selected, but no Fortran compiler was found.")
    endif()

    # default property to free form
    set(CMAKE_Fortran_FORMAT FREE)
else()
    message(STATUS  "Fortran support disabled.")
endif()
