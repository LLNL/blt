##
## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-689114
##
## All rights reserved.
##
## For release details and restrictions, please see RAJA/LICENSE.
##

###########################################################
# standard ROCm HCC compiler
###########################################################

set(ENABLE_ROCM ON CACHE BOOL "")
set(ENABLE_OPENMP OFF CACHE BOOL "")

set(ROCM_ROOT_DIR "/opt/rocm" CACHE PATH "ROCm ROOT directory path")

set(ROCM_INCLUDE_PATH "${ROCM_ROOT_DIR}/hcc/include"  CACHE PATH "")
set(ROCM_CXX_LIBRARIES "-L${ROCM_ROOT_DIR}/lib -lhc_am -lhip_hcc" CACHE STRING "")

###########################################################
# specify the target architecture
#  Default with ROCm 1.7 is gfx803 (Fiji)
#  Other options:
#    gfx700  Hawaii
#    gfx803  Polaris (RX580)
#    gfx900  Vega
#    gfx901  
###########################################################
set(ROCM_ARCH_FLAG "-amdgpu-target=${ROCM_ARCH}" CACHE STRING "")

###########################################################
# get compile/link flags from hcc-config
###########################################################
execute_process(COMMAND ${ROCM_ROOT_DIR}/hcc/bin/hcc-config --cxxflags OUTPUT_VARIABLE ROCM_CXX_COMPILE_FLAGS)
execute_process(COMMAND ${ROCM_ROOT_DIR}/hcc/bin/hcc-config --ldflags OUTPUT_VARIABLE ROCM_CXX_LINK_FLAGS)

set(ROCM_CXX_COMPILE_FLAGS "${ROCM_CXX_COMPILE_FLAGS} -Wno-unused-command-line-argument -DHCC_ENABLE_ACCELERATOR_PRINTF" CACHE STRING "") 
set(ROCM_CXX_LINK_FLAGS "${ROCM_CXX_LINK_FLAGS} ${ROCM_ARCH_FLAG} ${ROCM_CXX_LIBRARIES}" CACHE STRING "")

###########################################################
# set CMake cache variables
###########################################################
set(CMAKE_CXX_COMPILER "${ROCM_ROOT_DIR}/hcc/bin/hcc" CACHE FILEPATH "ROCm HCC compiler")
set(CMAKE_CXX_FLAGS "${ROCM_CXX_COMPILE_FLAGS}" CACHE STRING "HCC compiler flags")

set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_COMPILER} ${ROCM_CXX_LINK_FLAGS} <OBJECTS> <LINK_LIBRARIES> -o <TARGET>" CACHE STRING "HCC linker command line")
