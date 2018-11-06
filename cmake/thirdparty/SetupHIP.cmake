###############################################################################
#
# Copyright (c) 2018 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#
###############################################################################


################################
# HIP
################################
set (CMAKE_MODULE_PATH "${BLT_ROOT_DIR}/cmake/thirdparty;${CMAKE_MODULE_PATH}")
find_package(HIP REQUIRED)

message(STATUS "HIP version:      ${HIP_VERSION_STRING}")
message(STATUS "HIP platform:     ${HIP_PLATFORM}")
#message(STATUS "HIP Include Path: ${HIP_INCLUDE_DIRS}")
#message(STATUS "HIP Libraries:    ${HIP_LIBRARIES}")

# depend on 'hip', if you need to use hip
# headers, link to hip libs, and need to run your source
# through a hip compiler (hipcc)
blt_register_library(NAME hip
                     INCLUDES ${HIP_INCLUDE_DIRS}
                     LIBRARIES ${HIP_LIBRARIES})

# depend on 'hip_runtime', if you only need to use hip
# headers or link to hip libs, but don't need to run your source
# through a hip compiler (hipcc)
blt_register_library(NAME hip_runtime
                     INCLUDES ${HIP_INCLUDE_DIRS}
                     LIBRARIES ${HIP_LIBRARIES})
