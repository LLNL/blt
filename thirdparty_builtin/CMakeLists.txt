###############################################################################
# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
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

set(_blt_tpl_targets) # tracks names of enabled tpl targets

if(ENABLE_TESTS)
    include(CTest)

    # Note: Users can pass extra compiler flags to gtest 
    #       with the 'gtest_extra_flags' variable and extra 
    #       compile definitions with the 'gtest_defines' variable.


    # The following flag should be added to gtest's dependents, 
    # but not to gtest in shared Windows builds
    if(WIN32 AND BUILD_SHARED_LIBS)
        list(APPEND gtest_export_defines "-DGTEST_LINKED_AS_SHARED_LIBRARY=1")
    endif()

    # Explicitly enable/disable death tests
    if(ENABLE_GTEST_DEATH_TESTS)
        list(APPEND gtest_defines "-DGTEST_HAS_DEATH_TEST=1")
    else()
        list(APPEND gtest_defines "-DGTEST_HAS_DEATH_TEST=0")
    endif()

    #
    # Use BLT's ENABLE_GMOCK option to set the build options for 
    # gtest's cmake project
    #
    
    if(ENABLE_GMOCK)
        
        # In BLT ENABLE_GMOCK implies ENABLE_GTEST,
        # IF ENABLE_GTEST is OFF, force to ON
        if(NOT ENABLE_GTEST)
            set(ENABLE_GTEST ON CACHE BOOL "")
        endif()
        
        set(BUILD_GMOCK ON  CACHE BOOL "")
        set(BUILD_GTEST OFF CACHE BOOL "")
    else()
        set(BUILD_GMOCK OFF CACHE BOOL "")
        set(BUILD_GTEST ON  CACHE BOOL "")
    endif()

    message(STATUS "Google Test Support is ${ENABLE_GTEST}")
    message(STATUS "Google Mock Support is ${ENABLE_GMOCK}")
    
    #
    # Guard of googletest w/ ENABLE_GTEST
    # In BLT, ENABLE_GTEST is also required when using ENABLE_GMOCK
    #
    
    if(ENABLE_GTEST)
        
        #
        # gtest 1.8 emits many warnings related to dll-interface 
        # issues on windows, so we add flags to work around these
        # warnings, so they don't mask warnings we care about
        #
        # For more info see: https://github.com/LLNL/blt/issues/79
        # 
        
        blt_append_custom_compiler_flag( FLAGS_VAR gtest_extra_flags
                                         DEFAULT " " 
                                         MSVC  "/wd4251" "/wd4275" )

        set (INSTALL_GMOCK OFF CACHE BOOL "")
        set (INSTALL_GTEST OFF CACHE BOOL "")
                
        # Enable builtin google test 
        add_subdirectory(googletest-master-2018-02-01
                         ${BLT_BUILD_DIR}/thirdparty_builtin/googletest-master-2018-02-01)

        if(UNIX AND NOT APPLE)
            find_package( Threads )
            set(gtest_extra_libs ${CMAKE_THREAD_LIBS_INIT})
        endif()

        blt_register_library(NAME gtest
                             INCLUDES ${gtest_SOURCE_DIR}/include
                             LIBRARIES gtest_main gtest ${gtest_extra_libs}
                             COMPILE_FLAGS ${gtest_extra_flags}
                             DEFINES  ${gtest_defines} ${gtest_export_defines}
                             TREAT_INCLUDES_AS_SYSTEM ON
                             )
                             
        blt_add_target_definitions(TO gtest TARGET_DEFINITIONS ${gtest_defines})

        list(APPEND _blt_tpl_targets gtest gtest_main)

        if(ENABLE_GMOCK)
            blt_register_library(NAME gmock
                                 INCLUDES ${gmock_SOURCE_DIR}/include
                                 LIBRARIES gmock_main gmock
                                 COMPILE_FLAGS ${gtest_extra_flags}
                                 DEFINES  ${gtest_defines}
                                 TREAT_INCLUDES_AS_SYSTEM ON
                                 )
            list(APPEND _blt_tpl_targets gmock gmock_main)
        endif()
    endif()

    # Enable Fruit (FortRan UnIT testing) support
    if (ENABLE_FORTRAN)
        message(STATUS "Fruit Support is ${ENABLE_FRUIT}")
        if(ENABLE_FRUIT)
            add_subdirectory(fruit-3.4.1
                             ${BLT_BUILD_DIR}/thirdparty_builtin/fruit-3.4.1)
            list(APPEND _blt_tpl_targets fruit)
        endif()
    endif()

endif()

if(ENABLE_BENCHMARKS)
    if(WIN32 AND BUILD_SHARED_LIBS)
      message(FATAL_ERROR "Benchmarks cannot be built when BUILD_SHARED_LIBS=On")
    endif()

    ## google benchmark support
    add_subdirectory(gbenchmark-master-2017-05-19
                     ${BLT_BUILD_DIR}/thirdparty_builtin/gbenchmark-master-2017-05-19)

    if (UNIX AND NOT APPLE)
      find_library(RT_LIBRARIES rt)
    endif()

    blt_register_library(NAME gbenchmark
                         INCLUDES ${benchmark_SOURCE_DIR}/include ${benchmark_SOURCE_DIR}
                         LIBRARIES benchmark ${RT_LIBRARIES}
                         TREAT_INCLUDES_AS_SYSTEM ON
                         )

    list(APPEND _blt_tpl_targets benchmark)

    if(ENABLE_TESTS)
      # This sets up a target to run the benchmarks
      add_custom_target(run_benchmarks 
                        COMMAND ctest -C Benchmark -VV -R benchmark
                        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                        )
    endif()
endif()

# Set the folder property of the blt thirdparty libraries 
if(ENABLE_FOLDERS)
    foreach(tpl ${_blt_tpl_targets})
        blt_set_target_folder(TARGET ${tpl} FOLDER blt/thirdparty)
    endforeach()
endif()


