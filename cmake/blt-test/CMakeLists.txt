################################
# BLT Internal Testing Project
################################

################################
cmake_minimum_required(VERSION 3.1)

project(blt-example LANGUAGES CXX)

if (ENABLE_FORTRAN)
   enable_language(Fortran)
endif()

################################
# Include BLT
################################
include(blt/SetupBLT.cmake)

################################
# Add a library
################################

blt_add_library( NAME example
                 SOURCES "src/Example.cpp"
                 HEADERS "src/Example.hpp"
                 )

################################
# Add an executable 
# (which happens to be a test) 
################################
blt_add_executable(
         NAME t_example_smoke
         SOURCES "src/t_example_smoke.cpp"
         DEPENDS_ON example gtest)

################################
# Register our test w/ ctest
################################
blt_add_test(NAME t_example_smoke
             COMMAND t_example_smoke)


####################
#  Header-only test
####################

blt_add_library(NAME blt_header_only
                HEADERS "src/HeaderOnly.hpp")

# This executable depends on the header-only library

blt_add_executable(
  NAME t_header_only_smoke
  SOURCES "src/t_header_only_smoke.cpp"
  DEPENDS_ON blt_header_only gtest)

blt_add_test(NAME t_header_only_smoke
             COMMAND t_header_only_smoke)
