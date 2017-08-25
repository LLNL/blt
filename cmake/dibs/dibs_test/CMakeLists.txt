project(dibs_test)
cmake_minimum_required(VERSION 2.8)
include(../build/dibs.cmake)

require_spack_package(caliper raja)

find_package(caliper)
find_package(raja)

MESSAGE(STATUS "Adopted RAJA Flags: ${RAJA_COMPILE_FLAGS}")

add_executable(a a.cpp)
target_link_libraries(a caliper caliper-common caliper-reader)
