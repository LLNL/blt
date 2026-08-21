cmake_minimum_required(VERSION 3.14)

project(mpi-mixed-language LANGUAGES CXX)

set(BLT_CXX_STD c++14 CACHE STRING "")
set(ENABLE_MPI On CACHE BOOL "")
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../fake_mpi")

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

blt_install_tpl_setups(DESTINATION lib/cmake/${PROJECT_NAME})

blt_add_library(
    NAME mpi-mixed-language
    SOURCES mpi-mixed-language.cpp
    DEPENDS_ON blt::mpi)

install(TARGETS mpi-mixed-language
    EXPORT mpi-mixed-language-targets)

install(FILES
    ${PROJECT_SOURCE_DIR}/mpi-mixed-language-config.cmake
    DESTINATION lib/cmake/mpi-mixed-language)

install(EXPORT mpi-mixed-language-targets
    DESTINATION lib/cmake/mpi-mixed-language)
