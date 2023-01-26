#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.21.1/bin/cmake
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: clang@14.0.0
#------------------------------------------------------------------------------
set(CMAKE_C_COMPILER "/opt/rocm-5.1.1/llvm/bin/amdclang" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/opt/rocm-5.1.1/llvm/bin/amdclang++" CACHE PATH "")

set(CMAKE_Fortran_COMPILER "/opt/rocm-5.1.1/llvm/bin/amdflang" CACHE PATH "")

set(CMAKE_Fortran_FLAGS "-Mfreeform" CACHE STRING "")

set(ENABLE_FORTRAN ON CACHE BOOL "")

#------------------------------------------------------------------------------
# MPI
#------------------------------------------------------------------------------

set(MPI_C_COMPILER "/usr/tce/packages/cray-mpich-tce/cray-mpich-8.1.16-rocmcc-5.1.1/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/usr/tce/packages/cray-mpich-tce/cray-mpich-8.1.16-rocmcc-5.1.1/bin/mpicxx" CACHE PATH "")

set(MPI_Fortran_COMPILER "/usr/tce/packages/cray-mpich-tce/cray-mpich-8.1.16-rocmcc-5.1.1/bin/mpif90" CACHE PATH "")

set(MPIEXEC_EXECUTABLE "/usr/bin/flux mini run" CACHE PATH "")

set(MPIEXEC_NUMPROC_FLAG "-n" CACHE STRING "")

set(ENABLE_MPI ON CACHE BOOL "")

#------------------------------------------------------------------------------
# Hardware
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

# HIP

#------------------------------------------------------------------------------


set(ENABLE_HIP ON CACHE BOOL "")

set(HIP_ROOT_DIR "/opt/rocm-5.1.1/hip" CACHE STRING "")

set(HIP_CLANG_PATH "/opt/rocm-5.1.1/hip/../llvm/bin" CACHE STRING "")

set(CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "")

set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags -L/opt/rocm-5.1.1/hip/../llvm/lib -L/opt/rocm-5.1.1/hip/lib -Wl,-rpath,/opt/rocm-5.1.1/hip/../llvm/lib:/opt/rocm-5.1.1/hip/lib -lpgmath -lflang -lflangrti -lompstub -lamdhip64 -L/opt/rocm-5.1.1/hip/../lib64 -Wl,-rpath,/opt/rocm-5.1.1/hip/../lib64 -lhsakmt -lamd_comgr" CACHE STRING "")

#------------------------------------------------
# Hardware Specifics
#------------------------------------------------

set(ENABLE_OPENMP OFF CACHE BOOL "")

set(ENABLE_GTEST_DEATH_TESTS ON CACHE BOOL "")
