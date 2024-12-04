#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.21.1/bin/cmake
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: clang@16.0.0
#------------------------------------------------------------------------------

#_blt_tutorial_hip_compiler_start
set(_compiler_root "/opt/rocm-5.6.0/llvm")

set(CMAKE_C_COMPILER "${_compiler_root}/bin/amdclang" CACHE PATH "")

set(CMAKE_CXX_COMPILER "${_compiler_root}/bin/amdclang++" CACHE PATH "")

set(CMAKE_Fortran_COMPILER "${_compiler_root}/bin/amdflang" CACHE PATH "")

set(CMAKE_Fortran_FLAGS "-Mfreeform" CACHE STRING "")

set(ENABLE_FORTRAN ON CACHE BOOL "")
#_blt_tutorial_hip_compiler_end

#------------------------------------------------------------------------------
# MPI
#------------------------------------------------------------------------------

set(_mpi_root "/usr/tce/packages/cray-mpich-tce/cray-mpich-8.1.28-rocmcc-5.6.0")

set(MPI_C_COMPILER "${_mpi_root}/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "${_mpi_root}/bin/mpicxx" CACHE PATH "")

set(MPI_Fortran_COMPILER "${_mpi_root}/bin/mpif90" CACHE PATH "")

set(MPIEXEC_EXECUTABLE "/usr/global/tools/flux_wrappers/bin/srun" CACHE PATH "")

set(MPIEXEC_NUMPROC_FLAG "-n" CACHE STRING "")

set(ENABLE_MPI ON CACHE BOOL "")

#------------------------------------------------------------------------------
# Hardware
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

# HIP

#------------------------------------------------------------------------------

#_blt_tutorial_useful_hip_variables_start
set(_rocm_root "/opt/rocm-5.6.0")

set(ENABLE_HIP ON CACHE BOOL "")

set(HIP_ROOT_DIR "${_rocm_root}/hip" CACHE STRING "")

set(CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "")

set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags -L${_rocm_root}/hip/../llvm/lib -L${_rocm_root}/hip/lib -Wl,-rpath,${_rocm_root}/hip/../llvm/lib:${_rocm_root}/hip/lib -lpgmath -lflang -lflangrti -lompstub -lamdhip64  -L${_rocm_root}/hip/../lib64 -Wl,-rpath,${_rocm_root}/hip/../lib64  -L${_rocm_root}/hip/../lib -Wl,-rpath,${_rocm_root}/hip/../lib -lamd_comgr -lhsa-runtime64 " CACHE STRING "")
#_blt_tutorial_useful_hip_variables_end

#------------------------------------------------
# Hardware Specifics
#------------------------------------------------

set(ENABLE_OPENMP OFF CACHE BOOL "")

set(ENABLE_GTEST_DEATH_TESTS ON CACHE BOOL "")
