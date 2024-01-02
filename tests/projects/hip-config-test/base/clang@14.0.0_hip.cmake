#------------------------------------------------------------------------------
# This is meant to be a lightweight host config file to tell CMake where to find
# HIP on Tioga, for CI purposes.
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.21.1/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_C_COMPILER "/opt/rocm-5.1.1/llvm/bin/amdclang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/opt/rocm-5.1.1/llvm/bin/amdclang++" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/opt/rocm-5.1.1/llvm/bin/amdflang" CACHE PATH "")
set(CMAKE_Fortran_FLAGS "-Mfreeform" CACHE STRING "")

#------------------------------------------------------------------------------
# HIP
#------------------------------------------------------------------------------

set(ENABLE_HIP ON CACHE BOOL "")
set(HIP_ROOT_DIR "/opt/rocm-5.1.1/hip" CACHE STRING "")
set(HIP_CLANG_PATH "/opt/rocm-5.1.1/hip/../llvm/bin" CACHE STRING "")
set(CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags -L/opt/rocm-5.1.1/hip/../llvm/lib -L/opt/rocm-5.1.1/hip/lib -Wl,-rpath,/opt/rocm-5.1.1/hip/../llvm/lib:/opt/rocm-5.1.1/hip/lib -lpgmath -lflang -lflangrti -lompstub -lamdhip64 -L/opt/rocm-5.1.1/hip/../lib64 -Wl,-rpath,/opt/rocm-5.1.1/hip/../lib64 -lhsakmt -lamd_comgr" CACHE STRING "")
