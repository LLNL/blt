#####################################
# Test CUDA virtual device method call from kernels across library boundary
# CUDA needs to be enabled, with appropriate Flags set up for building and running
# on your current GPU architecture. 
#####################################
set(t_cuda_device_call_from_kernel_headers
    Child.h
    Parent.h)

set(t_cuda_device_call_from_kernel_sources
    Child.cxx
    Parent.cxx)

set(t_cuda_device_call_from_kernel_exec_src
    CudaTests.cxx)

set(t_cuda_device_call_from_kernel_depends cuda)


blt_add_library(NAME t_cuda_device_call_from_kernel_lib
                SOURCES ${t_cuda_device_call_from_kernel_sources}
                HEADERS ${t_cuda_device_call_from_kernel_headers}
                DEPENDS_ON ${t_cuda_device_call_from_kernel_depends}
                )


blt_add_executable( NAME t_cuda_device_call_from_kernel_exec
                    SOURCES ${t_cuda_device_call_from_kernel_exec_src}
                    DEPENDS_ON t_cuda_device_call_from_kernel_lib gtest cuda)

blt_add_test(NAME t_cuda_device_call_from_kernel COMMAND t_cuda_device_call_from_kernel_exec)


