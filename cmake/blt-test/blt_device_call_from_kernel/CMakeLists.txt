#####################################
# Test CUDA virtual device method call from kernels across library boundary
# CUDA needs to be enabled, with appropriate Flags set up for building and running
# on your current GPU architecture. 
#####################################
set(t_device_call_from_kernel_headers
    blt_device_call_from_kernel/Child.h
    blt_device_call_from_kernel/Parent.h)

set(t_device_call_from_kernel_sources
    blt_device_call_from_kernel/Child.cxx
    blt_device_call_from_kernel/Parent.cxx)

set(t_device_call_from_kernel_exec_src
    blt_device_call_from_kernel/CudaTests.cxx)

set(t_device_call_from_kernel_depends cuda)


blt_add_library(NAME t_device_call_from_kernel_lib
                SOURCES ${t_device_call_from_kernel_sources}
                HEADERS ${t_device_call_from_kernel_headers}
                DEPENDS_ON ${t_device_call_from_kernel_depends}
                )


blt_add_executable( NAME t_device_call_from_kernel_exec
                    SOURCES ${t_device_call_from_kernel_exec_src}
                    DEPENDS_ON t_device_call_from_kernel_lib gtest cuda)

blt_add_test(NAME t_device_call_from_kernel COMMAND t_device_call_from_kernel_exec)


