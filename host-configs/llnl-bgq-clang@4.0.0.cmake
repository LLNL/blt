set(CMAKE_CXX_COMPILER "/usr/apps/gnu/clang/r284961-stable/bin/bgclang++11" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/apps/gnu/clang/r284961-stable/bin/bgclang" CACHE PATH "")

# Use clang's libc++
set(CMAKE_CXX_FLAGS "-stdlib=libc++" CACHE STRING "")
