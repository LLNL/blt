set(CLANG_VERSION "clang-14.0.6")
set(CLANG_HOME "/usr/tce/packages/clang/${CLANG_VERSION}")
set(CMAKE_C_COMPILER "${CLANG_HOME}/bin/clang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${CLANG_HOME}/bin/clang++" CACHE PATH "")
