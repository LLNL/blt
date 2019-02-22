#####################################
# Test clang-query static analysis
# Clang Query needs to be enabled, with appropriate Flags set up for building and running
# your code
#####################################

set(BLT_CLANG_QUERY_CHECKER_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/static_analysis_passes)
blt_add_executable( NAME well_analyzed_executable
                    SOURCES well_analyzed_source.cpp
)
blt_add_code_checks(PREFIX all
  SOURCES well_analyzed_source.cpp)
blt_add_clang_query_target(NAME filtered CHECKERS "if-stmt"
  SRC_FILES well_analyzed_source.cpp)

