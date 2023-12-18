This test checks if the flags used to compile OpenMP cpp source code
are generator expressions.  This test will fail if the expected generator 
expression is not present in the `openmp` target's compile flags.  This test 
is necessary because the flags needed for `gfortran` to compile OpenMP are 
different from those needed by clang.

This test will need to be updated if the generator expression for compile flags
inside BLTSetupOpenMP.cmake are changed.
