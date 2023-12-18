This test will fail during compilation of the downstream library without the
BLT macro blt_install_tpl_setups. This failure is because the flag `BLT_ENABLE_FORTRAN`
won't be set in the base project, so the MPI FORTRAN headers won't be found
correctly.
