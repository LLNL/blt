This test creates a base library with a library function declared inside a header, 
and installed for use by the downstream library.  Then, the downstream library
calls the library functions on each host (using MPI).  This test makes sure that
both a base library can call `blt_install_tpl_setups` and a library downstream from
base can still use BLT.
