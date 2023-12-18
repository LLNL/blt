This test creates a base library requiring CUDA, then creates a downstream 
library requiring MPI.  This tests the inclusion of BLTSetupCUDA 
from a downstream library.  In particular, this tests
- if all necessary macros required by BLTSetupCUDA have been installed
- if BLTSetupMPI's generator expressions are evaluated correctly
- if ENABLE_CUDA is correctly forwarded from the base project to the downstream
