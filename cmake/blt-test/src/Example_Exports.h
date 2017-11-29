//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-725085
//
// All rights reserved.
//
// This file is part of BLT.
//
// For additional details, please also read BLT/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef _EXAMPLE_EXPORTS_H_
#define _EXAMPLE_EXPORTS_H_

//-----------------------------------------------------------------------------
// -- define proper lib exports for various platforms -- 
//-----------------------------------------------------------------------------
#if defined(_WIN32)
    #if defined(WIN32_SHARED_LIBS)
        #if defined(EXAMPLE_EXPORTS) || defined(example_EXPORTS)
            #define EXAMPLE_API __declspec(dllexport)
        #else
            #define EXAMPLE_API __declspec(dllimport)
        #endif
    #else
        #define EXAMPLE_API /* not needed for static on windows */
    #endif
    #if defined(_MSC_VER)
        /* Turn off warning about lack of DLL interface */
        #pragma warning(disable:4251)
        /* Turn off warning non-dll class is base for dll-interface class. */
        #pragma warning(disable:4275)
        /* Turn off warning about identifier truncation */
        #pragma warning(disable:4786)
    #endif
#else
# if __GNUC__ >= 4 && (defined(EXAMPLE_EXPORTS) || defined(example_EXPORTS))
#   define EXAMPLE_API __attribute__ ((visibility("default")))
# else
#   define EXAMPLE_API /* hidden by default */
# endif
#endif

#endif
