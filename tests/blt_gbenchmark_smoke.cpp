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

#include "benchmark/benchmark_api.h"

//------------------------------------------------------------------------------

#define BASIC_BENCHMARK_TEST(x) \
    BENCHMARK(x)->Arg( 1<<3 )->Arg( 1<<9 )->Arg( 1 << 13 )

void benchmark_smoke_empty(benchmark::State& state) {
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(state.iterations());
  }
}
BENCHMARK(benchmark_smoke_empty);


void benchmark_smoke_spin_loop(benchmark::State& state) {
    while (state.KeepRunning()) {
        for (int i=0; i < state.range(0); ++i) {
          benchmark::DoNotOptimize(i);
      }
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));

}
BASIC_BENCHMARK_TEST(benchmark_smoke_spin_loop);


void benchmark_smoke_accum_loop(benchmark::State& state) {
  while (state.KeepRunning()) {
      int accum = 0;
      for (int i=0; i <  state.range(0); ++i) {
        accum += i;
      }
      benchmark::DoNotOptimize(accum);
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));

}
BASIC_BENCHMARK_TEST(benchmark_smoke_accum_loop);

BENCHMARK_MAIN()

