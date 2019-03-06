// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other BLT Project Developers. See the top-level COPYRIGHT file for details
// 
// SPDX-License-Identifier: (BSD-3-Clause)

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

