/*
 * Copyright (c) 2015, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

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
        for (int i=0; i < state.range_x(); ++i) {
          benchmark::DoNotOptimize(i);
      }
  }
  state.SetItemsProcessed(state.iterations() * state.range_x());

}
BASIC_BENCHMARK_TEST(benchmark_smoke_spin_loop);


void benchmark_smoke_accum_loop(benchmark::State& state) {
  while (state.KeepRunning()) {
      int accum = 0;
      for (int i=0; i <  state.range_x(); ++i) {
        accum += i;
      }
      benchmark::DoNotOptimize(accum);
  }
  state.SetItemsProcessed(state.iterations() * state.range_x());

}
BASIC_BENCHMARK_TEST(benchmark_smoke_accum_loop);

BENCHMARK_MAIN()

