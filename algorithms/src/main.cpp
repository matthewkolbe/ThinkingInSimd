// Copyright 2023 Matthew Kolbe

#include <benchmark/benchmark.h>
#include "avx_binary_search.cc"
#include <bits/stdc++.h>
#include <iostream>
#include <new>

static void avx_full(benchmark::State &state) {
    int * v = new (std::align_val_t(64)) int[state.range(0)];
    int * lkup = new (std::align_val_t(64)) int[state.range(0)];
    //std::cout << alignof(v) << std::endl;
    size_t n = state.range(0);
    for(int i = 0; i < n; ++i) {
        v[i] = i;
        lkup[i] = std::rand() / (1+(RAND_MAX / n));
        //std::cout << lkup[i] << std::endl;
    }

    __m512i match;
    for (auto _ : state)
    {
        for(int i = 0; i + 15 < n; i+=16) {
            auto l = _mm512_loadu_epi32(lkup + i);
            match = bulk_index_match(v, n, l);
            benchmark::DoNotOptimize(match);
        }
    }
    
    ::operator delete[] (v, std::align_val_t(64));
    ::operator delete[] (lkup, std::align_val_t(64));
}
BENCHMARK(avx_full)->Args({16})
              ->Args({64})
              ->Args({256})
              ->Args({1<<10})
              ->Args({1<<12})
              ->Args({1<<14})
              ->Args({1<<16})
              ->Args({1<<20})
              ->Args({1<<24})
              ->Args({1<<26});

static void avx(benchmark::State &state) {
    int * v = new (std::align_val_t(64)) int[state.range(0)];
    int * lkup = new (std::align_val_t(64)) int[state.range(0)];
    //std::cout << alignof(v) << std::endl;
    size_t n = state.range(0);
    for(int i = 0; i < n; ++i) {
        v[i] = i;
        lkup[i] = std::rand() / (1+(RAND_MAX / n));
    }

    size_t match = -1;
    for (auto _ : state)
    {
        for(int i = 0; i < n; ++i) {
            match = index_match(v, n, lkup[i]);
            benchmark::DoNotOptimize(match);
        }
    }
    
    for(int i = 0; i < state.range(0); ++i) {
        match = index_match(v, state.range(0), i);
        assert(match == i);
    }

    ::operator delete[] (v, std::align_val_t(64));
    ::operator delete[] (lkup, std::align_val_t(64));
}
BENCHMARK(avx)->Args({7})
              ->Args({15})
              ->Args({83})
              ->Args({256})
              ->Args({503})
              ->Args({1<<10})
              ->Args({1<<12})
              ->Args({1<<14})
              ->Args({1<<16})
              ->Args({1<<20})
              ->Args({1<<24})
              ->Args({1<<26});



static void stl(benchmark::State &state) {
    int * v = new (std::align_val_t(64)) int[state.range(0)];
    int * lkup = new (std::align_val_t(64)) int[state.range(0)];
    //std::cout << alignof(v) << std::endl;
    size_t n = state.range(0);
    for(int i = 0; i < n; ++i) {
        v[i] = i;
        lkup[i] = std::rand() / (1+(RAND_MAX / n));
    }

    size_t match = -1;
    for (auto _ : state)
    {
        for(int i = 0; i < n; ++i) {
            match = std::lower_bound(v , v + n, lkup[i]) - v;
            benchmark::DoNotOptimize(match);
        }
    }

    ::operator delete[] (v, std::align_val_t(64));
    ::operator delete[] (lkup, std::align_val_t(64));
}
BENCHMARK(stl)->Args({7})
              ->Args({15})
              ->Args({64})
              ->Args({83})
              ->Args({256})
              ->Args({503})
              ->Args({1<<10})
              ->Args({1<<12})
              ->Args({1<<14})
              ->Args({1<<16})
              ->Args({1<<20})
              ->Args({1<<24})
              ->Args({1<<26});


BENCHMARK_MAIN();

// int main() {
//     alignas(64) int* v = new int[N_SIZE];

//     for(int i = 0; i < N_SIZE; ++i)
//         v[i] = i;

//     auto match = index_match(v, N_SIZE, 100000);

//     std::cout << match << std::endl;
// }