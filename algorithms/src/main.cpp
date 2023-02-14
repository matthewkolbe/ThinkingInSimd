#include <benchmark/benchmark.h>
#include "avx_binary_search.cc"
#include <bits/stdc++.h>
#include <iostream>

static void avx(benchmark::State &state) {
    alignas(64) int* v = new int[state.range(0)];
    size_t n = state.range(0);
    for(int i = 0; i < n; ++i) {
        v[i] = i;
    }

    size_t match = -1;
    for (auto _ : state)
    {
        for(int i = 0; i < n; ++i) {
            match = index_match(v, n, i);
            benchmark::DoNotOptimize(match);
        }
    }
    
    for(int i = 0; i < state.range(0); ++i) {
        match = index_match(v, state.range(0), i);
        assert(match == i);
    }
}
BENCHMARK(avx)->Args({7})
              ->Args({15})
              ->Args({83})
              ->Args({503})
              ->Args({1031})
              ->Args({8<<8})
              ->Args({16<<8})
              ->Args({32<<8})
              ->Args({64<<8})
              ->Args({128<<8})
              ->Args({256<<8})
              ->Args({512<<8})
              ->Args({1024<<8});

static void avx_long(benchmark::State &state) {
    alignas(64) long long* v = new long long[state.range(0)];
    size_t n = state.range(0);
    for(long long i = 0; i < n; ++i) {
        v[(size_t)i] = i;
    }

    size_t match = -1;
    for (auto _ : state)
    {
        for(long long i = 0; i < n; ++i) {
            
            match = index_match(v, n, i);
            benchmark::DoNotOptimize(match);
        }
    }
    
    for(long long i = 0; i < state.range(0); ++i) {
        match = index_match(v, state.range(0), i);
        assert(match == i);
    }
}
BENCHMARK(avx_long)->Args({7})
              ->Args({15})
              ->Args({83})
              ->Args({503})
              ->Args({1031})
              ->Args({8<<8})
              ->Args({16<<8})
              ->Args({32<<8})
              ->Args({64<<8})
              ->Args({128<<8})
              ->Args({256<<8})
              ->Args({512<<8});

static void avx_short(benchmark::State &state) {
    alignas(64) short* v = new short[state.range(0)];
    size_t n = state.range(0);
    for(short i = 0; i < n; ++i) {
        v[(size_t)i] = i;
    }

    size_t match = -1;
    for (auto _ : state)
    {
        for(short i = 0; i < n; ++i) {
            match = index_match(v, n, i);
            benchmark::DoNotOptimize(match);
        }
    }
    
    for(short i = 0; i < state.range(0); ++i) {
        match = index_match(v, state.range(0), i);
        assert(match == i);
    }
}
BENCHMARK(avx_short)->Args({7})
              ->Args({15})
              ->Args({83})
              ->Args({503})
              ->Args({1031})
              ->Args({1<<12})
              ->Args({1<<13})
              ->Args({1<<14});


static void stl(benchmark::State &state) {
    alignas(64) int* v = new int[state.range(0)];
    size_t n = state.range(0);
    for(int i = 0; i < n; ++i) {
        v[i] = i;
    }

    size_t match = -1;
    for (auto _ : state)
    {
        for(int i = 0; i < n; ++i) {
            match = std::lower_bound(v , v + n, i) - v;
            benchmark::DoNotOptimize(match);
        }
    }
}
BENCHMARK(stl)->Args({7})
              ->Args({15})
              ->Args({83})
              ->Args({503})
              ->Args({1031})
              ->Args({8<<8})
              ->Args({16<<8})
              ->Args({32<<8})
              ->Args({64<<8})
              ->Args({128<<8})
              ->Args({256<<8})
              ->Args({512<<8})
              ->Args({1024<<8});

static void stl_long(benchmark::State &state) {
    alignas(64) long long* v = new long long[state.range(0)];
    size_t n = state.range(0);
    for(long long i = 0; i < n; ++i) {
        v[i] = i;
    }

    size_t match = -1;
    for (auto _ : state)
    {
        for(long long i = 0; i < n; ++i) {
            match = std::lower_bound(v , v + n, i) - v;
            benchmark::DoNotOptimize(match);
        }
    }
}
BENCHMARK(stl_long)->Args({7})
              ->Args({15})
              ->Args({83})
              ->Args({503})
              ->Args({1031})
              ->Args({8<<8})
              ->Args({16<<8})
              ->Args({32<<8})
              ->Args({64<<8})
              ->Args({128<<8})
              ->Args({256<<8})
              ->Args({512<<8});

static void stl_short(benchmark::State &state) {
    alignas(64) short* v = new short[state.range(0)];
    size_t n = state.range(0);
    for(short i = 0; i < n; ++i) {
        v[(size_t)i] = i;
    }

    size_t match = -1;
    for (auto _ : state)
    {
        for(short i = 0; i < n; ++i) {
            match = std::lower_bound(v , v + n, i) - v;
            benchmark::DoNotOptimize(match);
        }
    }

}
BENCHMARK(stl_short)->Args({7})
              ->Args({15})
              ->Args({83})
              ->Args({503})
              ->Args({1031})
              ->Args({1<<12})
              ->Args({1<<13})
              ->Args({1<<14});


BENCHMARK_MAIN();

// int main() {
//     alignas(64) int* v = new int[N_SIZE];

//     for(int i = 0; i < N_SIZE; ++i)
//         v[i] = i;

//     auto match = index_match(v, N_SIZE, 100000);

//     std::cout << match << std::endl;
// }