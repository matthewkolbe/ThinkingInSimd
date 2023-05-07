// Copyright 2023 Matthew Kolbe

#include <benchmark/benchmark.h>
#include <iostream>
#include <unordered_map>
#include <list>
#include <new>
#include <string>
#include <vector>
#include <assert.h> 

#include "fash.hh"

//#define ARGS ->Args({8})->Args({10})->Args({11})->Args({12})->Args({13})->Args({14})->Args({15})->Args({16})->Args({18})->Args({20});
#define ARGS ->Args({10})->Args({11})->Args({12})->Args({13})->Args({14})->Args({15})->Args({16})->Args({17})->Args({18})->Args({19})->Args({20})->Args({21})->Args({22})->Args({23})->Args({24})->Args({25})->Args({26});
//#define ARGS ->Args({20});

#define LOOKUPCOUNT 731

struct Point {
    int x, y;
    Point(int xx, int yy) {
        x = xx;
        y = yy;
    }
    Point() {
    }
    Point(const Point & p) {
        x = p.x;
        y = p.y;
    }
};

struct Empty {};

static void fash2_at_bmk(benchmark::State &state) {
    auto bits  = state.range(0);
    auto n = 1 << bits;
    fash2<uint64_t, uint64_t> table(bits);
    
    for(int i = 0; i < n; ++i) {
        table.insert_no_intrinsic_int64(i+ (1<<20), i);
    }


    for (auto _ : state)
    {
        for(int i = 0; i < n; i++) {
            auto found = table.at_no_intrinsic_int64(i+ (1<<20));
            benchmark::DoNotOptimize(found);
            benchmark::ClobberMemory();
        }
    }

    assert(table.at_no_intrinsic_int64(41 + (1<<20)) == 41);

}
BENCHMARK(fash2_at_bmk)ARGS

static void fash128_at_no_intr_bmk(benchmark::State &state) {
    auto bits  = state.range(0);
    auto n = 1 << bits;
    fash128x<uint64_t, uint64_t> table(bits);
    
    for(int i = 0; i < n; ++i) {
        table.insert_no_intrinsic_int64(i+ (1<<20), i);
    }
    
    for (auto _ : state)
    {
        for(int i = 0; i < n; i++)  {
            auto found = table.at_no_intrinsic_int64(i+ (1<<20));
            benchmark::DoNotOptimize(found);
            benchmark::ClobberMemory();
        }
    }

    assert(table.at_no_intrinsic_int64(41 + (1<<20)) == 41);

}
BENCHMARK(fash128_at_no_intr_bmk)ARGS

static void fash128x2_bmk(benchmark::State &state) {
    auto bits  = state.range(0);
    auto n = 1 << bits;
    fash128x2<uint64_t, uint64_t> table(bits);

    for(int i = 0; i < n; ++i) {
        table.insert_no_intrinsic_int64(i+ (1<<20), i);
        
    }

    for (auto _ : state)
    {
        for(int i = 0; i < n; i++) {
            auto found = table.at_no_intrinsic_int64(i+ (1<<20));
            benchmark::DoNotOptimize(found);
            benchmark::ClobberMemory();
        }
    }

    assert(table.at_no_intrinsic_int64(41 + (1<<20)) == 41);

}
BENCHMARK(fash128x2_bmk)ARGS


// static void fash_at_fullintr_bmk(benchmark::State &state) {
//     auto bits  = state.range(0);
//     auto n = 1 << bits;
//     fash<uint64_t, uint64_t> table(bits);
//     std::list<int> memthrasher;
//     auto keys =  new (std::align_val_t(64)) uint64_t[n];
    
//     for(int i = 0; i < n; ++i) {
//         for(int x = 0; x < 1000; ++x)
//             memthrasher.push_back(x);

//         keys[i] = (i + (1<<20));
//         table.insert_no_intrinsic_int64(i+ (1<<20), i);
        
//     }
    
//     memthrasher.clear();
//     auto a = memthrasher.empty();

//     for (auto _ : state)
//     {
//         for(int i = 0; i + 7 < n; i+=8) {
//             const auto k = _mm512_load_epi64(keys + i);
            
//             auto found = table.at512(k);
//             benchmark::DoNotOptimize(found);
//             benchmark::ClobberMemory();
//         }
//     }

//     ::operator delete[] (keys, std::align_val_t(64));

// }
// BENCHMARK(fash_at_fullintr_bmk)ARGS

static void fash_at_no_intr_bmk(benchmark::State &state) {
    auto bits  = state.range(0);
    auto n = 1 << bits;
    fash<uint64_t, uint64_t> table(bits);

    for(int i = 0; i < n; ++i) {
        table.insert_no_intrinsic_int64(i+ (1<<20), i);
        
    }

    for (auto _ : state)
    {
        for(int i = 0; i < n; i++)  {
            auto found = table.at_no_intrinsic_int64(i+ (1<<20));
            benchmark::DoNotOptimize(found);
            benchmark::ClobberMemory();
        }
    }

    assert(table.at_no_intrinsic_int64(41 + (1<<20)) == 41);
}
BENCHMARK(fash_at_no_intr_bmk)ARGS

static void fash_at_intr_bmk(benchmark::State &state) {
    auto bits  = state.range(0);
    auto n = 1 << bits;
    fash<uint64_t, uint64_t> table(bits);

    for(int i = 0; i < n; ++i) {
        table.insert_no_intrinsic_int64(i+ (1<<20), i);
        
    }

    for (auto _ : state)
    {
        for(int i = 0; i < n; i++)  {
            auto found = table.at_int64(i+ (1<<20));
            benchmark::DoNotOptimize(found);
            benchmark::ClobberMemory();
        }
    }

    assert(table.at_no_intrinsic_int64(41 + (1<<20)) == 41);
}
BENCHMARK(fash_at_intr_bmk)ARGS


struct i64hasher {
    size_t operator()( const uint64_t & xx ) const // <-- don't forget const
	{
		auto x = (xx ^ (xx >> 31) ^ (xx >> 62)) * UINT64_C(0x319642b2d24d8ec3);
        x = (x ^ (x >> 27) ^ (x >> 54)) * UINT64_C(0x96de1b173f119089);
        x = x ^ (x >> 30) ^ (x >> 60);
        return x;
	}
};

static void unmap_bmk(benchmark::State &state) {
    auto bits  = state.range(0);
    auto n = 1 << bits;
    std::unordered_map<uint64_t, uint64_t, i64hasher> table(bits);
    std::vector<uint64_t> keys;

    for(int i = 0; i < n; ++i) {
        table.insert( {i + (1<<20), i });
        
    }

    for (auto _ : state)
    {
        for(int i = 0; i < n; i++) {
            auto& found = table.at(i + (1<<20));
            benchmark::DoNotOptimize(found);
            benchmark::ClobberMemory();
        }
    }

}
BENCHMARK(unmap_bmk)ARGS

BENCHMARK_MAIN();
