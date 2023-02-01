// Copyright 2023 Matthew Kolbe
#include <benchmark/benchmark.h>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <memory>
#include <omp.h>
#include <thread>

#include "black_scholes.hpp"
#include "vcl/vectorclass.h"
#include "vec_black_scholes.hpp"

// compile with: g++ main.cpp -std=c++20 -O3 -lm -lstdc++ -march=native -fopenmp -ffast-math  -lbenchmark -lpthread
// format with: clang-format main.cpp -i -style=Microsoft

#define SIZE_N (1600 * 32)

#define BS_UL 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120
#define BS_TTE 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121
#define BS_STRIKE 2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122
#define BS_RATE 3, 11, 19, 27, 35, 43, 51, 59, 67, 75, 83, 91, 99, 107, 115, 123
#define BS_IV 4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124
#define BS_VOL 5, 13, 21, 29, 37, 45, 53, 61, 69, 77, 85, 93, 101, 109, 117, 125
#define BS_PX 6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110, 118, 126
#define BS_THEO 7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111, 119, 127

#define THRD (32)

// these are all standard layout types, so type punning on inactive members is okay?
union alignas(64) V16 {
    Vec16f vcl;
    float array[16];
    __m512 intr;
};

struct alignas(32) bs
{
  public:
    float ul, tte, strike, rate, iv, vol, px, theo;
};

struct alignas(4096) bsv
{
  public:
    bsv()
    {
        ul = std::make_unique<float[]>(SIZE_N);
        tte = std::make_unique<float[]>(SIZE_N);
        strike = std::make_unique<float[]>(SIZE_N);
        rate = std::make_unique<float[]>(SIZE_N);
        vol = std::make_unique<float[]>(SIZE_N);
        iv = std::make_unique<float[]>(SIZE_N);
        px = std::make_unique<float[]>(SIZE_N);
        theo = std::make_unique<float[]>(SIZE_N);
    }

    std::unique_ptr<float[]> ul;
    std::unique_ptr<float[]> tte;
    std::unique_ptr<float[]> strike;
    std::unique_ptr<float[]> rate;
    std::unique_ptr<float[]> iv;
    std::unique_ptr<float[]> vol;
    std::unique_ptr<float[]> px;
    std::unique_ptr<float[]> theo;
};

struct alignas(4096) bsv512
{
  public:
    bsv512()
    {
        ul = std::make_unique<V16[]>(SIZE_N);
        tte = std::make_unique<V16[]>(SIZE_N);
        strike = std::make_unique<V16[]>(SIZE_N);
        rate = std::make_unique<V16[]>(SIZE_N);
        vol = std::make_unique<V16[]>(SIZE_N);
        iv = std::make_unique<V16[]>(SIZE_N);
        px = std::make_unique<V16[]>(SIZE_N);
        theo = std::make_unique<V16[]>(SIZE_N);
    }

    std::unique_ptr<V16[]> ul;
    std::unique_ptr<V16[]> tte;
    std::unique_ptr<V16[]> strike;
    std::unique_ptr<V16[]> rate;
    std::unique_ptr<V16[]> iv;
    std::unique_ptr<V16[]> vol;
    std::unique_ptr<V16[]> px;
    std::unique_ptr<V16[]> theo;
};

static void iv_naive_bsv(benchmark::State &state)
{
    std::srand(1);
    bsv data;

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data.ul[i] = 100.0;
        data.tte[i] = 0.3;
        data.strike[i] = 110.0;
        data.rate[i] = 0.05;
        data.vol[i] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.px[i] = bsPrice(data.ul[i], data.tte[i], data.strike[i], data.rate[i], data.vol[i]);
    }

    for (auto _ : state)
    {
        for (auto i = 0; i < SIZE_N; ++i)
        {
            data.iv[i] = bisectIV(data.ul[i], data.tte[i], data.strike[i], data.rate[i], data.px[i]);
        }
    }

    for (auto i = 0; i < SIZE_N; ++i)
    {
        assert(std::abs(data.iv[i] - data.vol[i]) <= 1e-4);
    }
}
BENCHMARK(iv_naive_bsv);

static void iv_naive_bs(benchmark::State &state)
{
    std::srand(1);
    alignas(4096) auto data = std::make_unique<bs[]>(SIZE_N);

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data[i].ul = 100.0;
        data[i].tte = 0.3;
        data[i].strike = 110.0;
        data[i].rate = 0.05;
        data[i].vol = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data[i].px = bsPrice(data[i].ul, data[i].tte, data[i].strike, data[i].rate, data[i].vol);
    }

    for (auto _ : state)
    {
        for (auto i = 0; i < SIZE_N; ++i)
        {
            data[i].iv = bisectIV(data[i].ul, data[i].tte, data[i].strike, data[i].rate, data[i].px);
        }
    }

    for (auto i = 0; i < SIZE_N; ++i)
    {
        assert(std::abs(data[i].iv - data[i].vol) <= 1e-4);
    }
}
BENCHMARK(iv_naive_bs);

static void iv_avx_bsv(benchmark::State &state)
{
    std::srand(1);
    alignas(4096) bsv data;

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data.ul[i] = 100.0;
        data.tte[i] = 0.3;
        data.strike[i] = 110.0;
        data.rate[i] = 0.05;
        data.vol[i] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.px[i] = bsPrice(data.ul[i], data.tte[i], data.strike[i], data.rate[i], data.vol[i]);
        data.iv[i] = 0.0;
    }

    for (auto _ : state)
    {
        Vec16f u, t, s, r, p;
        for (auto i = 0; i < SIZE_N; i += 16)
        {
            u.load(data.ul.get() + i);
            t.load(data.tte.get() + i);
            s.load(data.strike.get() + i);
            r.load(data.rate.get() + i);
            p.load(data.px.get() + i);
            bisectIVVec(u, t, s, r, p).store(data.iv.get() + i);
        }
    }

    for (auto i = 0; i < SIZE_N; ++i)
        assert(std::abs(data.iv[i] - data.vol[i]) <= 1e-4);
}
BENCHMARK(iv_avx_bsv);

static void iv_avx_bsv_omp(benchmark::State &state)
{
    std::srand(1);
    bsv data;

    omp_set_num_threads(THRD);

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data.ul[i] = 100.0;
        data.tte[i] = 0.3;
        data.strike[i] = 110.0;
        data.rate[i] = 0.05;
        data.vol[i] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.px[i] = bsPrice(data.ul[i], data.tte[i], data.strike[i], data.rate[i], data.vol[i]);
        data.iv[i] = 0.0;
    }

    const size_t N = SIZE_N / (THRD);

    for (auto _ : state)
    {
#pragma omp parallel
        {
            size_t ii = omp_get_thread_num();
            Vec16f u, t, s, r, p;
            for (auto i = N * ii; i < (ii + 1) * N; i += 16)
            {
                t.load(data.tte.get() + i);
                bisectIVVec(u.load(data.ul.get() + i), t, s.load(data.strike.get() + i), r.load(data.rate.get() + i),
                            p.load(data.px.get() + i))
                    .store(data.iv.get() + i);
            }
        }
    }

    for (auto i = 0; i < SIZE_N; ++i)
        assert(std::abs(data.iv[i] - data.vol[i]) <= 1e-4);
}
BENCHMARK(iv_avx_bsv_omp);

static void iv_avx_bsv512(benchmark::State &state)
{
    std::srand(1);
    bsv512 data;

    for (auto i = 0; i < SIZE_N / 16; ++i)
    {
        for (int j = 0; j < 16; j++)
        {
            data.ul[i].array[j] = 100.0;
            data.tte[i].array[j] = 0.3;
            data.strike[i].array[j] = 110.0;
            data.rate[i].array[j] = 0.05;
            data.vol[i].array[j] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            data.px[i].array[j] = bsPrice(data.ul[i].array[j], data.tte[i].array[j], data.strike[i].array[j],
                                          data.rate[i].array[j], data.vol[i].array[j]);
            data.iv[i].array[j] = 0.0;
        }
    }

    for (auto _ : state)
    {
        for (auto i = 0; i < SIZE_N / 16; i++)
            data.iv[i].vcl =
                bisectIVVec(data.ul[i].vcl, data.tte[i].vcl, data.strike[i].vcl, data.rate[i].vcl, data.px[i].vcl);
    }

    for (auto i = 0; i < SIZE_N / 16; ++i)
        for (auto j = 0; j < 16; ++j)
            assert(std::abs(data.iv[i].array[j] - data.vol[i].array[j]) <= 1e-4);
}
BENCHMARK(iv_avx_bsv512);

static void iv_avx_bsv512_omp(benchmark::State &state)
{
    std::srand(1);
    bsv512 data;

    omp_set_num_threads(THRD);

    for (auto i = 0; i < SIZE_N / 16; ++i)
    {
        for (int j = 0; j < 16; j++)
        {
            data.ul[i].array[j] = 100.0;
            data.tte[i].array[j] = 0.3;
            data.strike[i].array[j] = 110.0;
            data.rate[i].array[j] = 0.05;
            data.vol[i].array[j] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            data.px[i].array[j] = bsPrice(data.ul[i].array[j], data.tte[i].array[j], data.strike[i].array[j],
                                          data.rate[i].array[j], data.vol[i].array[j]);
            data.iv[i].array[j] = 0.0;
        }
    }

    const size_t N = SIZE_N / (16 * THRD);

    for (auto _ : state)
    {
#pragma omp parallel
        {
            size_t ii = omp_get_thread_num();
            for (auto i = ii * N; i < (ii + 1) * N; i++)
                data.iv[i].vcl =
                    bisectIVVec(data.ul[i].vcl, data.tte[i].vcl, data.strike[i].vcl, data.rate[i].vcl, data.px[i].vcl);
        }
    }

    for (auto i = 0; i < SIZE_N / 16; ++i)
        for (auto j = 0; j < 16; ++j)
            assert(std::abs(data.iv[i].array[j] - data.vol[i].array[j]) <= 1e-4);
}
BENCHMARK(iv_avx_bsv512_omp);

static void iv_avx_bs(benchmark::State &state)
{
    std::srand(1);
    alignas(4096) auto data = std::make_unique<bs[]>(SIZE_N);

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data[i].ul = 100.0;
        data[i].tte = 0.3;
        data[i].strike = 110.0;
        data[i].rate = 0.05;
        data[i].iv = 0.0;
        data[i].vol = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data[i].px = bsPrice(data[i].ul, data[i].tte, data[i].strike, data[i].rate, data[i].vol);
        data[i].theo = 0.0;
    }

    for (auto _ : state)
    {
        for (auto i = 0; i < SIZE_N; i += 16)
        {
            auto ul = gather16f<BS_UL>(data.get() + i);
            auto tte = gather16f<BS_TTE>(data.get() + i);
            auto str = gather16f<BS_STRIKE>(data.get() + i);
            auto rate = gather16f<BS_RATE>(data.get() + i);
            auto px = gather16f<BS_PX>(data.get() + i);
            scatter<BS_IV>(bisectIVVec(ul, tte, str, rate, px), (float *)(data.get() + i));
        }
    }

    for (auto i = 0; i < SIZE_N; ++i)
        assert(std::abs(data[i].iv - data[i].vol) <= 1e-4);
}
BENCHMARK(iv_avx_bs);

static void iv_avx_bs_omp(benchmark::State &state)
{
    std::srand(1);
    alignas(4096) auto data = std::make_unique<bs[]>(SIZE_N);

    omp_set_num_threads(THRD);

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data[i].ul = 100.0;
        data[i].tte = 0.3;
        data[i].strike = 110.0;
        data[i].rate = 0.05;
        data[i].vol = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data[i].px = bsPrice(data[i].ul, data[i].tte, data[i].strike, data[i].rate, data[i].vol);
    }

    const size_t N = SIZE_N / THRD;

    for (auto _ : state)
    {
#pragma omp parallel
        {
            size_t ii = omp_get_thread_num();
            for (auto i = ii * N; i < (ii + 1) * N; i += 16)
            {
                auto ul = gather16f<BS_UL>(data.get() + i);
                auto tte = gather16f<BS_TTE>(data.get() + i);
                auto str = gather16f<BS_STRIKE>(data.get() + i);
                auto rate = gather16f<BS_RATE>(data.get() + i);
                auto px = gather16f<BS_PX>(data.get() + i);
                scatter<BS_IV>(bisectIVVec(ul, tte, str, rate, px), (float *)(data.get() + i));
            }
        }
    }

    for (auto i = 0; i < SIZE_N; ++i)
        assert(std::abs(data[i].iv - data[i].vol) <= 1e-4);
}
BENCHMARK(iv_avx_bs_omp);

static void pricer_naive_bsv(benchmark::State &state)
{
    std::srand(1);
    bsv data;

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data.ul[i] = 100.0;
        data.tte[i] = 0.3;
        data.strike[i] = 110.0;
        data.rate[i] = 0.05;
        data.vol[i] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    for (auto _ : state)
    {

        for (auto i = 0; i < SIZE_N; ++i)
            data.px[i] = bsPrice(data.ul[i], data.tte[i], data.strike[i], data.rate[i], data.vol[i]);
    }
}
BENCHMARK(pricer_naive_bsv);

static void pricer_naive_bs(benchmark::State &state)
{
    std::srand(1);
    alignas(4096) auto data = std::make_unique<bs[]>(SIZE_N);

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data[i].ul = 100.0;
        data[i].tte = 0.3;
        data[i].strike = 110.0;
        data[i].rate = 0.05;
        data[i].vol = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    for (auto _ : state)
    {

        for (auto i = 0; i < SIZE_N; ++i)
            data[i].px = bsPrice(data[i].ul, data[i].tte, data[i].strike, data[i].rate, data[i].vol);
    }
}
BENCHMARK(pricer_naive_bs);

static void pricer_avx_bsv(benchmark::State &state)
{
    std::srand(1);
    bsv data;

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data.ul[i] = 100.0;
        data.tte[i] = 0.3;
        data.strike[i] = 110.0;
        data.rate[i] = 0.05;
        data.vol[i] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.theo[i] = bsPrice(data.ul[i], data.tte[i], data.strike[i], data.rate[i], data.vol[i]);
    }

    for (auto _ : state)
    {
        Vec16f u, t, s, r, v;
        for (auto i = 0; i < SIZE_N; i += 16)
        {
            v.load(data.vol.get() + i);
            t.load(data.tte.get() + i);
            bsPriceVec(u.load(data.ul.get() + i), t, s.load(data.strike.get() + i), r.load(data.rate.get() + i), v)
                .store(data.px.get() + i);
        }
    }

    for (auto i = 0; i < SIZE_N; ++i)
        assert(std::abs(data.theo[i] - data.px[i]) <= 1e-4);
}
BENCHMARK(pricer_avx_bsv);

static void pricer_avx_bsv_omp(benchmark::State &state)
{
    std::srand(1);
    bsv data;

    omp_set_num_threads(THRD);

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data.ul[i] = 100.0;
        data.tte[i] = 0.3;
        data.strike[i] = 110.0;
        data.rate[i] = 0.05;
        data.vol[i] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    const size_t N = SIZE_N / (THRD);

    for (auto _ : state)
    {
#pragma omp parallel
        {
            size_t ii = omp_get_thread_num();
            Vec16f u, t, s, r, v;
            for (auto i = N * ii; i < (ii + 1) * N; i += 16)
            {
                v.load(data.vol.get() + i);
                t.load(data.tte.get() + i);
                bsPriceVec(u.load(data.ul.get() + i), t, s.load(data.strike.get() + i), r.load(data.rate.get() + i), v)
                    .store(data.px.get() + i);
            }
        }
    }
}
BENCHMARK(pricer_avx_bsv_omp);

static void pricer_avx_bsv512(benchmark::State &state)
{
    std::srand(1);
    bsv512 data;

    for (auto i = 0; i < SIZE_N / 16; ++i)
    {
        data.ul[i].vcl = 100.0;
        data.tte[i].vcl = 0.3;
        data.strike[i].vcl = 110.0;
        data.rate[i].vcl = 0.05;
        data.vol[i].vcl = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.theo[i].vcl = bsPrice(data.ul[i].array[0], data.tte[i].array[0], data.strike[i].array[0],
                                   data.rate[i].array[0], data.vol[i].array[0]);
    }

    for (auto _ : state)
    {
        for (auto i = 0; i < SIZE_N / 16; i++)
            data.px[i].vcl =
                bsPriceVec(data.ul[i].vcl, data.tte[i].vcl, data.strike[i].vcl, data.rate[i].vcl, data.vol[i].vcl);
    }

    for (auto i = 0; i < SIZE_N / 16; ++i)
        for (int j = 0; j < 16; ++j)
            assert(std::abs(data.theo[i].array[j] - data.px[i].array[j]) <= 1e-4);
}
BENCHMARK(pricer_avx_bsv512);

static void pricer_avx_bsv512_omp(benchmark::State &state)
{
    std::srand(1);
    bsv512 data;

    omp_set_num_threads(THRD);

    for (auto i = 0; i < SIZE_N / 16; ++i)
    {
        data.ul[i].vcl = 100.0;
        data.tte[i].vcl = 0.3;
        data.strike[i].vcl = 110.0;
        data.rate[i].vcl = 0.05;
        data.vol[i].vcl = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    const size_t N = SIZE_N / (16 * THRD);

    for (auto _ : state)
    {
#pragma omp parallel
        {
            size_t ii = omp_get_thread_num();

            for (auto i = ii * N; i < (ii + 1) * N; i++)
                data.px[i].vcl =
                    bsPriceVec(data.ul[i].vcl, data.tte[i].vcl, data.strike[i].vcl, data.rate[i].vcl, data.vol[i].vcl);
        }
    }
}
BENCHMARK(pricer_avx_bsv512_omp);

static void pricer_avx_bs(benchmark::State &state)
{
    std::srand(1);
    alignas(4096) auto data = std::make_unique<bs[]>(SIZE_N);

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data[i].ul = 100.0;
        data[i].tte = 0.3;
        data[i].strike = 110.0;
        data[i].rate = 0.05;
        data[i].vol = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    for (auto _ : state)
    {
        for (auto i = 0; i < SIZE_N; i += 16)
        {
            auto ul = gather16f<BS_UL>(data.get() + i);
            auto tte = gather16f<BS_TTE>(data.get() + i);
            auto str = gather16f<BS_STRIKE>(data.get() + i);
            auto rate = gather16f<BS_RATE>(data.get() + i);
            auto vol = gather16f<BS_VOL>(data.get() + i);
            scatter<BS_THEO>(bsPriceVec(ul, tte, str, rate, vol), (float *)(data.get() + i));
        }
    }
}
BENCHMARK(pricer_avx_bs);

static void pricer_avx_bs_omp(benchmark::State &state)
{
    std::srand(1);
    alignas(4096) auto data = std::make_unique<bs[]>(SIZE_N);

    omp_set_num_threads(THRD);

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data[i].ul = 100.0;
        data[i].tte = 0.3;
        data[i].strike = 110.0;
        data[i].rate = 0.05;
        data[i].vol = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    const size_t N = SIZE_N / THRD;

    for (auto _ : state)
    {
#pragma omp parallel
        {
            size_t ii = omp_get_thread_num();
            for (auto i = ii * N; i < (ii + 1) * N; i += 16)
            {
                auto ul = gather16f<BS_UL>(data.get() + i);
                auto tte = gather16f<BS_TTE>(data.get() + i);
                auto str = gather16f<BS_STRIKE>(data.get() + i);
                auto rate = gather16f<BS_RATE>(data.get() + i);
                auto vol = gather16f<BS_VOL>(data.get() + i);
                scatter<BS_THEO>(bsPriceVec(ul, tte, str, rate, vol), (float *)(data.get() + i));
            }
        }
    }
}
BENCHMARK(pricer_avx_bs_omp);

static void vol_edge_naive_bsv512(benchmark::State &state)
{
    std::srand(1);
    bsv512 data;

    for (auto i = 0; i < SIZE_N / 16; ++i)
    {
        data.iv[i].vcl = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.vol[i].vcl = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    auto theo = (float *__restrict__)data.theo.get();
    auto iv = (float *__restrict__)data.iv.get();
    auto vol = (float *__restrict__)data.vol.get();

    for (auto _ : state)
    {
        for (auto i = 0; i < SIZE_N; ++i)
            theo[i] = std::abs(iv[i] - vol[i]);
    }
}
BENCHMARK(vol_edge_naive_bsv512);

static void vol_edge_naive_bsv(benchmark::State &state)
{
    std::srand(1);
    bsv data;

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data.iv[i] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.vol[i] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    for (auto _ : state)
    {

        for (auto i = 0; i < SIZE_N; ++i)
            data.theo[i] = std::abs(data.iv[i] - data.vol[i]);
    }
}
BENCHMARK(vol_edge_naive_bsv);

static void vol_edge_naive_bs(benchmark::State &state)
{
    std::srand(1);
    alignas(4096) bs *__restrict__ data = new bs[SIZE_N];

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data[i].iv = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data[i].vol = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    for (auto _ : state)
    {

        for (auto i = 0; i < SIZE_N; ++i)
            data[i].theo = std::abs(data[i].iv - data[i].vol);
    }
}
BENCHMARK(vol_edge_naive_bs);

static void vol_edge_avx_bsv(benchmark::State &state)
{
    std::srand(1);
    bsv data;

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data.iv[i] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.vol[i] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    for (auto _ : state)
    {
        Vec16f v, vi;
        for (auto i = 0; i < SIZE_N; i += 16)
            abs(v.load(data.vol.get() + i) - vi.load(data.iv.get() + i)).store(data.theo.get() + i);
    }
}
BENCHMARK(vol_edge_avx_bsv);

static void vol_edge_avx_unrolled_bsv(benchmark::State &state)
{
    std::srand(1);
    bsv data;

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data.iv[i] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.vol[i] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.theo[i] = 0.0;
    }

    for (auto _ : state)
    {
        Vec16f v0, iv0;
        Vec16f v1, iv1;
        Vec16f v2, iv2;
        Vec16f v3, iv3;
        Vec16f v4, iv4;
        Vec16f v5, iv5;
        Vec16f v6, iv6;
        Vec16f v7, iv7;
        for (auto i = 0; i < SIZE_N; i += 128)
        {
            v0.load(data.vol.get() + i);
            v1.load(data.vol.get() + i + 16);
            v2.load(data.vol.get() + i + 32);
            v3.load(data.vol.get() + i + 48);
            v4.load(data.vol.get() + i + 64);
            v5.load(data.vol.get() + i + 80);
            v6.load(data.vol.get() + i + 96);
            v7.load(data.vol.get() + i + 112);
            iv0.load(data.iv.get() + i);
            iv1.load(data.iv.get() + i + 16);
            iv2.load(data.iv.get() + i + 32);
            iv3.load(data.iv.get() + i + 48);
            iv4.load(data.iv.get() + i + 64);
            iv5.load(data.iv.get() + i + 80);
            iv6.load(data.iv.get() + i + 96);
            iv7.load(data.iv.get() + i + 112);
            v0 -= iv0;
            v1 -= iv1;
            v2 -= iv2;
            v3 -= iv3;
            v4 -= iv4;
            v5 -= iv5;
            v6 -= iv6;
            v7 -= iv7;
            iv0 = abs(v0);
            iv1 = abs(v1);
            iv2 = abs(v2);
            iv3 = abs(v3);
            iv4 = abs(v4);
            iv5 = abs(v5);
            iv6 = abs(v6);
            iv7 = abs(v7);
            iv0.store(data.theo.get() + i);
            iv1.store(data.theo.get() + i + 16);
            iv2.store(data.theo.get() + i + 32);
            iv3.store(data.theo.get() + i + 48);
            iv4.store(data.theo.get() + i + 64);
            iv5.store(data.theo.get() + i + 80);
            iv6.store(data.theo.get() + i + 96);
            iv7.store(data.theo.get() + i + 112);
        }
    }
}
BENCHMARK(vol_edge_avx_unrolled_bsv);

static void vol_edge_avx_bsv512(benchmark::State &state)
{
    std::srand(1);
    bsv512 data;

    for (auto i = 0; i < SIZE_N / 16; ++i)
    {
        data.iv[i].vcl = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.vol[i].vcl = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    for (auto _ : state)
    {

        for (auto i = 0; i < SIZE_N / 16; ++i)
            data.theo[i].vcl = abs(data.vol[i].vcl - data.iv[i].vcl);
    }
}
BENCHMARK(vol_edge_avx_bsv512);

static void vol_edge_avx_unrolled_bsv512(benchmark::State &state)
{
    std::srand(1);
    bsv512 data;

    for (auto i = 0; i < SIZE_N / 16; ++i)
    {
        data.iv[i].vcl = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.vol[i].vcl = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.theo[i].vcl = 0.0;
    }

    for (auto _ : state)
    {
        Vec16f v0, iv0;
        Vec16f v1, iv1;
        Vec16f v2, iv2;
        Vec16f v3, iv3;
        Vec16f v4, iv4;
        Vec16f v5, iv5;
        Vec16f v6, iv6;
        Vec16f v7, iv7;
        for (auto i = 0; i < SIZE_N / 16; i += 8)
        {
            v0 = data.vol[i].vcl;
            v1 = data.vol[i + 1].vcl;
            v2 = data.vol[i + 2].vcl;
            v3 = data.vol[i + 3].vcl;
            v4 = data.vol[i + 4].vcl;
            v5 = data.vol[i + 5].vcl;
            v6 = data.vol[i + 6].vcl;
            v7 = data.vol[i + 7].vcl;
            iv0 = data.iv[i].vcl;
            iv1 = data.iv[i + 1].vcl;
            iv2 = data.iv[i + 2].vcl;
            iv3 = data.iv[i + 3].vcl;
            iv4 = data.iv[i + 4].vcl;
            iv5 = data.iv[i + 5].vcl;
            iv6 = data.iv[i + 6].vcl;
            iv7 = data.iv[i + 7].vcl;
            v0 -= iv0;
            v1 -= iv1;
            v2 -= iv2;
            v3 -= iv3;
            v4 -= iv4;
            v5 -= iv5;
            v6 -= iv6;
            v7 -= iv7;
            data.theo[i].vcl = abs(v0);
            data.theo[i + 1].vcl = abs(v1);
            data.theo[i + 2].vcl = abs(v2);
            data.theo[i + 3].vcl = abs(v3);
            data.theo[i + 4].vcl = abs(v4);
            data.theo[i + 5].vcl = abs(v5);
            data.theo[i + 6].vcl = abs(v6);
            data.theo[i + 7].vcl = abs(v7);
        }
    }

    for (int i = 0; i < SIZE_N / 16; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            auto compueted = data.theo[i].array[j];
            auto test = abs(data.iv[i].array[j] - data.vol[i].array[j]);
            assert(abs(compueted - test) < 1e-15);
        }
    }
}
BENCHMARK(vol_edge_avx_unrolled_bsv512);

static void vol_edge_avx_bs(benchmark::State &state)
{
    std::srand(1);
    alignas(2048) auto data = std::make_unique<bs[]>(SIZE_N);

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data[i].vol = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data[i].iv = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    for (auto _ : state)
    {
        for (auto i = 0; i < SIZE_N; i += 16)
        {
            auto vol = gather16f<BS_VOL>(data.get() + i);
            auto iv = gather16f<BS_IV>(data.get() + i);
            scatter<BS_THEO>(abs(iv - vol), (float *)(data.get() + i));
        }
    }
}
BENCHMARK(vol_edge_avx_bs);

static void vol_edge_avx_unrolled_bs(benchmark::State &state)
{
    std::srand(1);
    alignas(2048) auto data = std::make_unique<bs[]>(SIZE_N);

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data[i].vol = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data[i].iv = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    for (auto _ : state)
    {
        Vec16f v0, vi0;
        Vec16f v1, vi1;
        Vec16f v2, vi2;
        Vec16f v3, vi3;
        for (auto i = 0; i < SIZE_N; i += 64)
        {
            v0 = gather16f<BS_VOL>(data.get() + i);
            vi0 = gather16f<BS_IV>(data.get() + i);
            v1 = gather16f<BS_VOL>(data.get() + i + 16);
            vi1 = gather16f<BS_IV>(data.get() + i + 16);
            v2 = gather16f<BS_VOL>(data.get() + i + 32);
            vi2 = gather16f<BS_IV>(data.get() + i + 32);
            v3 = gather16f<BS_VOL>(data.get() + i + 48);
            vi3 = gather16f<BS_IV>(data.get() + i + 48);
            scatter<BS_THEO>(abs(vi0 - v0), (float *)(data.get() + i));
            scatter<BS_THEO>(abs(vi1 - v1), (float *)(data.get() + i + 16));
            scatter<BS_THEO>(abs(vi2 - v2), (float *)(data.get() + i + 32));
            scatter<BS_THEO>(abs(vi3 - v3), (float *)(data.get() + i + 48));
        }
    }
}
BENCHMARK(vol_edge_avx_unrolled_bs);

static void random_writes_bs(benchmark::State &state)
{
    std::srand(1);
    alignas(2048) auto data = std::make_unique<bs[]>(SIZE_N);
    alignas(64) auto ind = std::make_unique<size_t[]>(SIZE_N);

    for (auto i = 0; i < SIZE_N; ++i)
        ind[i] = SIZE_N * rand() / RAND_MAX;

    for (auto _ : state)
    {
        for (auto i = 0; i < SIZE_N;)
        {
            data[ind[i]].vol = 0.5;
            data[ind[i++]].px = 1.0;
            data[ind[i]].vol = 0.5;
            data[ind[i++]].px = 1.0;
            data[ind[i]].vol = 0.5;
            data[ind[i++]].px = 1.0;
            data[ind[i]].vol = 0.5;
            data[ind[i++]].px = 1.0;
        }
    }

    benchmark::DoNotOptimize(data);
}
BENCHMARK(random_writes_bs);

static void random_writes_bsv(benchmark::State &state)
{
    std::srand(1);
    bsv data;
    alignas(64) auto ind = std::make_unique<size_t[]>(SIZE_N);

    for (auto i = 0; i < SIZE_N; ++i)
        ind[i] = SIZE_N * rand() / RAND_MAX;

    for (auto _ : state)
    {
        float *__restrict__ px = data.px.get();
        float *__restrict__ vol = data.vol.get();
        for (auto i = 0; i < SIZE_N;)
        {
            vol[ind[i]] = 0.5;
            px[ind[i++]] = 1.0;
            vol[ind[i]] = 0.5;
            px[ind[i++]] = 1.0;
            vol[ind[i]] = 0.5;
            px[ind[i++]] = 1.0;
            vol[ind[i]] = 0.5;
            px[ind[i++]] = 1.0;
        }
    }

    benchmark::DoNotOptimize(data);
}
BENCHMARK(random_writes_bsv);

BENCHMARK_MAIN();