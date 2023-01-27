// Copyright 2023 Matthew Kolbe

#include "black_scholes.hpp"
#include "ubench.h"
#include "vcl/vectorclass.h"
#include "vec_black_scholes.hpp"
#include <cassert>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <memory>
#include <immintrin.h>
#include <thread>
#include <chrono>
#include <omp.h>

// compile with: g++ main.cpp -std=c++20 -O3 -lm -lstdc++ -march=native -fopenmp
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

static Vec16f WarmAvx(Vec16f seed)
{
    for(int i =0; i<10000; ++i){
        seed *= 1.000001;
    }

    return seed;
}

static double WarmReg(double seed)
{
    for(int i =0; i<10000; ++i){
        seed *= 1.000001;
    }

    return seed;
}

union V16 {
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
        ul = new float[SIZE_N];
        tte = new float[SIZE_N];
        strike = new float[SIZE_N];
        rate = new float[SIZE_N];
        vol = new float[SIZE_N];
        iv = new float[SIZE_N];
        px = new float[SIZE_N];
        theo = new float[SIZE_N];
    }

    ~bsv()
    {
        delete[] ul;
        delete[] tte;
        delete[] strike;
        delete[] rate;
        delete[] vol;
        delete[] iv;
        delete[] px;
        delete[] theo;
    }

    float *__restrict__ ul;
    float *__restrict__ tte;
    float *__restrict__ strike;
    float *__restrict__ rate;
    float *__restrict__ iv;
    float *__restrict__ vol;
    float *__restrict__ px;
    float *__restrict__ theo;
};

struct alignas(4096) bsv512
{
  public:
    bsv512()
    {
        ul      = new V16[SIZE_N / 16];
        tte     = new V16[SIZE_N / 16];
        strike  = new V16[SIZE_N / 16];
        rate    = new V16[SIZE_N / 16];
        vol     = new V16[SIZE_N / 16];
        iv      = new V16[SIZE_N / 16];
        px      = new V16[SIZE_N / 16];
        theo    = new V16[SIZE_N / 16];
    }

    ~bsv512()
    {
        delete[] ul;
        delete[] tte;
        delete[] strike;
        delete[] rate;
        delete[] vol;
        delete[] iv;
        delete[] px;
        delete[] theo;
    }

    V16 *__restrict__ ul;
    V16 *__restrict__ tte;
    V16 *__restrict__ strike;
    V16 *__restrict__ rate;
    V16 *__restrict__ iv;
    V16 *__restrict__ vol;
    V16 *__restrict__ px;
    V16 *__restrict__ theo;
};

UBENCH_MAIN();

UBENCH_EX(iv, naive_bsv)
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

    double warm = WarmReg(1.0);
    UBENCH_DO_NOTHING(&warm);

    UBENCH_DO_BENCHMARK()
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

UBENCH_EX(iv, naive_bs)
{
    std::srand(1);
    alignas(4096) bs *__restrict__ data = new bs[SIZE_N];

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data[i].ul = 100.0;
        data[i].tte = 0.3;
        data[i].strike = 110.0;
        data[i].rate = 0.05;
        data[i].vol = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data[i].px = bsPrice(data[i].ul, data[i].tte, data[i].strike, data[i].rate, data[i].vol);
    }

    double warm = WarmReg(1.0);
    UBENCH_DO_NOTHING(&warm);

    UBENCH_DO_BENCHMARK()
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

    delete[] data;
}

UBENCH_EX(iv, avx_bsv)
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

    Vec16f warm = WarmAvx(1.0);
    UBENCH_DO_NOTHING(&warm);;

    UBENCH_DO_BENCHMARK()
    {
        Vec16f u, t, s, r, p;
        for (auto i = 0; i < SIZE_N; i += 16)
        {
            u.load(data.ul + i);
            t.load(data.tte + i);
            s.load(data.strike + i);
            r.load(data.rate + i);
            p.load(data.px + i);
            bisectIVVec(u, t, s, r, p).store(data.iv + i);
        }
    }

    for (auto i = 0; i < SIZE_N; ++i)
        assert(std::abs(data.iv[i] - data.vol[i]) <= 1e-4);
}

UBENCH_EX(iv, avx_bsv_omp)
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

    Vec16f warm = WarmAvx(1.0);
    UBENCH_DO_NOTHING(&warm);

    const size_t N = SIZE_N / (THRD);

    UBENCH_DO_BENCHMARK()
    {
#pragma omp parallel
        {
            size_t ii = omp_get_thread_num();
            Vec16f u, t, s, r, p;
            for (auto i = N * ii; i < (ii + 1) * N; i += 16)
            {
                t.load(data.tte + i);
                bisectIVVec(u.load(data.ul + i), t, s.load(data.strike + i), r.load(data.rate + i), p.load(data.px + i))
                    .store(data.iv + i);
            }
        }
    }

    for (auto i = 0; i < SIZE_N; ++i)
        assert(std::abs(data.iv[i] - data.vol[i]) <= 1e-4);
}

UBENCH_EX(iv, avx_bsv512)
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
        data.px[i].vcl = bsPriceVec(data.ul[i].vcl, data.tte[i].vcl, data.strike[i].vcl, data.rate[i].vcl, data.vol[i].vcl);
        data.iv[i].vcl = 0.0;
    }

    Vec16f warm = WarmAvx(1.0);
    UBENCH_DO_NOTHING(&warm);

    UBENCH_DO_BENCHMARK()
    {
        for (auto i = 0; i < SIZE_N / 16; i++)
            data.iv[i].vcl = bisectIVVec(data.ul[i].vcl, data.tte[i].vcl, data.strike[i].vcl, data.rate[i].vcl, data.px[i].vcl);
    }

    for (auto i = 0; i < SIZE_N / 16; ++i)
        for (auto j = 0; j < 16; ++j)
            assert(std::abs(data.iv[i].array[j] - data.vol[i].array[j]) <= 1e-4);
}

UBENCH_EX(iv, avx_bsv512_omp)
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
        data.px[i].vcl = bsPriceVec(data.ul[i].vcl, data.tte[i].vcl, data.strike[i].vcl, data.rate[i].vcl, data.vol[i].vcl);
        data.iv[i].vcl = 0.0;
    }
    const size_t N = SIZE_N / (16 * THRD);

    Vec16f warm = WarmAvx(1.0);
    UBENCH_DO_NOTHING(&warm);

    UBENCH_DO_BENCHMARK()
    {
#pragma omp parallel
        {
            size_t ii = omp_get_thread_num();
            for (auto i = ii * N; i < (ii + 1) * N; i++)
                data.iv[i].vcl = bisectIVVec(data.ul[i].vcl, data.tte[i].vcl, data.strike[i].vcl, data.rate[i].vcl, data.px[i].vcl);
        }
    }

    for (auto i = 0; i < SIZE_N / 16; ++i)
        for (auto j = 0; j < 16; ++j)
            assert(std::abs(data.iv[i].array[j] - data.vol[i].array[j]) <= 1e-4);
}

UBENCH_EX(iv, avx_bs)
{
    std::srand(1);
    alignas(4096) bs *__restrict__ data = new bs[SIZE_N];

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

    Vec16f warm = WarmAvx(1.0);
    UBENCH_DO_NOTHING(&warm);

    UBENCH_DO_BENCHMARK()
    {
        for (auto i = 0; i < SIZE_N; i += 16)
        {
            auto ul = gather16f<BS_UL>(data + i);
            auto tte = gather16f<BS_TTE>(data + i);
            auto str = gather16f<BS_STRIKE>(data + i);
            auto rate = gather16f<BS_RATE>(data + i);
            auto px = gather16f<BS_PX>(data + i);
            scatter<BS_IV>(bisectIVVec(ul, tte, str, rate, px), (float *)(data + i));
        }
    }

    for (auto i = 0; i < SIZE_N; ++i)
        assert(std::abs(data[i].iv - data[i].vol) <= 1e-4);

    delete[] data;
}

UBENCH_EX(iv, avx_bs_omp)
{
    std::srand(1);
    alignas(4096) bs *__restrict__ data = new bs[SIZE_N];

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
    Vec16f warm = WarmAvx(1.0);
    UBENCH_DO_NOTHING(&warm);

    UBENCH_DO_BENCHMARK()
    {
#pragma omp parallel
        {
            size_t ii = omp_get_thread_num();
            for (auto i = ii * N; i < (ii + 1) * N; i += 16)
            {
                auto ul = gather16f<BS_UL>(data + i);
                auto tte = gather16f<BS_TTE>(data + i);
                auto str = gather16f<BS_STRIKE>(data + i);
                auto rate = gather16f<BS_RATE>(data + i);
                auto px = gather16f<BS_PX>(data + i);
                scatter<BS_IV>(bisectIVVec(ul, tte, str, rate, px), (float *)(data + i));
            }
        }
    }

    for (auto i = 0; i < SIZE_N; ++i)
        assert(std::abs(data[i].iv - data[i].vol) <= 1e-4);

    delete[] data;
}

UBENCH_EX(pricer, naive_bsv)
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

    double warm = WarmReg(1.0);
    UBENCH_DO_NOTHING(&warm);

    UBENCH_DO_BENCHMARK()
    {

        for (auto i = 0; i < SIZE_N; ++i)
            data.px[i] = bsPrice(data.ul[i], data.tte[i], data.strike[i], data.rate[i], data.vol[i]);
    }
}

UBENCH_EX(pricer, naive_bs)
{
    std::srand(1);
    alignas(4096) bs *__restrict__ data = new bs[SIZE_N];

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data[i].ul = 100.0;
        data[i].tte = 0.3;
        data[i].strike = 110.0;
        data[i].rate = 0.05;
        data[i].vol = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    double warm = WarmReg(1.0);
    UBENCH_DO_NOTHING(&warm);

    UBENCH_DO_BENCHMARK()
    {

        for (auto i = 0; i < SIZE_N; ++i)
            data[i].px = bsPrice(data[i].ul, data[i].tte, data[i].strike, data[i].rate, data[i].vol);
    }

    delete[] data;
}

UBENCH_EX(pricer, avx_bsv)
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

    auto warm = WarmAvx(1.0);
    UBENCH_DO_NOTHING(&warm);

    UBENCH_DO_BENCHMARK()
    {
        Vec16f u, t, s, r, v;
        for (auto i = 0; i < SIZE_N; i += 16)
        {
            v.load(data.vol + i);
            t.load(data.tte + i);

            bsPriceVec(u.load(data.ul + i), t, u.load(data.strike + i), r.load(data.rate + i), v).store(data.px + i);
        }
    }
}

UBENCH_EX(pricer, avx_bsv_omp)
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

    auto warm = WarmAvx(1.0);
    UBENCH_DO_NOTHING(&warm);

    UBENCH_DO_BENCHMARK()
    {
#pragma omp parallel
        {
            size_t ii = omp_get_thread_num();
            Vec16f u, t, s, r, v;
            for (auto i = N * ii; i < (ii + 1) * N; i += 16)
                bsPriceVec(u.load(data.ul + i), t.load(data.tte + i), s.load(data.strike + i), r.load(data.rate + i),
                           v.load(data.vol + i))
                    .store(data.px + i);
        }
    }
}

UBENCH_EX(pricer, avx_bsv512)
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
        data.px[i].vcl = bsPriceVec(data.ul[i].vcl, data.tte[i].vcl, data.strike[i].vcl, data.rate[i].vcl, data.vol[i].vcl);
    }

    auto warm = WarmAvx(1.0);
    UBENCH_DO_NOTHING(&warm);

    UBENCH_DO_BENCHMARK()
    {
        for (auto i = 0; i < SIZE_N / 16; i++)
            data.px[i].vcl = bsPriceVec(data.ul[i].vcl, data.tte[i].vcl, data.strike[i].vcl, data.rate[i].vcl, data.vol[i].vcl);
    }
}

UBENCH_EX(pricer, avx_bsv512_omp)
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

    auto warm = WarmAvx(1.0);
    UBENCH_DO_NOTHING(&warm);

    UBENCH_DO_BENCHMARK()
    {
#pragma omp parallel
        {
            size_t ii = omp_get_thread_num();
            for (auto i = ii * N; i < (ii + 1) * N; i++)
                data.px[i].vcl = bsPriceVec(data.ul[i].vcl, data.tte[i].vcl, data.strike[i].vcl, data.rate[i].vcl, data.vol[i].vcl);
        }
    }
}

UBENCH_EX(pricer, avx_bs)
{
    std::srand(1);
    alignas(4096) bs *__restrict__ data = new bs[SIZE_N];

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data[i].ul = 100.0;
        data[i].tte = 0.3;
        data[i].strike = 110.0;
        data[i].rate = 0.05;
        data[i].vol = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    UBENCH_DO_BENCHMARK()
    {
        for (auto i = 0; i < SIZE_N; i += 16)
        {
            auto ul = gather16f<BS_UL>(data + i);
            auto tte = gather16f<BS_TTE>(data + i);
            auto str = gather16f<BS_STRIKE>(data + i);
            auto rate = gather16f<BS_RATE>(data + i);
            auto vol = gather16f<BS_VOL>(data + i);
            scatter<BS_THEO>(bsPriceVec(ul, tte, str, rate, vol), (float *)(data + i));
        }
    }

    delete[] data;
}

UBENCH_EX(pricer, avx_bs_omp)
{
    std::srand(1);
    alignas(4096) bs *__restrict__ data = new bs[SIZE_N];

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

    auto warm = WarmAvx(1.0);
    UBENCH_DO_NOTHING(&warm);

    UBENCH_DO_BENCHMARK()
    {
#pragma omp parallel
        {
            size_t ii = omp_get_thread_num();
            for (auto i = ii * N; i < (ii + 1) * N; i += 16)
            {
                auto ul = gather16f<BS_UL>(data + i);
                auto tte = gather16f<BS_TTE>(data + i);
                auto str = gather16f<BS_STRIKE>(data + i);
                auto rate = gather16f<BS_RATE>(data + i);
                auto vol = gather16f<BS_VOL>(data + i);
                scatter<BS_THEO>(bsPriceVec(ul, tte, str, rate, vol), (float *)(data + i));
            }
        }
    }

    delete[] data;
}

UBENCH_EX(vol_edge, naive_bsv512)
{
    std::srand(1);
    bsv512 data;

    for (auto i = 0; i < SIZE_N/16; ++i)
    {
        data.iv[i].vcl = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.vol[i].vcl = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    UBENCH_DO_BENCHMARK()
    {
        auto theo = (float*__restrict__)data.theo;
        auto iv = (float*__restrict__)data.iv;
        auto vol = (float*__restrict__)data.vol;
        
        for (auto i = 0; i < SIZE_N; ++i)
            theo[i] = std::abs(iv[i] - vol[i]);
    }
}


UBENCH_EX(vol_edge, naive_bsv)
{
    std::srand(1);
    bsv data;

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data.iv[i] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.vol[i] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    UBENCH_DO_BENCHMARK()
    {

        for (auto i = 0; i < SIZE_N; ++i)
            data.theo[i] = std::abs(data.iv[i] - data.vol[i]);
    }
}


UBENCH_EX(vol_edge, naive_bs)
{
    std::srand(1);
    alignas(4096) bs *__restrict__ data = new bs[SIZE_N];

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data[i].iv = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data[i].vol = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    UBENCH_DO_BENCHMARK()
    {

        for (auto i = 0; i < SIZE_N; ++i)
            data[i].theo = std::abs(data[i].iv - data[i].vol);
    }

    delete[] data;
}

UBENCH_EX(vol_edge, avx_bsv)
{
    std::srand(1);
    bsv data;

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data.iv[i] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.vol[i] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    UBENCH_DO_BENCHMARK()
    {
        Vec16f v, vi;
        for (auto i = 0; i < SIZE_N; i += 16)
            abs(v.load(data.vol + i) - vi.load(data.iv + i)).store(data.theo + i);
    }
}

UBENCH_EX(vol_edge, avx_unrolled_bsv)
{
    std::srand(1);
    bsv data;

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data.iv[i] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.vol[i] = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.theo[i] = 0.0;
    }

    UBENCH_DO_BENCHMARK()
    {
        Vec16f v0, vi0;
        Vec16f v1, vi1;
        Vec16f v2, vi2;
        Vec16f v3, vi3;
        for (auto i = 0; i < SIZE_N; i += 64)
        {
            v0.load(data.vol + i);
            v1.load(data.vol + i + 16);
            v2.load(data.vol + i + 32);
            v3.load(data.vol + i + 48);
            vi0.load(data.iv + i);
            vi1.load(data.iv + i + 16);
            vi2.load(data.iv + i + 32);
            vi3.load(data.iv + i + 48);
            v0 -= vi0;
            v1 -= vi1;
            v2 -= vi2;
            v3 -= vi3;
            vi0 = abs(v0);
            vi1 = abs(v1);
            vi2 = abs(v2);
            vi3 = abs(v3);
            vi0.store(data.theo + i);
            vi1.store(data.theo + i + 16);
            vi2.store(data.theo + i + 32);
            vi3.store(data.theo + i + 48);
        }
    }
}

UBENCH_EX(vol_edge, avx_bsv512)
{
    std::srand(1);
    bsv512 data;

    for (auto i = 0; i < SIZE_N / 16; ++i)
    {
        data.iv[i].vcl = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.vol[i].vcl = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    UBENCH_DO_BENCHMARK()
    {

        for (auto i = 0; i < SIZE_N / 16; ++i)
            data.theo[i].vcl = abs(data.vol[i].vcl - data.iv[i].vcl);
    }
}

UBENCH_EX(vol_edge, avx_unrolled_bsv512)
{
    std::srand(1);
    bsv512 data;

    for (auto i = 0; i < SIZE_N / 16; ++i)
    {
        data.iv[i].vcl = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.vol[i].vcl = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data.theo[i].vcl = 0.0;
    }

    UBENCH_DO_BENCHMARK()
    {
        Vec16f v0;
        Vec16f v1;
        Vec16f v2;
        Vec16f v3;
        for (auto i = 0; i < SIZE_N / 16; i += 4)
        {
            v0 = data.vol[i].vcl;
            v1 = data.vol[i + 1].vcl;
            v2 = data.vol[i + 2].vcl;
            v3 = data.vol[i + 3].vcl;
            v0 -= data.iv[i].vcl;
            v1 -= data.iv[i + 1].vcl;
            v2 -= data.iv[i + 2].vcl;
            v3 -= data.iv[i + 3].vcl;
            data.theo[i].vcl = abs(v0);
            data.theo[i + 1].vcl = abs(v1);
            data.theo[i + 2].vcl = abs(v2);
            data.theo[i + 3].vcl = abs(v3);
        }
    }

    for(int i = 0; i < SIZE_N/16; ++i) {
        for(int j = 0; j < 16; ++j) {
            auto compueted = data.theo[i].array[j];
            auto test = abs(data.iv[i].array[j] - data.vol[i].array[j]);
            assert(abs(compueted - test) < 1e-15);
        }
    }
}

UBENCH_EX(vol_edge, avx_bs)
{
    std::srand(1);
    alignas(2048) bs *__restrict__ data = new bs[SIZE_N];

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data[i].vol = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data[i].iv = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    UBENCH_DO_BENCHMARK()
    {
        for (auto i = 0; i < SIZE_N; i += 16)
        {
            auto vol = gather16f<BS_VOL>(data + i);
            auto iv = gather16f<BS_IV>(data + i);
            scatter<BS_THEO>(abs(iv - vol), (float *)(data + i));
        }
    }

    delete[] data;
}

UBENCH_EX(vol_edge, avx_unrolled_bs)
{
    std::srand(1);
    alignas(2048) bs *__restrict__ data = new bs[SIZE_N];

    for (auto i = 0; i < SIZE_N; ++i)
    {
        data[i].vol = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data[i].iv = 0.2 + 0.4 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    UBENCH_DO_BENCHMARK()
    {
        Vec16f v0, vi0;
        Vec16f v1, vi1;
        Vec16f v2, vi2;
        Vec16f v3, vi3;
        for (auto i = 0; i < SIZE_N; i += 64)
        {
            v0 = gather16f<BS_VOL>(data + i);
            vi0 = gather16f<BS_IV>(data + i);
            v1 = gather16f<BS_VOL>(data + i + 16);
            vi1 = gather16f<BS_IV>(data + i + 16);
            v2 = gather16f<BS_VOL>(data + i + 32);
            vi2 = gather16f<BS_IV>(data + i + 32);
            v3 = gather16f<BS_VOL>(data + i + 48);
            vi3 = gather16f<BS_IV>(data + i + 48);
            scatter<BS_THEO>(abs(vi0 - v0), (float *)(data + i));
            scatter<BS_THEO>(abs(vi1 - v1), (float *)(data + i + 16));
            scatter<BS_THEO>(abs(vi2 - v2), (float *)(data + i + 32));
            scatter<BS_THEO>(abs(vi3 - v3), (float *)(data + i + 48));
        }
    }

    delete[] data;
}