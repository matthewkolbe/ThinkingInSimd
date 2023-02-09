// Copyright 2023 Matthew Kolbe

#include <immintrin.h>
#include <math.h>

#include <iostream>

#include "vcl/vectorclass.h"
#include "vcl/vectormath_exp.h"

#define _USE_MATH_DEFINES

const Vec16f VONE = Vec16f(1.0f);
const Vec16f VNEGATIVE_ZERO = Vec16f(-0.0f);
const Vec16f VE1 = Vec16f(0.254829592f);
const Vec16f VE2 = Vec16f(-0.284496736f);
const Vec16f VE3 = Vec16f(1.421413741f);
const Vec16f VE4 = Vec16f(-1.453152027f);
const Vec16f VE5 = Vec16f(1.061405429f);
const Vec16f VNEGHALF = Vec16f(-0.5f);
const Vec16f ONE_OVER_ROOT2 = Vec16f(0.70710678118f);

inline __attribute__((always_inline)) Vec16f erf(const Vec16f &x)
{
    auto xx = abs(x);
    auto le_mask = (x <= VNEGATIVE_ZERO);
    auto t = VONE / (0.3275911f * xx + VONE);

    auto yy = polynomial_4(t, VE1, VE2, VE3, VE4, VE5);
    yy *= t;
    t = exp(-xx * xx);
    yy = VONE - yy * t;

    return ((!le_mask) & yy) + (le_mask & (-yy));
}

inline __attribute__((always_inline)) Vec16f cdfnorm(const Vec16f &x)
{
    return 0.5f * (1.0f + erf(x * ONE_OVER_ROOT2));
}

inline __attribute__((always_inline)) Vec16f bsPriceVec(const Vec16f &ul, const Vec16f &tte, const Vec16f &strike,
                                                        const Vec16f &rate, const Vec16f &vol)
{
    auto vol_sqrt_t = vol * sqrt(tte);

    auto d1 = (log(ul / strike) + (rate + vol * vol * 0.5f) * tte) / vol_sqrt_t;
    auto d2 = d1 - vol_sqrt_t;
    return (cdfnorm(d1) * ul) - (cdfnorm(d2) * strike * exp(-rate * tte));
}

inline __attribute__((always_inline)) Vec16f bisectIVVec(const Vec16f &ul, const Vec16f &tte, const Vec16f &strike,
                                                         const Vec16f &rate, const Vec16f &price)
{
    auto low_vol = Vec16f(0.01f);
    auto high_vol = Vec16f(2.0f);
    auto mid_vol = Vec16f(0.995f);
    const Vec16f eps = Vec16f(1e-4f);

    auto mid_val = bsPriceVec(ul, tte, strike, rate, mid_vol);
    auto condition = abs(mid_val - price) > 1e-4f;
    while ((__mmask16)(static_cast<Vec16b>(condition)) != 0)
    {
        auto msk = price < mid_val;
        high_vol = (high_vol & (!msk)) + (mid_vol & msk);
        low_vol = (low_vol & msk) + (mid_vol & (!msk));

        mid_vol = ((0.5f * (low_vol + high_vol)) & condition) + ((!condition) & mid_vol);
        mid_val = bsPriceVec(ul, tte, strike, rate, mid_vol);
        condition = abs(mid_val - price) > eps;
    }

    return mid_vol;
}
