// Copyright 2023 Matthew Kolbe

#include <cmath>
#include <iomanip>

#define ONE_OVER_SQRT_TWO 0.707106781186547524400844362105

// standard normal CDF
float nCDF(const float &x)
{
    return std::erfc(-x * ONE_OVER_SQRT_TWO) * 0.5;
}

float bsPrice(const bool &iscall, const float &ul, const float &tte, const float &strike, const float &rate,
              const float &vol)
{
    auto vol_sqrt_t = vol * std::sqrt(tte);
    auto d1 = (std::log(ul / strike) + (rate + vol * vol * 0.5) * tte) / vol_sqrt_t;
    auto d2 = d1 - vol_sqrt_t;
    return nCDF(d1) * ul - nCDF(d2) * strike * std::exp(-rate * tte);
}

float bisectIV(const bool &iscall, const float &ul, const float &tte, const float &strike, const float &rate,
               const float &price)
{
    auto low_vol = 0.01f;
    auto high_vol = 2.0f;
    auto mid_vol = 0.5 * (low_vol + high_vol);
    auto mid_val = bsPrice(iscall, ul, tte, strike, rate, mid_vol);
    while (std::abs(mid_val - price) > 1e-4)
    {
        if (price < mid_val)
            high_vol = mid_vol;
        else
            low_vol = mid_vol;

        mid_vol = 0.5 * (low_vol + high_vol);
        mid_val = bsPrice(iscall, ul, tte, strike, rate, mid_vol);
    }

    return mid_vol;
}
