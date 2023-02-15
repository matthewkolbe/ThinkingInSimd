// Copyright 2023 Matthew Kolbe

#include <vectorclass.h>
#include <memory>


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
    bsv(const size_t & n)
    {
        ul = std::make_unique<float[]>(n);
        tte = std::make_unique<float[]>(n);
        strike = std::make_unique<float[]>(n);
        rate = std::make_unique<float[]>(n);
        vol = std::make_unique<float[]>(n);
        iv = std::make_unique<float[]>(n);
        px = std::make_unique<float[]>(n);
        theo = std::make_unique<float[]>(n);
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
    bsv512(const size_t & n)
    {
        ul = std::make_unique<V16[]>(n);
        tte = std::make_unique<V16[]>(n);
        strike = std::make_unique<V16[]>(n);
        rate = std::make_unique<V16[]>(n);
        vol = std::make_unique<V16[]>(n);
        iv = std::make_unique<V16[]>(n);
        px = std::make_unique<V16[]>(n);
        theo = std::make_unique<V16[]>(n);
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