// Copyright 2023 Matthew Kolbe

#include <vectorclass.h>
#include <memory>
#include <new>


template<class T>
struct DeleteAligned
{
    void operator()(T * data) const
    {
        free(data);
    }
};


template<class T>
std::unique_ptr<T[], DeleteAligned<T>> allocate_aligned(int alignment, const size_t length)
{
    // omitted: check minimum alignment, check error
    T * raw = 0;
    // using posix_memalign as an example, could be made platform dependent...
    int error = posix_memalign((void **)&raw, alignment, sizeof(T)*length);
    return std::unique_ptr<T[], DeleteAligned<T>>{raw};
}


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
        ul =      allocate_aligned<float>(64, n);
        tte =     allocate_aligned<float>(64, n);
        strike =  allocate_aligned<float>(64, n);
        rate =    allocate_aligned<float>(64, n);
        vol =     allocate_aligned<float>(64, n);
        iv =      allocate_aligned<float>(64, n);
        px =      allocate_aligned<float>(64, n);
        theo =    allocate_aligned<float>(64, n);
    }

    std::unique_ptr<float[], DeleteAligned<float>> ul;
    std::unique_ptr<float[], DeleteAligned<float>> tte;
    std::unique_ptr<float[], DeleteAligned<float>> strike;
    std::unique_ptr<float[], DeleteAligned<float>> rate;
    std::unique_ptr<float[], DeleteAligned<float>> iv;
    std::unique_ptr<float[], DeleteAligned<float>> vol;
    std::unique_ptr<float[], DeleteAligned<float>> px;
    std::unique_ptr<float[], DeleteAligned<float>> theo;
};

struct alignas(4096) bsv512
{
  public:
    bsv512(const size_t & n)
    {
        ul =      allocate_aligned<V16>(64, n);
        tte =     allocate_aligned<V16>(64, n);
        strike =  allocate_aligned<V16>(64, n);
        rate =    allocate_aligned<V16>(64, n);
        vol =     allocate_aligned<V16>(64, n);
        iv =      allocate_aligned<V16>(64, n);
        px =      allocate_aligned<V16>(64, n);
        theo =    allocate_aligned<V16>(64, n);
    }

    std::unique_ptr<V16[], DeleteAligned<V16>> ul;
    std::unique_ptr<V16[], DeleteAligned<V16>> tte;
    std::unique_ptr<V16[], DeleteAligned<V16>> strike;
    std::unique_ptr<V16[], DeleteAligned<V16>> rate;
    std::unique_ptr<V16[], DeleteAligned<V16>> iv;
    std::unique_ptr<V16[], DeleteAligned<V16>> vol;
    std::unique_ptr<V16[], DeleteAligned<V16>> px;
    std::unique_ptr<V16[], DeleteAligned<V16>> theo;
};