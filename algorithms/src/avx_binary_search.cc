// Copyright 2023 Matthew Kolbe

#include <vector>
#include <math.h>
#include <algorithm>
#include <strings.h>
#include <immintrin.h>
#include <iostream>


inline __attribute__((always_inline)) std::size_t index_match(const short* v, const std::size_t & n, const short & find)
{
    if(n==0) {return 0;}

    auto f = _mm512_set1_epi16(find);
    std::size_t lo = 0;
    std::size_t vec_n = n & 0xFFFFFFE0;
    std::size_t delta = (vec_n - 1) / 2;
    std::size_t midi = delta;
    __mmask32 eqmask;
    __m512i vv;

    while (lo < vec_n) {
        
        delta /= 2;

        // get the aligned index
        midi &= 0xFFFFFFE0;

        // no aligned load for shorts?
        vv = _mm512_loadu_epi16(v + midi);
        eqmask = _mm512_cmp_epi16_mask(vv, f, _MM_CMPINT_EQ);
        if(eqmask != 0) 
            return midi + ffs(eqmask) - 1;

        lo = v[midi] > find ? lo : midi + 32;
        midi = std::min(lo + delta, n-1);
    }
    
    vv = _mm512_mask_loadu_epi16(f, (1u << (n-vec_n)) - 1,v + vec_n);
    eqmask = _mm512_cmp_epi16_mask(vv, f, _MM_CMPINT_EQ);

    return vec_n + ffs(eqmask) - 1;
}


inline __attribute__((always_inline)) __m512i bulk_index_match(const int * __restrict v, const int & n, const __m512i & find) {
    if(n==0) {return _mm512_setzero_epi32();}
    __m512i delta = _mm512_set1_epi32(n / 2);
    auto vv = _mm512_i32gather_epi32(delta, v, 4);
    const __m512i one = _mm512_set1_epi32(1);
    
    __m512i midi = delta;
    
    __mmask16 eqmask, ltmask;

    while(true) {
        delta = _mm512_srli_epi32(delta, 1);
        delta = _mm512_max_epi32(delta, one);
        eqmask = _mm512_cmp_epi32_mask(vv, find, _MM_CMPINT_EQ);
        
        if(eqmask == 65535)
            return midi;

        ltmask = _mm512_cmp_epi32_mask(find, vv, _MM_CMPINT_LT);

        midi = _mm512_mask_add_epi32(midi, ~(ltmask | eqmask), delta, midi);
        midi = _mm512_mask_sub_epi32(midi, ltmask, midi, delta);
        vv = _mm512_i32gather_epi32(midi, v, 4);
    }
}

// todo: cut out avx-512 to see if it helps.
inline __attribute__((always_inline)) std::size_t index_match_no_avx(const int * __restrict v, const std::size_t & n, const int & find)
{
    if(n==0) {return 0;}

    std::size_t lo = 0;
    std::size_t vec_n = n & 0xFFFFFFF0;
    std::size_t delta = (vec_n - 1) / 2;
    std::size_t midi = delta & 0xFFFFFFF0;
    
    __mmask16 eqmask;
    __m512i vv;

    while (lo < vec_n) {
        // get the aligned index
        delta /= 2;
        if(v[midi] > find)
            midi -= delta;
        else if (v[midi + 15] < find)
            midi += delta;
        else {
            int mini_delta = 8;
            
             
        }

        lo = v[midi] > find ? lo : midi + 16;
        midi = std::min(lo + delta, n-1) & 0xFFFFFFF0;
    }
    
    for(;vec_n < n;) {
        if(v[vec_n] == find )
            return vec_n;
        else
            vec_n++;
    }

    throw;
}

inline __attribute__((always_inline)) std::size_t index_match(const int * __restrict v, const std::size_t & n, const int & find)
{
    if(n==0) {return 0;}

    const __m512i f = _mm512_set1_epi32(find);
    std::size_t lo = 0;
    const std::size_t vec_n = n & 0xFFFFFFF0;
    std::size_t delta = (vec_n - 1) / 2;
    std::size_t midi = delta & 0xFFFFFFF0;
    
    __mmask16 eqmask;
    __m512i vv;

    while (lo < vec_n) {
        // get the aligned index
        vv = _mm512_load_epi32(v+midi);
        delta /= 2;
        eqmask = _mm512_cmp_epi32_mask(vv, f, _MM_CMPINT_EQ);
        if(eqmask != 0)
            return midi + ffs(eqmask) - 1;

        lo = v[midi] > find ? lo : midi + 16;
        midi = std::min(lo + delta, n-1) & 0xFFFFFFF0;
    }
    
    vv = _mm512_mask_load_epi32(f, (((unsigned short)1) << (n-vec_n)) - 1,v + vec_n);
    eqmask = _mm512_cmp_epi32_mask(vv, f, _MM_CMPINT_EQ);

    return vec_n + ffs(eqmask) - 1;
}

inline __attribute__((always_inline)) std::size_t index_match(const long long* v, const std::size_t & n, const long long & find)
{
    if(n==0) {return 0;}

    auto f = _mm512_set1_epi64(find);
    std::size_t lo = 0;
    std::size_t vec_n = n & 0xFFFFFFF8;
    std::size_t delta = (vec_n - 1) / 2;
    std::size_t midi = lo + delta;
    __mmask8 eqmask;
    __m512i vv;

    while (lo < vec_n) {

        delta /= 2;

        // get the aligned index
        midi &= 0xFFFFFFF8;
        vv = _mm512_load_epi64(v + midi);
        eqmask = _mm512_cmp_epi64_mask(vv, f, _MM_CMPINT_EQ);
        if(eqmask != 0)
            return midi + ffs(eqmask) - 1;

        lo = v[midi] > find ? lo : midi + 8;
        midi = std::min(lo + delta, n-1);
    }
    
    vv = _mm512_mask_load_epi64(f, (((unsigned char)1) << (n-vec_n)) - 1,v + vec_n);
    eqmask = _mm512_cmp_epi64_mask(vv, f, _MM_CMPINT_EQ);

    return vec_n + ffs(eqmask) - 1;
}
