#include <immintrin.h>
#include <new>
#include <cstdlib>
#include <functional>
#include <strings.h>
#include <string.h>

// fash serialization:
//   - every value has a 64-bit hash
//   - the lower 32 hash bits are reserved for finding the bucket
//   - the whole 64 bit hash is used for determining equality. see birthday attack,
//     if you want to know if your data is too big for this assumption. For n=190,000,
//     there's a one in a billion chance of a collision. 
//
// bucket logic:
//   - the user passes in a value bit_size, which will determine the size of the
//     storage array: 16 * 2^bit_size. this is chosen so that each bucket can hold
//     16 32-bit keys, and can thus compare equality on 16 elements. so the first 
//     problem that can occur is that we have a bucket with >16 collisions, which 
//     would overflow the bucket allocation. this is much less likely than the chance
//     of the "birthday problem" collision mentioned above, but for bit_size==16, the
//     chance of overflowing the bucket buffer is 2 out of 10,000,000,000. 
//  
//
// math reference: https://crypto.stackexchange.com/questions/27370/formula-for-the-number-of-expected-collisions/27372

template <class K, class V>
class fash128x {
    uint64_t* __restrict m_location;
    V* __restrict m_data;
    unsigned char m_bitsz;
    uint64_t m_sz, m_sz_m1;
    __m512i m_vz_m1;
    const __m512i zero = _mm512_set1_epi64(0ULL);
    const __m512i one = _mm512_set1_epi64(1ULL);
    const __m512i m_a = _mm512_set1_epi64(UINT64_C(0x319642b2d24d8ec3));
    const __m512i m_b = _mm512_set1_epi64(UINT64_C(0x96de1b173f119089));

public: 
    fash128x(unsigned char bit_size) {
        m_bitsz = bit_size;
        m_sz = 1 << (m_bitsz + 1); // 7 - 6 = (bucket size) - (fraction of entries)
        m_sz_m1 = (1<<(m_bitsz-6)) - 1;
        m_vz_m1 = _mm512_set1_epi64(m_sz_m1);
        m_location = new (std::align_val_t(64)) uint64_t[m_sz];
        memset(m_location, 0, m_sz*sizeof(uint64_t));
        m_data = new (std::align_val_t(64)) V[m_sz];
        memset(m_data, 0, m_sz*sizeof(V));
    }

    ~fash128x() {
        ::operator delete[] (m_location, std::align_val_t(64));
        ::operator delete[] (m_data, std::align_val_t(64));
    }

    // inline __attribute__((always_inline))  std::tuple<V, V, V, V, V, V, V, V> at512(__m512i key) const {
    //     auto k = unhash(key);
        
    //     auto bucket = _mm512_slli_epi64(_mm512_and_epi64(k, m_vz_m1), 7);
    //     auto b = _mm512_i64gather_epi64(bucket, m_location, 8);
    //     auto neq = _mm512_cmp_epi64_mask(k, b, _MM_CMPINT_NE);
    //     //std::cout << 0 << " neq " << (int)neq << std::endl;

    //     while(neq) {
    //         bucket = _mm512_mask_add_epi64(bucket, neq, bucket, one);
    //         b = _mm512_i64gather_epi64(bucket, m_location, 8);
    //         neq = _mm512_cmp_epi64_mask(k, b, _MM_CMPINT_NE);
    //         //std::cout << " neq " << (int)neq << std::endl;
    //     }

    //     uint64_t ind[8];
    //     _mm512_storeu_epi64(ind, bucket);

    //     return std::make_tuple(m_data[ind[0]], m_data[ind[1]], m_data[ind[2]], m_data[ind[3]], m_data[ind[4]], m_data[ind[5]], m_data[ind[6]], m_data[ind[7]]);
    // }

    inline __attribute__((always_inline))  V & at_int64v2(const uint64_t & key) {
        const uint64_t k = unhash(key);
        const auto kk = _mm512_set1_epi64(key);
        const uint32_t bucket = (k & m_sz_m1) << 7;
        const unsigned int guess = ((18302628885633695744ULL & k)>>57) << 3;

        for(int i = 0; i < 128; i+=8) {
            const auto idx = bucket + ((i + guess) & 127);
            const auto b0 = _mm512_load_epi64(m_location + idx);
            unsigned short mask0 = _mm512_cmp_epi64_mask(kk, b0, _MM_CMPINT_EQ);
            if(mask0) {
                auto openindex = __builtin_ffs(mask0) - 1;
                return m_data[openindex + idx];
            }
        }

        throw;
    }

    inline __attribute__((always_inline))  V & at_int64(const uint64_t & key) {
        const uint64_t k = unhash(key);
        const auto kk = _mm512_set1_epi64(key);
        const uint32_t bucket = (k & m_sz_m1) << 7;
        unsigned int guess = ((18302628885633695744ULL & k)>>57) << 3;

        for(int i = 0; i < 128; i+=16) {
            auto idx0 = bucket + ((i + guess) & 127);
            auto idx1 = bucket + ((i + 8 + guess) & 127);
            // auto idx2 = bucket + ((i + 16 + guess) & 127);
            // auto idx3 = bucket + ((i + 24 + guess) & 127);
            auto b0 = _mm512_load_epi64(m_location + idx0);
            auto b1 = _mm512_load_epi64(m_location + idx1);
            // auto b2 = _mm512_load_epi64(m_location + idx2);
            // auto b3 = _mm512_load_epi64(m_location + idx3);
            unsigned short mask0 = _mm512_cmp_epi64_mask(kk, b0, _MM_CMPINT_EQ);
            unsigned short mask1 = _mm512_cmp_epi64_mask(kk, b1, _MM_CMPINT_EQ);
            // unsigned short mask2 = _mm512_cmp_epi64_mask(kk, b2, _MM_CMPINT_EQ);
            // unsigned short mask3 = _mm512_cmp_epi64_mask(kk, b3, _MM_CMPINT_EQ);
            if(mask0) {
                auto openindex = __builtin_ffs(mask0) - 1;
                return m_data[openindex + idx0];
            }
            if(mask1) {
                auto openindex = __builtin_ffs(mask1) - 1;
                return m_data[openindex + idx1];
            }
            // if(mask2) {
            //     auto openindex = __builtin_ffs(mask2) - 1;
            //     return m_data[openindex + idx2];
            // }
            // if(mask3) {
            //     auto openindex = __builtin_ffs(mask3) - 1;
            //     return m_data[openindex + idx3];
            // }
        }

        throw;
    }

    inline __attribute__((always_inline)) V & at_no_intrinsic_int64(const uint64_t & key) {
        const auto k = unhash(key);
        unsigned int guess = (18302628885633695744ULL & k)>>57;
        const unsigned int bucket = (k & m_sz_m1) << 7;
        unsigned int guess_next_bucket = (guess >>3) << 3;
        guess &= 7;


        for(int i = 0; i < 16; ++i)
        {
            for(int j = 0; j < 8; ++j)
            {
                auto idx = bucket + guess_next_bucket + ((j + guess) & 7);
                if(m_location[idx] == key) {
                    return m_data[idx];
                }
            }
            guess_next_bucket += 8;
            guess_next_bucket &= 127;
        }
        std::cout << "at_no_intrinsic_int64 key: " << key << std::endl;
        throw;
    }


    void insert_no_intrinsic_int64(const uint64_t & key, V data) {
        const auto k = unhash(key);
        unsigned int guess = (18302628885633695744ULL & k)>>57;
        const unsigned int bucket = (k & m_sz_m1) << 7;
        unsigned int guess_next_bucket = (guess >>3) << 3;
        guess &= 7;

        for(int i = 0; i < 16; ++i)
        {
            for(int j = 0; j < 8; ++j)
            {
                auto idx = bucket + guess_next_bucket + ((j + guess) & 7);
                if(m_location[idx] == 0) {
                    m_location[idx] = key;
                    m_data[idx] = data;
                    return;
                }
            }
            guess_next_bucket += 8;
            guess_next_bucket &= 127;
        }

        std::cout << "insert_no_intrinsic_int64 " << m_location[bucket] << std::endl;
        throw;
    }

    void insert_no_intrinsic(const K & key) {
        const std::size_t k = std::hash<K>{}(key);
        const unsigned int bucket = (k & m_sz_m1) << 7;
        for(auto b = 0; b < 128; ++b)
        {
            if(m_location[bucket + b] == 0) {
                m_location[bucket + b] = key;
                return;
            }
        }

        throw;
    }

    inline __attribute__((always_inline)) uint64_t unhash(uint64_t x) const {
        x = (x ^ (x >> 31) ^ (x >> 62)) * UINT64_C(0x319642b2d24d8ec3);
        x = (x ^ (x >> 27) ^ (x >> 54)) * UINT64_C(0x96de1b173f119089);
        x = x ^ (x >> 30) ^ (x >> 60);
        return x;
    }

    inline __attribute__((always_inline)) __m512i unhash(__m512i x) const {
        auto a = _mm512_srli_epi64(x, 31);
        auto b = _mm512_srli_epi64(x, 62);
        x = _mm512_mullox_epi64(_mm512_xor_epi64(x, _mm512_xor_epi64(a, b)), m_a);
        a = _mm512_srli_epi64(x, 27);
        b = _mm512_srli_epi64(x, 54);
        x = _mm512_mullox_epi64(_mm512_xor_epi64(x, _mm512_xor_epi64(a, b)), m_b);
        a = _mm512_srli_epi64(x, 30);
        b = _mm512_srli_epi64(x, 60);
        return _mm512_xor_epi64(x, _mm512_xor_epi64(a, b));
    }
};



template <class K, class V>
class fash {
    uint64_t* __restrict m_location;
    V* __restrict m_data;
    unsigned char m_bitsz;
    unsigned int m_sz, m_sz_m1;
    __m512i m_vz_m1;
    const __m512i zero = _mm512_set1_epi64(0ULL);
    const __m512i one = _mm512_set1_epi64(1ULL);
    const __m512i m_a = _mm512_set1_epi64(UINT64_C(0x319642b2d24d8ec3));
    const __m512i m_b = _mm512_set1_epi64(UINT64_C(0x96de1b173f119089));

public: 
    fash(unsigned char bit_size) {
        m_bitsz = bit_size;
        m_sz = 1 << (m_bitsz + 4);
        m_sz_m1 = (1<<m_bitsz) - 1;
        m_vz_m1 = _mm512_set1_epi64(m_sz_m1);
        m_location = new (std::align_val_t(64)) uint64_t[m_sz];
        memset(m_location, 0, m_sz*sizeof(uint64_t));
        m_data = new (std::align_val_t(64)) V[m_sz];
        memset(m_data, 0, m_sz*sizeof(V));
    }

    ~fash() {
        ::operator delete[] (m_location, std::align_val_t(64));
        ::operator delete[] (m_data, std::align_val_t(64));
    }

    bool contains(const K & key) const {
        std::size_t k = std::hash<K>{}(key);
        unsigned int bucket = (k & m_sz_m1) << 4;
        auto blo = _mm512_load_epi64(m_location + bucket);
        auto bhi = _mm512_load_epi64(m_location + bucket + 8);
        unsigned short openmasklo = _mm512_cmp_epi64_mask(zero, blo, _MM_CMPINT_EQ);
        unsigned short openmaskhi = _mm512_cmp_epi64_mask(zero, bhi, _MM_CMPINT_EQ);
        openmasklo |= (openmaskhi << 8);
        return openmasklo != 0;
    }

    inline __attribute__((always_inline)) std::tuple<V, V, V, V, V, V, V, V> at512(__m512i key) const {
        auto k = unhash(key);
        
        auto bucket = _mm512_slli_epi64(_mm512_and_epi64(k, m_vz_m1), 4);
        auto b = _mm512_i64gather_epi64(bucket, m_location, 8);
        auto neq = _mm512_cmp_epi64_mask(k, b, _MM_CMPINT_NE);

        while(neq) {
            bucket = _mm512_mask_add_epi64(bucket, neq, bucket, one);
            b = _mm512_i64gather_epi64(bucket, m_location, 8);
            neq = _mm512_cmp_epi64_mask(k, b, _MM_CMPINT_NE);
            
        }

        uint64_t ind[8];
        _mm512_storeu_epi64(ind, bucket);

        return std::make_tuple(m_data[ind[0]], m_data[ind[1]], m_data[ind[2]], m_data[ind[3]], m_data[ind[4]], m_data[ind[5]], m_data[ind[6]], m_data[ind[7]]);
    }

    inline __attribute__((always_inline))  uint32_t loc(const K & key) const {
        std::size_t k = std::hash<K>{}(key);
        unsigned int bucket = (k & m_sz_m1) << 4;
        auto b = _mm512_load_epi64(m_location + bucket);
        
        unsigned short omask = _mm512_cmp_epi64_mask(zero, b, _MM_CMPINT_EQ);
        if(omask) [[likely]] {
            auto openindex = __builtin_ffs(omask) - 1;
            return openindex + bucket;
        }

        b = _mm512_load_epi64(m_location + bucket + 8);
        omask = _mm512_cmp_epi64_mask(zero, b, _MM_CMPINT_EQ);
        if(omask) {
            auto openindex = __builtin_ffs(omask) - 1;
            return openindex + bucket + 8;
        }

        throw;
    }

    inline __attribute__((always_inline))  V& at_int64(const uint64_t & key) {
        auto k = unhash(key);
        unsigned int bucket = (k & m_sz_m1) << 4;
        auto b = _mm512_load_epi64(m_location + bucket);
        
        unsigned short omask = _mm512_cmp_epi64_mask(zero, b, _MM_CMPINT_EQ);
        if(omask) [[likely]] 
            return m_data[__builtin_ffs(omask) - 1 + bucket];
        

        b = _mm512_load_epi64(m_location + bucket + 8);
        omask = _mm512_cmp_epi64_mask(zero, b, _MM_CMPINT_EQ);
        if(omask) {
            auto openindex = __builtin_ffs(omask) - 1;
            return m_data[openindex + bucket + 8];
        }

        throw;
    }


    inline __attribute__((always_inline))  V& at_no_intrinsic_int64(const uint64_t & key) {
        const auto k = unhash(key);
        const unsigned int bucket = (k & m_sz_m1) << 4;
        for(int i = 0; i < 16; ++i) {
            if(key == m_location[bucket + i])
                return m_data[i + bucket];
        }

        throw;
    }


    inline V & at(const K & key) const {
        std::size_t k = std::hash<K>{}(key);
        unsigned int bucket = (k & m_sz_m1) << 4;
        auto blo = _mm512_load_epi64(m_location + bucket);
        auto bhi = _mm512_load_epi64(m_location + bucket + 8);
        // why won't prefetch improve this????????
        // __builtin_prefetch(m_data + bucket);
        // __builtin_prefetch(m_data + bucket + 8);
        unsigned short openmasklo = _mm512_cmp_epi64_mask(zero, blo, _MM_CMPINT_EQ);
        unsigned short openmaskhi = _mm512_cmp_epi64_mask(zero, bhi, _MM_CMPINT_EQ);
        openmasklo |= (openmaskhi << 8);

        if(openmasklo == 0)
            throw;

        auto openindex = __builtin_ffs(openmasklo) - 1;
        return m_data[openmasklo + bucket];
    }

    inline V & at_no_intrinsic(const K & key) const {
        std::size_t k = std::hash<K>{}(key);
        unsigned int bucket = (k & m_sz_m1) << 4;
        for(int i = 0; i < 16; ++i) {
            if(k == m_location[bucket + i])
                return m_data[i + bucket];
        }

        throw;
    }

    void insert_no_intrinsic_int64(const uint64_t & key, V data) {
        const auto k = unhash(key);
        const unsigned int bucket = (k & m_sz_m1) << 4;
        for(auto b = 0; b < 16; ++b)
        {
            if(m_location[bucket + b] == 0) {
                m_location[bucket + b] = key;
                m_data[bucket + b] = data;
                return;
            }
        }
        throw;
    }

    void insert_no_intrinsic_int64(const uint64_t & key) {
        const auto k = unhash(key);
        const unsigned int bucket = (k & m_sz_m1) << 4;
        for(auto b = 0; b < 16; ++b)
        {
            if(m_location[bucket + b] == 0) {
                m_location[bucket + b] = key;
                return;
            }
        }
        throw;
    }

    void insert_empty_int64(const uint64_t & key) {
        auto k = unhash(key);
        unsigned int bucket = (k & m_sz_m1) << 4;
        auto blo = _mm512_load_epi64(m_location + bucket);
        auto bhi = _mm512_load_epi64(m_location + bucket + 8);
        unsigned short openmasklo = _mm512_cmp_epi64_mask(zero, blo, _MM_CMPINT_EQ);
        unsigned short openmaskhi = _mm512_cmp_epi64_mask(zero, bhi, _MM_CMPINT_EQ);
        openmasklo |= (openmaskhi << 8);

        if(openmasklo == 0)
            throw;

        auto openindex = __builtin_ffs(openmasklo) - 1;
        m_location[openindex + bucket] = k;
        
    }

    void insert_empty(const K & key) {
        std::size_t k = std::hash<K>{}(key);
        unsigned int bucket = (k & m_sz_m1) << 4;
        auto blo = _mm512_load_epi64(m_location + bucket);
        auto bhi = _mm512_load_epi64(m_location + bucket + 8);
        unsigned short openmasklo = _mm512_cmp_epi64_mask(zero, blo, _MM_CMPINT_EQ);
        unsigned short openmaskhi = _mm512_cmp_epi64_mask(zero, bhi, _MM_CMPINT_EQ);
        openmasklo |= (openmaskhi << 8);

        if(openmasklo == 0)
            throw;

        auto openindex = __builtin_ffs(openmasklo) - 1;
        m_location[openindex + bucket] = k;
    }

    void insert(const K & key, V && value) {
        std::size_t k = std::hash<K>{}(key);
        unsigned int bucket = (k & m_sz_m1) << 4;
        auto blo = _mm512_load_epi64(m_location + bucket);
        auto bhi = _mm512_load_epi64(m_location + bucket + 8);
        unsigned short openmasklo = _mm512_cmp_epi64_mask(zero, blo, _MM_CMPINT_EQ);
        unsigned short openmaskhi = _mm512_cmp_epi64_mask(zero, bhi, _MM_CMPINT_EQ);
        openmasklo |= (openmaskhi << 8);

        if(openmasklo == 0)
            throw;

        auto openindex = __builtin_ffs(openmasklo) - 1;
        m_location[openindex + bucket] = k;
        m_data[openindex + bucket] = value;
    }

    void insert_no_intrinsic(const K & key) {
        std::size_t k = std::hash<K>{}(key);
        unsigned int bucket = (k & m_sz_m1) << 4;
        for(auto b = 0; b < 16; ++b)
        {
            if(m_location[bucket + b] == 0) {
                m_location[bucket + b] = (k >> 32) | 1;
                return;
            }
        }
    }

    inline __attribute__((always_inline)) uint64_t unhash(uint64_t x) const {
        x = (x ^ (x >> 31) ^ (x >> 62)) * UINT64_C(0x319642b2d24d8ec3);
        x = (x ^ (x >> 27) ^ (x >> 54)) * UINT64_C(0x96de1b173f119089);
        x = x ^ (x >> 30) ^ (x >> 60);
        return x;
    }

    inline __attribute__((always_inline)) __m512i unhash(__m512i x) const {
        auto a = _mm512_srli_epi64(x, 31);
        auto b = _mm512_srli_epi64(x, 62);
        x = _mm512_mullox_epi64(_mm512_xor_epi64(x, _mm512_xor_epi64(a, b)), m_a);
        a = _mm512_srli_epi64(x, 27);
        b = _mm512_srli_epi64(x, 54);
        x = _mm512_mullox_epi64(_mm512_xor_epi64(x, _mm512_xor_epi64(a, b)), m_b);
        a = _mm512_srli_epi64(x, 30);
        b = _mm512_srli_epi64(x, 60);
        return _mm512_xor_epi64(x, _mm512_xor_epi64(a, b));
    }
};

template<class T>
struct fash_kvp {
    uint64_t key;
    T value;
};

template <class K, class V>
class fash2 {
    fash_kvp<V>* __restrict m_data;
    unsigned char m_bitsz;
    unsigned int m_sz, m_sz_m1;
    __m512i m_vz_m1;
    const __m512i zero = _mm512_set1_epi64(0ULL);
    const __m512i one = _mm512_set1_epi64(1ULL);
    const __m512i m_a = _mm512_set1_epi64(UINT64_C(0x319642b2d24d8ec3));
    const __m512i m_b = _mm512_set1_epi64(UINT64_C(0x96de1b173f119089));

public: 
    fash2(unsigned char bit_size) {
        m_bitsz = bit_size;
        m_sz = 1 << (m_bitsz + 4);
        m_sz_m1 = (1<<m_bitsz) - 1;
        m_vz_m1 = _mm512_set1_epi64(m_sz_m1);
        m_data = new (std::align_val_t(64)) fash_kvp<V>[m_sz];
        memset(m_data, 0, m_sz*sizeof(fash_kvp<V>));
    }

    ~fash2() {
        ::operator delete[] (m_data, std::align_val_t(64));
    }

    inline __attribute__((always_inline))  V& at_no_intrinsic_int64(const uint64_t & key) {
        const auto k = unhash(key);
        const unsigned int bucket = (k & m_sz_m1) << 4;
        for(int i = 0; i < 16; ++i) {
            if(key == m_data[bucket + i].key)
                return m_data[bucket + i].value;
        }

        throw;
    }

    void insert_no_intrinsic_int64(const uint64_t & key, V data) {
        const auto k = unhash(key);
        const unsigned int bucket = (k & m_sz_m1) << 4;
        for(auto b = 0; b < 16; ++b)
        {
            if(m_data[bucket + b].key == 0) {
                m_data[bucket + b].key = key;
                m_data[bucket + b].value = data;
                return;
            }
        }
        throw;
    }

    void insert_no_intrinsic_int64(const uint64_t & key) {
        const auto k = unhash(key);
        const unsigned int bucket = (k & m_sz_m1) << 4;
        for(auto b = 0; b < 16; ++b)
        {
            if(m_data[bucket + b].key == 0) {
                m_data[bucket + b].key = key;
                return;
            }
        }
        throw;
    }

    inline __attribute__((always_inline)) uint64_t unhash(uint64_t x) const {
        x = (x ^ (x >> 31) ^ (x >> 62)) * UINT64_C(0x319642b2d24d8ec3);
        x = (x ^ (x >> 27) ^ (x >> 54)) * UINT64_C(0x96de1b173f119089);
        x = x ^ (x >> 30) ^ (x >> 60);
        return x;
    }

};


template <class K, class V>
class fash128x2 {
    fash_kvp<V> * __restrict m_data;
    unsigned char m_bitsz;
    uint64_t m_sz, m_sz_m1;
    __m512i m_vz_m1;
    const __m512i zero = _mm512_set1_epi64(0ULL);
    const __m512i one = _mm512_set1_epi64(1ULL);
    const __m512i m_a = _mm512_set1_epi64(UINT64_C(0x319642b2d24d8ec3));
    const __m512i m_b = _mm512_set1_epi64(UINT64_C(0x96de1b173f119089));

public: 
    fash128x2(unsigned char bit_size) {
        m_bitsz = bit_size;
        m_sz = 1 << (m_bitsz + 1); // 7 - 6 = (bucket size) - (fraction of entries)
        m_sz_m1 = (1<<(m_bitsz-6)) - 1;
        m_vz_m1 = _mm512_set1_epi64(m_sz_m1);
        m_data = new (std::align_val_t(64)) fash_kvp<V>[m_sz];
        memset(m_data, 0, m_sz*sizeof(fash_kvp<V>));
    }

    ~fash128x2() {
        ::operator delete[] (m_data, std::align_val_t(64));
    }

    inline __attribute__((always_inline)) V & at_no_intrinsic_int64(const uint64_t & key) {
        const auto k = unhash(key);
        const unsigned int bucket = (k & m_sz_m1) << 7;
        const unsigned int guess = (18302628885633695744ULL & k)>>57;
        for(int i = 0; i < 128; ++i)
        {
            auto idx = bucket + ((i + guess) & 127);
            if(m_data[idx].key == key) 
                return m_data[idx].value;
        }

        throw;
    }

    void insert_no_intrinsic_int64(const uint64_t & key) {
        const auto k = unhash(key);
        const unsigned int bucket = (k & m_sz_m1) << 7;
        const unsigned int guess = (18302628885633695744ULL & k)>>57;

        for(int i = 0; i < 128; ++i)
        {
            auto idx = bucket + ((i + guess) & 127);
            if(m_data[idx].key == 0) {
                m_data[idx].key = key;
                return;
            }
        }
        std::cout << m_data[bucket].key << std::endl;
        throw;
    }

    void insert_no_intrinsic_int64(const uint64_t & key, V data) {
        const auto k = unhash(key);
        const unsigned int bucket = (k & m_sz_m1) << 7;
        const unsigned int guess = (18302628885633695744ULL & k)>>57;

        for(int i = 0; i < 128; ++i)
        {
            auto idx = bucket + ((i + guess) & 127);
            if(m_data[idx].key == 0) {
                m_data[idx].key = key;
                m_data[idx].value = data;
                return;
            }
        }
        std::cout << m_data[bucket].key << std::endl;
        throw;
    }

    inline __attribute__((always_inline)) uint64_t unhash(uint64_t x) const {
        x = (x ^ (x >> 31) ^ (x >> 62)) * UINT64_C(0x319642b2d24d8ec3);
        x = (x ^ (x >> 27) ^ (x >> 54)) * UINT64_C(0x96de1b173f119089);
        x = x ^ (x >> 30) ^ (x >> 60);
        return x;
    }

};

