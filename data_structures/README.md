# A C/C++ case study on how to organize an AVX accelerated application.

In a recent CppCon talk, Optiver's David Gross said "if you don't design your data properly, it's just going to be slow everywhere," highlighting the idea that you cannot micro-optimize out bottleneck problems if they're built on memory-inefficient data structures. So, the fundamental question I want to answer is how we build a data processing solution that effectively integrates AVX intrinsics, and as an example, I'll draw from my field of work: financial derivatives pricing.

Financial derivatives are interesting because there are a lot of derivatives active at any one time--like tens of millions--and there are a lot of things you might need to calculate. The first step is usually to split the work up among different machines (or ignore irrelevant derivatives) to reduce it down to tens of thousands, but after that, organizing all of this data to be responsive and memory efficient is critical. 

There are several types of operations we want to be able to perform on our data: long [^1] (~1us per calculation), medium [^2] (~50ns per calculation) and short (a few clock cycles, e.g. add two numbers). 

## The type of AVX acceleration we're targeting

Each calculation I mentioned can be done serially. One interesting thing about all of them is that they can be written as AVX functions. That is, instead of calculating one implied volatility at a time, we can use the VCL to compose a function that calculates 16 at once. So, the goal will be to calculate the desired values in chunks of 16, rather than one at a time. [`vec_black_scholes.hpp`](vec_black_scholes.hpp) features the implementations of these functions, whereas [`black_scholes.hpp`](black_scholes.hpp) has the non-vectorized implementations.

## The data structures

We need to organize eight `floats` for each derivative. I'll try three ways.

The first I think is the "obvious" way for most programmers: put all eight in a structure for each derivative, then store an array of derivatives.

```
struct alignas(32)  bs {
    public:
    float ul, tte, strike, rate, iv, vol, px, theo;
};
```

The next would be to keep each data type in a contigious array inside a data structure:

```
struct alignas(4096) bsv {
    public:
    std::unique_ptr<float[]> ul;
    std::unique_ptr<float[]> tte;
    std::unique_ptr<float[]> strike;
    std::unique_ptr<float[]> rate;
    std::unique_ptr<float[]> iv;
    std::unique_ptr<float[]> vol;
    std::unique_ptr<float[]> px;
    std::unique_ptr<float[]> theo;
};
```

Neither of these leverages any intrinsic data structure. So in the third, we will replace the `float` arrays with arrays of vector types. For this, we will use Agner Fog's superb [Vector Class Library (VCL)](https://github.com/vectorclass/version2), and specifically use the `Vec16f` type, which is a 512-bit storage for 16 contiguous floats.

```
struct alignas(4096) bsv512
{
    std::unique_ptr<V16[]> ul;
    std::unique_ptr<V16[]> tte;
    std::unique_ptr<V16[]> strike;
    std::unique_ptr<V16[]> rate;
    std::unique_ptr<V16[]> iv;
    std::unique_ptr<V16[]> vol;
    std::unique_ptr<V16[]> px;
    std::unique_ptr<V16[]> theo;
};
```

where

```
union alignas(64) V16 {
    Vec16f vcl;
    float array[16];
    __m512 intr;
};
```

## Notes on `struct bs`

This structure is the most intuitive, but extremely AVX unfriendly. Inputs to the vectorized functions are of type `Vec16f`, so for example, they take in 16 different `ul` values, but an array of type `bsv` will not store its `ul` values contiguously. For this, we use a rather ugly hack by treating the `bsv` array as a `float` array, and then using `scatter` and `gather` operations to create the function inputs and apply the output. `scatter` and `gather` are [not particularly fast](https://www.agner.org/optimize/instruction_tables.pdf), especially on AMD processors, this should be slow. And given that many calculations do not call for the use of all the structure's variables, it will be bringing in unnecessary data to the cache page, and for operations that are memory-constrained, this can matter quite a bit.

## `bsv` vs. `bsv512`

There are advantages and disadvantages to each of these, despite how similar they are. `bsv` will be able to leverage autovectorization whenever the compiler sees an opportunity. This is ideal, but as noted earlier, it will not be able to recognize many opportunities, but when it does, you almost always want to just let it take over. Arrays of floats are much more familiar to compilers than `Vec16f` types. On the other hand, arrays of type `Vec16f` do not need to be changed into vector types in order to process them, thus saving time in the `load` and `store` stages of processing.

Although I will not be showing this here, `bsv512` has the additional complication that it is probably slower to load data into than `bsv`, but that all depends on how much you can leverage AVX up and down through your application.

## The test

I would like to see how all data structures perform in all compute paradigms. Additionally, I would like to see how it performs single threaded versus using 32 threads via OpenMP (OMP). I chose a value of `N=51200` because it's appropriately large for practical uses, and it divides all my partitions evenly. Of note is that `51200*8*4=1.6MB` for the total data operated on, and my processor has 64KB of L1 cache and 1MB of L2 per core. If we want to start exceeding cache, the usefulness of each data structure is bound to change. Long is prefixed with `iv_`, Medium is prefixed with `pricer_` and Short with `vol_edge`. I use [Google Benchmark](https://github.com/google/benchmark) as the microbenchmarking framework. 

To run it cd into the build folder `cd build`. Then make it with `cmake -DCMAKE_BUILD_TYPE=Release ..` and then `make`. Then run the `data_structures` executable.

Here are the results:

Run on (32 X 5881 MHz CPU s)

CPU Caches:
- L1 Data 32 KiB (x16)
- L1 Instruction 32 KiB (x16)
- L2 Unified 1024 KiB (x16)
- L3 Unified 32768 KiB (x2)

|Benchmark                    |   Time (GCC) |  Time (Clang)|
|:----------------------------|-------------:|-------------:|
|iv_naive_bsv                 |   17222.87 us :heavy_check_mark:|   18156.57 us|
|iv_naive_bs                  |   17239.20 us|   18116.14 us|
|iv_avx_bsv                   |    1578.78 us|    1331.91 us :heavy_check_mark:|
|iv_avx_bsv_omp               |      87.00 us|      69.54 us :heavy_check_mark:|
|iv_avx_bsv512                |    1575.17 us|    1350.81 us|
|iv_avx_bsv512_omp            |      84.88 us|      70.10 us :heavy_check_mark:|
|iv_avx_bs                    |    1647.47 us|    1428.12 us|
|iv_avx_bs_omp                |      89.64 us|      75.40 us|
|pricer_naive_bsv             |     143.17 us :heavy_check_mark:|     853.25 us|
|pricer_naive_bs              |     164.57 us|     808.51 us|
|pricer_avx_bsv               |      94.68 us|      88.51 us|
|pricer_avx_bsv_omp           |       8.40 us|       6.69 us :heavy_check_mark:|
|pricer_avx_bsv512            |      99.26 us|      87.82 us :heavy_check_mark:|
|pricer_avx_bsv512_omp        |       8.13 us|       6.71 us :heavy_check_mark:|
|pricer_avx_bs                |     152.03 us|     161.31 us|
|pricer_avx_bs_omp            |      10.98 us|      10.26 us|
|vol_edge_naive_bsv512        |       8.85 us|       5.70 us|
|vol_edge_naive_bsv           |       3.60 us :heavy_check_mark:|       3.64 us :heavy_check_mark:|
|vol_edge_naive_bs            |      20.24 us|      19.47 us|
|vol_edge_avx_bsv             |       3.57 us|       5.00 us|
|vol_edge_avx_unrolled_bsv    |       3.55 us|       3.67 us|
|vol_edge_avx_bsv512          |       9.15 us|       7.27 us|
|vol_edge_avx_unrolled_bsv512 |       4.36 us|       4.03 us|
|vol_edge_avx_bs              |      40.52 us|      41.18 us|
|vol_edge_avx_unrolled_bs     |      37.82 us|      38.88 us|

## Analysis

`bsv` and `bsv512` seem to be nearly the same[^3], with the exception being that autovectorization works much better with `bsv`. Notice how much manual effort it took to unroll/reorder `bsv512` for the Short (vol_edge_*) cases just to make it match the autovectorized version.

One other thing of note is how across-the-board `bs` is the worst, but if you are only doing Long calculations, then maybe other qualities of the data structure could make it the best to use. But when you're doing Short or Medium calculations, `scatter` and `gather` combined with cache thrashing is just too much overhead. Even when you use `bs` naively, the results are bad, because the compiler cannot autovectorize it easily.

## Comments

A big takeaway is just look at how much AVXifying your code can help. Single threaded Naive-to-AVX offers a >10x speedup. That's hard to ignore, and if the concept is hard to ignore, then you should consider from the very beginning whether your data structures will hamper your ability to leverage it. It's easy to mistakenly think ahead of time that `bs` would be a good data structure for this purpose. It's simple, and what's more is that by putting all your instrument's data together, you'll get speed ups from the idea that using one piece of data implies a high probability that you will use another, so they'll both be there on the cache line already. That sounds logical to me, but then, when you're SIMD optimizing down the road, or the compiler is trying to SIMD optimize for you, you lose big time, and probably by that point, you've built so many other applications around the `bs` data structure that you cannot go back without incurring a huge expense.


### Footnotes

[^1]: This will be calculating the inverse of the Black-Scholes option price with respect to its volatility parameter, and [you can read about Black-Scholes here](http://www.iam.fmph.uniba.sk/institute/stehlikova/fd14en/lectures/06_black_scholes_2.pdf). Aside from the formula itself being non-trivial, with calls to the ERF function, it's not analytically invertible, and it's best to use a bisection root finder to solve instead. 

[^2]: This will be calculating an option's value in the Black-Scholes formula.

[^3]: A subtle bug that led to fewer IV iterations in an earlier version of this essay had made `bsv512` look like a clear winner, and that didn't feel good because it's not a good data structure. I'm thankful to have found that bug.
